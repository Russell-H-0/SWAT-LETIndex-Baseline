"""M2 focused tests: the SWAT-style block-bin allocation planner (Issue #4).

Covered: the pinned distribution/sizing kernel (independently re-derived), the block-unit
geometry, the stochastic plan and its structural invariants, the seeded randomness
contract, the explicit insufficient-capacity failure, independence from record contents,
and the absence of every physical/observable surface.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from swat_m_block import (
    ATOMIC_BUCKET_CAPACITY_BLOCKS,
    SWAT_REFERENCE_PRIVACY_DELTA,
    SWAT_REFERENCE_PRIVACY_EPSILON,
    SWAT_REFERENCE_SECURITY_LAMBDA,
    BlockAllocationError,
    BlockAllocationPlan,
    BlockBinPlan,
    GeometricLoadSampler,
    InsufficientSampledCapacity,
    LaplaceSampler,
    SwatBlockAllocationConfig,
    allocate_block_bins,
    compute_bin_capacity_blocks,
    compute_bin_count,
    compute_factor,
    compute_noisy_prefix_sums,
    compute_weight,
    derive_stream_seed,
    geom_conv,
    swat_reference_config,
)
from swat_m_block.distribution import STREAM_DOMAIN_LAPLACE, STREAM_DOMAIN_LOADS

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "codes"
SRC_DIR = CODES_DIR / "src"
SWAT_PACKAGE = SRC_DIR / "swat_m_block"
MANIFEST_PATH = REPO_ROOT / "provenance" / "common-substrate.sha256"
VERIFY_SCRIPT = REPO_ROOT / "provenance" / "verify_manifest.py"

LAMBDA = SWAT_REFERENCE_SECURITY_LAMBDA           # 512
PRIVACY_EPSILON = SWAT_REFERENCE_PRIVACY_EPSILON  # 1.0
PRIVACY_DELTA = SWAT_REFERENCE_PRIVACY_DELTA      # 1e-12

#: The block-unit geometry of the pinned SWAT reference config, independently re-derived
#: by the sizing search reproduced in this module (see _independent_bin_capacity_blocks).
REFERENCE_BIN_CAPACITY_BLOCKS = 16
REFERENCE_FACTOR = 0.12829670090910578

M2_MODULES = ("distribution.py", "bin_allocator.py")

#: Mechanisms M2 still must not contain (M2 authorises the stochastic block-bin planner
#: itself; everything physical, the DO data path, the merge and the shuffle stay out).
FORBIDDEN_M2_TOKENS = (
    "DOAllocate", "DOMerge", "DOMerger", "do_allocate", "do_merge",
    "UntrustedStorage", "TraceEvent", "TraceOperation", "TraceCollector", "SlotId",
    "output_shuffle", "oblivious_shuffle", "bitonic", "sorting_network",
    "prp_writeback", "deamortiz", "de_amortiz", "interior_point", "DPInteriorPoint",
    "read_slot", "write_slot", "physical_slot", "ciphertext", "sgx_",
)


# ---------------------------------------------------------------------------
# independent re-derivations (deliberately NOT reusing the implementation)
# ---------------------------------------------------------------------------


def _independent_weights(bin_capacity: int, epsilon: float):
    """The pinned recurrence, written out again from the pinned source semantics."""
    alpha = math.exp(-epsilon)
    half = bin_capacity // 2
    weights = [0.0] * (bin_capacity + 1)
    weights[half] = (1.0 - alpha) / (1.0 + alpha - 2.0 * alpha ** (half + 1))
    for i in range(1, half + 1):
        weights[half + i] = weights[half + i - 1] * alpha ** i
        weights[half - i] = weights[half + i]
    return weights


def _independent_convolve(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    return out


def _independent_convolved_weights(bin_capacity: int, epsilon: float, count: int):
    current = _independent_weights(bin_capacity, epsilon)
    for _ in range(1, count):
        current = _independent_convolve(current, _independent_weights(bin_capacity, epsilon))
    return current


def _independent_bin_capacity_blocks(lam: int, epsilon: float, delta: float,
                                     bucket_capacity: int = 1) -> int:
    """The pinned sizing search, written out again independently."""
    def even_up(value: int) -> int:
        return value + (value & 1)

    upper = even_up(math.floor(math.log(lam) ** 5 / epsilon))
    lower = 2
    if lower + 2 >= upper:
        return max(lower, upper)
    capacity = upper
    while lower + 2 < upper:
        capacity = even_up((lower + upper) // 2)
        min_bins = math.ceil(2 * bucket_capacity
                             / (capacity * (1 - math.log(lam) ** -2)))
        pmf = _independent_convolved_weights(capacity, epsilon, min_bins)
        if sum(pmf[:bucket_capacity]) > delta:
            lower = capacity + 2
        else:
            upper = capacity
    return capacity


def _code_without_docstrings(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                source = source.replace(doc, "")
    return source


class _ConstantLoadSampler:
    """The narrow testing seam: a load sampler with a fixed answer."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def sample(self) -> int:
        self.calls += 1
        return self.value


def plan_for(block_count: int, seed: int = 7, **kwargs) -> BlockAllocationPlan:
    return allocate_block_bins(block_count, swat_reference_config(seed), **kwargs)


def succeeds(block_count: int, seed: int) -> bool:
    try:
        allocate_block_bins(block_count, swat_reference_config(seed))
        return True
    except InsufficientSampledCapacity:
        return False


# ---------------------------------------------------------------------------
# A. pinned distribution kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [2, 4, 8, 16, 64])
@pytest.mark.parametrize("epsilon", [0.25, 1.0, 3.5])
def test_a_compute_weight_reproduces_the_pinned_recurrence_independently(capacity, epsilon):
    assert compute_weight(capacity, epsilon) == pytest.approx(
        _independent_weights(capacity, epsilon), rel=1e-12, abs=1e-300
    )


def test_a_compute_weight_is_symmetric_about_the_centre_with_a_triangular_decay():
    capacity, epsilon = 8, 1.0
    weights = compute_weight(capacity, epsilon)
    half = capacity // 2
    assert len(weights) == capacity + 1
    for i in range(1, half + 1):
        assert weights[half + i] == weights[half - i]
        # pinned linear-geometric decay: alpha ** (i * (i + 1) / 2)
        expected = weights[half] * math.exp(-epsilon) ** (i * (i + 1) / 2)
        assert weights[half + i] == pytest.approx(expected, rel=1e-12)
    # ... which is NOT the textbook geometric ratio alpha ** i
    alpha = math.exp(-epsilon)
    assert weights[half + 3] != pytest.approx(weights[half] * alpha ** 3, rel=1e-6)
    assert weights[half + 3] == pytest.approx(weights[half] * alpha ** 6, rel=1e-12)


def test_a_compute_weight_returns_the_pinned_raw_vector_unrenormalised():
    """The pinned vector need not sum to one; it is normalised by the sampler."""
    weights = compute_weight(8, 1.0)
    total = sum(weights)
    assert total != pytest.approx(1.0, rel=1e-6)
    assert 0.0 < total < 1.0


@pytest.mark.parametrize("capacity", [0, 1, 3, 5, -4])
def test_a_compute_weight_rejects_non_even_or_too_small_capacity(capacity):
    with pytest.raises(BlockAllocationError):
        compute_weight(capacity, 1.0)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_a_compute_weight_rejects_invalid_epsilon(epsilon):
    with pytest.raises(BlockAllocationError):
        compute_weight(8, epsilon)


@pytest.mark.parametrize("capacity,epsilon,count", [(2, 1.0, 1), (4, 1.0, 2), (4, 0.5, 3),
                                                    (8, 2.0, 2), (16, 1.0, 1)])
def test_a_geom_conv_matches_an_independent_convolution(capacity, epsilon, count):
    result = geom_conv(capacity, epsilon, count)
    assert len(result) == capacity * count + 1
    assert result == pytest.approx(
        _independent_convolved_weights(capacity, epsilon, count), rel=1e-12, abs=1e-300
    )


def test_a_geom_conv_is_pure_and_repeatable():
    first = geom_conv(8, 1.0, 2)
    second = geom_conv(8, 1.0, 2)
    assert first == second


@pytest.mark.parametrize("count", [0, -1, True])
def test_a_geom_conv_rejects_invalid_bin_count(count):
    with pytest.raises(BlockAllocationError):
        geom_conv(8, 1.0, count)


def test_a_geometric_load_sampler_draws_inside_the_support_and_is_seeded():
    sampler = GeometricLoadSampler(16, 1.0, seed=5)
    draws = [sampler.sample() for _ in range(300)]
    assert all(0 <= draw <= 16 for draw in draws)
    reference = GeometricLoadSampler(16, 1.0, seed=5)
    assert draws == [reference.sample() for _ in range(300)]
    other = GeometricLoadSampler(16, 1.0, seed=6)
    assert draws != [other.sample() for _ in range(300)]


def test_a_geometric_load_sampler_rejects_an_odd_capacity():
    with pytest.raises(BlockAllocationError):
        GeometricLoadSampler(15, 1.0, seed=1)


def test_a_the_load_distribution_concentrates_near_the_centre():
    sampler = GeometricLoadSampler(16, 1.0, seed=11)
    draws = [sampler.sample() for _ in range(4000)]
    centre_hits = sum(1 for draw in draws if abs(draw - 8) <= 3)
    assert centre_hits > 0.75 * len(draws)
    assert max(draws) <= 16 and min(draws) >= 0


def test_a_laplace_sampler_is_deterministic_and_linear_in_its_scale():
    first = LaplaceSampler(seed=3)
    second = LaplaceSampler(seed=3)
    doubled = LaplaceSampler(seed=3)
    samples = [first.sample(1.0) for _ in range(50)]
    assert samples == [second.sample(1.0) for _ in range(50)]
    assert all(math.isfinite(value) for value in samples)
    # the pinned formula is exactly linear in the scale for a fixed draw
    scaled = [doubled.sample(2.0) for _ in range(50)]
    assert scaled == pytest.approx([2.0 * value for value in samples], rel=1e-12)


def test_a_the_laplace_scale_is_the_inverse_of_the_privacy_epsilon():
    """The pinned prefix code calls the sampler with ``1 / epsilon``."""
    loads = (5, 5)
    laplace = LaplaceSampler(seed=2)
    prefix, error = compute_noisy_prefix_sums(
        loads, privacy_epsilon=1.0, bin_capacity_blocks=16, laplace=laplace
    )
    assert len(prefix) == 3 and prefix[0] == 0
    assert error >= 0


# ---------------------------------------------------------------------------
# B. block-unit geometry
# ---------------------------------------------------------------------------


def test_b_reference_geometry_is_independently_reproduced():
    assert LAMBDA == 512 and PRIVACY_EPSILON == 1.0 and PRIVACY_DELTA == 1e-12
    derived = compute_bin_capacity_blocks(
        security_lambda=LAMBDA, privacy_epsilon=PRIVACY_EPSILON,
        privacy_delta=PRIVACY_DELTA,
    )
    assert derived == REFERENCE_BIN_CAPACITY_BLOCKS
    assert derived == _independent_bin_capacity_blocks(LAMBDA, PRIVACY_EPSILON, PRIVACY_DELTA)
    assert derived % 2 == 0 and derived > 0
    assert compute_factor(LAMBDA, derived) == pytest.approx(REFERENCE_FACTOR, rel=1e-15)


def test_b_the_reference_geometry_matches_the_pinned_formula_by_hand():
    capacity = REFERENCE_BIN_CAPACITY_BLOCKS
    decay = 1.0 - math.log(LAMBDA) ** -2
    assert decay == pytest.approx(0.9743040866542517, rel=1e-15)
    assert capacity == pytest.approx(2.0 / (REFERENCE_FACTOR * decay), rel=1e-12)


@pytest.mark.parametrize("lam,epsilon", [(512, 1.0), (512, 0.5), (1024, 1.0), (256, 2.0)])
def test_b_bin_capacity_is_always_an_even_positive_integer(lam, epsilon):
    capacity = compute_bin_capacity_blocks(
        security_lambda=lam, privacy_epsilon=epsilon, privacy_delta=1e-12
    )
    assert isinstance(capacity, int) and capacity > 0 and capacity % 2 == 0
    assert capacity == _independent_bin_capacity_blocks(lam, epsilon, 1e-12)


def test_b_the_bucket_capacity_is_frozen_to_one_block():
    assert ATOMIC_BUCKET_CAPACITY_BLOCKS == 1
    with pytest.raises(BlockAllocationError):
        compute_bin_capacity_blocks(
            security_lambda=LAMBDA, privacy_epsilon=1.0, privacy_delta=1e-12,
            bucket_capacity_blocks=4,
        )


def test_b_the_degenerate_upper_candidate_is_resolved_explicitly():
    """Pinned leaves the member unset when the loop never runs; M2 defines it."""
    lam, epsilon, delta = 3, 1.0, 1e-12
    assert math.floor(math.log(lam) ** 5 / epsilon) + 1 < 4  # loop body would not run
    capacity = compute_bin_capacity_blocks(
        security_lambda=lam, privacy_epsilon=epsilon, privacy_delta=delta
    )
    assert capacity == 2
    assert capacity == _independent_bin_capacity_blocks(lam, epsilon, delta)


@pytest.mark.parametrize("blocks,expected", [(0, 0), (1, 1), (8, 2), (16, 3), (64, 9),
                                             (512, 66)])
def test_b_bin_count_matches_the_pinned_ceiling(blocks, expected):
    factor = REFERENCE_FACTOR
    assert compute_bin_count(factor, blocks) == expected
    if blocks == 0:
        assert compute_bin_count(factor, 0) == 0
    else:
        assert compute_bin_count(factor, blocks) == math.ceil(factor * blocks)


def test_b_zero_blocks_needs_no_bin_at_all():
    assert compute_bin_count(REFERENCE_FACTOR, 0) == 0
    with pytest.raises(BlockAllocationError):
        compute_bin_count(REFERENCE_FACTOR, -1)


def test_b_bin_count_uses_floating_point_not_integer_division():
    """A C/C++ integer division here would floor to 0 for small inputs."""
    factor = compute_factor(LAMBDA, REFERENCE_BIN_CAPACITY_BLOCKS)
    assert 0.0 < factor < 1.0
    assert compute_bin_count(factor, 1) == 1  # integer division would give 0


# ---------------------------------------------------------------------------
# C. configuration validation
# ---------------------------------------------------------------------------


def test_c_the_reference_config_carries_the_swat_defaults():
    config = swat_reference_config(seed=1234)
    assert config.security_lambda == 512
    assert config.privacy_epsilon == 1.0
    assert config.privacy_delta == 1e-12
    assert config.seed == 1234
    assert config.decay == pytest.approx(1.0 - math.log(512) ** -2, rel=1e-15)


@pytest.mark.parametrize("lam", [1, 0, -5, 2, True, "512", 512.0, None])
def test_c_invalid_security_lambda_is_refused(lam):
    with pytest.raises(BlockAllocationError):
        SwatBlockAllocationConfig(security_lambda=lam, privacy_epsilon=1.0,
                                  privacy_delta=1e-12, seed=0)


@pytest.mark.parametrize("epsilon", [0, -1, True, "1", float("nan"), float("inf"), None])
def test_c_invalid_privacy_epsilon_is_refused(epsilon):
    with pytest.raises(BlockAllocationError):
        SwatBlockAllocationConfig(security_lambda=512, privacy_epsilon=epsilon,
                                  privacy_delta=1e-12, seed=0)


@pytest.mark.parametrize("delta", [0, 1, 1.5, -0.1, True, "1e-12", float("nan"), None])
def test_c_invalid_privacy_delta_is_refused(delta):
    with pytest.raises(BlockAllocationError):
        SwatBlockAllocationConfig(security_lambda=512, privacy_epsilon=1.0,
                                  privacy_delta=delta, seed=0)


@pytest.mark.parametrize("seed", [-1, True, "0", 1.5, None])
def test_c_invalid_seed_is_refused(seed):
    with pytest.raises(BlockAllocationError):
        SwatBlockAllocationConfig(security_lambda=512, privacy_epsilon=1.0,
                                  privacy_delta=1e-12, seed=seed)


def test_c_a_valid_small_lambda_is_accepted_when_the_decay_stays_positive():
    config = SwatBlockAllocationConfig(security_lambda=3, privacy_epsilon=1.0,
                                       privacy_delta=0.5, seed=0)
    assert config.decay > 0.0


def test_c_the_allocator_refuses_a_non_config_and_a_negative_block_count():
    with pytest.raises(BlockAllocationError):
        allocate_block_bins(4, {"security_lambda": 512})
    with pytest.raises(BlockAllocationError):
        allocate_block_bins(-1, swat_reference_config(0))
    with pytest.raises(BlockAllocationError):
        allocate_block_bins(True, swat_reference_config(0))


# ---------------------------------------------------------------------------
# D. the allocation plan
# ---------------------------------------------------------------------------


def test_d_zero_blocks_produces_an_empty_plan():
    plan = plan_for(0)
    assert plan.is_empty
    assert plan.block_count == 0
    assert plan.bin_count == 0
    assert plan.bins == ()
    assert plan.sampled_loads == ()
    assert plan.noisy_prefix_sums == (0,)
    assert plan.additive_error_blocks == 0
    assert plan.sampled_capacity == 0 and plan.dummy_count == 0
    # the geometry is still reported, because it is a property of the configuration
    assert plan.bin_capacity_blocks == REFERENCE_BIN_CAPACITY_BLOCKS
    assert plan.factor == pytest.approx(REFERENCE_FACTOR, rel=1e-15)


def test_d_one_block_is_planned_into_a_single_padded_bin():
    plan = plan_for(1)
    assert plan.block_count == 1
    assert plan.bin_count == 1
    assert len(plan.bins) == 1
    bin_ = plan.bins[0]
    assert bin_.bin_index == 0
    assert bin_.real_count == 1
    assert bin_.logical_rank_start == 0 and bin_.logical_rank_stop == 1
    assert bin_.real_count + bin_.dummy_count == plan.bin_capacity_blocks == 16


@pytest.mark.parametrize("blocks", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 21, 33, 64])
def test_d_small_block_counts_produce_a_well_formed_plan(blocks):
    plan = plan_for(blocks)
    assert plan.block_count == blocks
    assert plan.bin_count == compute_bin_count(plan.factor, blocks)
    assert plan.covered_ranks() == tuple(range(blocks))
    assert sum(bin_.real_count for bin_ in plan.bins) == blocks


@pytest.mark.parametrize("blocks", [1, 2, 4, 5, 8, 13, 16, 29, 40, 64, 100])
def test_d_every_bin_has_exactly_the_bin_capacity_slots(blocks):
    plan = plan_for(blocks)
    for bin_ in plan.bins:
        assert bin_.real_count + bin_.dummy_count == plan.bin_capacity_blocks
        assert bin_.capacity == plan.bin_capacity_blocks
        assert 0 <= bin_.sampled_load <= plan.bin_capacity_blocks


@pytest.mark.parametrize("blocks", [1, 2, 4, 8, 16, 33, 64])
def test_d_rank_intervals_are_contiguous_ordered_and_non_overlapping(blocks):
    plan = plan_for(blocks)
    expected_start = 0
    for index, bin_ in enumerate(plan.bins):
        assert bin_.bin_index == index
        assert bin_.logical_rank_start == expected_start
        assert bin_.logical_rank_stop == bin_.logical_rank_start + bin_.real_count
        expected_start = bin_.logical_rank_stop
    assert expected_start == blocks


def test_d_a_successful_plan_covers_every_rank_exactly_once():
    blocks = 40
    plan = plan_for(blocks)
    covered = plan.covered_ranks()
    assert covered == tuple(range(blocks))
    assert len(covered) == len(set(covered)) == blocks
    for rank in range(blocks):
        occurrences = sum(
            1 for bin_ in plan.bins if bin_.logical_rank_start <= rank < bin_.logical_rank_stop
        )
        assert occurrences == 1, rank


def test_d_the_final_exhaustion_separates_sampled_load_from_real_count():
    """At the end of the run a bin can be sampled far fuller than it is filled."""
    plan = plan_for(8, seed=7)
    loads = plan.sampled_loads
    reals = tuple(bin_.real_count for bin_ in plan.bins)
    assert loads == (9, 8)
    assert reals == (8, 0)
    assert reals[-1] != loads[-1]
    assert plan.bins[-1].dummy_count == plan.bin_capacity_blocks
    assert plan.sampled_capacity > plan.block_count


def test_d_the_plan_materialises_no_dummy_items():
    plan = plan_for(64)
    assert plan.dummy_count > 0
    # only counts and rank ranges are stored - no per-slot object exists anywhere
    for bin_ in plan.bins:
        assert isinstance(bin_, BlockBinPlan)
        assert isinstance(bin_.dummy_count, int)


def test_d_the_plan_exposes_no_physical_or_observational_field():
    plan = plan_for(16)
    for field in list(BlockAllocationPlan.__dataclass_fields__) + \
            list(BlockBinPlan.__dataclass_fields__):
        lowered = field.lower()
        for token in ("slot", "trace", "storage", "prp", "key", "record", "pgm", "shuffle"):
            assert token not in lowered, field


# ---------------------------------------------------------------------------
# E. randomness contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", [1, 8, 21, 64])
def test_e_the_same_seed_gives_the_identical_plan(blocks):
    first = plan_for(blocks, seed=20260920)
    second = plan_for(blocks, seed=20260920)
    assert first == second
    assert first.sampled_loads == second.sampled_loads
    assert first.noisy_prefix_sums == second.noisy_prefix_sums


def test_e_two_seeds_produce_distinct_stochastic_evidence():
    seeds = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    load_vectors = {allocate_block_bins(97, swat_reference_config(s)).sampled_loads
                    for s in seeds}
    assert len(load_vectors) > 1
    assert allocate_block_bins(21, swat_reference_config(7)).sampled_loads != \
        allocate_block_bins(21, swat_reference_config(9)).sampled_loads


def test_e_the_plan_is_reproducible_across_processes():
    program = (
        "import json\n"
        "from swat_m_block import allocate_block_bins, swat_reference_config\n"
        "plan = allocate_block_bins(21, swat_reference_config(20260920))\n"
        "print(json.dumps({'loads': list(plan.sampled_loads),\n"
        "                  'prefix': list(plan.noisy_prefix_sums),\n"
        "                  'error': plan.additive_error_blocks,\n"
        "                  'z': plan.bin_capacity_blocks}))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(CODES_DIR), env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    import json

    payload = json.loads(process.stdout)
    plan = plan_for(21, seed=20260920)
    assert payload["loads"] == list(plan.sampled_loads)
    assert payload["prefix"] == list(plan.noisy_prefix_sums)
    assert payload["error"] == plan.additive_error_blocks
    assert payload["z"] == plan.bin_capacity_blocks


def test_e_the_two_rng_domains_are_separated():
    assert STREAM_DOMAIN_LOADS != STREAM_DOMAIN_LAPLACE
    assert derive_stream_seed(7, STREAM_DOMAIN_LOADS) != \
        derive_stream_seed(7, STREAM_DOMAIN_LAPLACE)
    assert derive_stream_seed(7, STREAM_DOMAIN_LOADS) == \
        derive_stream_seed(7, STREAM_DOMAIN_LOADS)
    load_stream = GeometricLoadSampler(16, 1.0, seed=7)
    other_domain = GeometricLoadSampler(16, 1.0, seed=7, domain="another/domain")
    assert [load_stream.sample() for _ in range(50)] != \
        [other_domain.sample() for _ in range(50)]


def test_e_no_module_global_mutable_rng_state_exists():
    """Every stream is created inside the sampler that owns it, never at import time."""
    streams_module = (SWAT_PACKAGE / "distribution.py").read_text(encoding="utf-8")
    assert "import random" in streams_module
    for name in M2_MODULES:
        path = SWAT_PACKAGE / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                assert not (isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Attribute)
                            and getattr(value.func.value, "id", None) == "random"), \
                    f"{name}: module-level RNG state"
                assert not (isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)
                            and value.func.id == "Random"), f"{name}: module-level RNG state"
        # no module-level random.seed(...) call either
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "seed", f"{name}: module-level random.seed"


def test_e_the_plan_does_not_depend_on_unrelated_stream_usage():
    baseline = plan_for(21, seed=5)
    for _ in range(37):
        LaplaceSampler(seed=5).sample(1.0)
        GeometricLoadSampler(16, 1.0, seed=5).sample()
    assert plan_for(21, seed=5) == baseline


def test_e_every_random_stream_is_planner_owned_and_seeded():
    config = swat_reference_config(11)
    plan = allocate_block_bins(33, config)
    assert plan.config == config
    sampler = GeometricLoadSampler(plan.bin_capacity_blocks, config.privacy_epsilon,
                                   seed=config.seed)
    fresh = tuple(sampler.sample() for _ in range(plan.bin_count))
    assert fresh == plan.sampled_loads
    laplace = LaplaceSampler(seed=config.seed)
    prefix, error = compute_noisy_prefix_sums(
        plan.sampled_loads, privacy_epsilon=config.privacy_epsilon,
        bin_capacity_blocks=plan.bin_capacity_blocks, laplace=laplace,
    )
    assert prefix == plan.noisy_prefix_sums
    assert error == plan.additive_error_blocks


# ---------------------------------------------------------------------------
# F. explicit allocation failure
# ---------------------------------------------------------------------------


def test_f_insufficient_sampled_capacity_is_an_explicit_typed_failure():
    sampler = _ConstantLoadSampler(0)
    with pytest.raises(InsufficientSampledCapacity) as error:
        allocate_block_bins(20, swat_reference_config(1234), load_sampler=sampler)
    failure = error.value
    assert isinstance(failure, BlockAllocationError)
    assert failure.block_count == 20
    assert failure.sampled_capacity == 0
    assert failure.bin_count == compute_bin_count(REFERENCE_FACTOR, 20)
    assert failure.bin_capacity_blocks == REFERENCE_BIN_CAPACITY_BLOCKS
    assert failure.seed == 1234
    assert failure.shortfall_blocks == 20
    assert "cannot cover the run" in str(failure)


def test_f_a_partially_insufficient_sample_also_fails_explicitly():
    sampler = _ConstantLoadSampler(6)  # enough for some, never for all
    with pytest.raises(InsufficientSampledCapacity) as error:
        allocate_block_bins(1000, swat_reference_config(7), load_sampler=sampler)
    failure = error.value
    expected_capacity = 6 * compute_bin_count(REFERENCE_FACTOR, 1000)
    assert failure.sampled_capacity == expected_capacity
    assert failure.sampled_capacity < 1000
    assert failure.shortfall_blocks == 1000 - expected_capacity


def test_f_the_failure_is_not_silently_repaired():
    """No extra bin, no inflated load, no resampling until it fits."""
    sampler = _ConstantLoadSampler(0)
    with pytest.raises(InsufficientSampledCapacity):
        allocate_block_bins(20, swat_reference_config(3), load_sampler=sampler)
    bin_count = compute_bin_count(REFERENCE_FACTOR, 20)
    # exactly one sample per planned bin - the allocator never retries a bin
    assert sampler.calls == bin_count


def test_f_the_same_config_fails_the_same_way_every_time():
    for _ in range(5):
        with pytest.raises(InsufficientSampledCapacity) as error:
            allocate_block_bins(20, swat_reference_config(3),
                                load_sampler=_ConstantLoadSampler(0))
        assert error.value.sampled_capacity == 0
        assert error.value.bin_count == compute_bin_count(REFERENCE_FACTOR, 20)


def test_f_a_failure_never_yields_a_plan():
    result = None
    try:
        result = allocate_block_bins(20, swat_reference_config(1),
                                     load_sampler=_ConstantLoadSampler(1))
    except InsufficientSampledCapacity as failure:
        assert failure.shortfall_blocks > 0
    assert result is None


def test_f_an_out_of_range_sampled_load_is_refused():
    with pytest.raises(BlockAllocationError) as error:
        allocate_block_bins(4, swat_reference_config(1),
                            load_sampler=_ConstantLoadSampler(99))
    assert not isinstance(error.value, InsufficientSampledCapacity)
    assert "outside the permitted" in str(error.value)


def test_f_a_sampler_without_sample_is_refused():
    with pytest.raises(BlockAllocationError):
        allocate_block_bins(4, swat_reference_config(1), load_sampler=object())


def test_f_a_sufficient_injected_sampler_produces_an_exact_plan():
    sampler = _ConstantLoadSampler(16)
    plan = allocate_block_bins(24, swat_reference_config(2), load_sampler=sampler)
    assert plan.sampled_loads == (16, 16, 16, 16)  # 4 bins, factor * 24
    assert tuple(bin_.real_count for bin_ in plan.bins) == (16, 8, 0, 0)
    assert tuple(bin_.dummy_count for bin_ in plan.bins) == (0, 8, 16, 16)
    assert plan.covered_ranks() == tuple(range(24))
    assert sampler.calls == 4


# ---------------------------------------------------------------------------
# G. noisy prefix sums
# ---------------------------------------------------------------------------


def test_g_the_prefix_vector_has_bin_count_plus_one_entries():
    plan = plan_for(40)
    assert len(plan.noisy_prefix_sums) == plan.bin_count + 1
    assert plan.noisy_prefix_sums[0] == 0


@pytest.mark.parametrize("blocks,seed", [(8, 1), (21, 2), (64, 3), (100, 4), (200, 5)])
def test_g_the_noisy_prefix_is_monotone_and_clipped_into_the_permitted_window(blocks, seed):
    plan = plan_for(blocks, seed=seed)
    prefix = plan.noisy_prefix_sums
    assert list(prefix) == sorted(prefix)
    true_prefix = 0
    for t, load in enumerate(plan.sampled_loads, start=1):
        true_prefix += load
        assert abs(true_prefix - prefix[t]) <= plan.bin_capacity_blocks


@pytest.mark.parametrize("blocks,seed", [(8, 1), (21, 2), (64, 3), (100, 4), (200, 5)])
def test_g_the_additive_error_is_exactly_the_maximum_deviation(blocks, seed):
    plan = plan_for(blocks, seed=seed)
    true_prefix = 0
    worst = 0
    for t, load in enumerate(plan.sampled_loads, start=1):
        true_prefix += load
        worst = max(worst, abs(true_prefix - plan.noisy_prefix_sums[t]))
    assert plan.additive_error_blocks == worst


def test_g_the_prefix_of_an_empty_plan_is_the_single_zero():
    prefix, error = compute_noisy_prefix_sums(
        (), privacy_epsilon=1.0, bin_capacity_blocks=16, laplace=LaplaceSampler(seed=1)
    )
    assert prefix == (0,) and error == 0


def test_g_the_prefix_is_deterministic_for_a_fixed_stream():
    loads = (6, 7, 5, 8, 4, 9)
    first = compute_noisy_prefix_sums(loads, privacy_epsilon=1.0, bin_capacity_blocks=16,
                                      laplace=LaplaceSampler(seed=9))
    second = compute_noisy_prefix_sums(loads, privacy_epsilon=1.0, bin_capacity_blocks=16,
                                       laplace=LaplaceSampler(seed=9))
    assert first == second


def test_g_a_load_outside_the_support_is_refused():
    with pytest.raises(BlockAllocationError):
        compute_noisy_prefix_sums((17,), privacy_epsilon=1.0, bin_capacity_blocks=16,
                                  laplace=LaplaceSampler(seed=1))
    with pytest.raises(BlockAllocationError):
        compute_noisy_prefix_sums((-1,), privacy_epsilon=1.0, bin_capacity_blocks=16,
                                  laplace=LaplaceSampler(seed=1))


def test_g_a_foreign_noise_stream_is_refused():
    with pytest.raises(BlockAllocationError):
        compute_noisy_prefix_sums((3,), privacy_epsilon=1.0, bin_capacity_blocks=16,
                                  laplace=object())


def test_g_the_prefix_of_a_zero_load_vector_stays_monotone_and_bounded():
    loads = (0, 0, 0, 0)
    prefix, error = compute_noisy_prefix_sums(
        loads, privacy_epsilon=1.0, bin_capacity_blocks=16, laplace=LaplaceSampler(seed=4)
    )
    assert prefix[0] == 0
    assert list(prefix) == sorted(prefix)
    assert all(0 <= value <= 16 for value in prefix)
    assert error == max(prefix)


# ---------------------------------------------------------------------------
# H. independence from record contents / M1
# ---------------------------------------------------------------------------


def test_h_the_allocator_takes_only_a_block_count_and_a_config():
    signature = inspect.signature(allocate_block_bins)
    assert list(signature.parameters) == ["block_count", "config", "load_sampler"]
    for name in ("records", "keys", "values", "level", "source", "target", "pgm"):
        assert name not in signature.parameters


def test_h_m2_does_not_import_the_m1_oracle_or_the_frozen_merge():
    for name in M2_MODULES:
        tree = ast.parse((SWAT_PACKAGE / name).read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        expected = {
            "distribution.py": {"__future__", "bisect", "hashlib", "math", "random",
                                "typing"},
            "bin_allocator.py": {"__future__", "math", "dataclasses", "typing",
                                 "distribution"},
        }
        assert modules == expected[name], (name, modules)
        joined = " ".join(modules)
        for forbidden in ("functional_oracle", "incremental_merge", "engine", "storage",
                          "trace", "block", "pgm"):
            assert forbidden not in joined, (name, forbidden)


def test_h_no_representative_key_or_interior_point_appears_in_m2():
    tokens = ("first_key", "last_key", "midpoint", "representative", "interior",
              "RecordKey", "record_key", "block_key")
    for name in M2_MODULES:
        code = _code_without_docstrings(SWAT_PACKAGE / name).lower()
        for token in tokens:
            assert token.lower() not in code, (name, token)


def test_h_the_plan_depends_only_on_the_block_count_and_config():
    """Two runs with the same shape allocate identically: M2 never sees a key or value."""
    first = plan_for(33, seed=42)
    second = plan_for(33, seed=42)
    assert first == second
    # the plan cannot carry content: it stores counts and rank intervals only
    assert first.covered_ranks() == tuple(range(33))


# ---------------------------------------------------------------------------
# I. scope guard: no physical/observable surface, no SWAT mechanism
# ---------------------------------------------------------------------------


def test_i_no_m2_module_contains_a_forbidden_mechanism():
    for name in M2_MODULES:
        code = _code_without_docstrings(SWAT_PACKAGE / name)
        for token in FORBIDDEN_M2_TOKENS:
            assert token not in code, (name, token)


def test_i_no_m2_module_references_a_physical_or_observational_identifier():
    for name in M2_MODULES:
        tree = ast.parse((SWAT_PACKAGE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in {"SlotId", "TraceEvent", "TraceOperation",
                                       "TraceCollector", "UntrustedStorage"}, name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"read", "write", "store", "allocate",
                                         "clear"}, (name, node.attr)


def test_i_planning_performs_zero_physical_io_and_emits_zero_trace_events(monkeypatch):
    from enhanced_letindex.storage import UntrustedStorage
    from enhanced_letindex.trace import TraceCollector

    physical_calls: list = []

    def explode(method):
        def recorder(self, *args, **kwargs):
            physical_calls.append(method)
            raise AssertionError(f"M2 performed a physical {method}")

        return recorder

    for method in ("allocate", "store", "read", "write", "clear"):
        monkeypatch.setattr(UntrustedStorage, method, explode(method))

    recorded: list = []
    original_record = TraceCollector.record

    def counting_record(self, operation, slot_id, level_id=None, metadata=None):
        recorded.append((operation, slot_id))
        return original_record(self, operation, slot_id, level_id, metadata)

    monkeypatch.setattr(TraceCollector, "record", counting_record)

    storage = UntrustedStorage()
    plans = [plan_for(blocks, seed=seed)
             for blocks, seed in ((0, 1), (1, 2), (8, 3), (21, 4), (64, 5))]

    assert [plan.block_count for plan in plans] == [0, 1, 8, 21, 64]
    assert physical_calls == []
    assert recorded == []
    assert storage.size == 0 and storage.trace.events() == ()


def test_i_the_allocator_never_constructs_a_physical_object():
    """An allocation plan is a plan: no storage, no slot, no event is created."""
    plan = plan_for(40)
    assert plan.planned_slots == plan.bin_count * plan.bin_capacity_blocks
    assert not hasattr(plan, "slots")
    assert not hasattr(plan, "slot_ids")
    assert not hasattr(plan, "trace")


def test_i_m2_adds_no_third_module_and_keeps_the_m1_oracle_intact():
    assert {path.name for path in SWAT_PACKAGE.glob("*.py")} == {
        "__init__.py", "functional_oracle.py", "distribution.py", "bin_allocator.py"
    }


# ---------------------------------------------------------------------------
# J. the M0 frozen substrate is untouched
# ---------------------------------------------------------------------------


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("m2_verify_manifest", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_j_the_35_file_frozen_substrate_manifest_still_verifies():
    entries = [line.split() for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(entries) == 35
    assert _load_verify_module().main(["--quiet", "--root", str(REPO_ROOT)]) == 0


def test_j_the_m2_implementation_lives_outside_the_frozen_package():
    frozen = {line.split()[1] for line in
              MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert not any("swat_m_block" in relative for relative in frozen)
    assert (SWAT_PACKAGE / "distribution.py").is_file()
    assert (SWAT_PACKAGE / "bin_allocator.py").is_file()
