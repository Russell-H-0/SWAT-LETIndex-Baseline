"""M3 focused tests: the content-bound SWAT-M-Block bin schedule (Issue #6).

Covered: the trusted block-run view and its validation, exact M2 plan-to-content binding,
the pinned interior-point weighting re-derived independently, the ``DUMMY_POS_INF``
sentinel, domain-separated deterministic interior streams, the pinned signed-tag ordering,
the abstract bin-read schedule, the relationship to the M1 oracle and to M2's block-unit
prefix evidence, and the absence of every physical/observable surface.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from enhanced_letindex.block import Block
from enhanced_letindex.identifiers import BlockId, RecordKey
from enhanced_letindex.incremental_merge import LogicalRunView
from enhanced_letindex.pgm import build_batch_pgm
from enhanced_letindex.record import Record

from swat_m_block import (
    SwatBlockAllocationConfig,
    allocate_block_bins,
    functional_merge_oracle,
    swat_reference_config,
)
from swat_m_block.content_schedule import (
    DUMMY_POS_INF,
    SOURCE,
    STREAM_DOMAIN_INTERIOR_SOURCE,
    STREAM_DOMAIN_INTERIOR_TARGET,
    TARGET,
    AbstractBinRead,
    BoundBlockAllocation,
    BoundBlockBin,
    ContentScheduleError,
    InteriorPoint,
    LogicalBlockRunView,
    LogicalBlockSnapshot,
    SwatBlockMergeSchedule,
    TaggedBinInteriorPoint,
    bind_block_allocation,
    build_abstract_merge_schedule,
    interior_point_sort_key,
    interior_point_weights,
    interior_stream_domain,
    plan_swat_block_merge_schedule,
    sample_bin_interior_points,
    sorted_tagged_interior_points,
)
from swat_m_block.distribution import STREAM_DOMAIN_LAPLACE, STREAM_DOMAIN_LOADS
from swat_m_block.distribution import derive_stream_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "codes"
SRC_DIR = CODES_DIR / "src"
SWAT_PACKAGE = SRC_DIR / "swat_m_block"
MANIFEST_PATH = REPO_ROOT / "provenance" / "common-substrate.sha256"
VERIFY_SCRIPT = REPO_ROOT / "provenance" / "verify_manifest.py"

M3_MODULE = "content_schedule.py"
IPB = 4
PGM_EPSILON = 8

#: Mechanisms M3 still must not contain (M3 authorises content binding, interior points and
#: the abstract schedule; everything physical and every output-side mechanism stays out).
FORBIDDEN_M3_TOKENS = (
    "DOAllocate", "DOMerge", "DOMerger", "do_allocate", "do_merge",
    "UntrustedStorage", "TraceEvent", "TraceOperation", "TraceCollector", "SlotId",
    "output_shuffle", "oblivious_shuffle", "bitonic", "sorting_network",
    "prp_writeback", "deamortiz", "de_amortiz", "newer_wins", "read_slot", "write_slot",
    "physical_slot", "ciphertext", "sgx_", "safe_output", "bounded_buffer",
)

#: Length of the source/target fixtures below.
SOURCE_KEYS = [10 * (index + 1) for index in range(32)]                 # 8 blocks
TARGET_KEYS = [10 * (index + 1) + 5 for index in range(20)]             # 5 blocks


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def make_block(block_id: int, keys, *, ipb: int = IPB, value_of=None) -> Block:
    records = [
        Record(RecordKey(key), value_of(key) if value_of else f"v{key}") for key in keys
    ]
    return Block.sorted_block(BlockId(block_id), records, ipb)


def make_run(level: int, keys, *, ipb: int = IPB, id_base: int = 0, value_of=None):
    blocks = [
        make_block(id_base + index // ipb, keys[index:index + ipb], ipb=ipb,
                   value_of=value_of)
        for index in range(0, len(keys), ipb)
    ]
    return LogicalBlockRunView(level=level, blocks=tuple(blocks), items_per_block=ipb)


def make_run_with_blocks(level, keys, *, ipb=IPB, id_base=0):
    """The same run, plus the caller-owned mutable Block objects behind it."""
    blocks = [
        make_block(id_base + index // ipb, keys[index:index + ipb], ipb=ipb)
        for index in range(0, len(keys), ipb)
    ]
    return LogicalBlockRunView(level, tuple(blocks), ipb), blocks


def make_run_with_bin_starts(level, bin_start_keys, block_count, id_base, *, ipb=IPB):
    """A run whose bin-starting blocks begin at requested first keys (fallback fixture)."""
    keys = []
    next_key = 1
    for block_index in range(block_count):
        start = bin_start_keys.get(block_index, next_key)
        block_keys = [start + offset for offset in range(ipb)]
        keys.extend(block_keys)
        next_key = block_keys[-1] + 10
    return make_run(level, keys, ipb=ipb, id_base=id_base)


def pinned_projected_fetches(source_bins, target_bins, tagged):
    """The pinned ``DOMerge`` loop, projected onto the actual bin-fetch sequence.

    This mirrors ``DOMerger.hpp`` independently of the M3 implementation: the two preloads,
    then one iteration per sorted tagged interior point, where a left tag fetches the left
    side's next bin while one remains and **otherwise** the right side's next bin is
    fetched.  That ``elif`` is a real fallback: a left tag arriving after the left side is
    exhausted fetches from the right side.  No-op iterations are projected away, and the
    deferred safe-output work of the pinned loop is not modelled.
    """
    fetches = []
    fallbacks = []
    if source_bins > 0:
        fetches.append((SOURCE, 0))
    if target_bins > 0:
        fetches.append((TARGET, 0))
    left_next, right_next = -1, -1
    for side, _key_order in tagged:
        is_left_tag = side == SOURCE
        if is_left_tag and left_next + 2 < source_bins:
            left_next += 1
            fetches.append((SOURCE, left_next + 1))
        elif right_next + 2 < target_bins:
            right_next += 1
            fetches.append((TARGET, right_next + 1))
            if is_left_tag:
                fallbacks.append((SOURCE, right_next + 1))
    return fetches, fallbacks


def plan_for(blocks: int, seed: int = 7):
    return allocate_block_bins(blocks, swat_reference_config(seed))


def fixture(seed_source: int = 7, seed_target: int = 7, ipb: int = IPB):
    source = make_run(4, SOURCE_KEYS, ipb=ipb)
    target = make_run(5, TARGET_KEYS, ipb=ipb, id_base=100)
    return (source, target, plan_for(source.block_count, seed_source),
            plan_for(target.block_count, seed_target))


def structure(schedule: SwatBlockMergeSchedule):
    """The schedule *structure*: never a record value, only keys and ordering."""
    return (
        tuple((read.side, read.bin_index) for read in schedule.reads),
        tuple((point.side, point.bin_index, point.signed_tag,
               repr(point.interior_point.key)) for point in schedule.tagged_interior_points),
    )


def independent_weights(load: int, epsilon: float):
    """The pinned ``DPInteriorPoint`` weights, written out again independently."""
    base = math.exp(epsilon)
    return [base ** (min(index, load - index) + 1) for index in range(load)]


def independent_fast_power(base: float, exponent: int) -> float:
    """The pinned ``fastPower`` loop, written out again independently."""
    result = 1.0
    value = base
    exp = exponent
    while exp > 0:
        if exp & 1:
            result *= value
        exp >>= 1
        value *= value
    return result


def _code_without_docstrings(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                source = source.replace(doc, "")
    return source


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("m3_verify_manifest", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A. LogicalBlockRunView
# ---------------------------------------------------------------------------


def test_a_an_empty_run_is_legal_and_empty():
    run = LogicalBlockRunView(level=4, blocks=(), items_per_block=IPB)
    assert run.is_empty
    assert run.block_count == 0
    assert run.item_count == 0
    assert run.records() == () and run.keys() == ()


def test_a_one_block_is_legal_and_partial_is_allowed_for_the_final_block():
    run = make_run(4, [1, 2, 3])
    assert run.block_count == 1
    assert run.blocks[-1].size == 3
    assert run.item_count == 3
    assert run.keys() == (RecordKey(1), RecordKey(2), RecordKey(3))


def test_a_canonical_run_keeps_full_blocks_and_a_final_partial_block():
    run = make_run(4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert run.block_count == 3
    assert [block.size for block in run.blocks] == [4, 4, 2]
    for block in run.blocks[:-1]:
        assert block.is_full
    assert run.keys() == tuple(RecordKey(value) for value in range(1, 11))


def test_a_the_view_preserves_the_real_block_boundaries():
    run = make_run(4, [1, 2, 3, 4, 9, 10, 11, 12])
    assert [block.block_id for block in run.blocks] == [BlockId(0), BlockId(1)]
    assert run.blocks[0].max_key < run.blocks[1].min_key
    assert run.records_of(0, 1) == run.blocks[0].records
    assert run.records_of(1, 2) == run.blocks[1].records


def test_a_neighbouring_overlap_is_refused():
    left = make_block(0, [1, 2, 3, 4])
    right = make_block(1, [4, 5, 6, 7])          # overlaps on key 4
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=(left, right), items_per_block=IPB)
    assert "must not overlap" in str(error.value)


def test_a_blocks_in_the_wrong_logical_order_are_refused():
    high = make_block(0, [9, 10, 11, 12])
    low = make_block(1, [1, 2, 3, 4])
    with pytest.raises(ContentScheduleError):
        LogicalBlockRunView(level=4, blocks=(high, low), items_per_block=IPB)


def test_a_a_non_final_partial_block_is_refused():
    first = make_block(0, [1, 2, 3, 4])
    middle = make_block(1, [5, 6])
    last = make_block(2, [7, 8, 9, 10])
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=(first, middle, last), items_per_block=IPB)
    assert "must be full" in str(error.value)


def test_a_a_final_partial_block_is_legal():
    run = make_run(4, [1, 2, 3, 4, 5])
    assert run.block_count == 2
    assert [block.size for block in run.blocks] == [4, 1]


def test_a_an_inconsistent_capacity_is_refused():
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=(make_block(0, [1, 2, 3, 4]),),
                            items_per_block=8)
    assert "items_per_block" in str(error.value)


def test_a_an_empty_block_is_refused():
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=(Block(BlockId(0), IPB),),
                            items_per_block=IPB)
    assert "is empty" in str(error.value)


def test_a_duplicate_block_ids_are_refused():
    first = make_block(7, [1, 2, 3, 4])
    second = make_block(7, [5, 6, 7, 8])
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=(first, second), items_per_block=IPB)
    assert "unique" in str(error.value)


def test_a_a_non_block_element_is_refused():
    with pytest.raises(ContentScheduleError) as error:
        LogicalBlockRunView(level=4, blocks=("not a block",), items_per_block=IPB)
    assert "Block objects" in str(error.value)


@pytest.mark.parametrize("level", [-1, True, "4", 1.5, None])
def test_a_an_invalid_level_is_refused(level):
    with pytest.raises(ContentScheduleError):
        LogicalBlockRunView(level=level, blocks=(), items_per_block=IPB)


@pytest.mark.parametrize("capacity", [0, -4, True, "4"])
def test_a_an_invalid_items_per_block_is_refused(capacity):
    with pytest.raises(ContentScheduleError):
        LogicalBlockRunView(level=4, blocks=(), items_per_block=capacity)


def test_a_an_out_of_range_record_query_is_refused():
    run = make_run(4, [1, 2, 3, 4])
    with pytest.raises(ContentScheduleError):
        run.records_of(0, 2)
    with pytest.raises(ContentScheduleError):
        run.records_of(-1, 1)


# ---------------------------------------------------------------------------
# B. binding an M2 plan to contents
# ---------------------------------------------------------------------------


def test_b_binding_covers_every_real_block_rank_exactly_once():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    assert bound.bound_ranks() == tuple(range(source.block_count))
    assert len(bound.bound_ranks()) == len(set(bound.bound_ranks()))
    assert bound.block_count == source.block_count
    assert bound.bin_count == source_plan.bin_count


def test_b_binding_reconstructs_the_exact_run_records():
    source, target, source_plan, target_plan = fixture()
    for run, plan, side in ((source, source_plan, SOURCE), (target, target_plan, TARGET)):
        bound = bind_block_allocation(run, plan, side=side)
        assert bound.real_records() == run.records()
        assert bound.real_keys() == run.keys()
        assert len(bound.real_records()) == run.item_count


def test_b_binding_keeps_the_m2_bin_order_and_the_rank_intervals():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    for bin_, bound_bin in zip(source_plan.bins, bound.bins):
        assert bound_bin.bin_index == bin_.bin_index
        assert bound_bin.logical_rank_start == bin_.logical_rank_start
        assert bound_bin.logical_rank_stop == bin_.logical_rank_stop
        assert [block.block_id for block in bound_bin.blocks] == [
            block.block_id
            for block in source.blocks[bin_.logical_rank_start:bin_.logical_rank_stop]
        ]


def test_b_a_zero_real_bin_binds_no_block_but_keeps_its_dummy_count():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    empties = [bin_ for bin_ in bound.bins if bin_.is_empty_real]
    assert empties, "the fixture must contain an exhausted trailing bin"
    for bin_ in empties:
        assert bin_.blocks == ()
        assert bin_.real_block_count == 0
        assert bin_.dummy_block_count == bin_.bin_capacity_blocks
    assert sum(bin_.real_block_count for bin_ in bound.bins) == source.block_count


def test_b_the_m2_evidence_is_preserved_and_never_rewritten():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    # the same objects are carried through, so the evidence cannot be rewritten
    assert bound.sampled_loads is source_plan.sampled_loads
    assert bound.noisy_prefix_sums is source_plan.noisy_prefix_sums
    assert bound.additive_error_blocks == source_plan.additive_error_blocks
    # the M2 plan itself is untouched (it is frozen, and binding copies nothing into it)
    assert source_plan.bins == tuple(source_plan.bins)
    assert source_plan.block_count == source.block_count


def test_b_no_dummy_block_is_ever_materialised():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    for bin_ in bound.bins:
        assert len(bin_.blocks) == bin_.real_block_count
        assert all(isinstance(block, LogicalBlockSnapshot) for block in bin_.blocks)
    assert not hasattr(bound, "dummy_blocks")
    assert not hasattr(bound.bins[0], "dummy_blocks")
    assert bound.plan.dummy_count == sum(bin_.dummy_block_count for bin_ in bound.bins)


def test_a_the_view_snapshots_caller_owned_blocks():
    """A frozen view over mutable Blocks must not stay reachable through them."""
    original = make_block(0, [10, 11, 12])            # capacity 4: one spare slot
    run = LogicalBlockRunView(level=4, blocks=(original,), items_per_block=IPB)
    snapshot = run.blocks[0]
    assert isinstance(snapshot, LogicalBlockSnapshot)
    assert snapshot is not original
    before = run.records()
    original.add_record(Record(RecordKey(1), "INJECTED"))    # below the block's min key
    assert run.records() == before
    assert run.keys() == tuple(record.key for record in before)
    assert snapshot.size == 3
    assert snapshot.min_key == RecordKey(10)
    assert snapshot.max_key == RecordKey(12)


def test_a_a_later_block_mutation_cannot_change_view_binding_interiors_or_schedule():
    """Mutation across a bin's block boundary cannot escape the snapshot."""
    source_run, source_blocks = make_run_with_blocks(4, [10, 11, 12, 13, 20, 21, 22])
    target_run, target_blocks = make_run_with_blocks(5, [5, 6, 7, 8, 30, 31, 32],
                                                     id_base=100)
    source_plan, target_plan = plan_for(2, 7), plan_for(2, 7)
    assert source_plan.bin_count == 1        # one bin binds both blocks
    assert source_plan.bins[0].real_count == 2
    baseline = plan_swat_block_merge_schedule(
        source_run, target_run, source_plan=source_plan, target_plan=target_plan)
    baseline_records = source_run.records()
    baseline_bound = bind_block_allocation(source_run, source_plan, side=SOURCE)

    # mutate the caller-owned originals: the second source block gains a key *below* the
    # first block's max key, which would break the run's cross-block order if the view
    # still pointed at the live blocks
    source_blocks[1].add_record(Record(RecordKey(9), "INJECTED"))
    target_blocks[1].add_record(Record(RecordKey(40), "INJECTED"))
    assert source_blocks[1].min_key == RecordKey(9)

    assert source_run.records() == baseline_records
    assert source_run.keys() == tuple(record.key for record in baseline_records)
    assert bind_block_allocation(source_run, source_plan, side=SOURCE) == baseline_bound
    again = plan_swat_block_merge_schedule(
        source_run, target_run, source_plan=source_plan, target_plan=target_plan)
    assert structure(again) == structure(baseline)
    assert again.source.interiors == baseline.source.interiors
    assert again.target.interiors == baseline.target.interiors
    assert again.reads == baseline.reads


def test_b_a_plan_for_a_different_block_count_is_refused():
    source, _target, _ignored, _ = fixture()
    with pytest.raises(ContentScheduleError) as error:
        bind_block_allocation(source, plan_for(source.block_count + 1), side=SOURCE)
    assert "binds only the run it was allocated for" in str(error.value)


def test_b_binding_refuses_non_plan_and_non_run_inputs():
    source, _target, source_plan, _ = fixture()
    with pytest.raises(ContentScheduleError):
        bind_block_allocation(source, {"block_count": 8}, side=SOURCE)
    with pytest.raises(ContentScheduleError):
        bind_block_allocation("run", source_plan, side=SOURCE)
    with pytest.raises(ContentScheduleError):
        bind_block_allocation(source, source_plan, side="middle")


def test_b_the_bound_allocation_exposes_no_physical_field():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    for field in (list(BoundBlockAllocation.__dataclass_fields__)
                  + list(BoundBlockBin.__dataclass_fields__)):
        lowered = field.lower()
        for token in ("slot", "storage", "trace", "prp", "shuffle"):
            assert token not in lowered, field


# ---------------------------------------------------------------------------
# C. the pinned interior-point weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("load", [1, 2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.5])
def test_c_the_pinned_weight_expression_is_reproduced_exactly(load, epsilon):
    assert interior_point_weights(load, epsilon) == pytest.approx(
        independent_weights(load, epsilon), rel=1e-12)


def test_c_the_weights_use_the_pinned_recurrence_not_a_symmetrised_formula():
    """``baseExp ** (min(i, load - i) + 1)`` has an asymmetric tail; keep it exactly."""
    base = math.exp(1.0)
    weights = interior_point_weights(4, 1.0)
    assert weights == pytest.approx(
        (base ** 1, base ** 2, base ** 3, base ** 2), rel=1e-12)
    # a "symmetrised" formula such as base ** (min(i, load - 1 - i) + 1) would give
    # (b, b^2, b^2, b) and is explicitly NOT what is implemented
    assert weights[2] != pytest.approx(base ** 2, rel=1e-6)
    assert weights[1] == pytest.approx(weights[3], rel=1e-12)
    assert weights[0] != pytest.approx(weights[3], rel=1e-6)


def test_c_the_weights_are_exact_pinned_fast_power_values():
    weights = interior_point_weights(5, 1.0)
    expected = [independent_fast_power(math.exp(1.0), min(i, 5 - i) + 1) for i in range(5)]
    assert weights == tuple(expected)


def test_c_the_weight_vector_is_empty_for_an_empty_bin():
    assert interior_point_weights(0, 1.0) == ()


def test_c_the_middle_record_carries_the_most_weight():
    weights = interior_point_weights(9, 1.0)
    assert weights.index(max(weights)) == 4


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_c_an_invalid_epsilon_is_refused(epsilon):
    with pytest.raises(ContentScheduleError):
        interior_point_weights(4, epsilon)


@pytest.mark.parametrize("load", [-1, True, 1.5, "4"])
def test_c_an_invalid_load_is_refused(load):
    with pytest.raises(ContentScheduleError):
        interior_point_weights(load, 1.0)


# ---------------------------------------------------------------------------
# D. sampled interior points
# ---------------------------------------------------------------------------


def test_d_every_real_bin_interior_point_is_an_actual_key_of_that_bin():
    source, _target, source_plan, _ = fixture()
    sampled = sample_bin_interior_points(
        bind_block_allocation(source, source_plan, side=SOURCE))
    real_bins = 0
    for bin_, point in zip(sampled.bins, sampled.interiors):
        if bin_.is_empty_real:
            assert point.is_dummy
            continue
        real_bins += 1
        assert not point.is_dummy
        assert bin_.real_keys()[point.record_index] == point.key
        assert point.key in bin_.real_keys()
        assert isinstance(point.key, RecordKey)
        assert point.record_key == point.key
    assert real_bins > 0


def test_d_an_empty_real_bin_yields_the_dummy_sentinel():
    source, _target, source_plan, _ = fixture()
    sampled = sample_bin_interior_points(
        bind_block_allocation(source, source_plan, side=SOURCE))
    for bin_, point in zip(sampled.bins, sampled.interiors):
        if bin_.is_empty_real:
            assert point.is_dummy and point.key is DUMMY_POS_INF
            assert point.record_index is None
        else:
            assert not point.is_dummy


def test_d_sampling_twice_is_refused():
    source, _target, source_plan, _ = fixture()
    sampled = sample_bin_interior_points(
        bind_block_allocation(source, source_plan, side=SOURCE))
    with pytest.raises(ContentScheduleError):
        sample_bin_interior_points(sampled)


def test_d_the_narrow_index_sampler_seam_is_honoured():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    for index in (0, 1):
        sampled = sample_bin_interior_points(
            bound,
            interior_index_sampler=lambda weights, index=index: min(index, len(weights) - 1))
        for bin_, point in zip(sampled.bins, sampled.interiors):
            if bin_.is_empty_real:
                assert point.is_dummy
            else:
                assert point.record_index == min(index, len(bin_.real_keys()) - 1)


@pytest.mark.parametrize("bad", [-1, 10 ** 9, "0", 1.5, True])
def test_d_an_out_of_range_index_sampler_is_refused(bad):
    source, _target, source_plan, _ = fixture()
    with pytest.raises(ContentScheduleError):
        sample_bin_interior_points(
            bind_block_allocation(source, source_plan, side=SOURCE),
            interior_index_sampler=lambda weights, bad=bad: bad)


def test_d_the_public_sampler_exposes_no_randomness_parameter():
    """The public helper cannot be given an epsilon, a seed or a domain."""
    parameters = inspect.signature(sample_bin_interior_points).parameters
    assert list(parameters) == ["bound", "interior_index_sampler"]
    for forbidden in ("privacy_epsilon", "seed", "domain", "epsilon"):
        assert forbidden not in parameters, forbidden


def test_d_the_frozen_plan_config_is_the_only_source_of_epsilon_and_seed():
    """Only ``bound.plan.config`` decides the weights and the stream position."""
    source, _target, _plan, _ = fixture()
    run = source
    plan_a = plan_for(run.block_count, 5)
    sampled_a = sample_bin_interior_points(
        bind_block_allocation(run, plan_a, side=SOURCE))
    # a different config seed changes the draws ...
    plan_b = plan_for(run.block_count, 9)
    sampled_b = sample_bin_interior_points(
        bind_block_allocation(run, plan_b, side=SOURCE))
    assert plan_a.config.seed != plan_b.config.seed
    assert sampled_a.interiors != sampled_b.interiors
    # ... and a different privacy_epsilon changes the pinned weights
    other = SwatBlockAllocationConfig(
        security_lambda=512, privacy_epsilon=0.25, privacy_delta=1e-12, seed=5)
    plan_c = allocate_block_bins(run.block_count, other)
    sampled_c = sample_bin_interior_points(
        bind_block_allocation(run, plan_c, side=SOURCE))
    assert plan_c.config.privacy_epsilon == 0.25
    assert sampled_c.interiors != sampled_a.interiors
    # the same config reproduces the same evidence exactly
    assert sample_bin_interior_points(
        bind_block_allocation(run, plan_a, side=SOURCE)).interiors == sampled_a.interiors


def test_d_the_entry_point_takes_each_sides_own_plan_config():
    source, target, source_plan, target_plan = fixture(seed_source=7, seed_target=11)
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert schedule.source.interiors == sample_bin_interior_points(
        bind_block_allocation(source, source_plan, side=SOURCE)).interiors
    assert schedule.target.interiors == sample_bin_interior_points(
        bind_block_allocation(target, target_plan, side=TARGET)).interiors


def test_d_a_non_callable_sampler_is_refused():
    source, _target, source_plan, _ = fixture()
    with pytest.raises(ContentScheduleError):
        sample_bin_interior_points(
            bind_block_allocation(source, source_plan, side=SOURCE),
            interior_index_sampler=42)


def test_d_a_non_allocation_is_refused():
    with pytest.raises(ContentScheduleError):
        sample_bin_interior_points("bound")


# ---------------------------------------------------------------------------
# E. the DUMMY_POS_INF sentinel
# ---------------------------------------------------------------------------


def test_e_the_sentinel_is_not_a_record_key_and_refuses_to_be_one():
    assert not isinstance(DUMMY_POS_INF, RecordKey)
    assert repr(DUMMY_POS_INF) == "DUMMY_POS_INF"
    point = InteriorPoint(DUMMY_POS_INF)
    assert point.is_dummy
    with pytest.raises(ContentScheduleError):
        point.record_key


def test_e_the_sentinel_sorts_after_every_real_key():
    sentinel_key = interior_point_sort_key(InteriorPoint(DUMMY_POS_INF))
    for value in (-10 ** 18, -1, 0, 1, 10 ** 18):
        assert interior_point_sort_key(InteriorPoint(RecordKey(value))) < sentinel_key


def test_e_a_mixed_sequence_orders_reals_then_the_sentinel():
    points = [
        InteriorPoint(DUMMY_POS_INF),
        InteriorPoint(RecordKey(5), 0),
        InteriorPoint(RecordKey(-3), 0),
    ]
    ordered = sorted(points, key=interior_point_sort_key)
    assert [point.key.value for point in ordered[:-1]] == [-3, 5]
    assert ordered[-1].is_dummy


def test_e_an_interior_point_must_be_a_key_or_the_sentinel():
    with pytest.raises(ContentScheduleError):
        InteriorPoint("not a key")
    with pytest.raises(ContentScheduleError):
        InteriorPoint(DUMMY_POS_INF, 3)
    with pytest.raises(ContentScheduleError):
        InteriorPoint(RecordKey(1), -1)


# ---------------------------------------------------------------------------
# F. tagging and pinned order
# ---------------------------------------------------------------------------


def test_f_signed_tags_follow_the_pinned_dom_merge_convention():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for point in schedule.source.tagged_interior_points:
        assert point.side == SOURCE
        assert point.signed_tag == point.bin_index + 1
    for point in schedule.target.tagged_interior_points:
        assert point.side == TARGET
        assert point.signed_tag == -(point.bin_index + 1)


def test_f_a_wrong_signed_tag_is_refused():
    with pytest.raises(ContentScheduleError):
        TaggedBinInteriorPoint(side=SOURCE, bin_index=2,
                               interior_point=InteriorPoint(RecordKey(1)), signed_tag=-3)
    with pytest.raises(ContentScheduleError):
        TaggedBinInteriorPoint(side=TARGET, bin_index=0,
                               interior_point=InteriorPoint(RecordKey(1)), signed_tag=1)


def test_f_the_sorted_tagged_sequence_matches_an_independent_pair_sort():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    def pair_of(point):
        key = point.interior_point.key
        if isinstance(key, RecordKey):
            return ((0, key.value), point.signed_tag)
        return ((1, 0), point.signed_tag)          # the sentinel sorts after every key

    pairs = [pair_of(point) for point in schedule.tagged_interior_points]
    assert pairs == sorted(pairs)


def test_f_an_equal_interior_point_places_the_target_tag_before_the_source_tag():
    """The pinned ``std::pair`` ordering: equal key, then ascending signed tag."""
    source = make_run(4, [10, 20, 30, 40, 50, 60, 70, 80,
                          90, 100, 110, 120, 130, 140, 150, 160])
    target = make_run(5, [10, 25, 35, 45, 55, 65, 75, 85], id_base=200)
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=plan_for(source.block_count, 3),
        target_plan=plan_for(target.block_count, 3),
        interior_index_sampler=lambda weights: 0)
    tagged = schedule.tagged_interior_points
    assert [point.interior_point.key.value for point in tagged] == [10, 10]
    assert [point.side for point in tagged] == [TARGET, SOURCE]
    assert [point.signed_tag for point in tagged] == [-1, 1]
    # ... and an independent (key, tag) sort agrees
    assert ([(point.interior_point.key.value, point.signed_tag) for point in tagged]
            == sorted((point.interior_point.key.value, point.signed_tag)
                      for point in tagged))


def test_f_the_sentinel_sorts_last_in_the_tagged_sequence():
    source, _target, source_plan, _ = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, make_run(5, (), id_base=200), source_plan=source_plan,
        target_plan=plan_for(0))
    keys = [interior_point_sort_key(point.interior_point)
            for point in schedule.tagged_interior_points]
    assert keys == sorted(keys)
    assert schedule.tagged_interior_points[-1].interior_point.is_dummy


def test_f_the_tagged_sequence_holds_one_point_per_bin():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert len(schedule.tagged_interior_points) == source_plan.bin_count + target_plan.bin_count
    assert schedule.tagged_interior_points == sorted_tagged_interior_points(
        schedule.source, schedule.target)


# ---------------------------------------------------------------------------
# G. randomness contract
# ---------------------------------------------------------------------------


def test_g_the_interior_domain_is_fixed_by_the_side():
    assert interior_stream_domain(SOURCE) == STREAM_DOMAIN_INTERIOR_SOURCE
    assert interior_stream_domain(TARGET) == STREAM_DOMAIN_INTERIOR_TARGET
    with pytest.raises(ContentScheduleError):
        interior_stream_domain("middle")


def test_g_the_interior_domains_are_separate_from_each_other_and_from_m2():
    domains = (STREAM_DOMAIN_INTERIOR_SOURCE, STREAM_DOMAIN_INTERIOR_TARGET,
               STREAM_DOMAIN_LOADS, STREAM_DOMAIN_LAPLACE)
    assert len(set(domains)) == len(domains)
    seeds = {derive_stream_seed(7, domain) for domain in domains}
    assert len(seeds) == len(domains)


def test_g_the_same_config_seed_still_gives_the_two_sides_different_streams():
    """Identical content and identical config seed, but distinct domains."""
    keys = [10 * (index + 1) for index in range(80)]     # 20 blocks -> many interior draws
    run = make_run(4, keys)
    plan = plan_for(run.block_count, 5)
    as_source = sample_bin_interior_points(
        bind_block_allocation(run, plan, side=SOURCE))
    as_target = sample_bin_interior_points(
        bind_block_allocation(run, plan, side=TARGET))
    assert as_source.bin_count == as_target.bin_count
    assert as_source.interiors != as_target.interiors
    assert derive_stream_seed(5, STREAM_DOMAIN_INTERIOR_SOURCE) != \
        derive_stream_seed(5, STREAM_DOMAIN_INTERIOR_TARGET)


def test_g_the_schedule_is_deterministic_for_fixed_config_content_and_seed():
    source, target, source_plan, target_plan = fixture()
    first = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    second = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert structure(first) == structure(second)
    assert first.source.interiors == second.source.interiors
    assert first.target.interiors == second.target.interiors


def test_g_changing_values_while_keeping_keys_leaves_the_schedule_identical():
    source, target, source_plan, target_plan = fixture()
    reference = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    other_values = make_run(4, SOURCE_KEYS, value_of=lambda key: f"OTHER-{key}")
    changed = plan_swat_block_merge_schedule(
        other_values, target, source_plan=source_plan, target_plan=target_plan)
    assert structure(changed) == structure(reference)
    assert changed.source.interiors == reference.source.interiors


def test_g_changing_keys_may_change_the_interior_evidence():
    source, target, source_plan, target_plan = fixture()
    reference = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    shifted = make_run(4, [key + 1 for key in SOURCE_KEYS])
    other = plan_swat_block_merge_schedule(
        shifted, target, source_plan=source_plan, target_plan=target_plan)
    assert other.source.interiors != reference.source.interiors


def test_g_the_schedule_is_reproducible_across_processes():
    program = (
        "import json\n"
        "from enhanced_letindex.block import Block\n"
        "from enhanced_letindex.identifiers import BlockId, RecordKey\n"
        "from enhanced_letindex.record import Record\n"
        "from swat_m_block import allocate_block_bins, swat_reference_config\n"
        "from swat_m_block.content_schedule import (\n"
        "    LogicalBlockRunView, plan_swat_block_merge_schedule)\n"
        "def run_(level, keys, id_base):\n"
        "    blocks = []\n"
        "    for i in range(0, len(keys), 4):\n"
        "        chunk = keys[i:i + 4]\n"
        "        recs = [Record(RecordKey(k), 'v%d' % k) for k in chunk]\n"
        "        blocks.append(Block.sorted_block(BlockId(id_base + i // 4), recs, 4))\n"
        "    return LogicalBlockRunView(level, tuple(blocks), 4)\n"
        "src = run_(4, [10 * (i + 1) for i in range(32)], 0)\n"
        "tgt = run_(5, [10 * (i + 1) + 5 for i in range(20)], 100)\n"
        "sched = plan_swat_block_merge_schedule(\n"
        "    src, tgt,\n"
        "    source_plan=allocate_block_bins(src.block_count, swat_reference_config(7)),\n"
        "    target_plan=allocate_block_bins(tgt.block_count, swat_reference_config(7)))\n"
        "print(json.dumps({\n"
        "    'reads': [[r.side, r.bin_index] for r in sched.reads],\n"
        "    'source': [repr(p.key) for p in sched.source.interiors],\n"
        "    'target': [repr(p.key) for p in sched.target.interiors],\n"
        "}))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(CODES_DIR), env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert payload["reads"] == [[read.side, read.bin_index] for read in schedule.reads]
    assert payload["source"] == [repr(point.key) for point in schedule.source.interiors]
    assert payload["target"] == [repr(point.key) for point in schedule.target.interiors]


def test_g_no_module_global_mutable_rng_state_exists():
    path = SWAT_PACKAGE / M3_MODULE
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assert not (isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "Random"), "module-level RNG state"
            assert not (isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and getattr(value.func.value, "id", None) == "random"), \
                "module-level RNG state"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "seed", "module-level random.seed"


# ---------------------------------------------------------------------------
# H. the abstract bin-read schedule
# ---------------------------------------------------------------------------


def test_h_both_sides_empty_yields_an_empty_schedule():
    schedule = plan_swat_block_merge_schedule(
        make_run(4, ()), make_run(5, (), id_base=100),
        source_plan=plan_for(0), target_plan=plan_for(0))
    assert schedule.reads == ()
    assert schedule.is_empty
    assert schedule.read_count == 0
    assert schedule.tagged_interior_points == ()


def test_h_a_source_only_schedule_reads_source_zero_to_b_minus_one():
    source, _target, source_plan, _ = fixture()
    target = make_run(5, (), id_base=100)
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=plan_for(0))
    assert schedule.source_read_order == tuple(range(source_plan.bin_count))
    assert schedule.target_read_order == ()
    assert schedule.read_count == source_plan.bin_count


def test_h_a_target_only_schedule_reads_target_zero_to_b_minus_one():
    _source, target, _ignored, target_plan = fixture()
    source = make_run(4, ())
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=plan_for(0), target_plan=target_plan)
    assert schedule.target_read_order == tuple(range(target_plan.bin_count))
    assert schedule.source_read_order == ()


def test_h_a_two_sided_schedule_starts_with_source_zero_then_target_zero():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert [(read.side, read.bin_index) for read in schedule.reads[:2]] == \
        [(SOURCE, 0), (TARGET, 0)]


def test_h_every_bin_of_each_side_is_scheduled_exactly_once():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for side, bin_count in ((SOURCE, source_plan.bin_count),
                            (TARGET, target_plan.bin_count)):
        order = schedule.read_order(side)
        assert sorted(order) == list(range(bin_count))
        assert len(order) == len(set(order)) == bin_count
    assert schedule.read_count == source_plan.bin_count + target_plan.bin_count


def test_h_no_read_falls_outside_the_bin_range():
    source, target, source_plan, target_plan = fixture(seed_source=1, seed_target=2)
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for read in schedule.reads:
        limit = source_plan.bin_count if read.side == SOURCE else target_plan.bin_count
        assert 0 <= read.bin_index < limit


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13])
def test_h_the_invariant_holds_across_seeds(seed):
    source, target, source_plan, target_plan = fixture(seed_source=seed, seed_target=seed + 1)
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    assert sorted(schedule.source_read_order) == list(range(source_plan.bin_count))
    assert sorted(schedule.target_read_order) == list(range(target_plan.bin_count))


def test_h_a_hand_built_bad_schedule_is_refused():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    good = list(schedule.reads)
    # a missing bin
    with pytest.raises(ContentScheduleError):
        SwatBlockMergeSchedule(source=schedule.source, target=schedule.target,
                               reads=tuple(good[:-1]),
                               tagged_interior_points=schedule.tagged_interior_points)
    # a duplicated bin
    with pytest.raises(ContentScheduleError):
        SwatBlockMergeSchedule(source=schedule.source, target=schedule.target,
                               reads=tuple(good[:-1] + [good[0]]),
                               tagged_interior_points=schedule.tagged_interior_points)
    # an out-of-range bin
    with pytest.raises(ContentScheduleError):
        SwatBlockMergeSchedule(
            source=schedule.source, target=schedule.target,
            reads=tuple(good + [AbstractBinRead(SOURCE, source_plan.bin_count)]),
            tagged_interior_points=schedule.tagged_interior_points)


def test_h_an_unsampled_side_cannot_be_scheduled():
    source, _target, source_plan, _ = fixture()
    bound = bind_block_allocation(source, source_plan, side=SOURCE)
    sampled_target = sample_bin_interior_points(
        bind_block_allocation(source, source_plan, side=TARGET))
    with pytest.raises(ContentScheduleError):
        build_abstract_merge_schedule(bound, sampled_target)
    with pytest.raises(ContentScheduleError):
        build_abstract_merge_schedule(sampled_target, bound)


def test_h_the_schedule_carries_no_physical_identifier():
    assert set(AbstractBinRead.__dataclass_fields__) == {"side", "bin_index"}
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for read in schedule.reads:
        assert isinstance(read.side, str) and isinstance(read.bin_index, int)
        assert not hasattr(read, "slot")
        assert not hasattr(read, "slot_id")
        assert not hasattr(read, "storage")


# ---------------------------------------------------------------------------
# I. merge identity validation
# ---------------------------------------------------------------------------


def test_i_the_accepted_l_to_l_plus_one_identity_is_enforced():
    source, target, source_plan, target_plan = fixture()
    with pytest.raises(ContentScheduleError):
        plan_swat_block_merge_schedule(
            make_run(5, SOURCE_KEYS), target, source_plan=source_plan,
            target_plan=target_plan)
    with pytest.raises(ContentScheduleError):
        plan_swat_block_merge_schedule(
            source, make_run(4, TARGET_KEYS, id_base=100), source_plan=source_plan,
            target_plan=target_plan)


def test_i_a_mismatched_block_geometry_is_refused():
    source = make_run(4, SOURCE_KEYS, ipb=4)
    target = make_run(5, TARGET_KEYS, ipb=8, id_base=100)
    with pytest.raises(ContentScheduleError) as error:
        plan_swat_block_merge_schedule(
            source, target, source_plan=plan_for(source.block_count),
            target_plan=plan_for(target.block_count))
    assert "block geometry" in str(error.value)


def test_i_non_run_or_non_plan_inputs_are_refused():
    source, target, source_plan, target_plan = fixture()
    with pytest.raises(ContentScheduleError):
        plan_swat_block_merge_schedule(
            "source", target, source_plan=source_plan, target_plan=target_plan)
    with pytest.raises(ContentScheduleError):
        plan_swat_block_merge_schedule(
            source, target, source_plan=source_plan, target_plan=None)


def test_i_an_empty_plan_and_an_empty_run_are_a_legal_merge_of_one_side():
    schedule = plan_swat_block_merge_schedule(
        make_run(4, ()), make_run(5, ()), source_plan=plan_for(0), target_plan=plan_for(0))
    assert schedule.is_empty


# ---------------------------------------------------------------------------
# J. relationship to M1 and to the M2 evidence
# ---------------------------------------------------------------------------


def test_j_flattening_the_bound_bins_reconstructs_each_run_exactly():
    source, target, source_plan, target_plan = fixture()
    bound_source = bind_block_allocation(source, source_plan, side=SOURCE)
    bound_target = bind_block_allocation(target, target_plan, side=TARGET)
    assert bound_source.real_records() == source.records()
    assert bound_target.real_records() == target.records()
    assert bound_source.real_keys() == source.keys()
    assert bound_target.real_keys() == target.keys()


def test_j_m1_remains_the_exact_newer_wins_oracle_and_ignores_the_schedule():
    source, target, _ignored_source_plan, _ignored_target_plan = fixture()
    source_view = LogicalRunView.from_records(
        source.level, [(record.key.value, record.value) for record in source.records()])
    target_view = LogicalRunView.from_records(
        target.level, [(record.key.value, record.value) for record in target.records()])
    reference = functional_merge_oracle(
        source_view, target_view, items_per_block=IPB, epsilon=PGM_EPSILON)
    # a different M2 plan changes the schedule but never the M1 result
    other = plan_swat_block_merge_schedule(
        source, target, source_plan=plan_for(source.block_count, 11),
        target_plan=plan_for(target.block_count, 12))
    assert other.source_read_order != () or other.is_empty
    again = functional_merge_oracle(
        source_view, target_view, items_per_block=IPB, epsilon=PGM_EPSILON)
    assert again == reference
    # ... and the M1 result is still the exact newer-wins sorted union, checked
    #     independently here rather than by reusing any M1 machinery
    expected = {}
    for key, value in target_view.records():
        expected[key] = value
    for key, value in source_view.records():          # source is newer: it wins
        expected[key] = value
    assert reference.records() == tuple(sorted(expected.items()))
    assert reference.pgm == build_batch_pgm(reference.keys(), PGM_EPSILON)


def test_j_m3_emits_no_merged_output_and_no_output_blocks():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for field in SwatBlockMergeSchedule.__dataclass_fields__:
        lowered = field.lower()
        for token in ("output", "merged", "result", "newer_wins"):
            assert token not in lowered, field


def test_j_m3_does_not_import_the_m1_oracle_or_a_merge_implementation():
    tree = ast.parse((SWAT_PACKAGE / M3_MODULE).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert modules == {"__future__", "bisect", "math", "random", "dataclasses", "typing",
                       "enhanced_letindex.block", "enhanced_letindex.identifiers",
                       "enhanced_letindex.record", "bin_allocator", "distribution"}, modules
    joined = " ".join(modules)
    for forbidden in ("functional_oracle", "incremental_merge", "engine", "merge",
                      "storage", "trace", "pgm"):
        assert forbidden not in joined, forbidden


def test_j_the_m2_block_prefix_evidence_is_carried_in_block_units_untouched():
    source, target, source_plan, target_plan = fixture()
    schedule = plan_swat_block_merge_schedule(
        source, target, source_plan=source_plan, target_plan=target_plan)
    for bound, plan in ((schedule.source, source_plan), (schedule.target, target_plan)):
        assert bound.sampled_loads == plan.sampled_loads
        assert bound.noisy_prefix_sums == plan.noisy_prefix_sums
        assert bound.additive_error_blocks == plan.additive_error_blocks
        assert len(bound.noisy_prefix_sums) == bound.bin_count + 1
        assert bound.noisy_prefix_sums[0] == 0
        # block units, never record units: every entry stays inside the block-count
        # window [true_prefix - Z, true_prefix + Z]
        assert max(bound.noisy_prefix_sums) <= plan.sampled_capacity + bound.bins[0].bin_capacity_blocks
        assert all(isinstance(value, int) and value >= 0
                   for value in bound.noisy_prefix_sums)


def test_j_the_record_level_safe_output_frontier_is_not_reimplemented():
    code = _code_without_docstrings(SWAT_PACKAGE / M3_MODULE)
    for token in ("newCnt", "new_cnt", "safe_output", "frontier", "additive_error +",
                  "2 * additive", "2 * self.additive"):
        assert token not in code, token


#: A fixture whose pinned ``DOMerge`` loop really does trigger the ``else if`` fallback:
#: the source (3 bins) is exhausted while the target (4 bins) still has unread bins, and a
#: later source tag arrives.  Bin-start keys were chosen so the tagged order is
#: t0, s0, s1, s2, t1, t2, t3.
FALLBACK_SEED = 3
FALLBACK_SOURCE_BLOCKS = 16
FALLBACK_TARGET_BLOCKS = 24
FALLBACK_SOURCE_KEYS = (1_000, 1_128, 1_256)
FALLBACK_TARGET_KEYS = (1, 2_256, 2_384, 2_516)


def fallback_fixture():
    source_plan = plan_for(FALLBACK_SOURCE_BLOCKS, FALLBACK_SEED)
    target_plan = plan_for(FALLBACK_TARGET_BLOCKS, FALLBACK_SEED)
    assert source_plan.bin_count == 3 and target_plan.bin_count == 4
    assert all(bin_.real_count > 0 for bin_ in source_plan.bins)
    assert all(bin_.real_count > 0 for bin_ in target_plan.bins)
    source_starts = dict(zip([bin_.logical_rank_start for bin_ in source_plan.bins],
                             FALLBACK_SOURCE_KEYS))
    target_starts = dict(zip([bin_.logical_rank_start for bin_ in target_plan.bins],
                             FALLBACK_TARGET_KEYS))
    source_run = make_run_with_bin_starts(
        4, source_starts, FALLBACK_SOURCE_BLOCKS, 0)
    target_run = make_run_with_bin_starts(
        5, target_starts, FALLBACK_TARGET_BLOCKS, 100)
    return source_run, target_run, source_plan, target_plan


def test_h_the_pinned_dom_merge_fallback_fetches_the_same_sequence_as_m3():
    """The pinned ``else if`` fallback DOES trigger, and the fetch sequence still agrees.

    The pinned loop runs ``source_bin_count + target_bin_count`` iterations and, when a
    source tag arrives after the source is exhausted, fetches from the target side.  After
    projecting away no-op iterations (and the deferred safe-output work, which M3 does not
    model at all), the emitted bin-fetch sequence equals M3's simpler
    same-side-next-unread schedule on this well-formed input.
    """
    source_run, target_run, source_plan, target_plan = fallback_fixture()
    schedule = plan_swat_block_merge_schedule(
        source_run, target_run, source_plan=source_plan, target_plan=target_plan,
        interior_index_sampler=lambda weights: 0)
    tagged = [(point.side, interior_point_sort_key(point.interior_point))
              for point in schedule.tagged_interior_points]
    assert tagged == [(TARGET, (0, 1)), (SOURCE, (0, 1_000)), (SOURCE, (0, 1_128)),
                      (SOURCE, (0, 1_256)), (TARGET, (0, 2_256)), (TARGET, (0, 2_384)),
                      (TARGET, (0, 2_516))]

    pinned, fallbacks = pinned_projected_fetches(
        source_plan.bin_count, target_plan.bin_count, tagged)
    m3 = [(read.side, read.bin_index) for read in schedule.reads]
    assert pinned == m3
    # the fixture must genuinely exercise the fallback: a source tag arriving once the
    # source is exhausted fetches the target's next unread bin
    assert fallbacks == [(SOURCE, 2)]
    # ... and both rules still fetch every planned bin exactly once
    assert sorted(pinned) == sorted(
        [(SOURCE, index) for index in range(source_plan.bin_count)]
        + [(TARGET, index) for index in range(target_plan.bin_count)])


def test_h_the_pinned_loop_and_m3_agree_on_every_fixture():
    """Projected pinned-loop fetches equal M3 reads for a range of fixtures."""
    for seed in (0, 1, 2, 3, 5, 8):
        source, target, source_plan, target_plan = fixture(seed_source=seed,
                                                           seed_target=seed + 1)
        schedule = plan_swat_block_merge_schedule(
            source, target, source_plan=source_plan, target_plan=target_plan)
        tagged = [(point.side, interior_point_sort_key(point.interior_point))
                  for point in schedule.tagged_interior_points]
        pinned, _fallbacks = pinned_projected_fetches(
            source_plan.bin_count, target_plan.bin_count, tagged)
        assert pinned == [(read.side, read.bin_index) for read in schedule.reads], seed


# ---------------------------------------------------------------------------
# K. scope guard: no physical/observable surface
# ---------------------------------------------------------------------------


def test_k_the_m3_module_contains_no_forbidden_mechanism():
    code = _code_without_docstrings(SWAT_PACKAGE / M3_MODULE)
    for token in FORBIDDEN_M3_TOKENS:
        assert token not in code, token


def test_k_no_swat_m_block_module_references_a_physical_identifier():
    for path in sorted(SWAT_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in {"SlotId", "TraceEvent", "TraceOperation",
                                       "TraceCollector", "UntrustedStorage"}, path.name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"read_slot", "write_slot"}, (path.name, node.attr)


def test_k_the_m3_pipeline_performs_zero_physical_io_and_zero_trace_events(monkeypatch):
    from enhanced_letindex.storage import UntrustedStorage
    from enhanced_letindex.trace import TraceCollector

    physical_calls: list = []

    def explode(method):
        def recorder(self, *args, **kwargs):
            physical_calls.append(method)
            raise AssertionError(f"M3 performed a physical {method}")

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
    source, target, source_plan, target_plan = fixture()
    schedules = [
        plan_swat_block_merge_schedule(
            source, target, source_plan=source_plan, target_plan=target_plan),
        plan_swat_block_merge_schedule(
            source, make_run(5, (), id_base=100), source_plan=source_plan,
            target_plan=plan_for(0)),
        plan_swat_block_merge_schedule(
            make_run(4, ()), target, source_plan=plan_for(0), target_plan=target_plan),
        plan_swat_block_merge_schedule(
            make_run(4, ()), make_run(5, ()), source_plan=plan_for(0),
            target_plan=plan_for(0)),
    ]
    assert physical_calls == []
    assert recorded == []
    assert storage.size == 0 and storage.trace.events() == ()
    for schedule in schedules:
        assert sorted(schedule.source_read_order) == [
            index for index in range(schedule.source.bin_count)]
        assert sorted(schedule.target_read_order) == [
            index for index in range(schedule.target.bin_count)]


def test_k_m3_adds_only_the_content_schedule_module():
    assert {path.name for path in SWAT_PACKAGE.glob("*.py")} == {
        "__init__.py", "functional_oracle.py", "distribution.py", "bin_allocator.py",
        "content_schedule.py",
    }


# ---------------------------------------------------------------------------
# L. the M0 frozen substrate is untouched
# ---------------------------------------------------------------------------


def test_l_the_35_file_frozen_substrate_manifest_still_verifies():
    entries = [line.split() for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(entries) == 35
    assert _load_verify_module().main(["--quiet", "--root", str(REPO_ROOT)]) == 0


def test_l_the_m3_implementation_lives_outside_the_frozen_package():
    frozen = {line.split()[1] for line in
              MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert not any("swat_m_block" in relative for relative in frozen)
    assert (SWAT_PACKAGE / M3_MODULE).is_file()
