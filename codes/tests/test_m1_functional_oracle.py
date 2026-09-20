"""M1 focused tests: the functional SWAT-M-Block oracle (Issue #2).

These tests are written for this repository.  They cover the M1 acceptance criteria:

* exact record-level newer-wins sorted union, cross-checked against an independent
  dict oracle;
* canonical output block packing;
* the canonical PGM over the exact output key stream;
* the required edge-case matrix (empty / one-sided / alternating / disjoint /
  duplicate-heavy / mixed-sign keys / one full block / final partial block /
  ``items_per_block = 1`` / multiple PGM segments / ``epsilon = 0`` and several positive
  epsilons / invalid same-reverse-gapped levels);
* schedule invariance and determinism;
* absence of every physical and privacy mechanism (no storage, no trace, no slot
  scheduling, no randomness, no noise/padding/bins);
* the M0 35-file frozen-substrate manifest still verifies.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.incremental_merge import (
    IncrementalMergeError,
    IncrementalMergeJob,
    LogicalRunView,
)
from enhanced_letindex.pgm import build_batch_pgm, make_segmentation

from swat_m_block import (
    DEFAULT_SCHEDULE_POLICY,
    FunctionalMergeOracleResult,
    FunctionalOracleError,
    collect_output_blocks,
    flatten_blocks,
    functional_merge_oracle,
    output_key_stream,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / "codes"
SRC_DIR = CODES_DIR / "src"
SWAT_PACKAGE = SRC_DIR / "swat_m_block"
MANIFEST_PATH = REPO_ROOT / "provenance" / "common-substrate.sha256"
VERIFY_SCRIPT = REPO_ROOT / "provenance" / "verify_manifest.py"

SOURCE_LEVEL = 4
TARGET_LEVEL = 5
ITEMS_PER_BLOCK = 8
PGM_EPSILON = 8

#: The M1 merge must not contain any of these in executable code (naming them in
#: documentation or in a test is allowed; implementing them is not).
FORBIDDEN_MECHANISM_TOKENS = (
    "DOAllocate", "DOMerge", "DOMerger", "bin_allocator", "BinAllocator",
    "noisy_allocat", "padded_bin", "cover_io", "dummy_io", "output_shuffle",
    "oblique_permutation", "bitonic", "sorting_network", "prp_writeback",
    "deamortiz", "de_amortiz", "UntrustedStorage", "TraceEvent", "TraceOperation",
    "TraceCollector", "SlotId", "read_slot", "write_slot", "physical_slot",
)

#: Modules that must not be imported anywhere in the M1 oracle: the physical surface
#: and every excluded EnhancedLETIndex defence implementation.
FORBIDDEN_IMPORT_SUFFIXES = (
    "storage", "trace", "block_prp", "defense_state", "query_defense", "query_stash",
    "protected_merge", "deamortized_merge", "defended_index",
)

#: Randomness sources the *oracle* must never acquire (M2 owns seeded randomness).
FORBIDDEN_ORACLE_RANDOMNESS = ("random", "secrets", "uuid")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def view(level: int, items) -> LogicalRunView:
    return LogicalRunView.from_records(level, items)


def independent_oracle(source_items, target_items):
    """An independent dict-based newer-wins oracle (source is newer)."""
    newer, older = dict(source_items), dict(target_items)
    return tuple(
        (key, newer[key] if key in newer else older[key])
        for key in sorted(set(newer) | set(older))
    )


def merge(source_items, target_items, *, items_per_block=ITEMS_PER_BLOCK,
          epsilon=PGM_EPSILON, schedule=None, levels=(SOURCE_LEVEL, TARGET_LEVEL)):
    return functional_merge_oracle(
        view(levels[0], source_items),
        view(levels[1], target_items),
        items_per_block=items_per_block,
        epsilon=epsilon,
        schedule=schedule,
    )


def expected_keys(source_items, target_items):
    return tuple(RecordKey(key) for key, _ in independent_oracle(source_items, target_items))


def block_sizes(result: FunctionalMergeOracleResult):
    return tuple(block.item_count for block in result.blocks)


BASE_SOURCE = tuple((2_000 + rank, f"s{rank}") for rank in range(12))
BASE_TARGET = tuple((2_006 + rank, f"t{rank}") for rank in range(12))

#: 12 + 12 records sharing 6 keys -> 18 merged records, 6 duplicates, blocks 8/8/2.
BASE_EXPECTED_RECORDS = 18
BASE_EXPECTED_BLOCKS = 3
BASE_EXPECTED_DUPLICATES = 6
BASE_EXPECTED_SIZES = (8, 8, 2)

#: Three groups of ten keys separated by wide gaps: with epsilon = 0 the canonical
#: segmentation has three segments, so merge steps and output-block boundaries necessarily
#: fall inside a PGM segment.
SEGMENT_KEYS = tuple(
    group * 100_000 + rank for group in range(3) for rank in range(10)
)
SEGMENT_SOURCE = tuple((key, f"s{key}") for key in SEGMENT_KEYS[:15])
SEGMENT_TARGET = tuple((key, f"t{key}") for key in SEGMENT_KEYS[15:])
SEGMENT_EPSILON = 0
SEGMENT_ITEMS_PER_BLOCK = 4
SEGMENT_EXPECTED_SEGMENTS = 3

#: source L (older-looking keys) entirely before target, and the mirror image.
BEFORE_SOURCE = ((1, "s1"), (2, "s2"))
BEFORE_TARGET = ((8, "t8"), (9, "t9"))

#: A duplicate-heavy fixture: every key appears on both sides.
DUP_HEAVY_SOURCE = tuple((rank, f"s{rank}") for rank in range(10))
DUP_HEAVY_TARGET = tuple((rank, f"t{rank}") for rank in range(10))

#: Mixed-sign keys, with two duplicated keys whose values differ.
SIGNED_SOURCE = ((-9, "s-9"), (-4, "s-4"), (0, "s0"), (7, "s7"))
SIGNED_TARGET = ((-7, "t-7"), (-4, "t-4"), (0, "t0"), (9, "t9"))

EPSILONS = (0, 1, 2, 4, 8, 64)

ALL_FIXTURES = {
    "base": (BASE_SOURCE, BASE_TARGET, ITEMS_PER_BLOCK),
    "both_empty": ((), (), ITEMS_PER_BLOCK),
    "source_empty": ((), ((1, "t1"), (2, "t2")), ITEMS_PER_BLOCK),
    "target_empty": (((1, "s1"), (2, "s2")), (), ITEMS_PER_BLOCK),
    "one_record_each_same_key": (((5, "s5"),), ((5, "t5"),), ITEMS_PER_BLOCK),
    "one_record_each": (((4, "s4"),), ((9, "t9"),), ITEMS_PER_BLOCK),
    "alternating_disjoint": (
        tuple((2 * rank, f"s{rank}") for rank in range(6)),
        tuple((2 * rank + 1, f"t{rank}") for rank in range(6)),
        ITEMS_PER_BLOCK,
    ),
    "source_entirely_before": (BEFORE_SOURCE, BEFORE_TARGET, ITEMS_PER_BLOCK),
    "target_entirely_before": (BEFORE_TARGET, BEFORE_SOURCE, ITEMS_PER_BLOCK),
    "duplicate_heavy": (DUP_HEAVY_SOURCE, DUP_HEAVY_TARGET, ITEMS_PER_BLOCK),
    "mixed_sign_keys": (SIGNED_SOURCE, SIGNED_TARGET, ITEMS_PER_BLOCK),
    "exactly_one_full_block": (tuple((r, f"s{r}") for r in range(8)), (), ITEMS_PER_BLOCK),
    "full_plus_final_partial": (BASE_SOURCE, BASE_TARGET, ITEMS_PER_BLOCK),
    "items_per_block_one": (((1, "s1"), (3, "s3")), ((2, "t2"), (3, "t3")), 1),
    "multi_segment_epsilon_zero": (
        SEGMENT_SOURCE, SEGMENT_TARGET, SEGMENT_ITEMS_PER_BLOCK,
    ),
    "single_record_block": (((11, "s11"),), (), 1),
}


#: The M1 oracle module.  The guards below state M1's own guarantee — the oracle is
#: randomness-free and contains no mechanism — so they scan this module rather than the
#: whole package: M2 (Issue #4) legitimately adds the stochastic block-bin allocation
#: planner, whose own guards live in ``test_m2_block_bin_allocation.py``.
M1_MODULE = "functional_oracle.py"

#: The package content authorised so far (M0 marker + M1 oracle + M2 planner).
SWAT_MODULES = ("__init__.py", "functional_oracle.py", "distribution.py",
                "bin_allocator.py")


def _iter_swat_modules():
    return sorted(SWAT_PACKAGE.glob("*.py"))


def _m1_module() -> Path:
    return SWAT_PACKAGE / M1_MODULE


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


# ---------------------------------------------------------------------------
# A. exact newer-wins semantics vs an independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_a_every_fixture_matches_the_independent_dict_oracle(name):
    source_items, target_items, items_per_block = ALL_FIXTURES[name]
    result = merge(source_items, target_items, items_per_block=items_per_block)
    expected = independent_oracle(source_items, target_items)
    assert result.records() == expected, name
    assert flatten_blocks(result.blocks) == expected
    assert result.record_count == len(expected)
    assert result.keys() == expected_keys(source_items, target_items)


def test_a_duplicate_keys_keep_the_source_value_and_consume_both_inputs():
    result = merge(BASE_SOURCE, BASE_TARGET)
    merged = dict(result.records())
    # keys 2006..2011 exist on both sides with different values
    for key in range(2_006, 2_012):
        assert merged[key] == f"s{key - 2_000}"
        assert dict(BASE_TARGET)[key] == f"t{key - 2_006}"
    # a source-only key keeps its source value, a target-only key its target value
    assert merged[2_000] == "s0"
    assert merged[2_017] == "t11"
    assert result.duplicate_count == BASE_EXPECTED_DUPLICATES


def test_a_exact_counters_for_the_base_fixture():
    result = merge(BASE_SOURCE, BASE_TARGET)
    assert result.record_count == BASE_EXPECTED_RECORDS
    assert result.block_count == BASE_EXPECTED_BLOCKS
    assert result.duplicate_count == BASE_EXPECTED_DUPLICATES
    assert result.first_key == RecordKey(2_000)
    assert result.last_key == RecordKey(2_017)
    assert result.source_level == SOURCE_LEVEL
    assert result.target_level == TARGET_LEVEL
    assert result.items_per_block == ITEMS_PER_BLOCK
    assert result.epsilon == PGM_EPSILON


def test_a_all_keys_in_one_side_are_duplicates():
    result = merge(DUP_HEAVY_SOURCE, DUP_HEAVY_TARGET)
    assert result.record_count == 10
    assert result.duplicate_count == 10
    assert all(value.startswith("s") for _, value in result.records())


# ---------------------------------------------------------------------------
# B. empty / one-sided / ordering edge cases
# ---------------------------------------------------------------------------


def test_b_both_inputs_empty():
    result = merge((), ())
    assert result.is_empty
    assert result.record_count == 0
    assert result.block_count == 0
    assert result.blocks == ()
    assert result.duplicate_count == 0
    assert result.first_key is None and result.last_key is None
    assert result.records() == ()


def test_b_one_sided_inputs_are_passed_through_in_order():
    source_only = merge(((1, "s1"), (2, "s2")), ())
    assert source_only.records() == ((1, "s1"), (2, "s2"))
    assert source_only.duplicate_count == 0
    target_only = merge((), ((1, "t1"), (2, "t2")))
    assert target_only.records() == ((1, "t1"), (2, "t2"))
    assert target_only.duplicate_count == 0


def test_b_one_record_each():
    same_key = merge(((5, "s5"),), ((5, "t5"),))
    assert same_key.records() == ((5, "s5"),)
    assert same_key.duplicate_count == 1
    assert same_key.block_count == 1
    different_keys = merge(((4, "s4"),), ((9, "t9"),))
    assert different_keys.records() == ((4, "s4"), (9, "t9"))
    assert different_keys.duplicate_count == 0


def test_b_alternating_disjoint_keys_interleave():
    result = merge(ALL_FIXTURES["alternating_disjoint"][0],
                   ALL_FIXTURES["alternating_disjoint"][1])
    assert [key for key, _ in result.records()] == list(range(12))


def test_b_source_and_target_entirely_before_each_other():
    before = merge(BEFORE_SOURCE, BEFORE_TARGET)
    assert before.records() == ((1, "s1"), (2, "s2"), (8, "t8"), (9, "t9"))
    # reversing the roles keeps every value attached to its own key: the target of the
    # reversed merge is BEFORE_SOURCE, so keys 1 and 2 still carry their source values
    after = merge(BEFORE_TARGET, BEFORE_SOURCE)
    assert after.records() == ((1, "s1"), (2, "s2"), (8, "t8"), (9, "t9"))
    assert after.records() == before.records()


def test_b_negative_zero_and_positive_keys_are_legal_and_ordered_exactly():
    result = merge(SIGNED_SOURCE, SIGNED_TARGET)
    assert result.records() == (
        (-9, "s-9"), (-7, "t-7"), (-4, "s-4"), (0, "s0"), (7, "s7"), (9, "t9")
    )
    assert result.duplicate_count == 2
    assert result.first_key == RecordKey(-9)
    assert result.last_key == RecordKey(9)


# ---------------------------------------------------------------------------
# C. canonical output block packing
# ---------------------------------------------------------------------------


def test_c_base_fixture_packs_full_blocks_then_a_final_partial_block():
    result = merge(BASE_SOURCE, BASE_TARGET)
    assert block_sizes(result) == BASE_EXPECTED_SIZES
    assert [block.rank for block in result.blocks] == [0, 1, 2]
    assert result.block_count == len(result.blocks)


def test_c_exactly_one_full_block_is_a_single_block():
    result = merge(tuple((rank, f"s{rank}") for rank in range(8)), ())
    assert block_sizes(result) == (8,)
    assert result.block_count == 1
    assert result.blocks[0].rank == 0


def test_c_items_per_block_one_emits_one_block_per_record():
    result = merge(((1, "s1"), (3, "s3")), ((2, "t2"), (3, "t3")), items_per_block=1)
    assert block_sizes(result) == (1, 1, 1)
    assert [block.rank for block in result.blocks] == [0, 1, 2]
    assert result.records() == ((1, "s1"), (2, "t2"), (3, "s3"))


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_c_packing_is_canonical_for_every_fixture(name):
    source_items, target_items, items_per_block = ALL_FIXTURES[name]
    result = merge(source_items, target_items, items_per_block=items_per_block)
    sizes = block_sizes(result)
    assert result.block_count == len(result.blocks)
    assert [block.rank for block in result.blocks] == list(range(len(result.blocks)))
    assert all(size == items_per_block for size in sizes[:-1])
    if sizes:
        assert 1 <= sizes[-1] <= items_per_block
    assert sum(sizes) == result.record_count
    keys = [key.value for key in output_key_stream(result.blocks)]
    assert keys == sorted(set(keys))


def test_c_empty_merge_emits_no_block_at_all():
    result = merge((), ())
    assert result.blocks == ()
    assert block_sizes(result) == ()


# ---------------------------------------------------------------------------
# D. canonical PGM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_d_finalized_pgm_equals_the_batch_builder(name, epsilon):
    source_items, target_items, items_per_block = ALL_FIXTURES[name]
    result = merge(source_items, target_items, items_per_block=items_per_block,
                   epsilon=epsilon)
    keys = expected_keys(source_items, target_items)
    assert result.pgm == build_batch_pgm(keys, epsilon), (name, epsilon)
    assert result.pgm.segments == make_segmentation(keys, epsilon), (name, epsilon)
    assert (result.first_key, result.last_key) == (
        result.pgm.first_key, result.pgm.last_key
    )


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_d_base_fixture_pgm_is_the_canonical_pgm_for_every_epsilon(epsilon):
    result = merge(BASE_SOURCE, BASE_TARGET, epsilon=epsilon)
    assert result.pgm == build_batch_pgm(expected_keys(BASE_SOURCE, BASE_TARGET), epsilon)


def test_d_epsilon_zero_produces_the_canonical_zero_epsilon_pgm():
    result = merge(BASE_SOURCE, BASE_TARGET, epsilon=0)
    assert result.epsilon == 0
    assert result.pgm == build_batch_pgm(expected_keys(BASE_SOURCE, BASE_TARGET), 0)


def test_d_a_multi_segment_pgm_is_produced_where_the_canonical_segmentation_splits():
    result = merge(SEGMENT_SOURCE, SEGMENT_TARGET,
                   items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON)
    segments = result.pgm.segments
    assert len(segments) == SEGMENT_EXPECTED_SEGMENTS
    assert len(segments) > 1
    assert segments == make_segmentation(expected_keys(SEGMENT_SOURCE, SEGMENT_TARGET),
                                         SEGMENT_EPSILON)
    # the canonical segment start ranks of this fixture: three groups of ten keys
    assert tuple(segment.start_rank for segment in segments) == (0, 10, 20)
    # the segmentation is not aligned with the output blocks: 8 blocks, 3 segments
    assert result.block_count == 8
    assert len(segments) != result.block_count


def test_d_empty_input_has_an_empty_pgm():
    result = merge((), ())
    assert result.pgm == build_batch_pgm((), PGM_EPSILON)
    assert result.pgm.segments == ()


# ---------------------------------------------------------------------------
# E. schedule invariance and determinism
# ---------------------------------------------------------------------------


SCHEDULES = (
    None,
    (1,),
    (2,),
    (3,),
    (7,),
    (1, 1, 1, 1, 1),
    (100,),
    (5, 3, 2, 9),
)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_e_the_whole_result_is_invariant_to_the_advance_schedule(schedule):
    """Issue #2 invariant 6 at full strength: the frozen result must compare EQUAL.

    Field-by-field comparison is not enough.  Any step-decomposition evidence embedded in
    the result would make two logically identical merges compare unequal, so the whole
    frozen dataclass is compared.
    """
    reference = merge(BASE_SOURCE, BASE_TARGET)
    result = merge(BASE_SOURCE, BASE_TARGET, schedule=schedule)
    assert result == reference
    assert result.records() == reference.records()
    assert block_sizes(result) == block_sizes(reference)
    assert result.pgm == reference.pgm


def test_e_the_oracle_result_carries_no_step_decomposition_evidence():
    """The corrected contract: the public logical result has no schedule field at all."""
    fields = set(FunctionalMergeOracleResult.__dataclass_fields__)
    assert "advance_schedule" not in fields
    assert fields == {
        "source_level", "target_level", "items_per_block", "epsilon", "record_count",
        "block_count", "duplicate_count", "blocks", "pgm", "first_key", "last_key",
    }


def test_e_the_default_schedule_policy_is_a_single_drain_budget():
    """The applied budgets are evaluator evidence from the helper, not result state."""
    assert DEFAULT_SCHEDULE_POLICY == "drain"
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, BASE_SOURCE), view(TARGET_LEVEL, BASE_TARGET),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    blocks, applied = collect_output_blocks(job)
    assert applied == (24,)  # 12 + 12 input records, one budget
    assert len(applied) == 1
    assert flatten_blocks(blocks) == independent_oracle(BASE_SOURCE, BASE_TARGET)


def test_e_a_consumed_schedule_is_drained_to_completion():
    result = merge(BASE_SOURCE, BASE_TARGET, schedule=(1,))
    assert result.records() == independent_oracle(BASE_SOURCE, BASE_TARGET)
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, BASE_SOURCE), view(TARGET_LEVEL, BASE_TARGET),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    blocks, applied = collect_output_blocks(job, (1,))
    assert len(applied) == BASE_EXPECTED_RECORDS  # drained one decision at a time
    assert flatten_blocks(blocks) == result.records()


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_e_two_runs_are_bit_identical(name):
    source_items, target_items, items_per_block = ALL_FIXTURES[name]
    first = merge(source_items, target_items, items_per_block=items_per_block)
    second = merge(source_items, target_items, items_per_block=items_per_block)
    assert first == second


def test_e_collect_output_blocks_also_works_on_a_raw_frozen_job():
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, BASE_SOURCE), view(TARGET_LEVEL, BASE_TARGET),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    blocks, applied = collect_output_blocks(job)
    assert flatten_blocks(blocks) == independent_oracle(BASE_SOURCE, BASE_TARGET)
    assert applied == (24,)
    merged = job.finalize()
    assert merged.block_count == BASE_EXPECTED_BLOCKS
    assert merged.pgm == build_batch_pgm(expected_keys(BASE_SOURCE, BASE_TARGET),
                                         PGM_EPSILON)


# ---------------------------------------------------------------------------
# F. invalid inputs
# ---------------------------------------------------------------------------


def test_f_same_level_merge_is_refused_before_any_output():
    with pytest.raises(FunctionalOracleError) as error:
        merge(BASE_SOURCE, BASE_TARGET, levels=(4, 4))
    assert isinstance(error.value, IncrementalMergeError)
    assert "same level" in str(error.value)


def test_f_reverse_direction_is_refused_before_any_output():
    with pytest.raises(FunctionalOracleError) as error:
        merge(BASE_SOURCE, BASE_TARGET, levels=(5, 4))
    assert isinstance(error.value, IncrementalMergeError)
    assert "reverse" in str(error.value)


@pytest.mark.parametrize("levels", [(4, 6), (0, 2), (9, 12)])
def test_f_gapped_levels_are_refused_before_any_output(levels):
    with pytest.raises(FunctionalOracleError) as error:
        merge(BASE_SOURCE, BASE_TARGET, levels=levels)
    assert isinstance(error.value, IncrementalMergeError)
    assert "gap" in str(error.value)


def test_f_the_frozen_job_refuses_the_same_identities_independently():
    for levels in ((4, 4), (5, 4), (4, 6)):
        with pytest.raises(IncrementalMergeError):
            IncrementalMergeJob.begin(
                view(levels[0], BASE_SOURCE), view(levels[1], BASE_TARGET),
                items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
            )


def test_f_non_view_inputs_are_refused():
    with pytest.raises(FunctionalOracleError):
        functional_merge_oracle(
            BASE_SOURCE, view(TARGET_LEVEL, BASE_TARGET),
            items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
        )
    with pytest.raises(FunctionalOracleError):
        functional_merge_oracle(
            view(SOURCE_LEVEL, BASE_SOURCE), dict(BASE_TARGET),
            items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
        )


@pytest.mark.parametrize("value", [0, -1, True, "8", 2.5, None])
def test_f_invalid_items_per_block_is_refused(value):
    with pytest.raises(IncrementalMergeError):
        merge(BASE_SOURCE, BASE_TARGET, items_per_block=value)


@pytest.mark.parametrize("value", [-1, True, "0", 1.5, None])
def test_f_invalid_epsilon_is_refused(value):
    with pytest.raises(IncrementalMergeError):
        merge(BASE_SOURCE, BASE_TARGET, epsilon=value)


@pytest.mark.parametrize("schedule", [(0,), (-1,), (True,), "abc", (1, 0), (None,)])
def test_f_invalid_schedule_is_refused(schedule):
    with pytest.raises(FunctionalOracleError):
        merge(BASE_SOURCE, BASE_TARGET, schedule=schedule)


def test_f_a_malformed_input_run_is_refused_by_the_frozen_view():
    with pytest.raises(IncrementalMergeError):
        view(SOURCE_LEVEL, ((3, "a"), (2, "b")))
    with pytest.raises(IncrementalMergeError):
        view(SOURCE_LEVEL, ((2, "a"), (2, "b")))


# ---------------------------------------------------------------------------
# G. no physical mechanism, no trace, no randomness
# ---------------------------------------------------------------------------


def test_g_the_m1_oracle_module_contains_no_forbidden_mechanism():
    code = _code_without_docstrings(_m1_module())
    for token in FORBIDDEN_MECHANISM_TOKENS:
        assert token not in code, (M1_MODULE, token)


def test_g_no_swat_m_block_module_imports_a_physical_or_defense_module():
    """The physical/defence import ban covers every module of the package.

    (The randomness ban below is scoped to the oracle: M2 owns seeded randomness and is
    guarded separately.)
    """
    for path in _iter_swat_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [alias.name for alias in node.names]
            for dotted in names:
                assert dotted.rsplit(".", 1)[-1] not in FORBIDDEN_IMPORT_SUFFIXES, (
                    path.name, dotted)



def test_g_the_m1_oracle_only_imports_the_frozen_common_substrate():
    path = SWAT_PACKAGE / "functional_oracle.py"
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "dataclasses",
        "typing",
        "enhanced_letindex.identifiers",
        "enhanced_letindex.incremental_merge",
        "enhanced_letindex.pgm",
    }, imported


def test_g_the_result_exposes_no_physical_or_observational_field():
    result = merge(BASE_SOURCE, BASE_TARGET)
    fields = set(result.__dataclass_fields__)
    for field in fields:
        lowered = field.lower()
        for token in ("slot", "trace", "storage", "prp", "rand", "noise", "bin"):
            assert token not in lowered, field


def test_g_running_the_oracle_loads_only_the_frozen_common_substrate():
    """A full run must reach only the frozen common substrate - no defence module.

    ``enhanced_letindex/__init__`` eagerly imports its own package surface, so this test
    asserts the *closure* is confinement to the frozen snapshot rather than the absence of
    the storage/trace vocabulary (which is proven operationally by the zero-I/O test).
    """
    program = (
        "import json, sys\n"
        "from enhanced_letindex.incremental_merge import LogicalRunView\n"
        "import swat_m_block\n"
        "from swat_m_block import functional_merge_oracle\n"
        "source = tuple((2000 + r, 's%d' % r) for r in range(12))\n"
        "target = tuple((2006 + r, 't%d' % r) for r in range(12))\n"
        "result = functional_merge_oracle(\n"
        "    LogicalRunView.from_records(4, source),\n"
        "    LogicalRunView.from_records(5, target),\n"
        "    items_per_block=8, epsilon=8,\n"
        ")\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('enhanced_letindex.'))\n"
        "print(json.dumps({\n"
        "    'loaded': loaded,\n"
        "    'records': result.record_count,\n"
        "    'blocks': result.block_count,\n"
        "}))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(CODES_DIR), env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    import json

    payload = json.loads(process.stdout)
    assert payload["records"] == BASE_EXPECTED_RECORDS
    assert payload["blocks"] == BASE_EXPECTED_BLOCKS
    loaded = payload["loaded"]
    assert "enhanced_letindex.incremental_merge" in loaded
    assert "enhanced_letindex.pgm" in loaded
    # every loaded enhanced_letindex module is part of the frozen 35-file snapshot, and
    # no defence / later-milestone module is reachable from the M1 oracle at all
    frozen_stems = {
        "enhanced_letindex." + line.split()[1].rsplit("/", 1)[-1][:-3]
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and "/enhanced_letindex/" in line
    }
    assert set(loaded) <= frozen_stems | {"enhanced_letindex"}, loaded
    for forbidden in (
        "enhanced_letindex.protected_merge", "enhanced_letindex.defended_index",
        "enhanced_letindex.query_defense", "enhanced_letindex.query_stash",
        "enhanced_letindex.block_prp", "enhanced_letindex.defense_state",
        "enhanced_letindex.deamortized_merge",
    ):
        assert forbidden not in loaded, forbidden


def test_g_a_full_merge_performs_zero_physical_io_and_emits_zero_trace_events(monkeypatch):
    """Operational proof: no storage operation and no observation event can occur.

    Every ``UntrustedStorage`` entry point is turned into a failure, and ``TraceCollector``
    is instrumented: a complete oracle run must not call either.
    """
    from enhanced_letindex.storage import UntrustedStorage
    from enhanced_letindex.trace import TraceCollector

    physical_calls: list = []

    def explode(name):
        def recorder(self, *args, **kwargs):
            physical_calls.append(name)
            raise AssertionError(f"the M1 oracle performed a physical {name}")

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
    result = merge(BASE_SOURCE, BASE_TARGET)

    assert result.records() == independent_oracle(BASE_SOURCE, BASE_TARGET)
    assert physical_calls == []
    assert recorded == []
    assert storage.size == 0
    assert storage.occupied_slots() == ()
    assert storage.trace.events() == ()
    assert len(storage.trace) == 0


def test_g_the_m1_oracle_uses_no_randomness():
    """M1's oracle stays deterministic: all randomness lives in the M2 planner."""
    path = _m1_module()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in set(FORBIDDEN_ORACLE_RANDOMNESS) | {"os"}, path.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in FORBIDDEN_ORACLE_RANDOMNESS, path.name
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            assert name not in {"random", "randint", "shuffle", "sample", "choice",
                                "urandom", "token_bytes"}, path.name


def test_g_the_package_holds_the_m1_oracle_and_the_m2_planner_only():
    assert {path.name for path in _iter_swat_modules()} == set(SWAT_MODULES)
    tree = ast.parse(_m1_module().read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ["FunctionalOracleError", "FunctionalMergeOracleResult"]
    public_functions = [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    assert public_functions == [
        "flatten_blocks", "output_key_stream", "collect_output_blocks",
        "functional_merge_oracle",
    ]


# ---------------------------------------------------------------------------
# H. the M0 frozen substrate is untouched
# ---------------------------------------------------------------------------


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("m1_verify_manifest", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h_the_35_file_frozen_substrate_manifest_still_verifies():
    entries = [
        line.split() for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 35
    assert _load_verify_module().main(["--quiet", "--root", str(REPO_ROOT)]) == 0


def test_h_the_m1_implementation_lives_outside_the_frozen_package():
    assert (SWAT_PACKAGE / "functional_oracle.py").is_file()
    frozen = {line.split()[1] for line in
              MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert not any("swat_m_block" in relative for relative in frozen)
