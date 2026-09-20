"""G5-A focused tests (Issue #21 §11).

The module under test is ``enhanced_letindex.incremental_merge``: the resumable
bounded-state newer-wins logical merge core and the step-invariant incremental PGM.
G4 (``protected_merge``) is used **only as an oracle** here, never as the engine.

Section map (Issue #21 §11):

```text
A  status closure / frozen baseline / G4 regressions / scope isolation
B  merge semantics, ordering validation, packing and streaming
C  work quantum, pause/resume, budgets, determinism, DONE policy
D  bounded merge working state
E  incremental PGM equivalence and step-boundary invisibility
F  G4 / oracle equivalence
G  no index, trace, schema, allocator or epoch mutation
```

.. note::

   DERIVED TEST, NOT BYTE-IDENTICAL IMPORT.
   Imported for M0 from EnhancedLETIndex@4e68be75 and reduced to the parts
   that exercise the frozen common LETIndex substrate only.
   Dropped tests (structurally require the excluded G4/G5-B defence modules
   or pin the original repository's milestone state):
     test_a1 (PROJECT-STATE.md / decision-diff pin),
     test_a2 (baseline-g2b0 tag pin), test_a3 (runs the excluded G4 suites),
     test_b2b's G4-engine section, section F (g4_oracle / g4_pgm_index /
     test_f10 / test_f10b), test_g20_a (DefendedExperiment).
   All retained tests are byte-identical to the source and none had an
   assertion weakened.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.incremental_merge import (
    FINALIZE_POLICY,
    POST_DONE_ADVANCE_POLICY,
    IncrementalMergeConfig,
    IncrementalMergeError,
    IncrementalMergeJob,
    IncrementalPgmBuilder,
    IncrementalPgmState,
    LogicalRunView,
)
from enhanced_letindex.pgm import BatchPgmIndex, build_batch_pgm, make_segmentation

CODES_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = CODES_DIR.parent
CLI_ENV = {**os.environ, "PYTHONPATH": str(CODES_DIR / "src")}
MODULE_PATH = CODES_DIR / "src" / "enhanced_letindex" / "incremental_merge.py"

ITEMS_PER_BLOCK = 8
PGM_EPSILON = 8
SEED = 31
PRP_KEY = 1234

SOURCE_LEVEL = 4
TARGET_LEVEL = 5
PGM_SOURCE_LEVEL = 5
PGM_TARGET_LEVEL = 6

#: The G4 fixture shape: 12 + 12 records sharing 6 keys -> 18 merged records.
SOURCE_ITEMS = tuple((2_000 + rank, f"s{rank}") for rank in range(12))
TARGET_ITEMS = tuple((2_006 + rank, f"t{rank}") for rank in range(12))
EXPECTED_MERGED_ITEMS = 18
EXPECTED_DUPLICATES = 6
EXPECTED_BLOCK_SIZES = (8, 8, 2)

#: Hand-worked toy fixture (Issue #21 §11.7): 4 + 4 records, two duplicates.
TOY_SOURCE = ((1, "s1"), (3, "s3"), (5, "s5"), (7, "s7"))
TOY_TARGET = ((2, "t2"), (3, "t3"), (6, "t6"), (7, "t7"))
TOY_MERGED = ((1, "s1"), (2, "t2"), (3, "s3"), (5, "s5"), (6, "t6"), (7, "s7"))

#: A fixture whose canonical segmentation has three 10-record segments, so merge-step
#: and output-block boundaries necessarily fall *inside* a PGM segment (§11.17).
SEGMENT_GROUPS = 3
SEGMENT_GROUP_SIZE = 10
SEGMENT_GAP = 100_000
SEGMENT_EPSILON = 0
SEGMENT_ITEMS_PER_BLOCK = 4


def _segment_keys() -> tuple:
    keys = []
    base = 0
    for _ in range(SEGMENT_GROUPS):
        keys.extend(base + rank for rank in range(SEGMENT_GROUP_SIZE))
        base += SEGMENT_GAP
    return tuple(keys)


SEGMENT_KEYS = _segment_keys()
SEGMENT_SOURCE = tuple((key, f"a{key}") for key in SEGMENT_KEYS[:15])
SEGMENT_TARGET = tuple((key, f"b{key}") for key in SEGMENT_KEYS[15:])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def logical_oracle(source_items, target_items):
    """An independent dict-based oracle of the frozen newer-wins merge."""
    newer, older = dict(source_items), dict(target_items)
    return tuple(
        (key, newer[key] if key in newer else older[key])
        for key in sorted(set(newer) | set(older))
    )


def streamed_records(blocks) -> tuple:
    """The logical record stream a caller observes from the emitted blocks."""
    return tuple(
        (key, value) for block in blocks for key, value in block.records()
    )


def key_stream(records) -> tuple:
    return tuple(RecordKey(key) for key, _ in records)


def view(level: int, items) -> LogicalRunView:
    return LogicalRunView.from_records(level, items)


def run_merge(
    source_items,
    target_items,
    schedule,
    *,
    items_per_block: int = ITEMS_PER_BLOCK,
    epsilon: int = PGM_EPSILON,
    sink=None,
    levels=(SOURCE_LEVEL, TARGET_LEVEL),
):
    """Drive a job with the given budget schedule; returns (blocks, result, steps, job).

    The schedule is consumed first; if it does not reach DONE the job is finished with
    ``advance(1)`` calls so that every helper run ends in a finalized job.
    """
    job = IncrementalMergeJob.begin(
        view(levels[0], source_items),
        view(levels[1], target_items),
        items_per_block=items_per_block,
        epsilon=epsilon,
        sink=sink,
    )
    blocks: list = []
    steps: list = []
    for q in schedule:
        step = job.advance(q)
        steps.append(step)
        blocks.extend(step.blocks)
        if step.done:
            break
    while not job.done:
        step = job.advance(1)
        steps.append(step)
        blocks.extend(step.blocks)
    return blocks, job.finalize(), steps, job


def fixed_schedule(q: int, count: int = 512) -> list:
    return [q] * count


def scan_schedule(total: int) -> list:
    """An irregular schedule that cannot align with any structural boundary."""
    pattern = [7, 1, 3, 2, 11, 1, 4, 1, 9, 2, 1, 5, 3, 1, 1, 6]
    schedule = []
    while len(schedule) * 5 < total * 3:
        schedule.extend(pattern)
    return schedule


def block_signature(blocks) -> tuple:
    return tuple((block.rank, block.item_count) for block in blocks)


# ---------------------------------------------------------------------------
# A. scope isolation (repo-state and milestone pins dropped in the derivation)
# ---------------------------------------------------------------------------


def test_a4_g5a_is_trusted_logical_computation_only():
    """Issue #21 §2/§11.20: no physical I/O, SlotId, allocator, epoch or later work."""
    import ast

    source = MODULE_PATH.read_text(encoding="utf-8")
    # The *executable* module must not touch the physical/later milestones at all; the
    # module docstring may only name them in its explicit out-of-scope list.
    tree = ast.parse(source)
    code = source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                code = code.replace(doc, "")
    forbidden = (
        "SlotId", "UntrustedStorage", "allocator", "Allocator",
        "structure_version", "publish_structural", "epoch_prp", "epoch_id",
        "run_protected_merge", "ProtectedMerge", "MERGE_PHASES",
        "TraceEvent", "TraceOperation", "reshuffle", "read_slot", "write_slot",
        "ORAM", "BORPStream", "CacheShuffle", "SWAT", "attack", "leakage_metric",
    )
    for token in forbidden:
        assert token not in code, f"incremental_merge.py code must not mention {token}"
    for module in ("protected_merge", "defended_index", "query_defense", "storage"):
        assert f"from .{module} import" not in code, module
    # the docstring states the deferrals that the code must not implement
    for deferred in ("G5-B", "G6", "out of scope", "physical-slot scheduling"):
        assert deferred in source, deferred
    # the module imports only the frozen canonical PGM machinery and the identifiers
    imports = {
        line.strip() for line in code.splitlines()
        if line.startswith("from ") or line.startswith("import ")
    }
    assert imports == {
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "from typing import Callable, Optional, Sequence, Tuple",
        "from .identifiers import RecordKey",
        "from .pgm import (",
    }, imports

    # ... and importing it really does not pull in the G4 / defended path
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import enhanced_letindex.incremental_merge, sys;"
            "print('enhanced_letindex.defended_index' in sys.modules,"
            "'enhanced_letindex.protected_merge' in sys.modules,"
            "'enhanced_letindex.query_defense' in sys.modules)",
        ],
        cwd=str(CODES_DIR), env=CLI_ENV, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False False False"


# ---------------------------------------------------------------------------
# B. merge semantics, validation, packing, streaming
# ---------------------------------------------------------------------------


def test_b2_newer_wins_semantics_for_source_only_target_only_and_equal_keys():
    """Issue #21 §11.2 + §4: source is newer, target older, duplicates take source."""
    blocks, result, _, _ = run_merge(SOURCE_ITEMS, TARGET_ITEMS, fixed_schedule(3))
    assert streamed_records(blocks) == logical_oracle(SOURCE_ITEMS, TARGET_ITEMS)
    assert result.record_count == EXPECTED_MERGED_ITEMS
    assert result.duplicate_count == EXPECTED_DUPLICATES
    # a duplicate keeps the *source* value even though the target value differs
    merged = dict(streamed_records(blocks))
    for key in range(2_006, 2_012):
        assert merged[key] == f"s{key - 2_000}"
        assert dict(TARGET_ITEMS)[key] == f"t{key - 2_006}"
    # a source-only key keeps the source value, a target-only key the target value
    assert merged[2_000] == "s0" and merged[2_017] == "t11"
    # the two runs are never reordered: the output is strictly sorted by key
    keys = [key for key, _ in streamed_records(blocks)]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_b2b_negative_zero_and_positive_keys_are_in_the_logical_domain():
    """Review round 1 item 4a: the G5-A key domain is the accepted G4 key domain."""
    source = ((-9, "s-9"), (-4, "s-4"), (0, "s0"), (7, "s7"))
    target = ((-7, "t-7"), (-4, "t-4"), (0, "t0"), (9, "t9"))
    merged = logical_oracle(source, target)
    assert [key for key, _ in merged] == [-9, -7, -4, 0, 7, 9]

    # a negative key is accepted and ordered exactly like the accepted RecordKey
    assert LogicalRunView.from_records(4, source).keys[0] == RecordKey(-9)
    assert LogicalRunView.from_records(5, target).keys[0] < RecordKey(-7 + 2)

    for epsilon in (0, 1, 4, 8, 64):
        blocks, result, _, _ = run_merge(
            source, target, scan_schedule(len(merged)), epsilon=epsilon
        )
        assert streamed_records(blocks) == merged, epsilon
        assert result.duplicate_count == 2, epsilon
        assert result.pgm == build_batch_pgm(key_stream(merged), epsilon), epsilon
        assert result.pgm.segments == make_segmentation(key_stream(merged), epsilon), epsilon
        assert (result.first_key, result.last_key) == (RecordKey(-9), RecordKey(9)), epsilon


def test_b3_malformed_input_is_rejected_at_construction():
    """Issue #21 §4: strict input ordering; malformed input never produces output."""
    for items in (
        ((2, "a"), (1, "b")),                      # descending
        ((1, "a"), (1, "b")),                      # duplicate key inside one run
    ):
        with pytest.raises(IncrementalMergeError, match="strictly sorted"):
            LogicalRunView.from_records(SOURCE_LEVEL, items)
    with pytest.raises(IncrementalMergeError, match="keys but"):
        LogicalRunView(level=SOURCE_LEVEL, keys=(RecordKey(1),), values=())
    with pytest.raises(IncrementalMergeError, match="RecordKey"):
        LogicalRunView(level=SOURCE_LEVEL, keys=(1,), values=("a",))
    with pytest.raises(IncrementalMergeError, match="\\(key, value\\) pair"):
        LogicalRunView.from_records(SOURCE_LEVEL, (1, 2, 3))
    with pytest.raises(IncrementalMergeError, match="same level"):
        IncrementalMergeJob.begin(
            view(SOURCE_LEVEL, SOURCE_ITEMS),
            view(SOURCE_LEVEL, TARGET_ITEMS),
            items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
        )


def test_b1b_the_frozen_merge_identity_source_l_to_target_l_plus_1_is_enforced():
    """Issue #21 §4 + review round 1 item 1: same-level / reverse / gapped refusals."""
    # same level
    with pytest.raises(IncrementalMergeError, match="same level") as same:
        IncrementalMergeJob.begin(
            view(5, SOURCE_ITEMS), view(5, TARGET_ITEMS),
            items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
        )
    assert "same level twice (L = 5)" in str(same.value)
    # reverse direction: the target must be the *older* run L + 1, so target <= source
    for source_level, target_level in ((5, 4), (4, 3), (6, 3), (2, 0)):
        with pytest.raises(IncrementalMergeError, match="reverse direction") as reverse:
            IncrementalMergeJob.begin(
                view(source_level, SOURCE_ITEMS), view(target_level, TARGET_ITEMS),
                items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
            )
        assert f"source {source_level} -> target {target_level}" in str(reverse.value)
    # gapped: non-adjacent forward requests
    for source_level, target_level, gap in ((4, 6, 1), (3, 9, 5), (0, 2, 1)):
        with pytest.raises(IncrementalMergeError, match="a gap of") as gapped:
            IncrementalMergeJob.begin(
                view(source_level, SOURCE_ITEMS), view(target_level, TARGET_ITEMS),
                items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
            )
        assert f"a gap of {gap}" in str(gapped.value)
    # the refusal is pre-output: no job object is created, so no decision was taken,
    # no record was buffered and no key reached the PGM
    job = None
    try:
        IncrementalMergeJob.begin(
            view(5, SOURCE_ITEMS), view(4, TARGET_ITEMS),
            items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
        )
        job = "created"
    except IncrementalMergeError:
        pass
    assert job is None
    # ... while the frozen identity L -> L + 1 still works and is adjacent
    accepted = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SOURCE_ITEMS), view(TARGET_LEVEL, TARGET_ITEMS),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    assert (accepted.source.level, accepted.target.level) == (SOURCE_LEVEL, TARGET_LEVEL)
    assert accepted.target.level == accepted.source.level + 1
    assert accepted.pgm_state.record_count == 0


def test_b3b_invalid_config_and_invalid_budget_are_refused():
    """Issue #21 §5: the budget is a positive plain int; configs are validated."""
    for bad in (0, -1, True, 1.5, None, "3"):
        with pytest.raises(IncrementalMergeError, match="items_per_block"):
            IncrementalMergeConfig(items_per_block=bad, epsilon=PGM_EPSILON)
    for bad in (-1, True, 2.5, None):
        with pytest.raises(IncrementalMergeError, match="epsilon"):
            IncrementalMergeConfig(items_per_block=ITEMS_PER_BLOCK, epsilon=bad)
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=2, epsilon=0,
    )
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(IncrementalMergeError, match="budget q"):
            job.advance(bad)
    assert job.state.source_cursor == 0 and job.state.target_cursor == 0


def test_b6_a_duplicate_consumes_both_inputs_atomically_but_counts_one_decision():
    """Issue #21 §11.6 + §5: equal keys consume both records, one output decision."""
    source = ((1, "s1"), (3, "s3"))
    target = ((1, "t1"), (4, "t4"))
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, source), view(TARGET_LEVEL, target),
        items_per_block=1, epsilon=0,
    )
    step = job.advance(1)
    assert step.decisions == 1
    assert step.emitted_records == 1
    assert [block.records() for block in step.blocks] == [((1, "s1"),)]
    assert job.state.source_cursor == 1 and job.state.target_cursor == 1
    assert job.state.output_records == 1
    # the duplicate pair is a single decision: q=1 could never have split it
    assert step.blocks[0].item_count == 1


def test_b4_b7_canonical_packing_block_ranks_and_final_partial_block():
    """Issue #21 §11.8/§11.9: full blocks stream out, one rank-ordered partial tail."""
    blocks, result, steps, job = run_merge(SOURCE_ITEMS, TARGET_ITEMS, fixed_schedule(1))
    assert [block.item_count for block in blocks] == list(EXPECTED_BLOCK_SIZES)
    assert [block.rank for block in blocks] == [0, 1, 2]
    assert [block.rank for block in blocks] == list(range(result.block_count))
    assert result.block_count == 3
    # packing is exactly the canonical target packing of the merged stream
    merged = logical_oracle(SOURCE_ITEMS, TARGET_ITEMS)
    for block in blocks:
        start = block.rank * ITEMS_PER_BLOCK
        assert block.records() == merged[start:start + block.item_count]
    # a full block is emitted in the very call that fills it ...
    for step in steps:
        if step.state.output_records == ITEMS_PER_BLOCK:
            assert [b.rank for b in step.blocks] == [0]
            assert step.state.pending_items == 0
    # ... while the final partial block waits for completion and appears exactly once
    partial_steps = [
        step for step in steps
        if any(block.item_count == EXPECTED_BLOCK_SIZES[-1] for block in step.blocks)
    ]
    assert len(partial_steps) == 1
    assert partial_steps[0].done is True
    assert job.state.pending_items == 0
    # the target packing capacity is respected
    assert all(block.item_count <= ITEMS_PER_BLOCK for block in blocks)


def test_b9_duplicate_across_a_block_boundary_and_boundary_on_a_duplicate_pair():
    """Issue #21 §10: the block boundary may fall inside a duplicate neighbourhood."""
    source = ((1, "s1"), (2, "s2"), (3, "s3"))
    target = ((2, "t2"), (3, "t3"), (4, "t4"))
    for items_per_block in (1, 2, 3):
        blocks, result, _, _ = run_merge(
            source, target, fixed_schedule(1), items_per_block=items_per_block, epsilon=1
        )
        merged = logical_oracle(source, target)
        assert streamed_records(blocks) == merged
        assert result.duplicate_count == 2
        assert all(
            b.records() == merged[b.rank * items_per_block:
                                  b.rank * items_per_block + b.item_count]
            for b in blocks
        )
        assert result.pgm == build_batch_pgm(key_stream(merged), 1)


def test_b11_empty_and_one_sided_inputs():
    """Issue #21 §11.11: empty source, empty target and both empty are legal."""
    for source, target, expected in (
        ((), TOY_TARGET, logical_oracle((), TOY_TARGET)),
        (TOY_SOURCE, (), logical_oracle(TOY_SOURCE, ())),
        ((), (), ()),
    ):
        blocks, result, _, _ = run_merge(source, target, fixed_schedule(2), epsilon=0)
        assert streamed_records(blocks) == expected
        assert result.record_count == len(expected)
        assert result.duplicate_count == 0
        assert result.pgm == build_batch_pgm(key_stream(expected), 0)
        assert [b.rank for b in blocks] == list(range(result.block_count))
    # an empty merge produces the empty canonical PGM and no block at all
    blocks, result, steps, _ = run_merge((), (), [4], epsilon=3)
    assert blocks == [] and result.block_count == 0
    assert result.pgm == BatchPgmIndex(3, 0, None, None, ())
    assert result.pgm.first_key is None and result.pgm.last_key is None
    assert steps[-1].done is True


# ---------------------------------------------------------------------------
# C. work quantum, pause/resume, budgets, determinism, DONE policy
# ---------------------------------------------------------------------------


def test_c5_q1_emits_at_most_one_output_record_per_advance():
    """Issue #21 §11.4 + §5: q is an output-decision budget."""
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SOURCE_ITEMS), view(TARGET_LEVEL, TARGET_ITEMS),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    decisions = []
    while not job.done:
        step = job.advance(1)
        assert step.decisions <= 1
        assert len(step.blocks) <= 1
        # a completed block may contain records decided in earlier calls, but never more
        # than one canonical block worth of them
        assert step.emitted_records <= ITEMS_PER_BLOCK
        decisions.append(step.decisions)
    assert decisions == [1] * EXPECTED_MERGED_ITEMS
    assert job.state.output_records == EXPECTED_MERGED_ITEMS
    # a budget larger than the remaining work returns the actual smaller progress
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=4, epsilon=0,
    )
    step = job.advance(1_000)
    assert step.decisions == len(TOY_MERGED) and step.done is True
    assert job.advance(5).decisions == 0


def test_c7_hand_worked_toy_trace_of_cursors_and_output_blocks():
    """Issue #21 §11.7: exact i/j/r trace over a hand-worked toy case."""
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=2, epsilon=0,
    )
    trace = []
    observed = []
    for q in (2, 2, 2):
        step = job.advance(q)
        observed.extend(step.blocks)
        trace.append(
            (
                q,
                step.decisions,
                step.state.source_cursor,
                step.state.target_cursor,
                step.state.output_records,
                [block.rank for block in step.blocks],
                step.state.pending_items,
                step.done,
            )
        )
    assert trace == [
        # q, decisions, i, j, r, emitted ranks, pending, done
        (2, 2, 1, 1, 2, [0], 0, False),   # 1(s) then 2(t): block 0 = [1s, 2t]
        (2, 2, 3, 2, 4, [1], 0, False),   # 3(s,duplicate) then 5(s): block 1 = [3s, 5s]
        (2, 2, 4, 4, 6, [2], 0, True),    # 6(t) then 7(s,duplicate): block 2 = [6t, 7s]
    ]
    assert streamed_records(observed) == TOY_MERGED
    assert [block.records() for block in observed] == [
        ((1, "s1"), (2, "t2")), ((3, "s3"), (5, "s5")), ((6, "t6"), (7, "s7"))
    ]
    assert job.finalize().record_count == len(TOY_MERGED)
    assert job.finalize().duplicate_count == 2
    assert job.finalize().block_count == 3

    # the same fixture with q=1 must show the same structural milestones
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=2, epsilon=0,
    )
    milestones = []
    while not job.done:
        step = job.advance(1)
        milestones.append(
            (step.state.output_records, step.state.source_cursor,
             step.state.target_cursor, [b.rank for b in step.blocks])
        )
    assert milestones == [
        (1, 1, 0, []),
        (2, 1, 1, [0]),
        (3, 2, 2, []),
        (4, 3, 2, [1]),
        (5, 3, 3, []),
        (6, 4, 4, [2]),
    ]

    # a single q=3 call on the same fixture finishes it in two steps, same packing
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=2, epsilon=0,
    )
    step = job.advance(3)
    # decisions 1(s), 2(t) and 3(s, duplicate): the duplicate advances BOTH cursors
    assert (step.decisions, step.state.source_cursor, step.state.target_cursor) == (3, 2, 2)
    assert [b.rank for b in step.blocks] == [0] and step.done is False
    assert step.state.pending_items == 1
    step = job.advance(3)
    assert (step.decisions, step.state.source_cursor, step.state.target_cursor) == (3, 4, 4)
    assert step.done is True and [b.rank for b in step.blocks] == [1, 2]


def test_c12_pause_resume_at_every_possible_boundary():
    """Issue #21 §11.12: the job is resumable at every boundary of a small fixture."""
    reference_blocks, reference_result, _, _ = run_merge(
        TOY_SOURCE, TOY_TARGET, fixed_schedule(1), items_per_block=2, epsilon=0
    )
    reference = streamed_records(reference_blocks)
    total = len(TOY_MERGED)
    for prefix in range(1, total + 2):
        blocks, result, _, _ = run_merge(
            TOY_SOURCE, TOY_TARGET, [prefix] + [1] * total, items_per_block=2, epsilon=0
        )
        assert streamed_records(blocks) == reference, prefix
        assert result.pgm == reference_result.pgm, prefix
        assert block_signature(blocks) == block_signature(reference_blocks), prefix
    # ... and pausing exactly *at* a block boundary keeps the same packing
    for prefix in range(1, total + 1):
        job = IncrementalMergeJob.begin(
            view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
            items_per_block=2, epsilon=0,
        )
        blocks = []
        hit = False
        while not job.done:
            step = job.advance(prefix)
            blocks.extend(step.blocks)
            if step.state.output_records >= 1:
                hit = True
        assert hit and streamed_records(blocks) == reference


def test_c13_three_or_more_budget_schedules_give_the_identical_result():
    """Issue #21 §11.5: q=1, q=N, item size, huge q and irregular budgets agree."""
    schedules = {
        "q=1": fixed_schedule(1),
        "q=2": fixed_schedule(2),
        "q=3": fixed_schedule(3),
        "q=items_per_block": fixed_schedule(ITEMS_PER_BLOCK),
        "q=huge": fixed_schedule(10_000, 4),
        "irregular": scan_schedule(EXPECTED_MERGED_ITEMS),
    }
    results = {}
    for label, schedule in schedules.items():
        blocks, result, steps, job = run_merge(SOURCE_ITEMS, TARGET_ITEMS, schedule)
        results[label] = (
            streamed_records(blocks),
            block_signature(blocks),
            result.record_count,
            result.block_count,
            result.duplicate_count,
            result.pgm,
            result.pgm_segment_start_ranks,
            job.working_state_report()["retained_output_blocks"],
        )
    reference = results["q=1"]
    for label, value in results.items():
        assert value == reference, label
    assert reference[2] == EXPECTED_MERGED_ITEMS
    assert reference[7] == 0


def test_c8_determinism_of_identical_schedules():
    """Issue #21 §8: same input + same budget sequence -> identical step results."""
    runs = []
    for _ in range(2):
        blocks, result, steps, job = run_merge(
            SOURCE_ITEMS, TARGET_ITEMS, scan_schedule(EXPECTED_MERGED_ITEMS)
        )
        runs.append(
            (
                streamed_records(blocks),
                block_signature(blocks),
                tuple(
                    (s.decisions, s.state.source_cursor, s.state.target_cursor,
                     s.state.output_records, [b.rank for b in s.blocks], s.done)
                    for s in steps
                ),
                result.pgm,
                job.working_state_report()["step_counter"],
            )
        )
    assert runs[0] == runs[1]


def test_c14_finalize_before_done_is_refused_and_post_done_rules_are_frozen():
    """Issue #21 §11.13/§11.14 + §8: DONE policy and finalize policy are frozen."""
    assert POST_DONE_ADVANCE_POLICY == "no-op"
    assert FINALIZE_POLICY == "idempotent"
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, TOY_SOURCE), view(TARGET_LEVEL, TOY_TARGET),
        items_per_block=2, epsilon=0,
    )
    job.advance(3)
    with pytest.raises(IncrementalMergeError, match="not DONE"):
        job.finalize()
    while not job.done:
        job.advance(1)
    state_before = job.state
    noop = job.advance(9)
    assert noop.decisions == 0 and noop.blocks == () and noop.done is True
    assert noop.state == state_before
    assert job.state == state_before
    first = job.finalize()
    assert job.finalize() is first          # idempotent
    assert first.record_count == len(TOY_MERGED)


def test_c14b_post_done_advance_is_a_real_no_op():
    """Review round 1 item 3: a valid post-DONE advance() mutates no job-owned field."""
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SOURCE_ITEMS), view(TARGET_LEVEL, TARGET_ITEMS),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    while not job.done:
        job.advance(2)
    before = job.working_state_report()
    before_pgm = job.pgm_state
    finalized = job.finalize()
    for q in (1, 5, 10_000):
        step = job.advance(q)
        assert (step.decisions, step.blocks, step.done) == (0, (), True)
        assert step.state == job.state
    # the *complete* working-state report (including the call counter) is unchanged
    assert job.working_state_report() == before
    assert job.pgm_state == before_pgm
    assert job.finalize() is finalized
    # an invalid budget is still refused after DONE, and still mutates nothing
    for bad in (0, -1, True):
        with pytest.raises(IncrementalMergeError, match="budget q"):
            job.advance(bad)
    assert job.working_state_report() == before
    assert job.finalize() is finalized


# ---------------------------------------------------------------------------
# D. bounded merge working state
# ---------------------------------------------------------------------------


#: The complete job-owned state surface; nothing here may accumulate output.
EXPECTED_JOB_SLOTS = {
    "_config", "_source", "_target",
    "_source_cursor", "_target_cursor", "_output_records", "_emitted_blocks",
    "_duplicate_count",
    "_pending_keys", "_pending_values",
    "_done", "_step_counter", "_advance_calls", "_max_pending_items",
    "_pgm", "_finalized", "_sink",
}


def test_d10_the_runtime_job_never_accumulates_completed_blocks():
    """Issue #21 §11.10 + §6: emitted blocks leave the job immediately."""
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SOURCE_ITEMS), view(TARGET_LEVEL, TARGET_ITEMS),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    assert set(job.__slots__) == EXPECTED_JOB_SLOTS
    observables = 0
    while not job.done:
        step = job.advance(1)
        observables += len(step.blocks)
        report = job.working_state_report()
        assert report["retained_output_blocks"] == 0
        assert report["retained_step_objects"] == 0
        assert report["retained_record_history"] == 0
        assert report["retained_output_records"] <= ITEMS_PER_BLOCK
        assert report["partial_block_items"] <= ITEMS_PER_BLOCK
        assert report["max_partial_block_items"] <= ITEMS_PER_BLOCK
    assert observables == 3
    # the only containers the job owns are the partial buffer and the PGM metadata
    growable = {
        slot for slot in EXPECTED_JOB_SLOTS
        if isinstance(getattr(job, slot), (list, dict, tuple))
    }
    assert growable == {"_pending_keys", "_pending_values"}
    assert len(job._pending_keys) <= ITEMS_PER_BLOCK


def test_d10b_the_reported_peak_is_the_true_instantaneous_pending_peak():
    """Review round 1 item 4b: the peak is recorded before any possible flush."""
    # 18 records with items_per_block = 8: the buffer necessarily reaches 8
    _, _, _, job = run_merge(SOURCE_ITEMS, TARGET_ITEMS, fixed_schedule(1))
    report = job.working_state_report()
    assert report["max_partial_block_items"] == ITEMS_PER_BLOCK == 8
    assert report["partial_block_items"] == 0
    # a run whose whole output is smaller than one block reports its own true peak
    _, _, _, tail_job = run_merge(
        ((1, "a"), (2, "b"), (3, "c")), (), fixed_schedule(1),
        items_per_block=8, epsilon=0,
    )
    assert tail_job.working_state_report()["max_partial_block_items"] == 3
    # and a full ipb=16 run reports 16, never the between-call residual 15
    source = tuple((rank * 2, f"s{rank}") for rank in range(4_000))
    target = tuple((rank * 2 + 1, f"t{rank}") for rank in range(3_000, 7_000))
    blocks, result, _, big_job = run_merge(
        source, target, fixed_schedule(1), items_per_block=16, epsilon=4
    )
    report = big_job.working_state_report()
    assert report["max_partial_block_items"] == 16
    assert report["max_partial_block_items"] >= report["partial_block_items"]
    assert report["retained_output_records"] == 0        # after the final flush
    assert result.block_count == 500
    assert all(block.item_count <= 16 for block in blocks)


def test_d19_a_large_tiny_q_run_keeps_the_ordinary_merge_state_bounded():
    """Issue #21 §11.19 + §9: bounded working state under tiny budgets."""
    source = tuple((rank * 2, f"s{rank}") for rank in range(4_000))
    target = tuple((rank * 2 + 1, f"t{rank}") for rank in range(3_000, 7_000))
    merged = logical_oracle(source, target)
    assert len(merged) == 8_000
    blocks, result, steps, job = run_merge(
        source, target, fixed_schedule(1), items_per_block=16, epsilon=4
    )
    report = job.working_state_report()
    assert result.record_count == len(merged)
    assert result.block_count == 500
    assert streamed_records(blocks) == merged
    assert report["retained_output_blocks"] == 0
    assert report["retained_step_objects"] == 0
    assert report["retained_record_history"] == 0
    assert report["max_partial_block_items"] == 16
    assert report["advance_calls"] == len(merged)
    assert report["step_counter"] == len(merged)
    # the allowed growth is only the finalized PGM segment metadata / active canonical
    # state, which is far smaller than the output stream
    assert report["pgm_closed_segments"] < len(merged)
    assert report["pgm_active_points_max"] <= len(merged)
    assert report["pgm_retained_keys"] == 0
    assert result.pgm == build_batch_pgm(key_stream(merged), 4)


# ---------------------------------------------------------------------------
# E. incremental PGM
# ---------------------------------------------------------------------------


PGM_FIXTURES = {
    "standard": (SOURCE_ITEMS, TARGET_ITEMS, ITEMS_PER_BLOCK),
    "toy": (TOY_SOURCE, TOY_TARGET, 2),
    "segmented": (SEGMENT_SOURCE, SEGMENT_TARGET, SEGMENT_ITEMS_PER_BLOCK),
    "source_only": (SOURCE_ITEMS, (), ITEMS_PER_BLOCK),
    "target_only": ((), TARGET_ITEMS, ITEMS_PER_BLOCK),
    "empty": ((), (), ITEMS_PER_BLOCK),
    "single": (((5, "s"),), ((5, "t"),), 1),
}


def test_e15_incremental_pgm_equals_build_batch_pgm_for_every_fixture():
    """Issue #21 §11.15 + §7: finalize() == build_batch_pgm(all_output_keys, epsilon)."""
    for label, (source, target, items_per_block) in PGM_FIXTURES.items():
        merged = logical_oracle(source, target)
        for epsilon in (0, 1, 2, 4, 8, 64):
            blocks, result, _, job = run_merge(
                source, target, scan_schedule(max(len(merged), 1)),
                items_per_block=items_per_block, epsilon=epsilon,
            )
            batch = build_batch_pgm(key_stream(merged), epsilon)
            assert result.pgm == batch, (label, epsilon)
            assert result.pgm_segment_start_ranks == tuple(
                segment.start_rank for segment in batch.segments
            ), (label, epsilon)
            assert streamed_records(blocks) == merged, (label, epsilon)
            # the builder consumed exactly the record stream it was asked to
            assert job.pgm_state.record_count == len(merged), (label, epsilon)
            assert result.pgm_segment_start_ranks == job.pgm_state.segment_start_ranks


def test_e16_the_pgm_result_is_independent_of_the_merge_step_boundaries():
    """Issue #21 §11.16: step endpoints are invisible to the PGM."""
    merged = logical_oracle(SEGMENT_SOURCE, SEGMENT_TARGET)
    batch = build_batch_pgm(key_stream(merged), SEGMENT_EPSILON)
    assert len(batch.segments) == 3          # three 10-record canonical segments
    schedule_signatures = set()
    for schedule in (
        fixed_schedule(1),
        fixed_schedule(3),
        fixed_schedule(SEGMENT_ITEMS_PER_BLOCK),
        fixed_schedule(10),
        scan_schedule(len(merged)),
    ):
        blocks, result, _, job = run_merge(
            SEGMENT_SOURCE, SEGMENT_TARGET, schedule,
            items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON,
        )
        schedule_signatures.add(
            (
                result.pgm,
                result.pgm_segment_start_ranks,
                block_signature(blocks),
                streamed_records(blocks),
            )
        )
    assert len(schedule_signatures) == 1
    assert schedule_signatures.pop()[0] == batch


def test_e17_a_pgm_segment_crosses_advance_and_block_boundaries_without_a_forced_split():
    """Issue #21 §11.17 + §7: merge step endpoint != PGM segment endpoint."""
    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SEGMENT_SOURCE), view(TARGET_LEVEL, SEGMENT_TARGET),
        items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON,
    )
    observed = []
    blocks = []
    while not job.done:
        step = job.advance(3)                      # q=3 vs items_per_block=4: misaligned
        blocks.extend(step.blocks)
        snapshot = job.pgm_state
        observed.append(
            (
                step.state.output_records,
                snapshot.closed_segment_start_ranks,
                snapshot.active_segment_span,
                snapshot.closed_in_last_add,
                [block.rank for block in step.blocks],
            )
        )
    result = job.finalize()
    # the canonical segmentation is exactly the batch one: (0, 10, 20).  Any forced
    # split at a step or block boundary would show up as an extra start rank.
    assert result.pgm_segment_start_ranks == (0, 10, 20)
    batch = build_batch_pgm(key_stream(logical_oracle(SEGMENT_SOURCE, SEGMENT_TARGET)), 0)
    assert result.pgm == batch
    assert result.pgm_segment_start_ranks == tuple(s.start_rank for s in batch.segments)

    by_records = {entries[0]: entries for entries in observed}
    # merge-step boundaries fell inside segment 0 (r = 3, 6, 9) and the segment stayed open
    for records in (3, 6, 9):
        closed, span, closed_now = by_records[records][1:4]
        assert closed == (), records
        assert closed_now == 0, records
        assert span == (0, records - 1), records
    # ... and output-block boundaries (items_per_block = 4 -> ranks 4 and 8) fell inside
    # the very same open segment: block 0 completed in the call ending at r = 6, block 1
    # in the call ending at r = 9
    assert by_records[6][4] == [0]
    assert by_records[9][4] == [1]
    assert [block.item_count for block in blocks[:2]] == [4, 4]
    # the segment closes exactly at rank 10 — the canonical boundary — not at r = 3/6/9/12
    assert by_records[12][1] == (0,)
    assert by_records[12][2] == (10, 11)
    # the second segment likewise spans two step boundaries and two block boundaries
    assert by_records[15][1] == (0,) and by_records[15][2] == (10, 14)
    assert by_records[18][1] == (0,) and by_records[21][1] == (0, 10)
    assert by_records[24][1] == (0, 10) and by_records[24][2] == (20, 23)
    assert by_records[30][1] == (0, 10) and by_records[30][2] == (20, 29)

    # the same segmentation under a completely different budget
    _, other, _, other_job = run_merge(
        SEGMENT_SOURCE, SEGMENT_TARGET, fixed_schedule(1),
        items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON,
    )
    assert other.pgm == batch
    assert other.pgm_segment_start_ranks == (0, 10, 20)
    assert other_job.pgm_state.closed_segment_start_ranks == (0, 10)


def test_e18_the_pgm_builder_never_retains_the_output_key_stream():
    """Issue #21 §11.18 + §7: no full output-key list is kept for finalization."""
    builder = IncrementalPgmBuilder(PGM_EPSILON)
    merged = logical_oracle(SOURCE_ITEMS, TARGET_ITEMS)
    for key, _ in merged:
        builder.add_key(RecordKey(key))
        assert builder.retained_keys == 0
    attributes = vars(builder)
    assert set(attributes) == {
        "_epsilon", "_oplm", "_segments", "_closed_ranks", "_current_start_rank",
        "_count", "_last_key", "_first_key", "_max_active_points", "_closed_in_last_add",
    }
    for name, value in attributes.items():
        if isinstance(value, (list, tuple)):
            assert not any(isinstance(item, RecordKey) for item in value), name
    assert builder.finalize() == build_batch_pgm(key_stream(merged), PGM_EPSILON)


def test_e19_the_job_exposes_only_immutable_pgm_diagnostics():
    """Review round 1 item 2: the job never hands out its live mutable PGM builder."""
    import dataclasses

    job = IncrementalMergeJob.begin(
        view(SOURCE_LEVEL, SOURCE_ITEMS), view(TARGET_LEVEL, TARGET_ITEMS),
        items_per_block=ITEMS_PER_BLOCK, epsilon=PGM_EPSILON,
    )
    # the mutable builder is not published under any public name
    assert not hasattr(IncrementalMergeJob, "pgm_builder")
    assert not hasattr(job, "pgm_builder")
    snapshot = job.pgm_state
    assert isinstance(snapshot, IncrementalPgmState)
    assert not isinstance(snapshot, IncrementalPgmBuilder)
    # the snapshot is frozen: it cannot be used to move the job-owned PGM state
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.record_count = 999
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.segment_start_ranks = (7,)
    # no public attribute is the owned builder or a key-feeding API
    public = [name for name in dir(job) if not name.startswith("_")]
    assert "pgm_state" in public
    for name in public:
        value = getattr(job, name)
        assert not isinstance(value, IncrementalPgmBuilder), name
        assert not (callable(value) and "add" in name.lower()), name
    assert "add_key" not in public

    # the read-only surface tracks exactly the decided stream: a phantom key injected
    # through it (or through any other public API) would show up as a count mismatch
    stream = key_stream(logical_oracle(SOURCE_ITEMS, TARGET_ITEMS))
    counts = []
    while not job.done:
        step = job.advance(3)
        snapshot = job.pgm_state
        decided = step.state.output_records
        assert snapshot.record_count == decided
        assert snapshot.segment_start_ranks == tuple(
            segment.start_rank
            for segment in make_segmentation(stream[:decided], PGM_EPSILON)
        )
        counts.append(decided)
        # merely observing the diagnostics never feeds the PGM
        assert job.pgm_state.record_count == decided
    assert counts == sorted(counts) and counts[-1] == EXPECTED_MERGED_ITEMS
    result = job.finalize()
    assert result.pgm == build_batch_pgm(stream, PGM_EPSILON)
    assert job.pgm_state.record_count == EXPECTED_MERGED_ITEMS
    assert job.pgm_state.retained_keys == 0

    # the standalone builder stays public and usable, independent of any job
    standalone = IncrementalPgmBuilder(PGM_EPSILON)
    standalone.add_keys(key_stream(SOURCE_ITEMS))
    assert standalone.finalize() == build_batch_pgm(key_stream(SOURCE_ITEMS), PGM_EPSILON)
    assert standalone.snapshot().record_count == len(SOURCE_ITEMS)
    # the job itself keeps no key history either (see the bounded-state tests)
    _, _, _, job = run_merge(SOURCE_ITEMS, TARGET_ITEMS, fixed_schedule(1))
    report = job.working_state_report()
    assert report["pgm_retained_keys"] == 0
    assert report["retained_record_history"] == 0
    assert report["retained_output_records"] == 0


def test_e15b_randomized_differential_against_the_canonical_pgm():
    """Issue #21 §7: exact online equivalence over many legal shapes."""
    rng = random.Random(7)
    for trial in range(120):
        source_keys = sorted(rng.sample(range(0, 4_000), rng.randint(0, 40)))
        target_keys = sorted(rng.sample(range(0, 4_000), rng.randint(0, 40)))
        source = tuple((key, f"a{key}") for key in source_keys)
        target = tuple((key, f"b{key}") for key in target_keys)
        epsilon = rng.choice([0, 1, 2, 3, 5, 8, 17, 64])
        items_per_block = rng.choice([1, 2, 3, 5, 8, 16])
        merged = logical_oracle(source, target)
        schedule = [rng.randint(1, 6) for _ in range(len(merged) + 4)]
        blocks, result, _, _ = run_merge(
            source, target, schedule,
            items_per_block=items_per_block, epsilon=epsilon,
        )
        assert streamed_records(blocks) == merged, trial
        assert result.pgm == build_batch_pgm(key_stream(merged), epsilon), trial
        assert result.pgm.segments == make_segmentation(key_stream(merged), epsilon), trial
        assert all(block.item_count <= items_per_block for block in blocks), trial


# ---------------------------------------------------------------------------
# G. remaining isolation checks
# ---------------------------------------------------------------------------


