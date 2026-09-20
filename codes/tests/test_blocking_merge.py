"""G1-D blocking merge baseline tests.

Covers `decisions/0006-g1-d-blocking-merge.md`: upsert as a fresh newest level,
explicit blocking two-level merge with caller-supplied precedence, newer-wins
duplicate resolution, fresh canonical output construction, output PGM through
the accepted G1-B1 builder, publication-before-retirement ordering, exact
baseline trace composition, and post-merge query correctness through G1-C.

Test labels ``# [n]`` refer to the required-test list in GitHub Issue #5.
"""

from __future__ import annotations

import random

import pytest

from enhanced_letindex.builder import validate_level
from enhanced_letindex.config import Config
from enhanced_letindex.dataset import DuplicateKeyError
from enhanced_letindex.engine import (
    MergeResult,
    RetiredLevelError,
    TrustedEngine,
)
from enhanced_letindex.identifiers import BlockId, LevelId, RecordKey, SlotId
from enhanced_letindex.mapping import MissingLogicalBlockError
from enhanced_letindex.merge import EmptyUpdateBatchError, MergeInputError
from enhanced_letindex.pgm import PgmConfigurationError, build_batch_pgm
from enhanced_letindex.record import Record
from enhanced_letindex.trace import TraceOperation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rec(key, tag="v"):
    return Record(RecordKey(key), f"{tag}:{key}")


def build_engine(capacity=3, epsilon=2):
    return TrustedEngine(Config(block_capacity=capacity, pgm_epsilon=epsilon))


def build_pgm_level(engine, keys, tag="v"):
    """Build a normal (already existing, older-style) level with its PGM."""
    level_id = engine.build_level([rec(k, tag) for k in keys])
    engine.build_level_pgm(level_id)
    return level_id


def update_level(engine, keys, tag="NEW", with_pgm=True):
    level_id = engine.create_update_level([rec(k, tag) for k in keys])
    if with_pgm:
        engine.build_level_pgm(level_id)
    return level_id


def block_slots(engine, level_id):
    return [engine.mapping.lookup(b).value for b in engine.level(level_id).block_ids]


def flatten_level(engine, level_id):
    """All records of a level in logical order (own trace window, then cleared)."""
    engine.trace.clear()
    records = []
    for block_id in engine.level(level_id).block_ids:
        records.extend(engine.read_block(block_id).records)
    engine.trace.clear()
    return records


def observe(engine, plan, key):
    """One G1-C query on a cleared trace -> (result, [slot ids])."""
    engine.trace.clear()
    result = engine.lookup_levels(list(plan), RecordKey(key))
    return result, [ev.slot_id.value for ev in engine.trace.events()]


def merge_trace_events(engine, newer, older):
    """Run a full blocking merge and return (result, [(operation, slot)])."""
    engine.trace.clear()
    result = engine.merge_levels_blocking(newer, older)
    return result, [(ev.operation.value, ev.slot_id.value) for ev in engine.trace.events()]


def expected_merge_trace(newer_slots, older_slots, output_slots):
    """Frozen baseline composition (decisions/0006 §11), derived independently.

    A: READ newer blocks then older blocks (logical order)
    B: WRITE one per output block (construction order)
    C: READ one per output block (G1-B1 PGM construction, logical order)
    D: WRITE one per retired input block (newer then older, logical order)
    """
    return (
        [("READ", s) for s in newer_slots]
        + [("READ", s) for s in older_slots]
        + [("WRITE", s) for s in output_slots]
        + [("READ", s) for s in output_slots]
        + [("WRITE", s) for s in newer_slots]
        + [("WRITE", s) for s in older_slots]
    )


def truth_map(older_keys, newer_keys):
    """Newer-wins ground truth: key -> value string."""
    truth = {k: f"OLD:{k}" for k in older_keys}
    truth.update({k: f"NEW:{k}" for k in newer_keys})
    return truth


def records_as_pairs(records):
    return [(r.key.value, r.value) for r in records]


def physical_ids_reachable(obj, depth=0):
    if depth > 6:
        return []
    if isinstance(obj, (SlotId, BlockId)):
        return [obj]
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return []
    found = []
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            found += physical_ids_reachable(item, depth + 1)
        return found
    if isinstance(obj, dict):
        for item in obj.values():
            found += physical_ids_reachable(item, depth + 1)
        return found
    if hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            found += physical_ids_reachable(value, depth + 1)
    return found


# ---------------------------------------------------------------------------
# [1] upsert creates a fresh level with fresh blocks
# ---------------------------------------------------------------------------


def test_update_batch_creates_fresh_level_and_blocks():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    old_blocks = set(eng.level(older).block_ids)

    level_id = update_level(eng, range(4, 10), tag="NEW")

    assert level_id != older
    assert eng.is_active_level(level_id)
    assert len(eng.level(level_id).block_ids) == 2  # 6 records at capacity 3
    assert not (set(eng.level(level_id).block_ids) & old_blocks)
    assert records_as_pairs(flatten_level(eng, level_id)) == [
        (k, f"NEW:{k}") for k in range(4, 10)
    ]
    # the older level is untouched by the upsert
    assert records_as_pairs(flatten_level(eng, older)) == [
        (k, f"OLD:{k}") for k in range(6)
    ]
    assert eng.storage.size == len(list(eng.mapping.items()))


def test_update_level_is_not_queryable_until_pgm_is_built_explicitly():
    eng = build_engine(capacity=3, epsilon=2)
    level_id = eng.create_update_level([rec(1, "NEW"), rec(2, "NEW")])

    # no implicit PGM construction anywhere in G1-D either
    with pytest.raises(KeyError):
        eng.pgm_index(level_id)
    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.lookup_levels([level_id], RecordKey(1))
    assert len(eng.trace) == 0

    eng.build_level_pgm(level_id)
    result, _ = observe(eng, [level_id], 1)
    assert result.found is True and result.hit_level_id == level_id


def test_empty_update_batch_is_rejected_without_publishing():
    eng = build_engine(capacity=3, epsilon=2)
    build_pgm_level(eng, range(4), tag="OLD")
    before_levels = eng._next_level_id
    before_storage = eng.storage.size

    eng.trace.clear()
    with pytest.raises(EmptyUpdateBatchError):
        eng.create_update_level([])

    assert eng._next_level_id == before_levels  # no level was allocated
    assert eng.storage.size == before_storage   # no block was written
    assert len(eng.trace) == 0


# ---------------------------------------------------------------------------
# [2] duplicate keys inside one update batch
# ---------------------------------------------------------------------------


def test_duplicate_keys_in_update_batch_fail_without_publishing():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(4), tag="OLD")
    before_level_id_counter = eng._next_level_id
    before_blocks = eng._next_block_id
    before_storage = eng.storage.size
    before_mapping = list(eng.mapping.items())

    eng.trace.clear()
    with pytest.raises(DuplicateKeyError):
        eng.create_update_level([rec(5, "NEW"), rec(7, "NEW"), rec(5, "NEW")])

    assert eng._next_level_id == before_level_id_counter
    assert eng._next_block_id == before_blocks
    assert eng.storage.size == before_storage
    assert list(eng.mapping.items()) == before_mapping
    assert len(eng.trace) == 0
    assert eng.is_active_level(older)


# ---------------------------------------------------------------------------
# [3] overlapping key resolved by G1-C newest-first before merge
# ---------------------------------------------------------------------------


def test_overlapping_key_uses_g1c_newest_first_before_merge():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, range(4, 10), tag="NEW")

    hit, _ = observe(eng, [newer, older], 5)
    assert hit.found is True and hit.hit_level_id == newer
    assert hit.record is not None and hit.record.value == "NEW:5"

    # G1-C is unchanged: reversing the plan reverses precedence
    reversed_hit, _ = observe(eng, [older, newer], 5)
    assert reversed_hit.hit_level_id == older
    assert reversed_hit.record is not None and reversed_hit.record.value == "OLD:5"

    # key only in the older level still resolves through the newer level first
    only_old, _ = observe(eng, [newer, older], 1)
    assert only_old.hit_level_id == older
    assert only_old.searched_level_ids == (newer, older)


# ---------------------------------------------------------------------------
# [4] [5] merge semantics
# ---------------------------------------------------------------------------


def test_merge_of_disjoint_levels_is_sorted_union():
    eng = build_engine(capacity=4, epsilon=2)
    older = build_pgm_level(eng, [0, 2, 4, 6], tag="OLD")
    newer = update_level(eng, [1, 3, 5, 7], tag="NEW")
    newer_slots, older_slots = block_slots(eng, newer), block_slots(eng, older)

    result, events = merge_trace_events(eng, newer, older)

    assert events == expected_merge_trace(
        newer_slots, older_slots, block_slots(eng, result.output_level_id)
    )
    assert result.newer_record_count == 4
    assert result.older_record_count == 4
    assert result.output_record_count == 8
    assert result.discarded_older_record_count == 0
    assert records_as_pairs(flatten_level(eng, result.output_level_id)) == [
        (k, f"{'NEW' if k % 2 else 'OLD'}:{k}") for k in range(8)
    ]


def test_merge_of_overlapping_levels_keeps_newer_record_once():
    eng = build_engine(capacity=3, epsilon=2)
    older_keys = list(range(8))            # 0..7, 3 blocks
    newer_keys = list(range(5, 11))        # 5..10, 2 blocks; overlaps 5,6,7
    older = build_pgm_level(eng, older_keys, tag="OLD")
    newer = update_level(eng, newer_keys, tag="NEW")
    newer_slots, older_slots = block_slots(eng, newer), block_slots(eng, older)

    result, events = merge_trace_events(eng, newer, older)

    assert events == expected_merge_trace(
        newer_slots, older_slots, block_slots(eng, result.output_level_id)
    )
    assert result.discarded_older_record_count == 3  # keys 5, 6, 7
    assert result.output_record_count == len(set(older_keys) | set(newer_keys)) == 11
    pairs = records_as_pairs(flatten_level(eng, result.output_level_id))
    assert pairs == sorted(truth_map(older_keys, newer_keys).items())
    # exactly once per key, newer value kept for overlaps
    keys = [k for k, _ in pairs]
    assert keys == sorted(set(keys))
    assert dict(pairs)[5] == "NEW:5" and dict(pairs)[7] == "NEW:7"
    assert dict(pairs)[4] == "OLD:4" and dict(pairs)[8] == "NEW:8"


# ---------------------------------------------------------------------------
# [6] [7] canonical output, fresh block ids
# ---------------------------------------------------------------------------


def test_output_level_is_canonical_and_passes_validate_level():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(8), tag="OLD")
    newer = update_level(eng, range(6, 14), tag="NEW")
    input_blocks = len(eng.level(older).block_ids) + len(eng.level(newer).block_ids)

    result, events = merge_trace_events(eng, newer, older)
    output = result.output_level_id
    output_blocks = len(eng.level(output).block_ids)

    # no validation READ pass inside the merge trace: READs are exactly the input
    # phase (A) plus the output PGM pass (C)
    assert sum(1 for op, _ in events if op == "READ") == input_blocks + output_blocks

    # canonical structure, checked with the accepted validator in its own window
    eng.trace.clear()
    validate_level(eng, output)
    validation_reads = len(eng.trace)
    eng.trace.clear()

    blocks = [eng.read_block(b) for b in eng.level(output).block_ids]
    eng.trace.clear()
    assert all(not b.is_empty for b in blocks)
    assert all(b.is_full for b in blocks[:-1])
    assert 1 <= blocks[-1].size <= eng.config.block_capacity
    keys = [r.key.value for b in blocks for r in b.records]
    assert keys == sorted(set(keys))
    assert validation_reads == len(blocks)  # validator is trace-visible by design


def test_output_block_ids_are_fresh_and_never_reused():
    eng = build_engine(capacity=2, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [3, 7, 8], tag="NEW")
    input_blocks = set(eng.level(older).block_ids) | set(eng.level(newer).block_ids)

    result, _ = merge_trace_events(eng, newer, older)
    output_blocks = eng.level(result.output_level_id).block_ids

    assert not (set(output_blocks) & input_blocks)
    assert len(set(output_blocks)) == len(output_blocks)
    assert min(output_blocks) > max(input_blocks)


# ---------------------------------------------------------------------------
# [8] caller-supplied precedence
# ---------------------------------------------------------------------------


def test_caller_supplied_precedence_not_level_id_order():
    for newer_first in (True, False):
        eng = build_engine(capacity=3, epsilon=2)
        first = build_pgm_level(eng, [1, 2, 3], tag="FIRST")   # lower LevelId
        second = update_level(eng, [2, 3, 4], tag="SECOND")   # higher LevelId
        assert first.value < second.value

        newer, older = (second, first) if newer_first else (first, second)
        result, _ = merge_trace_events(eng, newer, older)
        values = dict(records_as_pairs(flatten_level(eng, result.output_level_id)))

        winner = "SECOND" if newer_first else "FIRST"
        assert values[2] == f"{winner}:2" and values[3] == f"{winner}:3"
        assert result.newer_level_id == newer and result.older_level_id == older
        assert result.discarded_older_record_count == 2


# ---------------------------------------------------------------------------
# [9] [10] invalid inputs fail before merge I/O
# ---------------------------------------------------------------------------


def test_same_level_as_both_inputs_is_rejected_before_io():
    eng = build_engine(capacity=3, epsilon=2)
    level_id = build_pgm_level(eng, range(5), tag="L")
    blocks_before = eng.level(level_id).block_ids
    mapping_before = list(eng.mapping.items())

    eng.trace.clear()
    with pytest.raises(MergeInputError):
        eng.merge_levels_blocking(level_id, level_id)

    assert len(eng.trace) == 0
    assert eng.is_active_level(level_id)
    assert eng.level(level_id).block_ids == blocks_before
    assert list(eng.mapping.items()) == mapping_before


def test_unknown_and_retired_inputs_fail_before_merge_io():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(4), tag="OLD")
    newer = update_level(eng, [3, 4, 5], tag="NEW")
    stranger = LevelId(9999)

    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.merge_levels_blocking(newer, stranger)
    with pytest.raises(KeyError):
        eng.merge_levels_blocking(stranger, older)
    assert len(eng.trace) == 0

    # a real merge, then the retired inputs must fail clearly (still no I/O)
    result, _ = merge_trace_events(eng, newer, older)
    output = result.output_level_id
    eng.trace.clear()
    with pytest.raises(RetiredLevelError):
        eng.merge_levels_blocking(output, newer)
    with pytest.raises(RetiredLevelError):
        eng.merge_levels_blocking(older, output)
    assert len(eng.trace) == 0
    assert eng.is_active_level(output)


# ---------------------------------------------------------------------------
# [11] [12] [13] [14] exact baseline trace composition
# ---------------------------------------------------------------------------


def test_exact_baseline_trace_composition():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(8), tag="OLD")      # 3 blocks
    newer = update_level(eng, range(6, 12), tag="NEW")     # 2 blocks
    newer_blocks = eng.level(newer).block_ids
    older_blocks = eng.level(older).block_ids
    newer_slots = [eng.mapping.lookup(b).value for b in newer_blocks]
    older_slots = [eng.mapping.lookup(b).value for b in older_blocks]

    result, events = merge_trace_events(eng, newer, older)
    output_slots = block_slots(eng, result.output_level_id)
    assert len(output_slots) == 4  # 12 disjoint records at capacity 3

    expected = expected_merge_trace(newer_slots, older_slots, output_slots)
    assert events == expected

    reads = [slot for op, slot in events if op == "READ"]
    writes = [slot for op, slot in events if op == "WRITE"]
    assert reads == newer_slots + older_slots + output_slots
    assert writes == output_slots + newer_slots + older_slots

    # A: input read phase uses LOGICAL block order, newer level first
    assert reads[: len(newer_slots)] == newer_slots
    assert reads[len(newer_slots) : len(newer_slots) + len(older_slots)] == older_slots
    # B: exactly one WRITE per output block, construction order
    assert writes[: len(output_slots)] == output_slots
    # C: exactly one READ per output block for the PGM pass, logical order
    assert reads[len(newer_slots) + len(older_slots) :] == output_slots
    # D: exactly one WRITE per retired input block, newer level first
    assert writes[len(output_slots) :] == newer_slots + older_slots


def test_output_pgm_uses_the_accepted_g1b1_builder():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(9), tag="OLD")
    newer = update_level(eng, [7, 8, 9, 10], tag="NEW")

    result, _ = merge_trace_events(eng, newer, older)
    output_records = flatten_level(eng, result.output_level_id)
    keys = [r.key for r in output_records]

    # the output PGM equals what the accepted batch builder produces for those keys
    reference = build_batch_pgm(keys, eng.config.pgm_epsilon)
    assert eng.pgm_index(result.output_level_id) == reference
    assert reference.record_count == len(keys)


def test_retirement_writes_happen_only_after_publication():
    eng = build_engine(capacity=2, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [4, 5, 6], tag="NEW")
    newer_slots, older_slots = block_slots(eng, newer), block_slots(eng, older)

    result, events = merge_trace_events(eng, newer, older)
    output_slots = block_slots(eng, result.output_level_id)

    first_retirement = events.index(("WRITE", newer_slots[0]))
    last_pgm_read = max(
        i for i, (op, slot) in enumerate(events) if op == "READ" and slot in output_slots
    )
    last_output_write = max(
        i for i, (op, slot) in enumerate(events) if op == "WRITE" and slot in output_slots
    )
    assert last_pgm_read < first_retirement
    assert last_output_write < first_retirement
    # retirement order: newer level's blocks then older level's blocks
    assert [s for op, s in events if op == "WRITE" and s in newer_slots + older_slots] == (
        newer_slots + older_slots
    )
    assert eng.storage.size == len(output_slots)


# ---------------------------------------------------------------------------
# [15] [16] retirement state
# ---------------------------------------------------------------------------


def test_successful_merge_retires_inputs_and_drops_pgm_metadata():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [5, 6, 7], tag="NEW")
    input_blocks = list(eng.level(newer).block_ids) + list(eng.level(older).block_ids)

    result, _ = merge_trace_events(eng, newer, older)

    for level_id in (newer, older):
        assert eng.is_active_level(level_id) is False
        with pytest.raises(RetiredLevelError):
            eng.level(level_id)
        with pytest.raises(RetiredLevelError):
            eng.pgm_index(level_id)
        with pytest.raises(RetiredLevelError):
            eng.lookup_levels([level_id], RecordKey(1))
    assert eng.is_active_level(result.output_level_id)


def test_mapping_bijection_survives_merge_and_retired_blocks_are_gone():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [5, 6, 7], tag="NEW")
    input_blocks = list(eng.level(newer).block_ids) + list(eng.level(older).block_ids)

    result, _ = merge_trace_events(eng, newer, older)
    output_blocks = eng.level(result.output_level_id).block_ids

    eng.mapping.validate_bijection()
    mapped = {block_id for block_id, _slot in eng.mapping.items()}
    assert mapped == set(output_blocks)
    for block_id in input_blocks:
        assert block_id not in mapped
        with pytest.raises(MissingLogicalBlockError):
            eng.mapping.lookup(block_id)
    assert eng.storage.size == len(output_blocks)
    assert sorted(eng.storage.occupied_slots()) == sorted(
        eng.mapping.lookup(b) for b in output_blocks
    )


# ---------------------------------------------------------------------------
# [17] [18] query correctness after merge
# ---------------------------------------------------------------------------


def test_output_remains_queryable_through_g1c():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [4, 5, 6, 7, 8], tag="NEW")
    truth = truth_map(list(range(6)), [4, 5, 6, 7, 8])

    result, _ = merge_trace_events(eng, newer, older)

    for key in sorted(truth) + [3, 9, 99, -1]:
        hit, _ = observe(eng, [result.output_level_id], key)
        if key in truth:
            assert hit.found is True and hit.hit_level_id == result.output_level_id
            assert hit.record is not None and hit.record.value == truth[key]
        else:
            assert hit.found is False and hit.record is None


def test_pre_merge_and_post_merge_results_are_logically_equivalent():
    eng = build_engine(capacity=2, epsilon=2)
    older_keys = list(range(10))
    newer_keys = list(range(7, 15))
    older = build_pgm_level(eng, older_keys, tag="OLD")
    newer = update_level(eng, newer_keys, tag="NEW")
    truth = truth_map(older_keys, newer_keys)

    before = {}
    for key in sorted(set(list(truth) + [-1, 6, 15, 100])):
        before[key] = observe(eng, [newer, older], key)[0]

    result, _ = merge_trace_events(eng, newer, older)
    output = result.output_level_id

    for key, prior in before.items():
        after, _ = observe(eng, [output], key)
        assert after.found == prior.found
        assert after.record == prior.record
        if key in truth:
            assert after.record.value == truth[key]
        else:
            assert after.record is None
    # no query needs the retired levels any more
    assert eng.is_active_level(newer) is False
    assert eng.is_active_level(older) is False


# ---------------------------------------------------------------------------
# [19] failure atomicity boundary
# ---------------------------------------------------------------------------


def test_failure_in_output_pgm_construction_leaves_inputs_active(monkeypatch):
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [5, 6, 7], tag="NEW")
    next_level_id = LevelId(eng._next_level_id)
    mapping_before = sorted(list(eng.mapping.items()))
    input_slots = block_slots(eng, newer) + block_slots(eng, older)

    def explode(self, level_id):
        raise RuntimeError("injected PGM construction failure")

    monkeypatch.setattr(TrustedEngine, "build_level_pgm", explode)

    eng.trace.clear()
    with pytest.raises(RuntimeError):
        eng.merge_levels_blocking(newer, older)

    # both inputs are still active, still mapped, still queryable
    assert eng.is_active_level(newer) and eng.is_active_level(older)
    assert eng.is_active_level(next_level_id) is False  # no output advertised
    assert sorted(list(eng.mapping.items())) == mapping_before
    assert all(eng.storage.contains(SlotId(s)) for s in input_slots)
    assert eng.storage.size == len(input_slots)

    monkeypatch.undo()
    hit, _ = observe(eng, [newer, older], 6)
    assert hit.found is True and hit.hit_level_id == newer
    assert hit.record is not None and hit.record.value == "NEW:6"


def test_unconfigured_epsilon_fails_merge_without_retiring_inputs():
    eng = TrustedEngine(Config(block_capacity=3, pgm_epsilon=None))
    older = eng.build_level([rec(k, "OLD") for k in range(4)])
    newer = eng.create_update_level([rec(k, "NEW") for k in range(3, 6)])
    next_level_id = LevelId(eng._next_level_id)
    mapping_before = sorted(list(eng.mapping.items()))

    eng.trace.clear()
    with pytest.raises(PgmConfigurationError):
        eng.merge_levels_blocking(newer, older)

    assert eng.is_active_level(newer) and eng.is_active_level(older)
    assert eng.is_active_level(next_level_id) is False
    assert sorted(list(eng.mapping.items())) == mapping_before


def test_non_canonically_packed_input_is_rejected():
    """Ordered but non-canonically packed input (partial non-final block) fails.

    Issue #5 freezes that merge inputs must be canonical levels; decision 0006 §4
    requires the packing invariant to be checked during the input READ pass with
    zero extra I/O.
    """
    eng = build_engine(capacity=4, epsilon=2)
    good = build_pgm_level(eng, [10, 11, 12, 13], tag="GOOD")  # canonical: 1 full block

    # hand-assembled level: keys ordered, but block 0 is non-final AND partial
    #   block 0 = [0,1]      -> NON-CANONICAL (partial non-final block)
    #   block 1 = [2,3,4,5]  -> final, full
    bad = eng.create_level()
    eng.create_block([rec(0, "BAD"), rec(1, "BAD")], level_id=bad)
    eng.create_block(
        [rec(2, "BAD"), rec(3, "BAD"), rec(4, "BAD"), rec(5, "BAD")], level_id=bad
    )
    eng.build_level_pgm(bad)  # ordered keys still permit a PGM; packing is illegal

    good_slots = block_slots(eng, good)
    storage_before = eng.storage.size
    mapping_before = sorted(eng.mapping.items())
    reserved = LevelId(eng._next_level_id)

    for newer, older, label in ((good, bad, "bad-as-older"), (bad, good, "bad-as-newer")):
        eng.trace.clear()
        with pytest.raises(MergeInputError):
            eng.merge_levels_blocking(newer, older)
        events = [(e.operation.value, e.slot_id.value) for e in eng.trace.events()]
        # validation happens inside the input READ pass: READs only, so no output
        # materialization WRITE and no retirement WRITE ever occurs
        assert all(op == "READ" for op, _slot in events), (label, events)
        assert not any(op == "WRITE" for op, _slot in events), (label, events)
        assert not any(slot in good_slots for _op, slot in events if _op == "WRITE")

    # nothing was published, nothing was retired, engine state is untouched
    assert eng.is_active_level(good) is True
    assert eng.is_active_level(bad) is True          # stays active for repair
    assert eng.is_active_level(reserved) is False
    assert sorted(eng.mapping.items()) == mapping_before
    assert eng.storage.size == storage_before
    eng.mapping.validate_bijection()

    # the valid input remains usable and the malformed one is still readable
    good_hit, _ = observe(eng, [good], 11)
    assert good_hit.found is True and good_hit.record.value == "GOOD:11"
    bad_hit, _ = observe(eng, [bad], 3)
    assert bad_hit.found is True and bad_hit.record.value == "BAD:3"


def test_materialization_failure_discards_reserved_output(monkeypatch):
    """A mid-materialization exception must not leave a partial output active."""
    eng = build_engine(capacity=2, epsilon=2)
    older = build_pgm_level(eng, [0, 1], tag="OLD")      # 1 block
    newer = update_level(eng, [1, 2, 3], tag="NEW")      # 2 blocks
    # merged = 0:OLD, 1:NEW, 2:NEW, 3:NEW -> 2 output blocks at capacity 2
    input_slots = block_slots(eng, newer) + block_slots(eng, older)
    mapping_before = sorted(eng.mapping.items())
    storage_before = eng.storage.size
    reserved = LevelId(eng._next_level_id)

    original_create_block = TrustedEngine.create_block
    calls = {"n": 0}

    def flaky(self, records, level_id=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure on the second output block")
        return original_create_block(self, records, level_id=level_id)

    monkeypatch.setattr(TrustedEngine, "create_block", flaky)

    eng.trace.clear()
    with pytest.raises(RuntimeError):
        eng.merge_levels_blocking(newer, older)
    events = [(e.operation.value, e.slot_id.value) for e in eng.trace.events()]

    assert calls["n"] == 2  # the failure really happened inside materialization

    # no input retirement WRITE
    assert not any(op == "WRITE" and slot in input_slots for op, slot in events)
    # both inputs stay active and usable
    assert eng.is_active_level(newer) is True
    assert eng.is_active_level(older) is True
    # the reserved output is no longer an active level and has no PGM
    assert eng.is_active_level(reserved) is False
    with pytest.raises(KeyError):
        eng.pgm_index(reserved)
    # normally cleanable mapping/storage state is cleaned
    assert sorted(eng.mapping.items()) == mapping_before
    assert eng.storage.size == storage_before
    eng.mapping.validate_bijection()

    monkeypatch.undo()
    hit, _ = observe(eng, [newer, older], 2)
    assert hit.found is True and hit.hit_level_id == newer
    assert hit.record is not None and hit.record.value == "NEW:2"


# ---------------------------------------------------------------------------
# [20] result contract
# ---------------------------------------------------------------------------


def test_merge_result_exposes_no_physical_internals():
    eng = build_engine(capacity=3, epsilon=2)
    older = build_pgm_level(eng, range(6), tag="OLD")
    newer = update_level(eng, [4, 5, 6], tag="NEW")

    result, _ = merge_trace_events(eng, newer, older)

    assert isinstance(result, MergeResult)
    assert set(vars(result)) == {
        "output_level_id",
        "newer_level_id",
        "older_level_id",
        "newer_record_count",
        "older_record_count",
        "output_record_count",
        "discarded_older_record_count",
    }
    assert physical_ids_reachable(result) == []
    for forbidden in ("slot", "slot_id", "physical_schedule", "block_ids", "trace"):
        assert not hasattr(result, forbidden)


# ---------------------------------------------------------------------------
# [21] randomized merge oracle
# ---------------------------------------------------------------------------


def randomized_merge_oracle(seed):
    rng = random.Random(seed)
    for _ in range(6):
        capacity = rng.choice([1, 2, 3, 4, 5])
        eps = rng.choice([0, 1, 2, 3, 4])
        eng = build_engine(capacity=capacity, epsilon=eps)

        older_keys = sorted(rng.sample(range(0, 60), rng.randint(0, 20)))
        older = build_pgm_level(eng, older_keys, tag="OLD")
        newer_keys = sorted(rng.sample(range(0, 60), rng.randint(1, 15)))
        newer = update_level(eng, newer_keys, tag="NEW")
        truth = truth_map(older_keys, newer_keys)

        newer_slots = block_slots(eng, newer)
        older_slots = block_slots(eng, older)

        # pre-merge oracle through G1-C
        queries = sorted(set(list(truth) + [k - 1 for k in truth] + [-1, 60, 61]))
        for key in queries:
            result, _ = observe(eng, [newer, older], key)
            if key in truth:
                assert result.found is True
                assert result.record is not None and result.record.value == truth[key]
            else:
                assert result.found is False and result.record is None

        merge_result, events = merge_trace_events(eng, newer, older)
        output = merge_result.output_level_id
        output_slots = block_slots(eng, output)

        # exact frozen trace composition
        assert events == expected_merge_trace(newer_slots, older_slots, output_slots)
        assert len(output_slots) == (
            0 if not truth else (len(truth) + capacity - 1) // capacity
        )

        # functional merge result
        assert merge_result.newer_record_count == len(newer_keys)
        assert merge_result.older_record_count == len(older_keys)
        assert merge_result.output_record_count == len(truth)
        assert merge_result.discarded_older_record_count == len(
            set(older_keys) & set(newer_keys)
        )
        assert records_as_pairs(flatten_level(eng, output)) == sorted(truth.items())

        # canonical output, checked in its own trace window
        eng.trace.clear()
        validate_level(eng, output)
        eng.trace.clear()

        # post-merge equivalence and retirement state
        for key in queries:
            after, _ = observe(eng, [output], key)
            if key in truth:
                assert after.found is True
                assert after.record is not None and after.record.value == truth[key]
            else:
                assert after.found is False and after.record is None
        for level_id in (newer, older):
            assert eng.is_active_level(level_id) is False
        eng.mapping.validate_bijection()
        assert eng.storage.size == len(output_slots)


def test_randomized_merge_oracle_many_seeds():
    for seed in range(5):
        randomized_merge_oracle(9000 + seed)


def test_randomized_sequential_merges_keep_state_consistent():
    """Chained merges: output of one merge can be an input of the next."""
    for seed in range(3):
        rng = random.Random(5000 + seed)
        capacity = rng.choice([2, 3, 4])
        eng = build_engine(capacity=capacity, epsilon=2)
        truth = {}
        live = None

        for step in range(4):
            keys = sorted(rng.sample(range(0, 50), rng.randint(1, 8)))
            new_level = update_level(eng, keys, tag=f"L{step}")
            truth.update({k: f"L{step}:{k}" for k in keys})
            if live is None:
                live = new_level
                continue

            result, _ = merge_trace_events(eng, new_level, live)
            live = result.output_level_id
            assert records_as_pairs(flatten_level(eng, live)) == sorted(truth.items())

        for key in sorted(truth) + [-1, 100]:
            hit, _ = observe(eng, [live], key)
            if key in truth:
                assert hit.found is True
                assert hit.record is not None and hit.record.value == truth[key]
            else:
                assert hit.found is False
        eng.mapping.validate_bijection()
        assert eng.storage.size == len(eng.level(live).block_ids)


def test_merge_of_empty_older_level_is_allowed_and_canonical():
    eng = build_engine(capacity=3, epsilon=2)
    older = eng.build_level([])          # empty level: no blocks
    eng.build_level_pgm(older)
    newer = update_level(eng, [1, 2, 3], tag="NEW")
    newer_slots = block_slots(eng, newer)

    result, events = merge_trace_events(eng, newer, older)

    assert result.older_record_count == 0
    assert result.output_record_count == 3
    assert events == expected_merge_trace(
        newer_slots, [], block_slots(eng, result.output_level_id)
    )
    assert records_as_pairs(flatten_level(eng, result.output_level_id)) == [
        (k, f"NEW:{k}") for k in (1, 2, 3)
    ]
    assert eng.is_active_level(older) is False
