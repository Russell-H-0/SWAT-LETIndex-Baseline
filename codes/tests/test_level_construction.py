"""G1-A level construction tests: dataset -> sorted blocks -> logical level."""

import pytest

from enhanced_letindex.builder import LevelValidationError, validate_level
from enhanced_letindex.config import Config
from enhanced_letindex.engine import TrustedEngine
from enhanced_letindex.identifiers import BlockId, RecordKey, SlotId
from enhanced_letindex.record import Record
from enhanced_letindex.storage import StorageError
from enhanced_letindex.trace import TraceOperation

EXAMPLE_KEYS = (50, 10, 70, 20, 40, 30, 60)  # spec example, capacity 3


def rec(k):
    return Record(RecordKey(k), f"v{k}")


def records_from(keys):
    return [rec(k) for k in keys]


def sorted_seven():
    return sorted(EXAMPLE_KEYS)


def test_empty_input_creates_empty_level():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level([])

    level = eng.level(level_id)
    assert level.size == 0
    assert list(level.block_ids) == []
    assert eng.storage.size == 0
    # no dummy blocks and no storage writes for an empty dataset
    assert len(eng.trace) == 0


def test_logical_block_ids_are_unique_within_a_level():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    block_ids = list(eng.level(level_id).block_ids)
    assert len(block_ids) == 3
    assert len(set(block_ids)) == len(block_ids)


def test_two_levels_do_not_reuse_active_block_ids():
    eng = TrustedEngine(Config(block_capacity=3))
    a = eng.build_level(records_from((10, 20, 30, 40)))
    b = eng.build_level(records_from((100, 110, 120)))

    ida = set(eng.level(a).block_ids)
    idb = set(eng.level(b).block_ids)
    assert ida and idb
    assert ida.isdisjoint(idb)


def test_block_ids_distinct_from_slot_ids_conceptually():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    level = eng.level(level_id)

    for block_id in level.block_ids:
        slot = eng.physical_slot(block_id)
        # distinct namespaces: never equal even when the integers coincide
        assert block_id != slot
        assert not isinstance(slot, BlockId)
        assert not isinstance(block_id, SlotId)


def test_physical_slots_preserve_vanilla_logical_order():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    level = eng.level(level_id)

    # logical order is sorted-key order
    logical = [eng.read_block(b).records for b in level.block_ids]
    assert [r.key.value for r in logical[0]] == [10, 20, 30]
    assert [r.key.value for r in logical[1]] == [40, 50, 60]
    assert [r.key.value for r in logical[2]] == [70]

    # deterministic, order-preserving physical placement: consecutive slots
    slots = [eng.physical_slot(b) for b in level.block_ids]
    assert slots == [SlotId(0), SlotId(1), SlotId(2)]


def test_block_key_ranges_strictly_ordered_and_non_overlapping():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    level = eng.level(level_id)

    blocks = [eng.read_block(b) for b in level.block_ids]
    for left, right in zip(blocks, blocks[1:]):
        assert left.max_key < right.min_key
    assert blocks[0].max_key == RecordKey(30)
    assert blocks[1].min_key == RecordKey(40)


def test_construction_produces_one_write_per_materialized_block():
    eng = TrustedEngine(Config(block_capacity=3))
    eng.build_level(records_from(EXAMPLE_KEYS))  # 7 records -> 3 blocks

    events = eng.trace.events()
    writes = [e for e in events if e.operation is TraceOperation.WRITE]
    reads = [e for e in events if e.operation is TraceOperation.READ]
    assert len(writes) == 3
    assert len(reads) == 0  # pure construction only writes


def test_trace_does_not_expose_key_block_id_or_contents():
    eng = TrustedEngine(Config(block_capacity=3))
    eng.build_level(records_from(EXAMPLE_KEYS))

    for ev in eng.trace.events():
        data = ev.as_dict()
        assert data["operation"] == "WRITE"
        assert isinstance(data["slot_id"], int)
        assert set(data) == {"seq", "operation", "slot_id", "level_id", "metadata"}
        assert data["level_id"] is None
        assert data["metadata"] is None
        assert "block_id" not in data
        assert "key" not in data
        assert "min_key" not in data
        assert "max_key" not in data


def test_validate_level_accepts_constructed_level():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    validate_level(eng, level_id)  # must not raise


def test_validate_level_accepts_exact_multiple_of_capacity():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from((1, 2, 3, 4, 5, 6)))
    validate_level(eng, level_id)


def test_validate_level_accepts_empty_level():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level([])
    validate_level(eng, level_id)


def test_validate_rejects_duplicate_logical_block_ids():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    level = eng.level(level_id)
    level._block_ids.append(level.block_ids[0])  # corrupt: duplicate id
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)


def test_validate_rejects_non_full_intermediate_block():
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))  # B0 full, B1 full, B2 partial
    level = eng.level(level_id)
    # corrupt B0 by removing one record so it is no longer full but non-final
    b0 = eng.read_block(level.block_ids[0])
    b0._records.pop()
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)


# ---------------------------------------------------------------------------
# G1-A.1 hardening: negative coverage for the remaining validate_level branches


def _build_example():
    """A fresh engine with [10,20,30 | 40,50,60 | 70] at capacity 3."""
    eng = TrustedEngine(Config(block_capacity=3))
    level_id = eng.build_level(records_from(EXAMPLE_KEYS))
    return eng, level_id


def test_validate_rejects_missing_mapping_reference():
    eng, level_id = _build_example()
    level = eng.level(level_id)
    b0 = level.block_ids[0]
    slot = eng.physical_slot(b0)
    # white-box: drop the mapping so b0 has no physical reference
    eng.mapping._logical_to_physical.pop(b0)
    eng.mapping._physical_to_logical.pop(slot)
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)


def test_validate_rejects_missing_storage_reference():
    eng, level_id = _build_example()
    level = eng.level(level_id)
    b0 = level.block_ids[0]
    slot = eng.physical_slot(b0)
    # white-box: vacate the physical slot while the mapping still references it
    eng.storage._slots.pop(slot)
    # the invariant checker surfaces the missing storage through the traced
    # read path, which fails loudly rather than fabricating a block
    with pytest.raises(StorageError):
        validate_level(eng, level_id)


def test_validate_rejects_non_increasing_keys_in_referenced_block():
    eng, level_id = _build_example()
    level = eng.level(level_id)
    b0 = eng.read_block(level.block_ids[0])  # [10, 20, 30]
    b0._records[1] = rec(10)  # -> [10, 10, 30]: equal adjacent keys
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)


def test_validate_rejects_overlapping_neighbor_ranges():
    eng, level_id = _build_example()
    level = eng.level(level_id)
    b1 = eng.read_block(level.block_ids[1])  # [40, 50, 60]
    b1._records[0] = rec(30)  # -> [30, 50, 60]: min touches B0's max of 30
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)


def test_validate_rejects_final_block_over_capacity():
    eng, level_id = _build_example()
    level = eng.level(level_id)
    final = eng.read_block(level.block_ids[2])  # [70], capacity 3
    final._records.extend([rec(80), rec(90), rec(100)])  # size 4 > capacity 3
    with pytest.raises(LevelValidationError):
        validate_level(eng, level_id)
