"""End-to-end TrustedEngine tests for the G0 vertical slice."""

import pytest

from enhanced_letindex.block import Block
from enhanced_letindex.config import Config
from enhanced_letindex.engine import TrustedEngine
from enhanced_letindex.identifiers import BlockId, RecordKey, SlotId
from enhanced_letindex.mapping import DuplicatePhysicalSlotError
from enhanced_letindex.record import Record
from enhanced_letindex.storage import StorageConsistencyError
from enhanced_letindex.trace import TraceOperation


def make_records(n, offset=0):
    return [Record(RecordKey(i), f"value-{i}") for i in range(offset, offset + n)]


def test_engine_retrieves_block_through_mapping_layer():
    eng = TrustedEngine(Config(block_capacity=16))
    records = make_records(4)
    bid = eng.create_block(records)

    eng.trace.clear()
    block = eng.read_block(bid)

    assert block.block_id == bid
    assert [r.key for r in block.records] == [
        RecordKey(0),
        RecordKey(1),
        RecordKey(2),
        RecordKey(3),
    ]
    # exactly one observable trace event: a READ of the block's physical slot
    assert len(eng.trace) == 1
    ev = eng.trace.events()[0]
    assert ev.operation.value == "READ"
    assert ev.slot_id == eng.physical_slot(bid)


def test_relocation_preserves_logical_identity_and_changes_trace():
    eng = TrustedEngine(Config(block_capacity=16))
    records = make_records(4)
    bid = eng.create_block(records)  # first block -> physical slot 0
    assert eng.physical_slot(bid) == SlotId(0)

    # read through mapping: trace exposes slot 0
    eng.trace.clear()
    before = eng.read_block(bid)
    assert eng.trace.events()[0].slot_id == SlotId(0)

    # deterministic remap: logical block -> physical slot 5
    eng.relocate_block(bid, SlotId(5))
    assert eng.physical_slot(bid) == SlotId(5)

    # logical contents identical after relocation
    eng.trace.clear()
    after = eng.read_block(bid)
    assert [r.key for r in after.records] == [r.key for r in before.records]
    assert after.block_id == bid  # same logical identity, new physical slot

    # trace now exposes slot 5, not slot 0
    ev = eng.trace.events()[0]
    assert ev.operation.value == "READ"
    assert ev.slot_id == SlotId(5)


def test_relocate_to_occupied_slot_is_rejected_without_corruption():
    eng = TrustedEngine(Config(block_capacity=16))
    b1 = eng.create_block(make_records(2))      # slot 0
    b2 = eng.create_block(make_records(1, 10))  # slot 1

    with pytest.raises(DuplicatePhysicalSlotError):
        eng.relocate_block(b1, SlotId(1))

    assert eng.physical_slot(b1) == SlotId(0)
    assert [r.key for r in eng.read_block(b1).records] == [
        RecordKey(0),
        RecordKey(1),
    ]


def test_levels_hold_logical_block_ids_only():
    eng = TrustedEngine(Config())
    level_id = eng.create_level()
    bid = eng.create_block(make_records(2), level_id=level_id)

    level = eng.level(level_id)
    assert list(level.block_ids) == [bid]
    assert level.size == 1
    assert bid in level


def test_relocation_vacates_source_slot_and_preserves_contents():
    eng = TrustedEngine(Config(block_capacity=16))
    bid = eng.create_block(make_records(4))  # B0 -> P0

    # before: mapping and storage agree the block lives at P0
    assert eng.physical_slot(bid) == SlotId(0)
    assert eng.storage.contains(SlotId(0))
    assert eng.storage.read(SlotId(0)).block_id == bid
    occupied_before = eng.storage.size

    eng.relocate_block(bid, SlotId(5))

    # after: block is at P5 and P0 is vacant
    assert eng.physical_slot(bid) == SlotId(5)
    assert eng.storage.contains(SlotId(5))
    assert eng.storage.read(SlotId(5)).block_id == bid
    assert not eng.storage.contains(SlotId(0))
    assert SlotId(0) not in eng.storage.occupied_slots()
    assert eng.storage.size == occupied_before == 1

    # logical retrieval through the engine still returns identical contents
    after = eng.read_block(bid)
    assert after.block_id == bid
    assert [r.key for r in after.records] == [
        RecordKey(0),
        RecordKey(1),
        RecordKey(2),
        RecordKey(3),
    ]


def test_relocation_emits_expected_trace():
    eng = TrustedEngine(Config(block_capacity=16))
    bid = eng.create_block(make_records(4))  # B0 -> P0
    eng.trace.clear()

    eng.relocate_block(bid, SlotId(5))

    events = eng.trace.events()
    assert [(e.operation, e.slot_id) for e in events] == [
        (TraceOperation.READ, SlotId(0)),
        (TraceOperation.WRITE, SlotId(5)),
        (TraceOperation.WRITE, SlotId(0)),
    ]
    # the trace must expose only physical slot ids, never logical metadata
    for ev in events:
        data = ev.as_dict()
        assert "block_id" not in data
        assert "key" not in data
        assert data["level_id"] is None
        assert data["metadata"] is None


def test_vacated_slot_can_subsequently_be_reused():
    eng = TrustedEngine(Config(block_capacity=16))
    bid = eng.create_block(make_records(2))  # B0 -> P0
    eng.relocate_block(bid, SlotId(5))

    assert not eng.storage.contains(SlotId(0))

    replacement = Block.sorted_block(BlockId(100), make_records(1, 50), 8)
    eng.storage.write(SlotId(0), replacement)

    assert eng.storage.contains(SlotId(0))
    assert eng.storage.read(SlotId(0)).block_id == BlockId(100)


def test_relocation_refuses_destination_occupied_in_storage():
    """Destination must be free in storage too, not only in the mapping."""
    eng = TrustedEngine(Config(block_capacity=16))
    bid = eng.create_block(make_records(2))  # B0 -> P0
    stray = Block.sorted_block(BlockId(200), make_records(1, 99), 8)
    eng.storage.write(SlotId(5), stray)  # storage occupied, mapping says free

    with pytest.raises(StorageConsistencyError):
        eng.relocate_block(bid, SlotId(5))

    # nothing was moved, corrupted, or silently repaired
    assert eng.physical_slot(bid) == SlotId(0)
    assert eng.storage.contains(SlotId(0))
    assert eng.storage.read(SlotId(5)).block_id == BlockId(200)


def test_relocation_refuses_source_storage_mismatch():
    """Mapping says B0 -> P0, but P0 holds a different block: fail loudly."""
    eng = TrustedEngine(Config(block_capacity=16))
    bid = eng.create_block(make_records(2))  # B0 -> P0
    wrong = Block.sorted_block(BlockId(300), make_records(1, 7), 8)
    eng.storage.write(SlotId(0), wrong)  # inconsistent mapping/storage state

    with pytest.raises(StorageConsistencyError):
        eng.relocate_block(bid, SlotId(5))

    assert eng.physical_slot(bid) == SlotId(0)
