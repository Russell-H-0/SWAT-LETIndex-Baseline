"""Tests for UntrustedStorage and its trace recording."""

import pytest

from enhanced_letindex.block import Block
from enhanced_letindex.identifiers import BlockId, RecordKey, SlotId
from enhanced_letindex.record import Record
from enhanced_letindex.storage import StorageError, UntrustedStorage
from enhanced_letindex.trace import TraceOperation


def make_block(bid=BlockId(7)):
    return Block.sorted_block(
        bid,
        [Record(RecordKey(1), "a"), Record(RecordKey(2), "b")],
        capacity=8,
    )


def test_read_produces_exactly_one_trace_event():
    s = UntrustedStorage()
    slot = s.store(make_block())
    s.trace.clear()

    block = s.read(slot)

    assert block.block_id == BlockId(7)
    assert len(s.trace) == 1
    ev = s.trace.events()[0]
    assert ev.operation is TraceOperation.READ
    assert ev.slot_id == slot
    assert ev.level_id is None
    assert ev.metadata is None


def test_write_produces_exactly_one_trace_event():
    s = UntrustedStorage()
    s.trace.clear()

    s.write(SlotId(3), make_block())

    assert len(s.trace) == 1
    ev = s.trace.events()[0]
    assert ev.operation is TraceOperation.WRITE
    assert ev.slot_id == SlotId(3)


def test_trace_exposes_physical_slot_not_logical_block_id():
    s = UntrustedStorage()
    slot = s.store(make_block(BlockId(99)))
    s.trace.clear()

    s.read(slot)

    ev = s.trace.events()[0]
    assert ev.slot_id == slot
    data = ev.as_dict()
    assert data["slot_id"] == slot.value
    assert "block_id" not in data
    assert "key" not in data


def test_read_empty_slot_raises():
    s = UntrustedStorage()
    with pytest.raises(StorageError):
        s.read(SlotId(42))


def test_allocate_is_sequential_and_unique():
    s = UntrustedStorage()
    slots = [s.allocate() for _ in range(3)]
    assert slots == [SlotId(0), SlotId(1), SlotId(2)]


def test_store_returns_and_occupies_a_fresh_slot():
    s = UntrustedStorage()
    slot = s.store(make_block())
    assert s.contains(slot)
    assert s.size == 1


def test_trace_clear_and_serialize():
    s = UntrustedStorage()
    s.store(make_block())
    s.read(SlotId(0))

    assert len(s.trace) == 2
    dicts = s.trace.as_dicts()
    assert [d["operation"] for d in dicts] == ["WRITE", "READ"]

    s.trace.clear()
    assert len(s.trace) == 0


def test_clear_vacates_slot_and_records_write():
    s = UntrustedStorage()
    slot = s.store(make_block())
    s.trace.clear()

    s.clear(slot)

    assert not s.contains(slot)
    assert s.size == 0
    events = s.trace.events()
    assert len(events) == 1
    assert events[0].operation is TraceOperation.WRITE
    assert events[0].slot_id == slot


def test_clear_empty_slot_raises():
    s = UntrustedStorage()
    with pytest.raises(StorageError):
        s.clear(SlotId(7))
