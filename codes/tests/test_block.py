"""Tests for Block invariants."""

import pytest

from enhanced_letindex.block import Block, BlockFullError, BlockInvariantError
from enhanced_letindex.identifiers import BlockId, RecordKey
from enhanced_letindex.record import Record

BLOCK_ID = BlockId(7)


def rec(k):
    return Record(RecordKey(k), f"v{k}")


def test_records_are_sorted_on_construction():
    b = Block.sorted_block(BLOCK_ID, [rec(5), rec(1), rec(3)], capacity=8)
    assert [r.key.value for r in b.records] == [1, 3, 5]


def test_strict_ascending_keys_are_valid():
    b = Block.sorted_block(BLOCK_ID, [rec(10), rec(20), rec(30)], capacity=8)
    assert [r.key.value for r in b.records] == [10, 20, 30]


def test_duplicate_keys_are_rejected_on_sorted_construction():
    # Duplicate keys are invalid at the Block level, not only in Dataset.
    with pytest.raises(BlockInvariantError):
        Block.sorted_block(BLOCK_ID, [rec(10), rec(20), rec(20), rec(30)], capacity=8)


def test_duplicate_keys_are_rejected_on_direct_construction():
    # Direct construction goes through the same invariant check.
    with pytest.raises(BlockInvariantError):
        Block(BLOCK_ID, capacity=8, _records=[rec(10), rec(20), rec(20), rec(30)])


def test_descending_input_is_sorted_by_sorted_block_constructor():
    # The designated sorted-block constructor normalizes descending input.
    b = Block.sorted_block(BLOCK_ID, [rec(30), rec(20), rec(10)], capacity=8)
    assert [r.key.value for r in b.records] == [10, 20, 30]


def test_add_record_rejects_duplicate_key():
    b = Block.sorted_block(BLOCK_ID, [rec(10), rec(20), rec(30)], capacity=8)
    with pytest.raises(BlockInvariantError):
        b.add_record(rec(20))  # duplicate of an existing key


def test_add_record_maintains_sort_order():
    b = Block.sorted_block(BLOCK_ID, [rec(2), rec(4)], capacity=8)
    b.add_record(rec(1))
    b.add_record(rec(3))
    assert [r.key.value for r in b.records] == [1, 2, 3, 4]


def test_capacity_is_enforced_on_construction():
    with pytest.raises(BlockInvariantError):
        Block.sorted_block(BLOCK_ID, [rec(1), rec(2), rec(3)], capacity=2)


def test_capacity_is_enforced_by_add_record():
    b = Block(BLOCK_ID, capacity=2)
    b.add_record(rec(1))
    b.add_record(rec(2))
    with pytest.raises(BlockFullError):
        b.add_record(rec(3))
    assert b.size == 2  # failed add did not leak a record in


def test_min_max_keys():
    empty = Block(BLOCK_ID, capacity=4)
    assert empty.min_key is None
    assert empty.max_key is None

    b = Block.sorted_block(BLOCK_ID, [rec(7), rec(2)], capacity=4)
    assert b.min_key == RecordKey(2)
    assert b.max_key == RecordKey(7)


def test_size_and_full_flags():
    b = Block(BLOCK_ID, capacity=2)
    assert b.size == 0
    assert b.is_empty
    assert not b.is_full

    b.add_record(rec(1))
    b.add_record(rec(2))
    assert b.size == 2
    assert not b.is_empty
    assert b.is_full


def test_validate_raises_on_unsorted_records():
    b = Block(BLOCK_ID, capacity=4)
    b._records.append(rec(2))  # bypass the public API to test validation
    b._records.append(rec(1))
    with pytest.raises(BlockInvariantError):
        b.validate()
