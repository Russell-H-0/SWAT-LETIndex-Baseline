"""Tests for Dataset (G1-A input normalization and partitioning)."""

import pytest

from enhanced_letindex.dataset import Dataset, DuplicateKeyError
from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.record import Record


def rec(k):
    return Record(RecordKey(k), f"v{k}")


def keys_of(records):
    return [r.key.value for r in records]


def test_unsorted_input_is_globally_sorted():
    ds = Dataset.from_records([rec(k) for k in (50, 10, 70, 20, 40, 30, 60)])
    assert keys_of(ds.records) == [10, 20, 30, 40, 50, 60, 70]
    assert ds.keys == tuple(RecordKey(k) for k in (10, 20, 30, 40, 50, 60, 70))


def test_partition_exact_multiple_of_capacity():
    ds = Dataset.from_records([rec(k) for k in (1, 2, 3, 4, 5, 6)])
    parts = ds.partitions(block_capacity=3)
    assert len(parts) == 2
    assert keys_of(parts[0]) == [1, 2, 3]
    assert keys_of(parts[1]) == [4, 5, 6]


def test_partition_with_partial_final_block():
    ds = Dataset.from_records([rec(k) for k in (10, 20, 30, 40, 50, 60, 70)])
    parts = ds.partitions(block_capacity=3)
    assert len(parts) == 3
    assert keys_of(parts[0]) == [10, 20, 30]
    assert keys_of(parts[1]) == [40, 50, 60]
    assert keys_of(parts[2]) == [70]  # partial final block


def test_empty_dataset_has_no_partitions():
    ds = Dataset.from_records([])
    assert ds.is_empty()
    assert ds.size == 0
    assert ds.partitions(block_capacity=3) == ()


def test_duplicate_keys_are_rejected():
    with pytest.raises(DuplicateKeyError):
        Dataset.from_records([rec(1), rec(2), rec(1)])


def test_duplicate_keys_in_unordered_input_are_rejected():
    with pytest.raises(DuplicateKeyError):
        Dataset.from_records([rec(9), rec(1), rec(9)])


def test_dataset_is_immutable_and_sorted_stably():
    records = [rec(3), rec(1), rec(2)]
    ds = Dataset.from_records(records)
    assert keys_of(ds.records) == [1, 2, 3]
    # the source list is untouched
    assert keys_of(records) == [3, 1, 2]
