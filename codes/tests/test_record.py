"""Tests for Record."""

from dataclasses import FrozenInstanceError

import pytest

from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.record import Record


def test_record_holds_key_and_value():
    r = Record(key=RecordKey(10), value="ten")
    assert r.key == RecordKey(10)
    assert r.value == "ten"


def test_record_version_and_tombstone_defaults():
    r = Record(key=RecordKey(1), value=42)
    assert r.version is None
    assert r.tombstone is False


def test_record_key_is_comparable():
    a = Record(RecordKey(1), "x")
    b = Record(RecordKey(2), "y")
    assert a.key < b.key


def test_record_is_frozen():
    r = Record(RecordKey(1), "x")
    with pytest.raises(FrozenInstanceError):
        r.key = RecordKey(2)  # type: ignore[misc]


def test_identifier_namespaces_are_distinct():
    # A slot id must never silently equal a block id with the same integer.
    from enhanced_letindex.identifiers import BlockId, SlotId

    assert BlockId(3) != SlotId(3)
