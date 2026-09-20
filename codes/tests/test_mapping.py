"""Tests for the logical-to-physical mapping (the indirection layer)."""

import pytest

from enhanced_letindex.identifiers import BlockId, SlotId
from enhanced_letindex.mapping import (
    BijectionError,
    DuplicateLogicalBlockError,
    DuplicatePhysicalSlotError,
    LogicalPhysicalMapping,
    MissingLogicalBlockError,
)

B1 = BlockId(1)
B2 = BlockId(2)
P0 = SlotId(0)
P5 = SlotId(5)


def test_lookup_returns_assigned_slot():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    assert m.lookup(B1) == P0


def test_lookup_missing_block_raises():
    m = LogicalPhysicalMapping()
    with pytest.raises(MissingLogicalBlockError):
        m.lookup(B1)


def test_move_changes_physical_mapping():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    m.move(B1, P5)
    assert m.lookup(B1) == P5
    assert not m.slot_in_use(P0)
    m.validate_bijection()


def test_duplicate_physical_assignment_is_detected():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    with pytest.raises(DuplicatePhysicalSlotError):
        m.assign(B2, P0)


def test_duplicate_logical_assignment_is_detected():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    with pytest.raises(DuplicateLogicalBlockError):
        m.assign(B1, P5)


def test_move_to_occupied_slot_is_detected():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    m.assign(B2, P5)
    with pytest.raises(DuplicatePhysicalSlotError):
        m.move(B1, P5)


def test_bijection_validates_after_operations():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    m.assign(B2, P5)
    m.move(B2, SlotId(9))
    m.validate_bijection()
    assert m.slot_owner(SlotId(9)) == B2


def test_bijection_detects_corruption():
    m = LogicalPhysicalMapping()
    m.assign(B1, P0)
    m._physical_to_logical.pop(P0)  # corrupt the two-way bookkeeping
    with pytest.raises(BijectionError):
        m.validate_bijection()
