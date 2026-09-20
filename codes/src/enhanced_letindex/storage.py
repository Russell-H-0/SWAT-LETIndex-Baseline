"""Untrusted (REE) storage layer, keyed by physical slot."""

from __future__ import annotations

from typing import Optional

from .block import Block
from .identifiers import SlotId
from .trace import TraceCollector, TraceOperation

__all__ = ["UntrustedStorage", "StorageError", "StorageConsistencyError"]


class StorageError(Exception):
    """Raised for invalid physical-storage operations."""


class StorageConsistencyError(StorageError):
    """The storage layer and the logical mapping disagree about which
    physical slots contain active blocks.

    This signals a violated invariant (for example, a slot the mapping
    considers free still holds a block).  Such a state is never silently
    repaired; callers must fail loudly.
    """


class UntrustedStorage:
    """Simulated untrusted storage, addressed purely by physical slot.

    The storage layer does not understand plaintext query semantics and,
    critically, does not record logical block ids or plaintext keys in the
    access trace: it records only physical slot ids plus the operation type
    (and an optional level hint when explicitly supplied).  This is what keeps
    the trusted logical namespace hidden from the adversary.

    G0 stores plaintext :class:`Block` objects; encryption is a later
    milestone.  Slot reuse/overwrite at this layer is permitted; the
    uniqueness of active mapping is enforced one layer up, in
    ``LogicalPhysicalMapping``.
    """

    def __init__(self, trace: Optional[TraceCollector] = None) -> None:
        self._trace = trace if trace is not None else TraceCollector()
        self._slots: dict[SlotId, Block] = {}
        self._next_free_slot = 0

    # ---------------------------------------------------------------- introspect

    @property
    def trace(self) -> TraceCollector:
        """The trace collector recording all physical operations."""
        return self._trace

    @property
    def size(self) -> int:
        """Number of currently occupied physical slots (for testing)."""
        return len(self._slots)

    def occupied_slots(self) -> tuple[SlotId, ...]:
        """Physical slot ids that currently hold a block."""
        return tuple(sorted(self._slots))

    def contains(self, slot_id: SlotId) -> bool:
        """True when ``slot_id`` currently holds a block."""
        return slot_id in self._slots

    # ---------------------------------------------------------------- operations

    def allocate(self) -> SlotId:
        """Reserve and return the next free physical slot."""
        slot = SlotId(self._next_free_slot)
        self._next_free_slot += 1
        return slot

    def store(self, block: Block) -> SlotId:
        """Allocate a fresh slot and write ``block`` into it; return the slot."""
        slot = self.allocate()
        self.write(slot, block)
        return slot

    def read(self, slot_id: SlotId) -> Block:
        """Read the block held at physical ``slot_id`` (records a READ trace)."""
        self._trace.record(TraceOperation.READ, slot_id)
        try:
            return self._slots[slot_id]
        except KeyError:
            raise StorageError(f"physical slot {slot_id} is empty") from None

    def write(self, slot_id: SlotId, block: Block) -> None:
        """Write ``block`` into physical ``slot_id`` (records a WRITE trace)."""
        self._trace.record(TraceOperation.WRITE, slot_id)
        self._slots[slot_id] = block

    def clear(self, slot_id: SlotId) -> None:
        """Vacate an occupied physical slot, observable as a WRITE.

        Clearing a slot is an explicit, externally observable mutation: it is
        modeled on the access trace as a ``WRITE`` to ``slot_id`` so that the
        adversary sees the slot was touched.  After clearing, the slot is
        available for later reuse, no longer yields the old block on ``read``,
        and the occupied-slot count decreases by one.

        This is state bookkeeping only, not secure erase: removing the Python
        reference to the block provides no cryptographic deletion.

        Raises :class:`StorageError` when ``slot_id`` is not occupied.
        """
        if slot_id not in self._slots:
            raise StorageError(
                f"cannot clear physical slot {slot_id}: not occupied"
            )
        self._trace.record(TraceOperation.WRITE, slot_id)
        del self._slots[slot_id]
