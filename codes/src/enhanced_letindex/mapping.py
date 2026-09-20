"""Logical-to-physical block mapping (the indirection layer)."""

from __future__ import annotations

from typing import Iterator, Optional

from .identifiers import BlockId, SlotId

__all__ = [
    "LogicalPhysicalMapping",
    "MappingError",
    "DuplicateLogicalBlockError",
    "DuplicatePhysicalSlotError",
    "MissingLogicalBlockError",
    "BijectionError",
]


class MappingError(Exception):
    """Base error for mapping problems."""


class DuplicateLogicalBlockError(MappingError):
    """A logical block was assigned twice."""


class DuplicatePhysicalSlotError(MappingError):
    """A physical slot was assigned to two different logical blocks."""


class MissingLogicalBlockError(MappingError):
    """A lookup targeted a logical block with no physical mapping."""


class BijectionError(MappingError):
    """The active mapping is not a valid bijection."""


class LogicalPhysicalMapping:
    """Explicit indirection between logical block ids and physical slot ids.

    The mapping maintains two-way bookkeeping so the *active* mapping is
    always a bijection: each active logical block has exactly one physical
    slot, and each physical slot hosts at most one logical block.

    This layer is the security-critical seam where future random-permutation
    and reshuffle logic will live.  G0 provides only deterministic assignment
    and explicit (caller-requested) remapping; no randomness is applied yet.
    """

    def __init__(self) -> None:
        self._logical_to_physical: dict[BlockId, SlotId] = {}
        self._physical_to_logical: dict[SlotId, BlockId] = {}

    # ---------------------------------------------------------------- queries

    def lookup(self, block_id: BlockId) -> SlotId:
        """Return the physical slot hosting ``block_id``."""
        try:
            return self._logical_to_physical[block_id]
        except KeyError:
            raise MissingLogicalBlockError(
                f"logical block {block_id} has no physical mapping"
            ) from None

    def physical_slot(self, block_id: BlockId) -> SlotId:
        """Alias for :meth:`lookup`."""
        return self.lookup(block_id)

    def slot_owner(self, slot_id: SlotId) -> Optional[BlockId]:
        """Return the logical block mapped to ``slot_id``, or ``None``."""
        return self._physical_to_logical.get(slot_id)

    def slot_in_use(self, slot_id: SlotId) -> bool:
        """True when some active logical block is mapped to ``slot_id``."""
        return slot_id in self._physical_to_logical

    def is_mapped(self, block_id: BlockId) -> bool:
        """True when ``block_id`` has a physical mapping."""
        return block_id in self._logical_to_physical

    def items(self) -> Iterator[tuple[BlockId, SlotId]]:
        """Yield ``(block_id, slot_id)`` pairs in logical-block order."""
        yield from sorted(self._logical_to_physical.items())

    # ---------------------------------------------------------------- mutation

    def assign(self, block_id: BlockId, slot_id: SlotId) -> None:
        """Map ``block_id`` to ``slot_id``.

        Raises :class:`DuplicateLogicalBlockError` if ``block_id`` is already
        mapped, and :class:`DuplicatePhysicalSlotError` if ``slot_id`` is
        already taken by another logical block.
        """
        if block_id in self._logical_to_physical:
            existing = self._logical_to_physical[block_id]
            if existing == slot_id:
                return  # idempotent re-assignment of the same pair
            raise DuplicateLogicalBlockError(
                f"logical block {block_id} is already mapped to slot {existing}"
            )
        if slot_id in self._physical_to_logical:
            raise DuplicatePhysicalSlotError(
                f"physical slot {slot_id} is already assigned to "
                f"logical block {self._physical_to_logical[slot_id]}"
            )
        self._logical_to_physical[block_id] = slot_id
        self._physical_to_logical[slot_id] = block_id
        self.validate_bijection()

    def move(self, block_id: BlockId, new_slot: SlotId) -> None:
        """Remap ``block_id`` from its current slot to ``new_slot``.

        Raises :class:`MissingLogicalBlockError` if ``block_id`` is not
        mapped, and :class:`DuplicatePhysicalSlotError` if ``new_slot`` is
        already taken by another logical block.
        """
        old_slot = self.lookup(block_id)
        if old_slot == new_slot:
            return
        if new_slot in self._physical_to_logical:
            raise DuplicatePhysicalSlotError(
                f"physical slot {new_slot} is already assigned to "
                f"logical block {self._physical_to_logical[new_slot]}"
            )
        del self._physical_to_logical[old_slot]
        self._physical_to_logical[new_slot] = block_id
        self._logical_to_physical[block_id] = new_slot
        self.validate_bijection()

    def unassign(self, block_id: BlockId) -> SlotId:
        """Remove ``block_id`` from the active mapping (retirement primitive).

        Returns the physical slot the block used to occupy, so the caller can
        make that slot's storage consistent (see
        ``TrustedEngine.retire_level``).  The mapping itself performs no storage
        operation and emits no trace event: retirement of the physical copy is
        the caller's responsibility.

        Raises :class:`MissingLogicalBlockError` when ``block_id`` is not part of
        the active mapping.  The two-way bookkeeping is re-validated so the
        active mapping remains a bijection.
        """
        slot_id = self.lookup(block_id)
        del self._logical_to_physical[block_id]
        del self._physical_to_logical[slot_id]
        self.validate_bijection()
        return slot_id

    # ---------------------------------------------------------------- validation

    def validate_bijection(self) -> None:
        """Validate that the active mapping is a bijection.

        Raises :class:`BijectionError` if a logical block maps to more than
        one slot, two logical blocks map to the same slot, or the two-way
        bookkeeping is inconsistent.
        """
        if len(self._logical_to_physical) != len(self._physical_to_logical):
            raise BijectionError(
                "mapping is not a bijection: "
                f"{len(self._logical_to_physical)} logical entries vs "
                f"{len(self._physical_to_logical)} physical entries"
            )
        for block_id, slot_id in self._logical_to_physical.items():
            if self._physical_to_logical.get(slot_id) != block_id:
                raise BijectionError(
                    f"mapping inconsistent for block {block_id} -> slot {slot_id}"
                )
