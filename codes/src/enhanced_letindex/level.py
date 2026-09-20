"""Level: a logical, ordered collection of blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .identifiers import BlockId, LevelId

__all__ = ["Level", "LevelError"]


class LevelError(Exception):
    """Raised for invalid level operations."""


@dataclass
class Level:
    """A logical level.

    Holds *ordered logical block identifiers* only.  It deliberately stores
    no physical location information: physical placement is owned by
    ``LogicalPhysicalMapping`` and ``UntrustedStorage``.  This keeps the
    level's logical ordering independent of physical storage order, which is
    the property future reshuffle/merge milestones will rely on.
    """

    level_id: LevelId
    _block_ids: list[BlockId] = field(default_factory=list, repr=False)

    # ---------------------------------------------------------------- properties

    @property
    def block_ids(self) -> tuple[BlockId, ...]:
        """The logical block ids in level order (immutable view)."""
        return tuple(self._block_ids)

    @property
    def size(self) -> int:
        """Number of logical blocks in this level."""
        return len(self._block_ids)

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._block_ids

    # ---------------------------------------------------------------- mutation

    def add_block(self, block_id: BlockId) -> None:
        """Append a logical block id to this level."""
        if block_id in self._block_ids:
            raise LevelError(
                f"block {block_id} already present in level {self.level_id}"
            )
        self._block_ids.append(block_id)

    def remove_block(self, block_id: BlockId) -> None:
        """Remove a logical block id from this level."""
        if block_id not in self._block_ids:
            raise LevelError(
                f"block {block_id} not present in level {self.level_id}"
            )
        self._block_ids.remove(block_id)
