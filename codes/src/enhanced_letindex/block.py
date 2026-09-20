"""Block: the basic REE I/O unit."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .identifiers import BlockId, RecordKey
from .record import Record

__all__ = [
    "Block",
    "BlockError",
    "BlockFullError",
    "BlockInvariantError",
]


class BlockError(Exception):
    """Base error for block construction/validation problems."""


class BlockFullError(BlockError):
    """Raised when adding a record would exceed the block's capacity."""


class BlockInvariantError(BlockError):
    """Raised when a block's internal invariants are violated."""


@dataclass
class Block:
    """A logical block holding records sorted by key.

    Invariants (checked on construction and by :meth:`validate`):

    * ``capacity >= 1``;
    * ``len(records) <= capacity``;
    * keys are strictly increasing by ascending order
      (``k_1 < k_2 < ... < k_n``); duplicate keys are rejected;
    * ``min_key <= max_key`` whenever the block is non-empty.

    ``_records`` is internal.  Mutations go through :meth:`add_record`, which
    keeps the list sorted.  Callers must not mutate the list directly.
    """

    block_id: BlockId
    capacity: int
    _records: list[Record] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise BlockInvariantError(
                f"block {self.block_id}: capacity must be >= 1, got {self.capacity}"
            )
        self.validate()

    # ---------------------------------------------------------------- constructors

    @classmethod
    def sorted_block(
        cls, block_id: BlockId, records: Iterable[Record], capacity: int
    ) -> "Block":
        """Build a Block, sorting ``records`` by key first."""
        return cls(block_id, capacity, sorted(records, key=lambda r: r.key))

    # ---------------------------------------------------------------- properties

    @property
    def records(self) -> tuple[Record, ...]:
        """Records as an immutable tuple, in ascending key order."""
        return tuple(self._records)

    @property
    def size(self) -> int:
        """Number of records currently in the block."""
        return len(self._records)

    @property
    def is_empty(self) -> bool:
        """True when the block holds no records."""
        return not self._records

    @property
    def is_full(self) -> bool:
        """True when the block has reached its record capacity."""
        return self.size >= self.capacity

    @property
    def min_key(self) -> Optional[RecordKey]:
        """Smallest key in the block, or ``None`` when empty."""
        return None if self.is_empty else self._records[0].key

    @property
    def max_key(self) -> Optional[RecordKey]:
        """Largest key in the block, or ``None`` when empty."""
        return None if self.is_empty else self._records[-1].key

    # ---------------------------------------------------------------- mutation

    def add_record(self, record: Record) -> None:
        """Insert a record at its sorted position.

        Raises :class:`BlockFullError` when the block is already at capacity.
        """
        self.validate()
        if self.is_full:
            raise BlockFullError(
                f"block {self.block_id} is full (capacity {self.capacity})"
            )
        bisect.insort(self._records, record, key=lambda r: r.key)
        self.validate()

    def validate(self) -> None:
        """Check all block invariants; raise :class:`BlockInvariantError`.

        Enforces the frozen strict-key invariant: keys must be strictly
        increasing (``k_1 < k_2 < ... < k_n``), so any adjacent pair with
        equal keys is rejected.  Duplicate keys are therefore invalid at the
        Block level, independently of :class:`~enhanced_letindex.dataset.Dataset`.
        """
        if self.capacity < 1:
            raise BlockInvariantError(
                f"block {self.block_id}: capacity must be >= 1"
            )
        if self.size > self.capacity:
            raise BlockInvariantError(
                f"block {self.block_id}: size {self.size} exceeds capacity {self.capacity}"
            )
        for left, right in zip(self._records, self._records[1:]):
            if left.key >= right.key:
                raise BlockInvariantError(
                    f"block {self.block_id}: keys not strictly increasing "
                    f"by ascending key ({left.key} >= {right.key})"
                )
