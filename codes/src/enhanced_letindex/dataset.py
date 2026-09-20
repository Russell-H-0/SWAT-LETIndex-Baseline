"""Dataset: normalized, sorted, duplicate-free records feeding level construction.

G1-A freezes the following data semantics before any block or level is built:

* keys are unique comparable integer keys;
* duplicate keys are invalid;
* input may arrive in arbitrary order and is sorted strictly by key here.

This module owns the *normalization* step.  It is independent of the engine so
that the sorting and partitioning behavior is testable in isolation and can be
reused by later milestones (e.g. G1-B batch PGM construction) without change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .identifiers import RecordKey
from .record import Record

__all__ = ["Dataset", "DuplicateKeyError"]


class DuplicateKeyError(ValueError):
    """Raised when a dataset contains two records with the same key.

    G1-A treats duplicate keys as invalid input, not as something to merge or
    overwrite (that is future insert/update semantics, explicitly out of scope
    here).
    """


@dataclass(frozen=True)
class Dataset:
    """An immutable, key-sorted, duplicate-free collection of records.

    This is the canonical pre-level form of any set of records.  It is immutable
    (``frozen=True``) so a built dataset can be safely shared across
    construction steps and later milestones.

    The empty dataset is a valid input: it represents an empty level with no
    blocks and no dummy blocks.
    """

    records: tuple[Record, ...]

    @classmethod
    def from_records(cls, records: Iterable[Record]) -> "Dataset":
        """Sort ``records`` strictly ascending by key and reject duplicates.

        Sorting is by key only.  Because keys are unique in G1-A, no tie-breaker
        is needed; two records with an equal key after sorting are a hard error.
        """
        ordered = tuple(sorted(records, key=lambda r: r.key))
        for left, right in zip(ordered, ordered[1:]):
            if left.key == right.key:
                raise DuplicateKeyError(f"duplicate record key {left.key}")
        return cls(ordered)

    @property
    def keys(self) -> tuple[RecordKey, ...]:
        """The sorted keys, ascending."""
        return tuple(r.key for r in self.records)

    @property
    def size(self) -> int:
        """Number of records."""
        return len(self.records)

    def is_empty(self) -> bool:
        """True when the dataset holds no records."""
        return not self.records

    def partitions(self, block_capacity: int) -> tuple[tuple[Record, ...], ...]:
        """Split the sorted records into full blocks plus a final partial.

        Deterministic packing per ``Config.block_capacity``:

        * every block except possibly the final one has exactly
          ``block_capacity`` records;
        * the final block, when present, has between 1 and ``block_capacity``
          records;
        * an empty dataset yields zero partitions (never a single empty one).
        """
        if block_capacity < 1:
            raise ValueError(
                f"block_capacity must be >= 1, got {block_capacity}"
            )
        if self.is_empty():
            return ()
        return tuple(
            self.records[i : i + block_capacity]
            for i in range(0, len(self.records), block_capacity)
        )
