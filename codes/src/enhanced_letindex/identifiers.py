"""Distinct identifier namespaces for EnhancedLETIndex.

The security-relevant invariant of this prototype is that *logical block
identity* and *physical storage position* are different concepts.  To make
namespace confusion hard to introduce, each kind of identifier gets its own
lightweight, immutable, ordered wrapper type instead of reusing bare ``int``.

All four types are frozen dataclasses wrapping a single ``int``.  They are
deliberately NOT the same type, so ``BlockId(3) == SlotId(3)`` is ``False``,
which prevents a physical slot id from being silently used where a logical
block id is expected (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RecordKey", "BlockId", "LevelId", "SlotId"]


@dataclass(frozen=True, order=True)
class RecordKey:
    """A plaintext record key.  Comparable integer keys in G0."""

    value: int


@dataclass(frozen=True, order=True)
class BlockId:
    """A logical block identifier (trusted-side namespace)."""

    value: int


@dataclass(frozen=True, order=True)
class LevelId:
    """A logical level identifier (trusted-side namespace)."""

    value: int


@dataclass(frozen=True, order=True)
class SlotId:
    """A physical storage slot identifier (untrusted-side namespace)."""

    value: int
