"""Record: the smallest logical unit of stored data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .identifiers import RecordKey

__all__ = ["Record"]


@dataclass(frozen=True)
class Record:
    """A single key/value entry.

    ``version`` and ``tombstone`` are reserved for future update/delete
    semantics and carry no behavioral meaning in G0; distinct LSM versions
    and tombstones will be layered on later milestones.
    """

    key: RecordKey
    value: Any
    version: Optional[int] = None
    tombstone: bool = False
