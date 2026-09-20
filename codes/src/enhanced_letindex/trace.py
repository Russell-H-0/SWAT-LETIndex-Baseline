"""Access trace: an ordered record of observable REE storage operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .identifiers import LevelId, SlotId

__all__ = ["TraceOperation", "TraceEvent", "TraceCollector"]


class TraceOperation(Enum):
    """The kinds of physical operations an adversary can observe."""

    READ = "READ"
    WRITE = "WRITE"


@dataclass(frozen=True)
class TraceEvent:
    """A single observed REE storage operation.

    Metadata is deliberately conservative: only the physical slot, the
    operation type, an optional level hint (when a caller deliberately
    exposes it), and optional caller-supplied metadata.  Plaintext key
    information is never auto-exposed.
    """

    seq: int
    operation: TraceOperation
    slot_id: SlotId
    level_id: Optional[LevelId] = None
    metadata: Optional[dict] = None

    def as_dict(self) -> dict:
        """Plain structure suitable for future attack/leakage experiments."""
        return {
            "seq": self.seq,
            "operation": self.operation.value,
            "slot_id": self.slot_id.value,
            "level_id": None if self.level_id is None else self.level_id.value,
            "metadata": self.metadata,
        }


class TraceCollector:
    """Collects an ordered sequence of :class:`TraceEvent`."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    def record(
        self,
        operation: TraceOperation,
        slot_id: SlotId,
        level_id: Optional[LevelId] = None,
        metadata: Optional[dict] = None,
    ) -> TraceEvent:
        """Append a new event with an auto-incremented sequence number."""
        event = TraceEvent(
            seq=len(self._events),
            operation=operation,
            slot_id=slot_id,
            level_id=level_id,
            metadata=metadata,
        )
        self._events.append(event)
        return event

    def clear(self) -> None:
        """Drop all recorded events."""
        self._events.clear()

    def events(self) -> tuple[TraceEvent, ...]:
        """The recorded events, in order, as an immutable tuple."""
        return tuple(self._events)

    def as_dicts(self) -> list[dict]:
        """Serialize the trace to a list of plain dictionaries."""
        return [event.as_dict() for event in self._events]

    def to_json(self) -> str:
        """Serialize the trace to an indented JSON string."""
        return json.dumps(self.as_dicts(), indent=2)
