"""Privileged evaluator truth for LETIndex-Ref-v1 (scoring/debug only).

**This module is evaluator-only.**  Nothing here may be consumed by an
adversary-facing path: the attack-facing interface is ``ref_adversary``
(``AdversaryTranscript`` + explicitly declared public configuration + the optional
opaque M1 channel), and ``ref_adversary`` deliberately does not import this module.

The truth answers, at minimum:

```text
physical SlotId            -> level
physical SlotId            -> logical block rank within that level
logical block rank         -> physical SlotId
logical block rank         -> key / item-rank interval contained in the block
query                      -> exact target key/rank and hit level
```

It exists so a later attack's *recovery* output can be scored against what the
reference implementation actually did; it never leaks into a transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .identifiers import SlotId
from .letindex_ref import (
    PROFILE_NAME,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    ReferenceIndex,
    ReferenceProfileError,
)

__all__ = [
    "TRUTH_SCHEMA_VERSION",
    "ReferenceEvaluatorTruth",
    "TRUTH_KEYS",
]

TRUTH_SCHEMA_VERSION = "g2b0-truth-1"

#: Top-level keys of a truth artifact (used by the separation tests).
TRUTH_KEYS = frozenset({
    "schema_version", "profile", "upstream", "classification", "levels",
    "projection", "queries",
})


class ReferenceEvaluatorTruth:
    """Privileged ground truth of one LETIndex-Ref-v1 index."""

    classification = "EVALUATOR-ONLY"

    def __init__(self, index: ReferenceIndex) -> None:
        self.index = index

    # -------------------------------------------------- SlotId <-> geometry

    def physical_offset_of(self, level: int, block_rank: int) -> int:
        """``logical block rank -> physical block offset`` (applies the placement)."""
        reference = self.index.level(level)
        if (
            isinstance(block_rank, bool)
            or not isinstance(block_rank, int)
            or not 0 <= block_rank < reference.block_count
        ):
            raise ReferenceProfileError(
                f"level {level} has {reference.block_count} logical blocks; "
                f"no rank {block_rank!r}"
            )
        return reference.physical_block(block_rank)

    def logical_block_rank(self, level: int, physical_offset: int) -> int:
        """``physical block offset -> logical block rank`` (inverts the placement)."""
        return self.index.level(level).logical_block(physical_offset)

    def slot_of(self, level: int, block_rank: int) -> SlotId:
        """``logical block rank -> physical SlotId``.

        Placement-aware: the rank is first mapped through the level's layout
        (``ordered`` = identity, ``static_prp`` = the hidden permutation) and only then
        through the ``(level, physical offset) -> SlotId`` projection.
        """
        return self.index.projection.slot(
            level, self.physical_offset_of(level, block_rank)
        )

    def slot_of_block(self, level: int, block_rank: int) -> SlotId:
        """Alias with an explicit name: ``logical block rank -> physical SlotId``."""
        return self.slot_of(level, block_rank)

    def slot_of_physical_offset(self, level: int, physical_offset: int) -> SlotId:
        """``physical block offset -> physical SlotId`` (no placement applied)."""
        return self.index.projection.slot(level, physical_offset)

    def locate(self, slot: SlotId | int) -> tuple[int, int]:
        """``physical SlotId -> (level, logical block rank)``.

        Placement-aware: the projection yields the physical offset, which is then
        inverted through the level's layout into a logical block rank.
        """
        level, physical_offset = self.index.projection.locate(slot)
        return level, self.logical_block_rank(level, physical_offset)

    def physical_offset(self, slot: SlotId | int) -> tuple[int, int]:
        """``physical SlotId -> (level, physical block offset)`` (raw projection)."""
        return self.index.projection.locate(slot)

    def level_of(self, slot: SlotId | int) -> int:
        return self.locate(slot)[0]

    def block_rank_of(self, slot: SlotId | int) -> int:
        return self.locate(slot)[1]

    def physical_offset_of_slot(self, slot: SlotId | int) -> int:
        """``physical SlotId -> physical block offset`` within its level."""
        return self.index.projection.locate(slot)[1]

    # ---------------------------------------------------------- block truth

    def block_key_interval(self, level: int, block_rank: int) -> dict:
        """Key/item-rank interval contained in one **logical** block of one level."""
        reference = self.index.level(level)
        physical_offset = self.physical_offset_of(level, block_rank)
        start, end = reference.block_range(block_rank)
        return {
            "level": level,
            "block_rank": block_rank,
            "logical_block_rank": block_rank,
            "physical_block_offset": physical_offset,
            "slot_id": self.slot_of(level, block_rank).value,
            "item_start": start,
            "item_end": end,
            "item_count": end - start,
            "key_lo": reference.keys[start].value if end > start else None,
            "key_hi": reference.keys[end - 1].value if end > start else None,
        }

    def item_rank_of(self, level: int, key: int) -> Optional[int]:
        """Exact item rank of ``key`` inside one level, or ``None``."""
        reference = self.index.level(level)
        lo, hi = 0, reference.item_count
        while lo < hi:
            mid = (lo + hi) // 2
            if reference.keys[mid].value < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < reference.item_count and reference.keys[lo].value == key:
            return lo
        return None

    # ---------------------------------------------------------- query truth

    def query_truth(self, key: int) -> dict:
        """Exact target key/rank and hit level for one query key."""
        outcome = self.index.query(key)
        per_level = []
        for level_outcome in outcome.level_outcomes:
            reference = self.index.level(level_outcome.level)
            per_level.append({
                "level": level_outcome.level,
                "path": level_outcome.path,
                "interval": list(level_outcome.interval),
                "pos": level_outcome.pos,
                "found": level_outcome.found,
                "item_rank": level_outcome.pos,
                "block_rank": (
                    level_outcome.pos // reference.items_per_block
                    if level_outcome.pos < reference.item_count
                    else None
                ),
                "physical_block_offset": (
                    self.physical_offset_of(
                        level_outcome.level,
                        level_outcome.pos // reference.items_per_block,
                    )
                    if level_outcome.pos < reference.item_count
                    else None
                ),
                "slot_id": (
                    self.slot_of(
                        level_outcome.level,
                        level_outcome.pos // reference.items_per_block,
                    ).value
                    if level_outcome.pos < reference.item_count
                    else None
                ),
                "probes": list(level_outcome.probes),
                "read_count": level_outcome.read_count,
            })
        hit_rank = None
        hit_block = None
        hit_slot = None
        if outcome.hit_level is not None:
            hit_rank = self.item_rank_of(outcome.hit_level, key)
            if hit_rank is not None:
                hit_block = hit_rank // self.index.level(outcome.hit_level).items_per_block
                hit_slot = self.slot_of(outcome.hit_level, hit_block).value
        return {
            "key": key,
            "found": outcome.found,
            "value": outcome.value,
            "hit_level": outcome.hit_level,
            "hit_item_rank": hit_rank,
            "hit_block_rank": hit_block,
            "hit_physical_block_offset": (
                self.physical_offset_of(outcome.hit_level, hit_block)
                if outcome.hit_level is not None and hit_block is not None
                else None
            ),
            "hit_slot_id": hit_slot,
            "searched_levels": list(outcome.searched_levels),
            "skipped_empty_levels": list(outcome.skipped_empty_levels),
            "per_level": per_level,
        }

    # ------------------------------------------------------------- artifact

    def to_dict(self, *, query_keys: Sequence[int] = ()) -> dict:
        profiles = self.index.profile
        levels = []
        for level in self.index.levels():
            reference = self.index.level(level)
            if reference.item_count == 0:
                continue
            levels.append({
                "level": level,
                "item_count": reference.item_count,
                "block_count": reference.block_count,
                "items_per_block": reference.items_per_block,
                "backing": reference.backing,
                "has_pgm": reference.has_pgm,
                "region_start": self.index.projection.region(level).region_start,
                "placement": list(reference.physical_blocks()),
                "blocks": [
                    self.block_key_interval(level, rank)
                    for rank in range(reference.block_count)
                ],
            })
        return {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "profile": PROFILE_NAME,
            "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT},
            "classification": {
                "truth": self.classification,
                "layout": profiles.layout,
                "prp_key": profiles.prp_key,
                "note": (
                    "evaluator-only scoring/debug data; must never be consumed by an "
                    "adversary-facing path (see ref_adversary).  block ranks are logical "
                    "ranks; physical offsets and SlotIds apply the level placement"
                ),
            },
            "projection": self.index.projection.to_dict(),
            "levels": levels,
            "queries": [self.query_truth(key) for key in query_keys],
        }

    def to_json(self, *, query_keys: Sequence[int] = ()) -> str:
        return json.dumps(self.to_dict(query_keys=query_keys), indent=2)

    def write_json(self, path: str | Path, *, query_keys: Sequence[int] = ()) -> dict:
        payload = self.to_dict(query_keys=query_keys)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
