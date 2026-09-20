"""Level construction: turn arbitrary Records into a sorted logical Level.

This module is the G1-A construction abstraction.  It deliberately lives in a
builder rather than inside ``Level`` (which stays a dumb ordered container of
logical block ids) or inside the whole of ``TrustedEngine``.

Construction pipeline (deterministic, vanilla baseline):

    records
      -> Dataset.from_records      (sort strictly by key, reject duplicates)
      -> Dataset.partitions        (pack into full blocks + final partial)
      -> one Block per partition   (fresh globally-unique BlockId)
      -> materialize each Block    (storage.store: allocate slot + WRITE)
      -> LogicalPhysicalMapping    (BlockId -> SlotId)
      -> Level                     (BlockIds in logical sorted order)

Physical placement is intentionally order-preserving: the first logical block
is written to the first free physical slot, the second to the next, and so on.
This is the *vanilla baseline* correlation that later security milestones
(G3/G4/G6) study and break.  It is an intentional, explicitly-ordered, NOT
secure placement.  ``relocate_block`` is never used here to disguise it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from .dataset import Dataset
from .identifiers import BlockId, LevelId
from .record import Record

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import TrustedEngine

__all__ = ["LevelBuilder", "LevelValidationError", "validate_level"]


class LevelValidationError(Exception):
    """Raised when a constructed level violates a G1-A invariant."""


class LevelBuilder:
    """Builds a logical Level from records using a :class:`TrustedEngine`.

    The builder reuses the engine's existing ``create_block`` primitive for
    every partition, so block id allocation, storage write, mapping assignment,
    and level attachment all go through the same G0 code paths that already
    preserve the mapping/storage invariants.  Construction therefore performs
    exactly one ``WRITE`` trace event per materialized block and never bypasses
    ``UntrustedStorage``.
    """

    def __init__(self, engine: "TrustedEngine") -> None:
        self.engine = engine

    def build(
        self, records: Iterable[Record], level_id: Optional[LevelId] = None
    ) -> LevelId:
        """Build a level from ``records`` and return its :class:`LevelId`.

        When ``level_id`` is ``None`` a fresh level is created; otherwise the
        blocks are appended to the caller's existing level.  An empty input
        yields an empty level (or appends nothing to an existing level) and
        produces no blocks, no dummies, and no storage writes.
        """
        dataset = Dataset.from_records(records)
        if level_id is None:
            level_id = self.engine.create_level()
        for partition in dataset.partitions(self.engine.config.block_capacity):
            self.engine.create_block(partition, level_id=level_id)
        return level_id


def validate_level(engine: "TrustedEngine", level_id: LevelId) -> None:
    """Validate the G1-A invariants of a constructed level.

    Establishes, raising :class:`LevelValidationError` on violation:

    1. all BlockIds in the level are unique;
    2. every block referenced by the level exists in the mapping and in
       storage, and the stored block carries the expected logical id;
    3. within each block, keys are strictly increasing;
    4. neighboring block key ranges do not overlap
       (``max_key(B_i) < min_key(B_(i+1))``), i.e. logical order agrees with
       increasing key ranges;
    5. every non-final block is full;
    6. for a non-empty level, the final block has size in ``[1, capacity]``.

    ``validate_level`` resolves each block through the normal
    ``mapping -> storage.read`` path, so every block it inspects emits a READ
    trace event — validation is a faithful (trace-visible) physical read, not a
    silent internal poke.  Tests that count construction WRITEs should validate
    separately from the construction under test.
    """
    level = engine.level(level_id)
    block_ids: tuple[BlockId, ...] = level.block_ids

    # 1. BlockIds within the level are unique.
    if len(block_ids) != len(set(block_ids)):
        raise LevelValidationError(
            f"level {level_id} contains duplicate logical BlockIds"
        )

    if not block_ids:
        return  # empty level: vacuously valid, no dummy blocks

    blocks = []
    for block_id in block_ids:
        # 2. Exists in mapping and storage, and the stored block matches.
        if not engine.mapping.is_mapped(block_id):
            raise LevelValidationError(
                f"level {level_id}: block {block_id} has no physical mapping"
            )
        block = engine.read_block(block_id)  # mapping lookup + READ trace
        if block.block_id != block_id:
            raise LevelValidationError(
                f"level {level_id}: slot for {block_id} holds "
                f"block {block.block_id}"
            )
        if block.is_empty:
            raise LevelValidationError(
                f"level {level_id}: block {block_id} is empty"
            )
        blocks.append(block)

    # 3. Keys strictly increasing within each block.
    for block in blocks:
        keys = [r.key for r in block.records]
        for left, right in zip(keys, keys[1:]):
            if left >= right:
                raise LevelValidationError(
                    f"level {level_id}: block {block.block_id} has "
                    f"non-increasing keys {left} -> {right}"
                )

    # 4. Neighboring ranges are strictly ordered and do not overlap.
    for left, right in zip(blocks, blocks[1:]):
        if not (left.max_key < right.min_key):
            raise LevelValidationError(
                f"level {level_id}: overlapping/out-of-order ranges "
                f"[{left.min_key}..{left.max_key}] vs "
                f"[{right.min_key}..{right.max_key}]"
            )

    # 5. All non-final blocks are full.
    for block in blocks[:-1]:
        if not block.is_full:
            raise LevelValidationError(
                f"level {level_id}: non-final block {block.block_id} is not "
                f"full (size {block.size} of {block.capacity})"
            )

    # 6. Final block size is in [1, capacity] (non-empty level).
    final = blocks[-1]
    if final.size < 1 or final.size > engine.config.block_capacity:
        raise LevelValidationError(
            f"level {level_id}: final block {final.block_id} has invalid "
            f"size {final.size} (capacity {engine.config.block_capacity})"
        )
