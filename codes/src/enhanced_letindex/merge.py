"""G1-D blocking merge baseline (decisions/0006-g1-d-blocking-merge.md).

Two operations, both deliberately simple and leakage-prone:

* ``create_update_level`` — upsert as a *fresh newest level* (never in-place
  mutation of an existing level);
* ``merge_blocking`` — an explicit blocking two-level merge with caller-supplied
  newer/older precedence, a complete input read, a fresh canonical output level,
  an output PGM built through the accepted G1-B1 batch path, and retirement of
  both inputs only after successful publication.

This is the vanilla baseline that G7 privacy-aware / de-amortized merge must
later improve.  It introduces no de-amortization, no merge scheduling or ratio
policy, no PGM-guided partition/writeback, no reshuffle/PRP/stash, no dummy or
cover traffic, no encryption/ORAM, and no tombstones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from .dataset import Dataset
from .engine import MergeResult
from .identifiers import LevelId, RecordKey
from .record import Record

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import TrustedEngine
    from .level import Level

__all__ = ["LevelMerger", "MergeInputError", "EmptyUpdateBatchError"]


class MergeInputError(ValueError):
    """A blocking merge was given inputs that cannot be merged."""


class EmptyUpdateBatchError(ValueError):
    """An upsert batch was empty, so no newest level can be published."""


class LevelMerger:
    """Blocking (non-de-amortized) merge baseline for one engine.

    The merger reuses the accepted G1-A construction path for every level it
    publishes, so canonical packing, globally unique block ids, deterministic
    sequential placement and one WRITE per output block are all inherited from
    ``LevelBuilder`` rather than reimplemented here.
    """

    def __init__(self, engine: "TrustedEngine") -> None:
        self.engine = engine

    # -------------------------------------------------------------- upsert (1)

    def create_update_level(self, records: Iterable[Record]) -> LevelId:
        """Publish one upsert batch as a fresh newest logical level.

        Duplicate keys *within one batch* are invalid (``DuplicateKeyError`` from
        ``Dataset.from_records``, raised before any level or block exists); an
        empty batch is rejected with ``EmptyUpdateBatchError`` rather than
        silently publishing an empty newest level.  Keys may overlap older
        levels — that is the normal update case and is resolved by caller-supplied
        G1-C query order (newest level first).

        The new level carries fresh ``LevelId``/``BlockId``s and is built with the
        canonical G1-A packing rules.  Its PGM is not built here: querying requires
        an explicit ``build_level_pgm`` call, exactly as for any other level.
        """
        batch = tuple(records)
        if not batch:
            raise EmptyUpdateBatchError(
                "update batch is empty: refusing to publish an empty newest level"
            )
        dataset = Dataset.from_records(batch)  # sorts by key, rejects duplicates
        return self.engine.build_level(dataset.records)

    # --------------------------------------------------------------- merge (2)

    def merge_blocking(
        self, newer_level_id: LevelId, older_level_id: LevelId
    ) -> MergeResult:
        """Blocking merge of two levels; ``newer_level_id`` wins duplicate keys.

        Precedence is taken only from the caller's parameter choice — never from
        ``LevelId`` value, level size, creation order, or physical placement.

        Trace composition of a successful merge (decisions/0006 §11):

        A. input read phase: every block of the newer level then every block of
           the older level, in logical block order, one READ each;
        B. output materialization: one WRITE per output block, in construction
           order;
        C. output PGM construction: one READ per output block, in logical block
           order (the accepted G1-B1 builder);
        D. retirement: one observable WRITE per retired input block.

        Nothing else is emitted: no validation READ pass, no dummy/cover traffic,
        no reshuffle, and no WRITE outside B and D.

        Canonical input validation happens inside phase A with zero additional
        I/O (see :meth:`_read_level_records`), and the output ``LevelId`` is
        reserved before phase B starts so a mid-materialization exception can
        never leave a partial output active (decisions/0006 §9).
        """
        engine = self.engine

        if newer_level_id == older_level_id:
            raise MergeInputError(
                "blocking merge requires two distinct levels; "
                f"got {newer_level_id} as both newer and older input"
            )

        # resolving the inputs is trusted state: unknown -> KeyError,
        # retired -> RetiredLevelError, both before any merge I/O
        newer = engine.level(newer_level_id)
        older = engine.level(older_level_id)

        # ---- phase A: complete blocking input read (logical block order) ----
        newer_records = self._read_level_records(newer, "newer")
        older_records = self._read_level_records(older, "older")

        # ---- phase B: trusted two-way merge, newer record wins ---------------
        merged, discarded = self._merge_sorted(newer_records, older_records)

        # ---- phase C-E: fresh canonical output, canonicality check, output PGM
        # The output LevelId is reserved BEFORE any block materialization (see
        # decisions/0006 §9): a `Level` entry exists from the first moment of
        # materialization, so an ordinary Python exception raised part-way through
        # is still cleanable and can never leave a partially built output in the
        # active registry.
        output_level_id = engine.create_level()
        try:
            engine.build_level(merged, level_id=output_level_id)
            self._assert_canonical_output(merged, output_level_id)
            engine.build_level_pgm(output_level_id)  # accepted G1-B1 path
        except Exception:
            # Best-effort: never advertise a partial output as a replacement
            # level.  The two input levels are still untouched at this point and
            # therefore remain active and queryable.
            self._discard_partial_output(output_level_id)
            raise

        # ---- phase F: retirement, only after successful publication ----------
        engine.retire_level(newer_level_id)
        engine.retire_level(older_level_id)

        return MergeResult(
            output_level_id=output_level_id,
            newer_level_id=newer_level_id,
            older_level_id=older_level_id,
            newer_record_count=len(newer_records),
            older_record_count=len(older_records),
            output_record_count=len(merged),
            discarded_older_record_count=discarded,
        )

    # ----------------------------------------------------------------- helpers

    def _discard_partial_output(self, output_level_id: LevelId) -> None:
        """Best-effort removal of a reserved or partially built output level.

        Guarantees the publication boundary of decisions/0006 §9: after an
        ordinary caught Python exception during output materialization, the
        reserved level must not remain an **active** level.  Its blocks are
        cleared from storage, unassigned from the mapping, its PGM metadata (if
        any) is dropped, and its registry entry is removed.

        The documented limitation is unchanged: an exception at an awkward point
        inside a storage/mapping mutation can still leave unreachable orphan slot
        contents, and G1-D claims no transaction/crash consistency.
        """
        engine = self.engine
        if not engine.is_active_level(output_level_id):
            return
        try:
            engine.retire_level(output_level_id)
        except Exception:  # pragma: no cover - cleanup is best effort
            pass

    def _read_level_records(self, level: "Level", role: str) -> list[Record]:
        """Read every block of one input level in logical block order.

        Uses the normal ``read_block`` path (mapping -> storage READ), so each
        block contributes exactly one trace event.  The records are checked for
        the full canonical per-level invariant while flattening — an empty level
        is legal, every non-final block must be full, the final block must be
        non-empty and legal, and keys must be strictly increasing across the
        level.  All checks use data that was already read, so they cost zero
        additional I/O and add no trace events, and a malformed input fails here,
        before any output is published or any input is retired.

        ``validate_level`` is deliberately NOT used: it reads every block again
        through the normal path and would add an undocumented READ pass.
        """
        engine = self.engine
        capacity = engine.config.block_capacity
        block_ids = level.block_ids
        records: list[Record] = []
        previous: Optional[RecordKey] = None

        if not block_ids:
            return records  # empty level: legal merge input, zero access

        last_index = len(block_ids) - 1
        for index, block_id in enumerate(block_ids):
            block = engine.read_block(block_id)  # mapping lookup + READ trace

            if block.is_empty:
                raise MergeInputError(
                    f"{role} level {level.level_id}: block {block_id} is empty; "
                    "refusing to merge a non-canonical level"
                )
            if block.capacity != capacity:
                raise MergeInputError(
                    f"{role} level {level.level_id}: block {block_id} has capacity "
                    f"{block.capacity} but the engine block capacity is {capacity}; "
                    "refusing to merge a non-canonical level"
                )
            if index != last_index and not block.is_full:
                raise MergeInputError(
                    f"{role} level {level.level_id}: non-final block {block_id} is "
                    f"not full (size {block.size} of capacity {capacity}); refusing "
                    "to merge a non-canonically packed level"
                )
            if index == last_index and not 1 <= block.size <= capacity:
                raise MergeInputError(
                    f"{role} level {level.level_id}: final block {block_id} has an "
                    f"illegal size {block.size} (capacity {capacity}); refusing to "
                    "merge a non-canonical level"
                )

            for record in block.records:
                if previous is not None and record.key <= previous:
                    raise MergeInputError(
                        f"{role} level {level.level_id}: keys are not strictly "
                        f"increasing in logical block order "
                        f"({previous} -> {record.key}); refusing to merge a "
                        "non-canonical level"
                    )
                records.append(record)
                previous = record.key
        return records

    @staticmethod
    def _merge_sorted(
        newer: list[Record], older: list[Record]
    ) -> tuple[list[Record], int]:
        """Two-way sorted merge by ``RecordKey``; newer record wins ties.

        Returns ``(merged_records, discarded_older_record_count)``.  The output is
        globally strictly increasing by key with no duplicate keys, and no
        tombstone, version, timestamp, or conflict metadata is introduced.
        """
        merged: list[Record] = []
        discarded = 0
        i = j = 0
        while i < len(newer) and j < len(older):
            newer_key = newer[i].key
            older_key = older[j].key
            if newer_key < older_key:
                merged.append(newer[i])
                i += 1
            elif older_key < newer_key:
                merged.append(older[j])
                j += 1
            else:  # duplicate key across the two inputs: newer wins
                merged.append(newer[i])
                i += 1
                j += 1
                discarded += 1
        merged.extend(newer[i:])
        merged.extend(older[j:])
        return merged, discarded

    def _assert_canonical_output(
        self, merged: list[Record], output_level_id: LevelId
    ) -> None:
        """Validate output canonicality from trusted construction state.

        Deliberately does NOT call ``validate_level``: that validator reads every
        block through the normal path and would add an undocumented READ pass to
        the frozen merge trace (decisions/0006 §11).  Instead the merged record
        stream (already strictly increasing by key) and the deterministic
        canonical partition count are checked against the freshly built level.
        """
        engine = self.engine
        capacity = engine.config.block_capacity

        for left, right in zip(merged, merged[1:]):
            if left.key >= right.key:
                raise MergeInputError(
                    "merged output keys are not strictly increasing "
                    f"({left.key} -> {right.key})"
                )

        expected_blocks = (
            0 if not merged else (len(merged) + capacity - 1) // capacity
        )
        actual_blocks = len(engine.level(output_level_id).block_ids)
        if actual_blocks != expected_blocks:
            raise MergeInputError(
                f"merged output level {output_level_id} has {actual_blocks} "
                f"blocks but canonical packing of {len(merged)} records at "
                f"capacity {capacity} requires {expected_blocks}"
            )
