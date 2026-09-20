"""TrustedEngine: the minimal G0 trusted-side facade."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from .block import Block
from .config import Config
from .identifiers import BlockId, LevelId, RecordKey, SlotId
from .level import Level
from .mapping import DuplicatePhysicalSlotError, LogicalPhysicalMapping
from .record import Record
from .storage import StorageConsistencyError, UntrustedStorage
from .trace import TraceCollector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pgm import BatchPgmIndex, SearchResult

__all__ = [
    "TrustedEngine",
    "SingleLevelLookupResult",
    "MultiLevelLookupResult",
    "MergeResult",
    "DuplicateLevelInPlanError",
    "UnknownLevelError",
    "RetiredLevelError",
]


class DuplicateLevelInPlanError(ValueError):
    """An ordered multi-level query plan lists the same ``LevelId`` twice.

    Duplicate levels in one plan are invalid: they would both add query traffic
    and make "first hit wins" ambiguous.  Detected during preflight, before any
    physical query read.
    """


class UnknownLevelError(KeyError):
    """A multi-level query plan references a ``LevelId`` this engine does not own.

    Subclasses :class:`KeyError` so that a trusted-state miss stays detectable
    exactly like the other loud G0/G1-B2 lookup failures.
    """


class RetiredLevelError(KeyError):
    """A retired level was used after its merge publication.

    Retired levels are removed from the active registry by
    :meth:`TrustedEngine.retire_level`; they are permanently unusable.  Subclasses
    :class:`KeyError` so existing "trusted-state miss" handling keeps working.
    """


@dataclass(frozen=True)
class SingleLevelLookupResult:
    """The functional result of one single-level physical lookup (G1-B2).

    Exposes only membership semantics: no SlotId, physical schedule, or
    internal storage address is leaked through the functional return API.
    ``AccessTrace`` remains the sole observation source.
    """

    found: bool
    record: Optional[Record]
    lower_bound_rank: int
    search_result: "SearchResult"


@dataclass(frozen=True)
class MultiLevelLookupResult:
    """The functional result of one multi-level query (G1-C).

    ``searched_level_ids`` lists, in caller-supplied query order, exactly the
    non-empty levels on which a G1-B2 lookup actually ran.  Empty levels are
    skipped and absent.  On a HIT the tuple ends with ``hit_level_id``; on a
    miss across every searched level ``hit_level_id`` is ``None`` and the tuple
    contains every non-empty level of the plan.

    Like :class:`SingleLevelLookupResult` this exposes no ``SlotId``, physical
    schedule, or ``BlockId``; the ``AccessTrace`` is the only observation
    surface.
    """

    found: bool
    record: Optional[Record]
    hit_level_id: Optional[LevelId]
    searched_level_ids: tuple[LevelId, ...]


@dataclass(frozen=True)
class MergeResult:
    """The functional result of one blocking two-level merge (G1-D).

    Exposes logical identifiers and record counts only: no ``SlotId``, no
    physical schedule, and no ``BlockId``.  ``discarded_older_record_count``
    counts older records dropped because the newer level carried the same key.
    """

    output_level_id: LevelId
    newer_level_id: LevelId
    older_level_id: LevelId
    newer_record_count: int
    older_record_count: int
    output_record_count: int
    discarded_older_record_count: int


class TrustedEngine:
    """Owns configuration, logical levels, the logical-to-physical mapping,
    and a reference to untrusted storage.

    G0 provides only the minimal vertical slice requested by the bootstrap:

    1. create records;
    2. create a logical block;
    3. assign it a physical slot;
    4. store it;
    5. retrieve it as: logical block id -> mapping -> physical read;
    6. produce an observable READ trace.

    No stash, dummies, reshuffle, encryption, merge, or PGM behavior lives
    here yet.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config if config is not None else Config()
        self.mapping = LogicalPhysicalMapping()
        self.storage = UntrustedStorage()
        self._levels: dict[LevelId, Level] = {}
        self._pgm_indexes: dict[LevelId, "BatchPgmIndex"] = {}
        self._retired_level_ids: set[LevelId] = set()
        self._next_block_id = 0
        self._next_level_id = 0

    # ---------------------------------------------------------------- accessors

    @property
    def trace(self) -> TraceCollector:
        """The observable access trace (shared with the storage layer)."""
        return self.storage.trace

    # ---------------------------------------------------------------- id alloc

    def _new_block_id(self) -> BlockId:
        block_id = BlockId(self._next_block_id)
        self._next_block_id += 1
        return block_id

    def _new_level_id(self) -> LevelId:
        level_id = LevelId(self._next_level_id)
        self._next_level_id += 1
        return level_id

    # ---------------------------------------------------------------- levels

    def create_level(self) -> LevelId:
        """Create an empty logical level and return its id."""
        level = Level(self._new_level_id())
        self._levels[level.level_id] = level
        return level.level_id

    def build_level(
        self, records: Iterable[Record], level_id: Optional[LevelId] = None
    ) -> LevelId:
        """Construct a sorted logical Level from arbitrary records (G1-A).

        Normalizes the input (strict key sort, duplicate rejection), packs it
        into capacity-bounded blocks, allocates globally-unique logical
        BlockIds, materializes each block into :class:`UntrustedStorage` with
        a deterministic order-preserving physical placement, registers the
        mapping, and returns the level id.  See ``builder.LevelBuilder``.
        """
        from .builder import LevelBuilder

        return LevelBuilder(self).build(records, level_id=level_id)


    def level(self, level_id: LevelId) -> Level:
        """Return the :class:`Level` with the given id.

        Raises :class:`RetiredLevelError` for a level retired by a merge
        publication, and :class:`KeyError` when the id is unknown.
        """
        if level_id in self._retired_level_ids:
            raise RetiredLevelError(
                f"level {level_id} was retired by a merge publication and is no "
                "longer active"
            )
        return self._levels[level_id]

    def add_block_to_level(self, level_id: LevelId, block_id: BlockId) -> None:
        """Attach an existing logical block to a level.

        Invalidate any already-built PGM for the level: its data set has
        changed, so the pre-existing index is no longer authoritative.
        """
        self.level(level_id).add_block(block_id)
        self._pgm_indexes.pop(level_id, None)

    # ---------------------------------------------------------------- records

    def create_record(self, key: int, value: object) -> Record:
        """Construct a :class:`Record` from a plain integer key."""
        return Record(key=RecordKey(key), value=value)

    # ---------------------------------------------------------------- blocks

    def create_block(
        self, records: Iterable[Record], level_id: Optional[LevelId] = None
    ) -> BlockId:
        """Create a sorted logical block, place it in a fresh physical slot,
        register the mapping, and optionally attach it to a level.

        Returns the new logical block id.
        """
        block_id = self._new_block_id()
        block = Block.sorted_block(block_id, records, self.config.block_capacity)
        slot = self.storage.store(block)
        self.mapping.assign(block_id, slot)
        if level_id is not None:
            self.add_block_to_level(level_id, block_id)
        return block_id

    def read_block(self, block_id: BlockId) -> Block:
        """Retrieve a block via logical id -> mapping -> physical read.

        Emits exactly the storage layer's READ trace event for the resolved
        physical slot.
        """
        slot = self.mapping.lookup(block_id)
        return self.storage.read(slot)

    def physical_slot(self, block_id: BlockId) -> SlotId:
        """Return the physical slot currently hosting ``block_id``."""
        return self.mapping.lookup(block_id)

    def relocate_block(self, block_id: BlockId, new_slot: SlotId) -> None:
        """Move a logical block to a different physical slot, vacating the
        source slot.

        The logical block identity and its contents are unchanged; only the
        physical location (and hence the observable trace) changes.  For a
        move ``B0: P0 -> P5`` the emitted trace is exactly::

            READ  P0
            WRITE P5
            WRITE P0

        where the final ``WRITE P0`` vacates/overwrites the old slot so that
        every active block keeps exactly one current physical copy.

        This is a functional storage/correctness primitive only; it is NOT a
        privacy-preserving reshuffle primitive (the READ-old / WRITE-new /
        WRITE-old sequence directly reveals old-to-new physical
        correspondence).  See ``docs/g0-architecture.md``.

        The destination must be free in the mapping, the destination must not
        already hold a block in storage, and the block read back from the
        source slot must have the expected logical id.  Inconsistent
        mapping/storage state is reported loudly, never silently repaired.
        """
        old_slot = self.mapping.lookup(block_id)
        if old_slot == new_slot:
            return
        owner = self.mapping.slot_owner(new_slot)
        if owner is not None and owner != block_id:
            raise DuplicatePhysicalSlotError(
                f"physical slot {new_slot} is already assigned to "
                f"logical block {owner}"
            )
        if self.storage.contains(new_slot):
            raise StorageConsistencyError(
                f"physical slot {new_slot} holds a block but is not mapped; "
                f"refusing to overwrite it"
            )
        block = self.storage.read(old_slot)  # emits READ P0
        if block.block_id != block_id:
            raise StorageConsistencyError(
                f"physical slot {old_slot} holds block {block.block_id}, "
                f"expected {block_id}: mapping/storage inconsistent"
            )
        self.storage.write(new_slot, block)  # emits WRITE P5
        self.mapping.move(block_id, new_slot)
        self.storage.clear(old_slot)  # emits WRITE P0 (vacate source)

    # ---------------------------------------------------------------- PGM (G1-B1)

    def build_level_pgm(self, level_id: LevelId) -> "BatchPgmIndex":
        """Build the canonical batch PGM over an existing logical level.

        Walks the level's logical ``BlockId`` order, reading each block
        exactly once through the normal ``read_block`` path (mapping ->
        storage read), so construction emits exactly one trace-visible
        READ per block and no WRITE.  It flattens the records, assigns
        monotonic level-local record ranks, verifies keys are globally
        strictly increasing (fail loudly, never sort-repair), and builds
        the sequential OPLM index.  The epsilon must be explicitly
        configured (``config.pgm_epsilon`` non-``None``).

        The resulting index is stored as trusted state and returned.
        """
        from .pgm import PgmConfigurationError, build_batch_pgm

        epsilon = self.config.pgm_epsilon
        if epsilon is None:
            raise PgmConfigurationError(
                "pgm_epsilon is not configured; set Config(pgm_epsilon=<int>=0) "
                "before building a PGM"
            )

        level = self.level(level_id)
        keys: list[RecordKey] = []
        previous: Optional[RecordKey] = None
        for block_id in level.block_ids:
            block = self.read_block(block_id)  # one READ per block, trace-visible
            for record in block.records:
                key = record.key
                if previous is not None and key <= previous:
                    from .pgm import PgmError

                    raise PgmError(
                        f"level {level_id}: keys not globally strictly "
                        f"increasing across blocks ({previous} -> {key}); "
                        "refusing to build a PGM on a corrupted level"
                    )
                keys.append(key)
                previous = key

        index = build_batch_pgm(keys, epsilon)
        self._pgm_indexes[level_id] = index
        return index

    def pgm_index(self, level_id: LevelId) -> "BatchPgmIndex":
        """Return the built PGM for ``level_id``.

        Raises :class:`RetiredLevelError` when the level was retired by a merge
        publication, and :class:`KeyError` when no PGM has been built (or when it
        was invalidated by a later level mutation).
        """
        if level_id in self._retired_level_ids:
            raise RetiredLevelError(
                f"level {level_id} was retired by a merge publication; its PGM "
                "metadata no longer exists"
            )
        return self._pgm_indexes[level_id]

    def search_level_pgm(self, level_id: LevelId, key: RecordKey) -> "SearchResult":
        """Search the trusted PGM metadata for ``key``.

        Pure trusted-state operation: it performs no storage access and
        appends no trace event.
        """
        return self._pgm_indexes[level_id].search(key)

    # ------------------------------------------------------- lookup (G1-B2)

    def lookup_level(
        self, level_id: LevelId, key: RecordKey
    ) -> SingleLevelLookupResult:
        """Perform one single-level physical lookup (G1-B2).

        Requires an already-built ``BatchPgmIndex`` for ``level_id``; it
        never builds one implicitly, so the observation window of a query is
        never polluted by construction trace.  The lookup is:

          - SearchResult from trusted PGM metadata (0 trace events);
          - minimal covering logical block range from ``[lo, hi)``;
          - physical-order schedule: one READ per candidate, ascending
            current SlotId; no duplicate, no omission, no early stop;
          - restore logical rank order, then bisect over ``[lo, hi)`` and
            equality-check for an exact HIT/MISS.

        Fails loudly (``KeyError``) when the level has no built PGM.
        """
        level = self.level(level_id)
        pgm = self.pgm_index(level_id)  # KeyError if missing -> fail loudly

        sr: SearchResult = pgm.search(key)

        if sr.lo == sr.hi:
            return SingleLevelLookupResult(
                found=False, record=None, lower_bound_rank=sr.lo, search_result=sr
            )

        capacity = self.config.block_capacity
        block_ids = level.block_ids

        first = sr.lo // capacity
        last = (sr.hi - 1) // capacity

        # (logical_block_rank, block_id, slot_id) for every candidate block
        schedule = []
        for logical_rank in range(first, last + 1):
            block_id = block_ids[logical_rank]
            slot_id = self.mapping.lookup(block_id)
            schedule.append((logical_rank, block_id, slot_id))

        # Physical-order scheduling: ascending current SlotId.
        schedule.sort(key=lambda item: item[2].value)

        # Read every candidate exactly once, in physical order.
        fetched: dict[int, Block] = {}
        for logical_rank, block_id, _slot_id in schedule:
            fetched[logical_rank] = self.read_block(block_id)

        # Restore logical rank order; keep only records in [lo, hi).
        candidate_records: list[Record] = []
        for logical_rank in sorted(fetched):
            block = fetched[logical_rank]
            for offset, record in enumerate(block.records):
                global_rank = logical_rank * capacity + offset
                if sr.lo <= global_rank < sr.hi:
                    candidate_records.append(record)

        candidate_keys = [record.key for record in candidate_records]
        local_pos = bisect.bisect_left(candidate_keys, key)
        lb_rank = sr.lo + local_pos

        if local_pos < len(candidate_records) and candidate_records[local_pos].key == key:
            return SingleLevelLookupResult(
                found=True,
                record=candidate_records[local_pos],
                lower_bound_rank=lb_rank,
                search_result=sr,
            )

        return SingleLevelLookupResult(
            found=False, record=None, lower_bound_rank=lb_rank, search_result=sr
        )

    # -------------------------------------------------- multi-level lookup (G1-C)

    def lookup_levels(
        self, level_ids: Sequence[LevelId], key: RecordKey
    ) -> MultiLevelLookupResult:
        """Perform one multi-level query over an explicit ordered level plan (G1-C).

        ``level_ids`` is the caller-supplied **query-priority order**: index 0 is
        searched first (newest / highest precedence).  Search priority is never
        inferred from a ``LevelId`` numeric value, a physical ``SlotId``, a
        ``BlockId``, level size, creation order, or ``lsm_level_ratio``.

        Preflight (trusted-state only, zero trace events) completes before any
        physical query READ, in the frozen Decision 0005 §5 order:

        * phase A — duplicate detection over the **whole** plan: any repeated
          ``LevelId`` is rejected (:class:`DuplicateLevelInPlanError`) before any
          level resolution happens;
        * phase B — resolve every referenced level (:class:`UnknownLevelError`
          for ids this engine does not own), identify and skip empty levels (no
          PGM requirement, no storage access, no entry in the result's
          searched-level list), and verify that every remaining non-empty level
          already carries a current built PGM (``KeyError`` otherwise).  A PGM
          is never built implicitly, and there is no binary-search /
          small-level fallback in G1-C;
        * phase C — execute the accepted G1-B2 lookups.

        Because phase A completes over the entire plan before phase B resolves
        anything, a plan such as ``[unknown, unknown]`` fails as a duplicate plan
        rather than as an unknown level.

        Because all checks complete before execution, an invalid plan or a
        malformed later level can never leave a half-executed query trace.

        Execution then invokes the accepted G1-B2 :meth:`lookup_level` behaviour
        once per non-empty level, in caller order, and stops at the first HIT
        (hit-and-stop *between* levels only).  Each level's observable schedule is
        exactly Decision 0004: the minimal covering block set for ``[lo, hi)``
        read once each in ascending current ``SlotId``, with no ``±1`` neighbours,
        no fixed three-block padding, no dummy/cover reads, and no early stop
        inside that level's candidate set.  The G1-C trace is therefore exactly
        the concatenation of the per-level G1-B2 traces; G1-C itself emits no
        event of its own.

        Duplicate keys across levels resolve to the first HIT in the supplied
        order (LSM shadowing / newest-first semantics).
        """
        plan = tuple(level_ids)

        # ---- phase A: duplicate detection over the complete plan ----------
        seen: set[LevelId] = set()
        for level_id in plan:
            if level_id in seen:
                raise DuplicateLevelInPlanError(
                    f"multi-level query plan lists level {level_id} more than once"
                )
            seen.add(level_id)

        # ---- phase B: resolve levels, skip empties, verify PGMs ----------
        searchable: list[LevelId] = []
        for level_id in plan:
            level = self._levels.get(level_id)
            if level is None:
                if level_id in self._retired_level_ids:
                    raise RetiredLevelError(
                        f"multi-level query plan references retired level "
                        f"{level_id}; it was replaced by a merge publication"
                    )
                raise UnknownLevelError(
                    f"multi-level query plan references unknown level {level_id}"
                )
            if not level.block_ids:
                continue  # empty level: skipped, no PGM required, zero access
            self.pgm_index(level_id)  # KeyError if missing or invalidated
            searchable.append(level_id)

        # ---- phase C: exactly G1-B2 per level, in caller order -----------
        searched: list[LevelId] = []
        for level_id in searchable:
            result = self.lookup_level(level_id, key)
            searched.append(level_id)
            if result.found:
                return MultiLevelLookupResult(
                    found=True,
                    record=result.record,
                    hit_level_id=level_id,
                    searched_level_ids=tuple(searched),
                )

        return MultiLevelLookupResult(
            found=False,
            record=None,
            hit_level_id=None,
            searched_level_ids=tuple(searched),
        )

    # ------------------------------------------------- upsert / merge (G1-D)

    def is_active_level(self, level_id: LevelId) -> bool:
        """True when ``level_id`` is an active (non-retired) level of this engine."""
        return level_id in self._levels

    def active_level_ids(self) -> tuple[LevelId, ...]:
        """The ids of all currently active (non-retired) levels, sorted.

        Read-only trusted-state introspection intended for verification harnesses:
        no storage access, no trace event, and no behavioural change to any
        accepted G1-A–G1-D path.  Retired levels are never included, and the
        returned tuple is a snapshot, not a live view.
        """
        return tuple(sorted(self._levels))

    def create_update_level(self, records: Iterable[Record]) -> LevelId:
        """Publish one upsert batch as a fresh newest level (G1-D).

        The batch is normalized by key and must not contain duplicate keys
        (``Dataset.from_records`` rejects them before anything is published), and
        it must not be empty (``EmptyUpdateBatchError``).  Keys may overlap older
        levels: that is the normal update case, and G1-C resolves it by
        caller-supplied order (newest level first).

        The returned level is a normal canonical G1-A level with fresh
        ``LevelId``/``BlockId``s.  Its PGM is NOT built here: as everywhere else
        in this repository a ``BatchPgmIndex`` is built by an explicit
        ``build_level_pgm`` call, so the level becomes queryable through G1-C only
        after that explicit step.

        See ``decisions/0006-g1-d-blocking-merge.md``.
        """
        from .merge import LevelMerger

        return LevelMerger(self).create_update_level(records)

    def merge_levels_blocking(
        self, newer_level_id: LevelId, older_level_id: LevelId
    ) -> MergeResult:
        """Merge two levels by explicit caller-supplied precedence (G1-D).

        ``newer_level_id`` wins on duplicate keys; the direction is never inferred
        from ``LevelId`` values, sizes, creation order, or physical placement.
        Both inputs are read completely in logical block order (newer first), the
        merged records are materialized as a fresh canonical output level, the
        output PGM is built through the accepted G1-B1 batch path, and only then
        are both input levels retired.

        See ``decisions/0006-g1-d-blocking-merge.md``.
        """
        from .merge import LevelMerger

        return LevelMerger(self).merge_blocking(newer_level_id, older_level_id)

    def retire_level(self, level_id: LevelId) -> int:
        """Retire an active level and free its physical storage (G1-D baseline).

        For every block of the level, in logical block order:

        * the block's physical slot is cleared through ``UntrustedStorage.clear``,
          which is observable as exactly one ``WRITE`` trace event;
        * the logical block is unassigned from the active mapping (the bijection
          is re-validated after every removal).

        Then the level's PGM metadata is dropped, the level is removed from the
        active registry, and its id is recorded as retired so later
        ``level(...)`` / ``pgm_index(...)`` / G1-C plans fail clearly.

        Returns the number of retired blocks.  This is baseline state cleanup, NOT
        secure erasure: the observable trace still records which slots were
        vacated, and the adversary's knowledge of earlier accesses is unchanged.
        """
        level = self.level(level_id)
        block_ids = tuple(level.block_ids)
        for block_id in block_ids:
            slot = self.mapping.lookup(block_id)
            self.storage.clear(slot)  # observable WRITE (vacate)
            self.mapping.unassign(block_id)
        self._pgm_indexes.pop(level_id, None)
        del self._levels[level_id]
        self._retired_level_ids.add(level_id)
        return len(block_ids)
