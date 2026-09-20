"""G5-A: resumable bounded-state logical merge core + step-invariant incremental PGM.

This module implements **only** the trusted, resumable *logical* merge of two
internally sorted logical runs, together with an incremental PGM that consumes the
exact output key stream online.  It is the logical core that the later
de-amortized merge (G5-B) and the PGM-Guided work (G6) will drive.

Frozen contract (Issue #21, `decisions/0014-g5a-incremental-logical-merge.md`):

```text
A = source L        (newer)      B = target L+1   (older)
C = newer-wins sorted union(A, B)
source only -> source value      target only -> target value
same key    -> source value (newer wins), consume BOTH inputs, ONE output decision
```

The merge identity is frozen: only ``source = L -> target = L + 1`` is a G5-A job.
Same-level, reverse and gapped requests are refused at construction — before any
output, cursor movement or PGM key — exactly like the accepted G4 direction gate.
Record keys are arbitrary integers (negative, zero, positive): the accepted
``RecordKey``/G4 domain imposes no non-negativity.

* the state machine is **resumable**: ``begin -> advance(q) -> ... -> done -> finalize``
  and the result is independent of where the caller places step boundaries;
* ``advance(q)`` takes a positive integer **output-decision budget** — one decision
  emits at most one logical output record, so an ``advance(q)`` call emits at most
  ``q`` records and returns the actually achieved smaller progress at EOF;
* output is packed canonically exactly as the target level would be packed: at most
  **one** not-yet-complete logical output block is retained; a block that becomes
  full is emitted immediately (returned in the step payload and/or handed to an
  optional sink) and its records are dropped from job working state; the final
  partial block is emitted exactly once at completion; every emitted block carries a
  monotonically increasing **logical output block rank** — no ``SlotId`` is ever
  assigned here;
* the incremental PGM consumes the stream as the records are decided, and

  ```text
  IncrementalPgmBuilder.finalize() == build_batch_pgm(tuple(all_output_keys), epsilon)
  ```

  for every legal input, independent of ``advance()`` boundaries: a merge-step
  endpoint is **not** a PGM-segment endpoint.  The builder reuses the frozen
  canonical OPLM of :mod:`enhanced_letindex.pgm` (imported, not reimplemented), keeps
  the live OPLM/hull state across calls and blocks, and never restarts segmentation
  at a call or block boundary.  ``IncrementalPgmBuilder`` is public as a standalone
  component, but a job never hands out its owned instance: the job exposes only the
  immutable :class:`IncrementalPgmState` snapshot, so no caller can inject a phantom
  key into (or reset) the PGM that the merge stream owns.

Explicitly **out of scope** (Issue #21 §2) and deliberately absent from this module:
any REE READ/WRITE event, physical-slot scheduling or windowed input scan, fresh
output-region allocation or WRITE, ``MERGE`` observation change, structural
publication / ``structure_version`` increment, epoch/PRP reset, final shuffle or
writeback, G5-B/G6 mechanism, ORAM/BORPStream/CacheShuffle/SWAT, attack or leakage
metric.  G4 (``protected_merge``) stays the accepted physical/security oracle; this
module neither calls nor rewrites it.

Bounded-state claim (§9): the *incremental working state* of a job is

```text
O(1) cursors / current records
+ <= one partial output block
+ canonical incremental-PGM active state (live OPLM hull + counters)
+ finalized PGM segment metadata
```

The job never accumulates emitted records, completed blocks, per-step history or
per-record history.  It does hold the two immutable input run views supplied by the
caller; bounded *REE* buffering of those inputs is a G5-B question and is **not**
claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .identifiers import RecordKey
from .pgm import (
    BatchPgmIndex,
    _OptimalPiecewiseLinearModel,
    _pgm_segment_from_geometry,
)

__all__ = [
    "IncrementalMergeError",
    "IncrementalMergeConfig",
    "LogicalRunView",
    "OutputBlock",
    "IncrementalPgmState",
    "IncrementalMergeState",
    "IncrementalMergeStep",
    "IncrementalMergeResult",
    "IncrementalPgmBuilder",
    "IncrementalMergeJob",
]

#: The frozen post-DONE behaviour of ``advance()``: a deterministic no-op step.
POST_DONE_ADVANCE_POLICY = "no-op"

#: The frozen ``finalize()`` behaviour: idempotent (the same result object is returned
#: for every call after DONE).
FINALIZE_POLICY = "idempotent"


class IncrementalMergeError(Exception):
    """A G5-A request, input or state transition violates the frozen contract."""


# ---------------------------------------------------------------------------
# configuration and inputs
# ---------------------------------------------------------------------------


def _require_int(value, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IncrementalMergeError(
            f"{name} must be a plain int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise IncrementalMergeError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class IncrementalMergeConfig:
    """The frozen packing/geometry parameters of one logical merge.

    ``items_per_block`` is the target level's items-per-block and ``epsilon`` the
    accepted canonical PGM epsilon of the frozen PGM implementation.
    """

    items_per_block: int
    epsilon: int

    def __post_init__(self) -> None:
        items_per_block = _require_int(self.items_per_block, "items_per_block", minimum=1)
        epsilon = _require_int(self.epsilon, "epsilon", minimum=0)
        object.__setattr__(self, "items_per_block", items_per_block)
        object.__setattr__(self, "epsilon", epsilon)


@dataclass(frozen=True)
class LogicalRunView:
    """An immutable, strictly key-sorted logical run supplied by the caller.

    ``source`` (level ``L``) is the **newer** run and ``target`` (level ``L+1``) the
    **older** one.  Keys are arbitrary integers (negative, zero and positive are all
    legal — the accepted ``RecordKey``/G4 domain imposes no non-negativity), and they
    must be strictly increasing; values are opaque trusted payloads.  A malformed run
    is rejected at construction, so no ``advance()`` call can ever produce output from
    a malformed input.
    """

    level: int
    keys: Tuple[RecordKey, ...]
    values: Tuple[object, ...]

    def __post_init__(self) -> None:
        level = _require_int(self.level, "level", minimum=0)
        keys = tuple(self.keys)
        values = tuple(self.values)
        if len(keys) != len(values):
            raise IncrementalMergeError(
                f"logical run {level} has {len(keys)} keys but {len(values)} values"
            )
        for key in keys:
            if not isinstance(key, RecordKey):
                raise IncrementalMergeError(
                    f"logical run {level} keys must be RecordKey, got {type(key).__name__}"
                )
        for previous, current in zip(keys, keys[1:]):
            if current <= previous:
                raise IncrementalMergeError(
                    f"the input run of level {level} must be strictly sorted by key, "
                    f"got {current.value} after {previous.value}"
                )
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_records(cls, level: int, records: Sequence[Tuple[int, object]]) -> "LogicalRunView":
        """Build a view from ``(key_value, value)`` pairs."""
        keys = []
        values = []
        for record in records:
            try:
                key_value, value = record
            except (TypeError, ValueError):
                raise IncrementalMergeError(
                    f"a logical record must be a (key, value) pair, got {record!r}"
                ) from None
            keys.append(RecordKey(_require_int(key_value, "record key")))
            values.append(value)
        return cls(level=level, keys=tuple(keys), values=tuple(values))

    def records(self) -> Tuple[Tuple[int, object], ...]:
        """``(key_value, value)`` pairs of the run (evaluator convenience)."""
        return tuple(
            (key.value, value) for key, value in zip(self.keys, self.values)
        )

    @property
    def item_count(self) -> int:
        return len(self.keys)


# ---------------------------------------------------------------------------
# emitted output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputBlock:
    """One canonically packed logical output block (no physical slot is assigned).

    ``rank`` is the monotonically increasing logical output block rank (``0, 1, 2,
    ...``) and is part of the payload so that G5-B/G6 can address the stream without
    re-deriving the packing.
    """

    rank: int
    keys: Tuple[RecordKey, ...]
    values: Tuple[object, ...]

    @property
    def item_count(self) -> int:
        return len(self.keys)

    def records(self) -> Tuple[Tuple[int, object], ...]:
        return tuple(
            (key.value, value) for key, value in zip(self.keys, self.values)
        )


@dataclass(frozen=True)
class IncrementalPgmState:
    """An immutable, read-only snapshot of the job-owned incremental PGM state.

    The merge job **never** hands out its live mutable
    :class:`IncrementalPgmBuilder`: because ``add_key()`` is public, an owned instance
    would let a caller inject a phantom key between ``advance()`` calls and invalidate
    the central invariant ``final PGM == build_batch_pgm(exact merged output key
    stream)``.  This frozen snapshot is the only public observation surface, and it
    carries the diagnostics an evaluator needs (record count, segment start ranks,
    active segment span/point count, whether the last call closed a segment).
    """

    epsilon: int
    record_count: int
    segment_count: int
    closed_segment_start_ranks: Tuple[int, ...]
    segment_start_ranks: Tuple[int, ...]
    active_segment_start_rank: int
    active_segment_span: Tuple[int, int]
    active_point_count: int
    max_active_point_count: int
    closed_in_last_add: int
    retained_keys: int


@dataclass(frozen=True)
class IncrementalMergeState:
    """A frozen snapshot of the resumable job state.

    ``source_cursor``/``target_cursor`` are the input cursors ``i``/``j``, and
    ``output_records`` is the number of output decisions performed so far ``r``.
    """

    source_cursor: int
    target_cursor: int
    output_records: int
    emitted_blocks: int
    pending_items: int
    done: bool


@dataclass(frozen=True)
class IncrementalMergeStep:
    """The result of one ``advance(q)`` call.

    ``blocks`` are the canonical blocks **completed by this call**, in rank order;
    the job does not retain them afterwards.
    """

    decisions: int
    blocks: Tuple[OutputBlock, ...]
    state: IncrementalMergeState
    done: bool

    #: Records the caller can newly observe in this call's blocks.  A block completed by a
    #: call may contain records that were decided in earlier calls, so this is not the
    #: decision count of the call (see :attr:`decisions`).
    @property
    def emitted_records(self) -> int:
        return sum(block.item_count for block in self.blocks)


@dataclass(frozen=True)
class IncrementalMergeResult:
    """The finalize() result: the completed logical merge, without the records.

    ``pgm`` is the exact incremental PGM over the full output key stream, identical to
    ``build_batch_pgm(tuple(all_output_keys), epsilon)``.
    """

    source_level: int
    target_level: int
    record_count: int
    block_count: int
    duplicate_count: int
    first_key: Optional[RecordKey]
    last_key: Optional[RecordKey]
    pgm: BatchPgmIndex
    state: IncrementalMergeState
    pgm_segment_start_ranks: Tuple[int, ...]


# ---------------------------------------------------------------------------
# incremental PGM
# ---------------------------------------------------------------------------


class IncrementalPgmBuilder:
    """Online canonical PGM over the output key stream (G5-A §7).

    The builder drives the **frozen** canonical OPLM of
    :mod:`enhanced_letindex.pgm` — the very class/machinery used by
    ``make_segmentation``/``build_batch_pgm`` — one point at a time, so the resulting
    :class:`BatchPgmIndex` is exactly the batch result.  It keeps only the live OPLM
    hull state, the closed segment metadata and O(1) counters: the full output key
    stream is never retained.
    """

    def __init__(self, epsilon: int) -> None:
        self._epsilon = _require_int(epsilon, "epsilon", minimum=0)
        self._oplm = _OptimalPiecewiseLinearModel(self._epsilon)
        self._segments: list = []
        self._closed_ranks: list = []
        self._current_start_rank = 0
        self._count = 0
        self._last_key: Optional[RecordKey] = None
        self._first_key: Optional[RecordKey] = None
        self._max_active_points = 0
        self._closed_in_last_add = 0

    # -- feeding ------------------------------------------------------------

    def add_key(self, key: RecordKey) -> None:
        """Consume the next output key (must be strictly greater than the last)."""
        if not isinstance(key, RecordKey):
            raise IncrementalMergeError(
                f"incremental PGM keys must be RecordKey, got {type(key).__name__}"
            )
        if self._last_key is not None and key <= self._last_key:
            raise IncrementalMergeError(
                "the incremental PGM input keys must be strictly increasing, got "
                f"{key.value} after {self._last_key.value}"
            )
        self._closed_in_last_add = 0
        rank = self._count
        if not self._oplm.add_point(key.value, rank):
            # exactly the closing step of ``pgm.make_segmentation``: freeze the geometry
            # of the segment that just ended (ranks ..rank-1, ending at the previously
            # accepted key) and restart the model *at this same point*, so segmentation
            # continues across the boundary instead of restarting there
            self._segments.append(
                _pgm_segment_from_geometry(
                    self._oplm.get_segment(), self._current_start_rank,
                    end_rank=rank - 1, end_key=self._last_key,
                )
            )
            self._closed_ranks.append(self._current_start_rank)
            self._current_start_rank = rank
            self._oplm._reset()
            if not self._oplm.add_point(key.value, rank):    # pragma: no cover - guard
                raise IncrementalMergeError("the canonical OPLM failed to restart")
            self._closed_in_last_add = 1
        if self._first_key is None:
            self._first_key = key
        self._last_key = key
        self._count += 1
        points = self._oplm.points_in_hull
        if points > self._max_active_points:
            self._max_active_points = points

    def add_keys(self, keys: Sequence[RecordKey]) -> None:
        for key in keys:
            self.add_key(key)

    # -- observability ------------------------------------------------------

    @property
    def epsilon(self) -> int:
        return self._epsilon

    @property
    def record_count(self) -> int:
        """Number of keys consumed so far."""
        return self._count

    @property
    def segment_count(self) -> int:
        """Closed segments plus the still-open active segment."""
        return len(self._segments) + (1 if self._count else 0)

    @property
    def closed_segment_start_ranks(self) -> Tuple[int, ...]:
        return tuple(self._closed_ranks)

    @property
    def active_segment_start_rank(self) -> int:
        return self._current_start_rank

    @property
    def active_segment_span(self) -> Tuple[int, int]:
        """``(first_rank, last_rank)`` of the still-open segment."""
        if not self._count:
            return (0, -1)
        return (self._current_start_rank, self._count - 1)

    @property
    def active_point_count(self) -> int:
        """Points currently inside the live OPLM hull (canonical active state)."""
        return self._oplm.points_in_hull

    @property
    def max_active_point_count(self) -> int:
        return self._max_active_points

    @property
    def segment_start_ranks(self) -> Tuple[int, ...]:
        """Start ranks of all segments, as they would appear in the batch result."""
        return tuple(self._closed_ranks) + ((self._current_start_rank,) if self._count else ())

    @property
    def closed_in_last_add(self) -> int:
        """``1`` if the most recent ``add_key`` closed a segment, else ``0``."""
        return self._closed_in_last_add

    @property
    def retained_keys(self) -> int:
        """The builder never retains the key stream: always ``0``."""
        return 0

    # -- finalize -----------------------------------------------------------

    def snapshot(self) -> "IncrementalPgmState":
        """An immutable read-only view of the current canonical PGM state."""
        return IncrementalPgmState(
            epsilon=self._epsilon,
            record_count=self._count,
            segment_count=self.segment_count,
            closed_segment_start_ranks=self.closed_segment_start_ranks,
            segment_start_ranks=self.segment_start_ranks,
            active_segment_start_rank=self._current_start_rank,
            active_segment_span=self.active_segment_span,
            active_point_count=self.active_point_count,
            max_active_point_count=self._max_active_points,
            closed_in_last_add=self._closed_in_last_add,
            retained_keys=self.retained_keys,
        )

    def finalize(self) -> BatchPgmIndex:
        """The exact batch-equivalent :class:`BatchPgmIndex` of the consumed stream."""
        if not self._count:
            return BatchPgmIndex(self._epsilon, 0, None, None, ())
        segments = tuple(self._segments) + (
            _pgm_segment_from_geometry(
                self._oplm.get_segment(), self._current_start_rank,
                end_rank=self._count - 1, end_key=self._last_key,
            ),
        )
        assert segments and segments[0].start_rank == 0
        return BatchPgmIndex(
            self._epsilon,
            self._count,
            self._first_key,
            self._last_key,
            segments,
        )


# ---------------------------------------------------------------------------
# the resumable merge job
# ---------------------------------------------------------------------------


class IncrementalMergeJob:
    """The resumable newer-wins logical merge state machine (G5-A §4-§9).

    The job is deterministic, holds bounded merge working state, and never mutates a
    ``DefendedIndex`` or touches storage.
    """

    __slots__ = (
        "_config",
        "_source",
        "_target",
        "_source_cursor",
        "_target_cursor",
        "_output_records",
        "_emitted_blocks",
        "_duplicate_count",
        "_pending_keys",
        "_pending_values",
        "_done",
        "_step_counter",
        "_advance_calls",
        "_max_pending_items",
        "_pgm",
        "_finalized",
        "_sink",
    )

    def __init__(
        self,
        config: IncrementalMergeConfig,
        source: LogicalRunView,
        target: LogicalRunView,
        *,
        sink: Optional[Callable[[OutputBlock], None]] = None,
    ) -> None:
        if not isinstance(config, IncrementalMergeConfig):
            raise IncrementalMergeError(
                f"a job needs an IncrementalMergeConfig, got {type(config).__name__}"
            )
        for view, name in ((source, "source"), (target, "target")):
            if not isinstance(view, LogicalRunView):
                raise IncrementalMergeError(
                    f"the {name} of a merge must be a LogicalRunView, got "
                    f"{type(view).__name__}"
                )
        if source.level == target.level:
            raise IncrementalMergeError(
                f"a G5-A merge needs a newer source L and an older target L + 1, got the "
                f"same level twice (L = {source.level}); a same-level merge is refused "
                "before any output is produced"
            )
        if source.level > target.level:
            raise IncrementalMergeError(
                "a G5-A merge merges a newer source L into the older target L + 1; source "
                f"{source.level} -> target {target.level} is the reverse direction and is "
                "refused before any output is produced"
            )
        if target.level != source.level + 1:
            raise IncrementalMergeError(
                "a G5-A merge only merges numerically adjacent levels, got source "
                f"{source.level} and target {target.level} (a gap of "
                f"{target.level - source.level - 1}); a gapped merge is refused before "
                "any output is produced"
            )
        if sink is not None and not callable(sink):
            raise IncrementalMergeError("the optional sink must be callable")
        self._config = config
        self._source = source
        self._target = target
        self._source_cursor = 0
        self._target_cursor = 0
        self._output_records = 0
        self._emitted_blocks = 0
        self._duplicate_count = 0
        self._pending_keys: list = []
        self._pending_values: list = []
        self._done = False
        self._step_counter = 0
        self._advance_calls = 0
        self._max_pending_items = 0
        self._pgm = IncrementalPgmBuilder(config.epsilon)
        self._finalized: Optional[IncrementalMergeResult] = None
        self._sink = sink

    # -- construction -------------------------------------------------------

    @classmethod
    def begin(
        cls,
        source: LogicalRunView,
        target: LogicalRunView,
        *,
        items_per_block: int,
        epsilon: int,
        sink: Optional[Callable[[OutputBlock], None]] = None,
    ) -> "IncrementalMergeJob":
        """Start a resumable merge of ``source`` (newer) into ``target`` (older)."""
        return cls(
            IncrementalMergeConfig(items_per_block=items_per_block, epsilon=epsilon),
            source,
            target,
            sink=sink,
        )

    # -- accessors ----------------------------------------------------------

    @property
    def config(self) -> IncrementalMergeConfig:
        return self._config

    @property
    def source(self) -> LogicalRunView:
        return self._source

    @property
    def target(self) -> LogicalRunView:
        return self._target

    @property
    def state(self) -> IncrementalMergeState:
        return IncrementalMergeState(
            source_cursor=self._source_cursor,
            target_cursor=self._target_cursor,
            output_records=self._output_records,
            emitted_blocks=self._emitted_blocks,
            pending_items=len(self._pending_keys),
            done=self._done,
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def pgm_state(self) -> IncrementalPgmState:
        """Read-only diagnostics of the job-owned incremental PGM (never the builder).

        The live mutable :class:`IncrementalPgmBuilder` stays private to the job: a
        caller can observe the canonical PGM state through this frozen snapshot but
        cannot add a key to it, reset it or otherwise desynchronize it from the merge
        stream.
        """
        return self._pgm.snapshot()

    def working_state_report(self) -> dict:
        """Evaluator-only evidence of the bounded merge working state (§9).

        Every value is O(1) or explicitly allowed (the canonical active PGM state and
        the finalized PGM segment metadata).  The job owns no completed blocks and no
        step/record history.  ``max_partial_block_items`` is the **true instantaneous
        peak** of the pending buffer, recorded at append time before any flush, so for
        ``items_per_block = n`` it reports ``n`` (never ``n - 1``).
        """
        return {
            "source_cursor": self._source_cursor,
            "target_cursor": self._target_cursor,
            "retained_output_records": len(self._pending_keys),
            "retained_output_values": len(self._pending_values),
            "partial_block_items": len(self._pending_keys),
            "max_partial_block_items": self._max_pending_items,
            "retained_output_blocks": 0,
            "retained_step_objects": 0,
            "retained_record_history": 0,
            "step_counter": self._step_counter,
            "advance_calls": self._advance_calls,
            "pgm_closed_segments": len(self._pgm.closed_segment_start_ranks),
            "pgm_active_points": self._pgm.active_point_count,
            "pgm_active_points_max": self._pgm.max_active_point_count,
            "pgm_retained_keys": self._pgm.retained_keys,
            "items_per_block": self._config.items_per_block,
        }

    # -- the quantum --------------------------------------------------------

    def advance(self, q: int) -> IncrementalMergeStep:
        """Perform at most ``q`` output decisions and return the newly completed blocks.

        Post-DONE behaviour is the frozen :data:`POST_DONE_ADVANCE_POLICY` no-op: a
        deterministic step with ``decisions == 0``, no blocks and ``done is True``.
        """
        if isinstance(q, bool) or not isinstance(q, int):
            raise IncrementalMergeError(
                f"the advance budget q must be a plain int, got {type(q).__name__}"
            )
        if q <= 0:
            raise IncrementalMergeError(f"the advance budget q must be positive, got {q}")
        if self._done:
            # the frozen POST_DONE_ADVANCE_POLICY: a *real* no-op — no job-owned field is
            # mutated, not even the call counter the working-state report exposes
            return IncrementalMergeStep(0, (), self.state, True)

        self._advance_calls += 1
        self._step_counter += 1
        decisions = 0
        blocks: list = []
        source_keys, source_values = self._source.keys, self._source.values
        target_keys, target_values = self._target.keys, self._target.values

        while decisions < q:
            if (
                self._source_cursor < len(source_keys)
                and self._target_cursor < len(target_keys)
            ):
                source_key = source_keys[self._source_cursor]
                target_key = target_keys[self._target_cursor]
                if source_key == target_key:
                    # source is newer: emit the source record, consume BOTH inputs, and
                    # count the pair as a single output decision
                    self._emit(
                        source_key, source_values[self._source_cursor], blocks
                    )
                    self._source_cursor += 1
                    self._target_cursor += 1
                    self._duplicate_count += 1
                elif source_key < target_key:
                    self._emit(
                        source_key, source_values[self._source_cursor], blocks
                    )
                    self._source_cursor += 1
                else:
                    self._emit(
                        target_key, target_values[self._target_cursor], blocks
                    )
                    self._target_cursor += 1
            elif self._source_cursor < len(source_keys):
                self._emit(
                    source_keys[self._source_cursor],
                    source_values[self._source_cursor],
                    blocks,
                )
                self._source_cursor += 1
            elif self._target_cursor < len(target_keys):
                self._emit(
                    target_keys[self._target_cursor],
                    target_values[self._target_cursor],
                    blocks,
                )
                self._target_cursor += 1
            else:
                self._done = True
                break
            decisions += 1

        if self._source_cursor >= len(source_keys) and self._target_cursor >= len(target_keys):
            self._done = True
        if self._done and self._pending_keys:
            blocks.append(self._flush_block())
        return IncrementalMergeStep(decisions, tuple(blocks), self.state, self._done)

    # -- internals ----------------------------------------------------------

    def _emit(self, key: RecordKey, value: object, blocks: list) -> None:
        """Append one decided output record; emit a full canonical block at once."""
        self._pending_keys.append(key)
        self._pending_values.append(value)
        # the *instantaneous* peak of the pending buffer: recorded before a possible
        # flush, so for items_per_block = n it correctly reports n (a between-call
        # residual would only ever report n - 1)
        pending = len(self._pending_keys)
        if pending > self._max_pending_items:
            self._max_pending_items = pending
        # the exact output key stream drives the incremental PGM, per record: step and
        # block boundaries are invisible to the canonical segmentation
        self._pgm.add_key(key)
        self._output_records += 1
        if pending == self._config.items_per_block:
            blocks.append(self._flush_block())

    def _flush_block(self) -> OutputBlock:
        """Move the pending records into a canonical block and drop them from the job."""
        block = OutputBlock(
            rank=self._emitted_blocks,
            keys=tuple(self._pending_keys),
            values=tuple(self._pending_values),
        )
        self._pending_keys.clear()
        self._pending_values.clear()
        self._emitted_blocks += 1
        if self._sink is not None:
            self._sink(block)
        return block

    # -- finalize -----------------------------------------------------------

    def finalize(self) -> IncrementalMergeResult:
        """Return the completed merge; refused before DONE, idempotent after.

        The returned :class:`IncrementalMergeResult` carries the exact final PGM
        (``build_batch_pgm`` equivalent) and O(1) counters — never the record stream.
        """
        if not self._done:
            raise IncrementalMergeError(
                "finalize() requires a completed merge: the job is not DONE yet "
                f"(source cursor {self._source_cursor}, target cursor "
                f"{self._target_cursor}, pending items {len(self._pending_keys)})"
            )
        if self._finalized is not None:
            return self._finalized
        pgm = self._pgm.finalize()
        self._finalized = IncrementalMergeResult(
            source_level=self._source.level,
            target_level=self._target.level,
            record_count=self._output_records,
            block_count=self._emitted_blocks,
            duplicate_count=self._duplicate_count,
            first_key=pgm.first_key,
            last_key=pgm.last_key,
            pgm=pgm,
            state=self.state,
            pgm_segment_start_ranks=self._pgm.segment_start_ranks,
        )
        return self._finalized
