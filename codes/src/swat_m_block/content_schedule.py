"""M3: content-bound SWAT-M-Block bin schedule.

M3 is the semantic bridge between the data-independent geometry of M2 and the physical
execution of a later milestone.  It binds M2's stochastic block-bin plan to the **actual
trusted contents** of one logical run, samples one SWAT-style interior point per bin, and
derives the abstract cross-run bin-read order.

**Two granularities, never conflated** (`decisions/0005-m3-content-bound-bin-schedule.md`):

```text
observable allocation / later I/O unit = LETIndex block
trusted merge comparison unit          = record
```

A block is the unit M2 allocates and a later milestone will fetch; a **record** is the unit
at which merge correctness is decided.  M3 uses the records *inside* a bin to sample an
interior point, which does **not** turn a block into one sortable database item.  Cross-run
order is never decided from a block's first key, last key, midpoint, rank, `BlockId` or any
physical identifier.

Provenance — the interior-point semantics are re-implemented from the pinned SWAT reference
(`CongGroup/SWAT` @ ``b33646061ec1899ccf75c9ba8fe43b2653c44a6b``):

| Pinned source | What is ported here |
|---|---|
| ``include/enclave/DPInteriorPoint.hpp`` | :func:`interior_point_weights` — the pinned ``baseExp ** (min(i, load - i) + 1)`` weighting, and the per-bin draw |
| ``include/enclave/DOMerger.hpp`` (``DOAllocate``) | one interior point per allocated bin, tagged ``+(i + 1)`` |
| ``include/enclave/DOMerger.hpp`` (``DOMerge``) | the right side's tags negated to ``-(j + 1)``, the ``std::pair`` ordering of the tagged sequence, and the fetch skeleton (preload bin 0 of each side, then request a side's next unread bin when its tag arrives) |

No C++ text is copied.  What M3 deliberately does **not** port: ciphertext handling,
SGX/AES, physical construction or fetching of padded bins, the pinned safe-output-count
arithmetic, and output publication.  M3 performs **zero** ``UntrustedStorage`` access, emits
**zero** ``TraceEvent`` and schedules **zero** physical slot: mapping a logical block rank to
its current physical slot and reading in schedule order would leak the logical-rank →
physical-access correlation, so physical staging is a later milestone's problem.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from enhanced_letindex.block import Block
from enhanced_letindex.identifiers import BlockId, RecordKey
from enhanced_letindex.record import Record

from .bin_allocator import BlockAllocationPlan
from .distribution import _fast_power as pinned_fast_power, derive_stream_seed

__all__ = [
    "DUMMY_POS_INF",
    "SOURCE",
    "STREAM_DOMAIN_INTERIOR_SOURCE",
    "STREAM_DOMAIN_INTERIOR_TARGET",
    "TARGET",
    "AbstractBinRead",
    "BoundBlockAllocation",
    "BoundBlockBin",
    "ContentScheduleError",
    "InteriorPoint",
    "InteriorIndexSampler",
    "LogicalBlockRunView",
    "LogicalBlockSnapshot",
    "SwatBlockMergeSchedule",
    "TaggedBinInteriorPoint",
    "bind_block_allocation",
    "build_abstract_merge_schedule",
    "interior_point_sort_key",
    "interior_point_weights",
    "interior_stream_domain",
    "plan_swat_block_merge_schedule",
    "sample_bin_interior_points",
    "sorted_tagged_interior_points",
]

#: Frozen side identities: the source is the *newer* level ``L`` and the target the
#: *older* level ``L + 1``.  Precedence is never inferred from a level magnitude beyond
#: validating that accepted adjacency.
SOURCE = "source"
TARGET = "target"
SIDES = (SOURCE, TARGET)

#: Domain-separated interior-point streams (distinct from every M2 stream).
STREAM_DOMAIN_INTERIOR_SOURCE = "swat-m-block/interior/source"
STREAM_DOMAIN_INTERIOR_TARGET = "swat-m-block/interior/target"

#: The narrow seam used by tie/order tests: given one bin's pinned weight vector, return
#: the index of the chosen real record.
InteriorIndexSampler = Callable[[Sequence[float]], int]


class ContentScheduleError(Exception):
    """An M3 content-binding, interior-point or schedule request is illegal."""


# ---------------------------------------------------------------------------
# the dummy +infinity sentinel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DummyPosInf:
    """The explicit ``+infinity`` sentinel of an empty-real bin interior point.

    The pinned C++ code operates on a dummy datum whose key is the numeric maximum, which
    Python integers do not have.  ``DUMMY_POS_INF`` therefore sorts after every real
    :class:`~enhanced_letindex.identifiers.RecordKey`, is deliberately **not** a
    ``RecordKey``, is never a database result, and exists only for trusted abstract
    scheduling.  Use :func:`interior_point_sort_key` to order it against real keys.
    """

    def __repr__(self) -> str:
        return "DUMMY_POS_INF"


#: The singleton sentinel — see :class:`_DummyPosInf`.
DUMMY_POS_INF = _DummyPosInf()

_SENTINEL_SORT_KEY = (1, 0)


def _reject_bound_block(index: int, value: object) -> "LogicalBlockSnapshot":
    raise ContentScheduleError(
        f"bin {index} may only bind Block objects, got {type(value).__name__}"
    )


def _require_plain_int(value: object, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContentScheduleError(
            f"{name} must be a plain int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise ContentScheduleError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContentScheduleError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ContentScheduleError(f"{name} must be a finite value > 0, got {value!r}")
    return number


def _require_side(side: object) -> str:
    if side not in SIDES:
        raise ContentScheduleError(
            f"a side must be {SOURCE!r} or {TARGET!r}, got {side!r}"
        )
    return str(side)


# ---------------------------------------------------------------------------
# the trusted logical block-run view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalBlockSnapshot:
    """An immutable snapshot of one logical block's content and geometry.

    ``enhanced_letindex.Block`` is *mutable* (``add_record`` keeps it sorted), so a view
    that held caller-owned ``Block`` objects could be invalidated after it was validated —
    including across an M2 bin boundary, which the per-bin recheck would not catch.  M3
    therefore snapshots every block at view construction and binds snapshots only; the
    caller's original block stays theirs, and nothing they do to it can change a view, a
    binding, an interior point or a schedule.

    The record tuple is immutable and the records themselves are frozen
    (:class:`enhanced_letindex.record.Record`); record values remain opaque trusted
    payloads, exactly as in the frozen substrate.
    """

    block_id: BlockId
    capacity: int
    records: Tuple[Record, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise ContentScheduleError(
                f"a block snapshot needs a BlockId, got {type(self.block_id).__name__}"
            )
        capacity = _require_plain_int(self.capacity, "capacity", minimum=1)
        records = tuple(self.records)
        for record in records:
            if not isinstance(record, Record):
                raise ContentScheduleError(
                    f"a block snapshot may only hold Record objects, got "
                    f"{type(record).__name__}"
                )
        if len(records) > capacity:
            raise ContentScheduleError(
                f"block {self.block_id} holds {len(records)} records, above its capacity "
                f"{capacity}"
            )
        previous = None
        for record in records:
            if previous is not None and record.key <= previous:
                raise ContentScheduleError(
                    f"block {self.block_id} keys must strictly increase, got "
                    f"{record.key} after {previous}"
                )
            previous = record.key
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "records", records)

    @classmethod
    def from_block(cls, block: Block) -> "LogicalBlockSnapshot":
        """Snapshot a caller-owned mutable block."""
        if not isinstance(block, Block):
            raise ContentScheduleError(
                f"a block snapshot needs a Block, got {type(block).__name__}"
            )
        return cls(
            block_id=block.block_id,
            capacity=block.capacity,
            records=block.records,
        )

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    @property
    def min_key(self) -> Optional[RecordKey]:
        return None if self.is_empty else self.records[0].key

    @property
    def max_key(self) -> Optional[RecordKey]:
        return None if self.is_empty else self.records[-1].key


@dataclass(frozen=True)
class LogicalBlockRunView:
    """An immutable trusted view of one level's blocks, with real block boundaries.

    Unlike M1's logical run view, this one preserves the **block** structure: M3 binds M2
    bins to actual blocks, so block boundaries must be real and verified.  Construction is
    pure trusted computation, performs zero storage I/O, and **snapshots** every input block
    into a :class:`LogicalBlockSnapshot`, so the view cannot be invalidated by a caller
    mutating a block afterwards.
    """

    level: int
    blocks: Tuple[LogicalBlockSnapshot, ...]
    items_per_block: int

    def __post_init__(self) -> None:
        level = _require_plain_int(self.level, "level", minimum=0)
        capacity = _require_plain_int(self.items_per_block, "items_per_block", minimum=1)
        snapshots: List[LogicalBlockSnapshot] = []
        for index, block in enumerate(tuple(self.blocks)):
            if isinstance(block, LogicalBlockSnapshot):
                snapshots.append(block)
            elif isinstance(block, Block):
                snapshots.append(LogicalBlockSnapshot.from_block(block))
            else:
                raise ContentScheduleError(
                    f"a logical block run may only hold Block objects, got "
                    f"{type(block).__name__} at rank {index}"
                )
        blocks = tuple(snapshots)
        for index, block in enumerate(blocks):
            if block.capacity != capacity:
                raise ContentScheduleError(
                    f"block {block.block_id} has capacity {block.capacity}, but the run "
                    f"fixes items_per_block = {capacity}"
                )
            if block.is_empty:
                raise ContentScheduleError(
                    f"block {block.block_id} at rank {index} is empty; a logical block "
                    "run holds non-empty blocks only"
                )
        if len({block.block_id for block in blocks}) != len(blocks):
            raise ContentScheduleError("block ids must be unique within a logical run")
        for index, (left, right) in enumerate(zip(blocks, blocks[1:])):
            if not left.max_key < right.min_key:
                raise ContentScheduleError(
                    f"neighbouring blocks must not overlap: block {left.block_id} "
                    f"(max key {left.max_key}) and block {right.block_id} "
                    f"(min key {right.min_key}) at ranks {index} and {index + 1}"
                )
        for index, block in enumerate(blocks[:-1]):
            if not block.is_full:
                raise ContentScheduleError(
                    f"every non-final block must be full, but block {block.block_id} at "
                    f"rank {index} holds {block.size} of {capacity} records"
                )
        if blocks and not 1 <= blocks[-1].size <= capacity:
            raise ContentScheduleError(
                f"the final block must hold between 1 and {capacity} records, got "
                f"{blocks[-1].size}"
            )
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "items_per_block", capacity)
        object.__setattr__(self, "blocks", blocks)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    @property
    def item_count(self) -> int:
        return sum(block.size for block in self.blocks)

    def records(self) -> Tuple[object, ...]:
        """Every record of the run, flattened in logical block order."""
        return tuple(record for block in self.blocks for record in block.records)

    def keys(self) -> Tuple[RecordKey, ...]:
        """Every record key of the run, in logical block order."""
        return tuple(record.key for record in self.records())

    def records_of(self, start: int, stop: int) -> Tuple[object, ...]:
        """The records of the logical rank interval ``[start, stop)``."""
        _require_plain_int(start, "start", minimum=0)
        _require_plain_int(stop, "stop", minimum=start)
        if stop > self.block_count:
            raise ContentScheduleError(
                f"rank interval [{start}, {stop}) leaves the run's {self.block_count} blocks"
            )
        return tuple(
            record for block in self.blocks[start:stop] for record in block.records
        )


# ---------------------------------------------------------------------------
# interior points
# ---------------------------------------------------------------------------


def interior_point_weights(load: int, privacy_epsilon: float) -> Tuple[float, ...]:
    """The pinned ``DPInteriorPoint.hpp`` weight vector of one bin (exact expression).

    The pinned source is::

        baseExp = exp(epsilon)
        for i in 0 .. bin.size():
            weights[i] = (i < load) ? fastPower(baseExp, min(i, load - i) + 1) : 0

    Only the first ``load`` entries are ever positive, and a zero-weight entry can never be
    selected, so the distribution over a bin's **real** records is exactly
    ``baseExp ** (min(i, load - i) + 1)`` for ``i`` in ``[0, load)`` — which is what M3
    implements.  The pinned expression is reproduced verbatim: it is deliberately *not*
    "symmetrised" into a different, tidier-looking formula, and the exponent is evaluated
    with the same exponentiation-by-squaring routine the pinned ``fastPower`` uses.
    """
    count = _require_plain_int(load, "load", minimum=0)
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    if count == 0:
        return ()
    base_exp = math.exp(epsilon)
    return tuple(
        pinned_fast_power(base_exp, min(index, count - index) + 1)
        for index in range(count)
    )


@dataclass(frozen=True)
class InteriorPoint:
    """One bin's sampled interior point, in trusted scheduling terms only.

    ``key`` is either a real :class:`RecordKey` taken from the bin's contents or the
    :data:`DUMMY_POS_INF` sentinel when the bin holds no real block.  ``record_index`` is
    the chosen record's index inside the bin when it is known, and is always ``None`` for
    the sentinel.
    """

    key: object
    record_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.key is DUMMY_POS_INF:
            if self.record_index is not None:
                raise ContentScheduleError(
                    "an empty-real bin interior point has no record index"
                )
            return
        if not isinstance(self.key, RecordKey):
            raise ContentScheduleError(
                "an interior point key must be a RecordKey or DUMMY_POS_INF, got "
                f"{type(self.key).__name__}"
            )
        if self.record_index is not None:
            object.__setattr__(
                self,
                "record_index",
                _require_plain_int(self.record_index, "record_index", minimum=0),
            )

    @property
    def is_dummy(self) -> bool:
        """True when this point is the empty-real-bin sentinel."""
        return self.key is DUMMY_POS_INF

    @property
    def record_key(self) -> RecordKey:
        """The point as a database key; refuses the sentinel.

        The sentinel exists only for abstract trusted scheduling, so a consumer that needs
        a real key cannot silently receive it: this raises instead.
        """
        if self.is_dummy:
            raise ContentScheduleError(
                "DUMMY_POS_INF marks an empty-real bin and is never a database key"
            )
        return self.key


def interior_point_sort_key(point: InteriorPoint) -> Tuple[int, int]:
    """The ordering key of an interior point.

    Every real key sorts before :data:`DUMMY_POS_INF`, and real keys sort by value, so a
    mixed sequence of interior points has the total order the pinned `std::pair` comparison
    gives a numeric maximum dummy key.
    """
    if not isinstance(point, InteriorPoint):
        raise ContentScheduleError(
            f"expected an InteriorPoint, got {type(point).__name__}"
        )
    if point.is_dummy:
        return _SENTINEL_SORT_KEY
    return (0, point.key.value)


def _draw_index(stream: "random.Random", weights: Sequence[float]) -> int:
    """Draw one index from ``weights`` with a planner-owned stream."""
    total = math.fsum(weights)
    if not total > 0.0:  # pragma: no cover - non-empty bins always carry weight
        raise ContentScheduleError("an interior-point weight vector must not be all zero")
    cumulative: List[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)
    draw = stream.random() * total
    index = bisect.bisect_right(cumulative, draw)
    if index >= len(weights):
        index = len(weights) - 1
    return index


# ---------------------------------------------------------------------------
# bound bins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundBlockBin:
    """One M2 bin bound to the real logical blocks it covers.

    ``blocks`` are immutable snapshots of the bin's **real** blocks, in logical rank
    order.  Padding is a count only: M3 materialises no dummy block, and no physical
    identifier appears here.
    """

    bin_index: int
    logical_rank_start: int
    logical_rank_stop: int
    sampled_load: int
    real_block_count: int
    dummy_block_count: int
    bin_capacity_blocks: int
    blocks: Tuple[LogicalBlockSnapshot, ...]

    def __post_init__(self) -> None:
        index = _require_plain_int(self.bin_index, "bin_index", minimum=0)
        start = _require_plain_int(self.logical_rank_start, "logical_rank_start", minimum=0)
        stop = _require_plain_int(self.logical_rank_stop, "logical_rank_stop", minimum=0)
        load = _require_plain_int(self.sampled_load, "sampled_load", minimum=0)
        real = _require_plain_int(self.real_block_count, "real_block_count", minimum=0)
        dummy = _require_plain_int(self.dummy_block_count, "dummy_block_count", minimum=0)
        capacity = _require_plain_int(self.bin_capacity_blocks, "bin_capacity_blocks",
                                      minimum=2)
        blocks: Tuple[LogicalBlockSnapshot, ...] = tuple(
            block if isinstance(block, LogicalBlockSnapshot)
            else LogicalBlockSnapshot.from_block(block) if isinstance(block, Block)
            else _reject_bound_block(index, block)
            for block in self.blocks
        )
        if len(blocks) != real:
            raise ContentScheduleError(
                f"bin {index} reports {real} real block(s) but binds {len(blocks)}"
            )
        if stop - start != real:
            raise ContentScheduleError(
                f"bin {index}: the rank interval [{start}, {stop}) does not hold "
                f"real_block_count = {real}"
            )
        if real + dummy != capacity:
            raise ContentScheduleError(
                f"bin {index}: real_block_count {real} + dummy_block_count {dummy} != "
                f"bin_capacity_blocks {capacity}"
            )
        if load > capacity:
            raise ContentScheduleError(
                f"bin {index}: sampled_load {load} exceeds bin_capacity_blocks {capacity}"
            )
        object.__setattr__(self, "bin_index", index)
        object.__setattr__(self, "logical_rank_start", start)
        object.__setattr__(self, "logical_rank_stop", stop)
        object.__setattr__(self, "sampled_load", load)
        object.__setattr__(self, "real_block_count", real)
        object.__setattr__(self, "dummy_block_count", dummy)
        object.__setattr__(self, "bin_capacity_blocks", capacity)
        object.__setattr__(self, "blocks", blocks)

    @property
    def is_empty_real(self) -> bool:
        """True when the bin holds no real block (its interior point is the sentinel)."""
        return self.real_block_count == 0

    def real_records(self) -> Tuple[object, ...]:
        """The bin's real records, flattened in logical block order."""
        return tuple(record for block in self.blocks for record in block.records)

    def real_keys(self) -> Tuple[RecordKey, ...]:
        """The bin's real record keys, in logical block order."""
        return tuple(record.key for record in self.real_records())


@dataclass(frozen=True)
class TaggedBinInteriorPoint:
    """A bin's interior point with the pinned DOMerge signed tag.

    Source bin ``i`` carries ``+(i + 1)`` and target bin ``j`` carries ``-(j + 1)``,
    matching the pinned tagging.  The pair ``(interior point, signed tag)`` orders exactly
    like the pinned ``std::pair`` comparison, so an equal interior point places the
    negative target tag before the positive source tag.
    """

    side: str
    bin_index: int
    interior_point: InteriorPoint
    signed_tag: int

    def __post_init__(self) -> None:
        side = _require_side(self.side)
        index = _require_plain_int(self.bin_index, "bin_index", minimum=0)
        if not isinstance(self.interior_point, InteriorPoint):
            raise ContentScheduleError(
                f"expected an InteriorPoint, got {type(self.interior_point).__name__}"
            )
        tag = _require_plain_int(self.signed_tag, "signed_tag")
        expected = index + 1 if side == SOURCE else -(index + 1)
        if tag != expected:
            raise ContentScheduleError(
                f"a {side} bin {index} must carry the signed tag {expected}, got {tag}"
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "bin_index", index)

    @property
    def ordering_key(self) -> Tuple[Tuple[int, int], int]:
        """The pinned ``(interior point, signed tag)`` sort key."""
        return (interior_point_sort_key(self.interior_point), self.signed_tag)


@dataclass(frozen=True)
class BoundBlockAllocation:
    """One side's M2 plan bound to real trusted contents.

    The M2 evidence — ``sampled_loads``, ``noisy_prefix_sums`` and
    ``additive_error_blocks`` — is **carried from the plan unchanged** (the properties below
    read straight through to it, so it cannot be silently rewritten), and the M2 dummy
    padding stays a count.
    """

    side: str
    level: int
    items_per_block: int
    bins: Tuple[BoundBlockBin, ...]
    plan: BlockAllocationPlan
    interiors: Tuple[InteriorPoint, ...] = ()
    tagged_interior_points: Tuple[TaggedBinInteriorPoint, ...] = ()
    sampled: bool = False

    def __post_init__(self) -> None:
        side = _require_side(self.side)
        level = _require_plain_int(self.level, "level", minimum=0)
        capacity = _require_plain_int(self.items_per_block, "items_per_block", minimum=1)
        bins = tuple(self.bins)
        if not isinstance(self.plan, BlockAllocationPlan):
            raise ContentScheduleError(
                f"a bound allocation needs a BlockAllocationPlan, got "
                f"{type(self.plan).__name__}"
            )
        if len(bins) != self.plan.bin_count:
            raise ContentScheduleError(
                f"the plan holds {self.plan.bin_count} bin(s) but {len(bins)} are bound"
            )
        for index, bin_ in enumerate(bins):
            if bin_.bin_index != index:
                raise ContentScheduleError(
                    f"bound bins must keep the M2 bin order: expected bin {index}, got "
                    f"{bin_.bin_index}"
                )
        if not isinstance(self.sampled, bool):
            raise ContentScheduleError("sampled must be a bool")
        interiors = tuple(self.interiors)
        tagged = tuple(self.tagged_interior_points)
        if not self.sampled and (interiors or tagged):
            raise ContentScheduleError(
                "an allocation carries interior points only once it is sampled"
            )
        if self.sampled:
            if len(interiors) != len(bins):
                raise ContentScheduleError(
                    f"one interior point per bin is required, got {len(interiors)} for "
                    f"{len(bins)} bin(s)"
                )
            if len(tagged) != len(bins):
                raise ContentScheduleError(
                    f"one tagged interior point per bin is required, got {len(tagged)} for "
                    f"{len(bins)} bin(s)"
                )
            for point in tagged:
                if point.side != side:
                    raise ContentScheduleError(
                        f"tagged interior points must belong to the {side} side"
                    )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "items_per_block", capacity)
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "interiors", interiors)
        object.__setattr__(self, "tagged_interior_points", tagged)

    # -- the preserved M2 evidence ------------------------------------------------

    @property
    def bin_count(self) -> int:
        return len(self.bins)

    @property
    def block_count(self) -> int:
        return self.plan.block_count

    @property
    def sampled_loads(self) -> Tuple[int, ...]:
        """M2's sampled bin loads in blocks — carried unchanged."""
        return self.plan.sampled_loads

    @property
    def noisy_prefix_sums(self) -> Tuple[int, ...]:
        """M2's noisy prefix sums in **blocks** — carried unchanged."""
        return self.plan.noisy_prefix_sums

    @property
    def additive_error_blocks(self) -> int:
        """M2's additive error in **blocks** — carried unchanged."""
        return self.plan.additive_error_blocks

    # -- content views -------------------------------------------------------------

    @property
    def is_sampled(self) -> bool:
        """True once interior points have been sampled for this side.

        This is tracked explicitly rather than inferred from the interior tuple: a side
        with zero bins legitimately has zero interior points and is still sampled.
        """
        return self.sampled

    @property
    def is_empty(self) -> bool:
        return not self.bins

    def bound_ranks(self) -> Tuple[int, ...]:
        """Every logical block rank bound by this side, in bin order."""
        return tuple(
            rank for bin_ in self.bins
            for rank in range(bin_.logical_rank_start, bin_.logical_rank_stop)
        )

    def real_records(self) -> Tuple[object, ...]:
        """Every real record bound by this side, in bin then block order."""
        return tuple(
            record for bin_ in self.bins for record in bin_.real_records()
        )

    def real_keys(self) -> Tuple[RecordKey, ...]:
        """Every real record key bound by this side, in bin then block order."""
        return tuple(record.key for record in self.real_records())

    def interior_point_of(self, bin_index: int) -> InteriorPoint:
        index = _require_plain_int(bin_index, "bin_index", minimum=0)
        if not self.is_sampled:
            raise ContentScheduleError("interior points have not been sampled yet")
        if index >= len(self.interiors):
            raise ContentScheduleError(
                f"bin {index} does not exist in this {len(self.interiors)}-bin allocation"
            )
        return self.interiors[index]


@dataclass(frozen=True)
class AbstractBinRead:
    """One abstract bin fetch: ``(side, bin_index)`` and nothing else.

    There is deliberately no slot, handle or storage reference here: M3 defines the
    semantic fetch order only, and the physical staging of a padded bin is a later
    milestone's decision.
    """

    side: str
    bin_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _require_side(self.side))
        object.__setattr__(
            self, "bin_index", _require_plain_int(self.bin_index, "bin_index", minimum=0)
        )


@dataclass(frozen=True)
class SwatBlockMergeSchedule:
    """The abstract SWAT-M-Block bin-read schedule of two bound sides.

    The invariant asserted here is the whole point of the schedule: **every** planned bin of
    **each** side appears **exactly once**, and no read falls outside ``[0, bin_count)``.
    """

    source: BoundBlockAllocation
    target: BoundBlockAllocation
    reads: Tuple[AbstractBinRead, ...]
    tagged_interior_points: Tuple[TaggedBinInteriorPoint, ...]

    def __post_init__(self) -> None:
        for side, bound in ((SOURCE, self.source), (TARGET, self.target)):
            if not isinstance(bound, BoundBlockAllocation):
                raise ContentScheduleError(
                    f"the {side} side must be a BoundBlockAllocation, got "
                    f"{type(bound).__name__}"
                )
            if bound.side != side:
                raise ContentScheduleError(
                    f"the {side} slot holds a {bound.side} allocation"
                )
            if not bound.is_sampled:
                raise ContentScheduleError(
                    f"the {side} interior points must be sampled before scheduling"
                )
        reads = tuple(self.reads)
        for read in reads:
            if not isinstance(read, AbstractBinRead):
                raise ContentScheduleError(
                    f"a read must be an AbstractBinRead, got {type(read).__name__}"
                )
            bound = self.source if read.side == SOURCE else self.target
            if read.bin_index >= bound.bin_count:
                raise ContentScheduleError(
                    f"a {read.side} read of bin {read.bin_index} is out of range for "
                    f"{bound.bin_count} bin(s)"
                )
        for side, bound in ((SOURCE, self.source), (TARGET, self.target)):
            scheduled = [read.bin_index for read in reads if read.side == side]
            if sorted(scheduled) != list(range(bound.bin_count)):
                raise ContentScheduleError(
                    f"every {side} bin must be scheduled exactly once, got {scheduled} "
                    f"for {bound.bin_count} bin(s)"
                )
        tagged = tuple(self.tagged_interior_points)
        expected_count = self.source.bin_count + self.target.bin_count
        if len(tagged) != expected_count:
            raise ContentScheduleError(
                f"the tagged sequence must hold one point per bin ({expected_count}), got "
                f"{len(tagged)}"
            )
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "tagged_interior_points", tagged)

    @property
    def is_empty(self) -> bool:
        return not self.reads

    @property
    def read_count(self) -> int:
        return len(self.reads)

    def read_order(self, side: str) -> Tuple[int, ...]:
        """The scheduled bin indices of one side, in fetch order."""
        checked = _require_side(side)
        return tuple(read.bin_index for read in self.reads if read.side == checked)

    @property
    def source_read_order(self) -> Tuple[int, ...]:
        return self.read_order(SOURCE)

    @property
    def target_read_order(self) -> Tuple[int, ...]:
        return self.read_order(TARGET)


# ---------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------


def bind_block_allocation(
    run: LogicalBlockRunView,
    plan: BlockAllocationPlan,
    *,
    side: str,
) -> BoundBlockAllocation:
    """Bind an M2 block-bin ``plan`` to the contents of ``run``.

    Binding is legal only when ``plan.block_count == run.block_count``.  Each M2 bin's
    ``[start, stop)`` rank interval is bound to ``run.blocks[start:stop]``, so every real
    block is bound exactly once, no block is duplicated or omitted, and the M2 bin order is
    preserved.  The M2 evidence (loads, noisy prefixes, additive error, dummy counts) is
    carried through untouched, and no dummy block is materialised.
    """
    if not isinstance(run, LogicalBlockRunView):
        raise ContentScheduleError(
            f"binding needs a LogicalBlockRunView, got {type(run).__name__}"
        )
    if not isinstance(plan, BlockAllocationPlan):
        raise ContentScheduleError(
            f"binding needs a BlockAllocationPlan, got {type(plan).__name__}"
        )
    checked_side = _require_side(side)
    if plan.block_count != run.block_count:
        raise ContentScheduleError(
            f"the plan covers {plan.block_count} block(s) but the run holds "
            f"{run.block_count}; a plan binds only the run it was allocated for"
        )

    bins: List[BoundBlockBin] = []
    for bin_plan in plan.bins:
        blocks = run.blocks[bin_plan.logical_rank_start:bin_plan.logical_rank_stop]
        bound = BoundBlockBin(
            bin_index=bin_plan.bin_index,
            logical_rank_start=bin_plan.logical_rank_start,
            logical_rank_stop=bin_plan.logical_rank_stop,
            sampled_load=bin_plan.sampled_load,
            real_block_count=bin_plan.real_count,
            dummy_block_count=bin_plan.dummy_count,
            bin_capacity_blocks=plan.bin_capacity_blocks,
            blocks=tuple(blocks),
        )
        keys = bound.real_keys()
        if list(keys) != sorted(set(keys)):
            raise ContentScheduleError(
                f"bin {bound.bin_index} record keys must strictly increase across its "
                "blocks; the bound block ranges are not in run order"
            )
        bins.append(bound)
    return BoundBlockAllocation(
        side=checked_side,
        level=run.level,
        items_per_block=run.items_per_block,
        bins=tuple(bins),
        plan=plan,
    )


# ---------------------------------------------------------------------------
# interior-point sampling
# ---------------------------------------------------------------------------


def interior_stream_domain(side: str) -> str:
    """The interior-point stream domain of one side — fixed by the side, never caller-set."""
    checked = _require_side(side)
    return (STREAM_DOMAIN_INTERIOR_SOURCE if checked == SOURCE
            else STREAM_DOMAIN_INTERIOR_TARGET)


def _sample_bin_interior_points(
    bound: BoundBlockAllocation,
    *,
    privacy_epsilon: float,
    seed: int,
    domain: str,
    interior_index_sampler: Optional[InteriorIndexSampler] = None,
) -> BoundBlockAllocation:
    """Private low-level sampler: one pinned interior point per bin of ``bound``.

    The public :func:`sample_bin_interior_points` always calls this with the frozen M2 plan
    config's ``privacy_epsilon``/``seed`` and the side's fixed domain; the parameters exist
    here only so that the binding rule is visible in one place.
    """
    if not isinstance(bound, BoundBlockAllocation):
        raise ContentScheduleError(
            f"sampling needs a BoundBlockAllocation, got {type(bound).__name__}"
        )
    if bound.is_sampled:
        raise ContentScheduleError("this allocation already carries interior points")
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    if not isinstance(domain, str) or not domain:
        raise ContentScheduleError("an interior-point domain must be a non-empty str")
    if interior_index_sampler is not None and not callable(interior_index_sampler):
        raise ContentScheduleError("interior_index_sampler must be callable")

    stream = None
    if interior_index_sampler is None:
        stream = random.Random(
            derive_stream_seed(_require_plain_int(seed, "seed", minimum=0), domain)
        )

    interiors: List[InteriorPoint] = []
    tagged: List[TaggedBinInteriorPoint] = []
    for bin_ in bound.bins:
        keys = bin_.real_keys()
        if not keys:
            point = InteriorPoint(DUMMY_POS_INF, None)
        else:
            weights = interior_point_weights(len(keys), epsilon)
            if interior_index_sampler is not None:
                index = interior_index_sampler(weights)
            else:
                index = _draw_index(stream, weights)
            if isinstance(index, bool) or not isinstance(index, int):
                raise ContentScheduleError(
                    f"an interior index sampler must return a plain int, got "
                    f"{type(index).__name__}"
                )
            if not 0 <= index < len(keys):
                raise ContentScheduleError(
                    f"an interior index sampler returned {index}, outside the bin's "
                    f"{len(keys)} real record(s)"
                )
            point = InteriorPoint(keys[index], index)
        interiors.append(point)
        tagged.append(
            TaggedBinInteriorPoint(
                side=bound.side,
                bin_index=bin_.bin_index,
                interior_point=point,
                signed_tag=(bin_.bin_index + 1 if bound.side == SOURCE
                            else -(bin_.bin_index + 1)),
            )
        )
    return replace(
        bound,
        interiors=tuple(interiors),
        tagged_interior_points=tuple(tagged),
        sampled=True,
    )


def sample_bin_interior_points(
    bound: BoundBlockAllocation,
    *,
    interior_index_sampler: Optional[InteriorIndexSampler] = None,
) -> BoundBlockAllocation:
    """Sample one pinned interior point per bin of ``bound``.

    A bin with real content yields an interior point drawn from the pinned weight vector
    over its **actual record keys**; a bin whose real blocks were exhausted by the run
    yields :data:`DUMMY_POS_INF`.

    The sampling parameters are **not** caller-supplied: ``privacy_epsilon`` and ``seed``
    come from the frozen M2 plan config this allocation is bound to
    (``bound.plan.config``) and the stream domain is fixed by ``bound.side``
    (:func:`interior_stream_domain`).  That keeps the pinned weight expression and the
    domain separation a property of the frozen plan rather than of a call site: a caller
    cannot pair an allocation with an unrelated epsilon, reuse another domain, or make the
    two sides share a stream.

    ``interior_index_sampler`` is the narrow seam for deterministic tie/order tests: it
    receives one bin's pinned weight vector and returns the chosen record index.
    """
    if not isinstance(bound, BoundBlockAllocation):
        raise ContentScheduleError(
            f"sampling needs a BoundBlockAllocation, got {type(bound).__name__}"
        )
    config = bound.plan.config
    return _sample_bin_interior_points(
        bound,
        privacy_epsilon=config.privacy_epsilon,
        seed=config.seed,
        domain=interior_stream_domain(bound.side),
        interior_index_sampler=interior_index_sampler,
    )


# ---------------------------------------------------------------------------
# the abstract schedule
# ---------------------------------------------------------------------------


def sorted_tagged_interior_points(
    source: BoundBlockAllocation,
    target: BoundBlockAllocation,
) -> Tuple[TaggedBinInteriorPoint, ...]:
    """The pinned ``(interior point, signed tag)`` ordering over both sides."""
    return tuple(
        sorted(
            tuple(source.tagged_interior_points) + tuple(target.tagged_interior_points),
            key=lambda point: point.ordering_key,
        )
    )


def build_abstract_merge_schedule(
    source: BoundBlockAllocation,
    target: BoundBlockAllocation,
) -> Tuple[AbstractBinRead, ...]:
    """The abstract bin-fetch order of a merge, at ``(side, bin_index)`` granularity.

    Skeleton (pinned ``DOMerge``): preload bin 0 of each non-empty side, scan the sorted
    tagged interior points, and when a side's tag arrives request that side's next unread
    bin if one exists.  Each planned bin is therefore fetched exactly once.

    This is the *projection* of the pinned loop onto actual bin fetches, not the pinned
    per-iteration machine: pinned ``DOMerge`` also runs a fixed ``binCnt`` iteration count,
    keeps ``j0``/``j1`` state, and has a real ``else if`` fallback (a tag whose own side is
    exhausted fetches from the other side), interleaved with safe-output/frontier work that
    this function does not model.  On well-formed input the projected sequences coincide,
    because once a side is exhausted no further same-side fetch can occur.  A later
    milestone that introduces that frontier state must replay the pinned loop instead of
    treating one read here as one merge iteration.
    """
    for side, bound in ((SOURCE, source), (TARGET, target)):
        if not isinstance(bound, BoundBlockAllocation):
            raise ContentScheduleError(
                f"the {side} side must be a BoundBlockAllocation, got "
                f"{type(bound).__name__}"
            )
        if not bound.is_sampled:
            raise ContentScheduleError(
                f"the {side} interior points must be sampled before scheduling"
            )

    reads: List[AbstractBinRead] = []
    next_unread = {SOURCE: 0, TARGET: 0}
    bin_counts = {SOURCE: source.bin_count, TARGET: target.bin_count}

    # preload bin 0 of each non-empty side, source first
    for side in SIDES:
        if bin_counts[side] > 0:
            reads.append(AbstractBinRead(side, 0))
            next_unread[side] = 1

    for point in sorted_tagged_interior_points(source, target):
        side = point.side
        if next_unread[side] < bin_counts[side]:
            reads.append(AbstractBinRead(side, next_unread[side]))
            next_unread[side] += 1
    return tuple(reads)


def plan_swat_block_merge_schedule(
    source_run: LogicalBlockRunView,
    target_run: LogicalBlockRunView,
    *,
    source_plan: BlockAllocationPlan,
    target_plan: BlockAllocationPlan,
    interior_index_sampler: Optional[InteriorIndexSampler] = None,
) -> SwatBlockMergeSchedule:
    """Bind both runs to their M2 plans and derive the abstract merge schedule.

    The accepted merge identity is ``source = L`` (newer) into ``target = L + 1`` (older);
    the relation is validated, never inferred from the level magnitudes.  Each side's
    interior points are drawn from its own frozen M2 plan config
    (``privacy_epsilon`` and ``seed``) and its own fixed side domain, so two sides that
    share a config seed still differ.  Neither parameter is a caller choice.

    M3 emits no merged output: the exact logical result ``C`` remains the M1 oracle's job.
    """
    for name, run in (("source", source_run), ("target", target_run)):
        if not isinstance(run, LogicalBlockRunView):
            raise ContentScheduleError(
                f"the {name} of a merge must be a LogicalBlockRunView, got "
                f"{type(run).__name__}"
            )
    if target_run.level != source_run.level + 1:
        raise ContentScheduleError(
            "a SWAT-M-Block merge binds a newer source level L to the older target L + 1, "
            f"got source {source_run.level} and target {target_run.level}"
        )
    if source_run.items_per_block != target_run.items_per_block:
        raise ContentScheduleError(
            "both sides of a merge must use the same block geometry, got "
            f"items_per_block {source_run.items_per_block} and "
            f"{target_run.items_per_block}"
        )
    for name, plan in (("source", source_plan), ("target", target_plan)):
        if not isinstance(plan, BlockAllocationPlan):
            raise ContentScheduleError(
                f"the {name} plan must be a BlockAllocationPlan, got "
                f"{type(plan).__name__}"
            )

    # each side's interior stream comes from its own frozen M2 plan config
    # (privacy_epsilon + seed) and its own fixed side domain - never from the caller
    source_bound = sample_bin_interior_points(
        bind_block_allocation(source_run, source_plan, side=SOURCE),
        interior_index_sampler=interior_index_sampler,
    )
    target_bound = sample_bin_interior_points(
        bind_block_allocation(target_run, target_plan, side=TARGET),
        interior_index_sampler=interior_index_sampler,
    )
    return SwatBlockMergeSchedule(
        source=source_bound,
        target=target_bound,
        reads=build_abstract_merge_schedule(source_bound, target_bound),
        tagged_interior_points=sorted_tagged_interior_points(source_bound, target_bound),
    )
