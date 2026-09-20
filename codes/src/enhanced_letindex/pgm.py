"""Batch PGM construction (G1-B1).

This module implements the canonical batch PGM baseline frozen in
``decisions/0002-batch-pgm-baseline.md`` and
``decisions/0003-g1-b1-implementation-contract.md``: a sequential
``OptimalPiecewiseLinearModel`` (OPLM), ported exactly from the official
PGM-index algorithm, using exact integer arithmetic everywhere so that
no floating-point value ever decides feasibility, hull updates, slope
ordering, segment boundaries, or segment count.

Scope (single already-constructed logical Level):
    - canonical data points ``(key_i, i)`` with ``i`` the level-local
      sorted record rank;
    - sequential OPLM construction over real points only;
    - ``SearchResult(pos, lo, hi)``, with the official epsilon window
      ``lo = max(0, pos - eps)``, ``hi = min(n, pos + eps + 2)``.

Not implemented here (later milestones): rank -> block -> slot physical
access, membership, multi-level lookup, hit-and-stop, small-level
policy, merge, incremental PGM, and any security mechanism.

Trust discipline: ``PgmSegment`` and ``BatchPgmIndex`` store only
trusted PGM metadata.  They never store or predict physical ``SlotId``,
never retain the full key array or the records, and ``search`` never
touches ``UntrustedStorage`` (so metadata search produces zero physical
trace events).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .identifiers import RecordKey

__all__ = [
    "SearchResult",
    "PgmSegment",
    "RankLineCertificate",
    "BatchPgmIndex",
    "PgmError",
    "PgmConfigurationError",
    "build_batch_pgm",
    "make_segmentation",
]


class PgmError(Exception):
    """Base error for PGM construction/search problems."""


class PgmConfigurationError(PgmError):
    """The engine's epsilon is not explicitly configured."""


# ---------------------------------------------------------------------------
# Exact integer helpers
# ---------------------------------------------------------------------------


def _trunc_sign(value: int) -> int:
    """Sign of ``value`` (``-1`` when negative, ``+1`` otherwise).

    Used to reproduce the official C++ halfway-toward-zero rounding term,
    which for a positive denominator reduces to ``sign(numerator)``.
    """
    return 1 if value >= 0 else -1


def _div_trunc(num: int, den: int) -> int:
    """Exact C++-style integer division (truncation toward zero).

    ``//`` is floor division and changes the result for negative
    numerators, which would be semantically wrong here.
    """
    if den == 0:
        raise ZeroDivisionError("PGM division by zero denominator")
    return num // den if num >= 0 else -((-num) // den)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A legal candidate record-rank interval ``[lo, hi)`` plus a predicted
    rank ``pos``.

    PGM search does NOT decide membership.  The caller must verify
    membership by checking whether ``record[pos].key == q`` under the
    condition ``pos < n``.  ``lo``/``hi`` are the exclusive-upper window
    within which a later ``bisect_left``-based lower_bound is guaranteed
    to find the true insertion point.
    """

    pos: int
    lo: int
    hi: int


# ---------------------------------------------------------------------------
# OptimalPiecewiseLinearModel (exact-integer port of the official PGM-index)
# ---------------------------------------------------------------------------


class _Point:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Point({self.x}, {self.y})"


class _Slope:
    """A slope as an exact ordered pair (dx, dy).

    Comparisons use integer cross multiplication exactly like the official
    ``Slope`` struct (``dy * p.dx < dx * p.dy``).  ``dx`` is an int here
    and is always positive for the slopes produced by the algorithm.
    """

    __slots__ = ("dx", "dy")

    def __init__(self, dx: int, dy: int) -> None:
        self.dx = dx
        self.dy = dy

    def __lt__(self, other: "_Slope") -> bool:
        return self.dy * other.dx < self.dx * other.dy

    def __gt__(self, other: "_Slope") -> bool:
        return self.dy * other.dx > self.dx * other.dy

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Slope):
            return NotImplemented
        return self.dy * other.dx == self.dx * other.dy

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _Slope):
            return NotImplemented
        return self.dy * other.dx != self.dx * other.dy

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Slope({self.dx}, {self.dy})"


def _sub_slope(a: _Point, b: _Point) -> _Slope:
    """``b - a`` as a slope, i.e. ``(b.x - a.x, b.y - a.y)``."""
    return _Slope(b.x - a.x, b.y - a.y)


def _cross(o: _Point, a: _Point, b: _Point) -> int:
    """2D cross product ``(a-o) x (b-o)``; positive is a left turn."""
    oa_dx = a.x - o.x
    oa_dy = a.y - o.y
    ob_dx = b.x - o.x
    ob_dy = b.y - o.y
    return oa_dx * ob_dy - oa_dy * ob_dx


class _OptimalPiecewiseLinearModel:
    """Sequential feasible-region builder (O'Rourke / official PGM-index).

    Maintains the upper/lower convex hulls and a 4-corner rectangle that
    describes the current canonical segment.  All geometry is exact
    integer arithmetic; slope direction and hull updates match the
    official ``OptimalPiecewiseLinearModel`` exactly.
    """

    def __init__(self, epsilon: int) -> None:
        self.epsilon: int = epsilon
        self.lower: list[_Point] = []
        self.upper: list[_Point] = []
        self.first_x: int = 0
        self.last_x: int = 0
        self.lower_start: int = 0
        self.upper_start: int = 0
        self.points_in_hull: int = 0
        self.rectangle: list[_Point] = [_Point(0, 0) for _ in range(4)]

    # -- helpers ------------------------------------------------------------

    def _eps_add(self, value: int) -> int:
        return value + self.epsilon

    def _eps_sub(self, value: int) -> int:
        """Lower epsilon band clipped at 0 (non-negative rank domain)."""
        if value <= self.epsilon:
            return 0
        return value - self.epsilon

    def _reset(self) -> None:
        self.points_in_hull = 0
        self.lower.clear()
        self.upper.clear()

    # -- the official add_point, in exact integer arithmetic ----------------

    def add_point(self, x: int, y: int) -> bool:
        if self.points_in_hull > 0 and x <= self.last_x:
            raise PgmError("OPLM points must be strictly increasing by x")

        self.last_x = x
        p1 = _Point(x, self._eps_add(y))
        p2 = _Point(x, self._eps_sub(y))

        if self.points_in_hull == 0:
            self.first_x = x
            self.rectangle[0] = _Point(p1.x, p1.y)
            self.rectangle[1] = _Point(p2.x, p2.y)
            self.upper.clear()
            self.lower.clear()
            self.upper.append(_Point(p1.x, p1.y))
            self.lower.append(_Point(p2.x, p2.y))
            self.upper_start = self.lower_start = 0
            self.points_in_hull += 1
            return True

        if self.points_in_hull == 1:
            self.rectangle[2] = _Point(p2.x, p2.y)
            self.rectangle[3] = _Point(p1.x, p1.y)
            self.upper.append(_Point(p1.x, p1.y))
            self.lower.append(_Point(p2.x, p2.y))
            self.points_in_hull += 1
            return True

        slope1 = _sub_slope(self.rectangle[0], self.rectangle[2])
        slope2 = _sub_slope(self.rectangle[1], self.rectangle[3])
        outside_line1 = _sub_slope(self.rectangle[2], p1) < slope1
        outside_line2 = _sub_slope(self.rectangle[3], p2) > slope2

        if outside_line1 or outside_line2:
            self.points_in_hull = 0
            return False

        lower = self.lower
        upper = self.upper

        # First (negative-slope) branch: find the lower hull node with the
        # minimum slope toward p1 (official code uses min == extreme here).
        if _sub_slope(self.rectangle[1], p1) < slope2:
            min_slope = _sub_slope(lower[self.lower_start], p1)
            min_i = self.lower_start
            for i in range(self.lower_start + 1, len(lower)):
                val = _sub_slope(lower[i], p1)
                if val > min_slope:
                    break
                min_slope = val
                min_i = i

            self.rectangle[1] = _Point(lower[min_i].x, lower[min_i].y)
            self.rectangle[3] = _Point(p1.x, p1.y)
            self.lower_start = min_i

            end = len(upper)
            while end >= self.upper_start + 2 and _cross(
                upper[end - 2], upper[end - 1], p1
            ) <= 0:
                end -= 1
            del upper[end:]
            upper.append(_Point(p1.x, p1.y))

        # Second (positive-slope) branch: find the upper hull node with the
        # maximum slope toward p2.
        if _sub_slope(self.rectangle[0], p2) > slope1:
            max_slope = _sub_slope(upper[self.upper_start], p2)
            max_i = self.upper_start
            for i in range(self.upper_start + 1, len(upper)):
                val = _sub_slope(upper[i], p2)
                if val < max_slope:
                    break
                max_slope = val
                max_i = i

            self.rectangle[0] = _Point(upper[max_i].x, upper[max_i].y)
            self.rectangle[2] = _Point(p2.x, p2.y)
            self.upper_start = max_i

            end = len(lower)
            while end >= self.lower_start + 2 and _cross(
                lower[end - 2], lower[end - 1], p2
            ) >= 0:
                end -= 1
            del lower[end:]
            lower.append(_Point(p2.x, p2.y))

        self.points_in_hull += 1
        return True

    def get_segment(self) -> "_CanonicalSegmentGeometry":
        if self.points_in_hull == 1:
            p0 = self.rectangle[0]
            p1 = self.rectangle[1]
            return _CanonicalSegmentGeometry(
                (_Point(p0.x, p0.y), _Point(p1.x, p1.y),
                 _Point(p0.x, p0.y), _Point(p1.x, p1.y)),
                self.first_x,
                one_point=True,
            )
        return _CanonicalSegmentGeometry(
            tuple(_Point(p.x, p.y) for p in self.rectangle), self.first_x
        )


class _CanonicalSegmentGeometry:
    """A frozen snapshot of a canonical segment's 4-corner rectangle plus
    its first x, consumed immediately to build a :class:`PgmSegment`.
    """

    __slots__ = ("rectangle", "first", "one_point")

    def __init__(
        self,
        rectangle: Tuple[_Point, _Point, _Point, _Point],
        first: int,
        one_point: bool = False,
    ) -> None:
        self.rectangle = rectangle
        self.first = first
        self.one_point = one_point


# ---------------------------------------------------------------------------
# PgmSegment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankLineCertificate:
    """The exact canonical continuous rank line of one real-data PGM segment (G5-A.1).

    A canonical OPLM segment admits an exact rational affine line inside its epsilon
    band for every point it covers.  The *search* model stored by :class:`PgmSegment`
    is a different object: it rounds the intercept to an integer (and
    :meth:`PgmSegment.predict` additionally truncates the slope term toward zero), so
    its training-point error may reach ``epsilon + 1`` and it must never be used as the
    continuous line of a later proof.

    This certificate preserves the unrounded canonical line as exact integer metadata
    over the segment's inclusive real-point span and inclusive rank span:

    ```text
    L_s(x) = (slope_num * (x - start_key) + origin_num) / denominator
    ```

    All coefficients are integers and ``denominator`` is strictly positive; no float or
    ``Decimal`` is ever used to build or evaluate it.  The normative invariant, checked
    by exact cross multiplication, is

    ```text
    max(0, rank - epsilon) * denominator <= L_num(key) <= (rank + epsilon) * denominator
    ```

    for every real point ``(key, rank)`` assigned to the segment, i.e.
    ``|L_s(key) - rank| <= epsilon``.  This is the frozen premise of the later
    Half-Rank Completion / envelope work; nothing in G5-A.1 consumes it.
    """

    start_key: RecordKey
    end_key: RecordKey
    start_rank: int
    end_rank: int
    slope_num: int
    origin_num: int
    denominator: int

    def __post_init__(self) -> None:
        for name in ("start_key", "end_key"):
            if not isinstance(getattr(self, name), RecordKey):
                raise PgmError(
                    f"certificate {name} must be a RecordKey, got "
                    f"{type(getattr(self, name)).__name__}"
                )
        for name in ("start_rank", "end_rank", "slope_num", "origin_num", "denominator"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PgmError(
                    f"certificate {name} must be a plain int, got {type(value).__name__}"
                )
        if self.denominator <= 0:
            raise PgmError(
                f"a certificate denominator must be strictly positive, got "
                f"{self.denominator}"
            )
        if self.start_rank < 0:
            raise PgmError(f"a certificate start_rank cannot be negative, got {self.start_rank}")
        if self.end_rank < self.start_rank:
            raise PgmError(
                f"a certificate span must satisfy end_rank >= start_rank, got "
                f"{self.start_rank}..{self.end_rank}"
            )
        if self.end_key < self.start_key:
            raise PgmError(
                f"a certificate span must satisfy end_key >= start_key, got "
                f"{self.start_key.value}..{self.end_key.value}"
            )

    # -- evaluation (exact integers only) ----------------------------------

    def numerator_at(self, key: RecordKey) -> int:
        """The exact numerator of ``L_s(key)``: ``slope_num * (key - start_key) + origin_num``."""
        delta = key.value - self.start_key.value
        return self.slope_num * delta + self.origin_num

    def within_band(self, key: RecordKey, rank: int, epsilon: int) -> bool:
        """The frozen clipped-band invariant, by exact cross multiplication.

        ```text
        max(0, rank - epsilon) * denominator <= numerator_at(key)
                                          <= (rank + epsilon) * denominator
        ```
        """
        value = self.numerator_at(key)
        denominator = self.denominator
        lower = rank - epsilon
        if lower < 0:
            lower = 0
        return lower * denominator <= value <= (rank + epsilon) * denominator

    @property
    def rank_span(self) -> tuple:
        return (self.start_rank, self.end_rank)

    @property
    def key_span(self) -> tuple:
        return (self.start_key, self.end_key)

    def to_dict(self) -> dict:
        """Exact integer description (no float ever appears here)."""
        return {
            "start_key": self.start_key.value,
            "end_key": self.end_key.value,
            "start_rank": self.start_rank,
            "end_rank": self.end_rank,
            "slope_num": self.slope_num,
            "origin_num": self.origin_num,
            "denominator": self.denominator,
        }


@dataclass(frozen=True)
class PgmSegment:
    """One canonical real-data OPLM segment (trusted metadata only).

    The five search fields (``start_key``, ``start_rank``, ``slope_num``,
    ``slope_den``, ``intercept``) are the **exact-integer representation of the
    rounded-intercept lookup model**: ``slope_num / slope_den`` is the exact rational
    slope, ``intercept`` is that slope's value at ``start_key`` rounded to an integer
    (half toward zero), and :meth:`predict` additionally truncates the slope term
    toward zero before the search path caps/clamps it.  Its contract is the
    ``SearchResult`` lower-bound window only; a training point may sit up to
    ``epsilon + 1`` away from its prediction.

    That lookup model is **not** the canonical feasible line of the segment: the
    unrounded continuous line the canonical OPLM geometry actually admits is stored
    separately, exactly, in :attr:`certificate` (G5-A.1) — the two objects must not be
    conflated, and no query behaviour depends on the certificate.

    No physical location (BlockId/SlotId) is stored or predicted.
    """

    start_key: RecordKey
    start_rank: int
    slope_num: int
    slope_den: int
    intercept: int
    #: The exact canonical rank line this segment was cut from (G5-A.1).  It is a frozen
    #: field so a segment's search metadata and its proof certificate can never drift
    #: apart; the five search fields above keep their names and values unchanged.
    certificate: RankLineCertificate

    def predict(self, key: RecordKey) -> int:
        """Exact canonical prediction truncated toward zero, then floored at
        0 (matching the official unsigned position cast; the final clamp
        to ``[0, n]`` is applied by the search entry point)."""
        delta = key.value - self.start_key.value
        value = _div_trunc(self.slope_num * delta, self.slope_den) + self.intercept
        return value if value > 0 else 0


def _trunc_div_round_half_toward_zero(num: int, den: int) -> int:
    """``(num + rounding_term) / den`` with C++ integer division.

    ``den`` is always positive here (it is the slope denominator
    ``dx = rectangle[3].x - rectangle[1].x > 0``), so the official
    rounding term
    ``((num < 0) ^ (den < 0) ? -1 : +1) * den / 2`` reduces to
    ``sign(num) * den // 2``.
    """
    rounding_term = _trunc_sign(num) * (den // 2)
    return _div_trunc(num + rounding_term, den)


def _rank_line_certificate(
    geom: "_CanonicalSegmentGeometry", start_rank: int, end_rank: int, end_key: RecordKey
) -> RankLineCertificate:
    """Freeze the exact unrounded canonical line of one segment (G5-A.1 §3).

    ``rectangle[1] = (x1, y1)`` and ``rectangle[3] = (x3, y3)`` are the two corners the
    accepted search model already uses, ``x0`` is the segment's first real key and

    ```text
    slope_num  = y3 - y1
    origin_num = slope_num * (x0 - x1) + y1 * denominator
    denominator = x3 - x1      (made strictly positive)
    ```

    so the represented line is the canonical rational line through those two corners,
    expressed relative to ``x0``.  A one-point segment keeps the exact midpoint of its
    clipped band instead of the existing integer midpoint (``0`` / ``r0.y + r1.y`` / ``2``).
    """
    r0, r1, _r2, r3 = geom.rectangle
    start_key = RecordKey(geom.first)
    if geom.one_point:
        return RankLineCertificate(
            start_key=start_key,
            end_key=end_key,
            start_rank=start_rank,
            end_rank=end_rank,
            slope_num=0,
            origin_num=r0.y + r1.y,
            denominator=2,
        )
    slope_num = r3.y - r1.y
    denominator = r3.x - r1.x
    if denominator < 0:
        # canonical orientation: x increasing, denominator strictly positive
        slope_num = -slope_num
        denominator = -denominator
    if denominator == 0:
        # fail closed: a non-singleton canonical geometry always spans two distinct x
        # values, so a zero x span means the geometry is not a canonical segment
        raise PgmError(
            "a non-singleton canonical geometry must have a non-zero x span "
            "(rectangle[3].x == rectangle[1].x); refusing to build a rank-line "
            "certificate from a degenerate segment"
        )
    return RankLineCertificate(
        start_key=start_key,
        end_key=end_key,
        start_rank=start_rank,
        end_rank=end_rank,
        slope_num=slope_num,
        origin_num=slope_num * (start_key.value - r1.x) + r1.y * denominator,
        denominator=denominator,
    )


def _pgm_segment_from_geometry(geom: "_CanonicalSegmentGeometry",
                               start_rank: int,
                               *,
                               end_rank: int,
                               end_key: RecordKey) -> PgmSegment:
    """Build a :class:`PgmSegment` from a canonical segment geometry,
    reproducing the official integral ``get_floating_point_segment`` in
    exact integers (no float slope / intercept is ever materialised).

    The frozen :class:`RankLineCertificate` of the same geometry is attached in the same
    step, so the rounded search model and its exact proof line cannot become misaligned.
    """
    r0, r1, r2, r3 = geom.rectangle
    start_key = RecordKey(geom.first)
    certificate = _rank_line_certificate(geom, start_rank, end_rank, end_key)
    if geom.one_point:
        a = r0.y
        b = r1.y
        intercept = (a + b) // 2  # both are non-negative
        return PgmSegment(start_key, start_rank, 0, 1, intercept, certificate)

    slope_dx = r3.x - r1.x
    slope_dy = r3.y - r1.y
    if slope_dx < 0:
        # Canonical orientation: x increasing, denominator positive.
        slope_dx = -slope_dx
        slope_dy = -slope_dy
    if slope_dx == 0:
        slope_dx = 1
        slope_dy = 0

    num = slope_dy * (start_key.value - r1.x)
    den = slope_dx
    intercept = _trunc_div_round_half_toward_zero(num, den) + r1.y
    return PgmSegment(start_key, start_rank, slope_dy, slope_dx, intercept, certificate)


def make_segmentation(
    keys: Sequence[RecordKey], epsilon: int
) -> tuple[PgmSegment, ...]:
    """Sequential OPLM segmentation of the real points ``(key_i, i)``.

    Returns the canonical real-data segments in order.  No synthetic
    tail/sentinel point is emitted: the official sentinel's role is
    reproduced at search time by capping the last real segment at ``n``.

    Every segment carries the exact :class:`RankLineCertificate` of its canonical
    geometry, frozen together with the segment's inclusive rank span and its last real
    key (G5-A.1 §3): when the point at rank ``r`` is rejected, the preceding segment ends
    at rank ``r - 1`` and at the previously accepted key, and that rejected point starts
    the next segment; the final segment ends at ``n - 1`` / ``keys[n - 1]``.
    """
    n = len(keys)
    opt = _OptimalPiecewiseLinearModel(epsilon)
    segments: list[PgmSegment] = []
    current_start_rank = 0

    def freeze(end_rank: int, end_key: RecordKey) -> None:
        segments.append(
            _pgm_segment_from_geometry(
                opt.get_segment(), current_start_rank, end_rank=end_rank, end_key=end_key
            )
        )

    def add_point(x: int, y: int) -> None:
        nonlocal current_start_rank
        if not opt.add_point(x, y):
            # rejected at rank y: this segment owns ranks ..y-1 and the previous key
            freeze(y - 1, keys[y - 1])
            current_start_rank = y
            opt._reset()
            if not opt.add_point(x, y):  # pragma: no cover - unreachable
                raise PgmError("OPLM failed to start a new segment")

    if n == 0:
        return ()

    add_point(keys[0].value, 0)
    for i in range(1, n):
        add_point(keys[i].value, i)

    freeze(n - 1, keys[n - 1])
    return tuple(segments)


# ---------------------------------------------------------------------------
# BatchPgmIndex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPgmIndex:
    """An immutable batch PGM over one logical level (trusted metadata).

    Retains only ``epsilon``, the record count ``n``, boundary keys, and
    the canonical real-data segments.  It deliberately does NOT retain
    the full key array or any record contents.
    """

    epsilon: int
    record_count: int
    first_key: Optional[RecordKey]
    last_key: Optional[RecordKey]
    segments: tuple[PgmSegment, ...]

    def _next_intercept(self, index: int, n: int) -> int:
        if index + 1 < len(self.segments):
            return self.segments[index + 1].intercept
        return n

    def search(self, key: RecordKey) -> SearchResult:
        """Return ``SearchResult(pos, lo, hi)`` for ``key``.

        Pure metadata operation: no storage access, no trace event.
        """
        n = self.record_count
        if n == 0:
            return SearchResult(0, 0, 0)

        first_key = self.first_key  # non-None because n > 0
        assert first_key is not None
        last_key = self.last_key  # non-None because n > 0
        assert last_key is not None

        if key < first_key:
            pos = 0
        elif key > last_key:
            pos = n
        else:
            i = self._segment_index_for(key)
            seg = self.segments[i]
            pos = seg.predict(key)
            cap = self._next_intercept(i, n)
            if pos > cap:
                pos = cap
            if pos < 0:
                pos = 0
            if pos > n:
                pos = n

        lo = pos - self.epsilon
        if lo < 0:
            lo = 0
        hi = pos + self.epsilon + 2
        if hi > n:
            hi = n
        return SearchResult(pos, lo, hi)

    def _segment_index_for(self, key: RecordKey) -> int:
        """Index of the rightmost segment whose ``start_key <= key``.

        Over the *real-data* segments (the last one is not a sentinel;
        in-domain keys always have a governing segment, and the last
        segment's prediction is capped at ``n``).
        """
        keys = [s.start_key for s in self.segments]
        pos = bisect.bisect_right(keys, key) - 1
        if pos < 0:
            return 0
        return pos


def build_batch_pgm(keys: Sequence[RecordKey], epsilon: int) -> BatchPgmIndex:
    """Build a :class:`BatchPgmIndex` from globally strictly increasing keys.

    ``keys`` are the flattened, level-local sorted records; ranks are the
    indices ``0..n-1``.  ``epsilon`` must be a non-negative integer.
    """
    if isinstance(epsilon, bool):
        raise PgmError("epsilon must be an int, not a bool")
    if not isinstance(epsilon, int):
        raise PgmError("epsilon must be an int")
    if epsilon < 0:
        raise PgmError("epsilon cannot be negative")

    n = len(keys)
    for left, right in zip(keys, keys[1:]):
        if left >= right:
            raise PgmError(
                f"PGM input keys must be strictly increasing "
                f"(found {left} >= {right})"
            )

    if n == 0:
        return BatchPgmIndex(epsilon, 0, None, None, ())

    segments = tuple(make_segmentation(keys, epsilon))
    assert segments and segments[0].start_rank == 0

    return BatchPgmIndex(
        epsilon,
        n,
        keys[0],
        keys[-1],
        segments,
    )
