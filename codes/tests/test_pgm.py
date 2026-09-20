"""G1-B1 batch PGM construction tests.

Covers:
  - config activation + validation;
  - basic segmentation edge cases;
  - per-segment legality and maximal-prefix properties;
  - an INDEPENDENT exact DP minimal-segment oracle (validates that the
    sequential OPLM reproduces the canonical minimal segmentation);
  - a lower_bound interval oracle driven by ``bisect_left``;
  - no-membership-cheating / no-full-key-retention checks;
  - block-capacity and physical-placement invariance;
  - trace semantics (construction READs, search = zero trace).

The DP oracle uses exact ``fractions.Fraction`` and does not reuse any
OPLM internal feasibility function.
"""

from __future__ import annotations

import bisect
import random
from fractions import Fraction

import pytest

from enhanced_letindex.config import Config, ConfigError
from enhanced_letindex.engine import TrustedEngine
from enhanced_letindex.identifiers import RecordKey, SlotId
from enhanced_letindex.pgm import (
    BatchPgmIndex,
    PgmConfigurationError,
    PgmError,
    make_segmentation,
)
from enhanced_letindex.record import Record
from enhanced_letindex.trace import TraceOperation

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rec(k):
    return Record(RecordKey(k), f"v{k}")


def build_level(keys, capacity=None):
    """Build a level from integer keys and return (engine, level_id, keys)."""
    cfg = {"pgm_epsilon": 4}
    if capacity is not None:
        cfg["block_capacity"] = capacity
    eng = TrustedEngine(Config(**cfg))
    level_id = eng.build_level([rec(k) for k in keys])
    return eng, level_id, sorted(keys)


def level_keys(eng, level_id):
    keys = []
    for block_id in eng.level(level_id).block_ids:
        for r in eng.read_block(block_id).records:
            keys.append(r.key.value)
    return keys


def unique_sorted(rng, n, lo, hi):
    return sorted(rng.sample(range(lo, hi), n))


# ---------------------------------------------------------------------------
# independent exact feasibility + DP oracle (does NOT reuse OPLM internals)
# ---------------------------------------------------------------------------


def _window_feasible(keys, ranks, lo, hi, eps):
    """Is points[lo..hi] (inclusive) feasible under one epsilon-band line?

    Exact, using Fraction.  Single point is always feasible.  Points must
    be strictly increasing in keys.
    """
    if hi <= lo:
        return True
    max_low = None
    min_high = None
    for p in range(lo, hi + 1):
        for q in range(p, hi + 1):
            yp = ranks[p]
            yq = ranks[q]
            if p == q:
                continue
            lp = max(0, yp - eps)
            up = yp + eps
            lq = max(0, yq - eps)
            uq = yq + eps
            dx = keys[q] - keys[p]
            assert dx > 0
            low_slope = Fraction(lq - up, dx)
            high_slope = Fraction(uq - lp, dx)
            if max_low is None or low_slope > max_low:
                max_low = low_slope
            if min_high is None or high_slope < min_high:
                min_high = high_slope
    return max_low <= min_high


def dp_min_segments(keys, ranks, eps):
    """Minimum number of feasible contiguous segments covering all points."""
    n = len(keys)
    inf = n + 1
    dp = [inf] * (n + 1)
    dp[0] = 0
    feasible = {}
    for i in range(n):
        for j in range(i, n):
            feasible[(i, j)] = _window_feasible(keys, ranks, i, j, eps)
    for j in range(1, n + 1):
        for i in range(j):
            if feasible[(i, j - 1)]:
                dp[j] = min(dp[j], dp[i] + 1)
    return dp[n]


# ---------------------------------------------------------------------------
# 7.2 config tests
# ---------------------------------------------------------------------------


def test_config_accepts_none_and_nonnegative_ints():
    assert Config(pgm_epsilon=None).pgm_epsilon is None
    assert Config(pgm_epsilon=0).pgm_epsilon == 0
    assert Config(pgm_epsilon=1).pgm_epsilon == 1
    assert Config(pgm_epsilon=10_000_000).pgm_epsilon == 10_000_000


def test_config_rejects_negative():
    with pytest.raises(ConfigError):
        Config(pgm_epsilon=-1)


def test_config_rejects_float():
    with pytest.raises(ConfigError):
        Config(pgm_epsilon=1.0)  # type: ignore[arg-type]


def test_config_rejects_bool():
    with pytest.raises(ConfigError):
        Config(pgm_epsilon=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        Config(pgm_epsilon=False)  # type: ignore[arg-type]


def test_build_pgm_with_none_epsilon_fails_loudly():
    eng = TrustedEngine(Config(pgm_epsilon=None, block_capacity=3))
    level_id = eng.build_level([rec(k) for k in (10, 20, 30)])
    with pytest.raises(PgmConfigurationError):
        eng.build_level_pgm(level_id)


# ---------------------------------------------------------------------------
# 7.3 basic segmentation
# ---------------------------------------------------------------------------


def test_build_batch_pgm_over_empty_level():
    from enhanced_letindex.pgm import SearchResult

    eng = TrustedEngine(Config(pgm_epsilon=2))
    level_id = eng.build_level([])
    idx = eng.build_level_pgm(level_id)
    assert idx.record_count == 0
    assert idx.segments == ()
    assert idx.first_key is None
    assert idx.last_key is None
    # an empty index maps every key to the empty candidate range
    assert idx.search(RecordKey(5)) == SearchResult(0, 0, 0)
    assert idx.search(RecordKey(-1)) == SearchResult(0, 0, 0)


def test_empty_search_returns_zero_result():
    from enhanced_letindex.pgm import SearchResult

    eng = TrustedEngine(Config(pgm_epsilon=2))
    level_id = eng.build_level([])
    idx = eng.build_level_pgm(level_id)
    assert idx.search(RecordKey(99)) == SearchResult(0, 0, 0)


def test_single_point_segment_is_flat():
    segs = make_segmentation([RecordKey(42)], 0)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.slope_num == 0
    assert seg.slope_den == 1
    assert seg.start_key == RecordKey(42)
    assert seg.start_rank == 0
    assert seg.intercept == 0
    assert seg.predict(RecordKey(42)) == 0


def test_single_point_with_epsilon_intercept_is_band_midpoint():
    segs = make_segmentation([RecordKey(7)], 3)
    assert segs[0].slope_num == 0
    # rank 0 with epsilon 3 -> epsilon band [0, 3], integer midpoint 1
    assert segs[0].intercept == 1


def test_two_collinear_points_single_segment():
    segs = make_segmentation([RecordKey(0), RecordKey(1)], 0)
    assert len(segs) == 1
    assert (segs[0].slope_num, segs[0].slope_den) == (1, 1)


def test_exactly_collinear_many_points_single_segment():
    segs = make_segmentation([RecordKey(k) for k in (0, 1, 2, 3, 4, 5)], 0)
    assert len(segs) == 1


def test_non_collinear_points_split_segments_at_eps0():
    # 0->0, 1->1, 100->2: two lines needed when epsilon = 0
    segs = make_segmentation([RecordKey(0), RecordKey(1), RecordKey(100)], 0)
    assert len(segs) == 2


def test_epsilon0_single_bad_point_is_its_own_segment():
    segs = make_segmentation([RecordKey(0), RecordKey(10), RecordKey(11)], 0)
    # 0->0,10->1 slope 1/10; 11->2 off that line: 2 segments
    assert len(segs) == 2


def test_large_key_gaps_allowed():
    keys = [0, 10**9, 2 * 10**9, 3 * 10**9, 10**12]
    segs = make_segmentation([RecordKey(k) for k in keys], 2)
    assert len(segs) >= 1
    assert [s.start_key.value for s in segs][0] == 0


def test_negative_keys_with_rank_from_zero():
    keys = [-10, -5, -1, 0, 3, 9]
    segs = make_segmentation([RecordKey(k) for k in keys], 0)
    assert len(segs) >= 1
    assert segs[0].start_rank == 0
    # ranks are indices: key -10 -> 0, ... key 9 -> 5
    assert segs[-1].start_rank <= 5


def test_very_large_integer_keys_ok():
    keys = [-(10**18), -(10**18) + 1, 0, 10**18]
    segs = make_segmentation([RecordKey(k) for k in keys], 3)
    assert len(segs) >= 1


# ---------------------------------------------------------------------------
# 7.4 / 7.5 segment legality and maximal-prefix property
# ---------------------------------------------------------------------------


def segment_point_ranges(keys, segs):
    """Assign each real point to its segment; return per-segment point lists."""
    starts = [s.start_rank for s in segs]
    starts.append(len(keys))
    ranges = []
    for si in range(len(segs)):
        ranges.append(list(range(starts[si], starts[si + 1])))
    return ranges


def _check_segment_legality(keys, ranks, segs, eps):
    ranges = segment_point_ranges(keys, segs)
    for pts in ranges:
        assert pts, "empty segment"
        lo = pts[0]
        hi = pts[-1]
        assert _window_feasible(keys, ranks, lo, hi, eps), (
            f"segment points {range(lo, hi + 1)} not epsilon-feasible"
        )


def _check_maximal_prefix(keys, ranks, segs, eps):
    for si in range(len(segs) - 1):
        # current segment must be feasible...
        cur_lo = segs[si].start_rank
        nxt_lo = segs[si + 1].start_rank
        # ...but adding the next segment's first real point must be infeasible
        assert not _window_feasible(keys, ranks, cur_lo, nxt_lo, eps), (
            f"segment {si} could absorb the next segment's first point"
        )


@pytest.mark.parametrize("eps", [0, 1, 2, 3, 5])
def test_legality_and_maximal_prefix_on_randomized_sets(eps):
    rng = random.Random(1000 + eps)
    for _ in range(40):
        n = rng.randint(2, 25)
        keys = unique_sorted(rng, n, -50, 200)
        ranks = list(range(n))
        segs = make_segmentation([RecordKey(k) for k in keys], eps)
        _check_segment_legality(keys, ranks, segs, eps)
        _check_maximal_prefix(keys, ranks, segs, eps)


# ---------------------------------------------------------------------------
# 7.6 independent DP minimal-segment oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0, 1, 2, 3, 4, 5])
def test_sequential_oplm_matches_dp_minimal_count(eps):
    rng = random.Random(2026 + eps)
    for _ in range(30):
        n = rng.randint(1, 14)
        keys = unique_sorted(rng, n, 0, 100)
        ranks = list(range(n))
        segs = make_segmentation([RecordKey(k) for k in keys], eps)
        optimal = dp_min_segments(keys, ranks, eps)
        assert len(segs) == optimal, (
            f"eps={eps} keys={keys}: sequential={len(segs)} optimal={optimal}"
        )


def test_dp_oracle_case_empty_trivial():
    assert dp_min_segments([], [], 3) == 0


# ---------------------------------------------------------------------------
# 7.7 / 7.8 SearchResult window + lower_bound + no-cheating
# ---------------------------------------------------------------------------


def _search_window_holds(idx, n, q, eps):
    res = idx.search(RecordKey(q))
    assert 0 <= res.lo <= res.hi <= n, (q, res)
    assert 0 <= res.pos <= n, (q, res)
    assert res.lo == max(0, res.pos - eps), (q, res)
    assert res.hi == min(n, res.pos + eps + 2), (q, res)
    return res


def test_search_window_boundaries():
    eng, level_id, keys = build_level(list(range(20)), capacity=4)
    eps = eng.config.pgm_epsilon
    idx = eng.build_level_pgm(level_id)
    n = len(keys)
    for q in (-100, -1, 0, 5, 19, 20, 100):
        _search_window_holds(idx, n, q, eps)


def test_lower_bound_oracle_randomized():
    rng = random.Random(4321)
    for _ in range(60):
        n = rng.randint(1, 30)
        base = rng.randint(-20, 20)
        keys = sorted(base + 3 * i + rng.randint(0, 1) for i in range(n))
        keys = sorted(set(keys))
        if not keys:
            continue
        eps = rng.choice([0, 1, 2, 3, 4, 6])
        eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=rng.randint(2, 5)))
        level_id = eng.build_level([rec(k) for k in keys])
        idx = eng.build_level_pgm(level_id)

        queries = list(keys)
        for a, b in zip(keys, keys[1:]):
            queries.append((a + b) // 2)
        queries += [min(keys) - rng.randint(1, 3), max(keys) + rng.randint(1, 3)]
        queries += [rng.randint(min(keys) - 5, max(keys) + 5) for _ in range(15)]

        for q in queries:
            res = idx.search(RecordKey(q))
            global_lb = bisect.bisect_left(keys, q)
            local_lb = bisect.bisect_left(keys, q, res.lo, res.hi)
            assert local_lb == global_lb, (q, keys, res, global_lb, local_lb)
            _search_window_holds(idx, len(keys), q, eps)


def test_boundary_queries_against_oracle():
    eng, level_id, keys = build_level([5, 9, 13, 20, 31], capacity=2)
    idx = eng.build_level_pgm(level_id)
    n = len(keys)
    for q in (-10**9, -1, 2, 4, 5, 31, 50, 10**9):
        res = idx.search(RecordKey(q))
        assert bisect.bisect_left(keys, q, res.lo, res.hi) == bisect.bisect_left(keys, q)
        _search_window_holds(idx, n, q, eng.config.pgm_epsilon)


def test_absent_key_returns_candidate_range_not_membership():
    # fixture keys live in the TEST, not inside BatchPgmIndex
    keys = [10, 20, 30, 40]
    eng, level_id, _ = build_level(keys, capacity=2)
    idx = eng.build_level_pgm(level_id)

    query = 25  # absent, between 20 and 30
    assert query not in keys  # the query key truly does not exist

    res = idx.search(RecordKey(query))
    # SearchResult only expresses a candidate range: it carries no HIT/MISS
    # and no Record.
    assert not hasattr(res, "found")
    assert not hasattr(res, "record")
    assert not hasattr(res, "key")

    # contract: the true insertion point IS inside the candidate range
    global_lb = bisect.bisect_left(keys, query)
    local_lb = bisect.bisect_left(keys, query, res.lo, res.hi)
    assert local_lb == global_lb

    # and the record at the insertion point is NOT the query key
    if global_lb < len(keys):
        assert keys[global_lb] != query
    else:  # pragma: no cover - insertion at end
        assert global_lb == len(keys)


def test_pgm_segment_does_not_carry_physical_ids():
    from enhanced_letindex.pgm import PgmSegment

    seg = make_segmentation([RecordKey(k) for k in (0, 1, 2)], 0)[0]
    for attr in ("start_key", "start_rank", "slope_num", "slope_den", "intercept"):
        assert hasattr(seg, attr)
    assert not hasattr(seg, "slot_id") and not hasattr(seg, "block_id")


def test_pathological_datasets_lower_bound_contract():
    """Light, deterministic pathological datasets (Finding 3).

    Large / irregular gaps, negative-to-positive keys, eps >= n, eps = 0
    and segment-boundary absent queries must all satisfy the lookup
    contract `bisect_left(keys, q, lo, hi) == bisect_left(keys, q)`.
    """
    data_epsilon = [
        ([0, 1, 2, 10**6, 10**12], 2),
        ([0, 1, 2, 10**6, 10**12], 1),
        ([0, 2, 3, 100, 9000, 9001, 10**9], 1),
        ([-10**9, -10**6, -3, -2, 0, 0 + 5, 10**6], 2),
        ([-50, -1, 0, 1, 7, 8, 100], 3),
        (list(range(6)), 10),  # epsilon >= n
        (list(range(6)), 0),   # epsilon = 0
        ([1, 2, 3, 4, 5, 6], 2),
    ]
    for keys, eps in data_epsilon:
        keys = sorted(set(keys))
        eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=3))
        level_id = eng.build_level([rec(k) for k in keys])
        idx = eng.build_level_pgm(level_id)
        n = len(keys)

        def check(q):
            res = idx.search(RecordKey(q))
            assert res.lo == max(0, res.pos - eps)
            assert res.hi == min(n, res.pos + eps + 2)
            assert 0 <= res.lo <= res.hi <= n
            assert 0 <= res.pos <= n
            assert bisect.bisect_left(keys, q, res.lo, res.hi) == bisect.bisect_left(keys, q)

        for k in keys:
            check(k)
        # absent queries: inside and just outside the key range, plus the
        # values right around the big-gap segment boundaries
        for q in (0 - 1, max(keys) + 1, -10**12, 10**13, 1 + 1,
                  10**6 - 1, 10**6 + 1, 9000 - 1, -10**6 + 1):
            check(q)


def test_batch_pgm_does_not_retain_full_keys():
    segs = make_segmentation([RecordKey(k) for k in list(range(10))], 1)
    idx = BatchPgmIndex(1, 10, RecordKey(0), RecordKey(9), segs)
    # only metadata fields exist; no keys/records container
    data = vars(idx)
    assert set(data) == {"epsilon", "record_count", "first_key", "last_key", "segments"}
    # segments store boundary info + line params, not the key sequences; the G5-A.1
    # certificate adds exactly one exact-rank-line metadata field per segment
    for seg in idx.segments:
        assert set(vars(seg)) == {
            "start_key", "start_rank", "slope_num", "slope_den", "intercept",
            "certificate",
        }
        assert set(vars(seg.certificate)) == {
            "start_key", "end_key", "start_rank", "end_rank", "slope_num",
            "origin_num", "denominator",
        }


# ---------------------------------------------------------------------------
# 7.9 block-capacity invariance
# ---------------------------------------------------------------------------


def _pgm_fingerprint(keys, capacity, eps):
    eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=capacity))
    level_id = eng.build_level([rec(k) for k in keys])
    idx = eng.build_level_pgm(level_id)
    assert [s.start_key for s in idx.segments][0] == RecordKey(keys[0])
    return len(idx.segments), tuple(
        (s.start_key.value, s.start_rank, s.slope_num, s.slope_den, s.intercept)
        for s in idx.segments
    )


def test_block_capacity_invariance():
    keys = sorted(set(random.Random(5).sample(range(-20, 500), 50)))
    eps = 3
    fp2 = _pgm_fingerprint(keys, 2, eps)
    fp5 = _pgm_fingerprint(keys, 5, eps)
    fp11 = _pgm_fingerprint(keys, 11, eps)
    assert fp2 == fp5 == fp11


# ---------------------------------------------------------------------------
# 7.10 physical-placement invariance
# ---------------------------------------------------------------------------


def test_physical_placement_invariance_after_relocation():
    keys = sorted(set(random.Random(9).sample(range(-100, 900), 60)))
    eps = 3
    eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=4))
    level_id = eng.build_level([rec(k) for k in keys])
    before = eng.build_level_pgm(level_id)
    fp_before = tuple(
        (s.start_key.value, s.start_rank, s.slope_num, s.slope_den, s.intercept)
        for s in before.segments
    )

    # functionally relocate every block to a fresh high slot
    block_ids = eng.level(level_id).block_ids
    for i, b in enumerate(block_ids):
        eng.relocate_block(b, SlotId(1000 + i))

    after = eng.build_level_pgm(level_id)
    fp_after = tuple(
        (s.start_key.value, s.start_rank, s.slope_num, s.slope_den, s.intercept)
        for s in after.segments
    )
    assert fp_before == fp_after


# ---------------------------------------------------------------------------
# 7.11 trace tests
# ---------------------------------------------------------------------------


def test_construction_produces_one_read_per_block_and_no_write():
    eng, level_id, keys = build_level(list(range(10)), capacity=3)
    block_count = eng.level(level_id).size
    eng.trace.clear()

    eng.build_level_pgm(level_id)

    events = eng.trace.events()
    assert all(e.operation is TraceOperation.READ for e in events)
    assert len(events) == block_count
    # reads follow logical block order against current physical slots
    expect_slots = [eng.physical_slot(b) for b in eng.level(level_id).block_ids]
    assert [e.slot_id for e in events] == expect_slots


def test_search_produces_zero_trace_events():
    eng, level_id, keys = build_level(list(range(20)), capacity=4)
    idx = eng.build_level_pgm(level_id)
    eng.trace.clear()
    assert len(eng.trace) == 0

    for q in (-5, 0, 3, 10, 19, 25):
        eng.search_level_pgm(level_id, RecordKey(q))
    idx.search(RecordKey(7))

    assert len(eng.trace) == 0


def test_level_block_append_invalidates_pgm():
    eng, level_id, _ = build_level([1, 2, 3], capacity=2)
    idx = eng.build_level_pgm(level_id)
    assert eng.pgm_index(level_id) is idx
    # mutate the level via the public path
    extra = eng.create_block([rec(99)], level_id=level_id)
    assert extra is not None
    with pytest.raises(KeyError):
        eng.pgm_index(level_id)
