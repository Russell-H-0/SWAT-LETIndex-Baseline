"""G1-B2 single-level physical lookup tests.

Covers decisions/0004: basic HIT/MISS, minimal covering blocks, physical-order
scheduling, logical-rank restoration, [lo,hi) restriction, no early stop, no
auto-build, no validation reread, relocation invariance, and the randomized
membership/trace oracles.
"""

from __future__ import annotations

import bisect
import random

import pytest

from enhanced_letindex.config import Config
from enhanced_letindex.engine import SingleLevelLookupResult, TrustedEngine
from enhanced_letindex.identifiers import RecordKey, SlotId
from enhanced_letindex.pgm import SearchResult
from enhanced_letindex.record import Record
from enhanced_letindex.trace import TraceOperation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rec(k):
    return Record(RecordKey(k), f"v{k}")


def build_level(keys, capacity=None, epsilon=4):
    cfg = {}
    if capacity is not None:
        cfg["block_capacity"] = capacity
    eng = TrustedEngine(Config(pgm_epsilon=epsilon, **cfg))
    level_id = eng.build_level([rec(k) for k in keys])
    eng.build_level_pgm(level_id)
    return eng, level_id, sorted(keys)


def lookup_slots(eng, level_id, key):
    """Run one lookup and return (result, [slot_ids in observation order])."""
    before = len(eng.trace)
    result = eng.lookup_level(level_id, RecordKey(key))
    events = eng.trace.events()[before:]
    return result, [ev.slot_id.value for ev in events]


def expected_covering_slots(eng, level_id, sr, capacity):
    """Independently derive the minimal covering slot set from SearchResult."""
    if sr.lo == sr.hi:
        return []
    first = sr.lo // capacity
    last = (sr.hi - 1) // capacity
    block_ids = eng.level(level_id).block_ids[first : last + 1]
    return sorted(eng.mapping.lookup(b).value for b in block_ids)


def scramble_mapping(eng, level_id, slot_perm):
    """Relocate each block to a distinct fresh slot per ``slot_perm``.

    ``slot_perm[rank]`` is the destination SlotId for the block at logical
    rank ``rank``.  Destinations must be distinct, currently free, and live
    beyond the initial sequential placement.
    """
    block_ids = eng.level(level_id).block_ids
    n = len(block_ids)
    dest = [SlotId(v) for v in slot_perm]
    assert len(dest) == n
    assert len(set(dest)) == n
    for d in dest:
        assert not eng.storage.contains(d)
    for block_id, d in zip(block_ids, dest):
        eng.relocate_block(block_id, d)


# ---------------------------------------------------------------------------
# 24.2 basic HIT / 24.3 basic MISS
# ---------------------------------------------------------------------------


def test_basic_hit_single_block():
    eng, level_id, keys = build_level([10, 20, 30, 40], capacity=8)
    result, slots = lookup_slots(eng, level_id, 30)
    assert result.found is True
    assert result.record.key == RecordKey(30)
    assert result.lower_bound_rank == 2
    # single block level: one READ of the single physical slot
    assert slots == [0]
    # every lookup event is a READ; no WRITE
    events = eng.trace.events()
    assert all(ev.operation == TraceOperation.READ for ev in events[-len(slots) :])
    assert result.search_result.lo <= 2 < result.search_result.hi


def test_basic_miss_between_absent():
    eng, level_id, keys = build_level([10, 20, 30, 40], capacity=8)
    result, slots = lookup_slots(eng, level_id, 25)
    assert result.found is False
    assert result.record is None
    assert result.lower_bound_rank == bisect.bisect_left(keys, 25) == 2
    assert slots == [0]  # full level is one block


def test_basic_miss_below_min():
    eng, level_id, keys = build_level([10, 20, 30, 40], capacity=8)
    result, _ = lookup_slots(eng, level_id, 5)
    assert result.found is False
    assert result.record is None
    assert result.lower_bound_rank == 0


def test_basic_miss_above_max():
    eng, level_id, keys = build_level([10, 20, 30, 40], capacity=8)
    result, _ = lookup_slots(eng, level_id, 500)
    assert result.found is False
    assert result.record is None
    assert result.lower_bound_rank == len(keys)


# ---------------------------------------------------------------------------
# 24.4 empty level
# ---------------------------------------------------------------------------


def test_empty_level_lookup_miss_zero_read():
    eng, level_id, keys = build_level([], capacity=8)
    result, slots = lookup_slots(eng, level_id, 7)
    assert result.found is False
    assert result.record is None
    assert result.lower_bound_rank == 0
    assert slots == []
    assert result.search_result == SearchResult(0, 0, 0)


# ---------------------------------------------------------------------------
# 24.5 / 24.6 / 24.16 cross-block interval + minimal covering blocks
# ---------------------------------------------------------------------------


def test_cross_block_interval_reads_minimal_covering_set():
    # C=4; keys 0..15 -> B0 0..3, B1 4..7, B2 8..11, B3 12..15
    eng, level_id, keys = build_level(list(range(16)), capacity=4)
    idx = eng.pgm_index(level_id)
    sr = idx.search(RecordKey(8))
    result, slots = lookup_slots(eng, level_id, 8)
    expected = expected_covering_slots(eng, level_id, sr, 4)
    assert slots == expected
    # this key's [lo,hi) genuinely crosses >= 2 blocks (structural sanity)
    first = sr.lo // 4
    last = (sr.hi - 1) // 4
    assert last - first + 1 >= 2
    assert result.found is True
    assert result.record.key == RecordKey(8)
    assert result.lower_bound_rank == 8


def test_minimal_candidate_blocks_no_neighbour():
    # C=4, 20 keys -> 5 blocks.  A key inside B2 must read only its covering
    # set (B2 alone, or B2+B3 if [lo,hi) spans), never B1/B3 as "+/-1".
    eng, level_id, keys = build_level(list(range(20)), capacity=4, epsilon=2)
    idx = eng.pgm_index(level_id)

    q = 10  # rank 10 -> B2 (8..11)
    sr = idx.search(RecordKey(q))
    result, slots = lookup_slots(eng, level_id, q)
    expected = expected_covering_slots(eng, level_id, sr, 4)
    assert slots == expected
    first = sr.lo // 4
    last = (sr.hi - 1) // 4
    assert set(slots) == set(expected)
    assert len(slots) == (last - first + 1)
    assert result.found is True


# ---------------------------------------------------------------------------
# 24.7 read every candidate exactly once / no early stop
# ---------------------------------------------------------------------------


def test_no_data_dependent_early_stop():
    # A key in the FIRST candidate block still forces reads of all later
    # candidate blocks (no early stop, no data-dependent set).
    eng, level_id, keys = build_level(list(range(8)), capacity=4, epsilon=3)
    idx = eng.pgm_index(level_id)
    q = 2  # in B0 (0..3); [lo,hi)=[0,6] -> covers B0 and B1
    sr = idx.search(RecordKey(q))
    assert sr.lo // 4 == 0 and (sr.hi - 1) // 4 >= 1  # spans >= 2 blocks
    result, slots = lookup_slots(eng, level_id, q)
    expected = expected_covering_slots(eng, level_id, sr, 4)
    assert slots == expected
    assert len(slots) == len(set(slots)) == len(expected)
    assert result.found is True


def test_read_exactly_once_with_scrambled_mapping():
    # 3-block level, scrambled physical order; all candidate blocks read once.
    eng, level_id, keys = build_level(list(range(12)), capacity=4, epsilon=4)
    scramble_mapping(eng, level_id, [10, 11, 8])
    idx = eng.pgm_index(level_id)
    q = 2
    sr = idx.search(RecordKey(q))
    first = sr.lo // 4
    last = (sr.hi - 1) // 4
    assert last - first + 1 >= 2
    result, slots = lookup_slots(eng, level_id, q)
    expected = expected_covering_slots(eng, level_id, sr, 4)
    assert slots == expected
    assert len(slots) == len(set(slots)) == len(expected)
    assert result.found is True


# ---------------------------------------------------------------------------
# 24.8 / 24.9 physical-order schedule + restore logical order
# ---------------------------------------------------------------------------


def test_physical_order_schedule_is_slotid_ascending():
    eng, level_id, keys = build_level(list(range(16)), capacity=4)
    scramble_mapping(eng, level_id, [9, 10, 11, 8])
    # logical slots now [0,9,11,10]; a 3-block candidate read must be
    # ascending by slot.
    idx = eng.pgm_index(level_id)
    q = 7
    sr = idx.search(RecordKey(q))
    first = sr.lo // 4
    last = (sr.hi - 1) // 4
    block_ids = eng.level(level_id).block_ids[first : last + 1]
    logical_slots = [eng.mapping.lookup(b).value for b in block_ids]
    assert logical_slots != sorted(logical_slots), "fixture must be scrambled"
    result, slots = lookup_slots(eng, level_id, q)
    assert slots == sorted(logical_slots)
    assert slots != logical_slots
    assert len(slots) == len(set(slots)) == len(expected_covering_slots(eng, level_id, sr, 4))
    assert result.found is True
    assert result.record.key == RecordKey(7)
    assert result.lower_bound_rank == bisect.bisect_left(keys, 7)


def test_restore_logical_order_after_scrambled_read():
    eng, level_id, keys = build_level(list(range(16)), capacity=4)
    scramble_mapping(eng, level_id, [9, 10, 11, 8])
    idx = eng.pgm_index(level_id)
    for q, want in [
        (0, bisect.bisect_left(keys, 0)),
        (15, bisect.bisect_left(keys, 15)),
        (16, 16),   # above max
        (6, 6),
        (10, 10),
    ]:
        sr = idx.search(RecordKey(q))
        result, slots = lookup_slots(eng, level_id, q)
        expected = expected_covering_slots(eng, level_id, sr, 4)
        assert slots == expected, (q, slots, expected)
        assert result.lower_bound_rank == want
        if q in keys:
            assert result.found is True
            assert result.record.key == RecordKey(q)
        else:
            assert result.found is False
            assert result.record is None


# ---------------------------------------------------------------------------
# 24.10 relocation invariance
# ---------------------------------------------------------------------------


def test_relocation_invariance():
    eng, level_id, keys = build_level(list(range(16)), capacity=4)
    key_q = 7
    before_sr = eng.pgm_index(level_id).search(RecordKey(key_q))
    before_res, before_slots = lookup_slots(eng, level_id, key_q)

    scramble_mapping(eng, level_id, [9, 10, 11, 8])
    eng.trace.clear()

    after_sr = eng.pgm_index(level_id).search(RecordKey(key_q))
    after_res, after_slots = lookup_slots(eng, level_id, key_q)

    assert before_sr == after_sr
    assert before_res.found == after_res.found
    assert before_res.lower_bound_rank == after_res.lower_bound_rank
    assert before_res.record == after_res.record
    assert before_slots != after_slots  # physical trace changed


# ---------------------------------------------------------------------------
# 24.11 present-key interval containment
# ---------------------------------------------------------------------------


def test_present_key_interval_containment():
    keys = sorted(set(random.Random(1).sample(range(-100, 1000), 30)))
    eng, level_id, _ = build_level(keys, capacity=5, epsilon=3)
    idx = eng.pgm_index(level_id)
    for i, key in enumerate(keys):
        sr = idx.search(RecordKey(key))
        assert sr.lo <= i < sr.hi
        result, slots = lookup_slots(eng, level_id, key)
        assert result.found is True
        assert result.record.key == RecordKey(key)
        assert result.lower_bound_rank == i


# ---------------------------------------------------------------------------
# 24.12 randomized membership oracle
# ---------------------------------------------------------------------------


def randomized_membership_oracle(seed):
    rng = random.Random(seed)
    for _ in range(50):
        n = rng.randint(1, 60)
        spacing = rng.choice([1, 2, 3, 10])
        lo = rng.randint(-1000, 0)
        keys = []
        cur = lo
        for _ in range(n):
            cur += rng.randint(1, spacing)
            keys.append(cur)
        keys = list(dict.fromkeys(keys))
        eps = rng.choice([0, 1, 2, 3, 5])
        capacity = rng.choice([2, 4, 7])
        eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=capacity))
        level_id = eng.build_level([rec(k) for k in keys])
        idx = eng.build_level_pgm(level_id)

        queries = list(keys)
        queries += [min(keys) - 1, max(keys) + 1]
        for a, b in zip(keys, keys[1:]):
            queries.append((a + b) // 2)
        for _ in range(10):
            queries.append(rng.randint(min(keys) - 5, max(keys) + 5))
        queries += [rng.randint(-10**6, 10**6) for _ in range(3)]

        for q in queries:
            sr = idx.search(RecordKey(q))
            result, slots = lookup_slots(eng, level_id, q)
            global_pos = bisect.bisect_left(keys, q)
            expected_found = global_pos < len(keys) and keys[global_pos] == q
            assert result.lower_bound_rank == global_pos, (q, keys)
            assert result.found == expected_found
            if expected_found:
                assert result.record.key.value == q
            else:
                assert result.record is None
            # trace oracle: reads == minimal covering slot set
            expected = expected_covering_slots(eng, level_id, sr, capacity)
            assert slots == expected, (q, slots, expected)


def test_randomized_membership_oracle_many_seeds():
    for seed in range(6):
        randomized_membership_oracle(2000 + seed)


# ---------------------------------------------------------------------------
# 24.13 randomized trace oracle
# ---------------------------------------------------------------------------


def randomized_trace_oracle(seed):
    rng = random.Random(seed)
    for _ in range(40):
        n = rng.randint(1, 40)
        keys = sorted(rng.sample(range(-500, 5000), n))
        capacity = rng.choice([2, 3, 5, 8])
        eps = rng.choice([0, 1, 2, 4])
        eng = TrustedEngine(Config(pgm_epsilon=eps, block_capacity=capacity))
        level_id = eng.build_level([rec(k) for k in keys])
        idx = eng.build_level_pgm(level_id)

        q = rng.randint(-600, 5500)
        sr = idx.search(RecordKey(q))
        result, slots = lookup_slots(eng, level_id, q)
        expected = expected_covering_slots(eng, level_id, sr, capacity)
        assert slots == expected
        assert len(slots) == len(expected)
        assert result.lower_bound_rank == bisect.bisect_left(keys, q)


def test_randomized_trace_oracle_multiple_seeds():
    for seed in range(5):
        randomized_trace_oracle(1000 + seed)
    for seed in range(5):
        randomized_trace_oracle(5000 + seed)


# ---------------------------------------------------------------------------
# 24.14 / 24.15 no auto-build / no validation reread
# ---------------------------------------------------------------------------


def test_no_automatic_pgm_build():
    eng = TrustedEngine(Config(pgm_epsilon=2, block_capacity=2))
    level_id = eng.build_level([rec(k) for k in (10, 20, 30)])
    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.lookup_level(level_id, RecordKey(20))
    assert len(eng.trace) == 0  # no auto-build trace


def test_lookup_reads_only_candidate_blocks_not_full_level():
    # level with 8 blocks; a narrow query must read only its covering blocks.
    eng, level_id, keys = build_level(list(range(16)), capacity=2, epsilon=1)
    idx = eng.pgm_index(level_id)
    q = 5
    sr = idx.search(RecordKey(q))
    result, slots = lookup_slots(eng, level_id, q)
    expected = expected_covering_slots(eng, level_id, sr, 2)
    assert slots == expected
    assert len(slots) <= 3  # no full-level scan (would be 8 reads)
    assert result.found is True


# ---------------------------------------------------------------------------
# 24.17 boundary q > max
# ---------------------------------------------------------------------------


def test_q_above_max_reads_covering_blocks_not_zero_read():
    eng, level_id, keys = build_level(list(range(8)), capacity=4, epsilon=1)
    result, slots = lookup_slots(eng, level_id, 1000)
    assert result.found is False
    assert result.record is None
    assert result.lower_bound_rank == len(keys)
    sr = eng.pgm_index(level_id).search(RecordKey(1000))
    assert sr.lo == len(keys) - 1
    assert sr.hi == len(keys)
    # covering block for rank n-1 is the last block -> exactly one READ
    last_block = eng.level(level_id).block_ids[-1]
    assert slots == [eng.mapping.lookup(last_block).value]


# ---------------------------------------------------------------------------
# result object shape
# ---------------------------------------------------------------------------


def test_result_exposes_no_physical_internals():
    eng, level_id, keys = build_level(list(range(8)), capacity=2)
    result, _ = lookup_slots(eng, level_id, 3)
    assert isinstance(result, SingleLevelLookupResult)
    data = vars(result)
    assert set(data) == {"found", "record", "lower_bound_rank", "search_result"}
    assert isinstance(result.search_result, SearchResult)
    assert set(vars(result.search_result)) == {"pos", "lo", "hi"}
