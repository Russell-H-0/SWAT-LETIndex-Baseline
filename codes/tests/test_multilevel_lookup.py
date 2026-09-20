"""G1-C multi-level lookup tests.

Covers `decisions/0005-g1-c-multilevel-lookup.md`: caller-authoritative ordered
level plans, empty-level skipping, no small-level/binary fallback,
preflight-before-any-READ, exact G1-B2 reuse per level, cross-level
hit-and-stop, first-level-wins duplicate-key precedence, exact trace
composition, and the multi-level result contract.

Test labels ``# [n]`` refer to the required-test list in GitHub Issue #3.
"""

from __future__ import annotations

import random

import pytest

from enhanced_letindex.config import Config
from enhanced_letindex.engine import (
    DuplicateLevelInPlanError,
    MultiLevelLookupResult,
    TrustedEngine,
    UnknownLevelError,
)
from enhanced_letindex.identifiers import BlockId, LevelId, RecordKey, SlotId
from enhanced_letindex.record import Record
from enhanced_letindex.trace import TraceOperation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rec(key, tag="v"):
    """A record whose value names the level that owns it (tag per level)."""
    return Record(RecordKey(key), f"{tag}:{key}")


def build_engine(capacity=4, epsilon=2):
    return TrustedEngine(Config(block_capacity=capacity, pgm_epsilon=epsilon))


def add_level(engine, keys, tag="v", build_pgm=True):
    """Build a level with ``keys`` (empty list -> empty level, no blocks)."""
    level_id = engine.build_level([rec(k, tag) for k in keys])
    if build_pgm:
        engine.build_level_pgm(level_id)
    return level_id


def observe(engine, plan, key):
    """Run one multi-level query on a cleared trace.

    Returns ``(result, [slot ids in observation order], events)``.
    """
    engine.trace.clear()
    result = engine.lookup_levels(list(plan), RecordKey(key))
    events = engine.trace.events()
    return result, [ev.slot_id.value for ev in events], events


def level_cover(engine, level_id, key, capacity):
    """Independently derived G1-B2 cover for one level.

    Decision 0004 §3/§4: minimal covering logical block set for ``[lo, hi)``,
    resolved to current SlotIds, ascending.  Derived here from the contract,
    not from the implementation.
    """
    sr = engine.pgm_index(level_id).search(RecordKey(key))
    if sr.lo == sr.hi:
        return []
    first = sr.lo // capacity
    last = (sr.hi - 1) // capacity
    blocks = engine.level(level_id).block_ids[first : last + 1]
    return sorted(engine.mapping.lookup(b).value for b in blocks)


def oracle(engine, plan, key, capacity, contents):
    """Independent newest-first multi-level oracle.

    ``contents`` maps ``level_id -> (set_of_keys, tag)``.  Returns
    ``(found, hit_level_id, searched_level_ids, expected_slots, expected_value)``.
    """
    searched: list[LevelId] = []
    slots: list[int] = []
    for level_id in plan:
        keys, tag = contents[level_id]
        if not engine.level(level_id).block_ids:
            continue  # empty levels are skipped
        searched.append(level_id)
        slots += level_cover(engine, level_id, key, capacity)
        if key in keys:
            return True, level_id, tuple(searched), slots, f"{tag}:{key}"
    return False, None, tuple(searched), slots, None


def scramble_all(engine, level_ids, *, reverse=False, rng=None):
    """Relocate every block of the given levels to fresh slots."""
    block_ids = [b for lid in level_ids for b in engine.level(lid).block_ids]
    dest = [engine.storage.allocate() for _ in block_ids]
    if reverse:
        dest.reverse()
    elif rng is not None:
        rng.shuffle(dest)
    for block_id, slot in zip(block_ids, dest):
        engine.relocate_block(block_id, slot)


def physical_ids_reachable(obj, depth=0):
    """Every SlotId / BlockId reachable from a functional result object."""
    if depth > 6:
        return []
    if isinstance(obj, (SlotId, BlockId)):
        return [obj]
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return []
    found = []
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            found += physical_ids_reachable(item, depth + 1)
        return found
    if isinstance(obj, dict):
        for item in obj.values():
            found += physical_ids_reachable(item, depth + 1)
        return found
    if hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            found += physical_ids_reachable(value, depth + 1)
    return found


# ---------------------------------------------------------------------------
# [1] empty input plan
# ---------------------------------------------------------------------------


def test_empty_plan_is_miss_with_zero_reads():
    eng = build_engine()
    result, slots, events = observe(eng, [], 5)
    assert result.found is False
    assert result.record is None
    assert result.hit_level_id is None
    assert result.searched_level_ids == ()
    assert slots == []
    assert events == ()


# ---------------------------------------------------------------------------
# [2] plan containing only empty levels
# ---------------------------------------------------------------------------


def test_only_empty_levels_miss_with_zero_reads_and_no_pgm_required():
    eng = build_engine()
    empty_a = eng.create_level()
    empty_b = eng.create_level()
    assert eng.level(empty_a).block_ids == ()

    result, slots, events = observe(eng, [empty_a, empty_b], 7)

    assert result.found is False and result.record is None
    assert result.hit_level_id is None
    assert result.searched_level_ids == ()
    assert slots == [] and events == ()
    # no PGM was required or built as a side effect
    for level_id in (empty_a, empty_b):
        with pytest.raises(KeyError):
            eng.pgm_index(level_id)


def test_empty_level_skip_does_not_require_pgm_next_to_real_levels():
    eng = build_engine(capacity=4, epsilon=2)
    empty = eng.create_level()
    real = add_level(eng, range(8), tag="L")

    result, slots, events = observe(eng, [empty, real], 3)

    assert result.found is True
    assert result.searched_level_ids == (real,)  # empty level absent
    assert slots == level_cover(eng, real, 3, 4)
    with pytest.raises(KeyError):
        eng.pgm_index(empty)  # still no PGM for the skipped empty level
    assert all(ev.operation == TraceOperation.READ for ev in events)


# ---------------------------------------------------------------------------
# [3] HIT in the first level leaves later levels untouched
# ---------------------------------------------------------------------------


def test_first_level_hit_leaves_later_levels_untouched():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")

    result, slots, events = observe(eng, [l0, l1], 3)

    assert result.found is True
    assert result.hit_level_id == l0
    assert result.searched_level_ids == (l0,)
    assert slots == level_cover(eng, l0, 3, 4)
    l1_slots = {eng.mapping.lookup(b).value for b in eng.level(l1).block_ids}
    assert not (set(slots) & l1_slots), "later level received query traffic"


# ---------------------------------------------------------------------------
# [4] MISS on the first level, HIT on the second -> exact concatenation
# ---------------------------------------------------------------------------


def test_second_level_hit_produces_exact_two_level_trace():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")

    result, slots, events = observe(eng, [l0, l1], 103)

    assert result.found is True
    assert result.hit_level_id == l1
    assert result.searched_level_ids == (l0, l1)
    assert result.record is not None and result.record.value == "L1:103"
    expected = level_cover(eng, l0, 103, 4) + level_cover(eng, l1, 103, 4)
    assert slots == expected
    assert all(ev.operation == TraceOperation.READ for ev in events)
    assert len(slots) == len(set(slots))  # disjoint levels: no repeated slot


# ---------------------------------------------------------------------------
# [5] all levels MISS
# ---------------------------------------------------------------------------


def test_all_levels_miss_searches_every_non_empty_level():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")
    l2 = add_level(eng, range(200, 208), tag="L2")

    result, slots, events = observe(eng, [l0, l1, l2], 999)

    assert result.found is False and result.record is None
    assert result.hit_level_id is None
    assert result.searched_level_ids == (l0, l1, l2)
    expected = (
        level_cover(eng, l0, 999, 4)
        + level_cover(eng, l1, 999, 4)
        + level_cover(eng, l2, 999, 4)
    )
    assert slots == expected
    assert all(ev.operation == TraceOperation.READ for ev in events)


# ---------------------------------------------------------------------------
# [6] empty levels interleaved between non-empty levels
# ---------------------------------------------------------------------------


def test_interleaved_empty_levels_are_skipped_without_reordering():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    empty_a = eng.create_level()
    l1 = add_level(eng, range(100, 108), tag="L1")
    empty_b = eng.create_level()
    l2 = add_level(eng, range(200, 208), tag="L2")

    result, slots, _ = observe(eng, [l0, empty_a, l1, empty_b, l2], 103)

    assert result.found is True
    assert result.hit_level_id == l1
    assert result.searched_level_ids == (l0, l1)  # order preserved, empties gone
    assert slots == level_cover(eng, l0, 103, 4) + level_cover(eng, l1, 103, 4)


# ---------------------------------------------------------------------------
# [7] caller-supplied order is authoritative
# ---------------------------------------------------------------------------


def test_caller_order_is_authoritative_not_level_id_order():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, list(range(8)), tag="L0")          # owns key 3
    l1 = add_level(eng, [3, 50, 51], tag="L1")             # owns key 3
    l2 = add_level(eng, [3, 70, 71], tag="L2")             # owns key 3
    assert (l0.value, l1.value, l2.value) == (0, 1, 2)

    # descending LevelId numeric order wins if and only if it is supplied first
    result, _, _ = observe(eng, [l2, l1, l0], 3)
    assert result.hit_level_id == l2
    assert result.record is not None and result.record.value == "L2:3"
    assert result.searched_level_ids == (l2,)

    result, _, _ = observe(eng, [l0, l2, l1], 3)
    assert result.hit_level_id == l0
    assert result.record is not None and result.record.value == "L0:3"
    assert result.searched_level_ids == (l0,)

    result, _, _ = observe(eng, [l1, l0, l2], 3)
    assert result.hit_level_id == l1
    assert result.searched_level_ids == (l1,)


# ---------------------------------------------------------------------------
# [8] duplicate key in several levels -> first level in supplied order wins
# ---------------------------------------------------------------------------


def test_duplicate_key_across_levels_first_in_order_wins():
    eng = build_engine(capacity=4, epsilon=2)
    old = add_level(eng, [10, 20, 30], tag="OLD")
    new = add_level(eng, [10, 20, 30], tag="NEW")
    contents = {old: ({10, 20, 30}, "OLD"), new: ({10, 20, 30}, "NEW")}

    for plan, expected_hit, expected_value in [
        ([new, old], new, "NEW:20"),
        ([old, new], old, "OLD:20"),
    ]:
        found, hit, searched, slots, value = oracle(eng, plan, 20, 4, contents)
        result, observed_slots, _ = observe(eng, plan, 20)
        assert (result.found, result.hit_level_id) == (found, hit) == (True, expected_hit)
        assert result.searched_level_ids == searched
        assert result.record is not None and result.record.value == value == expected_value
        assert observed_slots == slots


# ---------------------------------------------------------------------------
# [9] no binary / small-level fallback
# ---------------------------------------------------------------------------


def test_small_level_requires_and_uses_pgm_no_binary_fallback():
    eng = build_engine(capacity=8, epsilon=2)
    small = add_level(eng, [42], tag="S", build_pgm=False)

    # a one-record level is not answered by any binary-search shortcut:
    # it must have a built PGM, and without one the plan fails before any READ
    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.lookup_levels([small], RecordKey(42))
    assert len(eng.trace) == 0
    with pytest.raises(KeyError):
        eng.pgm_index(small)

    eng.build_level_pgm(small)
    result, slots, events = observe(eng, [small], 42)
    assert result.found is True and result.hit_level_id == small
    assert result.record is not None and result.record.value == "S:42"
    # the G1-B2 path is used: exactly its single-block cover, one READ
    assert slots == level_cover(eng, small, 42, 8) == [eng.mapping.lookup(
        eng.level(small).block_ids[0]).value]
    assert len(events) == 1 and events[0].operation == TraceOperation.READ


# ---------------------------------------------------------------------------
# [10] a non-empty level without a valid PGM fails before ANY query READ
# ---------------------------------------------------------------------------


def test_later_missing_pgm_fails_before_any_query_read_even_on_would_be_hit():
    eng = build_engine(capacity=4, epsilon=2)
    good = add_level(eng, range(8), tag="G")          # owns key 3 -> would HIT
    bad = add_level(eng, range(100, 108), tag="B", build_pgm=False)

    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.lookup_levels([good, bad], RecordKey(3))
    assert len(eng.trace) == 0, "preflight must precede every physical query read"


def test_later_invalidated_pgm_fails_before_any_query_read():
    eng = build_engine(capacity=4, epsilon=2)
    good = add_level(eng, range(8), tag="G")
    bad = add_level(eng, range(100, 104), tag="B")
    eng.create_block([rec(999, "B")], level_id=bad)   # mutates -> PGM invalidated
    with pytest.raises(KeyError):
        eng.pgm_index(bad)

    eng.trace.clear()
    with pytest.raises(KeyError):
        eng.lookup_levels([good, bad], RecordKey(3))
    assert len(eng.trace) == 0


# ---------------------------------------------------------------------------
# [11] duplicate LevelId in a plan
# ---------------------------------------------------------------------------


def test_duplicate_level_in_plan_fails_before_any_query_read():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")
    unknown_a = LevelId(5150)
    unknown_b = LevelId(5151)

    plans = [
        [l0, l0],
        [l0, l1, l0],
        [l1, l1, l1],
        # Decision 0005 §5 phase A completes over the WHOLE plan before phase B
        # resolves anything, so duplicate validation outranks unknown-level
        # detection even when unknown ids appear first.
        [unknown_a, unknown_a],
        [unknown_a, l0, l0],
        [l0, unknown_b, unknown_b],
    ]
    for plan in plans:
        eng.trace.clear()
        with pytest.raises(DuplicateLevelInPlanError):
            eng.lookup_levels(plan, RecordKey(3))
        assert len(eng.trace) == 0
    assert issubclass(DuplicateLevelInPlanError, ValueError)


# ---------------------------------------------------------------------------
# [12] unknown LevelId
# ---------------------------------------------------------------------------


def test_unknown_level_fails_before_any_query_read():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    stranger = LevelId(4242)
    other_stranger = LevelId(4243)

    # no duplicates in any of these plans -> unknown-level detection is the
    # failure (duplicate detection only outranks it when a plan repeats an id)
    for plan in ([stranger], [l0, stranger], [stranger, l0],
                 [stranger, l0, other_stranger]):
        eng.trace.clear()
        with pytest.raises(UnknownLevelError):
            eng.lookup_levels(plan, RecordKey(3))
        assert len(eng.trace) == 0
    assert issubclass(UnknownLevelError, KeyError)


def test_all_miss_with_interleaved_empty_levels_skips_empties_in_order():
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    empty = eng.create_level()
    l1 = add_level(eng, range(100, 108), tag="L1")

    result, slots, events = observe(eng, [l0, empty, l1], 900)

    assert result.found is False and result.hit_level_id is None
    assert result.searched_level_ids == (l0, l1)
    assert slots == level_cover(eng, l0, 900, 4) + level_cover(eng, l1, 900, 4)
    assert all(ev.operation == TraceOperation.READ for ev in events)


# ---------------------------------------------------------------------------
# [13] per-level schedule == independently derived G1-B2 cover
# ---------------------------------------------------------------------------


def test_per_level_schedule_equals_independent_g1b2_cover():
    eng = build_engine(capacity=2, epsilon=3)
    l0 = add_level(eng, range(12), tag="L0")
    l1 = add_level(eng, range(100, 112), tag="L1")
    key = 103

    result, slots, events = observe(eng, [l0, l1], key)
    cover0 = level_cover(eng, l0, key, 2)
    cover1 = level_cover(eng, l1, key, 2)

    assert result.searched_level_ids == (l0, l1)
    assert slots == cover0 + cover1
    # each level's segment is ascending and exactly its minimal cover
    assert slots[: len(cover0)] == sorted(cover0)
    assert slots[len(cover0) :] == sorted(cover1)
    for segment in (cover0, cover1):
        assert segment == sorted(segment)
        assert len(segment) == len(set(segment))
    assert all(ev.operation == TraceOperation.READ for ev in events)


# ---------------------------------------------------------------------------
# [14] no fixed-3 behaviour: 0, 1, 2 and >3 candidate blocks are respected
# ---------------------------------------------------------------------------


def test_no_fixed_three_block_last_mile_padding_or_truncation():
    specs = [
        # (capacity, epsilon, keys, query, expected candidate block count)
        (4, 0, list(range(8)), 1000, 0),    # lo == hi: zero candidates
        (8, 2, list(range(6)), 3, 1),       # one block holds the level
        (2, 1, list(range(8)), 1, 2),       # exactly two blocks
        (2, 6, list(range(20)), 10, 7),     # far more than three blocks
    ]
    observed_counts = []
    for capacity, eps, keys, query, expected_count in specs:
        eng = build_engine(capacity=capacity, epsilon=eps)
        level_id = add_level(eng, keys, tag="L")
        expected_slots = level_cover(eng, level_id, query, capacity)

        result, slots, events = observe(eng, [level_id], query)

        assert set(result.searched_level_ids) == {level_id}
        assert slots == expected_slots
        assert len(slots) == expected_count, (capacity, eps, query, slots)
        assert all(ev.operation == TraceOperation.READ for ev in events)
        observed_counts.append(len(slots))

    assert observed_counts == [0, 1, 2, 7]
    # explicit anti-"fixed three" statement: at least one fixture is above and
    # at least one below the old 3-block simplification
    assert max(observed_counts) > 3 and min(observed_counts) < 3


# ---------------------------------------------------------------------------
# [15] hit-and-stop is cross-level only
# ---------------------------------------------------------------------------


def test_hit_and_stop_is_cross_level_only():
    eng = build_engine(capacity=2, epsilon=4)
    l0 = add_level(eng, range(12), tag="L0")     # key 0 sits in the FIRST candidate
    l1 = add_level(eng, range(50, 56), tag="L1")

    cover0 = level_cover(eng, l0, 0, 2)
    assert len(cover0) >= 3, "fixture must span several candidate blocks"

    result, slots, events = observe(eng, [l0, l1], 0)

    assert result.found is True and result.hit_level_id == l0
    assert result.searched_level_ids == (l0,)     # later level not searched
    assert slots == cover0, "must read every G1-B2 candidate before stopping"
    assert len(slots) >= 3
    assert all(ev.operation == TraceOperation.READ for ev in events)
    l1_slots = {eng.mapping.lookup(b).value for b in eng.level(l1).block_ids}
    assert not (set(slots) & l1_slots)


def test_hit_and_stop_still_reads_full_cover_when_hit_is_last_candidate():
    eng = build_engine(capacity=2, epsilon=4)
    l0 = add_level(eng, range(12), tag="L0")
    l1 = add_level(eng, range(50, 56), tag="L1")

    cover0 = level_cover(eng, l0, 5, 2)
    result, slots, _ = observe(eng, [l0, l1], 5)
    assert result.found is True and result.hit_level_id == l0
    assert result.searched_level_ids == (l0,)
    assert slots == cover0


# ---------------------------------------------------------------------------
# [16] scrambled physical mappings
# ---------------------------------------------------------------------------


def test_scrambled_mapping_preserves_result_and_per_level_order():
    eng = build_engine(capacity=4, epsilon=3)
    l0 = add_level(eng, range(16), tag="L0")
    l1 = add_level(eng, range(100, 116), tag="L1")
    queries = [0, 3, 7, 15, 16, 103, 115, 999, 50]
    contents = {l0: (set(range(16)), "L0"), l1: (set(range(100, 116)), "L1")}

    before = {q: observe(eng, [l0, l1], q)[0] for q in queries}
    scramble_all(eng, [l0, l1], reverse=True)

    for q in queries:
        found, hit, searched, expected_slots, expected_value = oracle(
            eng, [l0, l1], q, 4, contents
        )
        result, slots, events = observe(eng, [l0, l1], q)
        prior = before[q]

        # functional semantics unchanged by relocation
        assert result.found == prior.found == found
        assert result.hit_level_id == prior.hit_level_id == hit
        assert result.searched_level_ids == prior.searched_level_ids == searched
        assert result.record == prior.record
        if found:
            assert result.record is not None and result.record.value == expected_value
        else:
            assert result.record is None

        # physical schedule recomputed against the new mapping, per-level
        # ascending and composed level by level
        assert slots == expected_slots
        offset = 0
        for level_id in searched:
            segment = level_cover(eng, level_id, q, 4)
            assert slots[offset : offset + len(segment)] == sorted(segment)
            offset += len(segment)
        assert offset == len(slots)
        assert all(ev.operation == TraceOperation.READ for ev in events)
        assert len(slots) == len(set(slots))


def test_scrambled_mapping_trace_differs_but_semantics_hold():
    eng = build_engine(capacity=4, epsilon=3)
    l0 = add_level(eng, range(16), tag="L0")

    _, before_slots, _ = observe(eng, [l0], 7)
    scramble_all(eng, [l0], reverse=True)
    result, after_slots, _ = observe(eng, [l0], 7)

    assert result.found is True and result.record is not None
    assert result.record.value == "L0:7"
    assert after_slots != before_slots           # physical schedule changed
    assert after_slots == level_cover(eng, l0, 7, 4)
    assert after_slots == sorted(after_slots)


# ---------------------------------------------------------------------------
# [17] randomized multi-level oracle
# ---------------------------------------------------------------------------


def randomized_multilevel_oracle(seed):
    rng = random.Random(seed)
    for _ in range(12):
        capacity = rng.choice([1, 2, 3, 4, 8])
        eps = rng.choice([0, 1, 2, 3, 5])
        eng = build_engine(capacity=capacity, epsilon=eps)

        plan: list[LevelId] = []
        contents: dict[LevelId, tuple[set[int], str]] = {}
        for index in range(rng.randint(1, 5)):
            tag = f"L{index}"
            style = rng.choice(["dense", "sparse", "empty", "wide"])
            if style == "empty":
                keys: list[int] = []
                level_id = eng.create_level()
            else:
                if style == "dense":
                    keys = list(range(rng.randint(1, 18)))
                elif style == "sparse":
                    keys = sorted(rng.sample(range(0, 200), rng.randint(1, 12)))
                else:
                    keys = sorted(rng.sample(range(0, 10**6), rng.randint(1, 10)))
                level_id = add_level(eng, keys, tag=tag)
            contents[level_id] = (set(keys), tag)
            plan.append(level_id)

        rng.shuffle(plan)  # caller order is independent of creation order

        all_keys = sorted(set().union(*[c[0] for c in contents.values()]) or {0})
        queries = list(all_keys)
        queries += [k - 1 for k in all_keys] + [k + 1 for k in all_keys]
        queries += [min(all_keys) - 3, max(all_keys) + 3]
        queries += [rng.randint(min(all_keys) - 10, max(all_keys) + 10) for _ in range(8)]

        for q in queries:
            found, hit, searched, expected_slots, expected_value = oracle(
                eng, plan, q, capacity, contents
            )
            result, slots, events = observe(eng, plan, q)

            assert result.found == found, (seed, q, contents)
            assert result.hit_level_id == hit, (seed, q)
            assert result.searched_level_ids == searched, (seed, q)
            assert slots == expected_slots, (seed, q, slots, expected_slots)
            assert all(ev.operation == TraceOperation.READ for ev in events)
            assert len(slots) == len(set(slots))

            if found:
                assert result.record is not None
                assert result.record.value == expected_value
                assert result.hit_level_id == result.searched_level_ids[-1]
            else:
                assert result.record is None
                assert result.searched_level_ids == tuple(
                    lid for lid in plan if eng.level(lid).block_ids
                )


def test_randomized_multilevel_oracle_many_seeds():
    for seed in range(5):
        randomized_multilevel_oracle(7000 + seed)


# ---------------------------------------------------------------------------
# [18] result object exposes no physical internals
# ---------------------------------------------------------------------------


def test_multilevel_result_exposes_no_physical_internals():
    eng = build_engine(capacity=2, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")

    hit_result, _, _ = observe(eng, [l0, l1], 103)
    miss_result, _, _ = observe(eng, [l0, l1], 900)

    for result in (hit_result, miss_result):
        assert isinstance(result, MultiLevelLookupResult)
        assert set(vars(result)) == {
            "found",
            "record",
            "hit_level_id",
            "searched_level_ids",
        }
        assert physical_ids_reachable(result) == []
        for forbidden in ("slot", "slot_id", "physical_schedule", "block_ids", "trace"):
            assert not hasattr(result, forbidden)

    assert hit_result.searched_level_ids[-1] == hit_result.hit_level_id
    assert miss_result.hit_level_id is None


# ---------------------------------------------------------------------------
# extra contract checks
# ---------------------------------------------------------------------------


def test_multilevel_lookup_is_deterministic_and_write_free():
    eng = build_engine(capacity=3, epsilon=2)
    l0 = add_level(eng, range(9), tag="L0")
    l1 = add_level(eng, range(100, 109), tag="L1")
    plan = [l0, l1]

    first = observe(eng, plan, 4)
    second = observe(eng, plan, 4)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert all(ev.operation == TraceOperation.READ for ev in first[2] + second[2])


def test_g1b2_single_level_result_is_reused_not_reimplemented(monkeypatch):
    """The multi-level path must delegate to the accepted G1-B2 lookup."""
    eng = build_engine(capacity=4, epsilon=2)
    l0 = add_level(eng, range(8), tag="L0")
    l1 = add_level(eng, range(100, 108), tag="L1")

    calls = []
    original = TrustedEngine.lookup_level

    def spy(self, level_id, key):
        result = original(self, level_id, key)
        calls.append((level_id, key, result))
        return result

    monkeypatch.setattr(TrustedEngine, "lookup_level", spy)

    multi, _, _ = observe(eng, [l0, l1], 103)

    # one delegation per searched level, in caller order
    assert [level_id for level_id, _key, _result in calls] == [l0, l1]
    assert all(key == RecordKey(103) for _level, key, _result in calls)
    # the miss on the first level and the hit on the second come from G1-B2
    assert calls[0][2].found is False
    assert calls[1][2].found is True
    assert multi.record == calls[1][2].record
    assert multi.hit_level_id == l1
    assert multi.searched_level_ids == (l0, l1)
