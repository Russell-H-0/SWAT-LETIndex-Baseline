"""G2-B0 LETIndex-Ref-v1 reference-profile tests (Issue #11).

Covers the upstream-faithful geometry, the hand-derivable search paths
(1/2/3-block conditional prefetch, cached and uncached later probes, non-PGM
full-level search), the multi-level contract, the physical SlotId projection, the
``ordered`` / ``static_prp`` layouts, the adversary/evaluator separation, the
passive workload generators and reproducibility.

Every expectation below is derived by hand from the pinned upstream algorithm
(``chiiips/LETIndex`` @ ``9b28fce``) and written out with its derivation in the
comment, so the tests document the semantics they check.

Hand-derivation conventions used in the comments:

* ``C`` = items per block; ``item r`` holds key ``1000 + r`` for the small
  synthetic levels, so ``item r`` is hit by the query key ``1000 + r``;
* the PGM-bounded search is
  ``n = last - first; mid = first + n//2`` (first probe, with the conditional
  adjacent prefetch), then ``while n > 1: half = n//2; probe first + half``,
  then ``pos = first + (item[first] < q)`` and one final read of ``item[pos]``;
* the non-PGM search is the same binary search over ``[0, item_count)`` with
  single-block reads and no prefetch cache.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from enhanced_letindex import letindex_ref, ref_adversary, ref_truth, ref_workload
from enhanced_letindex.identifiers import SlotId
from enhanced_letindex.leakage import (
    SUMMARY_KEYS,
    AdversaryTranscript,
    LeakageError,
    validate_transcript,
)
from enhanced_letindex.leakage_metrics import leakage_metrics, per_operation_metrics
from enhanced_letindex.letindex_ref import (
    BASE,
    BLOCK_SIZE,
    CLASS_BUILD_INITIAL,
    CLASS_BUILD_PGM,
    CLASS_QUERY,
    ITEM_SIZE,
    ITEMS_PER_BLOCK,
    LAYOUT_ORDERED,
    LAYOUT_STATIC_PRP,
    LEGACY_SIMULATOR_SETTINGS,
    MIN_DISK_LEVEL,
    MIN_INDEX_LEVEL,
    MIN_LEVEL,
    PROFILE_NAME,
    READ,
    REFERENCE_EPSILON,
    REFERENCE_ARTIFACT_SCHEMA_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    WRITE,
    ReferenceExperiment,
    ReferenceIndex,
    ReferenceProfile,
    ReferenceProfileError,
    ReferenceTraceBuilder,
    SeededPermutation,
    SlotProjection,
    default_scenario,
    dumps_reference_artifact,
    validate_reference_transcript,
)
from enhanced_letindex.ref_adversary import (
    ALLOWED_FIELD_UNIVERSE,
    PUBLIC_CONFIG_KEYS,
    AdversaryInput,
    AdversaryInputError,
    scan_forbidden_fields,
)
from enhanced_letindex.ref_truth import ReferenceEvaluatorTruth
from enhanced_letindex.ref_workload import (
    UPSTREAM_WORKLOAD_NOTE,
    WORKLOAD_DATA_EMPIRICAL,
    WORKLOAD_DOMAIN_UNIFORM,
    WORKLOAD_MODES,
    WORKLOAD_RECORD_UNIFORM,
    WorkloadError,
    generate_passive_workload,
)

# small synthetic levels for hand-derivable traces
BLOCK_SWEEP_LABEL = "block-size-sweep"
LEVEL4 = 4

#: ``codes/`` — the CLI tests run ``python -m enhanced_letindex...`` from here.
CODES_DIR = Path(__file__).resolve().parents[1]
CLI_ENV = {**os.environ, "PYTHONPATH": str(CODES_DIR / "src")}


def variant(items_per_block, *, epsilon=8, label=BLOCK_SWEEP_LABEL, layout=LAYOUT_ORDERED,
            seed=0):
    return ReferenceProfile.experimental_variant(
        label, layout=layout, seed=seed, items_per_block=items_per_block, epsilon=epsilon
    )


def small_level(count, *, start=1000, level=LEVEL4, items_per_block=16, epsilon=8,
                label=BLOCK_SWEEP_LABEL, layout=LAYOUT_ORDERED, seed=0):
    """A single-level index whose items are ``start + r`` for ``r in range(count)``."""
    profile = variant(items_per_block, epsilon=epsilon, label=label, layout=layout, seed=seed)
    items = [(start + rank, f"v{rank}") for rank in range(count)]
    return ReferenceIndex.build(profile, {level: items})


def slots(outcome, index, level=LEVEL4):
    """Slot **values** of one level lookup, in event order."""
    projection = index.projection
    return [event_slot for event_slot in slots_of(outcome)]


def slots_of(outcome):
    return [slot for _, slot in outcome.events]


def blocks_of(outcome):
    """Physical block offsets of the events, recovered from the level's projection."""
    return list(outcome.fetched_blocks)


# ---------------------------------------------------------------------------
# 1. upstream geometry (frozen constants)
# ---------------------------------------------------------------------------


def test_reference_constants_equal_pinned_upstream_defaults():
    assert UPSTREAM_REPO == "https://github.com/chiiips/LETIndex"
    assert UPSTREAM_COMMIT == "9b28fce03638a3cd97402d43c10bca644d0a95a5"
    assert (BASE, MIN_LEVEL, MIN_INDEX_LEVEL, MIN_DISK_LEVEL) == (8, 3, 6, 7)
    assert (BLOCK_SIZE, ITEM_SIZE) == (4096, 16)
    assert ITEMS_PER_BLOCK == 256
    assert REFERENCE_EPSILON == 64
    profile = ReferenceProfile.reference()
    assert profile.is_reference_defaults()
    assert profile.variant_label is None


def test_items_per_block_is_block_size_over_item_size():
    assert BLOCK_SIZE // ITEM_SIZE == ITEMS_PER_BLOCK == 256
    assert ReferenceProfile.reference().block_count(4096) == 16
    assert ReferenceProfile.reference().block_count(1) == 1
    assert ReferenceProfile.reference().block_count(0) == 0


def test_level_capacity_follows_base_8():
    profile = ReferenceProfile.reference()
    assert [profile.max_size(level) for level in range(3, 8)] == [
        512, 4096, 32768, 262144, 2097152,
    ]
    assert profile.max_size(3) == BASE ** 3


def test_capacity_and_level_floor_are_enforced():
    profile = ReferenceProfile.reference()
    with pytest.raises(ReferenceProfileError):
        ReferenceIndex.build(profile, {2: [(1, "a")]})
    with pytest.raises(ReferenceProfileError):
        ReferenceIndex.build(profile, {3: [(1, "a"), (1, "b")]})
    with pytest.raises(ReferenceProfileError):
        ReferenceIndex.build(profile, {3: [(k, "a") for k in range(513)]})


def test_variant_must_be_labelled_and_legacy_settings_are_named():
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile(items_per_block=16)
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile(epsilon=8)
    labelled = ReferenceProfile.experimental_variant("sweep", items_per_block=16)
    assert labelled.variant_label == "sweep"
    assert not labelled.is_reference_defaults()
    assert set(LEGACY_SIMULATOR_SETTINGS) == {
        "ratio=4", "[4,16,64,...]", "fixed 3-block last mile", "hit±1",
    }


def test_backing_store_marks_levels_below_min_disk_level():
    profile = ReferenceProfile.reference()
    assert [profile.level_backing(level) for level in (3, 6, 7, 8)] == [
        "memory", "memory", "disk", "disk",
    ]
    assert profile.has_index(MIN_INDEX_LEVEL - 1) is False
    assert profile.has_index(MIN_INDEX_LEVEL) is True


# ---------------------------------------------------------------------------
# 2. hand-derivable search paths (variant C=16 so 1/2/3-block fetches all occur)
# ---------------------------------------------------------------------------


def test_interval_contained_in_one_block_first_fetch_is_one_block():
    """[0,10), q=1025: mid=item5 (block 0); first/last blocks both 0 -> no prefetch."""
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 0, 10)
    assert blocks_of(outcome) == [0]
    assert outcome.probes == (5, 7, 8, 9, 9)
    assert outcome.events == ((READ, index.projection.slot(LEVEL4, 0).value),)
    assert outcome.pos == 10 and outcome.found is False   # 1010 != 1025


def test_interval_crosses_left_boundary_first_fetch_is_two_blocks():
    """[14,30), q=1025: mid=item22 (block 1); first block 0 -> left prefetch only.

    n=16 -> half=8 -> mid=22; block(14)=0 < block(22)=1 -> save_left;
    block(30)=1 == block(22) -> no save_right; fetch blocks {0,1}.
    Then item[22]=1022<1025 -> first=22, n=8; probes 26,24,25; final pos=25.
    """
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 14, 30)
    assert blocks_of(outcome) == [0, 1]
    assert outcome.probes == (22, 26, 24, 25, 24)
    assert outcome.pos == 25 and outcome.found is True


def test_interval_crosses_right_boundary_first_fetch_is_two_blocks():
    """[2,20), q=1025: mid=item11 (block 0); last block 1 -> right prefetch only.

    n=18 -> half=9 -> mid=11; block(2)=0 == block(11)=0 -> no save_left;
    block(20)=1 > 0 -> save_right; fetch blocks {0,1}.  The later probe at
    item 17 lands in the prefetched block 1, so it adds **no** read.
    """
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 2, 20)
    assert blocks_of(outcome) == [0, 1]
    assert outcome.probes == (11, 15, 17, 18, 19, 19)
    assert outcome.pos == 20 and outcome.found is False


def test_interval_crosses_both_boundaries_first_fetch_is_three_blocks():
    """[5,40), q=1025: mid=item22 (block 1); first block 0, last block 2 -> both.

    n=35 -> half=17 -> mid=22; block(5)=0 < 1 -> save_left;
    block(40)=2 > 1 -> save_right; fetch blocks {0,1,2} in ascending order.
    """
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 5, 40)
    assert blocks_of(outcome) == [0, 1, 2]
    assert outcome.probes == (22, 31, 26, 24, 25, 25, 24)
    assert outcome.pos == 25 and outcome.found is True
    assert outcome.read_count == 3


def test_cached_neighbour_probe_adds_no_physical_read():
    """Same two-block right prefetch, but the trace stays at two reads."""
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 2, 20)
    in_prefetched_block = [
        probe for probe in outcome.probes if probe // 16 == 1
    ]
    assert in_prefetched_block, "test must exercise a probe in the prefetched block"
    assert outcome.read_count == 2, "a probe served by the neighbour cache must not read"


def test_uncached_later_probe_adds_a_physical_read():
    """[0,64), q=1005: mid=item32 (block 2) caches blocks 1 and 3 (fetch 1,2,3).

    n=64 -> half=32 -> mid=32 -> block 2; block(0)=0 < 2 -> save_left -> cache 1;
    block(64)=4 > 2 -> save_right -> cache 3.  item[32]=1032 > 1005 -> first stays 0,
    n=32; probe 16 -> block 1 (cached, no read); probe 8 -> block 0, **not** cached
    -> physical read; then probes 4, 6, 5 stay in block 0 (the new current block).
    Event order: prefetch 1,2,3 then the uncached block 0.
    """
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1005, 0, 64)
    assert blocks_of(outcome) == [1, 2, 3, 0]
    assert outcome.probes == (32, 16, 8, 4, 6, 5, 4)
    assert outcome.pos == 5 and outcome.found is True


def test_exact_hit_and_miss_at_the_interval_edges():
    index = small_level(64)
    hit = index.lookup_level_with_interval(LEVEL4, 1025, 17, 30)
    assert (hit.pos, hit.found, hit.value, hit.read_count) == (25, True, "v25", 1)

    miss_low = index.lookup_level_with_interval(LEVEL4, 999, 0, 10)
    assert (miss_low.pos, miss_low.found, miss_low.value) == (0, False, None)

    miss_high = index.lookup_level_with_interval(LEVEL4, 1064, 60, 64)
    assert (miss_high.pos, miss_high.found, miss_high.value) == (64, False, None)

    empty = index.lookup_level_with_interval(LEVEL4, 1000, 7, 7)
    assert (empty.pos, empty.found, empty.events) == (7, False, ())


def test_single_item_interval_reads_item_and_its_successor():
    """n == 1 skips the binary-search loop: read item[first], then item[first+1].

    ``[25,26)`` with q=1025: probes to item 25 only; item[25]=1025 is not < 1025,
    so pos = 25 and the final read is item 25 (one block, one read).  With q=1026
    the same probe yields pos = 26 and the final read stays inside the same block.
    """
    index = small_level(64)
    outcome = index.lookup_level_with_interval(LEVEL4, 1025, 25, 26)
    assert outcome.probes == (25,)
    assert (outcome.pos, outcome.found, outcome.value) == (25, True, "v25")
    assert blocks_of(outcome) == [1] and outcome.read_count == 1

    successor = index.lookup_level_with_interval(LEVEL4, 1026, 25, 26)
    assert (successor.pos, successor.found, successor.value) == (26, True, "v26")
    assert blocks_of(successor) == [1] and successor.read_count == 1


def test_no_pgm_level_uses_full_level_binary_search_without_prefetch():
    """Level 3 has no PGM (MIN_INDEX_LEVEL=6): full-level search, single-block reads.

    C=4, 16 items (blocks 0..3), q=1000:
    n=16, probe 8 -> block 2 (read); item[8]=1008>1000 -> first 0, n=8;
    probe 4 -> block 1 (read, current was 2); item[4]>1000 -> first 0, n=4;
    probe 2 -> block 0 (read); item[2]>1000 -> first 0, n=2;
    probe 1 -> block 0 (current, no read); item[1]>1000 -> first 0, n=1;
    pos = 0 + (item[0]=1000 < 1000 ? no) = 0; final read item[0] -> block 0 (current).
    """
    profile = variant(4, epsilon=8)
    index = ReferenceIndex.build(
        profile, {3: [(1000 + rank, f"f{rank}") for rank in range(16)]}
    )
    level = index.level(3)
    assert level.has_pgm is False
    outcome = index.lookup_level(3, 1000)
    assert outcome.path == "full_level"
    assert outcome.interval == (0, 16)
    assert outcome.probes == (8, 4, 2, 1, 0)
    assert blocks_of(outcome) == [2, 1, 0]
    assert (outcome.pos, outcome.found, outcome.value) == (0, True, "f0")

    miss = index.lookup_level(3, 9999)
    assert miss.found is False and miss.pos == 16
    assert blocks_of(miss) == [2, 3]  # no prefetch cache: each new block is a read


def test_reference_defaults_prefetch_is_never_three_blocks():
    """Derived bound: with eps=64 and 256 items per block the **initial prefetch**
    is 1 or 2 blocks, never 3.

    A three-block first fetch needs ``block(first) < mid_block`` **and**
    ``block(last) > mid_block``, hence ``last - first > 256``; but the PGM window is
    at most ``2*64 + 2 = 130`` wide.  (A lookup can still touch a *third* block
    through the final ``item[pos]`` read, because ``pos`` may equal ``hi`` and
    ``item[hi]`` lies outside the window — that is a later read, not a prefetch.)
    The variant sweep (small blocks) shows the three-block prefetch is reachable once
    the bound is lifted.
    """
    profile = ReferenceProfile.reference()
    items = [(1000 + rank, f"r{rank}") for rank in range(1200)]
    index = ReferenceIndex.build(profile, {6: items})
    prefetch_sizes = set()
    extra_reads = 0
    for key in range(999, 2201):
        outcome = index.lookup_level(6, key)
        interval = outcome.interval
        assert interval[1] - interval[0] <= 2 * profile.epsilon + 2
        if outcome.prefetch_size is not None:
            prefetch_sizes.add(outcome.prefetch_size)
            # the two-block window cannot leave the prefetched trio, so the whole
            # lookup costs exactly as many reads as the initial prefetch
            extra_reads += outcome.read_count - outcome.prefetch_size
    assert prefetch_sizes == {1, 2}, "reference defaults prefetch only 1 or 2 blocks"
    assert extra_reads == 0, "no later probe can miss the current/cached buffers here"

    index_variant = small_level(64)
    variant_prefetch = set()
    for first in range(0, 64):
        outcome = index_variant.lookup_level_with_interval(LEVEL4, 1025, first, 64)
        if outcome.prefetch_size is not None:
            variant_prefetch.add(outcome.prefetch_size)
    assert 3 in variant_prefetch, "the labelled variant must exercise three blocks"


# ---------------------------------------------------------------------------
# 3. multi-level contract
# ---------------------------------------------------------------------------


def test_empty_levels_are_skipped_and_order_is_ascending():
    profile = variant(16, epsilon=8)
    levels = {
        3: [(1000 + rank, f"a{rank}") for rank in range(4)],
        5: [],                                                  # declared but empty
        6: [(2000 + rank, f"b{rank}") for rank in range(4)],
    }
    index = ReferenceIndex.build(profile, levels)
    assert index.scanned_levels() == (3, 4, 5, 6)
    outcome = index.query(2002)
    assert outcome.searched_levels == (3, 6)
    assert outcome.skipped_empty_levels == (4, 5)
    assert outcome.paths == ((3, "full_level"), (6, "pgm_bounded"))
    assert outcome.hit_level == 6 and outcome.value == "b2"


def test_hit_and_stop_between_levels():
    profile = variant(16, epsilon=8)
    levels = {
        3: [(1000 + rank, f"a{rank}") for rank in range(4)],
        6: [(1000 + rank, f"b{rank}") for rank in range(4)],   # same keys, newer level first
    }
    index = ReferenceIndex.build(profile, levels)
    outcome = index.query(1001)
    assert outcome.found is True
    assert outcome.hit_level == 3                        # first level wins
    assert outcome.searched_levels == (3,)               # later levels receive no read
    assert outcome.paths == ((3, "full_level"),)
    assert all(
        event[1] < index.projection.region(6).region_start for event in outcome.events
    ), "no level-6 slot may appear once level 3 hit"


# ---------------------------------------------------------------------------
# 4. physical projection
# ---------------------------------------------------------------------------


def test_projection_is_injective_with_disjoint_level_regions():
    projection = SlotProjection.from_block_counts({3: 2, 4: 3, 6: 4})
    assert [projection.region(level).region_start for level in (3, 4, 6)] == [0, 2, 5]
    assert projection.total_blocks == 9
    seen = {}
    for level in (3, 4, 6):
        region = projection.region(level)
        for offset in range(region.block_count):
            slot = projection.slot(level, offset)
            assert slot.value not in seen, "levels must occupy disjoint SlotId regions"
            seen[slot.value] = (level, offset)
            assert projection.locate(slot) == (level, offset)
    assert len(seen) == 9
    with pytest.raises(ReferenceProfileError):
        projection.slot(4, 3)
    with pytest.raises(ReferenceProfileError):
        projection.slot(5, 0)
    with pytest.raises(ReferenceProfileError):
        projection.locate(SlotId(99))


def test_ordered_projection_is_monotone_within_a_level():
    profile = ReferenceProfile.reference()
    index = ReferenceIndex.build(
        profile, {4: [(k, "v") for k in range(700)], 6: [(k, "v") for k in range(900)]}
    )
    for level in (4, 6):
        reference = index.level(level)
        offsets = [index.projection.slot(level, rank).value for rank in range(reference.block_count)]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == reference.block_count


# ---------------------------------------------------------------------------
# 5. layouts: ordered vs static_prp
# ---------------------------------------------------------------------------


def test_static_prp_permutation_is_a_seeded_bijection():
    permutation = SeededPermutation(8, seed=11)
    assert sorted(permutation.as_tuple()) == list(range(8))
    assert [permutation.inverse(permutation.forward(i)) for i in range(8)] == list(range(8))
    assert SeededPermutation(8, seed=11).as_tuple() == permutation.as_tuple()
    assert SeededPermutation(8, seed=12).as_tuple() != permutation.as_tuple()
    with pytest.raises(ReferenceProfileError):
        permutation.forward(8)


def test_static_prp_changes_only_placement_not_logical_decisions():
    levels = {
        3: [(1000 + rank, f"a{rank}") for rank in range(512)],    # 2 blocks, no PGM
        6: [(2000 + rank, f"b{rank}") for rank in range(900)],    # 4 blocks, PGM
    }
    query_keys = (2000, 2013, 1999, 2400)
    ordered = ReferenceTraceBuilder(
        ReferenceExperiment(
            profile=ReferenceProfile.reference(layout=LAYOUT_ORDERED, seed=3),
            levels=tuple((level, tuple(items)) for level, items in levels.items()),
            query_keys=query_keys,
        )
    ).load().run_queries()
    prp = ReferenceTraceBuilder(
        ReferenceExperiment(
            profile=ReferenceProfile.reference(
                layout=LAYOUT_STATIC_PRP, seed=3, prp_key=91
            ),
            levels=tuple((level, tuple(items)) for level, items in levels.items()),
            query_keys=query_keys,
        )
    ).load().run_queries()

    for ordered_outcome, prp_outcome in zip(ordered.query_outcomes, prp.query_outcomes):
        # identical logical answers
        assert (ordered_outcome.found, ordered_outcome.value, ordered_outcome.hit_level) == (
            prp_outcome.found, prp_outcome.value, prp_outcome.hit_level
        )
        assert ordered_outcome.searched_levels == prp_outcome.searched_levels
        # identical logical search decisions: same interval, same probed item ranks
        assert [
            (o.level, o.interval, o.probes, o.pos) for o in ordered_outcome.level_outcomes
        ] == [
            (o.level, o.interval, o.probes, o.pos) for o in prp_outcome.level_outcomes
        ]

    # different physical traces: at least one distinct slot sequence
    ordered_slots = [tuple(o.slots()) for o in ordered.query_outcomes]
    prp_slots = [tuple(o.slots()) for o in prp.query_outcomes]
    assert ordered_slots != prp_slots
    # ... and the placements really are permuted, not shifted
    assert ordered.index.level(6).physical_blocks() == tuple(range(ordered.index.level(6).block_count))
    prp_placement = prp.index.level(6).physical_blocks()
    assert sorted(prp_placement) == list(range(len(prp_placement)))
    assert prp_placement != tuple(range(len(prp_placement)))


def test_static_prp_is_per_level_and_deterministic_across_runs():
    levels = {4: [(1000 + rank, "v") for rank in range(600)],
              6: [(2000 + rank, "v") for rank in range(600)]}
    run = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=77),
        levels=tuple((level, tuple(items)) for level, items in levels.items()),
        query_keys=(),
    )).load()
    again = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=77),
        levels=tuple((level, tuple(items)) for level, items in levels.items()),
        query_keys=(),
    )).load()
    assert run.index.level(4).physical_blocks() == again.index.level(4).physical_blocks()
    assert run.index.level(6).physical_blocks() == again.index.level(6).physical_blocks()
    assert run.index.level(4).physical_blocks() != run.index.level(6).physical_blocks()
    other_key = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=78),
        levels=tuple((level, tuple(items)) for level, items in levels.items()),
        query_keys=(),
    )).load()
    # the layout key drives the placement: over several keys the placements differ
    # (a 3-block level can coincide for two specific keys, so test the family)
    placements = {other_key.index.level(4).physical_blocks()}
    for key in range(79, 89):
        built = ReferenceTraceBuilder(ReferenceExperiment(
            profile=ReferenceProfile.reference(
                layout=LAYOUT_STATIC_PRP, seed=5, prp_key=key
            ),
            levels=tuple((level, tuple(items)) for level, items in levels.items()),
            query_keys=(),
        )).load()
        placements.add(built.index.level(4).physical_blocks())
    assert len(placements) > 1
    # the published analysis seed must not influence the placement at all
    same_key_other_seed = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=6, prp_key=77),
        levels=tuple((level, tuple(items)) for level, items in levels.items()),
        query_keys=(),
    )).load()
    assert (
        same_key_other_seed.index.level(4).physical_blocks()
        == run.index.level(4).physical_blocks()
    )
    assert (
        same_key_other_seed.index.level(6).physical_blocks()
        == run.index.level(6).physical_blocks()
    )


# ---------------------------------------------------------------------------
# 6. observation separation
# ---------------------------------------------------------------------------


def test_reference_artifact_leaks_no_evaluator_field():
    builder = ReferenceTraceBuilder(default_scenario(layout=LAYOUT_ORDERED, seed=7)).load().run_queries()
    artifact = builder.artifact()
    assert scan_forbidden_fields(artifact) == ()
    assert scan_forbidden_fields(artifact["transcript"]) == ()
    assert scan_forbidden_fields(artifact["public_config"]) == ()
    assert scan_forbidden_fields(artifact["workload"]) == ()

    text = dumps_reference_artifact(artifact)
    for token in ("item_rank", "block_rank", "placement", "permutation", "slot_of",
                  "\"keys\"", "\"values\"", "hit_level", "probes"):
        assert token not in text, token


def test_transcript_only_carries_the_frozen_observation_fields():
    builder = ReferenceTraceBuilder(default_scenario(seed=2)).load().run_queries()
    payload = builder.transcript_payload()
    assert set(payload) == {
        "schema_version", "source_schema_version", "equality_channel", "run_metadata",
        "operations", "summary",
    }
    assert payload["schema_version"] == "g2a-1"
    assert payload["equality_channel"] == "M0"
    assert {op["class"] for op in payload["operations"]} == {
        CLASS_BUILD_INITIAL, CLASS_BUILD_PGM, CLASS_QUERY,
    }
    for operation in payload["operations"]:
        assert set(operation) == {
            "index", "class", "events", "read_count", "write_count",
        }
        for event in operation["events"]:
            assert set(event) == {"seq", "operation", "slot_id"}
            assert event["operation"] in {READ, WRITE}
    assert set(payload["summary"]) == set(SUMMARY_KEYS)
    # the reference transcript is g2a-1-shaped; the G1-E-source validator differs
    # from it in exactly one field
    with pytest.raises(LeakageError):
        validate_transcript(payload)
    g1e_sourced = dict(payload, source_schema_version="g1e-1")
    validate_transcript(g1e_sourced)   # no other difference exists
    metrics = leakage_metrics(builder.transcript())
    assert metrics["query"]["query_count"] == payload["summary"]["query_count"]
    assert sorted(metrics) == [
        "equality_channel", "merge", "per_operation", "query", "schema_version",
        "source_schema_version", "transcript_schema_version",
    ]


def test_adversary_input_exposes_only_transcript_and_public_config():
    builder = ReferenceTraceBuilder(default_scenario(seed=4)).load().run_queries()
    artifact = builder.artifact()
    adversary_input = AdversaryInput.from_reference_artifact(artifact)
    assert scan_forbidden_fields(adversary_input.to_dict()) == ()
    assert set(adversary_input.to_dict()) == {"equality_channel", "public_config", "transcript"}
    assert len(adversary_input.query_traces()) == artifact["transcript"]["summary"]["query_count"]
    assert "truth" not in json.dumps(adversary_input.to_dict())

    with pytest.raises(AdversaryInputError):
        AdversaryInput(
            transcript=builder.transcript(),
            public_config={**artifact["public_config"], "keys": [1, 2, 3]},
        )
    with pytest.raises(AdversaryInputError):
        AdversaryInput.from_reference_artifact({"schema_version": "g1e-1"})
    with pytest.raises(AdversaryInputError):
        AdversaryInput.from_reference_artifact({"schema_version": REFERENCE_ARTIFACT_SCHEMA_VERSION})


def test_adversary_input_supports_the_optional_m1_channel():
    builder = ReferenceTraceBuilder(default_scenario(seed=4)).load().run_queries()
    artifact = builder.artifact()
    queries = builder.query_operations()
    labels = {op.index: f"QClass{position % 2}" for position, op in enumerate(queries)}
    adversary_input = AdversaryInput.from_reference_artifact(artifact, equality_labels=labels)
    assert adversary_input.equality_channel == "M1"
    assert all(
        op.query_equality_class == labels[op.index] for op in adversary_input.query_traces()
    )
    assert "M0" == AdversaryInput.from_reference_artifact(artifact).equality_channel


def test_adversary_module_does_not_import_evaluator_truth():
    source = Path(ref_adversary.__file__).read_text(encoding="utf-8")
    assert "import ref_truth" not in source
    assert "from .ref_truth" not in source
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    assert ".ref_truth" not in imported
    assert imported <= {".leakage", "json", "dataclasses", "typing", "__future__"}
    truth_symbols = sorted(
        name for name in ("ReferenceEvaluatorTruth", "query_truth", "block_key_interval")
        if name in ref_adversary.__dict__
    )
    assert truth_symbols == []


# ---------------------------------------------------------------------------
# 7. reproducibility
# ---------------------------------------------------------------------------


def test_same_config_and_seed_give_byte_identical_artifacts():
    first = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_STATIC_PRP, seed=9, prp_key=404, query_count=5)
    ).load().run_queries().artifact()
    second = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_STATIC_PRP, seed=9, prp_key=404, query_count=5)
    ).load().run_queries().artifact()
    assert dumps_reference_artifact(first) == dumps_reference_artifact(second)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_different_seed_changes_only_randomized_components():
    ordered = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_ORDERED, seed=1, query_count=5)
    ).load().run_queries().artifact()
    ordered_other = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_ORDERED, seed=2, query_count=5)
    ).load().run_queries().artifact()
    # same geometry and layout rule
    assert ordered["public_config"]["levels"] == ordered_other["public_config"]["levels"]
    assert ordered["public_config"]["base"] == ordered_other["public_config"]["base"]
    # the workload draw is a randomized component
    assert ordered["workload"] != ordered_other["workload"]
    assert ordered["transcript"]["run_metadata"]["seed"] == 1
    assert ordered_other["transcript"]["run_metadata"]["seed"] == 2


def test_prp_key_changes_placement_and_the_analysis_seed_does_not():
    levels = tuple(
        (level, tuple((1000 + rank, f"v{rank}") for rank in range(600)))
        for level in (4, 6)
    )
    first = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=21, prp_key=500),
        levels=levels, query_keys=(1000, 1100),
    )).load().run_queries()
    second = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=21, prp_key=501),
        levels=levels, query_keys=(1000, 1100),
    )).load().run_queries()
    assert [o.value for o in first.query_outcomes] == [o.value for o in second.query_outcomes]
    assert [o.slots() for o in first.query_outcomes] != [o.slots() for o in second.query_outcomes]
    # a different analysis seed with the same layout key leaves the placement alone
    third = ReferenceTraceBuilder(ReferenceExperiment(
        profile=ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=22, prp_key=500),
        levels=levels, query_keys=(1000, 1100),
    )).load().run_queries()
    assert [o.slots() for o in first.query_outcomes] == [o.slots() for o in third.query_outcomes]


# ---------------------------------------------------------------------------
# 8. passive workload generators
# ---------------------------------------------------------------------------


def test_record_uniform_is_seeded_and_draws_only_existing_keys():
    records = list(range(100, 140))
    workload = generate_passive_workload(
        WORKLOAD_RECORD_UNIFORM, seed=5, records=records, count=20
    )
    assert workload.mode == WORKLOAD_RECORD_UNIFORM
    assert set(workload.queries) <= set(records)
    assert workload.queries == generate_passive_workload(
        WORKLOAD_RECORD_UNIFORM, seed=5, records=records, count=20
    ).queries
    assert workload.queries == tuple(workload.queries)
    with pytest.raises(WorkloadError):
        generate_passive_workload(WORKLOAD_RECORD_UNIFORM, seed=1, count=3)


def test_domain_uniform_stays_in_the_declared_domain():
    workload = generate_passive_workload(
        WORKLOAD_DOMAIN_UNIFORM, seed=3, count=50, domain=(10, 20)
    )
    assert all(10 <= key < 20 for key in workload.queries)
    assert workload.public_dict()["domain"] == [10, 20]
    with pytest.raises(WorkloadError):
        generate_passive_workload(WORKLOAD_DOMAIN_UNIFORM, seed=3, count=5, domain=(5, 5))


def test_data_empirical_respects_the_supplied_distribution():
    workload = generate_passive_workload(
        WORKLOAD_DATA_EMPIRICAL, seed=4, count=60,
        empirical={1: 1.0, 2: 0.0, 3: 3.0},
    )
    assert set(workload.queries) <= {1, 3}
    assert 2 not in workload.queries           # zero weight never sampled
    assert workload.queries.count(3) > workload.queries.count(1)
    with pytest.raises(WorkloadError):
        generate_passive_workload(WORKLOAD_DATA_EMPIRICAL, seed=4, count=2, empirical={1: 0.0})
    with pytest.raises(WorkloadError):
        generate_passive_workload(WORKLOAD_DATA_EMPIRICAL, seed=4, count=2, empirical={1: -1.0})
    with pytest.raises(WorkloadError):
        generate_passive_workload("nope", seed=4, count=2)


def test_workload_provenance_never_claims_the_upstream_distribution():
    workload = generate_passive_workload(
        WORKLOAD_RECORD_UNIFORM, seed=1, records=[1, 2, 3], count=4
    )
    assert workload.provenance == UPSTREAM_WORKLOAD_NOTE
    assert "NOT the original upstream workload" in workload.provenance
    assert "queries" in workload.to_dict()
    assert "queries" not in workload.public_dict()
    assert WORKLOAD_MODES == (
        WORKLOAD_RECORD_UNIFORM, WORKLOAD_DOMAIN_UNIFORM, WORKLOAD_DATA_EMPIRICAL,
    )


# ---------------------------------------------------------------------------
# 9. evaluator truth
# ---------------------------------------------------------------------------


def test_evaluator_truth_answers_the_frozen_questions(tmp_path):
    experiment = default_scenario(seed=8, query_count=3)
    builder = ReferenceTraceBuilder(experiment).load().run_queries()
    truth = ReferenceEvaluatorTruth(builder.index)

    slot = truth.slot_of(6, 2)
    assert truth.level_of(slot) == 6
    assert truth.block_rank_of(slot) == 2
    assert truth.locate(slot) == (6, 2)
    assert truth.slot_of_block(6, 2) == slot

    interval = truth.block_key_interval(6, 2)
    reference = builder.index.level(6)
    assert interval["item_start"] == 2 * reference.items_per_block
    assert interval["item_end"] == min(3 * reference.items_per_block, reference.item_count)
    assert interval["key_lo"] == reference.keys[interval["item_start"]].value
    assert interval["key_hi"] == reference.keys[interval["item_end"] - 1].value
    assert interval["slot_id"] == slot.value

    hit_key = experiment.query_keys[0]
    outcome = builder.index.query(hit_key)
    query_truth = truth.query_truth(hit_key)
    assert query_truth["found"] == outcome.found
    assert query_truth["hit_level"] == outcome.hit_level
    assert query_truth["hit_item_rank"] == truth.item_rank_of(outcome.hit_level, hit_key)
    assert query_truth["searched_levels"] == list(outcome.searched_levels)
    assert query_truth["per_level"][0]["path"] in {"full_level", "pgm_bounded"}

    miss_truth = truth.query_truth(-1)
    assert miss_truth["found"] is False and miss_truth["hit_level"] is None

    payload = truth.to_dict(query_keys=(hit_key,))
    assert set(payload) == set(ref_truth.TRUTH_KEYS)
    assert payload["classification"]["truth"] == "EVALUATOR-ONLY"
    assert payload["queries"][0]["key"] == hit_key
    written = truth.write_json(tmp_path / "truth.json", query_keys=(hit_key,))
    assert written["levels"] and written["projection"]["regions"]


def test_evaluator_truth_never_reaches_the_adversary_artifact():
    experiment = default_scenario(seed=8, query_count=2)
    builder = ReferenceTraceBuilder(experiment).load().run_queries()
    artifact_text = dumps_reference_artifact(builder.artifact())
    truth_text = ReferenceEvaluatorTruth(builder.index).to_json(
        query_keys=experiment.query_keys
    )
    assert "query_truth" not in artifact_text
    for field in ("item_rank", "hit_block_rank", "physical_block_offset", "key_lo"):
        assert field not in artifact_text
        assert field in truth_text


# ---------------------------------------------------------------------------
# 10. reference vs enhanced baseline, and no attack code
# ---------------------------------------------------------------------------


def test_reference_profile_leaves_the_enhanced_baseline_untouched():
    """The two lookup profiles coexist: G1-B2 stays minimal-cover read-all."""
    from enhanced_letindex.builder import validate_level
    from enhanced_letindex.config import Config
    from enhanced_letindex.engine import TrustedEngine
    from enhanced_letindex.record import Record
    from enhanced_letindex.identifiers import RecordKey

    engine = TrustedEngine(Config(block_capacity=4, pgm_epsilon=2))
    level_id = engine.build_level([Record(RecordKey(k), f"v{k}") for k in range(8)])
    engine.build_level_pgm(level_id)
    engine.trace.clear()
    engine.lookup_level(level_id, RecordKey(3))
    enhanced_events = engine.trace.events()
    assert enhanced_events, "the accepted G1-B2 path still performs reads"

    profile = ReferenceProfile.reference()
    index = ReferenceIndex.build(profile, {3: [(k, f"v{k}") for k in range(8)]})
    outcome = index.lookup_level(3, 3)
    assert outcome.path == "full_level"      # a different, upstream-shaped path

    # the reference profile never touches G1 modules: it is a separate profile
    source = Path(letindex_ref.__file__).read_text(encoding="utf-8")
    assert "from .engine" not in source
    assert "from .builder" not in source
    assert "import engine" not in source
    assert "TrustedEngine(" not in source


def test_no_attack_recovery_or_defense_code_is_introduced():
    forbidden = (
        "attack", "recover", "recovery", "infer", "cdf", "ord_forced", "rank_width",
        "adj_recall", "defence", "defense", "reshuffle", "stash", "encrypt", "oram",
        "privacy", "score", "success", "correlate", "correlation", "budget",
    )
    names = set()
    for module in (letindex_ref, ref_workload, ref_truth, ref_adversary):
        names.update(dir(module))
        for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    offenders = sorted(
        name for name in (n.lower() for n in names)
        for token in forbidden if token in name
    )
    assert offenders == []
    # the only mention of recovery terms is the explicit "not implemented" note
    assert "no attack" in letindex_ref.__doc__.lower()
    with pytest.raises(ReferenceProfileError):
        # a profile can never be constructed that claims legacy semantics
        ReferenceProfile(items_per_block=16)


# ---------------------------------------------------------------------------
# 9. reviewer round-1 corrections (static_prp truth, layout-key separation)
# ---------------------------------------------------------------------------


def placement_truth_fixture():
    """A ``static_prp`` level with 8 logical blocks (C=8), for rank-level checks."""
    profile = ReferenceProfile.experimental_variant(
        "placement-truth", layout=LAYOUT_STATIC_PRP, seed=31, prp_key=1234,
        items_per_block=8, epsilon=8,
    )
    items = [(1000 + rank, f"v{rank}") for rank in range(64)]
    index = ReferenceIndex.build(profile, {LEVEL4: items})
    return profile, index, ReferenceEvaluatorTruth(index)


def test_static_prp_truth_maps_logical_ranks_through_the_placement():
    """Review fix 1: ``slot_of`` takes a logical rank, ``locate`` returns one.

    The truth must apply the level placement in **both** directions: a logical block
    rank goes through ``physical_block(rank)`` before the ``(level, offset) -> SlotId``
    projection, and a SlotId goes through the inverse placement after it.
    """
    _profile, index, truth = placement_truth_fixture()
    reference = index.level(LEVEL4)
    placement = reference.physical_blocks()
    assert reference.block_count == 8
    assert sorted(placement) == list(range(8))
    assert placement != tuple(range(8))          # a real permutation, not the identity

    identity_slots = {
        index.projection.slot(LEVEL4, rank).value
        for rank in range(reference.block_count)
    }
    for rank in range(reference.block_count):
        physical = placement[rank]
        assert truth.physical_offset_of(LEVEL4, rank) == physical
        assert truth.logical_block_rank(LEVEL4, physical) == rank
        slot = truth.slot_of(LEVEL4, rank).value
        assert slot == index.projection.slot(LEVEL4, physical).value
        # both directions round-trip
        assert truth.locate(SlotId(slot)) == (LEVEL4, rank)
        assert truth.level_of(SlotId(slot)) == LEVEL4
        assert truth.block_rank_of(SlotId(slot)) == rank
        assert truth.physical_offset_of_slot(SlotId(slot)) == physical
        assert truth.physical_offset(SlotId(slot)) == (LEVEL4, physical)
        assert truth.slot_of_block(LEVEL4, rank) == truth.slot_of(LEVEL4, rank)

    # the old (wrong) mapping used the logical rank as a physical offset, so the two
    # disagree for this layout — this is exactly what the review caught
    assert {
        truth.slot_of(LEVEL4, rank).value for rank in range(reference.block_count)
    } == identity_slots
    assert any(
        truth.slot_of(LEVEL4, rank).value != index.projection.slot(LEVEL4, rank).value
        for rank in range(reference.block_count)
    )

    with pytest.raises(ReferenceProfileError):
        truth.slot_of(LEVEL4, 8)
    with pytest.raises(ReferenceProfileError):
        truth.physical_offset_of(LEVEL4, -1)
    with pytest.raises(ReferenceProfileError):
        truth.logical_block_rank(LEVEL4, 99)

    # ordered contrast: the placement is the identity, so both agree
    ordered_profile = ReferenceProfile.reference()
    ordered_index = ReferenceIndex.build(
        ordered_profile, {LEVEL4: [(1000 + rank, "v") for rank in range(2048)]}
    )
    ordered_truth = ReferenceEvaluatorTruth(ordered_index)
    assert ordered_index.level(LEVEL4).block_count == 8
    for rank in range(8):
        assert ordered_truth.slot_of(LEVEL4, rank).value == (
            ordered_index.projection.slot(LEVEL4, rank).value
        )
        assert ordered_truth.locate(ordered_index.projection.slot(LEVEL4, rank)) == (
            LEVEL4, rank,
        )


def test_static_prp_truth_block_and_query_fields_use_the_placement():
    """Review fix 1: ``block_key_interval`` / ``query_truth`` slot fields are placed."""
    _profile, index, truth = placement_truth_fixture()
    reference = index.level(LEVEL4)
    placement = reference.physical_blocks()

    for rank in range(reference.block_count):
        interval = truth.block_key_interval(LEVEL4, rank)
        assert interval["block_rank"] == rank
        assert interval["logical_block_rank"] == rank
        assert interval["physical_block_offset"] == placement[rank]
        assert interval["slot_id"] == truth.slot_of(LEVEL4, rank).value
        assert interval["slot_id"] == index.projection.slot(
            LEVEL4, placement[rank]
        ).value
        assert interval["key_lo"] == 1000 + rank * 8
        assert interval["item_end"] == min((rank + 1) * 8, 64)

    hit_key = 1000 + 5 * 8 + 3        # item 43 -> logical block 5
    truth_entry = truth.query_truth(hit_key)
    assert truth_entry["found"] is True
    assert truth_entry["hit_item_rank"] == 43
    assert truth_entry["hit_block_rank"] == 5
    assert truth_entry["hit_physical_block_offset"] == placement[5]
    assert truth_entry["hit_slot_id"] == truth.slot_of(LEVEL4, 5).value
    assert truth_entry["hit_slot_id"] == index.projection.slot(LEVEL4, placement[5]).value

    for entry in truth_entry["per_level"]:
        if entry["slot_id"] is None:
            continue
        assert entry["physical_block_offset"] == placement[entry["block_rank"]]
        assert entry["slot_id"] == truth.slot_of(
            entry["level"], entry["block_rank"]
        ).value

    payload = truth.to_dict(query_keys=(hit_key,))
    (level_entry,) = payload["levels"]
    assert level_entry["placement"] == list(placement)
    assert payload["classification"]["prp_key"] == 1234
    assert payload["classification"]["layout"] == LAYOUT_STATIC_PRP


def test_attack_input_cannot_reconstruct_the_static_prp():
    """Review fix 2: the PRP-driving key never reaches the attack-facing input.

    The frozen ``g2a-1`` transcript must publish its run seed, so the permutation has to
    be keyed by something else entirely.  A G2-B1 consumer that tries to rederive the
    placement from everything it is given (layout mode, geometry, block counts, the run
    seed) must fail.
    """
    experiment = default_scenario(
        layout=LAYOUT_STATIC_PRP, seed=17, prp_key=8642, query_count=4
    )
    builder = ReferenceTraceBuilder(experiment).load().run_queries()
    artifact = builder.artifact()
    adversary_input = AdversaryInput.from_reference_artifact(artifact)
    payload = adversary_input.to_dict()
    text = adversary_input.to_json()

    assert "seed" not in payload["public_config"]
    assert "prp_key" not in payload["public_config"]
    assert "prp_key" not in text
    assert "placement" not in text and "permutation" not in text
    assert payload["public_config"]["layout"] == LAYOUT_STATIC_PRP   # the mode is public
    assert payload["transcript"]["run_metadata"]["seed"] == 17       # the run seed is not

    level = builder.index.level(6)
    placement = level.physical_blocks()
    assert placement != tuple(range(level.block_count))

    # the attacker's best attempt from its own input: the documented derivation keyed by
    # the published run seed, over the publicly known block count
    import random

    for key_material in (17, payload["transcript"]["run_metadata"]["seed"]):
        for level_number in range(3, 7):
            order = list(range(level.block_count))
            random.Random(key_material * 1_000_003 + level_number).shuffle(order)
            assert tuple(order) != placement, level_number

    # the layout key itself stays in the evaluator-side reproducibility record only
    assert artifact["reproducibility"]["prp_key"] == 8642
    assert artifact["public_config"] == payload["public_config"]


def test_attack_public_config_rejects_the_seed_and_the_layout_key():
    builder = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_STATIC_PRP, seed=2, prp_key=9, query_count=2)
    ).load().run_queries()
    artifact = builder.artifact()
    public_config = artifact["public_config"]
    transcript = builder.transcript()

    assert AdversaryInput(transcript=transcript, public_config=public_config)
    for leaked in ({"seed": 2}, {"prp_key": 9}, {"analysis_seed": 2}, {"placement": [0, 1]}):
        with pytest.raises(AdversaryInputError):
            AdversaryInput(
                transcript=transcript,
                public_config={**public_config, **leaked},
            )
    # the scan of the whole artifact stays clean: the evaluator-side sections use only
    # documented artifact metadata names
    assert scan_forbidden_fields(artifact) == ()
    # and the attack-public whitelist itself excludes the seed and the layout key
    assert PUBLIC_CONFIG_KEYS.isdisjoint({"seed", "prp_key", "analysis_seed", "placement"})
    assert "geometry" not in PUBLIC_CONFIG_KEYS
    assert ALLOWED_FIELD_UNIVERSE.isdisjoint({"placement", "permutation"})


def test_static_prp_profile_is_never_labelled_upstream_faithful():
    ordered = ReferenceProfile.reference()
    assert ordered.is_reference_defaults() and ordered.is_upstream_faithful()
    assert ordered.classification() == {
        "profile": "UPSTREAM-FAITHFUL",
        "geometry": "UPSTREAM-FAITHFUL",
        "layout": "UPSTREAM-FAITHFUL",
        "lookup_semantics": "UPSTREAM-FAITHFUL",
    }
    assert ordered.to_dict()["classification"]["profile"] == "UPSTREAM-FAITHFUL"

    prp = ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=1, prp_key=2)
    assert prp.is_reference_defaults()             # geometry is unchanged ...
    assert not prp.is_upstream_faithful_layout()   # ... but the layout is an extension
    assert not prp.is_upstream_faithful()
    assert prp.classification()["profile"] == "EXPERIMENTAL EXTENSION"
    assert prp.classification()["geometry"] == "UPSTREAM-FAITHFUL"
    assert prp.classification()["layout"] == "EXPERIMENTAL EXTENSION"

    prp_artifact = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_STATIC_PRP, seed=1, prp_key=2, query_count=2)
    ).load().run_queries().artifact()
    assert prp_artifact["classification"]["profile"] == "EXPERIMENTAL EXTENSION"
    assert prp_artifact["classification"]["layout"] == "EXPERIMENTAL EXTENSION"
    assert prp_artifact["classification"]["geometry"] == "UPSTREAM-FAITHFUL"
    assert prp_artifact["classification"]["lookup_semantics"] == "UPSTREAM-FAITHFUL"

    ordered_artifact = ReferenceTraceBuilder(
        default_scenario(layout=LAYOUT_ORDERED, seed=1, query_count=2)
    ).load().run_queries().artifact()
    assert ordered_artifact["classification"]["profile"] == "UPSTREAM-FAITHFUL"


def test_prp_key_must_be_explicit_and_never_derived_from_the_run_seed():
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5)
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=5)
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile.reference(layout=LAYOUT_ORDERED, prp_key=5)
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=True)
    with pytest.raises(ReferenceProfileError):
        ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key="5")
    with pytest.raises(ReferenceProfileError):        # ordered has no permutation key
        ReferenceProfile.reference().layout_key()

    prp = ReferenceProfile.reference(layout=LAYOUT_STATIC_PRP, seed=5, prp_key=99)
    assert prp.layout_key() == 99
    assert ReferenceProfile.from_dict(prp.to_dict()).prp_key == 99
    public = prp.public_dict()
    assert "seed" not in public and "prp_key" not in public
    assert public["layout"] == LAYOUT_STATIC_PRP
    assert set(public) < set(prp.to_dict())


def test_cli_requires_the_layout_key_for_static_prp(tmp_path):
    out = tmp_path / "reference.json"
    missing = subprocess.run(
        [sys.executable, "-m", "enhanced_letindex.letindex_ref",
         "--layout", "static_prp", "--seed", "4", "--out", str(out)],
        cwd=str(CODES_DIR), env=CLI_ENV, capture_output=True, text=True,
    )
    assert missing.returncode == 2
    assert "prp-key" in missing.stdout
    assert not out.exists()

    ordered_with_key = subprocess.run(
        [sys.executable, "-m", "enhanced_letindex.letindex_ref",
         "--layout", "ordered", "--prp-key", "7", "--out", str(out)],
        cwd=str(CODES_DIR), env=CLI_ENV, capture_output=True, text=True,
    )
    assert ordered_with_key.returncode == 2
    assert not out.exists()

    ok = subprocess.run(
        [sys.executable, "-m", "enhanced_letindex.letindex_ref",
         "--layout", "static_prp", "--seed", "4", "--prp-key", "123",
         "--queries", "4", "--out", str(out)],
        cwd=str(CODES_DIR), env=CLI_ENV, capture_output=True, text=True,
    )
    assert ok.returncode == 0, ok.stderr
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["reproducibility"]["prp_key"] == 123
    assert artifact["reproducibility"]["analysis_seed"] == 4
    assert "prp_key" not in json.dumps(artifact["public_config"])
    assert "seed" not in artifact["public_config"]
    # the attack-facing view of that artifact still cannot see the layout key
    adversary_input = AdversaryInput.from_reference_artifact(artifact)
    assert adversary_input.public_config == artifact["public_config"]
    assert '"prp_key"' not in adversary_input.to_json()

