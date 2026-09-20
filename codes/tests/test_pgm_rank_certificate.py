"""G5-A.1 focused tests (Issue #23 §8): exact certified OPLM rank-line metadata.

The module under test is the accepted canonical PGM machinery in
``enhanced_letindex.pgm`` plus the incremental builder in
``enhanced_letindex.incremental_merge``.  G4/G5-A behaviour is used as the oracle and
must stay unchanged.

Section map (Issue #23 §8):

```text
 2  empty PGM carries no certificate
 3  one-point segments: exact clipped-band midpoint (denominator 2)
 4  multi-point certificate formula vs an independent Fraction oracle
 5  exact clipped-band invariant by integer cross multiplication
 6  exact start/end key and inclusive-rank spans, contiguous 0..n-1
 7  edge fixtures (eps 0, eps >= n, negative/mixed-sign keys, huge gaps, ...)
 8  hard-coded rounding-gap fixture (predictor error > epsilon, certificate <= epsilon)
 9  legacy search fields / predict / SearchResult / lower_bound unchanged
10  batch vs incremental certificate identity across schedules
11  a certificate crosses advance and output-block boundaries without splitting
12  no full key stream retained; one certificate per segment, O(segments)
13  no float/Decimal in the production certificate path
14  newer-wins overlap guard counterexample (no (key, RID) semantics)
15  scope guard: the imported certificate modules add nothing physical
```

.. note::

   DERIVED TEST, NOT BYTE-IDENTICAL IMPORT.
   Imported for M0 from EnhancedLETIndex@4e68be75 and reduced to the parts
   that exercise the frozen common LETIndex substrate only.
   Dropped tests: test_1 (G5-A status closure pin against the original
   repository) and test_15b (runs the excluded G4 suites).
   test_15 is reduced to its common-substrate assertion (source-scope tokens
   of the imported pgm.py / incremental_merge.py); its original baseline-tag,
   milestone-diff and defence-vocabulary assertions are repository-specific.
   All other tests are byte-identical to the source; no assertion was weakened.
"""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.incremental_merge import (
    IncrementalMergeJob,
    IncrementalPgmBuilder,
    LogicalRunView,
)
from enhanced_letindex.pgm import (
    BatchPgmIndex,
    PgmError,
    PgmSegment,
    RankLineCertificate,
    _CanonicalSegmentGeometry,
    _OptimalPiecewiseLinearModel,
    _Point,
    _rank_line_certificate,
    build_batch_pgm,
    make_segmentation,
)

CODES_DIR = Path(__file__).resolve().parents[1]
CLI_ENV = {**os.environ, "PYTHONPATH": str(CODES_DIR / "src")}
PGM_PATH = CODES_DIR / "src" / "enhanced_letindex" / "pgm.py"
INCREMENTAL_PATH = CODES_DIR / "src" / "enhanced_letindex" / "incremental_merge.py"


#: The five accepted search fields; G5-A.1 must not change them.
LEGACY_SEARCH_FIELDS = ("start_key", "start_rank", "slope_num", "slope_den", "intercept")
CERTIFICATE_FIELDS = (
    "start_key", "end_key", "start_rank", "end_rank", "slope_num", "origin_num",
    "denominator",
)

ITEMS_PER_BLOCK = 8
SEED = 31

#: The hand-found rounding-gap fixture (Issue #23 §8.8): with epsilon = 1 the accepted
#: integer search predictor predicts rank 0 for the point whose true rank is 2 (error 2 >
#: epsilon) while the certificate line stays exactly inside the band.
ROUNDING_GAP_KEYS = (0, 1, 2, 9)
ROUNDING_GAP_EPSILON = 1
ROUNDING_GAP_SEGMENT = (3, 1, 7)          # slope_num, origin_num, denominator

#: A three-segment fixture (Issue #23 §8.11): 3 groups of 10 keys, gaps of 100000.
SEGMENT_GROUPS = 3
SEGMENT_GROUP_SIZE = 10
SEGMENT_GAP = 100_000
SEGMENT_EPSILON = 0
SEGMENT_ITEMS_PER_BLOCK = 4


def segment_group_keys() -> tuple:
    keys = []
    base = 0
    for _ in range(SEGMENT_GROUPS):
        keys.extend(base + rank for rank in range(SEGMENT_GROUP_SIZE))
        base += SEGMENT_GAP
    return tuple(keys)


SEGMENT_KEYS = segment_group_keys()


def key_tuple(values) -> tuple:
    return tuple(RecordKey(value) for value in values)


def segments_of(values, epsilon) -> tuple:
    return make_segmentation(key_tuple(values), epsilon)


def fraction_of(certificate: RankLineCertificate, key: RecordKey) -> Fraction:
    """The represented line value at ``key``, computed in exact rationals."""
    return Fraction(certificate.numerator_at(key), certificate.denominator)


def independent_canonical_line(
    keys: tuple, epsilon: int, certificate: RankLineCertificate
) -> tuple:
    """The canonical line of one segment, recomputed *independently* in Fractions.

    A fresh canonical OPLM is replayed over the segment's own real points (its true
    keys with their global ranks) and its rectangle is read directly; the expected
    slope/origin are then computed with ``Fraction`` arithmetic only — never with
    ``certificate.numerator_at`` or ``certificate.within_band``.  Returns
    ``(expected_slope, expected_origin)`` for
    ``L(x) = expected_slope * (x - start_key) + expected_origin``:

    ```text
    non-singleton   expected_slope  = Fraction(r3.y - r1.y, r3.x - r1.x)
                    expected_origin = expected_slope * (x0 - r1.x) + r1.y
    singleton       expected_slope  = 0
                    expected_origin = Fraction(r0.y + r1.y, 2)
    ```
    """
    model = _OptimalPiecewiseLinearModel(epsilon)
    for rank in range(certificate.start_rank, certificate.end_rank + 1):
        assert model.add_point(keys[rank].value, rank) is True
    geometry = model.get_segment()
    r0, r1, _r2, r3 = geometry.rectangle
    x0 = certificate.start_key.value
    if geometry.one_point:
        assert certificate.start_rank == certificate.end_rank
        return Fraction(0), Fraction(r0.y + r1.y, 2)
    expected_slope = Fraction(r3.y - r1.y, r3.x - r1.x)
    expected_origin = expected_slope * (x0 - r1.x) + r1.y
    return expected_slope, expected_origin


def assert_certificates_are_sound(values, epsilon) -> tuple:
    """The normative checks of §4/§5/§6 over one fixture; returns the segments."""
    keys = key_tuple(values)
    segments = make_segmentation(keys, epsilon)
    if not keys:
        assert segments == ()
        return segments
    assert segments[0].start_rank == 0
    assert segments[0].certificate.start_key == keys[0]
    for index, segment in enumerate(segments):
        certificate = segment.certificate
        # one certificate per segment, attached and aligned with its search fields
        assert segment.certificate is not None
        assert certificate.start_rank == segment.start_rank
        assert certificate.start_key == segment.start_key
        # exact spans and contiguous coverage
        assert certificate.end_rank >= certificate.start_rank
        assert certificate.end_key == keys[certificate.end_rank]
        if index + 1 < len(segments):
            assert segments[index + 1].start_rank == certificate.end_rank + 1
        else:
            assert certificate.end_rank == len(keys) - 1
            assert certificate.end_key == keys[-1]
        assert certificate.denominator > 0
        # the two stored representations agree on the rational slope
        assert Fraction(certificate.slope_num, certificate.denominator) == Fraction(
            segment.slope_num, segment.slope_den
        )
        for rank in range(certificate.start_rank, certificate.end_rank + 1):
            key = keys[rank]
            # the normative clipped-band inequality, by exact cross multiplication
            assert certificate.within_band(key, rank, epsilon), (index, rank)
            value = fraction_of(certificate, key)
            assert abs(value - rank) <= epsilon, (index, rank)
            # ... and the stored rounded-intercept line is at most 1/2 away (bound B)
            rounded = Fraction(segment.slope_num, segment.slope_den) * (
                key.value - certificate.start_key.value
            ) + segment.intercept
            assert abs(value - rounded) <= Fraction(1, 2), (index, rank)
    return segments


# ---------------------------------------------------------------------------
# 2. empty PGM
# ---------------------------------------------------------------------------


def test_2_an_empty_pgm_has_no_segment_and_no_certificate():
    """Issue #23 §8.2 + §4: the empty PGM has no segments/certificates."""
    assert make_segmentation((), 3) == ()
    index = build_batch_pgm((), 3)
    assert index.segments == ()
    assert (index.record_count, index.first_key, index.last_key) == (0, None, None)
    # both builders agree on the empty stream
    builder = IncrementalPgmBuilder(3)
    assert builder.finalize() == index
    assert builder.finalize().segments == ()
    assert builder.segment_start_ranks == ()


# ---------------------------------------------------------------------------
# 3. one-point segments
# ---------------------------------------------------------------------------


def code_without_docstrings(path: Path) -> str:
    """The executable source of a module (module/class/function docstrings removed)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    code = source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                code = code.replace(doc, "")
    return code


def test_3_one_point_segments_keep_the_exact_clipped_band_midpoint():
    """Issue #23 §3: slope 0, denominator 2, exact midpoint of the clipped band."""
    # a single-key PGM: one one-point segment whose clipped band is 0 .. epsilon
    for epsilon in (0, 1, 2, 7):
        segments = segments_of((42,), epsilon)
        assert len(segments) == 1
        certificate = segments[0].certificate
        assert certificate.slope_num == 0
        assert certificate.denominator == 2
        assert certificate.origin_num == epsilon + max(0, 0 - epsilon)      # r0.y + r1.y
        assert certificate.rank_span == (0, 0)
        assert certificate.key_span == (RecordKey(42), RecordKey(42))
        # the exact midpoint, not the integer midpoint of the legacy search model
        assert fraction_of(certificate, RecordKey(42)) == Fraction(epsilon, 2)
        assert segments[0].intercept == (epsilon + max(0, 0 - epsilon)) // 2
        assert certificate.within_band(RecordKey(42), 0, epsilon)
    # the odd clipped band is where the two representations really differ: rank 0 with
    # epsilon = 1 gives the exact 1/2 while the accepted intercept floors it to 0
    odd = segments_of((42,), 1)[0]
    assert (odd.certificate.origin_num, odd.certificate.denominator) == (1, 2)
    assert fraction_of(odd.certificate, RecordKey(42)) == Fraction(1, 2)
    assert odd.intercept == 0
    even = segments_of((42,), 2)[0]
    assert (even.certificate.origin_num, even.certificate.denominator) == (2, 2)
    assert fraction_of(even.certificate, RecordKey(42)) == 1
    assert even.intercept == 1

    # a one-point *final* segment (the huge last key starts its own segment)
    even_tail = segments_of((0, 1, 2, 1_000_000), 0)
    assert [segment.certificate.rank_span for segment in even_tail] == [(0, 2), (3, 3)]
    tail = even_tail[-1].certificate
    assert (tail.slope_num, tail.origin_num, tail.denominator) == (0, 6, 2)
    assert fraction_of(tail, RecordKey(1_000_000)) == 3
    assert even_tail[-1].intercept == 3

    odd_tail = segments_of((0, 1, 2, 3, 1_000_000), 0)
    assert [segment.certificate.rank_span for segment in odd_tail] == [(0, 3), (4, 4)]
    tail = odd_tail[-1].certificate
    assert (tail.slope_num, tail.origin_num, tail.denominator) == (0, 8, 2)
    assert fraction_of(tail, RecordKey(1_000_000)) == 4
    assert odd_tail[-1].intercept == 4

    # the frozen one-point rule: origin_num == (end_rank + epsilon) + max(0, end_rank - eps)
    for values, epsilon in (((42,), 1), ((42,), 2), ((0, 1, 2, 1_000_000), 0),
                            ((0, 1, 2, 3, 1_000_000), 0)):
        for segment in segments_of(values, epsilon):
            certificate = segment.certificate
            if certificate.start_rank == certificate.end_rank:
                rank = certificate.end_rank
                assert certificate.origin_num == (rank + epsilon) + max(0, rank - epsilon)
                assert certificate.denominator == 2 and certificate.slope_num == 0


# ---------------------------------------------------------------------------
# 4./5./6./7. exact invariants over a fixture table
# ---------------------------------------------------------------------------


#: Edge fixtures of Issue #23 §8.7 (name -> (keys, epsilon)).
EDGE_FIXTURES = {
    "empty": ((), 3),
    "single_key": ((42,), 1),
    "eps_zero_collinear": (tuple(range(0, 30)), 0),
    "eps_zero_singletons": ((0, 1, 5, 6, 100), 0),
    "eps_ge_n": (tuple(range(-5, 5)), 50),
    "negative_keys": ((-9, -4, -1, 0, 3), 1),
    "mixed_sign": ((-1_000, -1, 0, 1, 7, 10 ** 6), 3),
    "huge_gaps": ((0, 1, 2, 10 ** 9, 10 ** 9 + 1), 2),
    "multi_segment": (SEGMENT_KEYS, 0),
    "gappy": ((0, 1, 2, 3, 10 ** 6, 10 ** 6 + 1, 2 * 10 ** 6), 1),
    "one_point_final": ((0, 1, 2, 1_000_000), 2),
    "wide_gaps_many": (tuple([0, 1, 2, 50_000, 50_001, 50_002, 10 ** 7, 10 ** 7 + 3]), 1),
}


def test_4_5_6_7_certificate_invariants_over_the_edge_fixtures():
    """Issue #23 §8.4/§8.5/§8.6/§8.7: exact formula, band, spans, edge shapes."""
    for label, (values, epsilon) in EDGE_FIXTURES.items():
        segments = assert_certificates_are_sound(values, epsilon)
        if not values:
            continue
        # §8.7: multiple segments and singleton final segments really are exercised
        assert segments
        covered = sum(
            segment.certificate.end_rank - segment.certificate.start_rank + 1
            for segment in segments
        )
        assert covered == len(values), label
        if label in ("multi_segment", "gappy", "wide_gaps_many", "eps_zero_singletons"):
            assert len(segments) > 1, label


def test_4b_the_certificate_matches_a_fraction_oracle_on_randomized_fixtures():
    """Issue #23 §8.4/§8.5 with randomized support (not a substitute) + an
    independent Fraction oracle of the frozen formula (fresh OPLM replay per
    segment, no certificate evaluation API)."""
    rng = random.Random(2026)
    for trial in range(120):
        size = rng.randint(1, 60)
        span = rng.choice([20, 200, 10 ** 5, 10 ** 9])
        size = min(size, span)
        values = tuple(sorted(rng.sample(range(-span, span), size)))
        epsilon = rng.choice([0, 1, 2, 3, 5, 8, 64, len(values), len(values) + 3])
        assert_certificates_are_sound(values, epsilon)

    # ... and the frozen formula itself is checked against an *independent* oracle:
    # for every segment a fresh canonical OPLM is replayed over that segment's own
    # real points and its rectangle is read directly, with the expected slope/origin
    # (and the line values) computed in Fraction arithmetic only - never through
    # certificate.numerator_at() or certificate.within_band()
    fixtures = dict(EDGE_FIXTURES)
    fixtures["rounding_gap"] = (ROUNDING_GAP_KEYS, ROUNDING_GAP_EPSILON)
    fixtures["multi_segment_eps0"] = (SEGMENT_KEYS, 0)
    fixtures["multi_segment_eps3"] = (SEGMENT_KEYS, 3)
    fixtures["dense_gappy"] = (
        tuple(sorted({0, 1, 3, 4, 5, 100, 101, 10 ** 6, 10 ** 6 + 7})), 2,
    )
    checked = 0
    for label, (values, epsilon) in fixtures.items():
        keys = key_tuple(values)
        for segment in make_segmentation(keys, epsilon):
            certificate = segment.certificate
            expected_slope, expected_origin = independent_canonical_line(
                keys, epsilon, certificate
            )
            # stored integers, as Fractions, equal the independently recomputed line
            assert Fraction(certificate.slope_num, certificate.denominator) == expected_slope, label
            assert Fraction(certificate.origin_num, certificate.denominator) == expected_origin, label
            # ... at several points of the segment, evaluated with Fractions only
            ranks = {certificate.start_rank, certificate.end_rank,
                     (certificate.start_rank + certificate.end_rank) // 2}
            for rank in sorted(ranks):
                key = keys[rank]
                expected_value = expected_slope * (
                    key.value - certificate.start_key.value
                ) + expected_origin
                assert fraction_of(certificate, key) == expected_value, (label, rank)
                assert abs(expected_value - rank) <= epsilon, (label, rank)
            checked += 1
    assert checked >= 25                              # the oracle really ran many segments


# ---------------------------------------------------------------------------
# 8. the hard-coded rounding-gap fixture
# ---------------------------------------------------------------------------


def test_8_the_hard_coded_rounding_gap_fixture_separates_line_from_predictor():
    """Issue #23 §8.8: predictor error > epsilon while the certificate stays in band."""
    segments = segments_of(ROUNDING_GAP_KEYS, ROUNDING_GAP_EPSILON)
    assert len(segments) == 1
    segment = segments[0]
    certificate = segment.certificate

    # the literal accepted search model of this fixture (unchanged by G5-A.1)
    assert (segment.start_key.value, segment.start_rank) == (0, 0)
    assert (segment.slope_num, segment.slope_den, segment.intercept) == (3, 7, 0)
    # ... and the literal certificate
    assert (
        certificate.slope_num, certificate.origin_num, certificate.denominator
    ) == ROUNDING_GAP_SEGMENT
    assert certificate.rank_span == (0, 3)
    assert certificate.key_span == (RecordKey(0), RecordKey(9))

    keys = key_tuple(ROUNDING_GAP_KEYS)
    # the *search predictor* is wrong by 2 > epsilon = 1 at the training point of rank 2
    assert segment.predict(keys[2]) == 0
    assert abs(segment.predict(keys[2]) - 2) == 2 > ROUNDING_GAP_EPSILON
    # ... while the certificate line is exactly inside the band there (tight, = epsilon)
    assert fraction_of(certificate, keys[2]) == Fraction(1, 1)
    assert abs(fraction_of(certificate, keys[2]) - 2) == ROUNDING_GAP_EPSILON
    assert certificate.within_band(keys[2], 2, ROUNDING_GAP_EPSILON)

    # the accepted lookup contract is nevertheless intact: the SearchResult window
    # still contains the true lower-bound rank of that key
    index = build_batch_pgm(keys, ROUNDING_GAP_EPSILON)
    result = index.search(keys[2])
    assert result.lo <= 2 < result.hi
    assert result.lo == max(0, result.pos - ROUNDING_GAP_EPSILON)
    assert result.hi == min(len(keys), result.pos + ROUNDING_GAP_EPSILON + 2)
    # every point of the fixture is inside the certificate's band (and the predictor's
    # error is pinned for the whole fixture)
    errors = [
        abs(segment.predict(keys[rank]) - rank)
        for rank in range(certificate.start_rank, certificate.end_rank + 1)
    ]
    assert max(errors) == 2
    for rank in range(certificate.start_rank, certificate.end_rank + 1):
        assert certificate.within_band(keys[rank], rank, ROUNDING_GAP_EPSILON)


# ---------------------------------------------------------------------------
# 9. legacy semantics unchanged
# ---------------------------------------------------------------------------


def test_9_legacy_search_fields_and_lookup_behaviour_are_unchanged():
    """Issue #23 §2/§8.9: the five search fields and the lookup contract are frozen."""
    # (a) literal pins of the accepted values on the rounding-gap fixture
    segment = segments_of(ROUNDING_GAP_KEYS, ROUNDING_GAP_EPSILON)[0]
    assert (
        segment.start_key, segment.start_rank, segment.slope_num, segment.slope_den,
        segment.intercept,
    ) == (RecordKey(0), 0, 3, 7, 0)
    assert [field for field in LEGACY_SEARCH_FIELDS if hasattr(segment, field)] == list(
        LEGACY_SEARCH_FIELDS
    )
    assert segment.start_key.value == 0 and segment.start_rank == 0

    # (b) the exact field surface of a segment: the accepted five plus the certificate
    assert set(vars(segment)) == set(LEGACY_SEARCH_FIELDS) | {"certificate"}
    assert set(vars(segment.certificate)) == set(CERTIFICATE_FIELDS)

    # (c) the lower_bound contract on deterministic and randomized fixtures
    fixtures = {
        "rounding_gap": (ROUNDING_GAP_KEYS, ROUNDING_GAP_EPSILON),
        "multi_segment": (SEGMENT_KEYS, SEGMENT_EPSILON),
        "mixed_sign": (EDGE_FIXTURES["mixed_sign"][0], 3),
    }
    rng = random.Random(11)
    for label, (values, epsilon) in fixtures.items():
        keys = key_tuple(values)
        index = build_batch_pgm(keys, epsilon)
        queries = list(values) + [-10 ** 7, 10 ** 7] + [
            rng.randint(-(10 ** 6), 10 ** 6) for _ in range(300)
        ]
        for query in queries:
            key = RecordKey(query)
            expected = 0
            while expected < len(keys) and keys[expected] < key:
                expected += 1
            result = index.search(key)
            assert 0 <= result.pos <= len(keys)
            assert result.lo <= expected <= result.hi, (label, query)
            assert result.lo == max(0, result.pos - epsilon)
            assert result.hi == min(len(keys), result.pos + epsilon + 2)
        # predict() sees exactly the accepted search model, never the certificate
        for segment in index.segments:
            inside = [
                keys[rank]
                for rank in range(segment.certificate.start_rank,
                                  segment.certificate.end_rank + 1)
            ]
            for key in inside:
                expected = segment.predict(key)
                assert isinstance(expected, int)
        # search metadata does not carry physical identifiers
        for segment in index.segments:
            for name in vars(segment):
                assert "slot" not in name and "block" not in name


# ---------------------------------------------------------------------------
# 10. batch vs incremental identity
# ---------------------------------------------------------------------------


def drive_incremental(source_items, target_items, schedule, *, items_per_block, epsilon):
    job = IncrementalMergeJob.begin(
        LogicalRunView.from_records(4, source_items),
        LogicalRunView.from_records(5, target_items),
        items_per_block=items_per_block,
        epsilon=epsilon,
    )
    blocks = []
    for q in schedule:
        step = job.advance(q)
        blocks.extend(step.blocks)
        if step.done:
            break
    while not job.done:
        blocks.extend(job.advance(1).blocks)
    return blocks, job.finalize(), job


def test_10_batch_and_incremental_certificates_are_identical_across_schedules():
    """Issue #23 §7/§8.10: identical segments AND certificates for every schedule."""
    source = tuple((2_000 + rank, f"s{rank}") for rank in range(12))
    target = tuple((2_006 + rank, f"t{rank}") for rank in range(12))
    merged = tuple(
        sorted(
            {
                key: value
                for key, value in (*target, *source)
            }.items(),
            key=lambda item: item[0],
        )
    )
    merged_keys = key_tuple(key for key, _ in merged)
    batch = build_batch_pgm(merged_keys, 8)
    schedules = {
        "q=1": [1] * 64,
        "q=3": [3] * 64,
        "q=items_per_block": [8] * 64,
        "q=huge": [10_000],
        "irregular": [7, 1, 3, 2, 11, 1, 4, 1, 9, 2, 1, 5, 3, 1, 1, 6] * 4,
    }
    signatures = {}
    for label, schedule in schedules.items():
        blocks, result, _ = drive_incremental(
            source, target, schedule, items_per_block=8, epsilon=8
        )
        streamed = tuple((key, value) for block in blocks for key, value in block.records())
        assert streamed == merged, label
        assert result.pgm.segments == batch.segments, label
        assert tuple(segment.certificate for segment in result.pgm.segments) == tuple(
            segment.certificate for segment in batch.segments
        ), label
        signatures[label] = (
            tuple(segment.certificate.to_dict() for segment in result.pgm.segments),
            tuple(
                (segment.start_key.value, segment.start_rank, segment.slope_num,
                 segment.slope_den, segment.intercept)
                for segment in result.pgm.segments
            ),
        )
    assert len(set(map(repr, signatures.values()))) == 1

    # prefix identity with the read-only job surface: at every step the live certificate
    # set equals the batch certificates of exactly the decided prefix
    job = IncrementalMergeJob.begin(
        LogicalRunView.from_records(4, source), LogicalRunView.from_records(5, target),
        items_per_block=8, epsilon=8,
    )
    while not job.done:
        step = job.advance(3)
        decided = step.state.output_records
        prefix = make_segmentation(merged_keys[:decided], 8)
        snapshot = job.pgm_state
        assert snapshot.segment_count == len(prefix)
        assert snapshot.segment_start_ranks == tuple(
            segment.certificate.start_rank for segment in prefix
        )


def test_10b_certificates_are_identical_for_a_segment_fixture_and_epsilon_zero():
    """Issue #23 §7: epsilon = 0 and multi-segment fixtures keep the identity too."""
    source = tuple((key, f"a{key}") for key in SEGMENT_KEYS[:15])
    target = tuple((key, f"b{key}") for key in SEGMENT_KEYS[15:])
    merged_keys = key_tuple(SEGMENT_KEYS)
    batch = build_batch_pgm(merged_keys, SEGMENT_EPSILON)
    assert len(batch.segments) == 3
    for schedule in ([1] * 64, [3] * 64, [SEGMENT_ITEMS_PER_BLOCK] * 64, [10] * 16):
        _, result, _ = drive_incremental(
            source, target, schedule,
            items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON,
        )
        assert result.pgm.segments == batch.segments
        assert result.pgm == batch


# ---------------------------------------------------------------------------
# 11. a certificate crosses both boundaries
# ---------------------------------------------------------------------------


def test_11_a_certificate_crosses_advance_and_block_boundaries_without_splitting():
    """Issue #23 §8.11: step and block endpoints are invisible to certificates."""
    source = tuple((key, f"a{key}") for key in SEGMENT_KEYS[:15])
    target = tuple((key, f"b{key}") for key in SEGMENT_KEYS[15:])
    job = IncrementalMergeJob.begin(
        LogicalRunView.from_records(4, source), LogicalRunView.from_records(5, target),
        items_per_block=SEGMENT_ITEMS_PER_BLOCK, epsilon=SEGMENT_EPSILON,
    )
    observed = []
    while not job.done:
        step = job.advance(3)                       # step boundaries at r = 3, 6, 9, ...
        snapshot = job.pgm_state
        observed.append(
            (
                step.state.output_records,
                snapshot.closed_segment_start_ranks,
                snapshot.active_segment_span,
                [block.rank for block in step.blocks],
            )
        )
    result = job.finalize()
    by_records = {entry[0]: entry for entry in observed}

    # canonical certificates: spans 0-9, 10-19, 20-29 over the whole 30-record stream
    assert [segment.certificate.rank_span for segment in result.pgm.segments] == [
        (0, 9), (10, 19), (20, 29)
    ]
    assert [segment.certificate.start_key.value for segment in result.pgm.segments] == [
        0, 100_000, 200_000
    ]
    assert [segment.certificate.end_key.value for segment in result.pgm.segments] == [
        9, 100_009, 200_009
    ]

    # the first certificate is still open across the step boundaries r = 3, 6, 9 and
    # across the output-block boundaries r = 4 and r = 8 (items_per_block = 4)
    for records in (3, 6, 9):
        assert by_records[records][1] == ()
        assert by_records[records][2][0] == 0
        assert by_records[records][2][1] == records - 1
    assert by_records[6][3] == [0]                  # block 0 completed inside the span
    assert by_records[9][3] == [1]                  # block 1 completed inside the span
    # the boundary ranks that fall strictly inside the first certificate's rank span
    boundaries_inside_certificate = [r for r in (3, 4, 6, 8, 9) if 0 < r <= 9]
    assert boundaries_inside_certificate and max(boundaries_inside_certificate) == 9

    # the second certificate starts exactly at the canonical rank 10, not at a step or
    # block boundary, and the closed set never gained a synthetic split
    assert by_records[12][1] == (0,)
    assert result.pgm_segment_start_ranks == (0, 10, 20)
    assert result.pgm_segment_start_ranks == tuple(
        segment.certificate.start_rank for segment in result.pgm.segments
    )


# ---------------------------------------------------------------------------
# 12. metadata growth
# ---------------------------------------------------------------------------


def test_12_certificates_are_one_per_segment_and_no_key_stream_is_retained():
    """Issue #23 §8.12: O(segments) metadata, exactly one end boundary per segment."""
    values = tuple(sorted(random.Random(3).sample(range(-5_000, 5_000), 900)))
    epsilon = 3
    segments = segments_of(values, epsilon)
    assert len(segments) > 1
    # exactly one certificate and one end boundary per segment
    certificates = [segment.certificate for segment in segments]
    assert len(certificates) == len(segments)
    assert len({id(certificate) for certificate in certificates}) == len(certificates)
    assert [segment.certificate.end_rank for segment in segments] == [
        segments[index + 1].certificate.start_rank - 1 for index in range(len(segments) - 1)
    ] + [len(values) - 1]
    # a certificate stores only scalars and two boundary keys: no key sequence
    for certificate in certificates:
        for name, value in vars(certificate).items():
            assert not isinstance(value, (list, tuple, dict, set)), name
    # metadata is O(segments), not O(records)
    assert len(certificates) < len(values) / 4

    # the incremental builder keeps no key stream and only O(1) working state
    builder = IncrementalPgmBuilder(epsilon)
    for value in values:
        builder.add_key(RecordKey(value))
    assert builder.retained_keys == 0
    assert not any(
        isinstance(value, (list, tuple)) and any(
            isinstance(item, RecordKey) for item in value
        )
        for value in vars(builder).values()
    )
    finalized = builder.finalize()
    assert len(finalized.segments) == len(segments)
    assert [segment.certificate for segment in finalized.segments] == certificates

    # the merged job exposes only the read-only snapshot and reports the same count
    source = tuple((value, f"s{value}") for value in values[:500])
    target = tuple((value, f"t{value}") for value in values[500:])
    _, result, job = drive_incremental(
        source, target, [1] * 64, items_per_block=16, epsilon=epsilon
    )
    assert job.pgm_state.segment_count == len(result.pgm.segments)
    assert job.working_state_report()["retained_output_blocks"] == 0
    assert job.working_state_report()["retained_record_history"] == 0
    assert job.working_state_report()["pgm_retained_keys"] == 0


# ---------------------------------------------------------------------------
# 13. no floats in the production path
# ---------------------------------------------------------------------------


def test_13_the_production_certificate_path_uses_exact_integers_only():
    """Issue #23 §2/§8.13: no float/Decimal in construction or evaluation."""
    for path in (PGM_PATH, INCREMENTAL_PATH):
        code = code_without_docstrings(path)
        for token in ("import math", "import decimal", "from decimal", "from fractions",
                      "float(", "Decimal(", "math."):
            assert token not in code, (path.name, token)
        assert "Fraction" not in code, path.name

    # the certificate's numbers are ints and its evaluation is exact integer arithmetic
    certificate = segments_of(ROUNDING_GAP_KEYS, ROUNDING_GAP_EPSILON)[0].certificate
    for name, value in certificate.to_dict().items():
        assert isinstance(value, int) and not isinstance(value, bool), name
    assert isinstance(certificate.numerator_at(RecordKey(9)), int)
    assert isinstance(certificate.within_band(RecordKey(2), 2, 1), bool)
    assert all(
        isinstance(value, int)
        for value in (
            certificate.slope_num, certificate.origin_num, certificate.denominator,
            certificate.start_rank, certificate.end_rank,
        )
    )
    # the certificate is a frozen value object: no post-construction mutation
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        certificate.denominator = 1
    with pytest.raises(PgmError):
        RankLineCertificate(
            RecordKey(0), RecordKey(1), 0, 0, 1, 1, 0,            # denominator 0
        )
    with pytest.raises(PgmError):
        RankLineCertificate(RecordKey(0), RecordKey(1), 2, 1, 1, 1, 1)

    # fail closed: a *non-singleton* canonical geometry with a zero x span is not a
    # canonical segment, so freezing a certificate for it must raise instead of silently
    # substituting a fake slope/denominator
    degenerate = _CanonicalSegmentGeometry(
        (_Point(5, 1), _Point(5, 0), _Point(5, 1), _Point(5, 2)), 5,
    )
    with pytest.raises(PgmError, match="non-zero x span"):
        _rank_line_certificate(degenerate, 0, 0, RecordKey(5))
    # ... while the one-point geometry of the same x is legal (denominator 2)
    singleton = _CanonicalSegmentGeometry(
        (_Point(5, 3), _Point(5, 1), _Point(5, 3), _Point(5, 1)), 5, one_point=True,
    )
    certificate_from_geometry = _rank_line_certificate(singleton, 0, 0, RecordKey(5))
    assert (certificate_from_geometry.slope_num, certificate_from_geometry.origin_num,
            certificate_from_geometry.denominator) == (0, 4, 2)


# ---------------------------------------------------------------------------
# 14. the newer-wins overlap guard
# ---------------------------------------------------------------------------


def test_14_newer_wins_overlap_counterexample_is_pinned_as_a_theory_guard():
    """Issue #23 §6: R_A + R_B is not the output guide when keys overlap.

    Rank convention: the project freezes the **strict lower bound**
    ``R_X(x) = |{k in X : k < x}|`` because the PGM training points ``(k_i, i)`` are
    aligned with the *lower_bound* rank of a key.  Both conventions — strict and
    inclusive (``k <= x``) — satisfy ``R_C(x) = R_A(x) + R_B(x) - D_AB(x)`` as long as
    each is applied consistently to all three prefix counts; they are different
    numerical conventions and must never be mixed in one statement.  The guard below is
    stated over the frozen strict convention.
    """
    source = ((2, "s2"), (4, "s4"), (6, "s6"))
    target = ((2, "t2"), (4, "t4"), (6, "t6"))
    blocks, result, _ = drive_incremental(
        source, target, [1] * 8, items_per_block=2, epsilon=0
    )
    streamed = tuple((key, value) for block in blocks for key, value in block.records())
    # C collapses the shared keys and keeps the newer values: three records, not six
    assert streamed == ((2, "s2"), (4, "s4"), (6, "s6"))
    assert result.record_count == 3
    assert result.duplicate_count == 3
    assert result.pgm.record_count == 3

    def prefix_rank(items, threshold):
        """``R_X(x) = |{k in X : k < x}|`` — strictly less than the threshold."""
        return sum(1 for key, _ in items if key < threshold)

    def duplicate_prefix_rank(threshold):
        """``D_AB(x)``: the shared keys below the threshold."""
        newer_keys = {key for key, _ in source}
        return sum(1 for key, _ in target if key < threshold and key in newer_keys)

    # the hand-worked thresholds of the review: x = 3, 5, 7 give R_C = 1, 2, 3
    observed_c = []
    for threshold in (3, 5, 7):
        rank_a = prefix_rank(source, threshold)
        rank_b = prefix_rank(target, threshold)
        duplicates = duplicate_prefix_rank(threshold)
        rank_c = prefix_rank(streamed, threshold)
        observed_c.append(rank_c)
        # R_C(x) = R_A(x) + R_B(x) - D_AB(x)
        assert rank_c == rank_a + rank_b - duplicates, threshold
        # ... and on this fully-overlapping fixture R_A + R_B = 2 * R_C
        assert rank_a + rank_b == 2 * rank_c, threshold
    assert observed_c == [1, 2, 3]

    # the two conventions give different numbers and must not be mixed: at a threshold
    # equal to a key the strict count is 1 while the inclusive count is 2
    assert prefix_rank(source, 4) == 1
    assert sum(1 for key, _ in source if key <= 4) == 2
    assert prefix_rank(streamed, 4) == 1
    assert prefix_rank(source, 4) + prefix_rank(target, 4) - duplicate_prefix_rank(4) == 1

    # ... and the same identity holds under the *inclusive* convention, applied
    # consistently to all three prefix counts (so neither convention is "the wrong one";
    # the project merely freezes the strict lower bound)
    def inclusive_prefix_rank(items, threshold):
        return sum(1 for key, _ in items if key <= threshold)

    def inclusive_duplicate_prefix_rank(threshold):
        newer_keys = {key for key, _ in source}
        return sum(1 for key, _ in target if key <= threshold and key in newer_keys)

    for threshold in (2, 3, 4, 5, 6, 7):
        rank_c = inclusive_prefix_rank(streamed, threshold)
        assert rank_c == (
            inclusive_prefix_rank(source, threshold)
            + inclusive_prefix_rank(target, threshold)
            - inclusive_duplicate_prefix_rank(threshold)
        ), threshold
    assert [inclusive_prefix_rank(streamed, x) for x in (3, 5, 7)] == [1, 2, 3]
    assert [prefix_rank(streamed, x) for x in (3, 5, 7)] == [1, 2, 3]

    # the sum of the two *input* guides is still not the output rank
    assert prefix_rank(source, 7) + prefix_rank(target, 7) != prefix_rank(streamed, 7)

    # ... so a joint guide built by summing the two input PGMs would be wrong on this
    # fixture; the certificates are per-input metadata and no joint guide, corridor or
    # connector exists in the executable production code
    for path in (PGM_PATH, INCREMENTAL_PATH):
        code = code_without_docstrings(path)
        for token in ("GuideBoundary", "guide", "corridor", "connector", "joint",
                      "envelope", "half_rank", "HalfRank", "RID", "rid_"):
            assert token not in code, (path.name, token)
    # no (key, RID) semantics: exactly one record per key, source wins, no second version
    assert [key for key, _ in streamed] == sorted({key for key, _ in (*source, *target)})
    assert len({value for _, value in streamed}) == 3


# ---------------------------------------------------------------------------
# 15. scope guard
# ---------------------------------------------------------------------------


def test_15_the_imported_certificate_modules_stay_scope_clean():
    """Issue #23 §9/§8.15, common-substrate part only: the certificate slice adds
    certificates and nothing physical.  (The original test also pinned the source
    repository's baseline tag, milestone state and the excluded defence vocabulary;
    those parts are dropped in this derived copy.)"""
    for path in (PGM_PATH, INCREMENTAL_PATH):
        code = code_without_docstrings(path)
        for token in ("SlotId", "UntrustedStorage", "allocator", "writeback",
                      "ORAM", "BORPStream", "CacheShuffle", "SWAT", "attack",
                      "leakage_metric", "structure_version", "publish_structural"):
            assert token not in code, (path.name, token)


