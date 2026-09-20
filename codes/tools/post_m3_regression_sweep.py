"""Post-M3 deterministic differential / property-style regression sweep (QA-1, Issue #8).

This is a **stabilization harness**.  It freezes no new semantics: it only re-derives the
already accepted M1/M2/M3 contracts from independently written oracles and invariants, and
checks the shipped implementations against them over many generated cases.

What it covers:

* **M1** — exact record-level newer-wins sorted union against a harness-local dict evaluator;
  duplicate handling (newer source wins, consumed once, emitted once); strictly increasing
  output keys; canonical output block packing; full ``FunctionalMergeOracleResult`` equality
  across several ``advance(q)`` schedules; canonical PGM equality against the frozen batch
  builder; agreement with the frozen incremental job; and no physical/observational surface.
* **M2** — allocation invariants over a generated parameter grid: determinism, sampled loads
  inside the pinned support, fixed bin capacity, an exact contiguous partition of the logical
  ranks with no duplication or omission, the noisy-prefix window and additive-error identity,
  and reproducible typed failures that are never silently repaired.
* **M3** — immutable snapshots under post-construction mutation of the caller's ``Block``
  objects, exact reconstruction of both runs from the bound bins, interior points that are real
  keys of their own bin, ``DUMMY_POS_INF`` on zero-real bins, value-only invariance,
  determinism, source/target stream-domain separation, signed-tag tie order, every bin read
  exactly once, no out-of-range read, and zero SlotId/storage/trace surface.
* **Pinned ``DOMerge`` projection** — a harness-local simulator of the pinned preloads plus the
  per-tag ``if`` / ``else if`` fetch loop (``j0``/``j1``), written without importing or calling
  the M3 schedule builder, compared for exact equality against every two-sided M3 schedule.
  Cases that genuinely trigger the fallback branch are counted.

Nothing here belongs to the production package: the independent evaluators and the pinned
simulator live under ``codes/tools``, and :func:`check_static_separation` proves that no
``swat_m_block`` module references them.

Usage::

    python tools/post_m3_regression_sweep.py --quick
    python tools/post_m3_regression_sweep.py --full
    python tools/post_m3_regression_sweep.py --quick --seed 7

``--seed`` fixes the master seed; a fixed seed is deterministic in-process *and* across
processes (all generation goes through one ``random.Random(master_seed)``).  Elapsed time is
printed as operational information only — this harness makes no performance claim.
"""
from __future__ import annotations

import argparse
import ast
import math
import random
import sys
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "codes" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enhanced_letindex.block import Block                            # noqa: E402
from enhanced_letindex.identifiers import BlockId, RecordKey         # noqa: E402
from enhanced_letindex.incremental_merge import (                    # noqa: E402
    IncrementalMergeJob,
    LogicalRunView,
)
from enhanced_letindex.pgm import build_batch_pgm                    # noqa: E402
from enhanced_letindex.record import Record                         # noqa: E402

from swat_m_block import (                                          # noqa: E402
    DUMMY_POS_INF,
    SOURCE,
    STREAM_DOMAIN_INTERIOR_SOURCE,
    STREAM_DOMAIN_INTERIOR_TARGET,
    TARGET,
    BlockAllocationError,
    BlockAllocationPlan,
    GeometricLoadSampler,
    InsufficientSampledCapacity,
    LogicalBlockRunView,
    SwatBlockAllocationConfig,
    SwatBlockMergeSchedule,
    allocate_block_bins,
    bind_block_allocation,
    collect_output_blocks,
    compute_bin_capacity_blocks,
    compute_bin_count,
    compute_factor,
    derive_stream_seed,
    flatten_blocks,
    functional_merge_oracle,
    interior_point_sort_key,
    interior_stream_domain,
    output_key_stream,
    plan_swat_block_merge_schedule,
    sample_bin_interior_points,
    sorted_tagged_interior_points,
    swat_reference_config,
)
from swat_m_block.content_schedule import LogicalBlockSnapshot      # noqa: E402

__all__ = [
    "DEFAULT_MASTER_SEED",
    "FULL_BUDGET",
    "QUICK_BUDGET",
    "SweepBudget",
    "SweepFailure",
    "SweepReport",
    "check_m1_case",
    "check_m2_case",
    "check_m3_case",
    "check_static_separation",
    "generate_m1_cases",
    "generate_m2_cases",
    "generate_m3_cases",
    "independent_expected_block_sizes",
    "independent_newer_wins",
    "physical_io_guard",
    "pinned_projected_fetches",
    "run_sweep",
    "schedule_structure",
    "summary_lines",
]

DEFAULT_MASTER_SEED = 20260920
SWEEP_VERSION = "qa-1"

SOURCE_LEVEL = 4
TARGET_LEVEL = 5

#: How many keys a failure report shows before summarising the remainder.
CONTEXT_KEYS_SHOWN = 8

#: Identifiers that must never appear in a production module's executable code.
FORBIDDEN_MODULE_TOKENS = (
    "SlotId", "UntrustedStorage", "TraceEvent", "TraceCollector", "TraceOperation",
    "physical_slot", "read_slot", "write_slot",
)

#: The modules whose accepted semantics this sweep verifies.
PRODUCTION_MODULES = (
    "functional_oracle.py", "distribution.py", "bin_allocator.py", "content_schedule.py",
)

#: The only modules the production package may import: the standard library it already
#: uses, the frozen common substrate, and its own siblings.  An allowlist is used instead of
#: a denylist, so an excluded defence module, a physical module or a future rename of either
#: can never be imported into ``swat_m_block`` without failing this check.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__", "bisect", "dataclasses", "hashlib", "math", "random", "typing",
        "enhanced_letindex.block", "enhanced_letindex.identifiers",
        "enhanced_letindex.incremental_merge", "enhanced_letindex.pgm",
        "enhanced_letindex.record",
        "functional_oracle", "distribution", "bin_allocator", "content_schedule",
    }
)


# ---------------------------------------------------------------------------
# failure reporting
# ---------------------------------------------------------------------------


class SweepFailure(Exception):
    """One generated case violated one accepted invariant.

    Carries what is needed to reproduce the case and nothing more: the master seed, the case
    index, the subsystem, a compact description of the inputs, and the violated invariant.
    Whole objects are never included.
    """

    def __init__(self, *, subsystem: str, case_index: int, invariant: str, detail: str,
                 context: str) -> None:
        self.subsystem = subsystem
        self.case_index = case_index
        self.invariant = invariant
        self.detail = detail
        self.context = context
        super().__init__(f"[{subsystem}] case {case_index}: {invariant} :: {detail}")

    def report(self, master_seed: int) -> str:
        reproduce = (
            "python tools/post_m3_regression_sweep.py "
            f"--quick --seed {master_seed}"
        )
        return "\n".join(
            [
                "FAILURE",
                f"  master seed : {master_seed}",
                f"  case index  : {self.case_index}",
                f"  subsystem   : {self.subsystem}",
                f"  invariant   : {self.invariant}",
                f"  detail      : {self.detail}",
                f"  inputs      : {self.context}",
                f"  reproduce   : {reproduce}",
            ]
        )


def _fail(subsystem: str, case_index: int, invariant: str, detail: str,
          context: str) -> SweepFailure:
    return SweepFailure(subsystem=subsystem, case_index=case_index, invariant=invariant,
                        detail=detail, context=context)


def _compact_keys(keys: Sequence[int], label: str) -> str:
    values = list(keys)
    if len(values) <= CONTEXT_KEYS_SHOWN:
        return f"{label}={values}"
    head = values[:CONTEXT_KEYS_SHOWN]
    return f"{label}={head}...+{len(values) - CONTEXT_KEYS_SHOWN} more (n={len(values)})"


# ---------------------------------------------------------------------------
# independent evaluators  (harness-local; never imported by production code)
# ---------------------------------------------------------------------------


def independent_newer_wins(
    source_items: Sequence[Tuple[int, Any]],
    target_items: Sequence[Tuple[int, Any]],
) -> Tuple[Tuple[int, Any], ...]:
    """The expected exact newer-wins sorted union, written from scratch.

    ``source`` is the newer level, so its value wins every key it shares with ``target``.
    """
    newer = {key: value for key, value in source_items}
    older = {key: value for key, value in target_items}
    return tuple(
        (key, newer[key] if key in newer else older[key])
        for key in sorted(set(newer) | set(older))
    )


def independent_expected_block_sizes(record_count: int, items_per_block: int) -> Tuple[int, ...]:
    """Canonical packing: full blocks first, then exactly one final partial block."""
    if record_count == 0:
        return ()
    full, remainder = divmod(record_count, items_per_block)
    sizes = [items_per_block] * full
    if remainder:
        sizes.append(remainder)
    return tuple(sizes)


def pinned_projected_fetches(
    source_bins: int,
    target_bins: int,
    tagged: Sequence[Tuple[str, Any]],
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]:
    """The pinned ``DOMerge`` fetch loop, projected onto actual bin fetches.

    An independent mirror of the accepted pinned skeleton — it never calls the M3 schedule
    builder:

    1. preload bin 0 of each non-empty side;
    2. run ``source_bins + target_bins`` iterations over the sorted tagged interior points;
    3. on a left (source) tag, if the source still has an unfetched bin, fetch it (``j0``);
    4. otherwise (``else if``), if the target still has an unfetched bin, fetch it (``j1``).

    Step 4 is the real fallback: a left tag arriving after the source is exhausted fetches from
    the *target* side.  Iterations that fetch nothing are not emitted, and the pinned loop's
    deferred safe-output/frontier work is not modelled at all.

    Returns ``(projected_fetches, fallback_fetches)``, where every fallback fetch was triggered
    by a source tag while the source was exhausted.
    """
    fetches: List[Tuple[str, int]] = []
    fallbacks: List[Tuple[str, int]] = []
    if source_bins > 0:
        fetches.append((SOURCE, 0))
    if target_bins > 0:
        fetches.append((TARGET, 0))
    left_next = -1
    right_next = -1
    for side, _order in tagged:
        is_left_tag = side == SOURCE
        if is_left_tag and left_next + 2 < source_bins:
            left_next += 1
            fetches.append((SOURCE, left_next + 1))
        elif right_next + 2 < target_bins:
            right_next += 1
            fetches.append((TARGET, right_next + 1))
            if is_left_tag:
                fallbacks.append((SOURCE, right_next + 1))
    return tuple(fetches), tuple(fallbacks)


def schedule_structure(schedule: SwatBlockMergeSchedule) -> Tuple[Any, ...]:
    """The value-independent structure of a schedule.

    Record *values* never influence M3, so a structural comparison must not either: this
    returns the reads, the tagged (side, bin, sign, interior key order) stream and the interior
    points, with no record payload.
    """
    return (
        tuple((read.side, read.bin_index) for read in schedule.reads),
        tuple(
            (point.side, point.bin_index, point.signed_tag,
             interior_point_sort_key(point.interior_point))
            for point in schedule.tagged_interior_points
        ),
        tuple((point.key, point.record_index) for point in schedule.source.interiors),
        tuple((point.key, point.record_index) for point in schedule.target.interiors),
    )


# ---------------------------------------------------------------------------
# physical/observational guard  (harness-local)
# ---------------------------------------------------------------------------


class physical_io_guard:
    """Turn every physical and observational entry point into a hard failure.

    Proves operationally that a swept case performs no storage operation and emits no trace
    event.  Written without pytest so the sweep tool can use it directly.
    """

    def __init__(self) -> None:
        self.violations: List[str] = []
        self._patched: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "physical_io_guard":
        from enhanced_letindex.storage import UntrustedStorage
        from enhanced_letindex.trace import TraceCollector

        for owner, name in ((UntrustedStorage, "allocate"), (UntrustedStorage, "store"),
                            (UntrustedStorage, "read"), (UntrustedStorage, "write"),
                            (UntrustedStorage, "clear"), (TraceCollector, "record")):
            original = getattr(owner, name, None)
            if original is None:
                continue
            self._patched.append((owner, name, original))
            setattr(owner, name, self._explode(f"{owner.__name__}.{name}"))
        return self

    def _explode(self, name: str):
        def recorder(*args, **kwargs):
            self.violations.append(name)
            raise AssertionError(f"a swept case performed a physical {name}")

        return recorder

    def __exit__(self, *exc_info) -> None:
        for owner, name, original in self._patched:
            setattr(owner, name, original)
        self._patched.clear()
        return None


# ---------------------------------------------------------------------------
# budgets and cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepBudget:
    """How much of each subsystem one mode generates."""

    mode: str
    m1_cases: int
    m2_cases: int
    m3_cases: int
    max_records: int
    instrument_every: int


QUICK_BUDGET = SweepBudget(
    mode="quick", m1_cases=48, m2_cases=150, m3_cases=60, max_records=64,
    instrument_every=4,
)
FULL_BUDGET = SweepBudget(
    mode="full", m1_cases=640, m2_cases=2400, m3_cases=2000, max_records=220,
    instrument_every=16,
)
BUDGETS = {"quick": QUICK_BUDGET, "full": FULL_BUDGET}


@dataclass(frozen=True)
class M1Case:
    index: int
    name: str
    source_items: Tuple[Tuple[int, str], ...]
    target_items: Tuple[Tuple[int, str], ...]
    items_per_block: int
    epsilon: int
    source_level: int
    target_level: int
    schedules: Tuple[Optional[Tuple[int, ...]], ...]
    instrument: bool

    def context(self) -> str:
        return (
            f"family={self.name} "
            + _compact_keys([key for key, _ in self.source_items], "source_keys")
            + " "
            + _compact_keys([key for key, _ in self.target_items], "target_keys")
            + f" items_per_block={self.items_per_block} pgm_epsilon={self.epsilon}"
            + f" levels=({self.source_level},{self.target_level})"
        )


@dataclass(frozen=True)
class M2Case:
    index: int
    name: str
    block_count: int
    config: SwatBlockAllocationConfig
    fixed_load: Optional[int] = None

    def context(self) -> str:
        config = self.config
        return (
            f"family={self.name} block_count={self.block_count} configuration=( "
            f"security_lambda={config.security_lambda}, "
            f"privacy_epsilon={config.privacy_epsilon}, "
            f"privacy_delta={config.privacy_delta}, seed={config.seed}) "
            f"injected_load={self.fixed_load}"
        )


@dataclass(frozen=True)
class M3Case:
    index: int
    name: str
    source_keys: Tuple[int, ...]
    target_keys: Tuple[int, ...]
    items_per_block: int
    source_config: SwatBlockAllocationConfig
    target_config: SwatBlockAllocationConfig
    instrument: bool

    def context(self) -> str:
        return (
            f"family={self.name} "
            + _compact_keys(self.source_keys, "source_keys")
            + " "
            + _compact_keys(self.target_keys, "target_keys")
            + f" items_per_block={self.items_per_block} "
            + f"source_config=(security_lambda={self.source_config.security_lambda}, "
            + f"privacy_epsilon={self.source_config.privacy_epsilon}, "
            + f"privacy_delta={self.source_config.privacy_delta}, "
            + f"seed={self.source_config.seed}) "
            + f"target_config=(security_lambda={self.target_config.security_lambda}, "
            + f"privacy_epsilon={self.target_config.privacy_epsilon}, "
            + f"privacy_delta={self.target_config.privacy_delta}, "
            + f"seed={self.target_config.seed})"
        )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

M1_FAMILIES = (
    "source_only", "target_only", "both_empty", "disjoint_before", "disjoint_after",
    "interleaved", "duplicate_heavy", "identical_keys", "mixed_sign", "sparse_wide",
    "single_record", "many_records",
)
M2_FAMILIES = ("grid", "injected_zero_load", "injected_small_load", "out_of_support_load")
M3_FAMILIES = (
    "two_sided_interleaved", "two_sided_disjoint", "two_sided_duplicate_band",
    "source_only", "target_only", "both_empty", "single_block", "partial_final_block",
    "wide_sparse",
)

_BLOCK_CAPACITIES = (1, 2, 3, 4, 8, 16)
_PGM_EPSILONS = (0, 1, 2, 4, 8, 16, 64)
_LEVEL_PAIRS = ((0, 1), (1, 2), (4, 5), (17, 18), (999, 1000))
_SCHEDULES: Tuple[Optional[Tuple[int, ...]], ...] = (
    None, (1,), (2,), (7,), (1, 1, 1), (5, 3, 2, 9), (1000,),
)

_M2_LAMBDAS = (3, 4, 16, 512, 4096)
_M2_EPSILONS = (0.25, 0.5, 1.0, 2.0, 5.0)
_M2_DELTAS = (1e-12, 1e-6, 0.1, 0.5)
_M2_SEEDS = tuple(range(64))
_M2_BLOCK_COUNTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 40, 63, 64,
                    100, 127, 128, 255, 256, 512, 1000, 4096)

_M3_CAPACITIES = (1, 2, 3, 4, 8)


def _sorted_unique(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(values)))


def _band_keys(rng: random.Random, count: int, low: int, high: int) -> Tuple[int, ...]:
    span = high - low + 1
    if count <= 0:
        return ()
    if count >= span:
        return tuple(range(low, high + 1))
    return _sorted_unique(rng.sample(range(low, high + 1), count))


def _m1_items(rng: random.Random, family: str, max_records: int):
    small = max(1, max_records // 8)
    if family == "source_only":
        keys = _band_keys(rng, rng.randint(1, small), 10, 40 * small + 10)
        return tuple((key, f"s{key}") for key in keys), ()
    if family == "target_only":
        keys = _band_keys(rng, rng.randint(1, small), 10, 40 * small + 10)
        return (), tuple((key, f"t{key}") for key in keys)
    if family == "both_empty":
        return (), ()
    if family == "disjoint_before":
        source = _band_keys(rng, rng.randint(1, small), 0, 5 * small)
        target = _band_keys(rng, rng.randint(1, small), 5 * small + 1, 10 * small + 1)
        return (tuple((key, f"s{key}") for key in source),
                tuple((key, f"t{key}") for key in target))
    if family == "disjoint_after":
        source = _band_keys(rng, rng.randint(1, small), 5 * small + 1, 10 * small + 1)
        target = _band_keys(rng, rng.randint(1, small), 0, 5 * small)
        return (tuple((key, f"s{key}") for key in source),
                tuple((key, f"t{key}") for key in target))
    if family == "interleaved":
        keys = _band_keys(rng, rng.randint(2, 2 * small), 1, 4 * small)
        return (tuple((key, f"s{key}") for key in keys[0::2]),
                tuple((key, f"t{key}") for key in keys[1::2]))
    if family == "duplicate_heavy":
        shared = _band_keys(rng, rng.randint(1, small), 1, 3 * small)
        extra = _band_keys(rng, rng.randint(0, 2), 3 * small + 1, 5 * small + 1)
        return (tuple((key, f"s{key}") for key in shared),
                tuple((key, f"t{key}") for key in _sorted_unique(tuple(shared) + tuple(extra))))
    if family == "identical_keys":
        keys = _band_keys(rng, rng.randint(1, small), 1, 3 * small)
        return (tuple((key, f"s{key}") for key in keys),
                tuple((key, f"t{key}") for key in keys))
    if family == "mixed_sign":
        keys = _band_keys(rng, rng.randint(1, small), -3 * small, 3 * small)
        split = rng.randint(0, len(keys))
        return (tuple((key, f"s{key}") for key in keys[:split]),
                tuple((key, f"t{key}") for key in keys[split:]))
    if family == "sparse_wide":
        return (tuple((key, f"s{key}") for key in
                      _band_keys(rng, rng.randint(1, small), -10 * small, 10 * small)),
                tuple((key, f"t{key}") for key in
                      _band_keys(rng, rng.randint(1, small), -10 * small, 10 * small)))
    if family == "single_record":
        return (tuple((key, f"s{key}") for key in _band_keys(rng, 1, -5, 5)),
                tuple((key, f"t{key}") for key in _band_keys(rng, 1, -5, 5)))
    if family == "many_records":
        return (tuple((key, f"s{key}") for key in
                      _band_keys(rng, rng.randint(max_records // 2, max_records), 0,
                                 4 * max_records)),
                tuple((key, f"t{key}") for key in
                      _band_keys(rng, rng.randint(max_records // 2, max_records), 0,
                                 4 * max_records)))
    raise AssertionError(f"unknown M1 family {family!r}")


def generate_m1_cases(rng: random.Random, budget: SweepBudget) -> List[M1Case]:
    """Deterministically generate ``budget.m1_cases`` adjacent-run merge cases."""
    cases: List[M1Case] = []
    offset = rng.randrange(len(M1_FAMILIES))
    for index in range(budget.m1_cases):
        family = M1_FAMILIES[(index + offset) % len(M1_FAMILIES)]
        source_items, target_items = _m1_items(rng, family, budget.max_records)
        level_pair = _LEVEL_PAIRS[rng.randrange(len(_LEVEL_PAIRS))]
        total_records = len(source_items) + len(target_items)
        schedules = _SCHEDULES
        if total_records > budget.max_records // 2:
            schedules = (None, (3,), (1000,))
        elif rng.random() < 0.3:
            schedules = _SCHEDULES + ((rng.randint(1, 3), rng.randint(1, 5), 1),)
        cases.append(
            M1Case(
                index=index, name=family, source_items=source_items,
                target_items=target_items,
                items_per_block=_BLOCK_CAPACITIES[rng.randrange(len(_BLOCK_CAPACITIES))],
                epsilon=_PGM_EPSILONS[rng.randrange(len(_PGM_EPSILONS))],
                source_level=level_pair[0], target_level=level_pair[1],
                schedules=schedules, instrument=(index % budget.instrument_every == 0),
            )
        )
    return cases


def generate_m2_cases(rng: random.Random, budget: SweepBudget) -> List[M2Case]:
    """Deterministically generate ``budget.m2_cases`` allocation cases.

    ``injected_zero_load`` / ``injected_small_load`` drive the documented ``load_sampler``
    seam so an explicit ``InsufficientSampledCapacity`` is certain and countable rather than
    probabilistic; ``out_of_support_load`` injects a load outside ``[0, Z]``, which must be
    refused as a different typed error.
    """
    cases: List[M2Case] = []
    offset = rng.randrange(len(M2_FAMILIES))
    for index in range(budget.m2_cases):
        family = M2_FAMILIES[(index + offset) % len(M2_FAMILIES)]
        config = SwatBlockAllocationConfig(
            security_lambda=_M2_LAMBDAS[rng.randrange(len(_M2_LAMBDAS))],
            privacy_epsilon=_M2_EPSILONS[rng.randrange(len(_M2_EPSILONS))],
            privacy_delta=_M2_DELTAS[rng.randrange(len(_M2_DELTAS))],
            seed=_M2_SEEDS[rng.randrange(len(_M2_SEEDS))],
        )
        if family == "grid":
            blocks = _M2_BLOCK_COUNTS[rng.randrange(len(_M2_BLOCK_COUNTS))]
            fixed: Optional[int] = None
        elif family == "injected_zero_load":
            blocks = rng.randint(1, 200)
            fixed = 0
        elif family == "injected_small_load":
            blocks = rng.randint(1, 2000)
            fixed = (1, 2, 3, 6, 8)[rng.randrange(5)]
        else:
            # the support depends on the configuration, so derive the violating load from
            # this case's own bin capacity Z
            blocks = rng.randint(1, 50)
            capacity = _bin_capacity_of(config)
            fixed = (capacity + 1 + rng.randrange(4), -1 - rng.randrange(3))[rng.randrange(2)]
        cases.append(M2Case(index=index, name=family, block_count=blocks, config=config,
                            fixed_load=fixed))
    return cases


def _m3_keys(rng: random.Random, family: str, max_records: int):
    small = max(1, max_records // 10)
    if family == "two_sided_interleaved":
        keys = _band_keys(rng, 2 * small, 1, 8 * small)
        return keys[0::2], keys[1::2]
    if family == "two_sided_disjoint":
        return (_band_keys(rng, small, 0, 4 * small),
                _band_keys(rng, small, 4 * small + 1, 8 * small + 1))
    if family == "two_sided_duplicate_band":
        band = _band_keys(rng, small, 1, 4 * small)
        extra = _band_keys(rng, min(3, small // 2), 4 * small + 1, 6 * small + 1)
        return band, _sorted_unique(tuple(band) + tuple(extra))
    if family == "source_only":
        return _band_keys(rng, rng.randint(1, 2 * small), 1, 8 * small), ()
    if family == "target_only":
        return (), _band_keys(rng, rng.randint(1, 2 * small), 1, 8 * small)
    if family == "both_empty":
        return (), ()
    if family == "single_block":
        return _band_keys(rng, 1, 1, 50), _band_keys(rng, 1, 1, 50)
    if family == "partial_final_block":
        return (_band_keys(rng, 3 * small + rng.randint(1, 3), 1, 6 * small),
                _band_keys(rng, 3 * small + rng.randint(1, 3), 6 * small + 1, 12 * small + 1))
    if family == "wide_sparse":
        return (_band_keys(rng, rng.randint(1, small), -20 * small, 20 * small),
                _band_keys(rng, rng.randint(1, small), -20 * small, 20 * small))
    raise AssertionError(f"unknown M3 family {family!r}")


def _random_config(rng: random.Random) -> SwatBlockAllocationConfig:
    return SwatBlockAllocationConfig(
        security_lambda=_M2_LAMBDAS[rng.randrange(len(_M2_LAMBDAS))],
        privacy_epsilon=_M2_EPSILONS[rng.randrange(len(_M2_EPSILONS))],
        privacy_delta=_M2_DELTAS[rng.randrange(len(_M2_DELTAS))],
        seed=_M2_SEEDS[rng.randrange(len(_M2_SEEDS))],
    )


def generate_m3_cases(rng: random.Random, budget: SweepBudget) -> List[M3Case]:
    """Deterministically generate ``budget.m3_cases`` content-schedule cases."""
    cases: List[M3Case] = []
    offset = rng.randrange(len(M3_FAMILIES))
    for index in range(budget.m3_cases):
        family = M3_FAMILIES[(index + offset) % len(M3_FAMILIES)]
        source_keys, target_keys = _m3_keys(rng, family, budget.max_records)
        source_config = (swat_reference_config(rng.randrange(64)) if rng.random() < 0.6
                         else _random_config(rng))
        target_config = (swat_reference_config(rng.randrange(64)) if rng.random() < 0.6
                         else _random_config(rng))
        cases.append(
            M3Case(
                index=index, name=family, source_keys=source_keys, target_keys=target_keys,
                items_per_block=_M3_CAPACITIES[rng.randrange(len(_M3_CAPACITIES))],
                source_config=source_config, target_config=target_config,
                instrument=(index % budget.instrument_every == 0),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------


def _as_view(items: Sequence[Tuple[int, Any]], level: int) -> LogicalRunView:
    return LogicalRunView.from_records(level, [(key, value) for key, value in items])


def _build_blocks(keys: Sequence[int], items_per_block: int, id_base: int,
                  value_of) -> List[Block]:
    blocks: List[Block] = []
    for start in range(0, len(keys), items_per_block):
        records = [Record(RecordKey(key), value_of(key))
                   for key in keys[start:start + items_per_block]]
        blocks.append(Block.sorted_block(BlockId(id_base + start // items_per_block),
                                         records, items_per_block))
    return blocks


def _source_value(key: int) -> str:
    return f"s{key}"


def _target_value(key: int) -> str:
    return f"t{key}"


# ---------------------------------------------------------------------------
# M1 checker
# ---------------------------------------------------------------------------


def check_m1_case(case: M1Case) -> None:
    """Check every accepted M1 contract for one generated case."""

    def fail(invariant: str, detail: str) -> None:
        raise _fail("M1", case.index, invariant, detail, case.context())

    source = _as_view(case.source_items, case.source_level)
    target = _as_view(case.target_items, case.target_level)
    expected = independent_newer_wins(case.source_items, case.target_items)
    expected_keys = tuple(RecordKey(key) for key, _ in expected)

    def merge(schedule):
        return functional_merge_oracle(
            source, target, items_per_block=case.items_per_block,
            epsilon=case.epsilon, schedule=schedule,
        )

    # instrumentation: a complete merge must not touch storage or the trace
    if case.instrument:
        with physical_io_guard() as guard:
            result = merge(None)
        if guard.violations:
            fail("zero physical/observational surface", f"calls={guard.violations}")
    else:
        result = merge(None)

    # 1. exact record-level newer-wins sorted union (keys *and* values)
    if result.records() != expected:
        fail("exact newer-wins sorted union",
             f"got {len(result.records())} records, expected {len(expected)}")

    # 2. duplicates: the newer source wins, both inputs consumed, one emission
    duplicates = sorted(set(key for key, _ in case.source_items)
                        & set(key for key, _ in case.target_items))
    if result.duplicate_count != len(duplicates):
        fail("duplicate accounting",
             f"duplicate_count={result.duplicate_count} expected={len(duplicates)}")
    source_values = dict(case.source_items)
    for key, value in result.records():
        if key in source_values and source_values[key] != value:
            fail("newer source wins a duplicate key",
                 f"key {key} carries {value!r}, expected {source_values[key]!r}")
    emitted = tuple(output_key_stream(result.blocks))
    if len(set(emitted)) != len(emitted):
        fail("each key emitted exactly once",
             f"{len(emitted)} emissions, {len(set(emitted))} distinct")
    if emitted != expected_keys:
        fail("the output key stream is the exact merged key stream",
             f"{len(emitted)} keys, expected {len(expected_keys)}")

    # 3. output keys strictly increasing
    if list(emitted) != sorted(emitted):
        fail("output keys strictly increasing", f"{len(emitted)} keys")

    # 4. canonical block packing
    sizes = tuple(block.item_count for block in result.blocks)
    expected_sizes = independent_expected_block_sizes(len(expected), case.items_per_block)
    if sizes != expected_sizes:
        fail("canonical output block packing",
             f"sizes={sizes} expected={expected_sizes}")
    if result.block_count != len(expected_sizes) or result.record_count != len(expected):
        fail("result counters",
             f"block_count={result.block_count} record_count={result.record_count}")
    for position, block in enumerate(result.blocks):
        if block.rank != position:
            fail("canonical block ranks", f"block {position} carries rank {block.rank}")
        last = position == len(result.blocks) - 1
        if not last and block.item_count != case.items_per_block:
            fail("every non-final block is full", f"block {position}")
        if not 1 <= block.item_count <= case.items_per_block:
            fail("block sizes inside [1, items_per_block]", f"block {position}")

    # 5. schedule invariance: the whole frozen result must be identical
    for schedule in case.schedules:
        if merge(schedule) != result:
            fail("advance(q) schedule invariance", f"schedule={schedule} changed the result")

    # 6. canonical PGM over the exact output key stream
    if result.pgm != build_batch_pgm(expected_keys, case.epsilon):
        fail("canonical PGM equality",
             f"pgm_epsilon={case.epsilon} record_count={result.record_count}")

    # 7. the frozen incremental job agrees with the one-shot oracle
    job = IncrementalMergeJob.begin(source, target, items_per_block=case.items_per_block,
                                    epsilon=case.epsilon)
    budget = case.schedules[1] if len(case.schedules) > 1 else None
    blocks, _applied = collect_output_blocks(job, budget)
    if flatten_blocks(blocks) != expected:
        fail("frozen incremental job agreement",
             f"flattened {len(blocks)} blocks, expected {len(expected)} records")

    if expected:
        if result.first_key != expected_keys[0] or result.last_key != expected_keys[-1]:
            fail("first/last key", f"first={result.first_key} last={result.last_key}")
    elif result.first_key is not None or result.last_key is not None:
        fail("empty merge has no keys", f"first={result.first_key} last={result.last_key}")


# ---------------------------------------------------------------------------
# M2 checker
# ---------------------------------------------------------------------------


class _ConstantLoadSampler:
    """A counting constant load sampler, injected through the documented testing seam."""

    __slots__ = ("value", "calls")

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def sample(self) -> int:
        self.calls += 1
        return self.value


def _bin_capacity_of(config: SwatBlockAllocationConfig) -> int:
    return compute_bin_capacity_blocks(
        security_lambda=config.security_lambda,
        privacy_epsilon=config.privacy_epsilon,
        privacy_delta=config.privacy_delta,
        bucket_capacity_blocks=1,
    )


def check_m2_case(case: M2Case) -> str:
    """Check every accepted M2 contract for one generated case; returns ``plan``/``failure``."""

    def fail(invariant: str, detail: str) -> None:
        raise _fail("M2", case.index, invariant, detail, case.context())

    config = case.config
    capacity = _bin_capacity_of(config)
    expected_bin_count = compute_bin_count(compute_factor(config.security_lambda, capacity),
                                           case.block_count)

    # ---- facts that must hold whatever the outcome -------------------------------
    if capacity < 1 or capacity % 2:
        fail("bin capacity is an even positive integer", f"capacity={capacity}")

    # ---- injected samplers ------------------------------------------------------
    if case.fixed_load is not None:
        sampler = _ConstantLoadSampler(case.fixed_load)
        try:
            plan = allocate_block_bins(case.block_count, config, load_sampler=sampler)
        except InsufficientSampledCapacity as failure:
            if failure.bin_count != expected_bin_count:
                fail("failure bin count", f"{failure.bin_count} != {expected_bin_count}")
            if sampler.calls != failure.bin_count:
                fail("no resample-until-success",
                     f"{sampler.calls} samples drawn for {failure.bin_count} bins")
            if failure.sampled_capacity != max(case.fixed_load, 0) * expected_bin_count:
                fail("failure sampled capacity",
                     f"{failure.sampled_capacity} != "
                     f"{max(case.fixed_load, 0) * expected_bin_count}")
            if failure.shortfall_blocks != case.block_count - failure.sampled_capacity:
                fail("failure shortfall", f"{failure.shortfall_blocks}")
            if failure.seed != config.seed or failure.block_count != case.block_count:
                fail("failure carries the plan evidence",
                     f"seed={failure.seed} blocks={failure.block_count}")
            if not isinstance(failure, BlockAllocationError):
                fail("failure is a typed allocation error", f"{type(failure).__name__}")
            for _ in range(2):
                repeat = _ConstantLoadSampler(case.fixed_load)
                try:
                    allocate_block_bins(case.block_count, config, load_sampler=repeat)
                except InsufficientSampledCapacity as again:
                    if (again.sampled_capacity, again.bin_count, again.shortfall_blocks) != (
                            failure.sampled_capacity, failure.bin_count,
                            failure.shortfall_blocks):
                        fail("the same configuration fails the same way",
                             "a repeat disagreed")
                else:
                    fail("a failure never yields a plan", f"n={case.block_count}")
            return "failure"
        except BlockAllocationError as error:
            # a negative load is refused as ">= 0", a too-large load as "outside the
            # permitted [0, Z] range"; both are explicit typed refusals, never a silent
            # repair and never an insufficient-capacity failure
            message = str(error)
            if not ("must be >= 0" in message or "outside the permitted" in message):
                fail("an out-of-support load is refused explicitly", f"message {message}")
            return "failure"
        else:
            if not 0 <= case.fixed_load <= capacity:
                fail("an out-of-support load is refused explicitly",
                     f"load={case.fixed_load} was accepted with Z={capacity}")
            if plan.sampled_loads != (case.fixed_load,) * plan.bin_count:
                fail("no load inflation", f"loads={plan.sampled_loads}")
            if sampler.calls != plan.bin_count:
                fail("exactly one sample per planned bin",
                     f"{sampler.calls} samples for {plan.bin_count} bins")
            if tuple(bin_.sampled_load for bin_ in plan.bins) != plan.sampled_loads:
                fail("bin loads equal the sampled loads", "the plan disagrees with itself")
            return _check_m2_plan(case, plan, expected_bin_count, capacity, fail,
                                  injected=case.fixed_load)

    # ---- natural path -----------------------------------------------------------
    try:
        plan = allocate_block_bins(case.block_count, config)
    except InsufficientSampledCapacity as failure:
        sampler = GeometricLoadSampler(failure.bin_capacity_blocks, config.privacy_epsilon,
                                       seed=config.seed)
        independent_loads = tuple(sampler.sample() for _ in range(failure.bin_count))
        if sum(independent_loads) != failure.sampled_capacity:
            fail("the failure reflects the drawn loads exactly",
                 f"failure={failure.sampled_capacity} independent={sum(independent_loads)}")
        if failure.sampled_capacity >= case.block_count:
            fail("the failure is genuinely insufficient",
                 f"capacity={failure.sampled_capacity} n={case.block_count}")
        if failure.bin_count != expected_bin_count:
            fail("failure bin count", f"{failure.bin_count} != {expected_bin_count}")
        try:
            second = allocate_block_bins(case.block_count, config)
        except InsufficientSampledCapacity as again:
            if (again.sampled_capacity, again.bin_count, again.shortfall_blocks) != (
                    failure.sampled_capacity, failure.bin_count, failure.shortfall_blocks):
                fail("the same configuration fails the same way", "a repeat disagreed")
        else:
            fail("a failure never yields a plan", f"n={case.block_count} second={second}")
        return "failure"

    return _check_m2_plan(case, plan, expected_bin_count, capacity, fail)


def _check_m2_plan(case: M2Case, plan: BlockAllocationPlan, expected_bin_count: int,
                   capacity: int, fail, *, injected: Optional[int] = None) -> str:
    config = case.config

    if not is_dataclass(plan):
        fail("the plan is a dataclass", f"{type(plan).__name__}")
    if plan.block_count != case.block_count:
        fail("plan block count", f"{plan.block_count} != {case.block_count}")
    if plan.bin_capacity_blocks != capacity:
        fail("plan bin capacity", f"{plan.bin_capacity_blocks} != {capacity}")
    if plan.factor != compute_factor(config.security_lambda, capacity):
        fail("plan factor", f"{plan.factor}")
    if plan.bin_count != expected_bin_count:
        fail("plan bin count", f"{plan.bin_count} != {expected_bin_count}")
    if plan.bin_count != math.ceil(plan.factor * plan.block_count):
        fail("bin count is the pinned ceiling",
             f"bin_count={plan.bin_count} factor={plan.factor} n={plan.block_count}")
    if len(plan.bins) != plan.bin_count:
        fail("no append bin: one bin per planned bin",
             f"{len(plan.bins)} bins for bin_count={plan.bin_count}")
    if len(plan.sampled_loads) != plan.bin_count:
        fail("one sampled load per planned bin", f"{len(plan.sampled_loads)}")

    for load in plan.sampled_loads:
        if isinstance(load, bool) or not isinstance(load, int):
            fail("sampled load is a plain int", f"{load!r}")
        if not 0 <= load <= capacity:
            fail("sampled load inside [0, Z]", f"load={load} capacity={capacity}")

    consumed = 0
    for position, bin_ in enumerate(plan.bins):
        if bin_.bin_index != position:
            fail("bin order", f"bin {position} carries index {bin_.bin_index}")
        if bin_.capacity != capacity:
            fail("every bin has the fixed capacity", f"bin {position}: {bin_.capacity}")
        if bin_.real_count + bin_.dummy_count != capacity:
            fail("real + dummy == capacity",
                 f"bin {position}: {bin_.real_count}+{bin_.dummy_count} != {capacity}")
        if bin_.logical_rank_start != consumed:
            fail("contiguous rank intervals",
                 f"bin {position} starts at {bin_.logical_rank_start}, expected {consumed}")
        if bin_.logical_rank_stop - bin_.logical_rank_start != bin_.real_count:
            fail("rank interval width == real count", f"bin {position}")
        if bin_.sampled_load != plan.sampled_loads[position]:
            fail("bin load equals the sampled load", f"bin {position}")
        if bin_.real_count > bin_.sampled_load:
            fail("real count never exceeds the sampled load", f"bin {position}")
        consumed = bin_.logical_rank_stop
    if consumed != case.block_count:
        fail("every rank covered exactly once", f"consumed={consumed} n={case.block_count}")
    if plan.covered_ranks() != tuple(range(case.block_count)):
        fail("exact contiguous partition of 0..n-1", "covered_ranks() disagrees")
    if sum(bin_.real_count for bin_ in plan.bins) != case.block_count:
        fail("real counts sum to the run", f"{sum(bin_.real_count for bin_ in plan.bins)}")
    if plan.dummy_count != plan.bin_count * capacity - case.block_count:
        fail("dummy total", f"{plan.dummy_count}")
    if plan.sampled_capacity != sum(plan.sampled_loads):
        fail("sampled capacity", f"{plan.sampled_capacity}")
    if plan.is_empty != (case.block_count == 0):
        fail("empty plan flag", f"is_empty={plan.is_empty} n={case.block_count}")
    if plan.planned_slots != plan.bin_count * capacity:
        fail("planned slots", f"{plan.planned_slots}")

    prefix = plan.noisy_prefix_sums
    if len(prefix) != plan.bin_count + 1:
        fail("prefix length is bin_count + 1", f"{len(prefix)}")
    if prefix[0] != 0:
        fail("prefix starts at zero", f"{prefix[0]}")
    if list(prefix) != sorted(prefix):
        fail("prefix monotone non-decreasing", f"{prefix}")
    true_prefix = 0
    worst = 0
    for position, load in enumerate(plan.sampled_loads, start=1):
        true_prefix += load
        deviation = abs(true_prefix - prefix[position])
        if deviation > capacity:
            fail("prefix inside the permitted window",
                 f"t={position} deviation={deviation} Z={capacity}")
        worst = max(worst, deviation)
    if plan.additive_error_blocks != worst:
        fail("additive error is the exact maximum deviation",
             f"{plan.additive_error_blocks} != {worst}")

    if injected is None:
        again = allocate_block_bins(case.block_count, config)
    else:
        again = allocate_block_bins(case.block_count, config,
                                    load_sampler=_ConstantLoadSampler(injected))
    if again != plan:
        fail("the same configuration and sampler reproduce the identical plan",
             "a second plan differed")

    for field in fields(BlockAllocationPlan):
        if any(token in field.name for token in ("slot", "cipher", "prp", "trace", "storage")):
            fail("no physical field on the plan", f"field {field.name}")
    return "plan"


# ---------------------------------------------------------------------------
# M3 checker
# ---------------------------------------------------------------------------


@dataclass
class M3Observations:
    outcome: str = "schedule"
    two_sided: bool = False
    pinned_compared: bool = False
    fallback_triggered: bool = False
    mutation_checked: bool = False
    instrumented: bool = False


def _mutate_blocks(blocks: Sequence[Block]) -> int:
    """Inject one record into every block that still has spare capacity.

    The injected key is the preceding block's maximum key, which would break the run-level
    cross-block ordering if a view still pointed at the caller's live block.  Returns how many
    blocks were mutated.
    """
    mutated = 0
    for position, block in enumerate(blocks):
        if block.is_full:
            continue
        if position > 0:
            key = blocks[position - 1].max_key.value
        else:
            key = block.max_key.value + 1
        block.add_record(Record(RecordKey(key), "INJECTED"))
        mutated += 1
    return mutated


def _plan_or_none(config: SwatBlockAllocationConfig, block_count: int):
    try:
        return allocate_block_bins(block_count, config)
    except InsufficientSampledCapacity:
        return None


def check_m3_case(case: M3Case) -> M3Observations:
    """Check every accepted M3 contract for one generated case."""

    observations = M3Observations()

    def fail(invariant: str, detail: str) -> None:
        raise _fail("M3", case.index, invariant, detail, case.context())

    source_blocks = _build_blocks(case.source_keys, case.items_per_block, 0, _source_value)
    target_blocks = _build_blocks(case.target_keys, case.items_per_block, 100_000,
                                  _target_value)
    try:
        source_run = LogicalBlockRunView(SOURCE_LEVEL, tuple(source_blocks),
                                         case.items_per_block)
        target_run = LogicalBlockRunView(TARGET_LEVEL, tuple(target_blocks),
                                         case.items_per_block)
    except Exception as error:  # pragma: no cover - a harness construction bug
        fail("the harness builds a canonical run", f"{type(error).__name__}: {error}")
        raise

    # the view must hold immutable snapshots, never the caller's mutable blocks
    for position, original in enumerate(source_blocks):
        snapshot = source_run.blocks[position]
        if not isinstance(snapshot, LogicalBlockSnapshot):
            fail("the view holds LogicalBlockSnapshot", f"block {position}: {type(snapshot)}")
        if snapshot is original:
            fail("the view snapshots its input", f"block {position} is the caller's object")
        if snapshot.records != tuple(original.records):
            fail("snapshot content equals the block content", f"block {position}")
        if snapshot.block_id != original.block_id or snapshot.capacity != original.capacity:
            fail("snapshot geometry", f"block {position}")

    source_plan = _plan_or_none(case.source_config, source_run.block_count)
    target_plan = _plan_or_none(case.target_config, target_run.block_count)
    if source_plan is None or target_plan is None:
        observations.outcome = "skipped"
        return observations

    source_bound = bind_block_allocation(source_run, source_plan, side=SOURCE)
    target_bound = bind_block_allocation(target_run, target_plan, side=TARGET)
    _check_binding(source_run, source_bound, SOURCE, fail)
    _check_binding(target_run, target_bound, TARGET, fail)

    if case.instrument:
        observations.instrumented = True
        with physical_io_guard() as guard:
            schedule = plan_swat_block_merge_schedule(
                source_run, target_run, source_plan=source_plan, target_plan=target_plan
            )
        if guard.violations:
            fail("zero physical/observational surface", f"calls={guard.violations}")
    else:
        schedule = plan_swat_block_merge_schedule(
            source_run, target_run, source_plan=source_plan, target_plan=target_plan
        )

    _check_schedule(schedule, source_plan, target_plan, fail)

    for bound, side in ((source_bound, SOURCE), (target_bound, TARGET)):
        sampled = sample_bin_interior_points(bound)
        if len(sampled.interiors) != len(sampled.bins):
            fail("one interior point per bin", f"{side}")
        for bin_, point in zip(sampled.bins, sampled.interiors):
            if bin_.is_empty_real:
                if point.key is not DUMMY_POS_INF:
                    fail("a zero-real bin uses DUMMY_POS_INF",
                         f"{side} bin {bin_.bin_index} key={point.key!r}")
                if point.record_index is not None:
                    fail("the sentinel carries no record index", f"{side} bin {bin_.bin_index}")
                continue
            keys = bin_.real_keys()
            if point.key not in keys:
                fail("the interior point is an actual key of its own bin",
                     f"{side} bin {bin_.bin_index} key={point.key!r}")
            if keys[point.record_index] != point.key:
                fail("the interior record index agrees with the key",
                     f"{side} bin {bin_.bin_index}")
        if sampled.is_sampled is not True:
            fail("a sampled allocation reports itself sampled", f"{side}")
        if bound.is_sampled is not False:
            fail("sampling returns a new allocation instead of mutating the binding",
                 f"{side}")

    # stream-domain separation
    if interior_stream_domain(SOURCE) != STREAM_DOMAIN_INTERIOR_SOURCE:
        fail("the source domain is fixed by the side", f"{interior_stream_domain(SOURCE)}")
    if interior_stream_domain(TARGET) != STREAM_DOMAIN_INTERIOR_TARGET:
        fail("the target domain is fixed by the side", f"{interior_stream_domain(TARGET)}")
    for config in (case.source_config, case.target_config):
        if derive_stream_seed(config.seed, STREAM_DOMAIN_INTERIOR_SOURCE) == \
                derive_stream_seed(config.seed, STREAM_DOMAIN_INTERIOR_TARGET):
            fail("the two sides get domain-separated streams", f"seed={config.seed}")

    # determinism: identical inputs reproduce the identical schedule, values and all
    if plan_swat_block_merge_schedule(
            source_run, target_run, source_plan=source_plan, target_plan=target_plan
    ) != schedule:
        fail("same plan/config/content reproduces the schedule", "a second schedule differed")

    # values-only change: same keys, different payloads, identical structure
    other_run = LogicalBlockRunView(
        SOURCE_LEVEL,
        tuple(_build_blocks(case.source_keys, case.items_per_block, 0,
                            lambda key: f"other-s{key}")),
        case.items_per_block,
    )
    other_schedule = plan_swat_block_merge_schedule(
        other_run, target_run, source_plan=source_plan, target_plan=target_plan
    )
    if schedule_structure(other_schedule) != schedule_structure(schedule):
        fail("record values never influence the schedule", "changing values changed structure")

    # signed-tag tie order: on equal interior points the target's negative tag comes first
    ordered = sorted_tagged_interior_points(schedule.source, schedule.target)
    if tuple(schedule.tagged_interior_points) != ordered:
        fail("the schedule exposes the sorted tagged stream", "tagged stream disagreed")
    if len(ordered) != source_plan.bin_count + target_plan.bin_count:
        fail("one tagged interior point per planned bin",
             f"{len(ordered)} for {source_plan.bin_count}+{target_plan.bin_count} bins")
    for first, second in zip(ordered, ordered[1:]):
        if interior_point_sort_key(first.interior_point) != \
                interior_point_sort_key(second.interior_point):
            continue
        if first.signed_tag > second.signed_tag:
            fail("the target's negative tag sorts before the source's positive tag",
                 f"tags {first.signed_tag}/{second.signed_tag} "
                 f"bins {first.bin_index}/{second.bin_index}")

    # snapshot immutability under post-construction mutation of the caller's blocks
    baseline_records = source_run.records()
    baseline_keys = source_run.keys()
    baseline_bound = source_bound
    mutated = _mutate_blocks(source_blocks) + _mutate_blocks(target_blocks)
    if mutated:
        observations.mutation_checked = True
        if source_run.records() != baseline_records or source_run.keys() != baseline_keys:
            fail("a mutation cannot change the snapshotted view", "records or keys changed")
        if bind_block_allocation(source_run, source_plan, side=SOURCE) != baseline_bound:
            fail("a mutation cannot change the binding", "the bound allocation changed")
        if source_bound.real_records() != tuple(
                record for block in source_run.blocks for record in block.records):
            fail("a mutation cannot change the bound content", "bound records changed")
        after = plan_swat_block_merge_schedule(
            source_run, target_run, source_plan=source_plan, target_plan=target_plan
        )
        if schedule_structure(after) != schedule_structure(schedule):
            fail("a mutation cannot change the schedule", "the structure changed")

    # pinned DOMerge projection equivalence, for every two-sided case
    if source_plan.bin_count and target_plan.bin_count:
        observations.two_sided = True
        tagged = [(point.side, interior_point_sort_key(point.interior_point))
                  for point in ordered]
        pinned, fallbacks = pinned_projected_fetches(
            source_plan.bin_count, target_plan.bin_count, tagged
        )
        m3 = tuple((read.side, read.bin_index) for read in schedule.reads)
        if pinned != m3:
            fail("pinned projected fetches equal the M3 reads",
                 f"pinned={pinned} m3={m3}")
        observations.pinned_compared = True
        observations.fallback_triggered = bool(fallbacks)

    return observations


def _check_binding(run: LogicalBlockRunView, bound, side: str, fail) -> None:
    if bound.block_count != run.block_count:
        fail("the binding covers the whole run", f"{side}: {bound.block_count}")
    if bound.level != run.level or bound.items_per_block != run.items_per_block:
        fail("the binding carries the view identity", f"{side}")
    if bound.bound_ranks() != tuple(range(run.block_count)):
        fail("every real rank is bound exactly once", f"{side}: {bound.bound_ranks()}")
    consumed = 0
    for bin_ in bound.bins:
        if bin_.logical_rank_start != consumed:
            fail("contiguous bound rank intervals", f"{side} bin {bin_.bin_index}")
        if bin_.logical_rank_stop - bin_.logical_rank_start != bin_.real_block_count:
            fail("bound rank interval width == real block count", f"{side}")
        if len(bin_.blocks) != bin_.real_block_count:
            fail("a bin binds its real block count", f"{side} bin {bin_.bin_index}")
        if bin_.real_block_count + bin_.dummy_block_count != bin_.bin_capacity_blocks:
            fail("bound bin slots == capacity", f"{side} bin {bin_.bin_index}")
        if bin_.bin_capacity_blocks != bound.plan.bin_capacity_blocks:
            fail("bound bin capacity equals the plan capacity", f"{side}")
        for block in bin_.blocks:
            if not isinstance(block, LogicalBlockSnapshot):
                fail("bound bins hold snapshots", f"{side}: {type(block).__name__}")
        if bin_.sampled_load != bound.plan.sampled_loads[bin_.bin_index]:
            fail("bound bin load equals the planned load", f"{side}")
        consumed = bin_.logical_rank_stop
    if consumed != run.block_count:
        fail("the bins cover every rank", f"{side}: {consumed} != {run.block_count}")

    # flattening the bound bins reconstructs the input run exactly
    flattened = [record for bin_ in bound.bins for record in bin_.real_records()]
    if tuple(flattened) != run.records():
        fail("flatten(bound bins) == the input run records", f"{side}")
    if bound.real_keys() != run.keys():
        fail("flatten(bound bins) keys == the run keys", f"{side}")
    if bound.sampled_loads != bound.plan.sampled_loads:
        fail("sampled loads are passed through", f"{side}")
    if bound.noisy_prefix_sums != bound.plan.noisy_prefix_sums:
        fail("noisy prefix sums are passed through", f"{side}")
    if bound.additive_error_blocks != bound.plan.additive_error_blocks:
        fail("the additive error is passed through", f"{side}")
    if bound.bin_count != bound.plan.bin_count or bound.block_count != bound.plan.block_count:
        fail("the binding preserves the plan geometry", f"{side}")


def _check_schedule(schedule: SwatBlockMergeSchedule, source_plan, target_plan, fail) -> None:
    reads = tuple((read.side, read.bin_index) for read in schedule.reads)
    for side, bin_index in reads:
        if side == SOURCE:
            if not 0 <= bin_index < source_plan.bin_count:
                fail("no out-of-range read", f"source bin {bin_index}")
        elif side == TARGET:
            if not 0 <= bin_index < target_plan.bin_count:
                fail("no out-of-range read", f"target bin {bin_index}")
        else:
            fail("reads carry only the two sides", f"{side!r}")

    for side, plan in ((SOURCE, source_plan), (TARGET, target_plan)):
        scheduled = sorted(read.bin_index for read in schedule.reads if read.side == side)
        if scheduled != list(range(plan.bin_count)):
            fail("every planned bin of a side is read exactly once",
                 f"{side}: {scheduled} for bin_count={plan.bin_count}")

    if schedule.read_count != sum(1 for _ in reads):
        fail("read_count agrees with the reads", f"{schedule.read_count}")
    if reads and reads[0] != ((SOURCE, 0) if source_plan.bin_count else (TARGET, 0)):
        fail("the first read is the source's preload", f"reads={reads[:2]}")
    if source_plan.bin_count and target_plan.bin_count:
        if len(reads) < 2 or reads[1] != (TARGET, 0):
            fail("the second read is the target's preload", f"reads={reads[:2]}")
    if schedule.read_order(SOURCE) != schedule.source_read_order or             schedule.read_order(TARGET) != schedule.target_read_order:
        fail("the per-side read order agrees with the reads", "read_order disagreed")
    for side, plan in ((SOURCE, source_plan), (TARGET, target_plan)):
        order = (schedule.source_read_order if side == SOURCE else schedule.target_read_order)
        if tuple(order) != tuple(read.bin_index for read in schedule.reads if read.side == side):
            fail("the per-side read order matches the reads", f"{side}")


# ---------------------------------------------------------------------------
# static separation
# ---------------------------------------------------------------------------


def _module_executable_tokens(path: Path) -> Tuple[set, set]:
    """Identifiers/strings and imported module names of a module's executable code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body[0].value.value = ""
    tokens: set = set()
    imports: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value)
    return tokens, imports


def check_static_separation() -> None:
    """Repository-level separation checks that hold for the whole sweep."""
    package_dir = SRC_DIR / "swat_m_block"
    tools_dir = Path(__file__).resolve().parent
    for name in PRODUCTION_MODULES:
        path = package_dir / name
        if not path.exists():
            raise _fail("static", -1, "the production module exists", f"missing {name}",
                        f"module={name}")
        tokens, imports = _module_executable_tokens(path)
        for token in FORBIDDEN_MODULE_TOKENS:
            if token in tokens:
                raise _fail("static", -1, "no physical/observational surface",
                            f"{name} mentions {token} in executable code",
                            f"module={name}")
        for module in sorted(imports):
            if module not in ALLOWED_IMPORTS:
                raise _fail(
                    "static", -1,
                    "the production package imports only the frozen substrate and its siblings",
                    f"{name} imports {module!r}", f"module={name}",
                )
    for path in sorted(package_dir.glob("*.py")):
        if "post_m3_regression_sweep" in path.read_text(encoding="utf-8"):
            raise _fail("static", -1, "production code never imports the harness",
                        f"{path.name} references the harness", f"module={path.name}")
    if tools_dir.name != "tools" or tools_dir.parent.name != "codes":
        raise _fail("static", -1, "the harness lives outside the production package",
                    str(tools_dir), "codes/tools")


# ---------------------------------------------------------------------------
# runner / report
# ---------------------------------------------------------------------------


@dataclass
class SweepReport:
    mode: str
    master_seed: int
    total_cases: int = 0
    m1_cases: int = 0
    m2_plans: int = 0
    m2_failures: int = 0
    m3_schedules: int = 0
    m3_two_sided: int = 0
    m3_one_sided: int = 0
    m3_skipped: int = 0
    pinned_comparisons: int = 0
    fallback_triggered_cases: int = 0
    mutation_checked_cases: int = 0
    instrumented_cases: int = 0
    elapsed_seconds: float = 0.0
    passed: bool = True


def summary_lines(report: SweepReport) -> List[str]:
    return [
        f"post-M3 regression sweep ({SWEEP_VERSION}) — mode={report.mode} "
        f"master_seed={report.master_seed}",
        f"  total cases                     : {report.total_cases}",
        f"  M1 cases                        : {report.m1_cases}",
        f"  M2 successful plans             : {report.m2_plans}",
        f"  M2 explicit failures            : {report.m2_failures}",
        f"  M3 schedules                    : {report.m3_schedules} "
        f"(two-sided {report.m3_two_sided}, one-sided {report.m3_one_sided}, "
        f"skipped {report.m3_skipped})",
        f"  {'pinned merge-loop comparisons':<32}: {report.pinned_comparisons}",
        f"  pinned fallback-triggered cases : {report.fallback_triggered_cases}",
        f"  mutation-checked M3 cases       : {report.mutation_checked_cases}",
        f"  instrumented (zero-I/O) cases   : {report.instrumented_cases}",
        f"  elapsed                         : {report.elapsed_seconds:.2f} s "
        f"(operational only; no performance claim)",
        f"  {'PASS' if report.passed else 'FAIL'}",
    ]


def run_sweep(mode: str = "quick", master_seed: int = DEFAULT_MASTER_SEED,
              *, stream=None) -> SweepReport:
    """Run the sweep in ``mode``; raises :class:`SweepFailure` on the first mismatch."""
    budget = BUDGETS[mode]
    emit = (stream or sys.stdout).write
    check_static_separation()
    started = time.perf_counter()
    report = SweepReport(mode=mode, master_seed=master_seed)

    rng = random.Random(master_seed)
    m1_cases = generate_m1_cases(rng, budget)
    m2_cases = generate_m2_cases(rng, budget)
    m3_cases = generate_m3_cases(rng, budget)

    for case in m1_cases:
        check_m1_case(case)
        report.m1_cases += 1
        if case.instrument:
            report.instrumented_cases += 1

    for case in m2_cases:
        if check_m2_case(case) == "plan":
            report.m2_plans += 1
        else:
            report.m2_failures += 1

    for case in m3_cases:
        observations = check_m3_case(case)
        if observations.outcome == "skipped":
            report.m3_skipped += 1
            continue
        report.m3_schedules += 1
        if observations.two_sided:
            report.m3_two_sided += 1
        else:
            report.m3_one_sided += 1
        if observations.pinned_compared:
            report.pinned_comparisons += 1
        if observations.fallback_triggered:
            report.fallback_triggered_cases += 1
        if observations.mutation_checked:
            report.mutation_checked_cases += 1
        if observations.instrumented:
            report.instrumented_cases += 1

    report.total_cases = (report.m1_cases + report.m2_plans + report.m2_failures
                          + report.m3_schedules)
    report.elapsed_seconds = time.perf_counter() - started
    for line in summary_lines(report):
        emit(line + "\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="post_m3_regression_sweep.py",
        description="Post-M3 deterministic differential/invariant regression sweep (QA-1).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="a few seconds, suitable for local review (default)")
    mode.add_argument("--full", action="store_true",
                      help="thousands of combined generated cases, bounded runtime")
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED,
                        help=f"master seed (default {DEFAULT_MASTER_SEED})")
    args = parser.parse_args(argv)
    chosen = "full" if args.full else "quick"
    try:
        report = run_sweep(chosen, args.seed)
    except SweepFailure as failure:
        print(failure.report(args.seed))
        print("  FAIL")
        return 1
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
