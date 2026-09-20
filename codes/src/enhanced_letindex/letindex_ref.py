"""LETIndex-Ref-v1 reference profile: upstream-faithful LETIndex lookup semantics.

This module implements the **reference profile** frozen by
``decisions/0009-g2-b0-letindex-reference-profile.md``.  It answers one question
precisely: *what exact baseline are the later G2 attacks claiming to attack?*

Provenance: every "upstream-faithful" rule below is derived from the pinned
upstream implementation

```text
repo:   https://github.com/chiiips/LETIndex
commit: 9b28fce03638a3cd97402d43c10bca644d0a95a5
```

and the code paths that define them are named in the decision record
(``Server/index/include/pgm/pgm_index_dynamic.hpp``, ``.../pgm/defs.h``,
``.../pgm/storage_utility.hpp``).

What this module models (UPSTREAM-FAITHFUL):

* frozen geometry ``BASE=8``, ``MIN_LEVEL=3``, ``MIN_INDEX_LEVEL=6``,
  ``MIN_DISK_LEVEL=7``, ``BLOCK_SIZE=4096``, ``ITEM_SIZE=16``,
  ``ITEMS_PER_BLOCK=256`` and ``max_size(level) = BASE**level``;
* the PGM window ``lo = max(0, pos - eps)``, ``hi = min(n, pos + eps + 2)`` over
  **logical** item rank (the accepted canonical batch PGM of decision 0002/0003 is
  reused unchanged — no new approximation rule is introduced);
* PGM-bounded ``lower_bound_bl_disk_pgm(first, last, key)``: data-dependent
  binary search whose **first** probe is the midpoint and uses the conditional
  adjacent-block prefetch — midpoint block always, ``mid-1`` only when the
  interval reaches into the previous block, ``mid+1`` only when it reaches into
  the next block — so the first backing-store fetch is 1, 2 or 3 contiguous
  blocks.  Later probes are served from the current buffer or the two prefetched
  neighbour buffers (no backing-store read) and otherwise trigger a real read;
* the non-PGM ``lower_bound_bl_disk(0, n, key)`` full-level binary search with
  single-block reads (no prefetch cache);
* ascending level search ``for level in MIN_LEVEL .. used_levels-1`` with empty
  levels skipped and hit-and-stop between levels.

What this module adds as an **EXPERIMENTAL EXTENSION** (explicitly *not*
upstream): the ``(level, block_offset) -> SlotId`` adapter that expresses
upstream's physically distinguishable per-level regions in the frozen G2-A
transcript abstraction, the ``ordered`` / ``static_prp`` layout modes, the
deterministic loader that materialises a level, and the seeded passive workload
generators (``ref_workload``).

Legacy simulator settings (``ratio=4``, ``[4,16,64,...]`` capacities, a fixed
three-block last mile, ``hit±1``) are **LEGACY-SIMULATOR-ONLY** and are not
implemented here at all.  The accepted Enhanced-Baseline lookup
(``TrustedEngine.lookup_level`` minimal-cover read-all-candidates) is untouched
and remains a separate profile.

This milestone implements **no attack, no recovery and no defence**.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .identifiers import RecordKey, SlotId
from .leakage import (
    CLASS_BUILD_INITIAL,
    CLASS_BUILD_PGM,
    CLASS_QUERY,
    OPERATION_ALL_KEYS,
    OPERATION_KEYS,
    TRANSCRIPT_KEYS,
    AdversaryTranscript,
    LeakageError,
    ObservedEvent,
    ObservedOperation,
    SUMMARY_KEYS,
    dumps_transcript,
)
from .pgm import BatchPgmIndex, SearchResult, build_batch_pgm

__all__ = [
    "UPSTREAM_REPO",
    "UPSTREAM_COMMIT",
    "UPSTREAM_FAITHFUL",
    "EXPERIMENTAL_EXTENSION",
    "LEGACY_SIMULATOR_ONLY",
    "ENHANCED_BASELINE",
    "CLASSIFICATIONS",
    "PROFILE_NAME",
    "REFERENCE_ARTIFACT_SCHEMA_VERSION",
    "REFERENCE_TRANSCRIPT_SOURCE",
    "BASE",
    "MIN_LEVEL",
    "MIN_INDEX_LEVEL",
    "MIN_DISK_LEVEL",
    "BLOCK_SIZE",
    "ITEM_SIZE",
    "ITEMS_PER_BLOCK",
    "REFERENCE_EPSILON",
    "FROZEN_DEFAULTS",
    "LEGACY_SIMULATOR_SETTINGS",
    "LAYOUT_ORDERED",
    "LAYOUT_STATIC_PRP",
    "LAYOUT_MODES",
    "PUBLIC_PROFILE_KEYS",
    "READ",
    "WRITE",
    "EVENT_OPERATIONS",
    "ReferenceProfileError",
    "ReferenceProfile",
    "LevelRegion",
    "SlotProjection",
    "SeededPermutation",
    "ReferenceLevel",
    "ReferenceQueryOutcome",
    "LevelLookupOutcome",
    "ReferenceIndex",
    "ReferenceTraceBuilder",
    "ReferenceExperiment",
    "default_scenario",
    "dumps_reference_artifact",
    "write_reference_artifact_json",
    "validate_reference_transcript",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# provenance and classification
# ---------------------------------------------------------------------------

UPSTREAM_REPO = "https://github.com/chiiips/LETIndex"
UPSTREAM_COMMIT = "9b28fce03638a3cd97402d43c10bca644d0a95a5"

#: The four classification tags every G2-B0 item must carry (decision 0009).
UPSTREAM_FAITHFUL = "UPSTREAM-FAITHFUL"
EXPERIMENTAL_EXTENSION = "EXPERIMENTAL EXTENSION"
LEGACY_SIMULATOR_ONLY = "LEGACY-SIMULATOR-ONLY"
ENHANCED_BASELINE = "ENHANCED-BASELINE"
CLASSIFICATIONS = (
    UPSTREAM_FAITHFUL,
    EXPERIMENTAL_EXTENSION,
    LEGACY_SIMULATOR_ONLY,
    ENHANCED_BASELINE,
)

PROFILE_NAME = "LETINDEX-REF-V1"
REFERENCE_ARTIFACT_SCHEMA_VERSION = "g2b0-ref-1"

#: The observation block is the frozen ``g2a-1`` shape.  Its ``source_schema_version``
#: names where the observation was extracted from: for this profile that is the
#: reference artifact itself (``g2b0-ref-1``), *not* a G1-E run, so the transcript
#: is deliberately not accepted by the G1-E-source validator
#: ``leakage.validate_transcript`` — see ``validate_reference_transcript``.
REFERENCE_TRANSCRIPT_SOURCE = REFERENCE_ARTIFACT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# frozen upstream geometry
# ---------------------------------------------------------------------------

BASE = 8
MIN_LEVEL = 3
MIN_INDEX_LEVEL = 6
MIN_DISK_LEVEL = 7
BLOCK_SIZE = 4096
ITEM_SIZE = 16
ITEMS_PER_BLOCK = 256  # BLOCK_SIZE // ITEM_SIZE
REFERENCE_EPSILON = 64  # upstream ``epsilon_value`` default

FROZEN_DEFAULTS = {
    "base": BASE,
    "min_level": MIN_LEVEL,
    "min_index_level": MIN_INDEX_LEVEL,
    "min_disk_level": MIN_DISK_LEVEL,
    "block_size": BLOCK_SIZE,
    "item_size": ITEM_SIZE,
    "items_per_block": ITEMS_PER_BLOCK,
    "epsilon": REFERENCE_EPSILON,
}

#: Legacy-simulator settings that must never be presented as upstream LETIndex.
LEGACY_SIMULATOR_SETTINGS = (
    "ratio=4",
    "[4,16,64,...]",
    "fixed 3-block last mile",
    "hit±1",
)

READ = "READ"
WRITE = "WRITE"
EVENT_OPERATIONS = (READ, WRITE)

LAYOUT_ORDERED = "ordered"
LAYOUT_STATIC_PRP = "static_prp"
LAYOUT_MODES = (LAYOUT_ORDERED, LAYOUT_STATIC_PRP)

#: Ordered (deterministic key order) subset of profile fields an adversary may be told:
#: the frozen geometry and the layout *mode*.  The analysis seed and the layout key are
#: deliberately excluded — see ``ReferenceProfile.public_dict``.
PUBLIC_PROFILE_KEYS = (
    "profile",
    "base",
    "min_level",
    "min_index_level",
    "min_disk_level",
    "block_size",
    "item_size",
    "items_per_block",
    "epsilon",
    "layout",
    "variant_label",
)


class ReferenceProfileError(Exception):
    """A reference-profile request violates the frozen G2-B0 contract."""


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceProfile:
    """Frozen geometry + layout configuration of one LETIndex-Ref-v1 experiment.

    ``reference()`` returns the paper-faithful defaults.  Any deviation must go
    through ``experimental_variant()`` with an explicit label, so a parameter
    sweep can never be silently presented as the reference profile.
    """

    base: int = BASE
    min_level: int = MIN_LEVEL
    min_index_level: int = MIN_INDEX_LEVEL
    min_disk_level: int = MIN_DISK_LEVEL
    block_size: int = BLOCK_SIZE
    item_size: int = ITEM_SIZE
    items_per_block: int = ITEMS_PER_BLOCK
    epsilon: int = REFERENCE_EPSILON
    layout: str = LAYOUT_ORDERED
    #: Analysis/observation seed.  It is the value published in the frozen ``g2a-1``
    #: run metadata, so it must never drive anything that has to stay hidden.
    seed: int = 0
    #: Layout key for ``static_prp`` (EXPERIMENTAL EXTENSION).  Evaluator-side only:
    #: it is never part of an attack-facing payload, and it must be supplied
    #: explicitly so the permutation cannot be recomputed from the published seed.
    #: ``None`` for the ``ordered`` layout, which has no permutation.
    prp_key: Optional[int] = None
    variant_label: Optional[str] = None

    def __post_init__(self) -> None:
        for name, expected in FROZEN_DEFAULTS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ReferenceProfileError(f"{name} must be an integer, got {value!r}")
            if value != expected and not self.variant_label:
                raise ReferenceProfileError(
                    f"{name}={value!r} deviates from the frozen upstream default "
                    f"{expected!r}; deviations require an explicit variant_label so they "
                    "can never be presented as the reference profile"
                )
        if self.item_size <= 0 or self.items_per_block <= 0:
            raise ReferenceProfileError("item_size and items_per_block must be positive")
        if self.layout not in LAYOUT_MODES:
            raise ReferenceProfileError(
                f"layout must be one of {LAYOUT_MODES}, got {self.layout!r}"
            )
        if self.epsilon < 0:
            raise ReferenceProfileError("epsilon must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ReferenceProfileError(f"seed must be an integer, got {self.seed!r}")
        if self.prp_key is not None:
            if isinstance(self.prp_key, bool) or not isinstance(self.prp_key, int):
                raise ReferenceProfileError(
                    f"prp_key must be an integer or None, got {self.prp_key!r}"
                )
            if self.prp_key == self.seed:
                raise ReferenceProfileError(
                    "prp_key must differ from the analysis seed: the analysis seed is "
                    "published in the g2a-1 run metadata, so deriving the placement from "
                    "it would let an adversary recompute the hidden permutation"
                )
        if self.layout == LAYOUT_STATIC_PRP and self.prp_key is None:
            raise ReferenceProfileError(
                "layout=static_prp requires an explicit prp_key (evaluator-side, never "
                "attack-facing); deriving it from the published analysis seed would make "
                "the hidden permutation reconstructible from the attack input"
            )
        if self.layout == LAYOUT_ORDERED and self.prp_key is not None:
            raise ReferenceProfileError(
                "layout=ordered has no PRP key: the identity placement is the "
                "upstream-faithful layout and carries no key material"
            )
        if self.variant_label is not None and not str(self.variant_label).strip():
            raise ReferenceProfileError("variant_label must be a non-empty string")

    # ------------------------------------------------------------- factories

    @classmethod
    def reference(
        cls, *, layout: str = LAYOUT_ORDERED, seed: int = 0,
        prp_key: Optional[int] = None,
    ) -> "ReferenceProfile":
        """Paper-faithful defaults (the only profile that may be called reference)."""
        return cls(layout=layout, seed=seed, prp_key=prp_key)

    @classmethod
    def experimental_variant(
        cls, label: str, *, layout: str = LAYOUT_ORDERED, seed: int = 0,
        prp_key: Optional[int] = None,
        items_per_block: Optional[int] = None, epsilon: Optional[int] = None,
        base: Optional[int] = None, min_level: Optional[int] = None,
        min_index_level: Optional[int] = None, min_disk_level: Optional[int] = None,
        block_size: Optional[int] = None, item_size: Optional[int] = None,
    ) -> "ReferenceProfile":
        """An explicitly labelled deviation from the frozen defaults."""
        if not label or not str(label).strip():
            raise ReferenceProfileError("an experimental variant requires a non-empty label")
        overrides = {
            "items_per_block": items_per_block,
            "epsilon": epsilon,
            "base": base,
            "min_level": min_level,
            "min_index_level": min_index_level,
            "min_disk_level": min_disk_level,
            "block_size": block_size,
            "item_size": item_size,
        }
        kwargs = {key: value for key, value in overrides.items() if value is not None}
        return cls(
            layout=layout, seed=seed, prp_key=prp_key, variant_label=str(label), **kwargs
        )

    # ------------------------------------------------------------- geometry

    def max_size(self, level: int) -> int:
        """Upstream ``max_size(level) = 1 << (level * ceil_log2(base))``."""
        if isinstance(level, bool) or not isinstance(level, int):
            raise ReferenceProfileError(f"level must be an integer, got {level!r}")
        if level < 0:
            raise ReferenceProfileError(f"level must be non-negative, got {level!r}")
        return self.base ** level

    def block_count(self, item_count: int) -> int:
        """Upstream block packing: ``ceil(item_count / ItemCountPerBlock)``."""
        if item_count < 0:
            raise ReferenceProfileError("item_count must be non-negative")
        return (item_count + self.items_per_block - 1) // self.items_per_block

    def has_index(self, level: int) -> bool:
        """Upstream ``has_pgm(level) = level >= min_index_level``."""
        return level >= self.min_index_level

    def is_reference_defaults(self) -> bool:
        return all(
            getattr(self, name) == expected for name, expected in FROZEN_DEFAULTS.items()
        )

    def is_upstream_faithful_layout(self) -> bool:
        """``ordered`` is the upstream-faithful layout; ``static_prp`` is not."""
        return self.layout == LAYOUT_ORDERED

    def is_upstream_faithful(self) -> bool:
        """Both the frozen geometry **and** the upstream layout are in force."""
        return self.is_reference_defaults() and self.is_upstream_faithful_layout()

    def layout_key(self) -> int:
        """The key that seeds the per-level permutation (:meth:`SeededPermutation`).

        Only meaningful for ``static_prp``.  It is the explicit ``prp_key``, never the
        published analysis seed, and it is evaluator-side: it must not be handed to an
        adversary-facing consumer.
        """
        if self.prp_key is None:
            raise ReferenceProfileError(
                "layout=ordered has no permutation key; the identity placement is used"
            )
        return self.prp_key

    def classification(self) -> dict:
        """Per-aspect classification tags (EXPERIMENTAL EXTENSION unless upstream-faithful).

        The ``profile`` tag is ``UPSTREAM-FAITHFUL`` only when *both* the geometry and
        the layout are the upstream ones, so a ``static_prp`` profile can never be
        labelled upstream-faithful at profile level.
        """
        geometry = (
            UPSTREAM_FAITHFUL if self.is_reference_defaults() else EXPERIMENTAL_EXTENSION
        )
        layout = (
            UPSTREAM_FAITHFUL
            if self.is_upstream_faithful_layout()
            else EXPERIMENTAL_EXTENSION
        )
        return {
            "profile": (
                UPSTREAM_FAITHFUL
                if geometry == UPSTREAM_FAITHFUL and layout == UPSTREAM_FAITHFUL
                else EXPERIMENTAL_EXTENSION
            ),
            "geometry": geometry,
            "layout": layout,
            "lookup_semantics": UPSTREAM_FAITHFUL,
        }

    def level_backing(self, level: int) -> str:
        """``memory`` for levels below ``MIN_DISK_LEVEL``, else ``disk``.

        Both are backing-store accesses at block granularity; the distinction is
        kept as level metadata and never enters the adversary transcript.
        """
        return "disk" if level >= self.min_disk_level else "memory"

    # ------------------------------------------------------------- serde

    def to_dict(self) -> dict:
        """Full evaluator-side profile (round-trips; contains the layout key)."""
        return {
            "profile": PROFILE_NAME,
            "classification": self.classification(),
            "base": self.base,
            "min_level": self.min_level,
            "min_index_level": self.min_index_level,
            "min_disk_level": self.min_disk_level,
            "block_size": self.block_size,
            "item_size": self.item_size,
            "items_per_block": self.items_per_block,
            "epsilon": self.epsilon,
            "layout": self.layout,
            "seed": self.seed,
            "prp_key": self.prp_key,
            "variant_label": self.variant_label,
            "is_reference_defaults": self.is_reference_defaults(),
            "is_upstream_faithful": self.is_upstream_faithful(),
        }

    def public_dict(self) -> dict:
        """The attack-visible profile subset: geometry + layout mode only.

        The analysis seed **and** the layout key are deliberately absent: the seed is
        already published in the frozen ``g2a-1`` run metadata, and the layout key must
        stay unknown so that recovering the hidden permutation is a real experiment
        rather than a recomputation from the attack input.
        """
        full = self.to_dict()
        return {key: full[key] for key in PUBLIC_PROFILE_KEYS}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ReferenceProfile":
        return cls(
            base=int(data["base"]),
            min_level=int(data["min_level"]),
            min_index_level=int(data["min_index_level"]),
            min_disk_level=int(data["min_disk_level"]),
            block_size=int(data["block_size"]),
            item_size=int(data["item_size"]),
            items_per_block=int(data["items_per_block"]),
            epsilon=int(data["epsilon"]),
            layout=str(data["layout"]),
            seed=int(data["seed"]),
            prp_key=(
                None if data.get("prp_key") is None else int(data["prp_key"])
            ),
            variant_label=data.get("variant_label"),
        )


# ---------------------------------------------------------------------------
# layout / slot projection
# ---------------------------------------------------------------------------


class SeededPermutation:
    """Deterministic keyed bijection on ``range(block_count)`` (EXPERIMENTAL EXTENSION).

    This is an explicitly seeded permutation used as a sanity-hardening placement
    baseline.  It is **not** claimed to be a cryptographically secure PRP, and it
    is fixed for the whole experiment/epoch: no query-time rerandomisation, no
    reshuffle, no stash.
    """

    def __init__(self, block_count: int, seed: int) -> None:
        if block_count < 0:
            raise ReferenceProfileError("block_count must be non-negative")
        order = list(range(block_count))
        random.Random(seed).shuffle(order)
        self.block_count = int(block_count)
        self.seed = int(seed)
        self._forward = tuple(order)
        inverse = [0] * block_count
        for logical, physical in enumerate(self._forward):
            inverse[physical] = logical
        self._inverse = tuple(inverse)

    def forward(self, logical_block: int) -> int:
        """``logical block rank -> physical block offset``."""
        self._check(logical_block)
        return self._forward[logical_block]

    def inverse(self, physical_block: int) -> int:
        """``physical block offset -> logical block rank``."""
        self._check(physical_block)
        return self._inverse[physical_block]

    def _check(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ReferenceProfileError(f"block index must be an integer, got {index!r}")
        if not 0 <= index < self.block_count:
            raise ReferenceProfileError(
                f"block index {index} is outside [0, {self.block_count})"
            )

    def as_tuple(self) -> tuple[int, ...]:
        return self._forward


def _level_seed(layout_key: int, level: int) -> int:
    """Per-level deterministic permutation seed (documented derivation).

    Keyed by the explicit **layout key** (``ReferenceProfile.prp_key``), never by the
    analysis seed: the analysis seed is published in the frozen ``g2a-1`` run metadata,
    so deriving the placement from it would let an adversary recompute the hidden
    permutation from its own input.
    """
    return layout_key * 1_000_003 + level


@dataclass(frozen=True)
class LevelRegion:
    """One level's SlotId region: ``block_count`` consecutive ids from ``region_start``."""

    level: int
    region_start: int
    block_count: int

    @property
    def region_end(self) -> int:
        return self.region_start + self.block_count

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "region_start": self.region_start,
            "block_count": self.block_count,
        }


@dataclass(frozen=True)
class SlotProjection:
    """Deterministic injective ``(level, block_offset) -> SlotId`` (EXPERIMENTAL EXTENSION).

    Upstream keeps every level in its own file/shm region, so the regions are
    already physically distinguishable.  This adapter makes that explicit without
    touching the frozen ``g2a-1`` transcript: level regions are disjoint, and
    inside a level increasing block offsets map monotonically to increasing
    SlotIds *before* any optional PRP layer is applied.
    """

    regions: tuple[LevelRegion, ...]
    base: int = 0

    @classmethod
    def from_block_counts(
        cls, block_counts: Mapping[int, int], *, base: int = 0
    ) -> "SlotProjection":
        regions = []
        cursor = base
        for level in sorted(block_counts):
            count = int(block_counts[level])
            if count < 0:
                raise ReferenceProfileError(f"level {level} has a negative block count")
            regions.append(LevelRegion(level=level, region_start=cursor, block_count=count))
            cursor += count
        return cls(regions=tuple(regions), base=base)

    def region(self, level: int) -> LevelRegion:
        for region in self.regions:
            if region.level == level:
                return region
        raise ReferenceProfileError(f"level {level} has no SlotId region")

    def levels(self) -> tuple[int, ...]:
        return tuple(region.level for region in self.regions)

    @property
    def total_blocks(self) -> int:
        return sum(region.block_count for region in self.regions)

    def slot(self, level: int, block_offset: int) -> SlotId:
        region = self.region(level)
        if isinstance(block_offset, bool) or not isinstance(block_offset, int):
            raise ReferenceProfileError(
                f"block_offset must be an integer, got {block_offset!r}"
            )
        if not 0 <= block_offset < region.block_count:
            raise ReferenceProfileError(
                f"level {level} has no block offset {block_offset} "
                f"(block_count={region.block_count})"
            )
        return SlotId(region.region_start + block_offset)

    def locate(self, slot: SlotId | int) -> tuple[int, int]:
        """Inverse projection: ``SlotId -> (level, block_offset)``."""
        value = slot.value if isinstance(slot, SlotId) else int(slot)
        for region in self.regions:
            if region.region_start <= value < region.region_end:
                return region.level, value - region.region_start
        raise ReferenceProfileError(f"slot {value} is outside the projection")

    def to_dict(self) -> dict:
        return {"base": self.base, "regions": [r.to_dict() for r in self.regions]}


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceLevel:
    """One upstream level: sorted items, optional PGM, and its physical placement."""

    level: int
    items_per_block: int
    keys: tuple[RecordKey, ...]
    values: tuple[str, ...]
    pgm: Optional[BatchPgmIndex]
    placement: tuple[int, ...]  # logical block rank -> physical block offset
    backing: str

    @property
    def item_count(self) -> int:
        return len(self.keys)

    @property
    def block_count(self) -> int:
        return len(self.placement)

    @property
    def has_pgm(self) -> bool:
        return self.pgm is not None

    def key_at(self, position: int) -> RecordKey:
        return self.keys[position]

    def value_at(self, position: int) -> str:
        return self.values[position]

    def block_range(self, logical_block: int) -> tuple[int, int]:
        """Item-position range ``[start, end)`` of one logical block."""
        start = logical_block * self.items_per_block
        end = min(start + self.items_per_block, self.item_count)
        return start, end

    def physical_block(self, logical_block: int) -> int:
        return self.placement[logical_block]

    def logical_block(self, physical_block: int) -> int:
        """``physical block offset -> logical block rank`` (inverse placement)."""
        if not 0 <= physical_block < self.block_count:
            raise ReferenceProfileError(
                f"level {self.level} has {self.block_count} blocks; "
                f"no physical offset {physical_block}"
            )
        return self.placement.index(physical_block)

    def physical_blocks(self) -> tuple[int, ...]:
        return self.placement

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "item_count": self.item_count,
            "block_count": self.block_count,
            "backing": self.backing,
            "has_pgm": self.has_pgm,
            "items_per_block": self.items_per_block,
            "placement": list(self.placement),
        }


@dataclass(frozen=True)
class LevelLookupOutcome:
    """Evaluator-side result of one level lookup (never adversary-visible)."""

    level: int
    pos: int
    found: bool
    value: Optional[str]
    interval: tuple[int, int]
    path: str
    events: tuple[tuple[str, int], ...]
    probes: tuple[int, ...]
    fetched_blocks: tuple[int, ...]
    prefetch_size: Optional[int] = None

    @property
    def read_count(self) -> int:
        return sum(1 for operation, _ in self.events if operation == READ)


@dataclass(frozen=True)
class ReferenceQueryOutcome:
    """Evaluator-side result of one multi-level reference query."""

    key: int
    found: bool
    value: Optional[str]
    hit_level: Optional[int]
    searched_levels: tuple[int, ...]
    skipped_empty_levels: tuple[int, ...]
    paths: tuple[tuple[int, str], ...]
    events: tuple[tuple[str, int], ...]
    probes: tuple[tuple[int, int], ...]
    level_outcomes: tuple[LevelLookupOutcome, ...]

    @property
    def read_count(self) -> int:
        return sum(1 for operation, _ in self.events if operation == READ)

    def slots(self) -> tuple[int, ...]:
        return tuple(slot for _, slot in self.events)


class _LevelSearcher:
    """Mutable per-level-lookup state: buffers, prefetch cache and the event log.

    Models upstream ``lower_bound_bl_disk_pgm`` / ``lower_bound_bl_disk``:
    ``current_block`` is upstream's ``last_block``, and the two prefetched
    neighbour buffers are upstream's ``cached_block0`` / ``cached_block1``.

    Modelling decision (documented in decision 0009): upstream's neighbour cache
    is a *member* of the index object and is never invalidated, so a stale cache
    entry from an earlier level/query can serve a later lookup.  That is an
    accidental implementation artifact, not algorithmically observable access
    semantics, so this profile seeds the cache per level lookup (empty unless the
    first probe prefetches) and never carries it across lookups.
    """

    def __init__(self, level: ReferenceLevel, projection: SlotProjection) -> None:
        self.level = level
        self.projection = projection
        self.current_block: Optional[int] = None  # physical block in the main buffer
        self.cache_left: Optional[int] = None
        self.cache_right: Optional[int] = None
        self.events: list[tuple[str, int]] = []
        self.probes: list[int] = []
        self.fetched_blocks: list[int] = []
        self.prefetch_counts: list[int] = []

    # ----------------------------------------------------------- projections

    @property
    def _items_per_block(self) -> int:
        return self.level.items_per_block

    def _logical_block(self, item_position: int) -> int:
        return item_position // self._items_per_block

    def _physical(self, item_position: int) -> int:
        return self.level.physical_block(self._logical_block(item_position))

    def _slot_value(self, physical_block: int) -> int:
        return self.projection.slot(self.level.level, physical_block).value

    def _read(self, physical_block: int) -> None:
        self.events.append((READ, self._slot_value(physical_block)))
        self.fetched_blocks.append(physical_block)

    def _read_run(self, physical_blocks: Iterable[int]) -> None:
        """One contiguous multi-block backing-store read, recorded per block."""
        for physical_block in sorted(set(physical_blocks)):
            self._read(physical_block)

    # -------------------------------------------------------------- accessing

    def _key(self, item_position: int) -> RecordKey:
        return self.level.key_at(item_position)

    def _obtain_plain(self, item_position: int) -> RecordKey:
        """Upstream ``obtain_item_on_disk`` (``read_block_or_not``): no cache."""
        physical = self._physical(item_position)
        if physical != self.current_block:
            self._read(physical)
            self.current_block = physical
        return self._key(item_position)

    def _obtain_or_cache(self, item_position: int) -> RecordKey:
        """Upstream ``obtain_item_on_disk_or_cache``.

        A probe served from a prefetched neighbour buffer does **not** replace the
        current buffer: upstream memcpys straight out of ``cached_block0/1`` and
        leaves ``block_data`` / ``last_block`` untouched, so the midpoint block stays
        current and the neighbour cache stays valid for the rest of the lookup.
        Only a genuine cache miss changes the current buffer.
        """
        physical = self._physical(item_position)
        if physical != self.current_block:
            if physical in (self.cache_left, self.cache_right):
                pass  # served from a prefetched neighbour buffer: no backing-store read
            else:
                self._read(physical)
                self.current_block = physical
        return self._key(item_position)

    def _obtain_and_save(self, item_position: int, first: int, last: int) -> RecordKey:
        """Upstream ``obtain_item_on_disk_and_save`` + ``read_block_and_cache``.

        The prefetch *decision* is a property of the PGM interval relative to the
        midpoint block (logical, exactly as upstream computes it); the fetched
        neighbours are the **physically** adjacent blocks of the midpoint block,
        because upstream reads one contiguous run starting at
        ``block_id - save_left``.
        """
        mid_logical = self._logical_block(item_position)
        mid_physical = self.level.physical_block(mid_logical)
        first_logical = self._logical_block(first)
        last_logical = self._logical_block(last)

        save_left = first_logical < mid_logical
        save_right = last_logical > mid_logical

        fetch = [mid_physical]
        cache_left = mid_physical - 1 if save_left else None
        cache_right = mid_physical + 1 if save_right else None
        if cache_left is not None and not 0 <= cache_left < self.level.block_count:
            cache_left = None  # neighbours outside the level region are not fetched
        if cache_right is not None and not 0 <= cache_right < self.level.block_count:
            cache_right = None
        if cache_left is not None:
            fetch.append(cache_left)
        if cache_right is not None:
            fetch.append(cache_right)

        self._read_run(fetch)
        self.current_block = mid_physical
        self.cache_left = cache_left
        self.cache_right = cache_right
        self.prefetch_counts.append(len(set(fetch)))
        return self._key(item_position)

    # ------------------------------------------------------------- searches

    def lower_bound_pgm_bounded(self, first: int, last: int, key: int) -> int:
        """Upstream ``lower_bound_bl_disk_pgm(cur_level, fh, first, last, key, N, item)``."""
        n_total = self.level.item_count
        if first == last:
            return first
        n = last - first
        if n > 1:
            half = n // 2
            probe = first + half
            self.probes.append(probe)
            if self._obtain_and_save(probe, first, last).value < key:
                first = first + half
            n -= half
        while n > 1:
            half = n // 2
            probe = first + half
            self.probes.append(probe)
            if self._obtain_or_cache(probe).value < key:
                first = first + half
            n -= half
        self.probes.append(first)
        pos = first + (1 if self._obtain_or_cache(first).value < key else 0)
        if pos != n_total:
            self._obtain_or_cache(pos)
        return pos

    def lower_bound_full_level(self, key: int) -> int:
        """Upstream ``lower_bound_bl_disk(cur_level, fh, 0, item_count, key, N, item)``."""
        n_total = self.level.item_count
        first = 0
        last = n_total
        if first == last:
            return first
        n = last - first
        while n > 1:
            half = n // 2
            probe = first + half
            self.probes.append(probe)
            if self._obtain_plain(probe).value < key:
                first = first + half
            n -= half
        self.probes.append(first)
        pos = first + (1 if self._obtain_plain(first).value < key else 0)
        if pos != n_total:
            self._obtain_plain(pos)
        return pos


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


class ReferenceIndex:
    """A LETIndex-Ref-v1 index: levels, layouts, projection and lookup semantics."""

    def __init__(
        self,
        profile: ReferenceProfile,
        levels: Mapping[int, ReferenceLevel],
        used_levels: int,
        projection: SlotProjection,
        permutations: Mapping[int, SeededPermutation],
    ) -> None:
        self.profile = profile
        self._levels = dict(levels)
        self.used_levels = used_levels
        self.projection = projection
        self._permutations = dict(permutations)

    # ------------------------------------------------------------- building

    @classmethod
    def build(
        cls,
        profile: ReferenceProfile,
        levels: Mapping[int, Sequence[tuple[int, str]]],
    ) -> "ReferenceIndex":
        """Materialise levels (loading is our deterministic extension, not upstream)."""
        built: dict[int, ReferenceLevel] = {}
        permutations: dict[int, SeededPermutation] = {}
        declared: set[int] = set()

        for level, items in levels.items():
            if isinstance(level, bool) or not isinstance(level, int):
                raise ReferenceProfileError(f"level must be an integer, got {level!r}")
            if level < profile.min_level:
                raise ReferenceProfileError(
                    f"level {level} is below MIN_LEVEL={profile.min_level}; upstream "
                    "combines levels 0..MIN_LEVEL into the first level"
                )
            declared.add(level)
            pairs = sorted((int(key), str(value)) for key, value in items)
            if len({key for key, _ in pairs}) != len(pairs):
                raise ReferenceProfileError(f"level {level} has duplicate keys")
            if len(pairs) > profile.max_size(level):
                raise ReferenceProfileError(
                    f"level {level} holds {len(pairs)} items but its upstream capacity "
                    f"max_size({level}) = {profile.max_size(level)}"
                )
            block_count = profile.block_count(len(pairs))
            layout_key = (
                profile.layout_key()
                if profile.layout == LAYOUT_STATIC_PRP
                else 0
            )
            permutation = SeededPermutation(
                block_count, _level_seed(layout_key, level)
            )
            permutations[level] = permutation
            placement = (
                permutation.as_tuple()
                if profile.layout == LAYOUT_STATIC_PRP
                else tuple(range(block_count))
            )
            keys = tuple(RecordKey(key) for key, _ in pairs)
            values = tuple(value for _, value in pairs)
            pgm = (
                build_batch_pgm(keys, profile.epsilon)
                if profile.has_index(level) and keys
                else None
            )
            built[level] = ReferenceLevel(
                level=level,
                items_per_block=profile.items_per_block,
                keys=keys,
                values=values,
                pgm=pgm,
                placement=placement,
                backing=profile.level_backing(level),
            )

        non_empty = [level for level, lvl in built.items() if lvl.item_count > 0]
        used_levels = max(non_empty) + 1 if non_empty else profile.min_level
        scanned = sorted(level for level in declared if level < used_levels)
        block_counts = {
            level: built[level].block_count for level in scanned
        }
        projection = SlotProjection.from_block_counts(block_counts)
        return cls(profile, built, used_levels, projection, permutations)

    # ------------------------------------------------------------- accessors

    def levels(self) -> tuple[int, ...]:
        return tuple(sorted(self._levels))

    def level(self, level: int) -> ReferenceLevel:
        try:
            return self._levels[level]
        except KeyError:
            raise ReferenceProfileError(f"index has no level {level}") from None

    def permutation(self, level: int) -> SeededPermutation:
        try:
            return self._permutations[level]
        except KeyError:
            raise ReferenceProfileError(f"index has no permutation for level {level}") from None

    def scanned_levels(self) -> tuple[int, ...]:
        """Upstream scan order: ``MIN_LEVEL .. used_levels-1``."""
        return tuple(range(self.profile.min_level, self.used_levels))

    def block_key_interval(self, level: int, logical_block: int) -> dict:
        """Evaluator-side helper: key/rank interval covered by one logical block."""
        reference = self.level(level)
        start, end = reference.block_range(logical_block)
        return {
            "level": level,
            "block_rank": logical_block,
            "item_start": start,
            "item_end": end,
            "key_lo": reference.keys[start].value if end > start else None,
            "key_hi": reference.keys[end - 1].value if end > start else None,
        }

    # -------------------------------------------------------------- lookups

    def _searcher(self, level: int) -> _LevelSearcher:
        return _LevelSearcher(self.level(level), self.projection)

    def pgm_interval(self, level: int, key: int) -> SearchResult:
        """PGM window over **logical item rank** (pure metadata, emits no event)."""
        reference = self.level(level)
        if reference.pgm is None:
            raise ReferenceProfileError(f"level {level} has no PGM (has_pgm=False)")
        return reference.pgm.search(RecordKey(key))

    def lookup_level_with_interval(
        self, level: int, key: int, first: int, last: int
    ) -> LevelLookupOutcome:
        """PGM-bounded lookup with an explicit interval (hand-derivable traces)."""
        reference = self.level(level)
        if not 0 <= first <= last <= reference.item_count:
            raise ReferenceProfileError(
                f"interval [{first}, {last}) is outside level {level}'s "
                f"{reference.item_count} items"
            )
        searcher = self._searcher(level)
        pos = searcher.lower_bound_pgm_bounded(first, last, key)
        return self._level_outcome(level, key, pos, (first, last), "pgm_bounded", searcher)

    def lookup_level(self, level: int, key: int) -> LevelLookupOutcome:
        """One level lookup exactly as upstream dispatches it."""
        reference = self.level(level)
        if reference.has_pgm:
            interval = self.pgm_interval(level, key)
            return self.lookup_level_with_interval(level, key, interval.lo, interval.hi)
        searcher = self._searcher(level)
        pos = searcher.lower_bound_full_level(key)
        return self._level_outcome(
            level, key, pos, (0, reference.item_count), "full_level", searcher
        )

    def _level_outcome(
        self,
        level: int,
        key: int,
        pos: int,
        interval: tuple[int, int],
        path: str,
        searcher: _LevelSearcher,
    ) -> LevelLookupOutcome:
        reference = self.level(level)
        found = pos != reference.item_count and reference.keys[pos].value == key
        return LevelLookupOutcome(
            level=level,
            pos=pos,
            found=found,
            value=reference.values[pos] if found else None,
            interval=interval,
            path=path,
            events=tuple(searcher.events),
            probes=tuple(searcher.probes),
            fetched_blocks=tuple(searcher.fetched_blocks),
            prefetch_size=(
                searcher.prefetch_counts[0] if searcher.prefetch_counts else None
            ),
        )

    def query(self, key: int) -> ReferenceQueryOutcome:
        """Upstream multi-level lookup: ascending levels, skip empty, hit-and-stop."""
        searched: list[int] = []
        skipped: list[int] = []
        outcomes: list[LevelLookupOutcome] = []
        events: list[tuple[str, int]] = []
        probes: list[tuple[int, int]] = []
        paths: list[tuple[int, str]] = []

        for level in self.scanned_levels():
            reference = self._levels.get(level)
            if reference is None or reference.item_count == 0:
                skipped.append(level)
                continue
            outcome = self.lookup_level(level, key)
            searched.append(level)
            paths.append((level, outcome.path))
            events.extend(outcome.events)
            probes.extend((level, probe) for probe in outcome.probes)
            outcomes.append(outcome)
            if outcome.found:
                return ReferenceQueryOutcome(
                    key=key,
                    found=True,
                    value=outcome.value,
                    hit_level=level,
                    searched_levels=tuple(searched),
                    skipped_empty_levels=tuple(skipped),
                    paths=tuple(paths),
                    events=tuple(events),
                    probes=tuple(probes),
                    level_outcomes=tuple(outcomes),
                )
        return ReferenceQueryOutcome(
            key=key,
            found=False,
            value=None,
            hit_level=None,
            searched_levels=tuple(searched),
            skipped_empty_levels=tuple(skipped),
            paths=tuple(paths),
            events=tuple(events),
            probes=tuple(probes),
            level_outcomes=tuple(outcomes),
        )


# ---------------------------------------------------------------------------
# experiment + adversary-visible trace
# ---------------------------------------------------------------------------

OPERATION_CLASSES_USED = (CLASS_BUILD_INITIAL, CLASS_BUILD_PGM, CLASS_QUERY)


@dataclass(frozen=True)
class ReferenceExperiment:
    """One declared LETIndex-Ref-v1 experiment (public configuration + workload)."""

    profile: ReferenceProfile
    levels: tuple[tuple[int, tuple[tuple[int, str], ...]], ...]
    query_keys: tuple[int, ...]
    workload: Optional[dict] = None
    family: str = PROFILE_NAME

    def level_map(self) -> dict[int, tuple[tuple[int, str], ...]]:
        return {level: items for level, items in self.levels}

    @property
    def all_keys(self) -> tuple[int, ...]:
        return tuple(
            sorted(key for _, items in self.levels for key, _ in items)
        )

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "profile": self.profile.to_dict(),
            "levels": [
                {"level": level, "item_count": len(items)} for level, items in self.levels
            ],
            "query_count": len(self.query_keys),
            "workload": self.workload,
        }


def _summarize_operations(operations: Sequence[ObservedOperation]) -> dict:
    slots = {event.slot_id for operation in operations for event in operation.events}
    return {
        "operation_count": len(operations),
        "query_count": sum(
            1 for op in operations if op.operation_class == CLASS_QUERY
        ),
        "merge_count": 0,
        "total_reads": sum(
            1 for op in operations for event in op.events if event.operation == READ
        ),
        "total_writes": sum(
            1 for op in operations for event in op.events if event.operation == WRITE
        ),
        "distinct_slots": len(slots),
    }


def validate_reference_transcript(payload) -> None:
    """Strict shape validation of a reference observation block.

    The payload must have **exactly** the frozen ``g2a-1`` fields, operation
    fields and event fields — nothing is added for the reference profile — but its
    ``source_schema_version`` names the reference artifact rather than a G1-E run,
    because a LETIndex-Ref-v1 trace is not extracted from a G1-E artifact.  The
    G1-E-source validator ``leakage.validate_transcript`` still pins
    ``source_schema_version`` to ``g1e-1`` and is therefore *not* the right check
    for this producer; the difference between the two validators is exactly that
    one field.
    """
    from .leakage import SCHEMA_VERSION as G2A_SCHEMA_VERSION

    if not isinstance(payload, dict):
        raise LeakageError(
            f"a reference transcript must be a JSON object, got {type(payload).__name__}"
        )
    if set(payload) != TRANSCRIPT_KEYS:
        raise LeakageError(
            f"reference transcript must contain exactly {sorted(TRANSCRIPT_KEYS)}; "
            f"got {sorted(payload)}"
        )
    if payload["schema_version"] != G2A_SCHEMA_VERSION:
        raise LeakageError(
            f"reference transcript schema_version must be {G2A_SCHEMA_VERSION!r}, "
            f"got {payload['schema_version']!r}"
        )
    if payload["source_schema_version"] != REFERENCE_TRANSCRIPT_SOURCE:
        raise LeakageError(
            f"reference transcript source_schema_version must be "
            f"{REFERENCE_TRANSCRIPT_SOURCE!r}, got {payload['source_schema_version']!r}"
        )
    if payload["equality_channel"] != "M0":
        raise LeakageError(
            "a reference transcript is the base M0 view; equality_channel must be 'M0'"
        )
    metadata = payload["run_metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"family", "seed", "config"}:
        raise LeakageError("reference transcript run_metadata fields are frozen")
    if not isinstance(metadata["family"], str) or not metadata["family"]:
        raise LeakageError("run_metadata.family must be a non-empty string")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        raise LeakageError("run_metadata.seed must be an integer")
    config = metadata["config"]
    if not isinstance(config, dict) or set(config) != {"block_capacity", "pgm_epsilon"}:
        raise LeakageError(
            "run_metadata.config must carry exactly block_capacity and pgm_epsilon"
        )
    for key in ("block_capacity", "pgm_epsilon"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise LeakageError(f"run_metadata.config.{key} must be an integer")

    operations = payload["operations"]
    if not isinstance(operations, list) or not operations:
        raise LeakageError("reference transcript operations must be a non-empty list")
    parsed: list[ObservedOperation] = []
    for position, record in enumerate(operations):
        if not isinstance(record, dict):
            raise LeakageError(f"operations[{position}] must be an object")
        if set(record) != OPERATION_KEYS:
            raise LeakageError(
                f"operations[{position}] must contain exactly {sorted(OPERATION_KEYS)}; "
                f"got {sorted(record)}"
            )
        if record["class"] not in OPERATION_CLASSES_USED:
            raise LeakageError(
                f"operations[{position}].class must be one of "
                f"{list(OPERATION_CLASSES_USED)}, got {record['class']!r}"
            )
        if record["index"] != position:
            raise LeakageError(
                f"operations[{position}].index must equal its position"
            )
        events = record["events"]
        if not isinstance(events, list):
            raise LeakageError(f"operations[{position}].events must be a list")
        observed = []
        for seq, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {
                "seq", "operation", "slot_id"
            }:
                raise LeakageError(
                    "reference events must carry exactly seq/operation/slot_id"
                )
            if event["operation"] not in EVENT_OPERATIONS:
                raise LeakageError(
                    f"event operation must be one of {EVENT_OPERATIONS}"
                )
            for key in ("seq", "slot_id"):
                if isinstance(event[key], bool) or not isinstance(event[key], int):
                    raise LeakageError(f"event {key} must be an integer")
            if event["seq"] != seq:
                raise LeakageError("event seq must be 0-based and ascending")
            observed.append(ObservedEvent(
                seq=event["seq"], operation=event["operation"], slot_id=event["slot_id"]
            ))
        for key in ("read_count", "write_count"):
            if isinstance(record[key], bool) or not isinstance(record[key], int):
                raise LeakageError(f"operations[{position}].{key} must be an integer")
        operation = ObservedOperation(
            index=record["index"],
            operation_class=record["class"],
            events=tuple(observed),
            read_count=record["read_count"],
            write_count=record["write_count"],
        )
        reads = sum(1 for event in operation.events if event.operation == READ)
        writes = operation.event_count - reads
        if reads != operation.read_count or writes != operation.write_count:
            raise LeakageError(
                f"operations[{position}] read/write counts disagree with its events"
            )
        parsed.append(operation)

    summary = payload["summary"]
    if not isinstance(summary, dict) or set(summary) != set(SUMMARY_KEYS):
        raise LeakageError(
            f"reference transcript summary must contain exactly {sorted(SUMMARY_KEYS)}"
        )
    for key in SUMMARY_KEYS:
        if isinstance(summary[key], bool) or not isinstance(summary[key], int):
            raise LeakageError(f"summary.{key} must be a JSON integer")
    expected = _summarize_operations(tuple(parsed))
    for key in SUMMARY_KEYS:
        if summary[key] != expected[key]:
            raise LeakageError(
                f"summary.{key} is {summary[key]!r} but the events say {expected[key]!r}"
            )


class ReferenceTraceBuilder:
    """Builds the adversary-visible g2a-1-shaped trace of a reference experiment.

    Only observed physical operations are recorded: ``READ``/``WRITE`` with the
    projected ``SlotId``.  Logical level numbers, block ranks, item ranks, keys and
    values stay evaluator-side (see ``ref_truth``).
    """

    def __init__(self, experiment: ReferenceExperiment) -> None:
        self.experiment = experiment
        self.index = ReferenceIndex.build(experiment.profile, experiment.level_map())
        self._operations: list[ObservedOperation] = []
        self._outcomes: list[ReferenceQueryOutcome] = []
        self._loaded = False

    # ---------------------------------------------------------------- loading

    def load(self) -> "ReferenceTraceBuilder":
        """Materialise every level: one WRITE per occupied physical block."""
        if self._loaded:
            raise ReferenceProfileError("levels are already loaded")
        for level in self.index.levels():
            reference = self.index.level(level)
            if reference.item_count == 0:
                continue
            events = tuple(
                ObservedEvent(seq=seq, operation=WRITE, slot_id=slot)
                for seq, slot in enumerate(
                    self.index.projection.slot(level, physical).value
                    for physical in sorted(reference.physical_blocks())
                )
            )
            self._operations.append(ObservedOperation(
                index=len(self._operations),
                operation_class=CLASS_BUILD_INITIAL,
                events=events,
                read_count=0,
                write_count=len(events),
            ))
            if reference.has_pgm:
                # The PGM is trusted in-memory metadata upstream; building it emits
                # no backing-store access that this profile models (documented).
                self._operations.append(ObservedOperation(
                    index=len(self._operations),
                    operation_class=CLASS_BUILD_PGM,
                    events=(),
                    read_count=0,
                    write_count=0,
                ))
        self._loaded = True
        return self

    # ---------------------------------------------------------------- queries

    def run_queries(self) -> "ReferenceTraceBuilder":
        if not self._loaded:
            self.load()
        for key in self.experiment.query_keys:
            outcome = self.index.query(key)
            self._outcomes.append(outcome)
            events = tuple(
                ObservedEvent(seq=seq, operation=operation, slot_id=slot)
                for seq, (operation, slot) in enumerate(outcome.events)
            )
            self._operations.append(ObservedOperation(
                index=len(self._operations),
                operation_class=CLASS_QUERY,
                events=events,
                read_count=sum(1 for e in events if e.operation == READ),
                write_count=sum(1 for e in events if e.operation == WRITE),
            ))
        return self

    @property
    def operations(self) -> tuple[ObservedOperation, ...]:
        return tuple(self._operations)

    @property
    def query_outcomes(self) -> tuple[ReferenceQueryOutcome, ...]:
        return tuple(self._outcomes)

    def query_operations(self) -> tuple[ObservedOperation, ...]:
        return tuple(op for op in self._operations if op.operation_class == CLASS_QUERY)

    # -------------------------------------------------------------- transcript

    def transcript_payload(self) -> dict:
        profile = self.experiment.profile
        payload = {
            "schema_version": "g2a-1",
            "source_schema_version": REFERENCE_TRANSCRIPT_SOURCE,
            "equality_channel": "M0",
            "run_metadata": {
                "family": self.experiment.family,
                "seed": profile.seed,
                "config": {
                    "block_capacity": profile.items_per_block,
                    "pgm_epsilon": profile.epsilon,
                },
            },
            "operations": [op.to_dict() for op in self._operations],
            "summary": _summarize_operations(self._operations),
        }
        validate_reference_transcript(payload)
        return payload

    def transcript(self) -> AdversaryTranscript:
        """The frozen G2-A observation abstraction over this reference run."""
        payload = self.transcript_payload()
        return AdversaryTranscript(
            schema_version=payload["schema_version"],
            source_schema_version=payload["source_schema_version"],
            run_metadata=payload["run_metadata"],
            operations=tuple(self._operations),
            summary=payload["summary"],
            equality_channel=payload["equality_channel"],
        )

    # --------------------------------------------------------------- artifact

    def artifact(self) -> dict:
        profile = self.experiment.profile
        level_metadata = [
            self.index.level(level).to_dict() for level in self.index.levels()
            if self.index.level(level).item_count > 0
        ]
        for entry in level_metadata:
            entry.pop("placement", None)  # the resulting layout is not declared public
        return {
            "schema_version": REFERENCE_ARTIFACT_SCHEMA_VERSION,
            "profile": PROFILE_NAME,
            "upstream": {
                "repo": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "pinned": True,
            },
            "classification": {
                **profile.classification(),
                "slot_projection": EXPERIMENTAL_EXTENSION,
                "loader": EXPERIMENTAL_EXTENSION,
                "workload": EXPERIMENTAL_EXTENSION,
                "legacy_simulator_settings": list(LEGACY_SIMULATOR_SETTINGS),
            },
            "public_config": {
                **profile.public_dict(),
                "used_levels": self.index.used_levels,
                "scanned_levels": list(self.index.scanned_levels()),
                "levels": level_metadata,
            },
            "reproducibility": {
                "scenario": "default-scenario-v1",
                "analysis_seed": profile.seed,
                "prp_key": profile.prp_key,
                "layout": profile.layout,
                "note": (
                    "evaluator-side reproducibility record.  analysis_seed is the value "
                    "already published in the frozen g2a-1 run metadata (run_metadata.seed); "
                    "prp_key and the workload draw seed are NOT part of the attack-facing "
                    "configuration and must not be handed to an adversary-facing consumer"
                ),
            },
            "workload": self.experiment.workload
            or {"mode": None, "provenance": "unspecified"},
            "transcript": self.transcript_payload(),
        }

    def to_dict(self) -> dict:
        return self.artifact()


def dumps_reference_artifact(payload: Mapping) -> str:
    """Deterministic JSON text for a reference artifact (no timestamps, no paths)."""
    return json.dumps(dict(payload), indent=2)


def write_reference_artifact_json(payload: Mapping, path: str | Path) -> dict:
    text = dumps_reference_artifact(payload)
    Path(path).write_text(text + "\n", encoding="utf-8")
    return dict(payload)


def dumps_reference_transcript(transcript: AdversaryTranscript) -> str:
    """Deterministic ``g2a-1`` transcript bytes (delegates to the frozen writer)."""
    return dumps_transcript(transcript)


# ---------------------------------------------------------------------------
# default scenario + CLI
# ---------------------------------------------------------------------------

#: Default deterministic scenario: four levels, three of them below
#: ``MIN_INDEX_LEVEL`` (full-level binary search) and one indexed (PGM-bounded).
DEFAULT_LEVEL_SIZES = ((MIN_LEVEL, 120), (4, 600), (5, 700), (6, 900))


def default_scenario(
    *,
    layout: str = LAYOUT_ORDERED,
    seed: int = 0,
    prp_key: Optional[int] = None,
    query_count: int = 8,
    workload_mode: str = "record_uniform",
    profile: Optional[ReferenceProfile] = None,
    level_sizes: Sequence[tuple[int, int]] = DEFAULT_LEVEL_SIZES,
) -> ReferenceExperiment:
    """A deterministic reference experiment (our loader + our workload generator).

    ``seed`` is the published analysis seed; ``prp_key`` is the evaluator-side layout
    key and is required (explicitly, never derived from ``seed``) when
    ``layout=static_prp``.
    """
    from .ref_workload import generate_passive_workload

    profile = profile or ReferenceProfile.reference(
        layout=layout, seed=seed, prp_key=prp_key
    )
    levels = []
    for level, count in level_sizes:
        items = tuple(
            (level * 1_000_000 + 3 * index, f"L{level}:{3 * index}")
            for index in range(count)
        )
        levels.append((level, items))
    keys = sorted(key for _, items in levels for key, _ in items)
    workload = generate_passive_workload(
        workload_mode, seed=seed, records=keys, count=query_count
    )
    return ReferenceExperiment(
        profile=profile,
        levels=tuple(levels),
        query_keys=workload.queries,
        workload=workload.public_dict(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m enhanced_letindex.letindex_ref",
        description=(
            "Generate a LETIndex-Ref-v1 reference run: upstream-faithful lookup "
            "semantics over a deterministic loader and passive workload, exported as "
            "an adversary-visible g2a-1 observation block plus public configuration.  "
            "No attack, recovery or defence is implemented."
        ),
    )
    parser.add_argument("--layout", default=LAYOUT_ORDERED, choices=list(LAYOUT_MODES))
    parser.add_argument("--seed", type=int, default=0,
                        help="analysis seed (published in the g2a-1 run metadata)")
    parser.add_argument("--prp-key", type=int, default=None,
                        help="layout key for --layout static_prp (required, "
                             "evaluator-side, never attack-facing)")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--workload", default="record_uniform",
                        choices=["record_uniform", "domain_uniform", "data_empirical"])
    parser.add_argument("--variant-label", default=None,
                        help="required when deviating from the frozen upstream defaults")
    parser.add_argument("--items-per-block", type=int, default=None,
                        help="experimental variant only (default: frozen 256)")
    parser.add_argument("--epsilon", type=int, default=None,
                        help="experimental variant only (default: upstream 64)")
    parser.add_argument("--out", default="g2b0_reference.json",
                        help="output path for the reference artifact")
    parser.add_argument("--transcript-out", default=None,
                        help="optional path for the g2a-1 observation block alone")
    parser.add_argument("--truth-out", default=None,
                        help="optional path for the privileged evaluator truth artifact")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.layout == LAYOUT_STATIC_PRP and args.prp_key is None:
        print(
            "error: --layout static_prp requires --prp-key (evaluator-side layout key; "
            "it is never part of the attack-facing configuration)",
        )
        return 2
    if args.layout == LAYOUT_ORDERED and args.prp_key is not None:
        print("error: --prp-key is only meaningful with --layout static_prp")
        return 2
    if args.variant_label:
        profile = ReferenceProfile.experimental_variant(
            args.variant_label,
            layout=args.layout,
            seed=args.seed,
            prp_key=args.prp_key,
            items_per_block=args.items_per_block,
            epsilon=args.epsilon,
        )
    else:
        if args.items_per_block is not None or args.epsilon is not None:
            print(
                "error: deviating from the frozen upstream defaults requires "
                "--variant-label",
            )
            return 2
        profile = ReferenceProfile.reference(
            layout=args.layout, seed=args.seed, prp_key=args.prp_key
        )

    experiment = default_scenario(
        layout=profile.layout, seed=profile.seed, query_count=args.queries,
        workload_mode=args.workload, profile=profile,
    )
    builder = ReferenceTraceBuilder(experiment).load().run_queries()
    artifact = builder.artifact()
    write_reference_artifact_json(artifact, args.out)

    summary = artifact["transcript"]["summary"]
    print(f"profile={PROFILE_NAME} layout={profile.layout} seed={profile.seed} "
          f"items_per_block={profile.items_per_block} epsilon={profile.epsilon}")
    print(f"levels={len(artifact['public_config']['levels'])} "
          f"used_levels={artifact['public_config']['used_levels']} "
          f"queries={summary['query_count']}")
    print(f"reads={summary['total_reads']} writes={summary['total_writes']} "
          f"distinct_slots={summary['distinct_slots']} "
          f"operations={summary['operation_count']}")
    print(f"wrote {args.out}")

    if args.transcript_out:
        from .leakage import write_transcript_json

        write_transcript_json(builder.transcript(), args.transcript_out)
        print(f"wrote {args.transcript_out}")
    if args.truth_out:
        from .ref_truth import ReferenceEvaluatorTruth

        ReferenceEvaluatorTruth(builder.index).write_json(args.truth_out)
        print(f"wrote {args.truth_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
