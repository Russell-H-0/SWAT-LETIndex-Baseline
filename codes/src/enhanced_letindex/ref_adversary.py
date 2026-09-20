"""The attack-facing interface of a reference experiment (G2-B0 freeze).

G2-B1 may consume **only**:

```text
AdversaryTranscript / QUERY operation traces
+ public experiment configuration that is explicitly declared adversary-known
+ the optional opaque M1 equality classes when M1 is selected
```

It may not consume evaluator truth, plaintext keys, logical ranks, level ids,
block ids, the workload script or any result.  This module enforces that
structurally: it builds the input out of exactly two sections of a reference
artifact (``transcript`` and ``public_config``) and refuses anything else, and it
**does not import** ``ref_truth`` (a test asserts that).

No recovery logic lives here — this milestone adds no attack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional

from .leakage import (
    EVENT_KEYS,
    OPERATION_ALL_KEYS,
    RUN_METADATA_KEY_SET,
    TRANSCRIPT_KEYS,
    AdversaryTranscript,
    CONFIG_KEY_SET,
    SUMMARY_KEY_SET,
    ObservedOperation,
)

__all__ = [
    "PUBLIC_CONFIG_KEYS",
    "LEVEL_METADATA_KEYS",
    "ARTIFACT_METADATA_KEYS",
    "WORKLOAD_PUBLIC_KEYS",
    "ADVERSARY_INPUT_KEYS",
    "ALLOWED_FIELD_UNIVERSE",
    "FORBIDDEN_FIELD_NAMES",
    "AdversaryInputError",
    "AdversaryInput",
    "scan_forbidden_fields",
]

#: Everything the adversary is explicitly told about the experiment.
#:
#: ``seed`` and ``prp_key`` are deliberately **absent**:
#:
#: * the analysis seed is already published in the frozen ``g2a-1`` run metadata
#:   (``run_metadata.seed``), so re-stating it here adds nothing;
#: * the layout key drives the ``static_prp`` permutation.  Handing it (or anything
#:   the permutation is derived from) to the attack would let a G2-B1 consumer
#:   recompute the hidden placement from its own input, which would make "recover the
#:   static PRP from the trace" experimentally vacuous.
#:
#: The attack still knows the layout *mode*, the level geometry and the block counts.
PUBLIC_CONFIG_KEYS = frozenset({
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
    "used_levels",
    "scanned_levels",
    "levels",
})
LEVEL_METADATA_KEYS = frozenset({
    "level", "item_count", "block_count", "backing", "has_pgm", "items_per_block",
})

ADVERSARY_INPUT_KEYS = frozenset({"transcript", "public_config", "equality_channel"})

#: The complete set of field names an adversary-facing payload may contain.  It
#: covers the frozen ``g2a-1`` transcript fields, the declared public
#: configuration, level metadata, the reference artifact's own metadata sections
#: and the adversary-safe workload descriptor.  Anything else — in particular any
#: name in :data:`FORBIDDEN_FIELD_NAMES` — is reported by
#: :func:`scan_forbidden_fields`.
ARTIFACT_METADATA_KEYS = frozenset({
    "profile", "upstream", "repo", "commit", "pinned", "classification",
    "lookup_semantics", "slot_projection", "layout", "geometry", "loader", "workload",
    "legacy_simulator_settings", "reproducibility", "scenario", "query_count",
    "note", "truth", "analysis_seed", "prp_key",
})
WORKLOAD_PUBLIC_KEYS = frozenset({
    "mode", "seed", "count", "distinct_count", "record_count", "domain", "provenance",
    "classification",
})

ALLOWED_FIELD_UNIVERSE = (
    set(TRANSCRIPT_KEYS)
    | set(OPERATION_ALL_KEYS)
    | set(EVENT_KEYS)
    | set(SUMMARY_KEY_SET)
    | set(RUN_METADATA_KEY_SET)
    | set(CONFIG_KEY_SET)
    | set(PUBLIC_CONFIG_KEYS)
    | set(LEVEL_METADATA_KEYS)
    | set(ADVERSARY_INPUT_KEYS)
    | set(ARTIFACT_METADATA_KEYS)
    | set(WORKLOAD_PUBLIC_KEYS)
    | {"schema_version", "source_schema_version"}
)

# a few names that must never appear in an adversary-facing payload
FORBIDDEN_FIELD_NAMES = frozenset({
    "keys", "values", "key", "value", "item_rank", "hit_item_rank", "hit_block_rank",
    "block_rank", "physical_block_offset", "placement", "permutation", "prp",
    "slot_of", "locate", "probe", "probes", "target", "oracle", "truth", "script",
    "result", "verification", "region_start", "backing_store", "level_keys",
})


class AdversaryInputError(Exception):
    """The requested adversary-facing input would violate the frozen separation."""


def _collect_keys(payload, accumulator: set) -> set:
    if isinstance(payload, dict):
        for key, value in payload.items():
            accumulator.add(key)
            _collect_keys(value, accumulator)
    elif isinstance(payload, list):
        for item in payload:
            _collect_keys(item, accumulator)
    return accumulator


def scan_forbidden_fields(payload: Mapping | list) -> tuple[str, ...]:
    """Names that must never appear in an adversary-facing payload."""
    found = _collect_keys(payload, set())
    return tuple(sorted(
        name for name in found
        if name not in ALLOWED_FIELD_UNIVERSE or name in FORBIDDEN_FIELD_NAMES
    ))


@dataclass(frozen=True)
class AdversaryInput:
    """The frozen attack-facing view: transcript + declared public config (+ M1)."""

    transcript: AdversaryTranscript
    public_config: Mapping
    equality_channel: str = "M0"
    source_profile: str = "LETINDEX-REF-V1"

    def __post_init__(self) -> None:
        if not isinstance(self.public_config, Mapping):
            raise AdversaryInputError("public_config must be a mapping")
        extra = sorted(set(self.public_config) - PUBLIC_CONFIG_KEYS)
        if extra:
            raise AdversaryInputError(
                f"public_config contains fields that are not declared adversary-known: "
                f"{extra}  (the analysis seed lives in the g2a-1 run metadata, and the "
                "layout key and any placement/permutation data must never reach an "
                "adversary-facing consumer)"
            )
        forbidden = scan_forbidden_fields(self.to_dict())
        if forbidden:
            raise AdversaryInputError(
                f"adversary input would expose privileged fields: {list(forbidden)}"
            )

    # ------------------------------------------------------------- accessors

    def query_traces(self) -> tuple[ObservedOperation, ...]:
        """The QUERY operation traces (the only trace class an attack may read)."""
        return self.transcript.query_operations

    def operation_traces(self) -> tuple[ObservedOperation, ...]:
        return self.transcript.operations

    @property
    def summary(self) -> Mapping:
        return self.transcript.summary

    # ------------------------------------------------------------------ serde

    def to_dict(self) -> dict:
        return {
            "equality_channel": self.equality_channel,
            "public_config": dict(self.public_config),
            "transcript": self.transcript.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_reference_artifact(
        cls,
        artifact: Mapping,
        *,
        equality_labels: Optional[Mapping[int, str]] = None,
        expected_schema: str = "g2b0-ref-1",
    ) -> "AdversaryInput":
        """Build the attack-facing view from a reference artifact (two sections only)."""
        if not isinstance(artifact, Mapping):
            raise AdversaryInputError("a reference artifact must be a mapping")
        if artifact.get("schema_version") != expected_schema:
            raise AdversaryInputError(
                f"unsupported reference artifact schema_version "
                f"{artifact.get('schema_version')!r}; expected {expected_schema!r}"
            )
        transcript_section = artifact.get("transcript")
        public_config = artifact.get("public_config")
        if not isinstance(transcript_section, Mapping) or not isinstance(public_config, Mapping):
            raise AdversaryInputError(
                "a reference artifact must carry 'transcript' and 'public_config' objects"
            )
        transcript = AdversaryTranscript(
            schema_version=str(transcript_section["schema_version"]),
            source_schema_version=str(transcript_section["source_schema_version"]),
            run_metadata=dict(transcript_section["run_metadata"]),
            operations=tuple(
                ObservedOperation.from_dict(record)
                for record in transcript_section["operations"]
            ),
            summary={key: transcript_section["summary"][key] for key in SUMMARY_KEY_SET},
            equality_channel=str(transcript_section["equality_channel"]),
        )
        if equality_labels is not None:
            transcript = transcript.with_equality_channel(equality_labels)
        return cls(
            transcript=transcript,
            public_config=dict(public_config),
            equality_channel=transcript.equality_channel,
            source_profile=str(artifact.get("profile", "LETINDEX-REF-V1")),
        )
