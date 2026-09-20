"""G2-A adversary observation transcript over an exported G1-E artifact.

This module implements the **observation layer** frozen by
``decisions/0008-g2-a-observation-transcript.md``.  It converts an exported G1-E
run (``schema_version`` ``g1e-1``) into a versioned, primitive-only adversary
transcript (``schema_version`` ``g2a-1``) that contains **only** the fields the
adversary is allowed to observe:

```text
per operation:  index, class, events[] {seq, operation, slot_id}, read_count, write_count
per event:      seq, operation (READ/WRITE), slot_id
run metadata:   family, seed, block_capacity, pgm_epsilon
summary:        operation_count, query_count, merge_count, total_reads,
                total_writes, distinct_slots
```

Everything else in the G1-E artifact — plaintext keys/values, oracle answers,
``hit_ref`` / ``searched_refs``, logical ``BlockId`` / ``LevelId``, per-operation
results, verification summaries and the workload script — is evaluator-only
ground truth and is **never** copied into the transcript.

Layers:

* :class:`AdversaryTranscript` / :class:`ObservedOperation` / :class:`ObservedEvent`
  — the frozen transcript unit;
* :func:`extract_transcript` — whitelisting extraction from a G1-E artifact;
* :func:`validate_g1e_artifact` / :func:`validate_transcript` — loud schema checks;
* :class:`PrivilegedEvaluator` — the explicitly privileged namespace holding the
  only helper that consumes ground truth (the optional M1 equality classes);
* ``dumps_*`` / ``write_*`` — deterministic JSON export for transcript + metrics;
* :func:`main` — ``python -m enhanced_letindex.leakage`` CLI.

G2-A is **not** an attack.  It implements no CDF inversion, no passive recovery,
no trace reconstruction, no M0/M1 correlation attack, no attack budget/success
metric and no defence mechanism, and it changes no accepted G1 semantics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "SOURCE_OPERATION_KINDS",
    "EVENT_OPERATIONS",
    "EQUALITY_CHANNEL_M0",
    "EQUALITY_CHANNEL_M1",
    "EQUALITY_CLASS_PATTERN",
    "SUMMARY_KEYS",
    "RUN_METADATA_KEYS",
    "CONFIG_KEYS",
    "TRANSCRIPT_KEYS",
    "OPERATION_KEYS",
    "OPERATION_OPTIONAL_KEYS",
    "OPERATION_ALL_KEYS",
    "LeakageError",
    "ObservedEvent",
    "ObservedOperation",
    "AdversaryTranscript",
    "validate_g1e_artifact",
    "validate_transcript",
    "extract_transcript",
    "dumps_transcript",
    "write_transcript_json",
    "dumps_metrics",
    "write_metrics_json",
    "PrivilegedEvaluator",
    "build_parser",
    "main",
]

SCHEMA_VERSION = "g2a-1"
SOURCE_SCHEMA_VERSION = "g1e-1"
METRICS_SCHEMA_VERSION = SCHEMA_VERSION

# operation classes visible to the adversary: the coarse experiment segmentation
# already implied by the G1-E operation kinds (never plaintext query identity)
CLASS_BUILD_INITIAL = "BUILD_INITIAL"
CLASS_QUERY = "QUERY"
CLASS_UPSERT = "UPSERT"
CLASS_BUILD_PGM = "BUILD_PGM"
CLASS_MERGE = "MERGE"
SOURCE_OPERATION_KINDS = (
    CLASS_BUILD_INITIAL,
    CLASS_QUERY,
    CLASS_UPSERT,
    CLASS_BUILD_PGM,
    CLASS_MERGE,
)

EVENT_OPERATIONS = ("READ", "WRITE")
EVENT_KEYS = frozenset({"seq", "operation", "slot_id"})

EQUALITY_CHANNEL_M0 = "M0"  # equality hidden: base transcript, no labels at all
EQUALITY_CHANNEL_M1 = "M1"  # equality visible: opaque equality-class labels only
EQUALITY_CHANNELS = (EQUALITY_CHANNEL_M0, EQUALITY_CHANNEL_M1)

#: An M1 equality class must be opaque: never a plaintext key, never a value.
EQUALITY_CLASS_PATTERN = re.compile(r"^QClass[0-9]+$")

SUMMARY_KEYS = (
    "operation_count",
    "query_count",
    "merge_count",
    "total_reads",
    "total_writes",
    "distinct_slots",
)

RUN_METADATA_KEYS = ("family", "seed", "config")
CONFIG_KEYS = ("block_capacity", "pgm_epsilon")

# Documented transcript schema key sets.  The two schemas deliberately use
# different field names for the operation class — G1-E source records carry
# ``kind``, G2-A transcript records carry ``class`` — and neither field is
# accepted as an alias for the other.  A transcript is also rejected if it
# carries any undocumented field, so a schema-valid-looking file can never smuggle
# a privileged field (``key``, ``script``, ``level_id``, ...) past validation.
TRANSCRIPT_KEYS = frozenset({
    "schema_version",
    "source_schema_version",
    "equality_channel",
    "run_metadata",
    "operations",
    "summary",
})
OPERATION_KEYS = frozenset({
    "index",
    "class",
    "events",
    "read_count",
    "write_count",
})
OPERATION_OPTIONAL_KEYS = frozenset({"query_equality_class"})
OPERATION_ALL_KEYS = OPERATION_KEYS | OPERATION_OPTIONAL_KEYS
SUMMARY_KEY_SET = frozenset(SUMMARY_KEYS)
RUN_METADATA_KEY_SET = frozenset(RUN_METADATA_KEYS)
CONFIG_KEY_SET = frozenset(CONFIG_KEYS)


class LeakageError(Exception):
    """A G1-E artifact, a transcript or a request violates the G2-A contract."""


# ---------------------------------------------------------------------------
# frozen transcript unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedEvent:
    """One adversary-visible physical event: ``seq``, ``operation``, ``slot_id``."""

    seq: int
    operation: str
    slot_id: int

    def to_dict(self) -> dict:
        return {"seq": self.seq, "operation": self.operation, "slot_id": self.slot_id}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ObservedEvent":
        return cls(
            seq=_require_int(data, "seq", "event"),
            operation=_require_event_operation(data),
            slot_id=_require_int(data, "slot_id", "event"),
        )


@dataclass(frozen=True)
class ObservedOperation:
    """One observed operation: its class, physical event window and volumes.

    ``query_equality_class`` is the **optional** M1 side channel.  It is absent
    (``None``) in the base M0 transcript, and when present it is an opaque
    ``QClassN`` label — never a plaintext key.
    """

    index: int
    operation_class: str
    events: tuple[ObservedEvent, ...]
    read_count: int
    write_count: int
    query_equality_class: Optional[str] = None

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def slots(self) -> tuple[int, ...]:
        """The ordered physical slot sequence of this operation."""
        return tuple(event.slot_id for event in self.events)

    def to_dict(self) -> dict:
        payload = {
            "index": self.index,
            "class": self.operation_class,
            "events": [event.to_dict() for event in self.events],
            "read_count": self.read_count,
            "write_count": self.write_count,
        }
        if self.query_equality_class is not None:
            payload["query_equality_class"] = self.query_equality_class
        return payload

    @classmethod
    def from_dict(cls, data: Mapping) -> "ObservedOperation":
        if not isinstance(data, Mapping):
            raise LeakageError(f"transcript operations entries must be objects, got {data!r}")
        _reject_extra_fields(data, OPERATION_ALL_KEYS, "transcript operation")
        events = data.get("events")
        if not isinstance(events, list):
            raise LeakageError("transcript operations[].events must be a list")
        # Field *presence* is part of the frozen schema: an absent field and an
        # explicit JSON null are different states.  M0 requires the field to be
        # absent everywhere; M1 requires it on every QUERY operation with an opaque
        # ``QClassN`` value, and absent on every other operation.  ``null`` is
        # therefore never normalised to "absent".
        if "query_equality_class" in data:
            raw_label = data["query_equality_class"]
            if raw_label is None:
                raise LeakageError(
                    "transcript operation carries an explicit JSON null for "
                    "'query_equality_class'; an absent field and an explicit null are "
                    "different states — omit the field (M0) or attach an opaque QClassN "
                    "label (M1)"
                )
            query_equality_class = _require_opaque_class(raw_label)
        else:
            query_equality_class = None
        operation = cls(
            index=_require_int(data, "index", "operation"),
            operation_class=_require_transcript_class(data, "transcript operation"),
            events=tuple(_require_event(event) for event in events),
            read_count=_require_int(data, "read_count", "operation"),
            write_count=_require_int(data, "write_count", "operation"),
            query_equality_class=query_equality_class,
        )
        _check_event_integrity(operation, "transcript")
        return operation


@dataclass(frozen=True)
class AdversaryTranscript:
    """The frozen G2-A observation transcript (deterministic, primitive-only)."""

    schema_version: str
    source_schema_version: str
    run_metadata: dict
    operations: tuple[ObservedOperation, ...]
    summary: dict
    equality_channel: str = EQUALITY_CHANNEL_M0

    # ---------------------------------------------------------------- queries

    def operation(self, index: int) -> ObservedOperation:
        for candidate in self.operations:
            if candidate.index == index:
                return candidate
        raise LeakageError(f"transcript has no operation with index {index}")

    def of_class(self, operation_class: str) -> tuple[ObservedOperation, ...]:
        return tuple(op for op in self.operations if op.operation_class == operation_class)

    @property
    def query_operations(self) -> tuple[ObservedOperation, ...]:
        return self.of_class(CLASS_QUERY)

    @property
    def merge_operations(self) -> tuple[ObservedOperation, ...]:
        return self.of_class(CLASS_MERGE)

    # -------------------------------------------------- optional side channel

    def with_equality_channel(
        self,
        labels: Mapping[int, str],
        *,
        channel: str = EQUALITY_CHANNEL_M1,
    ) -> "AdversaryTranscript":
        """Return a copy annotated with opaque M1 equality classes.

        ``labels`` maps every ``QUERY`` operation index to exactly one opaque
        class label (``QClassN``); M1 is the model where query equality is fully
        visible, so the mapping must be a **complete partition** of the QUERY
        operations (vacuously empty when the transcript has no query).  Each QUERY
        entry is read by indexed access and must hold a valid opaque label — a
        present entry whose value is ``None`` is rejected, never treated as
        "unlabelled".  This is a deliberately separate, evaluator-requested side
        channel: it is never requested by default, the annotation is rejected on any
        non-``QUERY`` operation, and the label must match
        :data:`EQUALITY_CLASS_PATTERN` so a plaintext key can never be attached.
        """
        if channel != EQUALITY_CHANNEL_M1:
            raise LeakageError(
                f"the only supported equality annotation channel is "
                f"{EQUALITY_CHANNEL_M1!r}; got {channel!r}"
            )
        known = {op.index for op in self.operations}
        unknown = sorted(index for index in labels if index not in known)
        if unknown:
            raise LeakageError(
                f"equality classes reference unknown operation indices {unknown}"
            )
        query_indices = {
            op.index for op in self.operations if op.operation_class == CLASS_QUERY
        }
        misplaced = sorted(
            index for index in labels if index in known and index not in query_indices
        )
        if misplaced:
            raise LeakageError(
                f"equality classes may only annotate {CLASS_QUERY} operations; "
                f"got indices {misplaced}"
            )
        provided = set(labels)
        if provided != query_indices:
            missing = sorted(query_indices - provided)
            raise LeakageError(
                f"an {EQUALITY_CHANNEL_M1} equality partition must label every "
                f"{CLASS_QUERY} operation exactly once; missing labels for "
                f"{missing} (received {sorted(provided)})"
            )
        annotated: list[ObservedOperation] = []
        for operation in self.operations:
            if operation.operation_class != CLASS_QUERY:
                annotated.append(operation)
                continue
            # Indexed access, never ``labels.get()``: the completeness check above
            # only proves which indices are present.  A present entry whose value is
            # ``None`` (or any other non-opaque value) must reach
            # ``_require_opaque_class`` and fail, exactly as the serialized boundary
            # requires — presence and value validity are not inferable from a
            # ``.get()`` default.
            annotated.append(
                replace(
                    operation,
                    query_equality_class=_require_opaque_class(labels[operation.index]),
                )
            )
        return replace(self, equality_channel=channel, operations=tuple(annotated))

    # ------------------------------------------------------------- export

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "equality_channel": self.equality_channel,
            "run_metadata": _copy_run_metadata(self.run_metadata),
            "operations": [operation.to_dict() for operation in self.operations],
            "summary": {key: self.summary[key] for key in SUMMARY_KEYS},
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "AdversaryTranscript":
        validate_transcript(data)
        operations = tuple(
            ObservedOperation.from_dict(entry) for entry in data["operations"]
        )
        return cls(
            schema_version=str(data["schema_version"]),
            source_schema_version=str(data["source_schema_version"]),
            equality_channel=str(data["equality_channel"]),
            run_metadata=_copy_run_metadata(data["run_metadata"]),
            operations=operations,
            summary={key: data["summary"][key] for key in SUMMARY_KEYS},
        )


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _require_int(data: Mapping, key: str, where: str) -> int:
    if key not in data:
        raise LeakageError(f"{where} is missing required field {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeakageError(f"{where} field {key!r} must be an integer, got {value!r}")
    return value


def _require_event_operation(data: Mapping) -> str:
    value = data.get("operation")
    if value not in EVENT_OPERATIONS:
        raise LeakageError(
            f"event operation must be one of {EVENT_OPERATIONS}, got {value!r}"
        )
    return value


def _require_source_kind(data: Mapping, where: str) -> str:
    """A G1-E **source** operation must carry its own field name: ``kind``.

    The G2-A transcript field ``class`` is never accepted as an alias here.
    """
    if "kind" not in data:
        raise LeakageError(
            f"{where} is missing the required G1-E source field 'kind' "
            f"(a source operation record must use 'kind'; the transcript field "
            f"'class' is not accepted as an alias)"
        )
    value = data["kind"]
    if value not in SOURCE_OPERATION_KINDS:
        raise LeakageError(
            f"{where} kind must be one of {SOURCE_OPERATION_KINDS}, got {value!r}"
        )
    return value


def _require_transcript_class(data: Mapping, where: str) -> str:
    """A G2-A **transcript** operation must carry its own field name: ``class``.

    The G1-E source field ``kind`` is never accepted as an alias here.
    """
    if "class" not in data:
        raise LeakageError(
            f"{where} is missing the required transcript field 'class' "
            f"(a transcript operation must use 'class'; the G1-E source field "
            f"'kind' is not accepted as an alias)"
        )
    value = data["class"]
    if value not in SOURCE_OPERATION_KINDS:
        raise LeakageError(
            f"{where} class must be one of {SOURCE_OPERATION_KINDS}, got {value!r}"
        )
    return value


def _reject_extra_fields(data: Mapping, allowed, where: str) -> None:
    """Reject undocumented fields instead of silently dropping them."""
    extra = sorted(set(data) - set(allowed))
    if extra:
        raise LeakageError(
            f"{where} contains undocumented field(s) {extra}; "
            f"allowed fields are {sorted(allowed)}"
        )


def _require_opaque_class(label) -> str:
    if isinstance(label, bool) or not isinstance(label, str):
        raise LeakageError(
            f"an equality class must be an opaque string label, got {label!r}"
        )
    if not EQUALITY_CLASS_PATTERN.match(label):
        raise LeakageError(
            f"equality class {label!r} is not an opaque "
            f"{EQUALITY_CLASS_PATTERN.pattern} label"
        )
    return label


def _require_event(data: Mapping) -> ObservedEvent:
    if not isinstance(data, dict):
        raise LeakageError(f"event entries must be objects, got {data!r}")
    extra = sorted(set(data) - set(EVENT_KEYS))
    missing = sorted(set(EVENT_KEYS) - set(data))
    if extra or missing:
        raise LeakageError(
            f"event fields must be exactly {sorted(EVENT_KEYS)}; "
            f"extra={extra} missing={missing}"
        )
    return ObservedEvent.from_dict(data)


def _check_event_integrity(operation: ObservedOperation, where: str) -> None:
    """Event order, sequence numbering and volumes must match the events."""
    reads = sum(1 for event in operation.events if event.operation == "READ")
    writes = operation.event_count - reads
    if reads != operation.read_count or writes != operation.write_count:
        raise LeakageError(
            f"{where} operation {operation.index}: read_count/write_count "
            f"({operation.read_count}/{operation.write_count}) disagree with its "
            f"events ({reads}/{writes})"
        )
    for position, event in enumerate(operation.events):
        if event.seq != position:
            raise LeakageError(
                f"{where} operation {operation.index}: event seq {event.seq} at "
                f"position {position}; event sequence must be 0-based and ascending"
            )


def _copy_run_metadata(metadata: Mapping) -> dict:
    if not isinstance(metadata, dict):
        raise LeakageError("run_metadata must be an object")
    family = metadata.get("family")
    seed = metadata.get("seed")
    config = metadata.get("config")
    if not isinstance(family, str) or not family:
        raise LeakageError(f"run_metadata.family must be a non-empty string, got {family!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise LeakageError(f"run_metadata.seed must be an integer, got {seed!r}")
    if not isinstance(config, dict):
        raise LeakageError("run_metadata.config must be an object")
    copied = {}
    for key in CONFIG_KEYS:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise LeakageError(f"run_metadata.config.{key} must be an integer, got {value!r}")
        copied[key] = value
    return {"family": str(family), "seed": int(seed), "config": copied}


def _summarize(operations: Sequence[ObservedOperation]) -> dict:
    slots = {event.slot_id for op in operations for event in op.events}
    return {
        "operation_count": len(operations),
        "query_count": sum(1 for op in operations if op.operation_class == CLASS_QUERY),
        "merge_count": sum(1 for op in operations if op.operation_class == CLASS_MERGE),
        "total_reads": sum(1 for op in operations for event in op.events if event.operation == "READ"),
        "total_writes": sum(1 for op in operations for event in op.events if event.operation == "WRITE"),
        "distinct_slots": len(slots),
    }


# ---------------------------------------------------------------------------
# extraction from a G1-E artifact
# ---------------------------------------------------------------------------


def validate_g1e_artifact(payload) -> None:
    """Refuse a malformed, unsupported or unsuccessful G1-E artifact loudly.

    Source qualification: the artifact must be a completed G1-E run that G1-E
    itself reported as correct (``final`` present as an object and
    ``final.correctness_ok`` exactly the JSON boolean ``true``).  The ``final``
    section is privileged and is used **only** for this qualification — it is
    never copied into a transcript or into the metrics.
    """
    if not isinstance(payload, dict):
        raise LeakageError(f"a G1-E artifact must be a JSON object, got {type(payload).__name__}")
    version = payload.get("schema_version")
    if version != SOURCE_SCHEMA_VERSION:
        raise LeakageError(
            f"unsupported source schema_version {version!r}; "
            f"G2-A consumes {SOURCE_SCHEMA_VERSION!r} artifacts only"
        )
    final = payload.get("final")
    if not isinstance(final, dict):
        raise LeakageError(
            "G1-E artifact must carry a 'final' object recording its correctness "
            f"status; got {final!r}"
        )
    if "correctness_ok" not in final:
        raise LeakageError(
            "G1-E artifact final section is missing 'correctness_ok'; G2-A only "
            "consumes runs whose correctness status is recorded"
        )
    if final["correctness_ok"] is not True:
        raise LeakageError(
            "G1-E artifact is not an accepted successful run: "
            f"final.correctness_ok must be exactly true (JSON boolean), got "
            f"{final['correctness_ok']!r}"
        )
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise LeakageError("G1-E artifact must carry a non-empty operations list")
    for position, record in enumerate(operations):
        where = f"operations[{position}]"
        if not isinstance(record, dict):
            raise LeakageError(f"{where} must be an object")
        index = _require_int(record, "index", where)
        if index != position:
            raise LeakageError(
                f"{where}.index is {index}; G1-E records must be "
                "in execution order with index equal to their position"
            )
        operation_class = _require_source_kind(record, where)
        events = record.get("events")
        if not isinstance(events, list):
            raise LeakageError(f"{where}.events must be a list")
        observed_events = tuple(_require_event(event) for event in events)
        operation = ObservedOperation(
            index=index,
            operation_class=operation_class,
            events=observed_events,
            read_count=_require_int(record, "read_count", where),
            write_count=_require_int(record, "write_count", where),
        )
        _check_event_integrity(operation, "G1-E artifact")


def extract_transcript(artifact: Mapping) -> AdversaryTranscript:
    """Build the base (M0) adversary transcript from an exported G1-E artifact.

    Extraction is a whitelist: only the frozen observation fields are copied, so
    plaintext keys/values, oracle answers, logical ids, the workload script, the
    artifact's ``final`` section and every other evaluator-only field stay out of
    the adversary view by construction.  The source field name is schema-specific:
    a G1-E operation record is read through ``kind`` and written to the transcript
    as ``class``, with no alias fallback in either direction.  The caller must pass
    a G1-E **artifact dictionary** (as written by ``write_run_json``); the
    transcript never reads a live runner, engine, oracle or script object.
    """
    validate_g1e_artifact(artifact)
    operations = tuple(
        ObservedOperation(
            index=record["index"],
            operation_class=_require_source_kind(record, f"operations[{position}]"),
            events=tuple(_require_event(event) for event in record["events"]),
            read_count=record["read_count"],
            write_count=record["write_count"],
        )
        for position, record in enumerate(artifact["operations"])
    )
    metadata = _copy_run_metadata(
        {
            "family": artifact.get("family"),
            "seed": artifact.get("seed"),
            "config": artifact.get("config", {}),
        }
    )
    return AdversaryTranscript(
        schema_version=SCHEMA_VERSION,
        source_schema_version=SOURCE_SCHEMA_VERSION,
        run_metadata=metadata,
        operations=operations,
        summary=_summarize(operations),
        equality_channel=EQUALITY_CHANNEL_M0,
    )


def validate_transcript(payload) -> None:
    """Refuse a malformed or unsupported transcript payload loudly.

    Strictness: the payload must contain exactly the documented fields at every
    level (top-level, operation, run metadata, config, summary, event), so a
    transcript file can never look schema-valid while smuggling privileged fields
    such as ``key``, ``script`` or ``level_id``.  An ``M1`` transcript must carry a
    **complete** equality partition: every ``QUERY`` operation labelled exactly
    once with an opaque class, no label on any other operation, and no label at all
    in ``M0``.
    """
    if not isinstance(payload, dict):
        raise LeakageError(f"a transcript must be a JSON object, got {type(payload).__name__}")
    _reject_extra_fields(payload, TRANSCRIPT_KEYS, "transcript")
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise LeakageError(
            f"unsupported transcript schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    if payload.get("source_schema_version") != SOURCE_SCHEMA_VERSION:
        raise LeakageError(
            f"unsupported transcript source_schema_version "
            f"{payload.get('source_schema_version')!r}; expected {SOURCE_SCHEMA_VERSION!r}"
        )
    if "equality_channel" not in payload:
        raise LeakageError("transcript is missing the required field 'equality_channel'")
    channel = payload["equality_channel"]
    if channel not in EQUALITY_CHANNELS:
        raise LeakageError(
            f"equality_channel must be one of {EQUALITY_CHANNELS}, got {channel!r}"
        )
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise LeakageError("transcript run_metadata must be an object")
    _reject_extra_fields(metadata, RUN_METADATA_KEY_SET, "transcript run_metadata")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise LeakageError("transcript run_metadata.config must be an object")
    _reject_extra_fields(config, CONFIG_KEY_SET, "transcript run_metadata.config")
    _copy_run_metadata(metadata)
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise LeakageError("transcript operations must be a list")
    parsed: list[ObservedOperation] = []
    present_label_indices: set[int] = set()
    for position, record in enumerate(operations):
        if not isinstance(record, dict):
            raise LeakageError(f"transcript operations[{position}] must be an object")
        operation = ObservedOperation.from_dict(record)
        if operation.index != position:
            raise LeakageError(
                f"transcript operations[{position}].index is {operation.index}; "
                "operations must be in order"
            )
        # raw field presence, not the parsed optional value (absent != null)
        if "query_equality_class" in record:
            present_label_indices.add(operation.index)
        parsed.append(operation)

    query_indices = {op.index for op in parsed if op.operation_class == CLASS_QUERY}
    misplaced = sorted(
        index for index in present_label_indices if index not in query_indices
    )
    if misplaced:
        raise LeakageError(
            f"the 'query_equality_class' field is present on non-{CLASS_QUERY} "
            f"operation(s) {misplaced}; it must be absent on every non-QUERY operation"
        )
    if channel == EQUALITY_CHANNEL_M0:
        if present_label_indices:
            raise LeakageError(
                "an M0 transcript must not carry the 'query_equality_class' field on "
                f"any operation — it must be absent (not null); got "
                f"{sorted(present_label_indices)}"
            )
    elif present_label_indices != query_indices:
        raise LeakageError(
            f"an {EQUALITY_CHANNEL_M1} transcript must carry the "
            f"'query_equality_class' field on every {CLASS_QUERY} operation exactly "
            f"once; missing={sorted(query_indices - present_label_indices)} "
            f"unexpected={sorted(present_label_indices - query_indices)}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise LeakageError("transcript summary must be an object")
    if set(summary) != SUMMARY_KEY_SET:
        raise LeakageError(
            f"transcript summary must contain exactly {sorted(SUMMARY_KEY_SET)}; "
            f"got {sorted(summary)}"
        )
    # every summary field is a count: a JSON integer, and never a boolean
    # (``True == 1`` / ``1.0 == 1`` must not be able to satisfy the comparison)
    for key in SUMMARY_KEYS:
        _require_int(summary, key, "transcript summary")
    expected = _summarize(tuple(parsed))
    for key in SUMMARY_KEYS:
        if summary[key] != expected[key]:
            raise LeakageError(
                f"transcript summary.{key} is {summary[key]!r} but the events say "
                f"{expected[key]!r}"
            )


# ---------------------------------------------------------------------------
# explicitly privileged evaluator namespace
# ---------------------------------------------------------------------------


class PrivilegedEvaluator:
    """Evaluator-only helpers that consume G1-E **ground truth**.

    Everything in this namespace is privileged: it reads plaintext workload
    metadata that the adversary never sees.  Nothing here is used by
    :func:`extract_transcript` or by the baseline descriptive leakage metrics,
    and no field produced here may be merged into the base transcript — the only
    output that may leave this namespace is an opaque equality-class label.
    """

    @staticmethod
    def m1_equality_classes(artifact: Mapping) -> dict:
        """Map QUERY operation indices to opaque M1 equality classes.

        Two QUERY operations get the same ``QClassN`` label exactly when their
        G1-E workload operation carries the same plaintext key, i.e. when an M1
        adversary would be able to tell that they are equal.  Classes are
        assigned in first-occurrence order, so the mapping is deterministic.
        The plaintext key itself is never returned or attached.
        """
        validate_g1e_artifact(artifact)
        script = artifact.get("script")
        if not isinstance(script, dict) or not isinstance(script.get("operations"), list):
            raise LeakageError(
                "M1 equality classes require the G1-E artifact to carry its "
                "workload script (script.operations)"
            )
        script_operations = script["operations"]
        artifact_operations = artifact["operations"]
        if len(script_operations) != len(artifact_operations):
            raise LeakageError(
                f"script has {len(script_operations)} operations but the artifact "
                f"recorded {len(artifact_operations)}"
            )
        class_of_key: dict[int, str] = {}
        labels: dict[int, str] = {}
        for position, (record, script_operation) in enumerate(
            zip(artifact_operations, script_operations)
        ):
            if not isinstance(script_operation, dict):
                raise LeakageError(f"script.operations[{position}] must be an object")
            if script_operation.get("kind") != record.get("kind"):
                raise LeakageError(
                    f"script.operations[{position}].kind "
                    f"{script_operation.get('kind')!r} does not match the recorded "
                    f"operation kind {record.get('kind')!r}"
                )
            if record.get("kind") != CLASS_QUERY:
                continue
            keys = script_operation.get("keys") or []
            if len(keys) != 1:
                raise LeakageError(
                    f"script.operations[{position}] is a QUERY with {len(keys)} "
                    "keys; G2-A expects exactly one key per QUERY operation"
                )
            key = keys[0]
            if isinstance(key, bool) or not isinstance(key, int):
                raise LeakageError(f"query key must be an integer, got {key!r}")
            label = class_of_key.setdefault(key, f"QClass{len(class_of_key)}")
            labels[int(record["index"])] = label
        return labels

    @classmethod
    def m1_equality_transcript(cls, artifact: Mapping) -> AdversaryTranscript:
        """Base transcript plus the optional opaque M1 equality-class channel."""
        return extract_transcript(artifact).with_equality_channel(
            cls.m1_equality_classes(artifact)
        )


# ---------------------------------------------------------------------------
# deterministic JSON export
# ---------------------------------------------------------------------------


def dumps_transcript(transcript: AdversaryTranscript) -> str:
    """Deterministic JSON text for a transcript (no timestamps, no paths)."""
    return json.dumps(transcript.to_dict(), indent=2)


def write_transcript_json(transcript: AdversaryTranscript, path: str | Path) -> dict:
    payload = transcript.to_dict()
    Path(path).write_text(dumps_transcript(transcript) + "\n", encoding="utf-8")
    return payload


def dumps_metrics(metrics: Mapping) -> str:
    """Deterministic JSON text for a descriptive metrics payload."""
    return json.dumps(dict(metrics), indent=2)


def write_metrics_json(metrics: Mapping, path: str | Path) -> dict:
    payload = dict(metrics)
    Path(path).write_text(dumps_metrics(payload) + "\n", encoding="utf-8")
    return payload


def load_artifact(path: str | Path) -> dict:
    """Read a G1-E artifact from disk, failing loudly on unusable input."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LeakageError(f"cannot read G1-E artifact {path}: {exc}") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LeakageError(f"G1-E artifact {path} is not valid JSON: {exc}") from None
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m enhanced_letindex.leakage",
        description=(
            "Extract the G2-A adversary observation transcript and baseline "
            "descriptive leakage metrics from an exported G1-E artifact "
            "(schema g1e-1).  Observation only: no attack, no recovery and no "
            "defence is implemented.  Query-equality labels are off by default "
            "(M0); the optional M1 channel is evaluator-side and privileged."
        ),
    )
    parser.add_argument("--input", required=True,
                        help="path to an exported G1-E artifact (schema g1e-1)")
    parser.add_argument("--transcript-out", default="g2a_transcript.json",
                        help="output path for the adversary transcript")
    parser.add_argument("--metrics-out", default="g2a_metrics.json",
                        help="output path for the descriptive leakage metrics")
    parser.add_argument("--equality-channel", default="none", choices=["none", "m1"],
                        help="none (default, M0: no equality labels) or m1 "
                             "(privileged evaluator-side opaque equality classes)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .leakage_metrics import leakage_metrics  # local import keeps the CLI module self-contained

    args = build_parser().parse_args(argv)
    try:
        artifact = load_artifact(args.input)
        if args.equality_channel == "m1":
            transcript = PrivilegedEvaluator.m1_equality_transcript(artifact)
        else:
            transcript = extract_transcript(artifact)
        metrics = leakage_metrics(transcript)
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_transcript_json(transcript, args.transcript_out)
    write_metrics_json(metrics, args.metrics_out)
    summary = transcript.summary
    print(f"input={args.input} equality_channel={transcript.equality_channel}")
    print(f"operations={summary['operation_count']} "
          f"(query={summary['query_count']} merge={summary['merge_count']}) "
          f"reads={summary['total_reads']} writes={summary['total_writes']} "
          f"distinct_slots={summary['distinct_slots']}")
    print(f"family={transcript.run_metadata['family']} "
          f"seed={transcript.run_metadata['seed']}")
    print(f"wrote {args.transcript_out}")
    print(f"wrote {args.metrics_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
