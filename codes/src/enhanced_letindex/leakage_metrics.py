"""G2-A baseline **descriptive** leakage metrics (decisions/0008).

These metrics are the frozen descriptive leakage surfaces of the G2-A
observation layer.  They are not privacy scores and not attack-success metrics:
they only describe what the adversary transcript already shows.

Hard rule: **every metric here consumes an
:class:`~enhanced_letindex.leakage.AdversaryTranscript` and nothing else.** This
module deliberately has no runtime dependency on the engine, the workload
harness, the oracle or the exported G1-E artifact, so privileged ground truth is
unreachable from the metrics path by construction.  Any future evaluator-only
check that needs ground truth must live in the privileged namespace of
``leakage.py`` instead of being mixed in here.

The optional M1 query-equality channel is never consumed: baseline metrics are
always computed in M0 semantics, so annotating a transcript with opaque equality
classes cannot change a single number (see ``equality_channel`` in the payload).

Definitions:

* *slot sequence* — the ordered ``slot_id`` values of an operation's events;
* *repeated slot* — a slot id occurring more than once inside one operation;
* *Jaccard* — ``|A ∩ B| / |A ∪ B|`` for the physical slot **sets** of two QUERY
  operations, rounded to 6 decimals; two empty sets are reported as ``0.0``
  (no shared accessed-slot information), while ``set_equal`` / ``sequence_equal``
  are boolean facts and are ``True`` for two empty traces;
* *hotness* — how many distinct QUERY operations touched a slot;
* *adjacency* — consecutive steps inside one QUERY trace whose slot ids differ
  by exactly 1 (both directions), plus the signed step-difference histogram;
* *prior-query overlap* — the intersection of a MERGE operation's touched slots
  with the slots touched by QUERY operations that precede it in the transcript.

No phase labels (A/B/C/D) are derived or injected anywhere: the merge summary is
built strictly from the observed READ/WRITE sequence and slot ids.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from .leakage import AdversaryTranscript, ObservedOperation

from .leakage import (
    CLASS_MERGE,
    CLASS_QUERY,
    METRICS_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)

__all__ = [
    "METRICS_SCHEMA_VERSION",
    "MEAN_DIGITS",
    "leakage_metrics",
    "operation_metrics",
    "per_operation_metrics",
    "query_metrics",
    "merge_metrics",
]

MEAN_DIGITS = 6
"""Ratios and means are rounded to this many decimals so export is byte-stable."""


# ---------------------------------------------------------------------------
# small deterministic primitives
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float:
    """``numerator / denominator`` rounded to :data:`MEAN_DIGITS`, 0.0 if empty."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, MEAN_DIGITS)


def _mean(values: Sequence[int]) -> float:
    return _ratio(sum(values), len(values))


def _histogram(values: Iterable[int], key: str) -> list[dict]:
    """Deterministic ``[{key: value, "count": n}]`` sorted by value."""
    counts = Counter(values)
    return _count_histogram(counts, key)


def _count_histogram(counts: Counter, key: str, count_key: str = "count") -> list[dict]:
    """Histogram over an already-counted multiset, sorted by value."""
    return [{key: value, count_key: counts[value]} for value in sorted(counts)]


def _run_length_encoding(operations: Sequence[str]) -> list[dict]:
    runs: list[dict] = []
    for operation in operations:
        if runs and runs[-1]["operation"] == operation:
            runs[-1]["count"] += 1
        else:
            runs.append({"operation": operation, "count": 1})
    return runs


def _max_run(operations: Sequence[str], wanted: str) -> int:
    longest = 0
    current = 0
    for operation in operations:
        current = current + 1 if operation == wanted else 0
        longest = max(longest, current)
    return longest


def _slots(operation: "ObservedOperation") -> tuple[int, ...]:
    return tuple(event.slot_id for event in operation.events)


# ---------------------------------------------------------------------------
# per-operation metrics
# ---------------------------------------------------------------------------


def operation_metrics(operation: "ObservedOperation") -> dict:
    """Descriptive metrics of one observed operation, from its own events only."""
    slots = _slots(operation)
    counts = Counter(slots)
    repeated = sorted(slot for slot, count in counts.items() if count > 1)
    reads = sum(1 for event in operation.events if event.operation == "READ")
    writes = operation.event_count - reads
    return {
        "index": operation.index,
        "class": operation.operation_class,
        "event_count": len(slots),
        "read_volume": reads,
        "write_volume": writes,
        "distinct_slots": len(counts),
        "repeated_slot_count": len(repeated),
        "repeated_slot_ids": repeated,
        "repeat_access_count": len(slots) - len(counts),
        "slot_sequence": list(slots),
        "min_slot": min(slots) if slots else None,
        "max_slot": max(slots) if slots else None,
        "physical_span": (max(slots) - min(slots) + 1) if slots else None,
    }


def per_operation_metrics(transcript: "AdversaryTranscript") -> list[dict]:
    """Per-operation metrics for every operation, in transcript order."""
    return [operation_metrics(operation) for operation in transcript.operations]


# ---------------------------------------------------------------------------
# across QUERY operations
# ---------------------------------------------------------------------------


def query_metrics(transcript: "AdversaryTranscript") -> dict:
    """Descriptive leakage surfaces across QUERY operations (no equality labels)."""
    queries = transcript.query_operations
    if not queries:
        return {
            "query_count": 0,
            "trace_length": {"min": None, "max": None, "mean": 0.0, "total": 0,
                             "histogram": []},
            "distinct_slots": {"min": None, "max": None, "mean": 0.0, "total": 0,
                               "histogram": []},
            "pairwise": {
                "pair_count": 0, "mean_jaccard": 0.0, "max_jaccard": 0.0,
                "min_jaccard": 0.0, "pairs_with_intersection_count": 0,
                "exact_set_equality_count": 0, "exact_sequence_equality_count": 0,
                "exact_sequence_equality_rate": 0.0, "pairs": [],
            },
            "slot_hotness": {
                "distinct_query_slots": 0, "total_query_slot_accesses": 0,
                "repeated_query_slot_count": 0, "max_slot_query_count": 0,
                "slot_query_frequency": [], "hotness_histogram": [],
            },
            "adjacency": {
                "total_consecutive_steps": 0, "adjacent_by_one_count": 0,
                "adjacent_by_one_rate": 0.0, "step_difference_histogram": [],
                "per_query": [],
            },
        }

    sequences = {operation.index: _slots(operation) for operation in queries}
    slot_sets = {index: frozenset(slots) for index, slots in sequences.items()}
    trace_lengths = [len(sequences[operation.index]) for operation in queries]
    distinct_counts = [len(slot_sets[operation.index]) for operation in queries]

    pairs: list[dict] = []
    for position, first in enumerate(queries):
        for second in queries[position + 1:]:
            first_set = slot_sets[first.index]
            second_set = slot_sets[second.index]
            intersection = len(first_set & second_set)
            union = len(first_set | second_set)
            pairs.append({
                "index_a": first.index,
                "index_b": second.index,
                "intersection_size": intersection,
                "union_size": union,
                "jaccard": _ratio(intersection, union),
                "set_equal": first_set == second_set,
                "sequence_equal": sequences[first.index] == sequences[second.index],
            })
    pair_count = len(pairs)
    jaccards = [pair["jaccard"] for pair in pairs]

    slot_query_counts: Counter[int] = Counter()
    slot_access_counts: Counter[int] = Counter()
    for operation in queries:
        slot_query_counts.update(slot_sets[operation.index])
        slot_access_counts.update(sequences[operation.index])
    hotness_histogram = _count_histogram(
        Counter(slot_query_counts.values()), "query_count", "slot_count"
    )

    per_query_adjacency: list[dict] = []
    step_differences: Counter[int] = Counter()
    total_steps = 0
    adjacent_steps = 0
    for operation in queries:
        slots = sequences[operation.index]
        adjacent = 0
        for previous, following in zip(slots, slots[1:]):
            difference = following - previous
            step_differences[difference] += 1
            if abs(difference) == 1:
                adjacent += 1
        steps = max(len(slots) - 1, 0)
        total_steps += steps
        adjacent_steps += adjacent
        per_query_adjacency.append({
            "index": operation.index,
            "consecutive_steps": steps,
            "adjacent_by_one_count": adjacent,
        })

    return {
        "query_count": len(queries),
        "trace_length": {
            "min": min(trace_lengths),
            "max": max(trace_lengths),
            "mean": _mean(trace_lengths),
            "total": sum(trace_lengths),
            "histogram": _histogram(trace_lengths, "length"),
        },
        "distinct_slots": {
            "min": min(distinct_counts),
            "max": max(distinct_counts),
            "mean": _mean(distinct_counts),
            "total": sum(distinct_counts),
            "histogram": _histogram(distinct_counts, "distinct_slots"),
        },
        "pairwise": {
            "pair_count": pair_count,
            "mean_jaccard": _mean(jaccards) if pair_count else 0.0,
            "max_jaccard": max(jaccards) if pair_count else 0.0,
            "min_jaccard": min(jaccards) if pair_count else 0.0,
            "pairs_with_intersection_count": sum(
                1 for pair in pairs if pair["intersection_size"] > 0
            ),
            "exact_set_equality_count": sum(1 for pair in pairs if pair["set_equal"]),
            "exact_sequence_equality_count": sum(
                1 for pair in pairs if pair["sequence_equal"]
            ),
            "exact_sequence_equality_rate": _ratio(
                sum(1 for pair in pairs if pair["sequence_equal"]), pair_count
            ),
            "pairs": pairs,
        },
        "slot_hotness": {
            "distinct_query_slots": len(slot_query_counts),
            "total_query_slot_accesses": sum(slot_access_counts.values()),
            "repeated_query_slot_count": sum(
                1 for slot in slot_query_counts if slot_query_counts[slot] > 1
            ),
            "max_slot_query_count": max(slot_query_counts.values(), default=0),
            "slot_query_frequency": [
                {
                    "slot_id": slot,
                    "query_count": slot_query_counts[slot],
                    "access_count": slot_access_counts[slot],
                }
                for slot in sorted(slot_query_counts)
            ],
            "hotness_histogram": hotness_histogram,
        },
        "adjacency": {
            "total_consecutive_steps": total_steps,
            "adjacent_by_one_count": adjacent_steps,
            "adjacent_by_one_rate": _ratio(adjacent_steps, total_steps),
            "step_difference_histogram": _count_histogram(
                step_differences, "difference"
            ),
            "per_query": per_query_adjacency,
        },
    }


# ---------------------------------------------------------------------------
# across MERGE operations
# ---------------------------------------------------------------------------


def merge_metrics(transcript: "AdversaryTranscript") -> dict:
    """Descriptive leakage surfaces across MERGE operations, phase-agnostic.

    Only the observable READ/WRITE sequence and slot ids are used: no hidden
    A/B/C/D merge-phase label is derived, stored or exposed.
    """
    prior_query_slots: set[int] = set()
    prior_query_count = 0
    per_merge: list[dict] = []
    merge_slots_all: set[int] = set()
    total_reads = 0
    total_writes = 0

    for operation in transcript.operations:
        if operation.operation_class == CLASS_QUERY:
            prior_query_slots |= set(_slots(operation))
            prior_query_count += 1
            continue
        if operation.operation_class != CLASS_MERGE:
            continue
        slots = _slots(operation)
        merge_set = set(slots)
        merge_slots_all |= merge_set
        sequence = [event.operation for event in operation.events]
        reads = sequence.count("READ")
        writes = len(sequence) - reads
        total_reads += reads
        total_writes += writes
        overlap = merge_set & prior_query_slots
        per_merge.append({
            "index": operation.index,
            "event_count": len(slots),
            "read_volume": reads,
            "write_volume": writes,
            "distinct_slots": len(merge_set),
            "distinct_read_slots": len({event.slot_id for event in operation.events
                                        if event.operation == "READ"}),
            "distinct_write_slots": len({event.slot_id for event in operation.events
                                         if event.operation == "WRITE"}),
            "min_slot": min(slots) if slots else None,
            "max_slot": max(slots) if slots else None,
            "physical_span": (max(slots) - min(slots) + 1) if slots else None,
            "operation_runs": _run_length_encoding(sequence),
            "read_run_count": sum(1 for run in _run_length_encoding(sequence)
                                  if run["operation"] == "READ"),
            "write_run_count": sum(1 for run in _run_length_encoding(sequence)
                                   if run["operation"] == "WRITE"),
            "max_read_run": _max_run(sequence, "READ"),
            "max_write_run": _max_run(sequence, "WRITE"),
            "prior_query_overlap": {
                "prior_query_count": prior_query_count,
                "prior_query_slots": len(prior_query_slots),
                "merge_slots": len(merge_set),
                "overlap_slots": len(overlap),
                "overlap_ratio": _ratio(len(overlap), len(merge_set)),
            },
        })

    ratios = [entry["prior_query_overlap"]["overlap_ratio"] for entry in per_merge]
    return {
        "merge_count": len(per_merge),
        "total_reads": total_reads,
        "total_writes": total_writes,
        "distinct_slots": len(merge_slots_all),
        "mean_overlap_ratio": _mean(ratios) if ratios else 0.0,
        "max_overlap_ratio": max(ratios) if ratios else 0.0,
        "sum_prior_query_overlap_slots": sum(
            entry["prior_query_overlap"]["overlap_slots"] for entry in per_merge
        ),
        "per_merge": per_merge,
    }


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def leakage_metrics(transcript: "AdversaryTranscript") -> dict:
    """The complete descriptive metrics payload for one adversary transcript.

    Always M0: the optional query-equality channel is ignored, so the payload is
    identical for an M0 transcript and for an M1-annotated copy of it.
    """
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "transcript_schema_version": transcript.schema_version,
        "equality_channel": "M0",
        "per_operation": per_operation_metrics(transcript),
        "query": query_metrics(transcript),
        "merge": merge_metrics(transcript),
    }
