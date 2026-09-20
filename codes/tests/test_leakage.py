"""G2-A adversary observation transcript and baseline leakage metric tests.

Covers ``decisions/0008-g2-a-observation-transcript.md`` and the required-test
list of GitHub Issue #9.  Test labels ``# [n]`` follow that list.

The metrics tests deliberately re-derive every expectation from the raw event
stream (or from an independently built synthetic artifact), never from the
module under test.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from enhanced_letindex import leakage, leakage_metrics
from enhanced_letindex.leakage import (
    EQUALITY_CHANNEL_M0,
    EQUALITY_CHANNEL_M1,
    SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    AdversaryTranscript,
    LeakageError,
    PrivilegedEvaluator,
    dumps_metrics,
    dumps_transcript,
    extract_transcript,
    load_artifact,
    main,
    validate_g1e_artifact,
    validate_transcript,
    write_metrics_json,
    write_transcript_json,
)
from enhanced_letindex.leakage_metrics import (
    leakage_metrics as build_metrics,
    merge_metrics,
    operation_metrics,
    per_operation_metrics,
    query_metrics,
)
from enhanced_letindex.workload import export_dict, generate_workload, run_script

# ---------------------------------------------------------------------------
# documented schemas (stable primitive fields only)
# ---------------------------------------------------------------------------

TRANSCRIPT_KEYS = {
    "schema_version", "source_schema_version", "equality_channel", "run_metadata",
    "operations", "summary",
}
RUN_METADATA_KEYS = {"family", "seed", "config"}
RUN_CONFIG_KEYS = {"block_capacity", "pgm_epsilon"}
OPERATION_KEYS = {"index", "class", "events", "read_count", "write_count"}
OPERATION_KEYS_M1 = OPERATION_KEYS | {"query_equality_class"}
EVENT_KEYS = {"seq", "operation", "slot_id"}
SUMMARY_KEYS = {
    "operation_count", "query_count", "merge_count", "total_reads", "total_writes",
    "distinct_slots",
}

METRICS_KEYS = {
    "schema_version", "source_schema_version", "transcript_schema_version",
    "equality_channel", "per_operation", "query", "merge",
}
PER_OPERATION_KEYS = {
    "index", "class", "event_count", "read_volume", "write_volume", "distinct_slots",
    "repeated_slot_count", "repeated_slot_ids", "repeat_access_count", "slot_sequence",
    "min_slot", "max_slot", "physical_span",
}
QUERY_METRIC_KEYS = {
    "query_count", "trace_length", "distinct_slots", "pairwise", "slot_hotness",
    "adjacency",
}
DISTRIBUTION_KEYS = {"min", "max", "mean", "total", "histogram"}
PAIRWISE_KEYS = {
    "pair_count", "mean_jaccard", "max_jaccard", "min_jaccard",
    "pairs_with_intersection_count", "exact_set_equality_count",
    "exact_sequence_equality_count", "exact_sequence_equality_rate", "pairs",
}
PAIR_KEYS = {
    "index_a", "index_b", "intersection_size", "union_size", "jaccard", "set_equal",
    "sequence_equal",
}
SLOT_HOTNESS_KEYS = {
    "distinct_query_slots", "total_query_slot_accesses", "repeated_query_slot_count",
    "max_slot_query_count", "slot_query_frequency", "hotness_histogram",
}
ADJACENCY_KEYS = {
    "total_consecutive_steps", "adjacent_by_one_count", "adjacent_by_one_rate",
    "step_difference_histogram", "per_query",
}
MERGE_METRIC_KEYS = {
    "merge_count", "total_reads", "total_writes", "distinct_slots",
    "mean_overlap_ratio", "max_overlap_ratio", "sum_prior_query_overlap_slots",
    "per_merge",
}
PER_MERGE_KEYS = {
    "index", "event_count", "read_volume", "write_volume", "distinct_slots",
    "distinct_read_slots", "distinct_write_slots", "min_slot", "max_slot",
    "physical_span", "operation_runs", "read_run_count", "write_run_count",
    "max_read_run", "max_write_run", "prior_query_overlap",
}
PRIOR_OVERLAP_KEYS = {
    "prior_query_count", "prior_query_slots", "merge_slots", "overlap_slots",
    "overlap_ratio",
}

# every key that only privileged G1-E ground truth can produce
FORBIDDEN_KEYS = {
    "script", "operations_script", "final", "description", "note", "keys", "values",
    "key", "value", "found", "result", "oracle", "verification", "hit_ref",
    "searched_refs", "searched_level_refs", "expected_found", "expected_value",
    "expected_searched_refs", "level_ref", "level_id", "newer_ref", "older_ref",
    "block_id", "block_ids", "record_count", "block_count", "queryable",
    "active_levels", "retired_levels", "oracle_record_count", "oracle_mismatches",
    "query_mismatches", "merge_mismatches", "verification_failures", "correctness_ok",
    "detailed_checks", "matches_oracle", "output_contents_match",
}

# attack / recovery / defence vocabulary that must not appear as code in G2-A
FORBIDDEN_CODE_TOKENS = (
    "attack", "recover", "recovery", "infer", "cdf", "ord_forced", "rank_width",
    "adj_recall", "defence", "defense", "reshuffle", "stash", "encrypt", "oram",
    "privacy", "score", "success", "correlate", "correlation", "budget",
)

FAMILY_FIXTURES = (("Q", 101, 4), ("UQ", 202, 4), ("UMQ", 303, 3))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def g1e_artifact(family: str, seed: int, capacity: int = 4, epsilon: int = 2) -> dict:
    """Run a G1-E workload family and return its exported artifact dictionary."""
    script = generate_workload(family, seed, capacity=capacity, epsilon=epsilon)
    runner = run_script(script)
    return export_dict(script, runner)


def record(index, kind, events, keys=None, ref=None):
    """One schema-valid G1-E operation record with explicit events."""
    events = [
        {"seq": position, "operation": operation, "slot_id": slot_id}
        for position, (operation, slot_id) in enumerate(events)
    ]
    payload = {
        "index": index,
        "kind": kind,
        "operation": {"index": index, "kind": kind, "keys": list(keys or ()),
                      "values": [f"t:{key}" for key in (keys or ())], "ref": ref,
                      "level_ref": None, "newer_ref": None, "older_ref": None,
                      "note": ""},
        "events": events,
        "read_count": sum(1 for event in events if event["operation"] == "READ"),
        "write_count": sum(1 for event in events if event["operation"] == "WRITE"),
        "result": {},
        "verification": {},
        "oracle": None,
    }
    return payload


def synthetic_artifact(operations, *, family="SYNTH", seed=7, capacity=4, epsilon=2):
    """A minimal but schema-valid G1-E artifact built from explicit operations.

    ``operations`` is a sequence of ``(kind, events, keys)`` where ``events`` is a
    sequence of ``("READ"|"WRITE", slot_id)`` pairs.  The artifact carries the
    workload script so the optional M1 equality channel can be exercised.
    """
    records = []
    script_operations = []
    for index, (kind, events, keys) in enumerate(
        (item if len(item) == 3 else (item[0], item[1], ()) for item in operations)
    ):
        records.append(record(index, kind, events, keys=keys, ref=f"L{index}"))
        script_operations.append({
            "index": index, "kind": kind, "keys": list(keys),
            "values": [f"L{index}:{key}" for key in keys], "ref": f"L{index}",
            "level_ref": None, "newer_ref": None, "older_ref": None, "note": "",
        })
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "family": family,
        "seed": seed,
        "config": {"block_capacity": capacity, "pgm_epsilon": epsilon,
                   "random_seed": 0, "lsm_level_ratio": 2},
        "script": {"family": family, "seed": seed, "capacity": capacity,
                   "epsilon": epsilon, "description": "synthetic",
                   "operations": script_operations},
        "operations": records,
        "final": {"correctness_ok": True},
    }


def artifact_events(artifact, index):
    return [event["slot_id"] for event in artifact["operations"][index]["events"]]


def recursive_keys(payload) -> set:
    keys = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(key)
            keys |= recursive_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= recursive_keys(item)
    return keys


def recursive_leaves(payload) -> list:
    if isinstance(payload, dict):
        return [leaf for value in payload.values() for leaf in recursive_leaves(value)]
    if isinstance(payload, list):
        return [leaf for item in payload for leaf in recursive_leaves(item)]
    return [payload]


def jaccard(first, second):
    union = len(first | second)
    return 0.0 if union == 0 else round(len(first & second) / union, 6)


@pytest.fixture(scope="module")
def artifacts():
    return {
        family: g1e_artifact(family, seed, capacity)
        for family, seed, capacity in FAMILY_FIXTURES
    }


@pytest.fixture(scope="module")
def transcripts(artifacts):
    return {family: extract_transcript(artifact) for family, artifact in artifacts.items()}


@pytest.fixture(scope="module")
def metrics(transcripts):
    return {family: build_metrics(transcript) for family, transcript in transcripts.items()}


# ---------------------------------------------------------------------------
# [1] no privileged G1-E field reaches the base transcript
# ---------------------------------------------------------------------------


def test_no_privileged_g1e_fields_in_base_transcript(artifacts, transcripts):
    for family, artifact in artifacts.items():
        transcript = transcripts[family]
        text = dumps_transcript(transcript)

        # no privileged key anywhere in the transcript tree
        assert recursive_keys(transcript.to_dict()) & FORBIDDEN_KEYS == set()

        # no plaintext value / symbolic level ref / script note text either
        tokens = set()
        for operation in artifact["script"]["operations"]:
            tokens.update(str(value) for value in operation["values"])
            for field in ("ref", "level_ref", "newer_ref", "older_ref"):
                if operation[field]:
                    tokens.add(str(operation[field]))
            if operation["note"]:
                tokens.add(str(operation["note"]))
        assert tokens, "fixture must carry plaintext tokens"
        for token in tokens:
            assert token not in text, f"{family}: plaintext token {token!r} leaked"

        # the artifact really does carry those privileged sections
        assert any(
            source["oracle"] for source in artifact["operations"] if source["kind"] == "QUERY"
        )


def test_source_artifact_is_read_only_and_unchanged(artifacts):
    snapshot = copy.deepcopy(artifacts)
    for artifact in artifacts.values():
        extract_transcript(artifact)
        build_metrics(extract_transcript(artifact))
    assert artifacts == snapshot
    for family, artifact in artifacts.items():
        assert artifact["schema_version"] == SOURCE_SCHEMA_VERSION
        assert artifact["final"]["correctness_ok"] is True
        assert artifact["final"]["oracle_mismatches"] == 0


# ---------------------------------------------------------------------------
# [2] event order and slot ids are preserved exactly
# ---------------------------------------------------------------------------


def test_trace_events_preserve_exact_order_and_slot_ids(artifacts, transcripts):
    for family, artifact in artifacts.items():
        transcript = transcripts[family]
        for source in artifact["operations"]:
            observed = transcript.operation(source["index"])
            assert [event.to_dict() for event in observed.events] == source["events"]
            assert [event.seq for event in observed.events] == list(
                range(len(source["events"]))
            )


# ---------------------------------------------------------------------------
# [3] operation boundaries and classes are preserved
# ---------------------------------------------------------------------------


def test_operation_boundaries_and_classes_are_preserved(artifacts, transcripts):
    for family, artifact in artifacts.items():
        transcript = transcripts[family]
        assert len(transcript.operations) == len(artifact["operations"])
        for position, source in enumerate(artifact["operations"]):
            observed = transcript.operations[position]
            assert observed.index == source["index"] == position
            assert observed.operation_class == source["kind"]
            assert observed.read_count == source["read_count"]
            assert observed.write_count == source["write_count"]
            assert observed.event_count == len(source["events"])
        counts = Counter(source["kind"] for source in artifact["operations"])
        assert transcript.summary["operation_count"] == len(artifact["operations"])
        assert transcript.summary["query_count"] == counts["QUERY"]
        assert transcript.summary["merge_count"] == counts["MERGE"]


# ---------------------------------------------------------------------------
# [4] deterministic transcript and metrics
# ---------------------------------------------------------------------------


def test_transcript_and_metrics_are_byte_identical_for_the_same_input(artifacts):
    for family, artifact in artifacts.items():
        first = extract_transcript(artifact)
        second = extract_transcript(copy.deepcopy(artifact))
        assert dumps_transcript(first) == dumps_transcript(second)
        assert dumps_metrics(build_metrics(first)) == dumps_metrics(build_metrics(second))
        # a fresh G1-E run of the same family/seed is the same artifact, so the
        # adversary view must be identical as well
        rerun = g1e_artifact(
            family,
            dict((f, s) for f, s, _ in FAMILY_FIXTURES)[family],
            dict((f, c) for f, _, c in FAMILY_FIXTURES)[family],
        )
        assert dumps_transcript(extract_transcript(rerun)) == dumps_transcript(first)


# ---------------------------------------------------------------------------
# [5] malformed / unsupported source artifacts fail loudly
# ---------------------------------------------------------------------------


def broken_artifacts():
    artifact = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    cases = {}

    def mutated(name, mutate):
        payload = copy.deepcopy(artifact)
        mutate(payload)
        cases[name] = payload

    mutated("unsupported_schema_version",
            lambda p: p.__setitem__("schema_version", "g1e-2"))
    mutated("missing_schema_version", lambda p: p.pop("schema_version"))
    mutated("operations_not_a_list", lambda p: p.__setitem__("operations", {}))
    mutated("operations_empty", lambda p: p.__setitem__("operations", []))
    mutated("index_not_position",
            lambda p: p["operations"][0].__setitem__("index", 3))
    mutated("unknown_kind", lambda p: p["operations"][0].__setitem__("kind", "DELETE"))
    mutated("events_not_a_list",
            lambda p: p["operations"][0].__setitem__("events", "READ 1"))
    mutated("event_extra_field",
            lambda p: p["operations"][0]["events"][0].__setitem__("level_id", 0))
    mutated("event_missing_slot",
            lambda p: p["operations"][0]["events"][0].pop("slot_id"))
    mutated("event_bad_operation",
            lambda p: p["operations"][0]["events"][0].__setitem__("operation", "SCAN"))
    mutated("event_seq_gap",
            lambda p: p["operations"][0]["events"][0].__setitem__("seq", 4))
    mutated("read_count_lies",
            lambda p: p["operations"][0].__setitem__("read_count", 2))
    mutated("config_missing",
            lambda p: p["config"].pop("block_capacity"))
    mutated("seed_not_int", lambda p: p.__setitem__("seed", "seven"))
    # source `kind` is required; the transcript field `class` is not an alias
    mutated("kind_missing_class_valid",
            lambda p: (p["operations"][0].pop("kind"),
                       p["operations"][0].__setitem__("class", "QUERY")))
    mutated("kind_invalid_class_valid",
            lambda p: (p["operations"][0].__setitem__("kind", "DELETE"),
                       p["operations"][0].__setitem__("class", "QUERY")))
    mutated("kind_missing", lambda p: p["operations"][0].pop("kind"))
    # source qualification: only successful G1-E runs may be observed
    mutated("final_missing", lambda p: p.pop("final"))
    mutated("final_not_an_object", lambda p: p.__setitem__("final", "ok"))
    mutated("final_is_a_list", lambda p: p.__setitem__("final", []))
    mutated("correctness_ok_missing", lambda p: p["final"].pop("correctness_ok"))
    mutated("correctness_ok_false",
            lambda p: p["final"].__setitem__("correctness_ok", False))
    mutated("correctness_ok_int_one",
            lambda p: p["final"].__setitem__("correctness_ok", 1))
    mutated("correctness_ok_string_true",
            lambda p: p["final"].__setitem__("correctness_ok", "true"))
    cases["not_an_object"] = [1, 2, 3]
    return cases


@pytest.mark.parametrize("name", sorted(broken_artifacts()))
def test_malformed_or_unsupported_sources_fail_loudly(name):
    with pytest.raises(LeakageError):
        extract_transcript(broken_artifacts()[name])


def test_transcript_validation_rejects_malformed_payloads(artifacts):
    transcript = extract_transcript(artifacts["UMQ"])
    payload = transcript.to_dict()
    assert payload is not None

    wrong_version = copy.deepcopy(payload)
    wrong_version["schema_version"] = "g2a-2"
    with pytest.raises(LeakageError):
        validate_transcript(wrong_version)

    wrong_source = copy.deepcopy(payload)
    wrong_source["source_schema_version"] = "g1e-2"
    with pytest.raises(LeakageError):
        validate_transcript(wrong_source)

    wrong_channel = copy.deepcopy(payload)
    wrong_channel["equality_channel"] = "M2"
    with pytest.raises(LeakageError):
        validate_transcript(wrong_channel)

    wrong_summary = copy.deepcopy(payload)
    wrong_summary["summary"]["distinct_slots"] += 1
    with pytest.raises(LeakageError):
        validate_transcript(wrong_summary)

    mislabelled = copy.deepcopy(payload)
    mislabelled["operations"][0]["query_equality_class"] = "QClass0"
    with pytest.raises(LeakageError):
        validate_transcript(mislabelled)

    reordered = copy.deepcopy(payload)
    reordered["operations"] = list(reversed(reordered["operations"]))
    with pytest.raises(LeakageError):
        validate_transcript(reordered)

    assert AdversaryTranscript.from_dict(payload).to_dict() == payload


def test_load_artifact_rejects_unreadable_input(tmp_path):
    with pytest.raises(LeakageError):
        load_artifact(tmp_path / "missing.json")
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(LeakageError):
        load_artifact(garbage)


# ---------------------------------------------------------------------------
# schema hardening (review round 1): kind vs class, M1 partition, source
# qualification, strict extra-field rejection
# ---------------------------------------------------------------------------


def test_kind_and_class_are_never_interchangeable_aliases(artifacts):
    """G1-E source records use `kind`; G2-A transcript records use `class`."""
    source = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])

    # source record with a valid `class` but no `kind`
    aliased = copy.deepcopy(source)
    aliased["operations"][0].pop("kind")
    aliased["operations"][0]["class"] = "QUERY"
    assert aliased["operations"][0]["class"] == "QUERY"
    with pytest.raises(LeakageError):
        validate_g1e_artifact(aliased)
    with pytest.raises(LeakageError):
        extract_transcript(aliased)

    # source record with an invalid `kind` and a valid `class`
    wrong_kind = copy.deepcopy(source)
    wrong_kind["operations"][0]["kind"] = "DELETE"
    wrong_kind["operations"][0]["class"] = "QUERY"
    with pytest.raises(LeakageError):
        validate_g1e_artifact(wrong_kind)
    with pytest.raises(LeakageError):
        extract_transcript(wrong_kind)

    # `kind` is still required on its own
    missing_kind = copy.deepcopy(source)
    missing_kind["operations"][0].pop("kind")
    with pytest.raises(LeakageError):
        extract_transcript(missing_kind)

    # transcript record with no `class`
    payload = extract_transcript(artifacts["Q"]).to_dict()
    missing_class = copy.deepcopy(payload)
    missing_class["operations"][0].pop("class")
    with pytest.raises(LeakageError):
        validate_transcript(missing_class)

    # transcript record carrying the *source* spelling instead
    kind_instead_of_class = copy.deepcopy(payload)
    kind_instead_of_class["operations"][0].pop("class")
    kind_instead_of_class["operations"][0]["kind"] = "BUILD_INITIAL"
    with pytest.raises(LeakageError):
        validate_transcript(kind_instead_of_class)

    # transcript record carrying both spellings is rejected too
    both_spellings = copy.deepcopy(payload)
    both_spellings["operations"][0]["kind"] = "BUILD_INITIAL"
    with pytest.raises(LeakageError):
        validate_transcript(both_spellings)

    # the two schemas still round-trip on their own terms
    translate = extract_transcript(source)
    assert translate.operation(0).operation_class == "QUERY"
    assert translate.to_dict()["operations"][0]["class"] == "QUERY"
    assert "kind" not in translate.to_dict()["operations"][0]


def test_cli_rejects_aliased_kind_and_unsuccessful_source(tmp_path, capsys):
    source = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    aliased = copy.deepcopy(source)
    aliased["operations"][0].pop("kind")
    aliased["operations"][0]["class"] = "QUERY"
    unsuccessful = copy.deepcopy(source)
    unsuccessful["final"] = {"correctness_ok": False}

    for name, payload in (("aliased", aliased), ("unsuccessful", unsuccessful)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        transcript_out = tmp_path / f"{name}_transcript.json"
        metrics_out = tmp_path / f"{name}_metrics.json"
        exit_code = main(["--input", str(path),
                          "--transcript-out", str(transcript_out),
                          "--metrics-out", str(metrics_out)])
        assert exit_code == 2, name
        assert "error:" in capsys.readouterr().err
        assert not transcript_out.exists() and not metrics_out.exists()


@pytest.mark.parametrize("value", [False, 0, 1, "true", "True", None, [], {}])
def test_source_correctness_ok_must_be_exactly_true(value):
    artifact = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    artifact["final"] = {"correctness_ok": value}
    with pytest.raises(LeakageError):
        validate_g1e_artifact(artifact)
    with pytest.raises(LeakageError):
        extract_transcript(artifact)


@pytest.mark.parametrize("final", ["ok", [], 0, None, {}, {"oracle_mismatches": 0}])
def test_source_requires_a_final_object_carrying_correctness_ok(final):
    artifact = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    artifact["final"] = final
    with pytest.raises(LeakageError):
        extract_transcript(artifact)

    without_final = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    without_final.pop("final")
    with pytest.raises(LeakageError):
        extract_transcript(without_final)


def test_source_qualification_never_leaks_the_final_section(artifacts, metrics):
    for family, artifact in artifacts.items():
        transcript = extract_transcript(artifact)
        payload = transcript.to_dict()
        assert "final" not in payload
        assert not (recursive_keys(payload) & {"final", "correctness_ok",
                                              "detailed_checks", "oracle_mismatches"})
        text = dumps_transcript(transcript)
        assert "correctness_ok" not in text and '"final"' not in text
        assert not (recursive_keys(metrics[family]) & {"final", "correctness_ok"})
        assert "correctness_ok" not in dumps_metrics(metrics[family])
        # the artifact itself does carry it, i.e. the check above is meaningful
        assert artifact["final"]["correctness_ok"] is True


def test_m1_requires_a_complete_query_partition(artifacts):
    transcript = extract_transcript(artifacts["UMQ"])
    queries = transcript.query_operations
    assert len(queries) > 1
    full = {op.index: f"QClass{position % 3}" for position, op in enumerate(queries)}

    # partial partition -> reject
    partial = dict(list(full.items())[:-1])
    with pytest.raises(LeakageError):
        transcript.with_equality_channel(partial)
    # queries exist but no labels at all -> reject
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({})

    # the evaluator-produced complete partition round-trips
    annotated = PrivilegedEvaluator.m1_equality_transcript(artifacts["UMQ"])
    payload = annotated.to_dict()
    assert annotated.equality_channel == EQUALITY_CHANNEL_M1
    labelled = {
        op["index"] for op in payload["operations"] if "query_equality_class" in op
    }
    assert labelled == {op.index for op in queries}
    assert AdversaryTranscript.from_dict(payload).to_dict() == payload

    # a serialized M1 payload that lost one query label -> reject
    lossy = copy.deepcopy(payload)
    for operation in lossy["operations"]:
        if "query_equality_class" in operation:
            operation.pop("query_equality_class")
            break
    with pytest.raises(LeakageError):
        validate_transcript(lossy)

    # a serialized M1 payload with no labels at all -> reject
    unlabelled = copy.deepcopy(payload)
    for operation in unlabelled["operations"]:
        operation.pop("query_equality_class", None)
    with pytest.raises(LeakageError):
        validate_transcript(unlabelled)

    # a serialized M1 payload with a label on a non-QUERY operation -> reject
    misplaced = copy.deepcopy(payload)
    non_query = next(
        op for op in misplaced["operations"] if op["class"] != "QUERY"
    )
    non_query["query_equality_class"] = "QClass0"
    with pytest.raises(LeakageError):
        validate_transcript(misplaced)

    # zero queries: the empty partition is the only valid M1 mapping
    no_queries = extract_transcript(synthetic_artifact([
        ("BUILD_PGM", [("READ", 1)], ()),
    ]))
    assert no_queries.query_operations == ()
    empty_partition = no_queries.with_equality_channel({})
    assert empty_partition.equality_channel == EQUALITY_CHANNEL_M1
    assert AdversaryTranscript.from_dict(empty_partition.to_dict()).to_dict() == (
        empty_partition.to_dict()
    )
    with pytest.raises(LeakageError):
        no_queries.with_equality_channel({0: "QClass0"})


def test_transcript_validation_rejects_undocumented_extra_fields(artifacts):
    payload = extract_transcript(artifacts["UMQ"]).to_dict()
    validate_transcript(payload)  # the documented payload is valid

    def with_extra(field, value, where="top"):
        broken = copy.deepcopy(payload)
        if where == "top":
            broken[field] = value
        elif where == "operation":
            broken["operations"][0][field] = value
        elif where == "event":
            broken["operations"][0]["events"][0][field] = value
        elif where == "run_metadata":
            broken["run_metadata"][field] = value
        elif where == "config":
            broken["run_metadata"]["config"][field] = value
        elif where == "summary":
            broken["summary"][field] = value
        return broken

    hostile = [
        with_extra("script", {"operations": []}),
        with_extra("key", 7),
        with_extra("final", {"correctness_ok": True}),
        with_extra("level_id", 3),
        with_extra("key", 7, "operation"),
        with_extra("level_id", 3, "operation"),
        with_extra("kind", "QUERY", "operation"),
        with_extra("result", {}, "operation"),
        with_extra("level_id", 0, "event"),
        with_extra("lsm_level_ratio", 2, "run_metadata"),
        with_extra("family_name", "x", "run_metadata"),
        with_extra("random_seed", 0, "config"),
        with_extra("lsm_level_ratio", 2, "config"),
        with_extra("leakage_score", 0.0, "summary"),
    ]
    for broken in hostile:
        with pytest.raises(LeakageError):
            validate_transcript(broken)
        with pytest.raises(LeakageError):
            AdversaryTranscript.from_dict(broken)

    # documented fields are still required
    for field, where in (("equality_channel", "top"), ("run_metadata", "top"),
                         ("summary", "top"), ("class", "operation"),
                         ("index", "operation"), ("slot_id", "event")):
        broken = copy.deepcopy(payload)
        if where == "top":
            broken.pop(field)
        elif where == "operation":
            broken["operations"][0].pop(field)
        else:
            broken["operations"][0]["events"][0].pop(field)
        with pytest.raises(LeakageError):
            validate_transcript(broken)

    broken_summary = copy.deepcopy(payload)
    broken_summary["summary"].pop("distinct_slots")
    with pytest.raises(LeakageError):
        validate_transcript(broken_summary)


def test_absent_and_explicit_null_are_distinct_states(artifacts):
    """`query_equality_class` absence is part of the schema; null is not absence."""
    base = extract_transcript(artifacts["Q"])
    payload = base.to_dict()
    query_index = base.query_operations[0].index
    non_query_index = next(
        op.index for op in base.operations if op.operation_class != "QUERY"
    )
    m1 = PrivilegedEvaluator.m1_equality_transcript(artifacts["Q"]).to_dict()

    # M0: the field must be absent on every operation, null included
    for where, index in (("QUERY", query_index), ("non-QUERY", non_query_index)):
        null_field = copy.deepcopy(payload)
        null_field["operations"][index]["query_equality_class"] = None
        with pytest.raises(LeakageError):
            validate_transcript(null_field)
        with pytest.raises(LeakageError):
            AdversaryTranscript.from_dict(null_field)

        labelled_field = copy.deepcopy(payload)
        labelled_field["operations"][index]["query_equality_class"] = "QClass0"
        with pytest.raises(LeakageError):
            validate_transcript(labelled_field)

    # a null on *every* operation is still not an M0 transcript
    all_null = copy.deepcopy(payload)
    for record in all_null["operations"]:
        record["query_equality_class"] = None
    with pytest.raises(LeakageError):
        validate_transcript(all_null)

    # M1: present on every QUERY, absent on every non-QUERY — null is neither
    null_on_non_query = copy.deepcopy(m1)
    null_on_non_query["operations"][non_query_index]["query_equality_class"] = None
    with pytest.raises(LeakageError):
        validate_transcript(null_on_non_query)

    null_on_query = copy.deepcopy(m1)
    null_on_query["operations"][query_index]["query_equality_class"] = None
    with pytest.raises(LeakageError):
        validate_transcript(null_on_query)

    # controls: normal M0 and the evaluator-produced complete M1 still round-trip
    assert AdversaryTranscript.from_dict(payload).to_dict() == payload
    assert AdversaryTranscript.from_dict(m1).to_dict() == m1
    assert "query_equality_class" not in recursive_keys(payload)
    assert all(
        "query_equality_class" in op for op in m1["operations"] if op["class"] == "QUERY"
    )
    assert all(
        "query_equality_class" not in op for op in m1["operations"] if op["class"] != "QUERY"
    )


SUMMARY_KEY_LIST = sorted(SUMMARY_KEYS)
ONE_QUERY_SUMMARY = {
    "operation_count": 1,
    "query_count": 1,
    "merge_count": 0,
    "total_reads": 1,
    "total_writes": 0,
    "distinct_slots": 1,
}


def one_query_transcript_payload():
    payload = extract_transcript(
        synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    ).to_dict()
    assert payload["summary"] == ONE_QUERY_SUMMARY
    return payload


@pytest.mark.parametrize("key", SUMMARY_KEY_LIST)
@pytest.mark.parametrize("value", [True, False])
def test_summary_counts_reject_booleans(key, value):
    """`True == 1` / `False == 0` must not satisfy an integer count field."""
    payload = one_query_transcript_payload()
    payload["summary"][key] = value
    with pytest.raises(LeakageError):
        validate_transcript(payload)
    with pytest.raises(LeakageError):
        AdversaryTranscript.from_dict(payload)


@pytest.mark.parametrize("key", SUMMARY_KEY_LIST)
def test_summary_counts_reject_integer_valued_floats(key):
    """An integer-valued float such as 1.0 must not satisfy a count field."""
    payload = one_query_transcript_payload()
    payload["summary"][key] = float(ONE_QUERY_SUMMARY[key])
    with pytest.raises(LeakageError):
        validate_transcript(payload)
    with pytest.raises(LeakageError):
        AdversaryTranscript.from_dict(payload)


@pytest.mark.parametrize("value", ["1", "0", None, [], {}])
def test_summary_counts_reject_other_non_integer_values(value):
    payload = one_query_transcript_payload()
    payload["summary"]["query_count"] = value
    with pytest.raises(LeakageError):
        validate_transcript(payload)


def test_summary_counts_still_validate_when_correct():
    payload = one_query_transcript_payload()
    validate_transcript(payload)
    transcript = AdversaryTranscript.from_dict(payload)
    assert transcript.summary == ONE_QUERY_SUMMARY
    assert all(type(value) is int for value in transcript.summary.values())
    wrong = copy.deepcopy(payload)
    wrong["summary"]["query_count"] = 2
    with pytest.raises(LeakageError):
        validate_transcript(wrong)


# ---------------------------------------------------------------------------
# [6] the base transcript carries no query equality label
# ---------------------------------------------------------------------------


def test_base_transcript_contains_no_query_equality_labels(artifacts, transcripts):
    for family in artifacts:
        transcript = transcripts[family]
        assert transcript.equality_channel == EQUALITY_CHANNEL_M0
        assert all(op.query_equality_class is None for op in transcript.operations)
        payload = transcript.to_dict()
        assert "query_equality_class" not in recursive_keys(payload)
        assert "query_equality_class" not in dumps_transcript(transcript)


# ---------------------------------------------------------------------------
# [7] the optional M1 equality class is opaque and key-free
# ---------------------------------------------------------------------------


def test_m1_equality_classes_are_opaque_and_key_free(artifacts):
    for family, artifact in artifacts.items():
        labels = PrivilegedEvaluator.m1_equality_classes(artifact)
        assert labels, f"{family}: expected query operations"
        assert all(label.startswith("QClass") for label in labels.values())

        plaintext_keys = {
            key for operation in artifact["script"]["operations"]
            if operation["kind"] == "QUERY" for key in operation["keys"]
        }
        assert plaintext_keys
        for label in labels.values():
            assert label not in {str(key) for key in plaintext_keys}

        transcript = PrivilegedEvaluator.m1_equality_transcript(artifact)
        assert transcript.equality_channel == EQUALITY_CHANNEL_M1
        text = dumps_transcript(transcript)
        # equality classes partition exactly the plaintext equality relation:
        # one label per distinct plaintext query key, in first-occurrence order,
        # and the label is never the key
        per_label = {}
        for index, label in labels.items():
            per_label.setdefault(label, []).append(index)
        keys_of_label = set()
        for label, indices in per_label.items():
            label_keys = {
                artifact["script"]["operations"][index]["keys"][0] for index in indices
            }
            assert len(label_keys) == 1, f"{label} mixes plaintext keys"
            keys_of_label |= label_keys
        assert len(keys_of_label) == len(per_label) == len(plaintext_keys)
        assert set(per_label) == {
            f"QClass{index}" for index in range(len(plaintext_keys))
        }

        # still no privileged field and no plaintext value in the M1 view
        payload = transcript.to_dict()
        assert recursive_keys(payload) & FORBIDDEN_KEYS == set()
        for value in {str(v) for op in artifact["script"]["operations"]
                      for v in op["values"]}:
            assert value not in text
        for operation in transcript.operations:
            if operation.query_equality_class is not None:
                assert operation.operation_class == "QUERY"
                assert operation.query_equality_class in set(labels.values())


def test_equality_channel_rejects_non_opaque_and_mislaced_labels(artifacts):
    transcript = extract_transcript(artifacts["Q"])
    queries = transcript.query_operations
    full = {op.index: f"QClass{position}" for position, op in enumerate(queries)}
    non_query_index = next(
        op.index for op in transcript.operations if op.operation_class != "QUERY"
    )

    with pytest.raises(LeakageError):
        transcript.with_equality_channel({queries[0].index: "7", **{
            op.index: full[op.index] for op in queries[1:]
        }})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({**full, queries[0].index: 7})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({**full, queries[0].index: None})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({**full, queries[0].index: "key:7"})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({**full, non_query_index: "QClass0"})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel({**full, 999: "QClass0"})
    with pytest.raises(LeakageError):
        transcript.with_equality_channel(full, channel="M0")

    annotated = transcript.with_equality_channel(full)
    for operation in queries:
        assert annotated.operation(operation.index).query_equality_class == full[operation.index]
    assert annotated.operation(non_query_index).to_dict().get("query_equality_class") is None
    assert "query_equality_class" not in annotated.operation(non_query_index).to_dict()


def test_with_equality_channel_validates_every_query_value(artifacts):
    """A complete key set must not let a null-like value bypass label validation."""
    for family, artifact in artifacts.items():
        transcript = extract_transcript(artifact)
        queries = transcript.query_operations
        full = {op.index: f"QClass{position % 2}" for position, op in enumerate(queries)}
        assert queries and set(full) == {op.index for op in queries}

        for index in (queries[0].index, queries[-1].index):
            for bad_value in (None, 7, True, False, "7", "key:7", [""], {}):
                broken = {**full, index: bad_value}
                # the key set is still complete: only the value is wrong
                assert set(broken) == {op.index for op in queries}
                with pytest.raises(LeakageError):
                    transcript.with_equality_channel(broken)

        # a complete, valid mapping still works and can only produce a schema-valid M1 view
        annotated = transcript.with_equality_channel(full)
        assert annotated.equality_channel == EQUALITY_CHANNEL_M1
        assert all(
            op.query_equality_class == full[op.index]
            for op in annotated.query_operations
        )
        assert all(
            op.query_equality_class is None
            for op in annotated.operations
            if op.operation_class != "QUERY"
        )
        payload = annotated.to_dict()
        validate_transcript(payload)
        assert AdversaryTranscript.from_dict(payload).to_dict() == payload
        for op in payload["operations"]:
            if op["class"] == "QUERY":
                assert "query_equality_class" in op
            else:
                assert "query_equality_class" not in op


def test_m1_equality_classes_require_the_workload_script():
    artifact = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    artifact.pop("script")
    with pytest.raises(LeakageError):
        PrivilegedEvaluator.m1_equality_classes(artifact)
    mismatched = synthetic_artifact([("QUERY", [("READ", 1)], (7,))])
    mismatched["script"]["operations"][0]["kind"] = "MERGE"
    with pytest.raises(LeakageError):
        PrivilegedEvaluator.m1_equality_classes(mismatched)


# ---------------------------------------------------------------------------
# [8] the M0 transcript is unchanged when no equality channel is requested
# ---------------------------------------------------------------------------


def test_m0_transcript_is_unchanged_and_metrics_ignore_the_m1_channel(artifacts):
    for family, artifact in artifacts.items():
        base = extract_transcript(artifact)
        before = dumps_transcript(base)
        annotated = PrivilegedEvaluator.m1_equality_transcript(artifact)
        assert dumps_transcript(base) == before  # base transcript not mutated
        assert base.equality_channel == EQUALITY_CHANNEL_M0
        assert all(op.query_equality_class is None for op in base.operations)
        # the optional channel changes no metric: baseline metrics are M0-only
        assert dumps_metrics(build_metrics(base)) == dumps_metrics(
            build_metrics(annotated)
        )
        assert build_metrics(annotated)["equality_channel"] == EQUALITY_CHANNEL_M0
        # only the QUERY events differ in annotation; the events themselves never do
        assert [op.to_dict()["events"] for op in annotated.operations] == [
            op.to_dict()["events"] for op in base.operations
        ]


# ---------------------------------------------------------------------------
# [9] per-operation metrics match independently derived expectations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["Q", "UQ", "UMQ"])
def test_per_operation_metrics_match_independent_derivation(family, artifacts, metrics):
    artifact = artifacts[family]
    per_operation = metrics[family]["per_operation"]
    assert len(per_operation) == len(artifact["operations"])
    for position, source in enumerate(artifact["operations"]):
        slots = [event["slot_id"] for event in source["events"]]
        counts = Counter(slots)
        expected = {
            "index": source["index"],
            "class": source["kind"],
            "event_count": len(slots),
            "read_volume": sum(1 for e in source["events"] if e["operation"] == "READ"),
            "write_volume": sum(1 for e in source["events"] if e["operation"] == "WRITE"),
            "distinct_slots": len(counts),
            "repeated_slot_count": sum(1 for c in counts.values() if c > 1),
            "repeated_slot_ids": sorted(s for s, c in counts.items() if c > 1),
            "repeat_access_count": len(slots) - len(counts),
            "slot_sequence": slots,
            "min_slot": min(slots) if slots else None,
            "max_slot": max(slots) if slots else None,
            "physical_span": (max(slots) - min(slots) + 1) if slots else None,
        }
        assert per_operation[position] == expected


def test_per_operation_metrics_of_an_empty_and_a_repeating_operation():
    artifact = synthetic_artifact([
        ("BUILD_PGM", [], ()),
        ("QUERY", [("READ", 4), ("READ", 4), ("READ", 9)], (1,)),
        ("QUERY", [], (2,)),
    ])
    transcript = extract_transcript(artifact)
    empty = operation_metrics(transcript.operation(0))
    assert empty["event_count"] == 0
    assert empty["distinct_slots"] == 0
    assert empty["repeated_slot_count"] == 0
    assert empty["slot_sequence"] == []
    assert empty["min_slot"] is None and empty["max_slot"] is None
    assert empty["physical_span"] is None

    repeated = operation_metrics(transcript.operation(1))
    assert repeated["slot_sequence"] == [4, 4, 9]
    assert repeated["distinct_slots"] == 2
    assert repeated["repeated_slot_ids"] == [4]
    assert repeated["repeated_slot_count"] == 1
    assert repeated["repeat_access_count"] == 1
    assert (repeated["min_slot"], repeated["max_slot"], repeated["physical_span"]) == (4, 9, 6)

    assert per_operation_metrics(transcript) == [
        operation_metrics(op) for op in transcript.operations
    ]


# ---------------------------------------------------------------------------
# [10] query pairwise overlap is computed from event sets only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["Q", "UQ", "UMQ"])
def test_query_pairwise_overlap_matches_independent_oracle(family, artifacts, metrics):
    artifact = artifacts[family]
    query_indices = [
        source["index"] for source in artifact["operations"] if source["kind"] == "QUERY"
    ]
    slot_sets = {index: set(artifact_events(artifact, index)) for index in query_indices}
    sequences = {
        index: tuple(artifact_events(artifact, index)) for index in query_indices
    }

    expected_pairs = []
    for first, second in itertools.combinations(query_indices, 2):
        expected_pairs.append({
            "index_a": first,
            "index_b": second,
            "intersection_size": len(slot_sets[first] & slot_sets[second]),
            "union_size": len(slot_sets[first] | slot_sets[second]),
            "jaccard": jaccard(slot_sets[first], slot_sets[second]),
            "set_equal": slot_sets[first] == slot_sets[second],
            "sequence_equal": sequences[first] == sequences[second],
        })

    pairwise = metrics[family]["query"]["pairwise"]
    assert pairwise["pairs"] == expected_pairs
    assert pairwise["pair_count"] == len(expected_pairs)
    assert pairwise["exact_set_equality_count"] == sum(
        1 for pair in expected_pairs if pair["set_equal"]
    )
    assert pairwise["exact_sequence_equality_count"] == sum(
        1 for pair in expected_pairs if pair["sequence_equal"]
    )
    assert pairwise["pairs_with_intersection_count"] == sum(
        1 for pair in expected_pairs if pair["intersection_size"] > 0
    )
    if expected_pairs:
        jaccards = [pair["jaccard"] for pair in expected_pairs]
        assert pairwise["mean_jaccard"] == round(sum(jaccards) / len(jaccards), 6)
        assert pairwise["max_jaccard"] == max(jaccards)
        assert pairwise["min_jaccard"] == min(jaccards)
        assert pairwise["exact_sequence_equality_rate"] == round(
            sum(1 for pair in expected_pairs if pair["sequence_equal"]) / len(expected_pairs),
            6,
        )


def test_empty_trace_pairs_are_reported_explicitly():
    artifact = synthetic_artifact([
        ("QUERY", [], (1,)),
        ("QUERY", [], (2,)),
    ])
    pairwise = query_metrics(extract_transcript(artifact))["pairwise"]
    assert pairwise["pair_count"] == 1
    assert pairwise["pairs"][0]["intersection_size"] == 0
    assert pairwise["pairs"][0]["union_size"] == 0
    assert pairwise["pairs"][0]["jaccard"] == 0.0          # documented 0/0 convention
    assert pairwise["pairs"][0]["set_equal"] is True       # empty sets are equal
    assert pairwise["pairs"][0]["sequence_equal"] is True


# ---------------------------------------------------------------------------
# [11] sequence equality uses the observed trace, never the query key
# ---------------------------------------------------------------------------


def test_sequence_equality_uses_trace_sequence_not_query_key():
    equal_traces = [("READ", 5), ("READ", 6)]
    artifact = synthetic_artifact([
        ("QUERY", equal_traces, (111,)),
        ("QUERY", equal_traces, (222,)),          # different key, identical trace
        ("QUERY", [("READ", 6), ("READ", 5)], (333,)),
    ])
    transcript = extract_transcript(artifact)
    pairwise = query_metrics(transcript)["pairwise"]

    by_pair = {(pair["index_a"], pair["index_b"]): pair for pair in pairwise["pairs"]}
    assert by_pair[(0, 1)]["sequence_equal"] is True
    assert by_pair[(0, 1)]["set_equal"] is True
    assert by_pair[(0, 2)]["sequence_equal"] is False
    assert by_pair[(1, 2)]["sequence_equal"] is False
    assert by_pair[(0, 2)]["set_equal"] is True
    assert pairwise["exact_sequence_equality_count"] == 1
    assert pairwise["exact_set_equality_count"] == 3
    assert pairwise["exact_sequence_equality_rate"] == round(1 / 3, 6)

    # the same operations have three *different* plaintext keys and three
    # different equality classes, so the metric cannot be reading the key
    labels = PrivilegedEvaluator.m1_equality_classes(artifact)
    assert labels == {0: "QClass0", 1: "QClass1", 2: "QClass2"}


# ---------------------------------------------------------------------------
# [12] slot hotness histogram is deterministic and event-derived
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["Q", "UQ", "UMQ"])
def test_slot_hotness_histogram_is_deterministic(family, artifacts, metrics):
    artifact = artifacts[family]
    query_indices = [
        source["index"] for source in artifact["operations"] if source["kind"] == "QUERY"
    ]
    slot_query_counts = Counter()
    access_counts = Counter()
    for index in query_indices:
        slots = artifact_events(artifact, index)
        slot_query_counts.update(set(slots))
        access_counts.update(slots)

    hotness = metrics[family]["query"]["slot_hotness"]
    assert hotness["slot_query_frequency"] == [
        {"slot_id": slot, "query_count": slot_query_counts[slot],
         "access_count": access_counts[slot]}
        for slot in sorted(slot_query_counts)
    ]
    histogram = Counter(slot_query_counts.values())
    assert hotness["hotness_histogram"] == [
        {"query_count": count, "slot_count": histogram[count]}
        for count in sorted(histogram)
    ]
    assert hotness["distinct_query_slots"] == len(slot_query_counts)
    assert hotness["total_query_slot_accesses"] == sum(access_counts.values())
    assert hotness["repeated_query_slot_count"] == sum(
        1 for count in slot_query_counts.values() if count > 1
    )
    assert hotness["max_slot_query_count"] == max(slot_query_counts.values())
    assert json.dumps(hotness) == json.dumps(
        query_metrics(extract_transcript(artifact))["slot_hotness"]
    )


# ---------------------------------------------------------------------------
# [13] adjacency is derived from the observed slot sequence only
# ---------------------------------------------------------------------------


def test_adjacency_metric_is_derived_from_the_observed_sequence():
    artifact = synthetic_artifact([
        ("QUERY", [("READ", 3), ("READ", 4), ("READ", 9), ("READ", 10)], (1,)),
        ("QUERY", [("READ", 2), ("READ", 1)], (2,)),
    ])
    adjacency = query_metrics(extract_transcript(artifact))["adjacency"]
    assert adjacency["total_consecutive_steps"] == 3 + 1
    assert adjacency["adjacent_by_one_count"] == 2 + 1
    assert adjacency["adjacent_by_one_rate"] == round(3 / 4, 6)
    assert adjacency["step_difference_histogram"] == [
        {"difference": -1, "count": 1},
        {"difference": 1, "count": 2},
        {"difference": 5, "count": 1},
    ]
    assert adjacency["per_query"] == [
        {"index": 0, "consecutive_steps": 3, "adjacent_by_one_count": 2},
        {"index": 1, "consecutive_steps": 1, "adjacent_by_one_count": 1},
    ]

    # no adjacency steps when a trace is empty or a single event
    single = synthetic_artifact([
        ("QUERY", [("READ", 3)], (1,)),
        ("QUERY", [], (2,)),
    ])
    empty_adjacency = query_metrics(extract_transcript(single))["adjacency"]
    assert empty_adjacency["total_consecutive_steps"] == 0
    assert empty_adjacency["adjacent_by_one_count"] == 0
    assert empty_adjacency["adjacent_by_one_rate"] == 0.0
    assert empty_adjacency["step_difference_histogram"] == []


@pytest.mark.parametrize("family", ["Q", "UQ", "UMQ"])
def test_adjacency_matches_independent_derivation(family, artifacts, metrics):
    artifact = artifacts[family]
    differences = []
    adjacent = 0
    for source in artifact["operations"]:
        if source["kind"] != "QUERY":
            continue
        slots = [event["slot_id"] for event in source["events"]]
        for previous, following in zip(slots, slots[1:]):
            difference = following - previous
            differences.append(difference)
            if abs(difference) == 1:
                adjacent += 1
    adjacency = metrics[family]["query"]["adjacency"]
    assert adjacency["total_consecutive_steps"] == len(differences)
    assert adjacency["adjacent_by_one_count"] == adjacent
    histogram = Counter(differences)
    assert adjacency["step_difference_histogram"] == [
        {"difference": difference, "count": histogram[difference]}
        for difference in sorted(histogram)
    ]


# ---------------------------------------------------------------------------
# [14] merge / prior-query overlap uses only observed transcripts
# ---------------------------------------------------------------------------


def test_merge_prior_query_overlap_uses_only_observed_transcripts():
    artifact = synthetic_artifact([
        ("QUERY", [("READ", 1), ("READ", 4)], (1,)),
        ("MERGE", [("READ", 1), ("READ", 2), ("WRITE", 3), ("WRITE", 1),
                   ("WRITE", 2), ("READ", 3)], ()),
        ("QUERY", [("READ", 3), ("READ", 8)], (2,)),
        ("MERGE", [("WRITE", 8), ("READ", 8)], ()),
    ])
    merged = merge_metrics(extract_transcript(artifact))
    assert merged["merge_count"] == 2
    assert merged["total_reads"] == 4
    assert merged["total_writes"] == 4
    assert merged["distinct_slots"] == len({1, 2, 3, 8})
    assert merged["sum_prior_query_overlap_slots"] == 2  # slot 1, then slot 8

    first, second = merged["per_merge"]
    assert first["index"] == 1
    assert first["event_count"] == 6
    assert first["read_volume"] == 3 and first["write_volume"] == 3
    assert first["distinct_slots"] == 3
    assert first["distinct_read_slots"] == 3 and first["distinct_write_slots"] == 3
    assert (first["min_slot"], first["max_slot"], first["physical_span"]) == (1, 3, 3)
    assert first["operation_runs"] == [
        {"operation": "READ", "count": 2},
        {"operation": "WRITE", "count": 3},
        {"operation": "READ", "count": 1},
    ]
    assert first["read_run_count"] == 2 and first["write_run_count"] == 1
    assert first["max_read_run"] == 2 and first["max_write_run"] == 3
    assert first["prior_query_overlap"] == {
        "prior_query_count": 1,
        "prior_query_slots": 2,
        "merge_slots": 3,
        "overlap_slots": 1,
        "overlap_ratio": round(1 / 3, 6),
    }

    assert second["index"] == 3
    assert second["prior_query_overlap"] == {
        "prior_query_count": 2,
        "prior_query_slots": 4,          # {1, 4} then {3, 8}
        "merge_slots": 1,                # {8}
        "overlap_slots": 1,
        "overlap_ratio": 1.0,
    }
    assert merged["mean_overlap_ratio"] == round(
        (round(1 / 3, 6) + 1.0) / 2, 6
    )
    assert merged["max_overlap_ratio"] == 1.0

    # merging touches nothing a prior query touched when there is no prior query
    later_only = synthetic_artifact([
        ("MERGE", [("WRITE", 1)], ()),
        ("QUERY", [("READ", 1)], (1,)),
    ])
    lone = merge_metrics(extract_transcript(later_only))["per_merge"][0]
    assert lone["prior_query_overlap"] == {
        "prior_query_count": 0,
        "prior_query_slots": 0,
        "merge_slots": 1,
        "overlap_slots": 0,
        "overlap_ratio": 0.0,
    }


@pytest.mark.parametrize("family", ["UMQ", "Q", "UQ"])
def test_merge_metrics_match_independent_derivation(family, artifacts, metrics):
    artifact = artifacts[family]
    prior_slots = set()
    prior_queries = 0
    expected = []
    for source in artifact["operations"]:
        slots = set(artifact_events(artifact, source["index"]))
        if source["kind"] == "QUERY":
            prior_slots |= slots
            prior_queries += 1
            continue
        if source["kind"] != "MERGE":
            continue
        expected.append({
            "index": source["index"],
            "event_count": len(source["events"]),
            "read_volume": source["read_count"],
            "write_volume": source["write_count"],
            "distinct_slots": len(slots),
            "prior_query_overlap": {
                "prior_query_count": prior_queries,
                "prior_query_slots": len(prior_slots),
                "merge_slots": len(slots),
                "overlap_slots": len(slots & prior_slots),
                "overlap_ratio": round(len(slots & prior_slots) / len(slots), 6)
                if slots else 0.0,
            },
        })
    merged = metrics[family]["merge"]
    assert merged["merge_count"] == len(expected)
    assert [
        {key: entry[key] for key in ("index", "event_count", "read_volume", "write_volume",
                                    "distinct_slots", "prior_query_overlap")}
        for entry in merged["per_merge"]
    ] == expected


# ---------------------------------------------------------------------------
# [15] metrics consume the transcript, never the privileged G1-E object
# ---------------------------------------------------------------------------


def test_metrics_consume_the_transcript_only(artifacts, monkeypatch):
    transcript = extract_transcript(artifacts["UMQ"])
    metrics = build_metrics(transcript)
    assert metrics["query"]["query_count"] == transcript.summary["query_count"]

    # the raw G1-E artifact (or any other object) is not a valid metric input
    with pytest.raises(AttributeError):
        build_metrics(artifacts["UMQ"])
    with pytest.raises(AttributeError):
        query_metrics({"operations": []})
    with pytest.raises(AttributeError):
        merge_metrics(None)

    # metrics never re-extract or otherwise touch privileged extraction
    def explode(*args, **kwargs):
        raise AssertionError("metrics must not call privileged extraction")

    monkeypatch.setattr(leakage, "extract_transcript", explode)
    monkeypatch.setattr(leakage.PrivilegedEvaluator, "m1_equality_classes", explode)
    assert dumps_metrics(build_metrics(transcript)) == dumps_metrics(metrics)

    # and the module cannot import the harness at all
    source = Path(leakage_metrics.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    forbidden = {
        "enhanced_letindex.workload", "enhanced_letindex.engine",
        "enhanced_letindex.merge", "enhanced_letindex.builder",
        "enhanced_letindex.pgm", "enhanced_letindex.dataset",
        "enhanced_letindex.storage", ".workload", ".engine", ".merge", ".builder",
        ".pgm", ".dataset", ".storage",
    }
    assert imported & forbidden == set()
    assert imported <= {".leakage", "collections", "typing", "__future__"}


# ---------------------------------------------------------------------------
# [16] transcript / metrics schemas contain only documented primitive fields
# ---------------------------------------------------------------------------


def test_transcript_schema_uses_only_documented_primitive_fields(artifacts):
    for family in artifacts:
        transcript = extract_transcript(artifacts[family])
        payload = transcript.to_dict()
        assert set(payload) == TRANSCRIPT_KEYS
        assert set(payload["run_metadata"]) == RUN_METADATA_KEYS
        assert set(payload["run_metadata"]["config"]) == RUN_CONFIG_KEYS
        assert set(payload["summary"]) == SUMMARY_KEYS
        for operation in payload["operations"]:
            assert set(operation) == OPERATION_KEYS
            for event in operation["events"]:
                assert set(event) == EVENT_KEYS
                assert event["operation"] in {"READ", "WRITE"}
                assert isinstance(event["seq"], int) and isinstance(event["slot_id"], int)
        annotated = PrivilegedEvaluator.m1_equality_transcript(artifacts[family])
        for operation in annotated.to_dict()["operations"]:
            assert set(operation) in (OPERATION_KEYS, OPERATION_KEYS_M1)
        for leaf in recursive_leaves(payload):
            assert leaf is None or isinstance(leaf, (str, int, float, bool))


def test_metrics_schema_uses_only_documented_primitive_fields(artifacts, metrics):
    for family in artifacts:
        payload = metrics[family]
        assert set(payload) == METRICS_KEYS
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["source_schema_version"] == SOURCE_SCHEMA_VERSION
        assert payload["transcript_schema_version"] == SCHEMA_VERSION
        for operation in payload["per_operation"]:
            assert set(operation) == PER_OPERATION_KEYS
        query = payload["query"]
        assert set(query) == QUERY_METRIC_KEYS
        assert set(query["trace_length"]) == DISTRIBUTION_KEYS
        assert set(query["distinct_slots"]) == DISTRIBUTION_KEYS
        assert set(query["pairwise"]) == PAIRWISE_KEYS
        for pair in query["pairwise"]["pairs"]:
            assert set(pair) == PAIR_KEYS
        assert set(query["slot_hotness"]) == SLOT_HOTNESS_KEYS
        assert set(query["adjacency"]) == ADJACENCY_KEYS
        merge = payload["merge"]
        assert set(merge) == MERGE_METRIC_KEYS
        for entry in merge["per_merge"]:
            assert set(entry) == PER_MERGE_KEYS
            assert set(entry["prior_query_overlap"]) == PRIOR_OVERLAP_KEYS
            for run in entry["operation_runs"]:
                assert set(run) == {"operation", "count"}
        # no phase labels anywhere in the merge summary
        assert "phase" not in dumps_metrics(payload)
        for leaf in recursive_leaves(payload):
            assert leaf is None or isinstance(leaf, (str, int, float, bool))
        assert recursive_keys(payload) & FORBIDDEN_KEYS == set()


def test_metrics_of_a_query_only_or_merge_only_transcript_are_complete():
    only_queries = extract_transcript(synthetic_artifact([
        ("QUERY", [("READ", 1)], (1,)),
    ]))
    metrics = build_metrics(only_queries)
    assert set(metrics["merge"]) == MERGE_METRIC_KEYS
    assert metrics["merge"]["merge_count"] == 0
    assert metrics["merge"]["per_merge"] == []

    only_merges = extract_transcript(synthetic_artifact([
        ("MERGE", [("WRITE", 1)], ()),
    ]))
    metrics = build_metrics(only_merges)
    assert set(metrics["query"]) == QUERY_METRIC_KEYS
    assert metrics["query"]["query_count"] == 0
    assert metrics["query"]["pairwise"]["pairs"] == []
    assert metrics["query"]["trace_length"]["histogram"] == []
    assert metrics["query"]["trace_length"]["min"] is None
    assert metrics["query"]["adjacency"]["per_query"] == []


# ---------------------------------------------------------------------------
# [17] the CLI equals the direct API, and refuses bad input loudly
# ---------------------------------------------------------------------------


def cli_env():
    src = str(Path(leakage.__file__).parents[1])
    return {**os.environ, "PYTHONPATH": src}


def test_cli_output_equals_direct_api_output(tmp_path, artifacts):
    artifact_path = tmp_path / "g1e_run.json"
    artifact_path.write_text(json.dumps(artifacts["UMQ"]), encoding="utf-8")
    transcript_path = tmp_path / "transcript.json"
    metrics_path = tmp_path / "metrics.json"

    exit_code = main([
        "--input", str(artifact_path),
        "--transcript-out", str(transcript_path),
        "--metrics-out", str(metrics_path),
    ])
    assert exit_code == 0

    transcript = extract_transcript(artifacts["UMQ"])
    expected_transcript = dumps_transcript(transcript) + "\n"
    expected_metrics = dumps_metrics(build_metrics(transcript)) + "\n"
    assert transcript_path.read_text(encoding="utf-8") == expected_transcript
    assert metrics_path.read_text(encoding="utf-8") == expected_metrics

    # the direct API writers produce exactly the same bytes
    direct_transcript = tmp_path / "direct_transcript.json"
    direct_metrics = tmp_path / "direct_metrics.json"
    write_transcript_json(transcript, direct_transcript)
    write_metrics_json(build_metrics(transcript), direct_metrics)
    assert direct_transcript.read_bytes() == transcript_path.read_bytes()
    assert direct_metrics.read_bytes() == metrics_path.read_bytes()

    # and so does the real CLI subprocess
    process = subprocess.run(
        [sys.executable, "-m", "enhanced_letindex.leakage",
         "--input", str(artifact_path),
         "--transcript-out", str(tmp_path / "sub_transcript.json"),
         "--metrics-out", str(tmp_path / "sub_metrics.json")],
        cwd=str(tmp_path), env=cli_env(), capture_output=True, text=True,
    )
    assert process.returncode == 0, process.stderr
    assert (tmp_path / "sub_transcript.json").read_bytes() == transcript_path.read_bytes()
    assert (tmp_path / "sub_metrics.json").read_bytes() == metrics_path.read_bytes()
    assert "equality_channel=M0" in process.stdout


def test_cli_defaults_to_no_equality_labels(tmp_path, artifacts):
    artifact_path = tmp_path / "g1e_run.json"
    artifact_path.write_text(json.dumps(artifacts["Q"]), encoding="utf-8")
    assert main(["--input", str(artifact_path),
                 "--transcript-out", str(tmp_path / "t.json"),
                 "--metrics-out", str(tmp_path / "m.json")]) == 0
    payload = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert payload["equality_channel"] == EQUALITY_CHANNEL_M0
    assert all("query_equality_class" not in op for op in payload["operations"])

    # the optional evaluator-side M1 channel is opt-in and stays opaque
    assert main(["--input", str(artifact_path), "--equality-channel", "m1",
                 "--transcript-out", str(tmp_path / "t1.json"),
                 "--metrics-out", str(tmp_path / "m1.json")]) == 0
    annotated = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))
    assert annotated["equality_channel"] == EQUALITY_CHANNEL_M1
    labels = {
        op["query_equality_class"] for op in annotated["operations"]
        if "query_equality_class" in op
    }
    assert labels and all(label.startswith("QClass") for label in labels)
    assert json.loads((tmp_path / "m1.json").read_text(encoding="utf-8")) == json.loads(
        (tmp_path / "m.json").read_text(encoding="utf-8")
    )


def test_cli_refuses_malformed_artifacts_loudly(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "g1e-2", "operations": []}),
                   encoding="utf-8")
    exit_code = main(["--input", str(bad),
                      "--transcript-out", str(tmp_path / "t.json"),
                      "--metrics-out", str(tmp_path / "m.json")])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "t.json").exists()
    assert not (tmp_path / "m.json").exists()

    missing = main(["--input", str(tmp_path / "nope.json"),
                    "--transcript-out", str(tmp_path / "t2.json"),
                    "--metrics-out", str(tmp_path / "m2.json")])
    assert missing == 2

    garbage = tmp_path / "garbage.json"
    garbage.write_text("[]", encoding="utf-8")
    assert main(["--input", str(garbage),
                 "--transcript-out", str(tmp_path / "t3.json"),
                 "--metrics-out", str(tmp_path / "m3.json")]) == 2


# ---------------------------------------------------------------------------
# [18] representative Q / UQ / UMQ G1-E artifacts all extract successfully
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,seed,capacity", FAMILY_FIXTURES)
def test_representative_families_extract_successfully(family, seed, capacity, artifacts,
                                                     transcripts, metrics):
    transcript = transcripts[family]
    artifact = artifacts[family]
    recount = Counter(source["kind"] for source in artifact["operations"])
    assert transcript.summary == {
        "operation_count": len(artifact["operations"]),
        "query_count": recount["QUERY"],
        "merge_count": recount["MERGE"],
        "total_reads": sum(source["read_count"] for source in artifact["operations"]),
        "total_writes": sum(source["write_count"] for source in artifact["operations"]),
        "distinct_slots": len({
            event["slot_id"] for source in artifact["operations"]
            for event in source["events"]
        }),
    }
    payload = metrics[family]
    assert len(payload["per_operation"]) == transcript.summary["operation_count"]
    assert payload["query"]["query_count"] == recount["QUERY"]
    assert payload["merge"]["merge_count"] == recount["MERGE"]
    assert dumps_transcript(extract_transcript(artifact)) == dumps_transcript(transcript)


# ---------------------------------------------------------------------------
# [19] existing G0-G1-E behaviour is untouched by G2-A
# ---------------------------------------------------------------------------


def test_g1e_artifacts_and_harness_are_untouched():
    artifact = g1e_artifact("UMQ", 303, capacity=3)
    assert artifact["schema_version"] == SOURCE_SCHEMA_VERSION
    assert artifact["final"]["correctness_ok"] is True
    assert artifact["final"]["oracle_mismatches"] == 0
    assert set(artifact["operations"][0]) == {
        "index", "kind", "operation", "events", "read_count", "write_count",
        "result", "verification", "oracle",
    }
    # the accepted G1-E event export is still exactly seq / operation / slot_id
    for source in artifact["operations"]:
        for event in source["events"]:
            assert set(event) == EVENT_KEYS


# ---------------------------------------------------------------------------
# [20] no attack, recovery or defence code was introduced
# ---------------------------------------------------------------------------


def test_no_attack_recovery_or_defence_code_is_introduced():
    modules = [leakage, leakage_metrics]
    names = set()
    for module in modules:
        names.update(dir(module))
        for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    lowered = {name.lower() for name in names}
    for token in FORBIDDEN_CODE_TOKENS:
        offenders = sorted(name for name in lowered if token in name)
        assert offenders == [], f"{token!r} appears in API names {offenders}"

    public = set(leakage.__all__) | set(leakage_metrics.__all__)
    assert not any(token in name.lower() for name in public
                   for token in FORBIDDEN_CODE_TOKENS)
    # the transcript carries no attack-success style metric fields
    assert recursive_keys(build_metrics(extract_transcript(
        g1e_artifact("UMQ", 303, capacity=3)
    ))) & {"ord_forced", "rank_width", "adj_recall", "attack_budget",
           "success_rate", "leakage_score"} == set()
