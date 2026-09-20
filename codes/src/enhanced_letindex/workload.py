"""G1-E reproducible end-to-end baseline workload harness (decisions/0007).

This module is the Layer-1 experiment/control layer for the accepted vanilla
baseline: it scripts baseline operations, executes them through the **already
accepted public engine API only**, checks every step against an independent
logical oracle, captures operation-scoped access traces, and exports a stable
JSON artifact.

It adds no data-structure, privacy, attack or scheduling mechanism: no automatic
PGM build, no merge trigger/ratio policy, no de-amortization, no dummy/cover
access, no reshuffle, no encryption, and no delete/tombstone semantics.

Public pieces:

* ``WorkloadOperation`` / ``WorkloadScript`` — serializable workload description
  with symbolic level references (``L0``, ``L1``, ...);
* ``generate_workload`` and the ``Q`` / ``UQ`` / ``UMQ`` / ``RANDOM`` family
  generators (deterministic under an explicit seed);
* ``LogicalOracle`` — independent newest-first logical truth model;
* ``BaselineWorkloadRunner`` — executes operations, verifies after every step and
  records per-operation traces;
* ``run_script`` / ``export_dict`` / ``write_run_json`` — execution and stable
  export;
* ``main`` — ``python -m enhanced_letindex.workload`` CLI.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .builder import validate_level
from .config import Config
from .engine import MergeResult, RetiredLevelError, TrustedEngine
from .identifiers import LevelId, RecordKey
from .record import Record
from .trace import TraceEvent, TraceOperation

__all__ = [
    "SCHEMA_VERSION",
    "OPERATION_KINDS",
    "WORKLOAD_FAMILIES",
    "WorkloadError",
    "WorkloadOperation",
    "WorkloadScript",
    "LogicalOracle",
    "OperationRecord",
    "QueryOutcome",
    "BaselineWorkloadRunner",
    "generate_workload",
    "generate_query_workload",
    "generate_update_query_workload",
    "generate_update_merge_query_workload",
    "generate_random_workload",
    "run_script",
    "export_dict",
    "write_run_json",
    "main",
]

SCHEMA_VERSION = "g1e-1"

OP_BUILD_INITIAL = "BUILD_INITIAL"
OP_QUERY = "QUERY"
OP_UPSERT = "UPSERT"
OP_BUILD_PGM = "BUILD_PGM"
OP_MERGE = "MERGE"
OPERATION_KINDS = (
    OP_BUILD_INITIAL,
    OP_QUERY,
    OP_UPSERT,
    OP_BUILD_PGM,
    OP_MERGE,
)

FAMILY_Q = "Q"
FAMILY_UQ = "UQ"
FAMILY_UMQ = "UMQ"
FAMILY_RANDOM = "RANDOM"
WORKLOAD_FAMILIES = (FAMILY_Q, FAMILY_UQ, FAMILY_UMQ, FAMILY_RANDOM)


class WorkloadError(Exception):
    """The workload description or its execution violates the G1-E contract."""


# ---------------------------------------------------------------------------
# workload description (serializable, symbolic level references)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadOperation:
    """One scripted baseline operation.

    Symbolic level references (``ref`` / ``level_ref`` / ``newer_ref`` /
    ``older_ref``) are resolved to real ``LevelId``s at execution time, so a
    script is independent of engine id allocation.  ``keys``/``values`` are
    workload metadata (plaintext ground truth) and never enter the trace.
    """

    index: int
    kind: str
    keys: tuple[int, ...] = ()
    values: tuple[str, ...] = ()
    ref: Optional[str] = None
    level_ref: Optional[str] = None
    newer_ref: Optional[str] = None
    older_ref: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "keys": list(self.keys),
            "values": list(self.values),
            "ref": self.ref,
            "level_ref": self.level_ref,
            "newer_ref": self.newer_ref,
            "older_ref": self.older_ref,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkloadOperation":
        return cls(
            index=int(data["index"]),
            kind=str(data["kind"]),
            keys=tuple(int(k) for k in data.get("keys", ())),
            values=tuple(str(v) for v in data.get("values", ())),
            ref=data.get("ref"),
            level_ref=data.get("level_ref"),
            newer_ref=data.get("newer_ref"),
            older_ref=data.get("older_ref"),
            note=data.get("note", ""),
        )


@dataclass(frozen=True)
class WorkloadScript:
    """A deterministic, serializable baseline workload description."""

    family: str
    seed: int
    capacity: int
    epsilon: int
    operations: tuple[WorkloadOperation, ...]
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "seed": self.seed,
            "capacity": self.capacity,
            "epsilon": self.epsilon,
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkloadScript":
        return cls(
            family=str(data["family"]),
            seed=int(data["seed"]),
            capacity=int(data["capacity"]),
            epsilon=int(data["epsilon"]),
            description=str(data.get("description", "")),
            operations=tuple(
                WorkloadOperation.from_dict(op) for op in data.get("operations", ())
            ),
        )

    def operation_counts(self) -> dict:
        counts = {kind: 0 for kind in OPERATION_KINDS}
        for op in self.operations:
            counts[op.kind] = counts.get(op.kind, 0) + 1
        return counts


class _ScriptBuilder:
    """Deterministic script assembly with symbolic ref/order bookkeeping."""

    def __init__(self, family, seed, capacity, epsilon, description=""):
        self.family = family
        self.seed = seed
        self.capacity = capacity
        self.epsilon = epsilon
        self.description = description
        self._ops: list[WorkloadOperation] = []
        self._next_ref = 0
        self.live: list[str] = []  # newest-first, mirrors the oracle level order

    def new_ref(self) -> str:
        ref = f"L{self._next_ref}"
        self._next_ref += 1
        return ref

    def _add(self, kind, **kwargs) -> WorkloadOperation:
        op = WorkloadOperation(index=len(self._ops), kind=kind, **kwargs)
        self._ops.append(op)
        return op

    def build_initial(self, keys: Sequence[int], tag: Optional[str] = None) -> str:
        ref = self.new_ref()
        tag = ref if tag is None else tag
        keys = tuple(sorted(set(keys)))
        self._add(OP_BUILD_INITIAL, keys=keys,
                  values=tuple(f"{tag}:{k}" for k in keys), ref=ref)
        self.live.insert(0, ref)
        return ref

    def upsert(self, keys: Sequence[int], tag: Optional[str] = None) -> str:
        ref = self.new_ref()
        tag = ref if tag is None else tag
        keys = tuple(sorted(set(keys)))
        self._add(OP_UPSERT, keys=keys,
                  values=tuple(f"{tag}:{k}" for k in keys), ref=ref)
        self.live.insert(0, ref)
        return ref

    def build_pgm(self, ref: str) -> None:
        self._add(OP_BUILD_PGM, level_ref=ref)

    def merge(self, newer_ref: str, older_ref: str) -> str:
        i_newer = self.live.index(newer_ref)
        i_older = self.live.index(older_ref)
        if i_older != i_newer + 1:
            raise WorkloadError(
                f"G1-E merges must be between adjacent levels (newer immediately "
                f"above older); got {newer_ref} at {i_newer} and {older_ref} at "
                f"{i_older} in {self.live}"
            )
        ref = self.new_ref()
        self._add(OP_MERGE, ref=ref, newer_ref=newer_ref, older_ref=older_ref)
        self.live[i_newer : i_older + 1] = [ref]
        return ref

    def query(self, key: int) -> None:
        self._add(OP_QUERY, keys=(int(key),))

    def script(self) -> WorkloadScript:
        return WorkloadScript(
            family=self.family,
            seed=self.seed,
            capacity=self.capacity,
            epsilon=self.epsilon,
            operations=tuple(self._ops),
            description=self.description,
        )


# ---------------------------------------------------------------------------
# deterministic workload families
# ---------------------------------------------------------------------------


def generate_query_workload(seed: int, *, capacity: int = 4, epsilon: int = 2) -> WorkloadScript:
    """Q — query-only workload over three PGM-indexed levels."""
    rng = random.Random(seed)
    b = _ScriptBuilder(FAMILY_Q, seed, capacity, epsilon,
                       "query-only: three levels with PGMs, newest/older hits, "
                       "misses, repeated and rank-adjacent keys")
    newest_keys = sorted(rng.sample(range(0, 40), 6))
    newest = b.build_initial(newest_keys)
    b.build_pgm(newest)
    middle_keys = sorted(rng.sample(range(40, 90), 8))
    middle = b.upsert(middle_keys)
    b.build_pgm(middle)
    oldest_keys = sorted(rng.sample(range(90, 160), 10))
    oldest = b.upsert(oldest_keys)
    b.build_pgm(oldest)

    queries = list(newest_keys[:2])                 # hits in the newest level
    queries += list(middle_keys[:2])                # hits after newer misses
    queries += list(oldest_keys[:2])                # hits in the oldest level
    queries += [rng.randint(400, 500) for _ in range(2)]   # full misses
    queries += [newest_keys[0], newest_keys[0]]     # repeated keys
    for key in newest_keys[2:4] + middle_keys[2:4]:  # rank-adjacent keys
        queries += [key - 1, key + 1]
    for key in queries:
        b.query(key)
    return b.script()


def generate_update_query_workload(
    seed: int, *, capacity: int = 4, epsilon: int = 2, rounds: int = 3
) -> WorkloadScript:
    """UQ — fresh-level upserts interleaved with queries (newest-first)."""
    rng = random.Random(seed)
    b = _ScriptBuilder(FAMILY_UQ, seed, capacity, epsilon,
                       "update+query: fresh-level upserts interleaved with "
                       "queries, showing newest-first semantics before merge")
    base_keys = sorted(rng.sample(range(0, 30), 8))
    base = b.build_initial(base_keys)
    b.build_pgm(base)

    for round_index in range(rounds):
        # deliberate overlap with the existing data so newest-first shadowing
        # (and, after merge, newer-wins) is actually exercised
        overlap = sorted(set(rng.sample(base_keys, 2)) | set(rng.sample(range(0, 30), 2)))
        fresh = b.upsert(overlap)
        b.build_pgm(fresh)
        b.query(overlap[0])                             # newest-first hit
        b.query(overlap[-1] + 1)                        # adjacent query
        b.query(base_keys[round_index % len(base_keys)])  # older-level hit
    b.query(1000)                                        # full miss
    return b.script()


def generate_update_merge_query_workload(
    seed: int, *, capacity: int = 3, epsilon: int = 2, rounds: int = 3
) -> WorkloadScript:
    """UMQ — upserts plus explicit adjacent blocking merges, queried throughout."""
    rng = random.Random(seed)
    b = _ScriptBuilder(FAMILY_UMQ, seed, capacity, epsilon,
                       "update+merge+query: explicit adjacent blocking merges "
                       "with overlapping keys, queried before and after merge")
    base_keys = sorted(rng.sample(range(0, 25), 8))
    live = b.build_initial(base_keys)
    b.build_pgm(live)

    for round_index in range(rounds):
        # guaranteed overlap with the pre-existing data
        overlap = sorted(set(rng.sample(base_keys, 3)) | set(rng.sample(range(0, 25), 2)))
        fresh = b.upsert(overlap, tag=f"U{round_index}")
        b.build_pgm(fresh)
        b.query(overlap[0])                             # pre-merge newest-first
        b.query(base_keys[round_index % len(base_keys)])
        merged = b.merge(fresh, live)                   # adjacent pair
        assert merged is not None
        b.query(overlap[0])                             # post-merge equivalence
        b.query(base_keys[round_index % len(base_keys)])
        b.query(overlap[-1] + 1)
        live = merged
    b.query(9999)                                       # full miss after merges
    return b.script()


def generate_random_workload(
    seed: int, *, capacity: int = 4, epsilon: int = 2, steps: int = 14
) -> WorkloadScript:
    """Mixed randomized-but-deterministic workload used for oracle fuzzing."""
    rng = random.Random(seed)
    b = _ScriptBuilder(FAMILY_RANDOM, seed, capacity, epsilon,
                       "mixed randomized baseline sequence over upserts, "
                       "adjacent merges and queries")
    initial_keys = sorted(rng.sample(range(0, 40), 6))
    live = b.build_initial(initial_keys)
    b.build_pgm(live)
    b.query(initial_keys[0])

    for step in range(steps):
        # every third step forces a merge when possible, so chained merges are
        # always exercised deterministically
        force_merge = step % 3 == 2 and len(b.live) >= 2
        draw = rng.random()
        if (force_merge or draw < 0.35) and len(b.live) >= 2:
            index = rng.randrange(0, len(b.live) - 1)
            b.merge(b.live[index], b.live[index + 1])
            b.query(rng.randrange(0, 45))
        elif draw < 0.7:
            keys = sorted(rng.sample(range(0, 40), rng.randint(1, 5)))
            fresh = b.upsert(keys, tag=f"S{step}")
            b.build_pgm(fresh)
            b.query(keys[0])
        else:
            b.query(rng.randrange(0, 45))
    return b.script()


def generate_workload(
    family: str, seed: int, *, capacity: int = 4, epsilon: int = 2, steps: int = 14
) -> WorkloadScript:
    """Dispatch to the requested G1-E workload family."""
    if family == FAMILY_Q:
        return generate_query_workload(seed, capacity=capacity, epsilon=epsilon)
    if family == FAMILY_UQ:
        return generate_update_query_workload(seed, capacity=capacity, epsilon=epsilon)
    if family == FAMILY_UMQ:
        return generate_update_merge_query_workload(seed, capacity=capacity, epsilon=epsilon)
    if family == FAMILY_RANDOM:
        return generate_random_workload(
            seed, capacity=capacity, epsilon=epsilon, steps=steps
        )
    raise WorkloadError(
        f"unknown workload family {family!r}; expected one of {WORKLOAD_FAMILIES}"
    )


# ---------------------------------------------------------------------------
# independent logical oracle
# ---------------------------------------------------------------------------


class LogicalOracle:
    """Newest-first logical truth model, independent of the engine.

    It never calls ``lookup_level``, ``lookup_levels`` or
    ``merge_levels_blocking``: it only manipulates plain key/value maps.  Merges
    are only well defined between **adjacent** levels (no level strictly between
    the two inputs), because a merged output that carries the newer values must
    not jump above a level that previously shadowed part of its data; the runner
    enforces that precondition.
    """

    def __init__(self) -> None:
        self._levels: list[tuple[str, dict[int, str]]] = []  # index 0 = newest
        self._retired: list[str] = []

    # -- mutation

    def add_newest(self, ref: str, contents: dict[int, str]) -> None:
        if any(existing == ref for existing, _ in self._levels):
            raise WorkloadError(f"oracle already holds level {ref}")
        self._levels.insert(0, (ref, dict(contents)))

    def merge(self, newer_ref: str, older_ref: str, output_ref: str) -> dict:
        refs = self.refs()
        i_newer = refs.index(newer_ref)
        i_older = refs.index(older_ref)
        if i_older != i_newer + 1:
            raise WorkloadError(
                f"oracle merge requires adjacent levels (newer immediately above "
                f"older); got {newer_ref} at {i_newer} and {older_ref} at {i_older}"
            )
        newer_map = dict(self._levels[i_newer][1])
        older_map = dict(self._levels[i_older][1])
        union = dict(older_map)
        union.update(newer_map)  # newer wins duplicates
        self._levels[i_newer : i_older + 1] = [(output_ref, union)]
        self._retired.extend([newer_ref, older_ref])
        return union

    # -- queries

    def refs(self) -> tuple[str, ...]:
        return tuple(ref for ref, _ in self._levels)

    def contents(self, ref: str) -> dict[int, str]:
        for existing, contents in self._levels:
            if existing == ref:
                return dict(contents)
        raise WorkloadError(f"oracle has no level {ref}")

    def record_count(self) -> int:
        return sum(len(contents) for _, contents in self._levels)

    def query(self, key: int) -> Optional[str]:
        for _ref, contents in self._levels:
            if key in contents:
                return contents[key]
        return None

    def searched_refs(self, key: int) -> tuple[str, ...]:
        searched: list[str] = []
        for ref, contents in self._levels:
            searched.append(ref)
            if key in contents:
                break
        return tuple(searched)

    def retired_refs(self) -> tuple[str, ...]:
        return tuple(self._retired)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryOutcome:
    """Logical summary of one query operation, with its oracle expectation."""

    key: int
    found: bool
    value: Optional[str]
    hit_ref: Optional[str]
    searched_refs: tuple[str, ...]
    expected_found: bool
    expected_value: Optional[str]
    expected_searched_refs: tuple[str, ...]
    matches_oracle: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "found": self.found,
            "value": self.value,
            "hit_ref": self.hit_ref,
            "searched_refs": list(self.searched_refs),
            "expected_found": self.expected_found,
            "expected_value": self.expected_value,
            "expected_searched_refs": list(self.expected_searched_refs),
            "matches_oracle": self.matches_oracle,
        }


@dataclass(frozen=True)
class OperationRecord:
    """One executed operation with its own trace window and logical summary."""

    index: int
    kind: str
    operation: WorkloadOperation
    events: tuple[TraceEvent, ...]
    result: dict
    verification: dict
    oracle: Optional[dict] = None

    @property
    def read_count(self) -> int:
        return sum(1 for e in self.events if e.operation == TraceOperation.READ)

    @property
    def write_count(self) -> int:
        return sum(1 for e in self.events if e.operation == TraceOperation.WRITE)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "operation": self.operation.to_dict(),
            "events": [_event_to_dict(e) for e in self.events],
            "read_count": self.read_count,
            "write_count": self.write_count,
            "result": dict(self.result),
            "verification": dict(self.verification),
            "oracle": None if self.oracle is None else dict(self.oracle),
        }


def _event_to_dict(event: TraceEvent) -> dict:
    """Stable primitive trace-event export (no logical ids, no plaintext)."""
    return {
        "seq": event.seq,
        "operation": event.operation.value,
        "slot_id": event.slot_id.value,
    }


class BaselineWorkloadRunner:
    """Executes baseline operations through the accepted public engine API.

    After every mutating operation it verifies engine state against the oracle
    (active level set, retired levels, mapping bijection, canonical structure,
    PGM/queryability declarations); every query is compared with the oracle in
    found/value **and** searched-level semantics.  Canonical validation runs in a
    separate trace window so workload operation traces stay clean.
    """

    def __init__(self, engine: TrustedEngine, seed: int, *, family: str = "adhoc") -> None:
        self.engine = engine
        self.seed = seed
        self.family = family
        self.oracle = LogicalOracle()
        self._refs: dict[str, LevelId] = {}
        self._ref_of: dict[LevelId, str] = {}
        self._queryable: set[LevelId] = set()
        self._records: list[OperationRecord] = []
        self._next_ref = 0
        self._pending_merge: Optional[dict] = None
        self.query_mismatches = 0
        self.merge_mismatches = 0
        self.verification_failures = 0

    # ------------------------------------------------------------ properties

    @property
    def records(self) -> tuple[OperationRecord, ...]:
        return tuple(self._records)

    @property
    def mismatches(self) -> int:
        """Total correctness failures: query + merge + state verification."""
        return (
            self.query_mismatches + self.merge_mismatches + self.verification_failures
        )

    @property
    def correctness_ok(self) -> bool:
        """True only when no query, merge or state-verification failure occurred."""
        return self.mismatches == 0

    def level_id(self, ref: str) -> LevelId:
        try:
            return self._refs[ref]
        except KeyError:
            raise WorkloadError(f"workload has no level ref {ref!r}") from None

    # ----------------------------------------------------------- public ops

    def build_initial(self, records: Sequence[Record]) -> LevelId:
        """BUILD_INITIAL: canonical level construction (G1-A public API)."""
        op = WorkloadOperation(index=len(self._records), kind=OP_BUILD_INITIAL,
                              ref=f"L{self._next_ref}")
        return self._run(op, records=tuple(records))["level_id_value"]

    def upsert(self, records: Sequence[Record]) -> LevelId:
        """UPSERT: publish a fresh newest level (G1-D public API)."""
        op = WorkloadOperation(index=len(self._records), kind=OP_UPSERT,
                              ref=f"L{self._next_ref}")
        return self._run(op, records=tuple(records))["level_id_value"]

    def build_pgm(self, level_id: LevelId) -> int:
        """BUILD_PGM: explicit batch PGM build (G1-B1 public API)."""
        op = WorkloadOperation(index=len(self._records), kind=OP_BUILD_PGM,
                              level_ref=self._ref_of.get(level_id))
        return self._run(op, level_id=level_id)["block_count"]

    def merge(self, newer_level_id: LevelId, older_level_id: LevelId) -> MergeResult:
        """MERGE: explicit blocking two-level merge (G1-D public API)."""
        op = WorkloadOperation(index=len(self._records), kind=OP_MERGE,
                              ref=f"L{self._next_ref}",
                              newer_ref=self._ref_of.get(newer_level_id),
                              older_ref=self._ref_of.get(older_level_id))
        return self._run(op, newer_level_id=newer_level_id,
                         older_level_id=older_level_id)["merge_result"]

    def query(self, key: int) -> QueryOutcome:
        """QUERY: ordered multi-level lookup (G1-C public API) vs the oracle."""
        op = WorkloadOperation(index=len(self._records), kind=OP_QUERY,
                              keys=(int(key),))
        return self._run(op)["query_outcome"]

    # ------------------------------------------------------------ script run

    def run_script(self, script: WorkloadScript) -> tuple[OperationRecord, ...]:
        """Execute every operation of ``script`` in order."""
        for op in script.operations:
            records = None
            if op.kind in (OP_BUILD_INITIAL, OP_UPSERT):
                records = tuple(
                    Record(RecordKey(k), v) for k, v in zip(op.keys, op.values)
                )
            kwargs = {"records": records} if records is not None else {}
            if op.kind == OP_BUILD_PGM:
                kwargs["level_id"] = self.level_id(op.level_ref)
            elif op.kind == OP_MERGE:
                kwargs["newer_level_id"] = self.level_id(op.newer_ref)
                kwargs["older_level_id"] = self.level_id(op.older_ref)
            self._run(op, **kwargs)
        return tuple(self._records)

    # ------------------------------------------------------------- execution

    def _run(self, op: WorkloadOperation, **kwargs) -> dict:
        engine = self.engine
        self._pending_merge = None
        engine.trace.clear()
        try:
            payload = self._dispatch(op, **kwargs)
        finally:
            events = engine.trace.events()
            engine.trace.clear()  # verification reads use their own window

        # Merge verification (and the oracle commit) runs only after the operation
        # trace window is closed, so the MERGE trace stays exactly A/B/C/D.
        oracle_summary = payload.pop("_oracle", None)
        if self._pending_merge is not None:
            oracle_summary = self._finish_merge(op)

        verification = self._verify_state()
        if not self._verification_ok(verification):
            self.verification_failures += 1
        if oracle_summary is not None and not oracle_summary["matches_oracle"]:
            if op.kind == OP_MERGE:
                self.merge_mismatches += 1
            else:
                self.query_mismatches += 1
        # the exported result summary must stay primitive-only: internal objects
        # (LevelId / MergeResult / QueryOutcome) are returned to the caller but
        # never stored in the record that gets serialized
        internal = {
            key: payload.pop(key)
            for key in ("level_id_value", "merge_result", "query_outcome")
            if key in payload
        }
        record = OperationRecord(
            index=len(self._records),
            kind=op.kind,
            operation=op,
            events=tuple(events),
            result=payload,
            verification=verification,
            oracle=oracle_summary,
        )
        self._records.append(record)
        return {**payload, **internal}

    @staticmethod
    def _verification_ok(verification: dict) -> bool:
        return bool(
            verification["active_levels_match_oracle"]
            and verification["retired_levels_inactive"]
            and verification["mapping_bijection"]
            and verification["queryability_matches"]
        )

    def _dispatch(self, op: WorkloadOperation, **kwargs) -> dict:
        if op.kind == OP_BUILD_INITIAL:
            return self._do_build_initial(op, kwargs["records"])
        if op.kind == OP_UPSERT:
            return self._do_upsert(op, kwargs["records"])
        if op.kind == OP_BUILD_PGM:
            return self._do_build_pgm(op, kwargs["level_id"])
        if op.kind == OP_MERGE:
            return self._do_merge(op, kwargs["newer_level_id"], kwargs["older_level_id"])
        if op.kind == OP_QUERY:
            return self._do_query(op)
        raise WorkloadError(f"unknown operation kind {op.kind!r}")

    def _bind_new_level(self, op: WorkloadOperation, level_id: LevelId) -> str:
        ref = op.ref if op.ref is not None else f"L{self._next_ref}"
        self._refs[ref] = level_id
        self._ref_of[level_id] = ref
        self._next_ref += 1
        return ref

    def _do_build_initial(self, op, records) -> dict:
        contents = {r.key.value: r.value for r in records}
        level_id = self.engine.build_level(records)
        ref = self._bind_new_level(op, level_id)
        self.oracle.add_newest(ref, contents)
        return {
            "level_ref": ref,
            "level_id": level_id.value,
            "level_id_value": level_id,
            "record_count": len(contents),
            "block_count": len(self.engine.level(level_id).block_ids),
        }

    def _do_upsert(self, op, records) -> dict:
        contents = {r.key.value: r.value for r in records}
        level_id = self.engine.create_update_level(records)
        ref = self._bind_new_level(op, level_id)
        self.oracle.add_newest(ref, contents)
        return {
            "level_ref": ref,
            "level_id": level_id.value,
            "level_id_value": level_id,
            "record_count": len(contents),
            "block_count": len(self.engine.level(level_id).block_ids),
        }

    def _do_build_pgm(self, op, level_id) -> dict:
        index = self.engine.build_level_pgm(level_id)
        self._queryable.add(level_id)
        return {
            "level_ref": self._ref_of[level_id],
            "level_id": level_id.value,
            "record_count": index.record_count,
            "block_count": len(self.engine.level(level_id).block_ids),
        }

    def _do_merge(self, op, newer_level_id, older_level_id) -> dict:
        """MERGE: preflight the harness contract, then delegate to G1-D.

        The workload-level merge contract is checked **before** the engine is
        touched: both refs must be tracked live levels, the two inputs must be
        distinct, and ``older`` must be exactly one position below ``newer`` in
        the oracle's newest-first order.  A violation raises ``WorkloadError``
        with zero trace events and no engine mutation (no output level, no
        retirement, no mapping/storage change, no oracle change).
        """
        newer_ref = self._ref_of.get(newer_level_id)
        older_ref = self._ref_of.get(older_level_id)
        live_refs = self.oracle.refs()
        if newer_ref is None or older_ref is None:
            raise WorkloadError("merge inputs must be levels tracked by this workload")
        if newer_ref not in live_refs or older_ref not in live_refs:
            raise WorkloadError(
                f"merge inputs must be live levels; got newer={newer_ref} "
                f"older={older_ref} with live={live_refs}"
            )
        if newer_level_id == older_level_id:
            raise WorkloadError("merge inputs must be two distinct levels")
        index_newer = live_refs.index(newer_ref)
        index_older = live_refs.index(older_ref)
        if index_older != index_newer + 1:
            raise WorkloadError(
                "G1-E merges must be between adjacent levels (older immediately "
                f"below newer); got newer={newer_ref} at {index_newer} and "
                f"older={older_ref} at {index_older} in {live_refs}"
            )

        # independent expectations, derived from the oracle BEFORE the engine runs
        newer_map = self.oracle.contents(newer_ref)
        older_map = self.oracle.contents(older_ref)
        expected_union = dict(older_map)
        expected_union.update(newer_map)  # newer wins duplicates
        expected_discarded = len(newer_map.keys() & older_map.keys())

        result = self.engine.merge_levels_blocking(newer_level_id, older_level_id)

        # the output level id and PGM are published by G1-D; the oracle is only
        # committed after the verification step in _finish_merge
        output_ref = self._bind_new_level(op, result.output_level_id)
        self._queryable.add(result.output_level_id)
        self._pending_merge = {
            "result": result,
            "newer_ref": newer_ref,
            "older_ref": older_ref,
            "output_ref": output_ref,
            "newer_count": len(newer_map),
            "older_count": len(older_map),
            "expected_union": expected_union,
            "expected_discarded": expected_discarded,
        }
        return {
            "level_ref": output_ref,
            "level_id": result.output_level_id.value,
            "merge_result": result,
            "newer_ref": newer_ref,
            "older_ref": older_ref,
            "newer_record_count": result.newer_record_count,
            "older_record_count": result.older_record_count,
            "output_record_count": result.output_record_count,
            "discarded_older_record_count": result.discarded_older_record_count,
            "record_count": result.output_record_count,
            "block_count": len(self.engine.level(result.output_level_id).block_ids),
        }

    def _finish_merge(self, op: WorkloadOperation) -> dict:
        """Verify a completed merge against the pre-computed oracle expectation.

        Runs in its own trace window (the MERGE operation trace is already
        closed), compares the G1-D result counts *and* the output level's logical
        contents with the independent expectation, records a primitive summary,
        and only then commits the oracle merge state.
        """
        pending = self._pending_merge
        assert pending is not None
        self._pending_merge = None
        result = pending["result"]
        expected_union = pending["expected_union"]
        expected_discarded = pending["expected_discarded"]

        actual_contents = self._read_level_contents(result.output_level_id)
        counts_match = (
            result.output_record_count == len(expected_union)
            and result.discarded_older_record_count == expected_discarded
            and result.newer_record_count == pending["newer_count"]
            and result.older_record_count == pending["older_count"]
        )
        contents_match = actual_contents == expected_union

        summary = {
            "newer_ref": pending["newer_ref"],
            "older_ref": pending["older_ref"],
            "output_ref": pending["output_ref"],
            "expected_newer_record_count": pending["newer_count"],
            "actual_newer_record_count": result.newer_record_count,
            "expected_older_record_count": pending["older_count"],
            "actual_older_record_count": result.older_record_count,
            "expected_output_record_count": len(expected_union),
            "actual_output_record_count": result.output_record_count,
            "expected_discarded_older_record_count": expected_discarded,
            "actual_discarded_older_record_count": result.discarded_older_record_count,
            "expected_output_key_count": len(expected_union),
            "actual_output_key_count": len(actual_contents),
            "output_contents_match": contents_match,
            "matches_oracle": counts_match and contents_match,
        }

        # commit the oracle merge state after verification, always from the
        # independent expectation (never from engine-returned data)
        self.oracle.merge(pending["newer_ref"], pending["older_ref"], pending["output_ref"])
        return summary

    def _read_level_contents(self, level_id: LevelId) -> dict:
        """Logical contents of a level, read in its own trace window."""
        engine = self.engine
        engine.trace.clear()
        contents: dict[int, str] = {}
        for block_id in engine.level(level_id).block_ids:
            for record in engine.read_block(block_id).records:
                contents[record.key.value] = record.value
        engine.trace.clear()
        return contents

    def _do_query(self, op) -> dict:
        key = op.keys[0]
        plan = [self._refs[ref] for ref in self.oracle.refs()]
        for level_id in plan:
            if self.engine.level(level_id).block_ids and level_id not in self._queryable:
                raise WorkloadError(
                    f"level {self._ref_of[level_id]} is non-empty but has no built "
                    "PGM; G1-E workloads must declare queryable levels explicitly"
                )
        expected_value = self.oracle.query(key)
        expected_searched = self.oracle.searched_refs(key)
        result = self.engine.lookup_levels(plan, RecordKey(key))
        outcome = QueryOutcome(
            key=key,
            found=result.found,
            value=None if result.record is None else result.record.value,
            hit_ref=None if result.hit_level_id is None else self._ref_of[result.hit_level_id],
            searched_refs=tuple(self._ref_of[lid] for lid in result.searched_level_ids),
            expected_found=expected_value is not None,
            expected_value=expected_value,
            expected_searched_refs=expected_searched,
            matches_oracle=(
                result.found == (expected_value is not None)
                and (None if result.record is None else result.record.value) == expected_value
                and tuple(self._ref_of[lid] for lid in result.searched_level_ids)
                == expected_searched
            ),
        )
        return {
            "key": key,
            "searched_level_refs": list(outcome.searched_refs),
            "hit_ref": outcome.hit_ref,
            "found": outcome.found,
            "query_outcome": outcome,
            "_oracle": outcome.to_dict(),
        }

    # ----------------------------------------------------------- verification

    def _verify_state(self) -> dict:
        engine = self.engine
        oracle_refs = self.oracle.refs()
        oracle_active = {self._refs[ref] for ref in oracle_refs}
        actual_active = set(engine.active_level_ids())
        missing = sorted(lid.value for lid in oracle_active - actual_active)
        unexpected = sorted(lid.value for lid in actual_active - oracle_active)
        retired = self.oracle.retired_refs()
        retired_ok = all(
            not engine.is_active_level(self._refs[ref]) for ref in retired
        )
        retired_fail_loudly = True
        for ref in retired:
            try:
                engine.level(self._refs[ref])
                retired_fail_loudly = False
            except RetiredLevelError:
                pass
        engine.mapping.validate_bijection()

        # canonical validation in a dedicated trace window; any extra active level
        # is already reported by the two-way set comparison below
        engine.trace.clear()
        for ref in oracle_refs:
            if engine.is_active_level(self._refs[ref]):
                validate_level(engine, self._refs[ref])
        engine.trace.clear()

        queryable_ok = True
        for level_id in actual_active:
            declared = level_id in self._queryable
            try:
                engine.pgm_index(level_id)
                has_pgm = True
            except KeyError:
                has_pgm = False
            if declared != has_pgm:
                queryable_ok = False

        return {
            "active_levels": [
                ref for ref in oracle_refs if engine.is_active_level(self._refs[ref])
            ],
            "active_level_ids": sorted(lid.value for lid in actual_active),
            "unexpected_active_levels": unexpected,
            "missing_active_levels": missing,
            "active_levels_match_oracle": not missing and not unexpected,
            "retired_levels": list(retired),
            "retired_levels_inactive": retired_ok and retired_fail_loudly,
            "mapping_bijection": True,
            "mapping_size": len(list(engine.mapping.items())),
            "canonical_levels_validated": sum(
                1 for ref in oracle_refs if engine.is_active_level(self._refs[ref])
            ),
            "queryability_matches": queryable_ok,
            "oracle_record_count": self.oracle.record_count(),
        }


# ---------------------------------------------------------------------------
# execution + stable export
# ---------------------------------------------------------------------------


def run_script(script: WorkloadScript) -> BaselineWorkloadRunner:
    """Execute a script on a fresh engine built from the script's config."""
    engine = TrustedEngine(
        Config(block_capacity=script.capacity, pgm_epsilon=script.epsilon)
    )
    runner = BaselineWorkloadRunner(engine, seed=script.seed, family=script.family)
    runner.run_script(script)
    return runner


def _config_dict(config: Config) -> dict:
    return {
        "block_capacity": config.block_capacity,
        "pgm_epsilon": config.pgm_epsilon,
        "random_seed": config.random_seed,
        "lsm_level_ratio": config.lsm_level_ratio,
    }


def export_dict(script: WorkloadScript, runner: BaselineWorkloadRunner) -> dict:
    """Stable, primitive-only JSON payload for one workload run."""
    records = runner.records
    final_levels = []
    for ref in runner.oracle.refs():
        level_id = runner.level_id(ref)
        block_count = len(runner.engine.level(level_id).block_ids)
        final_levels.append({
            "ref": ref,
            "level_id": level_id.value,
            "record_count": len(runner.oracle.contents(ref)),
            "block_count": block_count,
            "queryable": level_id in runner._queryable,
        })
    counts = script.operation_counts()
    query_records = [r for r in records if r.kind == OP_QUERY and r.oracle is not None]
    merge_records = [r for r in records if r.kind == OP_MERGE and r.oracle is not None]
    detailed = {
        "active_levels_match_oracle": all(
            r.verification["active_levels_match_oracle"] for r in records
        ),
        "retired_levels_inactive": all(
            r.verification["retired_levels_inactive"] for r in records
        ),
        "mapping_bijection": all(r.verification["mapping_bijection"] for r in records),
        "queryability_matches": all(
            r.verification["queryability_matches"] for r in records
        ),
        "query_oracle_matches": all(r.oracle["matches_oracle"] for r in query_records),
        "merge_oracle_matches": all(r.oracle["matches_oracle"] for r in merge_records),
    }
    correctness_ok = runner.mismatches == 0 and all(detailed.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "family": script.family,
        "seed": script.seed,
        "config": _config_dict(runner.engine.config),
        "script": script.to_dict(),
        "operations": [record.to_dict() for record in records],
        "final": {
            "active_levels": final_levels,
            "retired_levels": list(runner.oracle.retired_refs()),
            "oracle_record_count": runner.oracle.record_count(),
            "operation_count": len(records),
            "operation_counts": {kind: counts.get(kind, 0) for kind in OPERATION_KINDS},
            "total_read_events": sum(r.read_count for r in records),
            "total_write_events": sum(r.write_count for r in records),
            "oracle_mismatches": runner.mismatches,
            "query_mismatches": runner.query_mismatches,
            "merge_mismatches": runner.merge_mismatches,
            "verification_failures": runner.verification_failures,
            "correctness_ok": correctness_ok,
            "detailed_checks": detailed,
        },
    }


def dumps_run(payload: dict) -> str:
    """Deterministic JSON text for an exported run (no timestamps, no paths)."""
    return json.dumps(payload, indent=2)


def write_run_json(script: WorkloadScript, runner: BaselineWorkloadRunner,
                   path: str | Path) -> dict:
    """Write the stable JSON artifact for one run and return the payload."""
    payload = export_dict(script, runner)
    Path(path).write_text(dumps_run(payload) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m enhanced_letindex.workload",
        description=(
            "Run a deterministic G1-E baseline workload (Q / UQ / UMQ / RANDOM) "
            "through the accepted public engine API and export a stable JSON "
            "artifact.  Correctness is checked against an independent logical "
            "oracle after every operation."
        ),
    )
    parser.add_argument("--family", default=FAMILY_UMQ, choices=list(WORKLOAD_FAMILIES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--epsilon", type=int, default=2)
    parser.add_argument("--steps", type=int, default=14,
                        help="number of randomized steps (RANDOM family only)")
    parser.add_argument("--out", default="g1e_workload_run.json",
                        help="output path for the JSON artifact")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    script = generate_workload(
        args.family, args.seed, capacity=args.capacity, epsilon=args.epsilon,
        steps=args.steps,
    )
    runner = run_script(script)
    payload = write_run_json(script, runner, args.out)
    final = payload["final"]
    counts = final["operation_counts"]
    print(f"family={payload['family']} seed={payload['seed']} "
          f"capacity={args.capacity} epsilon={args.epsilon}")
    print(f"operations={final['operation_count']} "
          f"(query={counts[OP_QUERY]} upsert={counts[OP_UPSERT]} "
          f"build_pgm={counts[OP_BUILD_PGM]} merge={counts[OP_MERGE]} "
          f"build_initial={counts[OP_BUILD_INITIAL]})")
    print(f"read_events={final['total_read_events']} "
          f"write_events={final['total_write_events']}")
    print(f"active_levels={len(final['active_levels'])} "
          f"retired_levels={len(final['retired_levels'])} "
          f"mismatches={final['oracle_mismatches']} "
          f"(query={final['query_mismatches']} merge={final['merge_mismatches']} "
          f"state={final['verification_failures']})")
    print(f"correctness_ok={final['correctness_ok']}")
    print(f"wrote {args.out}")
    return 0 if final["correctness_ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
