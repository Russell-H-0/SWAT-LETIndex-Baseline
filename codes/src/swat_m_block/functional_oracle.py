"""M1: the functional / logical SWAT-M-Block oracle (Issue #2).

M1 answers exactly one question:

    Given two valid adjacent LETIndex logical levels — source ``L`` (newer) and target
    ``L + 1`` (older) — what is the exact logical merged result, its canonical output
    blocks and its canonical PGM?

This module is a **logical correctness oracle only**.  It is the reference result that a
later SWAT-M-Block execution must reproduce; it is not an execution of any SWAT
mechanism, and it makes no privacy or security claim of any kind
(`decisions/0003-m1-functional-oracle.md`).

Frozen logical semantics (Decision 0001 / 0002, inherited unchanged):

```text
C = exact sorted union(source, target)          keys strictly increasing
source only -> source value
target only -> target value
same key    -> source value (newer wins), consume BOTH inputs, emit exactly ONE record
```

Implementation rule — compose, do not re-implement.  The merge, the canonical output
packing and the canonical PGM are the **frozen** common substrate, driven through
:class:`enhanced_letindex.incremental_merge.IncrementalMergeJob`:

* this module contains **no** newer-wins merge loop of its own, and no second packing or
  PGM rule; it only drives the frozen job to completion and collects what the job emits;
* the job's finalized PGM is exactly
  ``build_batch_pgm(tuple(all_output_keys), epsilon)``, so the canonical PGM requirement
  is satisfied by reuse rather than by recomputation;
* retaining the full logical output blocks is allowed here precisely because M1 is a
  *functional* oracle: M1 makes no bounded-memory claim and no physical-execution claim.

Explicitly absent from this module, and out of scope for M1 (Issue #2 §Out of scope):
``DOAllocate`` / ``DOMerge``, noisy or padded bin allocation, dummy or cover I/O,
oblique output permutation, ``PRP`` writeback, physical publication, de-amortisation,
attacks and performance experiments.  There is **zero** ``UntrustedStorage`` access,
**zero** ``TraceEvent``, **zero** physical ``SlotId`` scheduling and **zero** randomness
in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from enhanced_letindex.identifiers import RecordKey
from enhanced_letindex.incremental_merge import (
    IncrementalMergeError,
    IncrementalMergeJob,
    LogicalRunView,
    OutputBlock,
)
from enhanced_letindex.pgm import BatchPgmIndex

__all__ = [
    "DEFAULT_SCHEDULE_POLICY",
    "FunctionalOracleError",
    "FunctionalMergeOracleResult",
    "collect_output_blocks",
    "flatten_blocks",
    "functional_merge_oracle",
    "output_key_stream",
]

#: The default ``advance`` policy of :func:`functional_merge_oracle`: one budget large
#: enough to reach DONE, i.e. the oracle does not depend on a step decomposition.
DEFAULT_SCHEDULE_POLICY = "drain"


class FunctionalOracleError(IncrementalMergeError):
    """An M1 functional-oracle request violates the frozen logical contract.

    Subclasses the frozen :class:`enhanced_letindex.incremental_merge.IncrementalMergeError`
    so that callers have a single error vocabulary: M1-specific refusals are raised here,
    and every refusal coming from the frozen substrate propagates unchanged.
    """


# ---------------------------------------------------------------------------
# pure helpers over emitted blocks
# ---------------------------------------------------------------------------


def flatten_blocks(blocks: Sequence[OutputBlock]) -> Tuple[Tuple[int, object], ...]:
    """The logical record stream of ``blocks``, in rank order (no physical notion)."""
    records: List[Tuple[int, object]] = []
    for block in blocks:
        records.extend(block.records())
    return tuple(records)


def output_key_stream(blocks: Sequence[OutputBlock]) -> Tuple[RecordKey, ...]:
    """The exact output key stream of ``blocks``, in rank order."""
    keys: List[RecordKey] = []
    for block in blocks:
        keys.extend(block.keys)
    return tuple(keys)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _require_view(view: object, name: str) -> LogicalRunView:
    if not isinstance(view, LogicalRunView):
        raise FunctionalOracleError(
            f"the {name} of a SWAT-M-Block merge must be a LogicalRunView, got "
            f"{type(view).__name__}"
        )
    return view


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FunctionalOracleError(
            f"{name} must be a plain int, got {type(value).__name__}"
        )
    if value <= 0:
        raise FunctionalOracleError(f"{name} must be positive, got {value}")
    return value


def _validate_merge_identity(source: LogicalRunView, target: LogicalRunView) -> None:
    """Refuse same-level, reverse and gapped requests before any output exists.

    The frozen job enforces the same identity; M1 states it explicitly so the oracle's
    contract is visible at the oracle boundary, and so a refusal is reported with M1
    wording.  Only ``source = L -> target = L + 1`` is a legal merge.
    """
    if source.level == target.level:
        raise FunctionalOracleError(
            "a SWAT-M-Block merge needs a newer source L and an older target L + 1, got "
            f"the same level twice (L = {source.level}); a same-level merge is refused "
            "before any output is produced"
        )
    if source.level > target.level:
        raise FunctionalOracleError(
            "a SWAT-M-Block merge merges a newer source L into the older target L + 1; "
            f"source {source.level} -> target {target.level} is the reverse direction and "
            "is refused before any output is produced"
        )
    if target.level != source.level + 1:
        raise FunctionalOracleError(
            "a SWAT-M-Block merge only merges numerically adjacent levels, got source "
            f"{source.level} and target {target.level} (a gap of "
            f"{target.level - source.level - 1}); a gapped merge is refused before any "
            "output is produced"
        )


def _validate_schedule(schedule: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(schedule, (str, bytes)):
        raise FunctionalOracleError(
            f"a schedule must be a sequence of positive int budgets, got "
            f"{type(schedule).__name__}"
        )
    budgets = tuple(_require_positive_int(q, "an advance budget") for q in schedule)
    return budgets


def _check_canonical_packing(
    blocks: Sequence[OutputBlock],
    items_per_block: int,
    record_count: int,
    block_count: int,
) -> None:
    """Structural self-check of the canonical packing of the collected blocks.

    This validates the *result*; it decides nothing about record order or values, which
    remain the frozen job's business.
    """
    if len(blocks) != block_count:
        raise FunctionalOracleError(
            f"the merge reported {block_count} blocks but emitted {len(blocks)}"
        )
    if any(block.rank != index for index, block in enumerate(blocks)):
        raise FunctionalOracleError(
            "output block ranks must be exactly 0..B-1 in emission order, got "
            f"{[block.rank for block in blocks]}"
        )
    if sum(block.item_count for block in blocks) != record_count:
        raise FunctionalOracleError(
            f"the merge reported {record_count} records but the blocks hold "
            f"{sum(block.item_count for block in blocks)}"
        )
    for block in blocks[:-1]:
        if block.item_count != items_per_block:
            raise FunctionalOracleError(
                f"output block {block.rank} is not final but holds "
                f"{block.item_count} != {items_per_block} records"
            )
    if blocks:
        final = blocks[-1]
        if not 1 <= final.item_count <= items_per_block:
            raise FunctionalOracleError(
                f"the final output block must hold between 1 and {items_per_block} "
                f"records, got {final.item_count}"
            )
    keys = [key.value for key in output_key_stream(blocks)]
    if keys != sorted(set(keys)):
        raise FunctionalOracleError(
            "output keys must strictly increase across the emitted blocks"
        )


# ---------------------------------------------------------------------------
# the public result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalMergeOracleResult:
    """The immutable M1 result: the exact merged level, logically.

    ``blocks`` are the canonical logical output blocks in rank order and ``pgm`` the
    canonical PGM over the exact output key stream.  No physical slot, storage handle or
    observation event appears anywhere in this result.
    """

    source_level: int
    target_level: int
    items_per_block: int
    epsilon: int
    record_count: int
    block_count: int
    duplicate_count: int
    blocks: Tuple[OutputBlock, ...]
    pgm: BatchPgmIndex
    first_key: Optional[RecordKey]
    last_key: Optional[RecordKey]

    #: The ``advance`` budgets actually applied, in order (evidence of how the frozen job
    #: was driven; the result itself does not depend on them).
    advance_schedule: Tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return self.record_count == 0

    def records(self) -> Tuple[Tuple[int, object], ...]:
        """The flattened logical record stream ``((key, value), ...)``."""
        return flatten_blocks(self.blocks)

    def keys(self) -> Tuple[RecordKey, ...]:
        """The exact output key stream."""
        return output_key_stream(self.blocks)


# ---------------------------------------------------------------------------
# driving the frozen job
# ---------------------------------------------------------------------------


def collect_output_blocks(
    job: IncrementalMergeJob,
    schedule: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[OutputBlock, ...], Tuple[int, ...]]:
    """Drive ``job`` to DONE and collect its emitted blocks.

    Returns ``(blocks, applied_schedule)``.  With ``schedule=None`` the default
    :data:`DEFAULT_SCHEDULE_POLICY` ("drain") is used: a single budget large enough to
    cover every remaining decision.  A caller-supplied schedule is consumed in order, and
    the job is then drained one decision at a time so that the returned blocks are always
    a complete merge.

    The caller owns the returned blocks; the job retains none of them.
    """
    if not isinstance(job, IncrementalMergeJob):
        raise FunctionalOracleError(
            f"collect_output_blocks needs an IncrementalMergeJob, got {type(job).__name__}"
        )
    if schedule is None:
        budgets: Tuple[int, ...] = (
            max(1, job.source.item_count + job.target.item_count),
        )
    else:
        budgets = _validate_schedule(schedule)

    blocks: List[OutputBlock] = []
    applied: List[int] = []
    for budget in budgets:
        if job.done:
            break
        step = job.advance(budget)
        applied.append(budget)
        blocks.extend(step.blocks)
    while not job.done:
        step = job.advance(1)
        applied.append(1)
        blocks.extend(step.blocks)
    return tuple(blocks), tuple(applied)


def functional_merge_oracle(
    source: LogicalRunView,
    target: LogicalRunView,
    *,
    items_per_block: int,
    epsilon: int,
    schedule: Optional[Sequence[int]] = None,
) -> FunctionalMergeOracleResult:
    """Merge ``source`` (newer, level ``L``) into ``target`` (older, level ``L + 1``).

    Returns the exact logical result: the canonical output blocks of
    ``C = exact sorted union(source, target)`` under the frozen newer-wins rule, and the
    canonical PGM over ``C``'s exact key stream.

    ``items_per_block`` is the target level's block capacity and ``epsilon`` the canonical
    PGM epsilon; both are validated by the frozen configuration.  ``schedule`` optionally
    fixes the frozen job's ``advance`` budgets — the *result* must not depend on it, which
    is exactly what the M1 schedule-invariance tests assert.

    Nothing here is randomized, nothing is padded, nothing is written anywhere: the merge
    is decided record by record inside the trusted domain and observed only as its logical
    result.
    """
    source = _require_view(source, "source")
    target = _require_view(target, "target")
    _validate_merge_identity(source, target)

    job = IncrementalMergeJob.begin(
        source,
        target,
        items_per_block=items_per_block,
        epsilon=epsilon,
    )
    blocks, applied = collect_output_blocks(job, schedule)
    merged = job.finalize()
    _check_canonical_packing(
        blocks, job.config.items_per_block, merged.record_count, merged.block_count
    )

    return FunctionalMergeOracleResult(
        source_level=merged.source_level,
        target_level=merged.target_level,
        items_per_block=job.config.items_per_block,
        epsilon=job.config.epsilon,
        record_count=merged.record_count,
        block_count=merged.block_count,
        duplicate_count=merged.duplicate_count,
        blocks=blocks,
        pgm=merged.pgm,
        first_key=merged.first_key,
        last_key=merged.last_key,
        advance_schedule=applied,
    )
