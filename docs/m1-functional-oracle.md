# M1 — the functional SWAT-M-Block oracle

Milestone record for Issue #2.  The normative decision is
`decisions/0003-m1-functional-oracle.md`; this document explains the implementation and
how it is verified.

## 1. What M1 answers

Exactly one question:

> Given two valid adjacent LETIndex logical levels — source `L` (newer) and target `L+1`
> (older) — what is the exact logical merged result, its canonical output blocks and its
> canonical PGM?

Nothing else.  M1 is the reference answer that later, schedule-bearing milestones must
reproduce.

## 2. Public API

```python
from swat_m_block import functional_merge_oracle

result = functional_merge_oracle(
    source,                  # LogicalRunView (level L, newer) - frozen type
    target,                  # LogicalRunView (level L+1, older) - frozen type
    items_per_block=8,       # target level's block capacity, >= 1
    epsilon=8,               # canonical PGM epsilon, >= 0
    schedule=None,           # optional advance() budgets; the result must not depend on it
)
```

`FunctionalMergeOracleResult` (immutable, frozen dataclass):

| Field | Meaning |
|---|---|
| `source_level`, `target_level` | the merged levels (`L`, `L+1`) |
| `items_per_block`, `epsilon` | the validated geometry/PGM parameters |
| `record_count` | number of merged records `\|C\|` |
| `block_count` | number of canonical output blocks |
| `duplicate_count` | number of keys present on both sides |
| `blocks` | the canonical logical output blocks, in rank order |
| `pgm` | the canonical PGM over the exact output key stream |
| `first_key`, `last_key` | the merged key extremes (`None` when empty) |
| `advance_schedule` | the `advance` budgets actually applied (evidence, not semantics) |

plus `is_empty`, `records()`, `keys()`, and the module helpers `flatten_blocks(blocks)`,
`output_key_stream(blocks)`, `collect_output_blocks(job, schedule)`,
`DEFAULT_SCHEDULE_POLICY`, `FunctionalOracleError`.

Errors: `FunctionalOracleError` subclasses the frozen
`enhanced_letindex.incremental_merge.IncrementalMergeError`, so one vocabulary covers both
the M1 boundary refusals and everything the frozen substrate refuses.

## 3. Exact logical semantics

```text
C = exact sorted union(source, target)          keys strictly increasing
source only -> the source record
target only -> the target record
same key    -> the SOURCE value (newer wins); consume BOTH inputs; emit ONE record
```

Keys are arbitrary integers: negative, zero and positive are all legal and ordered
exactly as `RecordKey` orders them.  The result is independent of any future SWAT
bin/noise/shuffle policy — those policies may only change *what is observable*, never `C`.

## 4. How it is implemented (compose, don't re-implement)

```text
LogicalRunView (source L)  ┐
LogicalRunView (target L+1)┘
        |
        v
IncrementalMergeJob.begin(...)        <-- the FROZEN newer-wins merge core
        |
        v
advance(q) ... until DONE             <-- the frozen quantum; schedule is irrelevant
        |            \
        |             \--> emitted canonical OutputBlock values (collected by M1)
        v
finalize()  -------------------------> canonical PGM over the exact output key stream
        |
        v
FunctionalMergeOracleResult           <-- immutable: blocks + PGM + O(1) counters
```

* the newer-wins loop, the canonical packing and the PGM all belong to the frozen
  `enhanced_letindex.incremental_merge` / `enhanced_letindex.pgm` substrate; M1 contains
  no copy of any of them;
* M1 retains the full logical output blocks, which is legitimate because M1 is a
  *functional* oracle: no bounded-memory claim and no physical-execution claim is made;
* after collection, M1 runs one cheap structural self-check of the packing (block count,
  contiguous ranks `0..B-1`, non-final blocks exactly full, final block in
  `[1, capacity]`, strictly increasing keys).  This validates the result; it decides
  nothing about record order or values, which stay the frozen job's business.

## 5. What M1 deliberately does not do

No `DOAllocate` / `DOMerge`, no noisy or padded bin allocation, no dummy or cover block
I/O, no output blinding or permutation, no `PRP` writeback, no physical publication, no
de-amortisation, no attacks, no performance experiments.

Concretely, a full oracle run performs **zero** `UntrustedStorage` operations, produces
**zero** `TraceEvent`s, addresses **zero** physical `SlotId`s, allocates **no** padding,
and uses **no** randomness.  The M1 tests enforce this both statically (the package
imports no physical or defence module and no randomness source; no forbidden mechanism
token appears in M1 executable code) and operationally (every `UntrustedStorage` entry
point is replaced by a failure and `TraceCollector` is instrumented — a complete merge
must call neither).

## 6. Edge-case matrix covered by the tests

Both inputs empty; source empty; target empty; one record each (equal keys and distinct
keys); alternating disjoint keys; source entirely before target and the mirror case;
duplicate-heavy overlap where every key is duplicated; equal keys with differing values
where the source must win; negative/zero/positive keys; output of exactly one full block;
full blocks plus a final partial block; `items_per_block = 1`; multiple PGM segments
(three groups of ten keys with wide gaps at `epsilon = 0`, giving canonical segments
starting at ranks `0, 10, 20` while the output has eight blocks); `epsilon = 0`; several
positive epsilons; invalid same-level, reverse and gapped inputs; malformed runs;
invalid `items_per_block`, `epsilon` and schedules.

## 7. Invariants asserted for every legal fixture

1. the flattened M1 blocks equal an independent dict-based newer-wins oracle;
2. the packing is canonical: every non-final block is full, the final block is in
   `[1, items_per_block]`;
3. block ranks are exactly `0..B-1`;
4. output keys strictly increase;
5. the finalized PGM equals `build_batch_pgm(exact_output_keys, epsilon)` and its
   segmentation equals `make_segmentation(exact_output_keys, epsilon)`;
6. the result is invariant across several `advance(q)` schedules and is bit-identical
   between runs;
7. no physical I/O and no observation event is produced (instrumented).

## 8. Milestones

| Milestone | Content | State |
|---|---|---|
| M0 | repository / bootstrap freeze | **accepted** (Issue #1, CLOSED) |
| **M1** | functional oracle (this document) | **current** (Issue #2) |
| M2 | SWAT-style noisy/padded block-bin allocation | not authorized, **not started** |
| later | dummy/cover I/O, output shuffle, de-amortisation, attacks, experiments | out of scope |
