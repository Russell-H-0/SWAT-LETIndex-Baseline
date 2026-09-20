# M3 — the content-bound SWAT-M-Block bin schedule

Milestone record for Issue #6.  The normative decision is
`decisions/0005-m3-content-bound-bin-schedule.md`; this document explains the implementation.

## 1. What M3 answers

> Given M2's block-bin plan (`decisions/0004`) and the **actual trusted contents** of one
> logical run, which real blocks belong to each bin, what interior point does each bin
> expose, and in what order must the two sides' bins be fetched?

M3 performs no physical I/O, merges nothing, and emits no output.

## 2. The two granularities

```text
observable allocation / later I/O unit = LETIndex block     <- M2 allocates, M4 will fetch
trusted merge comparison unit          = record             <- correctness is decided here
```

M3 flattens the records of a bin's real blocks to validate content and to sample an interior
point.  It never collapses a block into one sortable representative, and it never derives
cross-run order from a block's first/last/midpoint key, its rank, a `BlockId` or a physical
identifier.

## 3. Public API

```python
from swat_m_block import (
    # the trusted view
    LogicalBlockRunView,
    # binding
    BoundBlockBin, BoundBlockAllocation, bind_block_allocation,
    # interior points
    InteriorPoint, interior_point_weights, interior_point_sort_key,
    sample_bin_interior_points, DUMMY_POS_INF,
    # tags and ordering
    TaggedBinInteriorPoint, sorted_tagged_interior_points,
    # the schedule
    AbstractBinRead, SwatBlockMergeSchedule,
    build_abstract_merge_schedule, plan_swat_block_merge_schedule,
    # errors / sides / domains
    ContentScheduleError, SOURCE, TARGET,
    STREAM_DOMAIN_INTERIOR_SOURCE, STREAM_DOMAIN_INTERIOR_TARGET,
)
```

The frozen element type of both views' ``blocks`` attributes lives in the module that defines
it, ``swat_m_block.content_schedule``:

```python
from swat_m_block.content_schedule import LogicalBlockSnapshot
```

Pipeline:

```text
LogicalBlockRunView (level, blocks, items_per_block)
        +  BlockAllocationPlan (M2)
        |
        v
bind_block_allocation(run, plan, side=...)          each bin -> run.blocks[start:stop]
        |
        v
sample_bin_interior_points(bound)                   one pinned interior point per bin
        |                                           (epsilon/seed/domain not caller-set)
        v
build_abstract_merge_schedule(source, target)       preload bin 0 per side, then
        |                                           follow the sorted tagged stream
        v
plan_swat_block_merge_schedule(source_run, target_run,
                               source_plan=..., target_plan=...)   -> SwatBlockMergeSchedule
```

`plan_swat_block_merge_schedule` is the single top-level entry point: it validates the
accepted `source = L` (newer) → `target = L + 1` (older) identity, binds both sides, samples
both sides' interior points from each side's own M2 config (`privacy_epsilon` and `seed`),
and derives the schedule.

The sampling helper's full current signature is

```python
sample_bin_interior_points(bound, *, interior_index_sampler=None)
```

and nothing about the randomness is caller-controlled:

- `privacy_epsilon` is `bound.plan.config.privacy_epsilon` — the epsilon of the frozen plan
  the bin was bound from, never a parameter;
- `seed` is `bound.plan.config.seed`, likewise;
- the RNG domain is fixed by `bound.side` through `interior_stream_domain`: `source` →
  `swat-m-block/interior/source`, `target` → `swat-m-block/interior/target`.

A caller therefore cannot pair an allocation with an unrelated epsilon or seed, reuse one of
the M2 domains, or put both sides on the same stream.  The only exposed knob is the
`interior_index_sampler` test seam, which picks an index inside a bin's already-validated
record list and cannot escape the config/side binding.

## 4. `LogicalBlockRunView` validation

The view preserves the real block boundaries that M1's logical run view deliberately drops.
Construction performs zero storage I/O, snapshots its input at once, and refuses:

- a `level` that is not a plain non-negative int;
- an `items_per_block` that is not a plain int ≥ 1;
- any element that is neither an `enhanced_letindex.Block` nor a `LogicalBlockSnapshot`;
- a block whose capacity differs from `items_per_block`;
- an empty block (a logical run holds non-empty blocks only);
- a block whose keys are not strictly increasing;
- duplicate `BlockId`s;
- neighbouring blocks whose key ranges touch or overlap (`left.max_key < right.min_key` is
  required);
- a non-final block that is not full;
- a final block outside `[1, items_per_block]`.

An empty run (zero blocks) is legal, as is a single block.  `records()`, `keys()` and
`records_of(start, stop)` expose the flattened trusted content.

**The view is immutable, not merely frozen.**  A frozen dataclass holding the caller's
`enhanced_letindex.Block` objects would still be mutable through them, so construction
replaces every input `Block` with the frozen `LogicalBlockSnapshot` in

```text
LogicalBlockSnapshot: block_id, capacity, records (immutable tuple), size,
                      is_empty, is_full, min_key, max_key
```

and validates the snapshots (not the live blocks).  `LogicalBlockRunView.blocks` is a
`Tuple[LogicalBlockSnapshot, ...]`; the caller keeps ownership of their own `Block` objects,
and `Block.add_record(...)` afterwards cannot change `records()`, `keys()`, the binding, the
interior points or the schedule.  The class is importable from
`swat_m_block.content_schedule`; `enhanced_letindex.Block` itself is untouched.  (`Record` and
`RecordKey` are already frozen in the substrate, so copying the tuple is sufficient; record
*values* are opaque payloads and are never inspected by M3.)

## 5. Binding an M2 plan to contents

Legal only when `plan.block_count == run.block_count`.  Each M2 bin's `[start, stop)` binds
`run.blocks[start:stop]`, and `BoundBlockBin` carries the bin index, the rank interval, its
`Tuple[LogicalBlockSnapshot, ...]` (never a reference to a caller-owned `Block`),
`real_block_count`, `dummy_block_count` and `bin_capacity_blocks` — nothing physical.  The
snapshots were taken when the view was constructed, so binding re-validates bin-local state
and cannot be invalidated afterwards.  Binding guarantees:

- every real block rank bound exactly once (`bound_ranks() == 0..n-1`);
- no block duplicated or omitted;
- the M2 bin order preserved;
- the M2 evidence carried through **unchanged** (the properties read straight off the frozen
  M2 plan object, so it cannot be silently rewritten);
- `real_count + dummy_count == bin_capacity_blocks` per bin, with padding kept as a count —
  no dummy block is materialised, and no `dummy_blocks` field exists (`real_count` counts
  real blocks, so the padding count is the only dummy evidence in the binding);
- the bin's flattened keys strictly increase across its blocks (re-validated).

Fixtures with an exhausted run produce trailing bins with `real_block_count == 0`; those bins
are legal, bind no block, and keep their dummy count.

## 6. Interior points (pinned `DPInteriorPoint`)

For a bin with `load` real records, `baseExp = exp(privacy_epsilon)` and

```text
weight[i] = baseExp ** (min(i, load - i) + 1)      for i in [0, load)
```

exactly as pinned — including the asymmetric tail (for `load = 4`: `b, b², b³, b²`) and the
pinned exponentiation-by-squaring evaluation.  The selected index is returned as one of the
bin's **actual** `RecordKey`s; `InteriorPoint.record_index` records which one, so
`bin.real_keys()[point.record_index] == point.key` is asserted for every real bin.

For `real_block_count == 0` there is nothing to sample, and the interior point is
`DUMMY_POS_INF`.

## 7. The `DUMMY_POS_INF` sentinel

A singleton that sorts after every real `RecordKey` (via `interior_point_sort_key`, which
maps a real key to `(0, value)` and the sentinel to `(1, 0)`), is **not** a `RecordKey`, and
is refused by `InteriorPoint.record_key` — so a consumer that needs a database key cannot
silently receive it.  It exists only for trusted abstract scheduling, standing in for the
pinned dummy datum whose key is a numeric maximum (which Python integers do not have).

## 8. Randomness

Planner-owned, deterministic, domain-separated:

```text
swat-m-block/interior/source
swat-m-block/interior/target
```

The domain is chosen internally from `bound.side` (`interior_stream_domain`) and neither it
nor `privacy_epsilon`/`seed` is a public parameter: both come from `bound.plan.config`.
`derive_stream_seed(seed, domain)` (SHA-256 based) makes the two sides' streams different
even when both M2 configs carry the same seed, and makes all four domains in the package
(two interior, two M2) mutually distinct.  No module-global mutable RNG state exists, and no
`random.seed()` is called at module scope.  The schedule is reproducible in-process *and*
across processes; changing values while keeping keys leaves it identical; changing keys may
change the interior evidence.

## 9. Signed tags and the pinned ordering

```text
source bin i -> +(i + 1)        target bin j -> -(j + 1)
sort by (interior_point, signed_tag)
```

i.e. the pinned `std::pair` order: interior point ascending, then signed tag ascending.  On an
equal interior point the **target's negative tag precedes the source's positive tag**, which
the tests pin with a controlled sampler that forces both sides onto the same key.  The sorted
sequence is exposed as `SwatBlockMergeSchedule.tagged_interior_points`.

## 10. The abstract bin-read schedule

```text
preload source bin 0 (if the source has bins)
preload target bin 0 (if the target has bins)
scan the sorted tagged interior points:
    when a side's tag arrives, fetch that side's next unread bin if one exists
```

Consequences, all asserted: two-sided schedules start `(source, 0), (target, 0)`; a
source-only schedule is `source 0..B-1`; a target-only schedule is `target 0..B-1`; a
both-empty schedule is empty; every planned bin of each side appears **exactly once**; no read
falls outside `[0, bin_count)`.  Those invariants are re-derived in the schedule's own
constructor, so a hand-built schedule with a missing, duplicated or out-of-range bin is
refused rather than trusted.

**Relation to the pinned loop — the `else if` fallback is real.**  The pinned `DOMerge` loop
runs `binCnt = leftBinCnt + rightBinCnt` iterations with `j0`/`j1` state, and its
`else if` branch is *not* dead code: when a tag's own side is exhausted it fetches from the
other side.  Example: the source has already fetched its last bin while the target still has
unread bins, and a later source tag arrives — the `left && ...` branch fails and the target's
next bin is fetched.  M3's rule ("a side's next unread bin is fetched when that side's own tag
arrives, if one exists") is therefore **not** the same state machine; it is the *projection* of
the pinned machine onto actual bin fetches, with no-op iterations and the deferred
safe-output/frontier work dropped.  On well-formed input the projected sequences are identical
(asserted by an independent pinned-loop regression over a fixture that really does trigger the
fallback, plus a sweep of ordinary fixtures), because once a side is exhausted no further
same-side fetch can happen, so a fallback fetch only pulls the other side's next bin forward.

**M4 must not assume `1 M3 read == 1 DOMerge iteration`.**  When M4 introduces the `j0`/`j1`
safe-output/frontier state it must replay the original per-tag loop and its fallback, not a
one-read-per-step abstraction of M3's schedule.

A read is an `AbstractBinRead(side, bin_index)` and nothing else — no slot, handle or storage
reference.

## 11. Why M3 is still zero physical I/O

Reading in schedule order straight from each bound block's *current* physical slot would
publish the logical-rank → physical-access correlation and would pre-empt M4's staging
decision.  M3 therefore performs zero `UntrustedStorage` operations, emits zero `TraceEvent`s
and schedules zero slots — enforced statically (token and identifier scans over executable
code) and operationally (every `UntrustedStorage` entry point replaced by a failure,
`TraceCollector` instrumented, four full pipelines — two-sided, source-only, target-only,
empty — executed; neither is called).

## 12. Relationship to M1 and to M2's prefixes

Flattening all real records of all bound bins reconstructs each input run exactly, so M1
remains the sole authority for `C = exact record-level newer-wins sorted union`.  M3
implements no second merge, emits no output blocks, and imports neither
`functional_oracle`, `incremental_merge`, `engine`, `merge` nor `pgm`.

M3 carries M2's `sampled_loads`, `noisy_prefix_sums` and `additive_error_blocks` unchanged
and **in block units**.  The pinned `DOMerge` expression
`newCnt = leftPrefix[...] + rightPrefix[...] - 2 * additiveError` is **not** ported: its
prefixes and error are in datum units, so applying it to block-unit evidence would be a unit
error.  The record-unit safe-output frontier and the bounded trusted merge buffer are deferred
to M4 (Decision 0005, point 11).

## 13. Scope guard

M3 contains no physical slot scheduling, no temporary-bin materialisation, no physical
dummy/cover I/O, no ciphertext/SGX/AES, no `DOAllocate` data path, no `DOMerge`, no
safe-output frontier, no bounded output buffer, no output block construction, no output
shuffle, no `PRP` publication, no level retirement, no de-amortisation, no attacks, no
benchmarks and no privacy theorem claim.

The milestone guards evolved narrowly, as Issue #6 allows: the M0 gate's repo-wide token list
dropped `interior_point` / `DPInteriorPoint` (M3 authorises interior points) and the package
surface guards now expect `content_schedule.py`.  Nothing unrelated was weakened, and the
M0/M1/M2 guarantees all still hold.

## 14. Milestones

| Milestone | Content | State |
|---|---|---|
| M0 | repository / bootstrap freeze | **accepted** (Issue #1, CLOSED) |
| M1 | functional oracle | **accepted** (Issue #2, CLOSED, PR #3 merged) |
| M2 | block-bin allocation planner | **accepted** (Issue #4, CLOSED, PR #5 merged) |
| **M3** | content-bound bin schedule (this document) | **current** (Issue #6) |
| M4 | physically stage/fetch padded bins; trusted bounded merge executor | not authorised, **not started** |
