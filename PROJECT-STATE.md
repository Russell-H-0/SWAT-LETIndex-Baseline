# PROJECT STATE

## Current milestone

```text
M3 — Content-bound SWAT-M-Block bin schedule
```

## Accepted

### M0 — repository / bootstrap freeze (Issue #1, CLOSED)

- **common LETIndex snapshot** — one-time byte-identical import of the LETIndex common
  substrate from `Russell-H-0/EnhancedLETIndex@4e68be75b0ac45e09ce6da8f6d587490fea4f35e`
  (`PROVENANCE.md` §3, `decisions/0001-common-letindex-substrate.md`).
- **provenance / hash gate** — SHA-256 manifest `provenance/common-substrate.sha256`
  covering every byte-identical import, enforced by `provenance/verify_manifest.py` and
  by `codes/tests/test_m0_provenance_gate.py`.
- **SWAT-M-Block contract** — the frozen semantics of the merge baseline
  (`decisions/0002-swat-m-block-contract.md`, `docs/swat-m-block.md`).

### M1 — functional SWAT-M-Block oracle (Issue #2, CLOSED, PR #3 merged)

- **the oracle** — `codes/src/swat_m_block/functional_oracle.py` exposes
  `functional_merge_oracle(...)` returning the immutable `FunctionalMergeOracleResult`
  (`decisions/0003-m1-functional-oracle.md`, `docs/m1-functional-oracle.md`).
- **exact logical semantics** — record-level exact newer-wins sorted union, canonical
  output block packing, canonical PGM, driven through the frozen
  `enhanced_letindex.incremental_merge` / `pgm` machinery rather than re-implemented.

### M2 — SWAT-style block-bin allocation planner (Issue #4, CLOSED, PR #5 merged)

- **the planner** — `codes/src/swat_m_block/distribution.py` (pinned SWAT distribution and
  sizing kernel) and `codes/src/swat_m_block/bin_allocator.py` (the plan)
  (`decisions/0004-m2-block-bin-allocation.md`, `docs/m2-block-bin-allocation.md`).
- **frozen adaptation boundary** — one allocation item is one logical LETIndex block and
  the atomic bucket capacity is one block, so bin capacity, bin loads and every
  prefix/error quantity are measured in **blocks**.
- **reference geometry** — for `lambda = 512`, `privacy_epsilon = 1.0`,
  `privacy_delta = 1e-12`: `bin_capacity_blocks Z = 16`,
  `factor = 0.12829670090910578`, `bin_count = ceil(factor * n)`.

### M3 — content-bound SWAT-M-Block bin schedule (Issue #6)

- **the schedule** — `codes/src/swat_m_block/content_schedule.py`
  (`decisions/0005-m3-content-bound-bin-schedule.md`,
  `docs/m3-content-bound-bin-schedule.md`).
- **frozen granularities** — a block is the allocation / future I/O unit, a record is the
  trusted merge comparison unit.  No block representative key is introduced.
- **public API** — `LogicalBlockRunView`, `BoundBlockBin`, `BoundBlockAllocation`,
  `InteriorPoint`, `TaggedBinInteriorPoint`, `AbstractBinRead`,
  `SwatBlockMergeSchedule`, `ContentScheduleError`, `DUMMY_POS_INF`, `SOURCE`, `TARGET`,
  `bind_block_allocation`, `sample_bin_interior_points`, `interior_point_weights`,
  `interior_point_sort_key`, `sorted_tagged_interior_points`,
  `build_abstract_merge_schedule`, `plan_swat_block_merge_schedule`.
- **interior points** — sampled from the bin's actual record keys with the pinned
  `baseExp ** (min(i, load - i) + 1)` weighting; an exhausted bin uses `DUMMY_POS_INF`.
- **abstract schedule** — `(side, bin_index)` fetch order only: preload bin 0 of each side,
  then follow the pinned `(interior point, signed tag)` stream; every planned bin of each
  side exactly once.

### QA-1 — post-M3 differential regression sweep (Issue #8, PR open, not yet reviewed)

- **the harness** — `codes/tools/post_m3_regression_sweep.py` with
  `codes/tests/test_post_m3_regression_sweep.py` and `docs/post-m3-regression-sweep.md`.
- **what it is** — a stabilization campaign over the already accepted M1/M2/M3 contracts.  It
  freezes no new semantics, establishes no new decision record and implements no mechanism; it
  re-derives the accepted behaviour from independently written evaluators (a dict-based exact
  newer-wins evaluator, arithmetic canonical packing, and an independent simulator of the
  pinned `DOMerge` preload/`if`/`else if` fetch loop) and checks the shipped implementations
  against them over a deterministic generated sweep.
- **status** — QA only, additive.  It does not change the current algorithmic milestone, does
  not authorize M4, and does not modify any module under `codes/src/swat_m_block/` or any of
  the 35 manifest-covered frozen substrate files.

## Implemented in the repository

```text
codes/src/enhanced_letindex/   frozen common substrate (35 manifest-covered files)
codes/src/swat_m_block/        M1 functional oracle, M2 block-bin planner,
                               M3 content-bound bin schedule
codes/tools/                   QA-1 post-M3 regression sweep (stabilization only;
                               not part of the production package)
codes/tests/                   13 imported + 2 derived + 4 repo test modules
                               + the QA-1 sweep tests
provenance/                    manifest + verification script
```

## Not implemented

- physical `SlotId` scheduling / physical staging of padded bins
- temporary padded-bin physical materialization
- physical dummy / cover block I/O
- the `DOAllocate` data path (ciphertext handling, SGX/AES, bitonic sort of decrypted data,
  bin publication)
- `DOMerge`
- the record-unit safe-output frontier (the pinned `newCnt` arithmetic)
- the bounded trusted merge buffer
- output block construction / output oblivious shuffle
- `PRP` publication / level retirement
- de-amortisation
- attacks
- benchmarks / performance experiments
- any `(epsilon, delta)` privacy theorem claim

The M1 oracle performs **zero** `UntrustedStorage` access, emits **zero** `TraceEvent`,
addresses **zero** physical `SlotId`, allocates **no** dummy or cover element and uses
**zero** randomness.  The M2 planner and the M3 binder use seeded, planner-owned randomness
to construct a plan and to sample interior points, but still perform **zero**
`UntrustedStorage` access, emit **zero** `TraceEvent` and schedule **zero** physical
`SlotId`: these are descriptions of intended placement, not REE observations.

## Out of scope for M3

- No EnhancedLETIndex modification; the source repository is untouched.
- No submodule / subtree / pip / git dependency between the two repositories.
- No original SWAT code was ported (provenance only, `PROVENANCE.md` §7).
- No physical schedule, no merge execution, no output, no theorem claim.

## Next authorized milestone

```text
M4 — Physically stage/fetch padded bins and implement the trusted bounded merge executor
```

**NOT AUTHORIZED by M3 and NOT STARTED.** M4 must not begin before it is explicitly
authorized by its own issue.  In particular M4 must separately resolve: physical
padded-bin materialization without accidental rank → slot leakage; dummy block
representation and physical reads; mapping from abstract bin fetch to an REE trace;
block-unit noisy-prefix evidence vs the record-unit safe-output frontier; bounded trusted
merge-buffer correctness; and the eventual output publication / oblivious shuffle boundary.

## Verification snapshot (M3)

```text
source commit (EnhancedLETIndex) : 4e68be75b0ac45e09ce6da8f6d587490fea4f35e
SWAT reference commit            : b33646061ec1899ccf75c9ba8fe43b2653c44a6b
frozen substrate                 : 35 files, SHA-256 manifest, unchanged since M0
M1 implementation                : codes/src/swat_m_block/functional_oracle.py
M2 implementation                : codes/src/swat_m_block/distribution.py
                                   codes/src/swat_m_block/bin_allocator.py
M3 implementation                : codes/src/swat_m_block/content_schedule.py
M3 public entry point            : plan_swat_block_merge_schedule
M3 schedule granularity          : (side, bin_index) abstract bin fetch order
physical I/O / trace events      : 0 / 0 (instrumented, see the M1 / M2 / M3 tests)
randomness                       : seeded, planner-owned, domain-separated, deterministic;
                                   not claimed C++ bit-identical
SWAT privacy theorem claimed     : none
```
