# PROJECT STATE

## Current milestone

```text
M2 — SWAT-style block-bin allocation planner (noisy/padded bins)
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
- **M1 focused tests** — `codes/tests/test_m1_functional_oracle.py`.

### M2 — SWAT-style block-bin allocation planner (Issue #4)

- **the planner** — `codes/src/swat_m_block/distribution.py` (pinned SWAT distribution and
  sizing kernel) and `codes/src/swat_m_block/bin_allocator.py` (the plan)
  (`decisions/0004-m2-block-bin-allocation.md`, `docs/m2-block-bin-allocation.md`).
- **frozen adaptation boundary** — one allocation item is one logical LETIndex block and
  the atomic bucket capacity is one block, so bin capacity, bin loads and every
  prefix/error quantity are measured in **blocks**.
- **public API** — `SwatBlockAllocationConfig`, `swat_reference_config`,
  `BlockBinPlan`, `BlockAllocationPlan`, `BlockAllocationError`,
  `InsufficientSampledCapacity`, `allocate_block_bins`, `compute_bin_capacity_blocks`,
  `compute_bin_count`, `compute_factor`, `compute_noisy_prefix_sums`, `compute_weight`,
  `geom_conv`, `GeometricLoadSampler`, `LaplaceSampler`, `derive_stream_seed`,
  `ATOMIC_BUCKET_CAPACITY_BLOCKS`.
- **reference geometry** — for `lambda = 512`, `privacy_epsilon = 1.0`,
  `privacy_delta = 1e-12`: `bin_capacity_blocks Z = 16`,
  `factor = 0.12829670090910578`, `bin_count = ceil(factor * n)`.
- **M2 focused tests** — `codes/tests/test_m2_block_bin_allocation.py`.

## Implemented in the repository

```text
codes/src/enhanced_letindex/   frozen common substrate (35 manifest-covered files)
codes/src/swat_m_block/        M1 functional oracle + M2 block-bin allocation planner
codes/tests/                   13 imported + 2 derived + 3 repo test modules
provenance/                    manifest + verification script
```

## Not implemented

- the `DOAllocate` data path (ciphertext handling, SGX/AES, bitonic sort of decrypted data,
  bin publication)
- `DOMerge` / cross-run merge schedule
- DP interior point and key/interior-point binding (deferred to M3)
- dummy / cover physical I/O
- output blocks and output oblivious shuffle
- `PRP` writeback / physical publication
- de-amortisation
- physical-slot scheduling of any kind
- attacks
- benchmarks / performance experiments
- any `(epsilon, delta)` privacy theorem claim

The M1 oracle performs **zero** `UntrustedStorage` access, emits **zero** `TraceEvent`,
addresses **zero** physical `SlotId`, allocates **no** dummy or cover element and uses
**zero** randomness.  The M2 planner uses seeded, planner-owned randomness to construct a
plan, but still performs **zero** `UntrustedStorage` access, emits **zero** `TraceEvent` and
schedules **zero** physical `SlotId` — the plan is a description of intended block
placement, not an REE observation.

## Out of scope for M2

- No EnhancedLETIndex modification; the source repository is untouched.
- No submodule / subtree / pip / git dependency between the two repositories.
- No original SWAT code was ported (provenance only, `PROVENANCE.md` §7).
- No physical schedule, no key binding, no theorem claim.

## Next authorized milestone

```text
M3 — Bind block-bin plans to trusted contents and build the SWAT-M-Block merge/read schedule
```

**NOT AUTHORIZED by M2 and NOT STARTED.** M3 must not begin before it is explicitly
authorized by its own issue.

## Verification snapshot (M2)

```text
source commit (EnhancedLETIndex) : 4e68be75b0ac45e09ce6da8f6d587490fea4f35e
SWAT reference commit            : b33646061ec1899ccf75c9ba8fe43b2653c44a6b
frozen substrate                 : 35 files, SHA-256 manifest, unchanged since M0
M1 implementation                : codes/src/swat_m_block/functional_oracle.py
M2 implementation                : codes/src/swat_m_block/distribution.py
                                   codes/src/swat_m_block/bin_allocator.py
M2 public entry point            : allocate_block_bins
reference geometry (block units) : Z = 16, factor = 0.12829670090910578
physical I/O / trace events      : 0 / 0 (instrumented, see the M1 and M2 tests)
randomness                       : seeded, planner-owned, deterministic; not C++ bit-identical
SWAT privacy theorem claimed     : none
```
