# PROJECT STATE

## Current milestone

```text
M1 — Functional SWAT-M-Block oracle
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

### M1 — functional SWAT-M-Block oracle (Issue #2)

- **the oracle** — `codes/src/swat_m_block/functional_oracle.py` exposes
  `functional_merge_oracle(source, target, *, items_per_block, epsilon, schedule=None)`
  returning the immutable `FunctionalMergeOracleResult`
  (`decisions/0003-m1-functional-oracle.md`, `docs/m1-functional-oracle.md`).
- **exact logical semantics** — record-level exact newer-wins sorted union, canonical
  output block packing, canonical PGM; the merge, packing and PGM are driven through the
  frozen `enhanced_letindex.incremental_merge` / `pgm` machinery, not re-implemented.
- **M1 focused tests** — `codes/tests/test_m1_functional_oracle.py`.

## Implemented in the repository

```text
codes/src/enhanced_letindex/   frozen common substrate (35 manifest-covered files)
codes/src/swat_m_block/        functional oracle (M1)            <- this milestone
codes/tests/                   13 imported + 2 derived + 2 repo test modules
provenance/                    manifest + verification script
```

## Not implemented

- noisy bin allocation / padded bins
- DOAllocate / DOMerge
- dummy / cover block I/O
- output oblivious shuffle
- PRP writeback / physical publication
- de-amortisation
- physical-slot scheduling of any kind
- attacks
- performance experiments

The M1 oracle performs **zero** `UntrustedStorage` access, emits **zero** `TraceEvent`,
addresses **zero** physical `SlotId`, allocates **no** dummy or cover element and uses
**zero** randomness.  It is a logical correctness oracle only and makes no privacy or
security claim of any kind.

## Out of scope for M1

- No EnhancedLETIndex modification; the source repository is untouched.
- No submodule / subtree / pip / git dependency between the two repositories.
- No original SWAT code was ported (provenance only, `PROVENANCE.md` §7).
- No SWAT privacy mechanism, no physical schedule, no `(epsilon, delta)` claim.

## Next authorized milestone

```text
M2 — SWAT-style noisy/padded block-bin allocation
```

**NOT AUTHORIZED by M1 and NOT STARTED.** M2 must not begin before it is explicitly
authorized by its own issue.

## Verification snapshot (M1)

```text
source commit (EnhancedLETIndex) : 4e68be75b0ac45e09ce6da8f6d587490fea4f35e
SWAT reference commit            : b33646061ec1899ccf75c9ba8fe43b2653c44a6b
frozen substrate                 : 35 files, SHA-256 manifest, unchanged since M0
M1 implementation                : codes/src/swat_m_block/functional_oracle.py
M1 public entry point            : functional_merge_oracle
physical I/O / trace events      : 0 / 0 (instrumented, see the M1 tests)
randomness                       : none
SWAT privacy mechanism present   : 0 (M1 is a functional oracle)
```
