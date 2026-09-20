# PROJECT STATE

## Current milestone

```text
M0 — Repository / bootstrap freeze
```

## Accepted

- **common LETIndex snapshot** — one-time byte-identical import of the LETIndex common
  substrate from `Russell-H-0/EnhancedLETIndex@4e68be75b0ac45e09ce6da8f6d587490fea4f35e`
  (`PROVENANCE.md` §3, `decisions/0001-common-letindex-substrate.md`).
- **provenance / hash gate** — SHA-256 manifest `provenance/common-substrate.sha256`
  covering every byte-identical import, enforced by `provenance/verify_manifest.py` and
  by `codes/tests/test_m0_provenance_gate.py`.
- **SWAT-M-Block contract** — the frozen semantics of the future merge baseline
  (`decisions/0002-swat-m-block-contract.md`, `docs/swat-m-block.md`).

## Not implemented

- noisy bin allocation
- DO merge
- dummy / cover block I/O
- output oblivious shuffle
- de-amortisation
- attacks
- performance experiments

`codes/src/swat_m_block/` contains an **empty package marker** and nothing else: no
`DOAllocate`, no `DOMerge`, no bin allocator, no output shuffle, no I/O of any kind.

## Out of scope for M0

- No EnhancedLETIndex modification; the source repository is untouched.
- No submodule / subtree / pip / git dependency between the two repositories; they are
  independent and evolve independently from this snapshot on.
- No original SWAT code was ported (provenance only, `PROVENANCE.md` §7).

## Next authorized milestone

```text
M1 — Functional SWAT-M-Block oracle
```

**NOT STARTED.** M1 must not begin before it is explicitly authorised. The M0 brief
stops here.

## Verification snapshot (M0)

```text
source commit (EnhancedLETIndex) : 4e68be75b0ac45e09ce6da8f6d587490fea4f35e
SWAT reference commit            : b33646061ec1899ccf75c9ba8fe43b2653c44a6b
imported byte-identical files    : 35 (22 substrate + 13 tests), SHA-256 manifest
derived test files               : 2 (DERIVED TEST, NOT BYTE-IDENTICAL IMPORT)
excluded defence modules present : 0
SWAT-M-Block algorithm present   : 0 (M0 implements nothing)
```
