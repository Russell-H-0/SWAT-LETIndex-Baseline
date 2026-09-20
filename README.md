# SWAT-LETIndex-Baseline

**Purpose.** An apples-to-apples **SWAT-style merge baseline** for EnhancedLETIndex
experiments: a block-granular adaptation of the merge-side defence of SWAT, implemented
over exactly the same frozen LETIndex substrate that EnhancedLETIndex uses, so that the
two can be compared on identical data models, block geometry, PGM semantics, physical
I/O vocabulary and correctness oracle.

**Primary comparison target.** EnhancedLETIndex vs SWAT-M-Block, over the same LETIndex
substrate. Fairness is obtained through a **shared experiment / artifact contract**, not
through live source sharing (see `decisions/0001-common-letindex-substrate.md`).

## What this repository is not

- **Not a fork of EnhancedLETIndex.** It is an independent repository with its own
  history. No submodule, no subtree, no pip/git dependency, no live synchronisation.
- **Not the original SWAT.** It is not a port of the SWAT enclave implementation and does
  not reuse its code. The official SWAT reference implementation
  ([CongGroup/SWAT](https://github.com/CongGroup/SWAT), commit
  `b33646061ec1899ccf75c9ba8fe43b2653c44a6b`, `include/enclave/DOMerger.hpp`) is recorded
  as **provenance only** — nothing from it has been copied.
- **Not claiming formal equivalence to original SWAT.** The mechanism here is adapted at
  **block-I/O granularity**, and the original SWAT record/element-level
  (ε, δ)-differential-obliviousness theorem does **not** automatically transfer to that
  adaptation. Until a separate proof is supplied, this mechanism is called
  *"SWAT-style block-granular adaptation"* / *SWAT-M-Block*, and it is **not** called
  formally proven (ε, δ)-DO.

## Provenance in one line

`codes/src/enhanced_letindex/` is a **one-time, byte-identical snapshot** of the LETIndex
common substrate taken from `Russell-H-0/EnhancedLETIndex` at commit
`4e68be75b0ac45e09ce6da8f6d587490fea4f35e` on **2026-09-20**. It is frozen: the SHA-256
manifest in `provenance/common-substrate.sha256` is the gate that guards it, and no
EnhancedLETIndex **defence implementation** may ever be imported here. Full detail,
including the justification for every imported file, is in `PROVENANCE.md`.

## Layout

```text
SWAT-LETIndex-Baseline/
├── README.md
├── PROVENANCE.md                  provenance freeze + imported/excluded file lists
├── PROJECT-STATE.md               milestone state
├── pyproject.toml                 (see codes/pyproject.toml for the test config)
├── decisions/
│   ├── 0001-common-letindex-substrate.md
│   └── 0002-swat-m-block-contract.md
├── docs/
│   └── swat-m-block.md            the frozen SWAT-M-Block semantics
├── provenance/
│   ├── common-substrate.sha256    machine-verifiable manifest
│   └── verify_manifest.py         standalone verification script
└── codes/
    ├── pyproject.toml
    ├── src/
    │   ├── enhanced_letindex/     [frozen imported common substrate]
    │   └── swat_m_block/          [M0: empty package marker only]
    └── tests/                     common-substrate regression tests + M0 gates
```

## Running the tests

```bash
cd codes
python -m pytest -q -p no:cacheprovider
```

The suite is configured to never write `.pytest_cache` (`-p no:cacheprovider` is part of
`addopts`). To print a pass/fail summary line use `--junitxml=report.xml` or
`-p no:cacheprovider -rA`.

Verify the frozen snapshot independently of pytest:

```bash
python provenance/verify_manifest.py
```

## Milestone state

M0 (repository / bootstrap freeze) is the current milestone: **no SWAT algorithm is
implemented yet**. See `PROJECT-STATE.md`.
