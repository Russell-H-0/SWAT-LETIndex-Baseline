# PROVENANCE

Frozen provenance for this repository. Everything below is a record of a **one-time
source snapshot**; nothing in this repository is synchronised with any other repository
at runtime, build time or test time.

## 1. Common substrate source

```text
Common substrate source:
    Russell-H-0/EnhancedLETIndex

Frozen source commit:
    4e68be75b0ac45e09ce6da8f6d587490fea4f35e

Import date:
    2026-09-20
```

The frozen commit corresponds to the **accepted and merged G5-B baseline** of
EnhancedLETIndex. The snapshot was read directly out of the source repository's git
object database (`git show <commit>:<path>`), never from the source working tree, and the
source checkout was **not** required to sit at that commit. Existence of the commit was
verified before any content was copied:

```bash
git -C <EnhancedLETIndex> cat-file -e 4e68be75b0ac45e09ce6da8f6d587490fea4f35e^{commit}
```

## 2. Policy

```text
Policy:
    - one-time source snapshot
    - no live synchronization
    - no submodule/subtree/package dependency
    - future source import requires explicit provenance revision
    - EnhancedLETIndex defense implementations are excluded
```

Consequences, all of them enforced by tests (see §6):

- This repository must never declare a git submodule, a git subtree merge, a
  `git+https://…#EnhancedLETIndex` requirement, or a local-path dependency on
  `F:\HHC\Projects\EnhancedLETIndex`.
- The imported files are **frozen**. Any change to them is a provenance violation and is
  caught by the SHA-256 manifest gate.
- Further imports from EnhancedLETIndex require an explicit revision of this document
  (a new dated section), not an ad-hoc copy.
- Fair comparison between EnhancedLETIndex and SWAT-M-Block is done through a **shared
  experiment / artifact contract**, never through live source sharing
  (`decisions/0001-common-letindex-substrate.md`).

## 3. Imported common substrate (`codes/src/enhanced_letindex/`)

All 22 files below are **byte-identical** to their source blobs at the frozen commit.
There was no package rename, no "cleanup", no reformatting and no opportunistic
refactoring. `enhanced_letindex` deliberately keeps its original package name.

### 3.1 Requested common substrate (21 files)

| File | Role |
|---|---|
| `__init__.py` | package surface |
| `identifiers.py` | `RecordKey`, `LevelId`, `BlockId`, `SlotId` — the logical/physical namespaces |
| `record.py` | `Record` logical unit |
| `block.py` | `Block`, `block_capacity` geometry |
| `dataset.py` | dataset construction |
| `config.py` | `Config` |
| `level.py` | logical `Level` representation |
| `mapping.py` | logical-rank to slot mapping |
| `storage.py` | `UntrustedStorage` observable surface (`READ`/`WRITE`, physical `SlotId`) |
| `trace.py` | physical trace vocabulary (`TraceEvent`, `TraceOperation`) |
| `pgm.py` | canonical PGM (segmentation, OPLM, rank-line certificate) |
| `builder.py` | level construction + `validate_level` oracle |
| `engine.py` | `TrustedEngine`, `MergeResult`, `RetiredLevelError` |
| `merge.py` | blocking logical merge |
| `incremental_merge.py` | incremental (resumable, bounded-state) logical merge + incremental PGM |
| `leakage.py` | observation transcript / privileged evaluator |
| `leakage_metrics.py` | transcript metrics |
| `letindex_ref.py` | LETIndex reference profile |
| `ref_workload.py` | reference workload |
| `ref_truth.py` | reference truth (evaluator side) |
| `ref_adversary.py` | reference adversary (attack side) |

### 3.2 Dependency-closure addition (1 file) — justified

| File | Reason |
|---|---|
| `workload.py` | **Minimal dependency closure, not "might be useful later".** The required common regression target *leakage transcript / metrics* is `codes/tests/test_leakage.py`, which imports `enhanced_letindex.workload` (`export_dict`, `generate_workload`, `run_script`). Without this file that required regression target cannot be imported. `workload.py` is the **G1-E shared baseline workload model** — the shared experiment artifact Definition 0001 points at — and it is not a defence mechanism: it imports only `builder`, `config`, `engine`, `identifiers`, `record`, `trace`, all of which are already part of the requested set, so it adds **no further file** to the closure. |

The static import closure of the 21 requested modules was computed before importing
anything, and it is **exactly those 21 modules** — no other file was pulled in, and none
of the forbidden defence modules is reachable from them.

### 3.3 Nothing else was copied

No other file of the source repository was imported — in particular no `design-spec.md`,
no `PROJECT-STATE.md`, no `decisions/`, no `docs/`, no `experiments/`, and none of the
source `pyproject.toml`. The documents in this repository were written fresh for M0.

## 4. Explicitly excluded EnhancedLETIndex defence implementations

The following source modules are **excluded by policy** and must not exist in this
repository in any form:

```text
block_prp.py
defense_state.py
query_defense.py
query_stash.py
protected_merge.py
deamortized_merge.py
defended_index.py
```

Also excluded, by the same policy:

- any **G6 / PGM-guided physical scheduling** implementation;
- any other future EnhancedLETIndex defence implementation or extension thereof.

`test_pgm_rank_certificate.py` in the source tree imports `defended_index` purely for
observation-schema constants used by its repository-state scope guard. Rather than
"quietly" importing the forbidden module, the import surface was narrowed without
changing any source semantic: this repository imports a **derived** copy of that test
with the defence-constant and repository-state parts removed (§5).

No allowed file had to be dropped: after narrowing, **no requested module statically
depends on any forbidden module** (verified by an AST import-closure pass over the frozen
commit before importing). No STOP condition in §13 of the M0 brief was triggered by
`rules 2`/`3`.

## 5. Imported tests (`codes/tests/`)

### 5.1 Byte-identical imports (13 files)

| File | Covered common substrate |
|---|---|
| `test_record.py` | `Record` |
| `test_block.py` | `Block` |
| `test_dataset.py` | `Dataset` |
| `test_mapping.py` | mapping |
| `test_storage.py` | storage / trace |
| `test_level_construction.py` | level construction |
| `test_lookup.py` | single-level lookup |
| `test_multilevel_lookup.py` | multi-level lookup |
| `test_engine.py` | engine |
| `test_pgm.py` | PGM |
| `test_blocking_merge.py` | blocking logical merge |
| `test_leakage.py` | leakage transcript / metrics (needs `workload.py`, §3.2) |
| `test_letindex_ref.py` | LETIndex reference profile + adversary/truth separation |

These files are covered by the SHA-256 manifest as well, because they are claimed
byte-identical.

### 5.2 DERIVED TEST, NOT BYTE-IDENTICAL IMPORT (2 files)

Both are reduced copies of a source test that exercised a *mix* of common-substrate
behaviour and EnhancedLETIndex defence / repository-state behaviour. **No functional
assertion of the common substrate was weakened**; only the parts that structurally
require an excluded module or pin the source repository's milestone state were removed.

| File | Removed, and why |
|---|---|
| `test_incremental_merge.py` | Removed `test_a1` (`PROJECT-STATE.md` + decision-diff status pin), `test_a2` (`baseline-g2b0-letindex-ref-v1` tag pin), `test_a3` (runs the excluded G4 suites), `test_b2b`'s trailing G4-engine section, the whole section F (`g4_oracle`, `g4_pgm_index`, `published_items`, `test_f10`, `test_f10b` — all use `protected_merge` / `defended_index` as oracle) and `test_g20_a` (`DefendedExperiment`). Kept byte-identical: every section B/C/D/E test, `test_a4_g5a_is_trusted_logical_computation_only` (it only *names* the forbidding modules in string literals and asserts `incremental_merge.py` imports nothing else), and the CLI/probe guards. |
| `test_pgm_rank_certificate.py` | Removed `test_1` (G5-A status-closure pin against the source repository), `test_15b` (runs the excluded G4 suites) and the source-repository half of `test_15`. `test_15` is kept as `test_15_the_imported_certificate_modules_stay_scope_clean`, preserving its source-scope token assertion over the imported `pgm.py` / `incremental_merge.py`. |
| `test_workload.py` | **Not imported.** Its subject is not one of the requested regression targets; `workload.py` is present only as the dependency of `test_leakage.py` (§3.2). |

### 5.3 New M0 tests (not imported, written for this repository)

| File | Purpose |
|---|---|
| `test_m0_provenance_gate.py` | verifies the SHA-256 manifest, the exclusion audit, the absence of any submodule/subtree/live dependency, and the absence of any SWAT-M-Block algorithm |

## 6. Machine-verifiable manifest

```text
provenance/common-substrate.sha256      <sha256>  <relative path>
```

It covers **every** file claimed byte-identical in this document: all 22 files of
`codes/src/enhanced_letindex/` and the 13 tests of §5.1 (35 entries).

The manifest verifies that **this repository's frozen snapshot is not modified
afterwards**. It is *not* a runtime connection to EnhancedLETIndex, and verification does
not read, fetch or require the source repository.

Verification entry points:

```bash
python provenance/verify_manifest.py          # standalone, exits non-zero on mismatch
cd codes && python -m pytest -q -p no:cacheprovider tests/test_m0_provenance_gate.py
```

## 7. SWAT reference source (provenance only — nothing copied)

```text
SWAT reference source:
    CongGroup/SWAT

SWAT reference commit:
    b33646061ec1899ccf75c9ba8fe43b2653c44a6b

Relevant upstream implementation:
    include/enclave/DOMerger.hpp

Reference paper:
    Zheng et al., "SWAT: A System-Wide Approach to Tunable Leakage Mitigation in
    Encrypted Data Stores", PVLDB 2024
```

`include/enclave/DOMerger.hpp` is the intended conceptual reference for the merge-side,
differential-oblivious allocation in M1+. **M0 records this provenance only** — no code
was ported, and no upstream file exists in this repository.
