# Decision 0001 — Common LETIndex substrate

```text
Status:   ACCEPTED
Date:     2026-09-20
Milestone: M0 (repository / bootstrap freeze)
```

## Context

SWAT-LETIndex-Baseline must produce a merge baseline that is comparable with
EnhancedLETIndex. A comparison is only meaningful if both implementations agree on the
data model, the block geometry, the PGM semantics, the physical I/O vocabulary and the
correctness oracle. Those five things are exactly the *common substrate* of
EnhancedLETIndex.

Two ways of obtaining them were considered:

1. **live source sharing** — submodule, subtree, a pip dependency on a git URL, or a
   relative path dependency on a sibling working copy;
2. **a one-time source snapshot** — copy the substrate once, freeze it cryptographically,
   and let both repositories evolve independently afterwards.

Live source sharing was rejected: it couples the two repositories' release cadence,
silently changes the baseline whenever the source moves, and makes the baseline
non-reproducible. It was also explicitly forbidden by the M0 brief.

## Decision

> **SWAT-LETIndex-Baseline imports a one-time snapshot of the LETIndex common
> substrate from EnhancedLETIndex@4e68be75. Imported common files define the shared
> logical data model, block geometry, PGM semantics, physical trace vocabulary, and
> correctness oracle, are thereafter frozen, and no EnhancedLETIndex defense mechanism
> may be imported into the SWAT baseline.**

Concretely:

- Source: `Russell-H-0/EnhancedLETIndex`, commit
  `4e68be75b0ac45e09ce6da8f6d587490fea4f35e` (the accepted, merged G5-B baseline).
- Read out of the git object database, byte-identical, no rename, no reformatting: the
  package keeps the name `enhanced_letindex`.
- Frozen from the moment of import, with a SHA-256 manifest
  (`provenance/common-substrate.sha256`) as the gate.
- **No defence mechanism of EnhancedLETIndex is imported**, now or later
  (`PROVENANCE.md` §4 lists the excluded modules).

## The frozen contract

The imported common files fix the following, and SWAT-M-Block must not redefine any of
it:

1. **same `Record` / `Block` semantics** — a record is the logical unit; a block is the
   unit of REE-observable I/O.
2. **same `block_capacity` semantics** — the capacity and the canonical packing of a
   block are unchanged.
3. **same logical `Level` representation** — level identity, keys, values and the
   canonical item ordering.
4. **same `BlockId` ≠ `SlotId` namespace separation** — logical block identity and
   physical slot identity are different namespaces and stay distinct.
5. **same `UntrustedStorage` observable surface** — the adversary-visible surface is
   `READ` / `WRITE` on physical `SlotId`s, nothing else.
6. **same canonical PGM implementation** — segmentation, OPLM and the rank-line
   certificate come from the frozen `pgm.py`, not from a re-implementation.
7. **same newer-wins merge semantics** — on a duplicated key the **source (newer)** level
   wins; the merged result is the sorted union over record keys.
8. **same canonical output block packing** — the merged record stream is packed into
   output blocks with the frozen canonical packing, including the final partial block.
9. **same evaluator-vs-adversary separation where imported** — the privileged evaluator
   side (`leakage_metrics`, `ref_truth`) and the adversary side (`ref_adversary`) remain
   separated; the adversary sees only the observation transcript.
10. **no Enhanced defence mechanism** — none of `block_prp`, `defense_state`,
    `query_defense`, `query_stash`, `protected_merge`, `deamortized_merge`,
    `defended_index`, and no G6 / PGM-guided physical scheduling work.

## Fair comparison

Future fair comparison with EnhancedLETIndex is achieved through a **shared
experiment / artifact contract** — identical inputs, identical datasets, identical
workload families, identical observation schema, identical metrics — and **not** through
live source sharing. The shared baseline workload model (`enhanced_letindex.workload`)
and the LETIndex reference profile (`letindex_ref`, `ref_workload`, `ref_truth`,
`ref_adversary`) are part of the frozen snapshot precisely so that both sides can be fed
the same artifacts.

## Revision rule

A future import from EnhancedLETIndex is allowed only via an **explicit provenance
revision**: a new dated section in `PROVENANCE.md` naming the new commit, the file list
and the reason, plus a regenerated manifest. Ad-hoc copying is a violation.

## Consequences

- The two repositories are independent; neither can break the other.
- The frozen files cannot be "improved" here. If SWAT-M-Block needs different behaviour,
  it must live in `swat_m_block/` on top of the frozen substrate, not inside it.
- Any edit to an imported file fails `provenance/verify_manifest.py` and the M0
  provenance gate test.
