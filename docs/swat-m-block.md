# SWAT-M-Block

Design notes for the merge-side baseline of this repository. The normative contract is
`decisions/0002-swat-m-block-contract.md`; this document explains it and maps it onto the
frozen LETIndex substrate. **M0 implements none of it.**

## 1. Where this comes from

Zheng et al., *"SWAT: A System-Wide Approach to Tunable Leakage Mitigation in Encrypted
Data Stores"*, PVLDB 2024, proposes the **differential obliviousness (DO)** framework as a
tunable middle ground between full ORAM-style obliviousness (heavy) and no protection at
all (leaky), and applies it across the whole system — including the **merge side**, where
the sequence of accesses during a compaction/merge is itself a leak.

Upstream reference implementation: [CongGroup/SWAT](https://github.com/CongGroup/SWAT),
commit `b33646061ec1899ccf75c9ba8fe43b2653c44a6b`; the merge-side piece of interest is
`include/enclave/DOMerger.hpp` (differential-oblivious allocation of elements to output
bins).

This repository freezes that provenance and **re-derives** the mechanism against the
LETIndex substrate. No SWAT source file exists in this repository.

## 2. The substrate mapping (from Decision 0001, frozen)

| SWAT / paper notion | LETIndex substrate here |
|---|---|
| element / record | `enhanced_letindex.Record` — logical unit |
| bin / output array chunk | `enhanced_letindex.Block` with frozen `block_capacity` |
| level (LSM run) | `enhanced_letindex.Level` — logical representation |
| physical address | `SlotId` (distinct namespace from `BlockId`) |
| observable I/O | `UntrustedStorage` `READ` / `WRITE` on `SlotId`s |
| observable vocabulary | `trace.py` (`TraceEvent`, `TraceOperation`) |
| index metadata | canonical `pgm.py` (segmentation, OPLM, rank-line certificate) |
| correctness oracle | `builder.validate_level` + frozen newer-wins merge semantics |
| evaluator vs adversary | `leakage_metrics` / `ref_truth` vs `ref_adversary` |

The **block** is the unit that the REE can observe; the **record** is the unit at which
correctness is decided. Keeping those apart is the whole design (§3).

## 3. The subtlety: interleaving key ranges

In a LETIndex/learned-index merge, a source block and a target block may have
**interleaving key ranges**: for two blocks `B_s` (source) and `B_t` (target) it can
happen that `B_s` holds keys that fall between keys of `B_t` and vice versa. Therefore:

- a block is **not** an atomic sortable unit, and "sort the blocks, then concatenate" does
  **not** yield the merged level;
- the merged level must be produced by **record-level** newer-wins merging, then
  re-packed into canonical blocks.

This is why the contract forbids modelling a block as one atomic record, and why the
pipeline has a trusted record-level merge/sort buffer between the observable READ
schedule and the output packing:

```text
observable (block granularity)      trusted (record granularity)
--------------------------------    ---------------------------------------
READ schedule over blocks      ->   merge/sort buffer over records
                                    -> exact newer-wins sorted output C
output block packing           <-   canonical LETIndex packing of C
```

Both granularities coexist in one run; the REE observes the left column only.

## 4. Pipeline (normative)

```text
source level + target level
    -> SWAT-style noisy/padded block-bin allocation
    -> block-granular observable READ schedule
    -> trusted record-level merge/sort buffer
    -> exact newer-wins sorted output C
    -> canonical LETIndex block packing
    -> canonical PGM construction
    -> oblivious output shuffle
    -> fresh physical output level
```

## 5. What is not claimed

- **No formal (ε, δ)-DO theorem.** The upstream theorem is stated at record/element
  granularity; adapting the mechanism to block-granular I/O changes the observable unit,
  so the theorem does not transfer automatically. The honest name for this construction
  is *"SWAT-style block-granular adaptation"* / *SWAT-M-Block*, not "proven (ε, δ)-DO".
- **No equivalence with original SWAT.** Different substrate, different observable unit,
  different implementation lineage.
- **No claim about EnhancedLETIndex.** SWAT-M-Block is a *baseline to compare against*
  EnhancedLETIndex; the comparison protocol is the shared experiment/artifact contract
  (Decision 0001), not a shared code path.

## 6. Output write order

Writing output blocks to `PRP(rank)` in logical rank order is **forbidden**: the write
order alone reveals the rank → slot mapping, regardless of how strong the PRP is.

The intended construction (frozen here, implemented in a later milestone) is:

```text
secret random tags
  + fixed sorting network (bitonic shuffle, data-independent)
  + fresh-region materialisation
```

so that the sequence of physical writes is independent of the logical ranks of the blocks
being written. **M0 does not implement it.**

## 7. Milestone plan (only M0 and M1 are authorised in any sense)

| Milestone | Content | State |
|---|---|---|
| **M0** | repository / bootstrap freeze: substrate snapshot, provenance gate, this contract | **current** |
| **M1** | *Functional SWAT-M-Block oracle*: correct block-granular pipeline whose logical output C equals the frozen newer-wins merge; no observable-schedule protection yet | **not started** |
| later | noisy/padded bin allocation, cover/dummy block I/O, oblivious output shuffle, de-amortisation, attacks, performance experiments | out of scope for now |

## 8. References

- Zheng et al., *SWAT: A System-Wide Approach to Tunable Leakage Mitigation in Encrypted
  Data Stores*, PVLDB 2024.
- `CongGroup/SWAT` @ `b33646061ec1899ccf75c9ba8fe43b2653c44a6b`,
  `include/enclave/DOMerger.hpp`.
- `decisions/0001-common-letindex-substrate.md` — the frozen substrate.
- `decisions/0002-swat-m-block-contract.md` — the normative contract.
- `PROVENANCE.md` — the freeze record and the excluded-module audit.
