# Decision 0002 — SWAT-M-Block contract

```text
Status:   ACCEPTED FOR M1+ IMPLEMENTATION
          M0 IMPLEMENTS NO SWAT ALGORITHM
Date:     2026-09-20
Milestone: M0 (freeze only) — implementation starts at M1
```

## 1. Definition

> **SWAT-M-Block is a block-observable adaptation of SWAT's differential-oblivious merge
> to the LETIndex substrate.**

Reference: Zheng et al., *"SWAT: A System-Wide Approach to Tunable Leakage Mitigation in
Encrypted Data Stores"*, PVLDB 2024; upstream merge-side implementation
`include/enclave/DOMerger.hpp` of [CongGroup/SWAT](https://github.com/CongGroup/SWAT) at
commit `b33646061ec1899ccf75c9ba8fe43b2653c44a6b` (provenance only — no code was ported
in M0).

## 2. Two granularities, kept explicitly distinct

| | granularity |
|---|---|
| **Security / I/O granularity** | the REE observable I/O unit is the **LETIndex block** |
| **Logical correctness granularity** | **record-level exact newer-wins sorted union** |

These two granularities must never be conflated. In particular it is **forbidden** to
define SWAT-M-Block as

> "treat each LETIndex block as one atomic sortable record"

because the key ranges of the source and target blocks may **interleave**: a block's key
range is not an atomic sort key here, and packing a block as a single unit into an output
block would produce a merged level whose contents are not the exact newer-wins sorted
union of the input records. Logical correctness is therefore decided **per record**, on
the trusted side, even though the *observable* schedule is decided per block.

## 3. Frozen pipeline

```text
source level + target level
    ->
SWAT-style noisy/padded block-bin allocation
    ->
block-granular observable READ schedule
    ->
trusted record-level merge/sort buffer
    ->
exact newer-wins sorted output C
    ->
canonical LETIndex block packing
    ->
canonical PGM construction
    ->
oblivious output shuffle
    ->
fresh physical output level
```

Step meanings:

1. **source level + target level** — two frozen logical levels (source is *newer*).
2. **SWAT-style noisy/padded block-bin allocation** — the DO-side idea: bins plus
   padded/noisy assignment so the observable access pattern does not follow the data.
3. **block-granular observable READ schedule** — what the REE can observe: which blocks
   are read, in what order.
4. **trusted record-level merge/sort buffer** — inside the trusted domain the actual
   records are merged; this step is *not* observable.
5. **exact newer-wins sorted output C** — the logical result C is exactly the frozen
   newer-wins merge: on an equal key the **source** (newer) record wins.
6. **canonical LETIndex block packing** — C is packed with the canonical packing.
7. **canonical PGM construction** — the output level's PGM comes from the frozen
   canonical PGM machinery.
8. **oblivious output shuffle** — the output blocks are written to fresh physical slots
   in an order that does not reveal the logical rank → slot mapping (§5).
9. **fresh physical output level** — the result is materialised in a fresh region; no
   in-place rewrite of an observable region.

## 4. Security claim — deliberately weak, and it stays weak

- The DO-style mechanism is adapted at **block-I/O granularity**.
- The original SWAT **record/element-level** (ε, δ)-DO theorem is **NOT automatically
  claimed** for this adaptation.
- Until a separate proof is supplied, this mechanism is called

  > **"SWAT-style block-granular adaptation"** or **"SWAT-M-Block"**

  and it must **not** be called formally proven (ε, δ)-DO.
- Any statement of the form "SWAT-M-Block is (ε, δ)-DO" is out of contract until M-proof
  supplies a theorem. Descriptive comparisons ("SWAT-style", "differential-oblivious
  merge adapted to blocks") are allowed; the formal claim is not.

## 5. Output write rule

**Forbidden:**

```text
logical-rank-order -> PRP(rank) writes
```

Writing output block *i* to `PRP(rank_i)` in logical rank order is rejected: the WRITE
**order itself** leaks the rank → slot mapping, independently of the PRP's secrecy.

**Required from the output-write milestone onward:** an **oblivious output shuffle**.

The first planned construction (frozen as the intended baseline; **not implemented in
M0**) is:

```text
secret random tags
    + fixed sorting network / bitonic shuffle
    + fresh-region materialisation
```

i.e. each output block receives a secret random tag, a fixed (data-independent) sorting
network — a bitonic network — permutes by tag, and the permuted blocks are materialised
into a freshly allocated region, so that neither the address sequence nor the write order
reveals the logical rank of a block.

## 6. Implementation boundary

- **M0 implements nothing** of this document. `codes/src/swat_m_block/` is an empty
  package marker: no noisy allocation, no DO merge, no dummy/cover I/O, no shuffle, no
  de-amortisation, and no `DOAllocate` / `DOMerge` port.
- M1 is scoped to a **functional oracle** only: a correct, non-oblivious-to-observable
  implementation of the frozen pipeline whose logical output C is exactly the frozen
  newer-wins merge result. Observable-schedule construction (noisy allocation, cover I/O)
  and the oblivious output shuffle are later milestones.
- SWAT-M-Block must be built **on top of** the frozen substrate of Decision 0001, never
  by editing it.

## 7. Non-goals

- No claim of formal equivalence with original SWAT.
- No modification of EnhancedLETIndex and no live dependency on it.
- No attack implementation and no performance experiment in M0.
