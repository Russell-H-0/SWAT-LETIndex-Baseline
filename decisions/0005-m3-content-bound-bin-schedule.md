# Decision 0005 — M3 content-bound bin schedule

```text
Status:   ACCEPTED
Date:     2026-09-20
Milestone: M3 (content-bound SWAT-M-Block bin schedule, Issue #6)
```

## Context

M2 plans SWAT-style padded bins without looking at any content (`decisions/0004`); M1
decides the exact logical result of a merge (`decisions/0003`).  Neither can drive a
physical merge, because the pinned merge order is derived from *interior points sampled from
actual contents*, and the adaptation has no such notion yet.

M3 is the bridge.  It binds M2's plan to the real trusted blocks of one run, samples one
pinned SWAT interior point per bin from the bin's actual record keys, and derives the
abstract cross-run bin-read order — while still performing no physical I/O.

Pinned provenance (read directly for this milestone):

```text
CongGroup/SWAT @ b33646061ec1899ccf75c9ba8fe43b2653c44a6b
include/enclave/DPInteriorPoint.hpp   the per-bin weight expression and draw
include/enclave/DOMerger.hpp          DOAllocate's per-bin tagging, DOMerge's negation of
                                      the right side's tags, the std::pair ordering of the
                                      tagged sequence, and the bin-fetch skeleton
```

## Decision

> **M3 binds M2 block-bin plans to real trusted block contents and derives the SWAT-style
> abstract bin-read schedule.  It uses record contents for interior points and bin ordering,
> but it never collapses a block into one representative database key, it emits no merged
> output, and it performs zero physical I/O.**

## The fourteen frozen points

### 1. Two granularities: block is the allocation / I/O unit, record is the trusted comparison unit

```text
observable allocation / later I/O unit = LETIndex block
trusted merge comparison unit          = record
```

M2 groups contiguous logical **block ranks** into bins.  M3 may flatten the records *inside*
those blocks, in their already-sorted order, to validate content, sample an interior point
and reason about record-level merge semantics.  That does not make a block an atomic
sortable item, and merge correctness is still decided per record.

### 2. An M2 plan binds exactly the contiguous logical blocks it covers

Binding is legal only when `plan.block_count == run.block_count`; each M2 bin's
`[start, stop)` rank interval binds `run.blocks[start:stop]`.  Every real block is bound
exactly once, none is duplicated or omitted, the M2 bin order is preserved, and the M2
evidence (`sampled_loads`, `noisy_prefix_sums`, `additive_error_blocks`, dummy counts) is
carried through **unchanged**.  Dummy padding stays a count: no dummy block is materialised.

### 3. No block representative key is introduced

Cross-run order is never decided from a block's first key, its last key, its midpoint, its
rank, a `BlockId` or any physical identifier.  A block's key range can interleave with
another level's, so any such surrogate would define a wrong merge order (Decision 0002).

### 4. The interior point is sampled from the bin's actual record keys

For a bin with real content, the real records of its real blocks are concatenated in logical
block order — an already strictly increasing key sequence, re-validated on binding — and one
interior point is sampled from that sequence.  The selected point is therefore **one actual
`RecordKey` of that bin**, never a synthetic representative.

### 5. An empty-real bin uses the explicit `DUMMY_POS_INF` sentinel

A bin whose real blocks are exhausted (`real_count == 0`) has no record to sample.  The
pinned C++ code would carry a dummy datum, whose key is a numeric maximum; Python integers
have none, so M3 uses one explicit sentinel which:

- sorts **after** every real `RecordKey` (through `interior_point_sort_key`);
- is **not** a `RecordKey` (`isinstance(DUMMY_POS_INF, RecordKey)` is false);
- is **never** a database result — `InteriorPoint.record_key` refuses it explicitly;
- exists only for trusted abstract scheduling.

### 6. The pinned `DPInteriorPoint` weight expression is the provenance basis

With `baseExp = exp(privacy_epsilon)` and `load` the number of real records, the weight of
real record index `i` is exactly

```text
weight[i] = baseExp ** (min(i, load - i) + 1)
```

reproduced verbatim, including its asymmetric tail (for `load = 4` the weights are
`b, b², b³, b²`).  It is deliberately **not** "symmetrised" into a tidier formula, and the
exponent is evaluated with the same exponentiation-by-squaring routine the pinned
`fastPower` uses.  Only real records carry positive weight, so the pinned zero-weight dummy
tail can never be selected — which is exactly why sampling over the real records alone is
faithful.

### 7. Source and target interior streams are deterministic and domain-separated

The streams are planner-owned (no module-global RNG state), derived from each side's M2
config seed with a side-specific domain:

```text
swat-m-block/interior/source
swat-m-block/interior/target
```

Both differ from each other and from every M2 domain, so two sides that carry the *same*
config seed still draw from different sequences.  For a fixed pair of plans, contents,
configs and seeds the schedule is identical (in-process and across processes); changing
values while keeping keys cannot change the schedule; changing keys may change it, because
M3 is the first content-aware milestone.  No bit-for-bit equivalence with
`std::mt19937` + `std::discrete_distribution` is claimed, and the pinned function-local
`static` generator lifetime is not reproduced.

### 8. Signed-tag tie ordering matches the pinned `DOMerge` pair ordering

```text
source bin i -> tag +(i + 1)
target bin j -> tag -(j + 1)
sort by      -> (interior_point, signed_tag)
```

which is the pinned `std::pair` lexicographic order: interior point ascending, then signed
tag ascending on a tie.  An equal interior point therefore places the **target's negative
tag before the source's positive tag**.  The sorted tagged sequence is exposed as trusted
schedule evidence.

### 9. M3 produces an abstract bin-read order only

The schedule is a sequence of `(side, bin_index)`.  It contains no slot, handle or storage
reference, and it is an abstract *bin-fetch order*, not a physical read plan: one future
physical fetch will mean fetching a fixed-size padded bin representation, which M3 does not
define or execute.

**One M3 read is not one pinned `DOMerge` iteration.**  The pinned loop iterates
`binCnt = leftBinCnt + rightBinCnt` times, carries `j0`/`j1` state and interleaves
safe-output/frontier work; M3 emits only the *projected* fetch sequence that remains after
dropping no-op iterations and that deferred work.  The two coincide on well-formed input
(asserted by an independent pinned-loop regression whose fixture genuinely triggers the
`else if` fallback), but they are not the same machine.  **M4 must replay the original
per-tag loop and its fallback semantics when it introduces the `j0`/`j1` frontier state**,
rather than treating one M3 read as one merge step.

### 10. Zero physical, slot and trace behaviour

M3 performs:

```text
0 UntrustedStorage READ
0 UntrustedStorage WRITE
0 TraceEvent
0 SlotId lookup or scheduling
```

This is deliberate and load-bearing.  Mapping each bound logical block to its *current*
physical slot and reading in schedule order would publish exactly the logical-rank →
physical-access correlation that the whole mechanism exists to hide, and would silently
choose a physical staging design ahead of M4.  M3 freezes the semantic schedule at bin
identity granularity *before* M4 decides how padded bins are staged and fetched.

### 11. M2's block-unit noisy prefixes are not reinterpreted as record-unit safe-output prefixes

M3 carries `sampled_loads`, `noisy_prefix_sums` and `additive_error_blocks` unchanged, and
they stay expressed in **blocks**.  In particular the pinned `DOMerge` expression

```text
newCnt = leftPrefix[...] + rightPrefix[...] - 2 * additiveError
```

is **not** ported: the upstream prefixes and additive error are in *datum* units, while M2's
are in *block* units, and mixing the two would be a unit error.  The record-unit safe-output
frontier and the bounded trusted merge buffer are explicitly deferred to M4.

### 12. M1 remains the only exact newer-wins output oracle

M3 implements no second record-level merge and emits no merged output `C`.  Flattening all
real records of all bound bins reproduces each input run exactly (asserted), and the M1
oracle remains the sole authority for `C`.  The M1 result is independent of the M3 schedule,
and M1 is unchanged by M3.

### 13. No privacy theorem claim

M3 is an adaptation step in a block-granular mechanism: content binding and bin ordering are
semantic plumbing, not a proof.  Nothing here may be called proven (epsilon, delta)-DO, and
the pinned theorem is not inherited — the observable unit is a block, not a record.

### 14. M4 is not authorized by M3

```text
M4 - Physically stage/fetch padded bins and implement the trusted bounded merge executor
```

is **not** started, not scoped and not implied by this decision.  It needs its own issue and
decision record.

## Documented divergences from the pinned source

| # | Pinned source | M3 | Why |
|---|---|---|---|
| D1 | `DPInteriorPoint` is called for **every** bin, including an all-dummy bin, and its distribution is over `bin.size()` slots with zero weight on the dummy tail | M3 samples over the bin's real records only, and a zero-real bin gets the sentinel without drawing | the zero-weight tail can never be selected; and a draw whose result is discarded is a C++ artifact, not semantics. Consequence: the stream offset depends on how many bins have real content — deterministic, but not claimed identical to the pinned call pattern |
| D2 | an all-dummy bin yields the dummy datum (numeric maximum key) | `DUMMY_POS_INF` sentinel (§5) | Python integers have no maximum; the sentinel's ordering *is* the pinned behaviour |
| D3 | `DOMerge`'s loop runs `binCnt = leftBinCnt + rightBinCnt` iterations and, when a tag's own side is exhausted, **falls back to fetching from the other side** (`else if`) | M3 requests a side's next unread bin only when its own tag arrives and a bin remains; it models no loop count and no per-iteration state | the fallback is **real**, not dead code: with the source exhausted while the target still holds unread bins, a later source tag does fetch from the target side. What holds is narrower — after projecting away no-op iterations *and* the deferred safe-output work (which M3 does not model at all), the **emitted bin-fetch sequence** equals M3's on every well-formed input, because once a side is exhausted no further same-side fetch can occur, so a fallback fetch only ever pulls the other side's next bin forward to an earlier iteration without reordering the sequence. M3 reproduces that projected sequence, **not** the pinned per-iteration machine — see §9 |
| D4 | `static std::mt19937` inside `DPInteriorPoint`, shared across calls | planner-owned stream taken per side and per sampling call | the pinned generator's lifetime spans calls and therefore depends on call order; M3's evidence must depend only on the plan, contents and seed |
| D5 | the pinned weight vector is built over `bin.size()` (slots) | built over the real record count | same selection behaviour, fewer entries; documented so the correspondence is explicit |

## Consequences

- The abstract schedule is now derived from real contents, which is the precondition for any
  executable SWAT-M-Block merge — and it is derived *without* yet committing to a physical
  staging design.
- Because the two sides' interior streams are domain-separated and deterministic, a schedule
  is fully reproducible from `(plans, contents, configs)`, which is what makes later
  physical milestones testable against a fixed expectation.
- The M2 block-unit prefix evidence now travels alongside record-level content.  The unit
  mismatch is recorded here rather than silently resolved, so M4 must decide the safe-output
  frontier deliberately.
