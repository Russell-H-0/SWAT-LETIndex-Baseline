# Decision 0004 — M2 block-bin allocation planner

```text
Status:   ACCEPTED
Date:     2026-09-20
Milestone: M2 (SWAT-style block-bin allocation planner, Issue #4)
```

## Context

M1 froze the exact logical answer of a merge (`decisions/0003`).  M2 introduces the first
stochastic mechanism of SWAT-M-Block: the merge-side allocation geometry of the pinned SWAT
reference implementation, adapted from *record* items to *LETIndex block* items.  Nothing
in the pinned code can be transferred literally — its allocation unit is a datum inside a
bucket, its addresses are AES ciphertext offsets, and its RNG is a C++ static — so the
adaptation boundary has to be frozen explicitly rather than left to the reader.

Pinned provenance (read directly for this milestone):

```text
CongGroup/SWAT @ b33646061ec1899ccf75c9ba8fe43b2653c44a6b
include/enclave/Dist.hpp          Geom::computeWeight, Geom::geomConv, Geom, Laplace
include/enclave/DOMerger.hpp      constructor binCapacity sizing, DOAllocate bin loads
include/enclave/DPPrefixSum.hpp   noisy prefix sums + additive error
```

## Decision

> **M2 implements a SWAT-style stochastic block-bin allocation planner over one logical
> run of LETIndex blocks.  It plans bin geometry, real/dummy block counts, and noisy
> prefix-sum evidence, using the pinned SWAT distribution and sizing semantics adapted to
> block units.  It is a planning kernel: it is not the pinned `DOAllocate` data path, it
> performs no physical I/O, it binds no keys, and it claims no privacy theorem.**

## The twelve frozen points

### 1. One allocation item is one logical LETIndex block

`blocks` are counted in logical LETIndex blocks (`Block`), never records.  The planner's
input is a **block count**, and it takes nothing else — no records, no keys, no values, no
level, no PGM.  Changing the contents of a run without changing its block count cannot
change the plan for a fixed config and seed, and the API has no parameter through which
content could influence it.

### 2. Atomic bucket capacity is one block

```text
ATOMIC_BUCKET_CAPACITY_BLOCKS == 1
```

`compute_bin_capacity_blocks(..., bucket_capacity_blocks=4)` is refused.  The pinned
`bucketCapacity` is what made `dataCnt / bucketCapacity` a bucket count; in the adaptation
one bucket is one block, so `bucket_count == block_count` and the pinned expressions keep
their shape (`factor = 2 * bucketCapacity / (...)` becomes `2 / (...)`).

### 3. Bin capacity and every prefix/error quantity are measured in blocks

`bin_capacity_blocks = Z`, each bin has exactly `Z` block slots, bin loads are in
`[0, Z]`, `real_count + dummy_count == Z` for every bin, `noisy_prefix_sums` and
`additive_error_blocks` are block counts.  No quantity in the plan is a record count, and
nothing silently falls back to record-level allocation.

### 4. The pinned SWAT source is the provenance basis

`distribution.py` re-implements, in Python, the pinned semantics of `Geom::computeWeight`,
`Geom::geomConv`, the `Geom` sampler, the `Laplace` sampler and the `DOMerger.hpp`
constructor sizing.  In particular the pinned `computeWeight` recurrence is reproduced
**verbatim**, including its linear-geometric (triangular-exponent) decay
`w[half ± i] = w[half] * alpha**(i*(i+1)/2)` and its pinned normalising constant — it is
*not* replaced by a textbook `alpha**i` geometric distribution, and the pinned raw weight
vector is not renormalised (the sampler normalises, exactly as `std::discrete_distribution`
does upstream).

### 5. The Python RNG is deterministic but is *not* claimed bit-identical to C++

A fixed config and seed give a reproducible plan, including across processes.  No claim is
made that these are the same random numbers as `std::mt19937` +
`std::discrete_distribution`, nor that the pinned static RNG lifetime is reproduced:
M2 uses planner-owned, domain-separated streams (one for the geometric loads, one for the
Laplace noise, both derived from the config seed with SHA-256), and there is no
module-global mutable RNG state anywhere.

### 6. No AES/datum byte alignment in the abstract block planner

The pinned constructor ends with `binCapacity = upperBound(datumSize, binCapacity)`, which
rounds the capacity so that a bucket of `binCapacity` datums plus the AEAD tag fits a
byte-aligned encrypted bucket.  M2 plans abstract block items, not ciphertext byte
addresses, so **the byte-alignment rounding is not ported** and the derived
`bin_capacity_blocks` is the convolutional sizing result itself.  The result is still an
even positive integer, as the pinned model requires.

### 7. Insufficient sampled capacity is an explicit failure, never silently repaired

If `sum(sampled_loads) < block_count` the allocator raises
`InsufficientSampledCapacity(BlockAllocationError)`, carrying `block_count`,
`sampled_capacity`, `bin_count`, `bin_capacity_blocks` and `seed` (plus a
`shortfall_blocks` property).  It never appends an extra bin, inflates the last sampled
load, resamples until the sample happens to suffice, or alters the sampled distribution;
one call samples exactly one load per planned bin.  The failure surface is deliberately
public so that a later milestone can measure the empirical failure rate, which is why the
result is not "fixed up" here.

### 8. M2 is not full `DOAllocate`

M2 = the stochastic block-bin allocation *planning* kernel: bin count, sampled loads,
padded bin geometry and noisy prefix-sum evidence for **one** run.  It is not the pinned
`DOAllocate` data path: no ciphertext handling, no SGX/AES, no bitonic sort of decrypted
data, no bin publication, no `DOMerge`, no output blocks, no output shuffle, no `PRP`
writeback, no de-amortisation, no cross-run merge schedule.

### 9. The DP interior point is deferred to M3

The pinned `DOAllocate` also computes a DP interior point from the actual decrypted
elements of each bin so that a later merge can route by key.  M2 has no bin contents and
invents no block representative key: no first key, no last key, no midpoint key, and no
block rank used as a cross-level key surrogate.  Any of those would prematurely define
cross-run merge semantics and would be wrong whenever two levels' block key ranges
interleave (Decision 0002).  Key/interior-point binding belongs to M3, where actual bin
contents exist in the trusted merge path.

### 10. No physical I/O and no observable surface

Zero `UntrustedStorage` READ, zero `UntrustedStorage` WRITE, zero `TraceEvent`, zero
physical `SlotId` scheduling.  M2 *does* use seeded randomness to construct the plan, but
the plan is not an REE observation: it is a description of intended block placement, and
constructing it touches no storage.  The tests enforce this statically and operationally
(every `UntrustedStorage` entry point is replaced by a failure and `TraceCollector` is
instrumented while plans are built).

### 11. No formal privacy theorem claim

The pinned (epsilon, delta) result is stated for the record/element-level mechanism.  M2 is
a *block-granular* adaptation, made of a distribution kernel and a planner that has not yet
been executed against a physical schedule, and it therefore inherits no theorem.  M2 is
described as a "SWAT-style block-granular adaptation" and nothing here may be called
proven (epsilon, delta)-differentially oblivious.

### 12. M3 is not authorised by M2

```text
M3 — Bind block-bin plans to trusted contents and build the SWAT-M-Block merge/read schedule
```

is **not** started, not scoped and not implied by this decision.  It requires its own issue
and decision record.

## Documented divergences from the pinned source

Every one of these is an explicit choice, not an accident, and each is asserted by the M2
tests where it is observable:

| # | Pinned source | M2 | Why |
|---|---|---|---|
| D1 | `binCapacity = upperBound(datumSize, binCapacity)` | not ported | AES/datum byte alignment has no meaning for abstract block items (§6) |
| D2 | `binCapacity` member is left uninitialised when the sizing loop body never executes (e.g. `lambda = 3`, `epsilon = 1`, where the upper candidate collapses to 2) | the degenerate range is resolved explicitly as `max(2, upper)` | the pinned read is indeterminate; M2 does not emulate an indeterminate read |
| D3 | the search's exit condition can return the last *probe*, which may be a candidate that failed the tail test | preserved faithfully (same variable updates, same exit condition, same returned value) | the request was to preserve the convolutional sizing mathematics; only the UB was replaced |
| D4 | `prefixSum[t] = (uint32_t) curNoise` casts a possibly-negative `double` — undefined behaviour | `pinned_uint32_cast`: identical to C++ truncation toward zero inside `[0, 2**32)`, and explicitly defined outside it (non-finite refused, out-of-range clamped) | do not emulate undefined behaviour; keep the intended integer semantics |
| D5 | `static Laplace laplace(seed)` — a function-local static whose stream lifetime spans calls | an explicit planner-owned `LaplaceSampler` passed into the prefix computation | the pinned stream depends on call order across the process; M2's noise must depend only on the plan and the seed |
| D6 | `const double epsilon_ = epsilon / log2(binCnt)` is computed and never used; the effective Laplace scale is `1.0 / epsilon` | the unused variable is omitted; the effective scale `1 / privacy_epsilon` is implemented | faithful to the *executed* pinned semantics |
| D7 | sampling clamps each load by the remaining data (`min(geom.sample(), dataCnt - totalLoad)`), so a short sample silently yields a shorter run | raw loads are sampled in `[0, Z]` and truncation happens at consumption (`real_count = min(load, n - consumed)`) | makes the short-sample case *detectable* (§7) instead of silent; the real counts are the same when the sample suffices |
| D8 | the pinned `Laplace` formula evaluates `log(0)` if the uniform draw is exactly `0.0`, giving a non-finite sample | the draw is kept on the open interval `(0, 1)` | the pinned continuation is undefined; the substitution is unobservable in practice (probability `2**-53`) |
| D9 | prefix sums are computed over the already-clamped `binLoads` | computed over the raw sampled loads | §3 of the issue fixes the raw loads as the prefix input |
| D10 | `std::mt19937` + `std::discrete_distribution` | Python `random.Random` over the same weights | stream identity is not claimed (§5) |

## Derived reference geometry

For the pinned SWAT reference configuration (`lambda = 512`, `privacy_epsilon = 1.0`,
`privacy_delta = 1e-12`), one block per bucket:

```text
1 - log(lambda)^-2            = 0.9743040866542517
upper candidate (raw floor)   = 9447
bin_capacity_blocks  Z        = 16
factor                        = 0.12829670090910578
bin_count(n) = ceil(factor*n) : n=1 -> 1, n=8 -> 2, n=16 -> 3, n=64 -> 9, n=512 -> 66
```

`Z = 16` and `factor = 0.12829670090910578` are pinned by the M2 tests, and the tests
*independently re-derive* both (separate weight, convolution and sizing-search
implementations) before comparing, so the pin is not a hard-coded guess.

## Consequences for the existing guards

M2 authorises the mechanism that the M0/M1 absence guards previously refused by name, so
those token lists were narrowed rather than left to fail: the mechanisms M2 still excludes
(DO data path, `DOMerge`, the physical surface, the output permutation, `PRP` writeback,
de-amortisation, the DP interior point) remain refused repo-wide, and M1's oracle-specific
guarantees (no randomness, no mechanism, no physical import) are now asserted against the
oracle module, with the M2 planner guarded by its own suite.  No guarantee was dropped.

## Consequences

- The geometry, the loads and the padding are now reproducible and measurable, which is the
  precondition for the empirical failure-rate work M2 deliberately exposes rather than
  hides.
- Because M2 is data-independent, it can be planned before any bin content exists — which
  is exactly why M3 has to supply the interior point: the plan says *how many* blocks go
  where, never *which keys*.
- The M1 oracle remains untouched and unmodified by M2; the two milestones meet only in a
  later, content-aware milestone.
