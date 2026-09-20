# M2 — the SWAT-style block-bin allocation planner

Milestone record for Issue #4.  The normative decision is
`decisions/0004-m2-block-bin-allocation.md`; this document explains the implementation.

## 1. What M2 answers

> Given a sorted logical run containing `n` LETIndex blocks and SWAT-style privacy
> parameters, how many fixed-capacity bins are planned, how many real block items are
> assigned to each bin, how much dummy padding each bin carries, and what noisy prefix-sum
> evidence is produced?

M2 does **not** perform physical I/O, does not merge two runs, does not bind keys, and does
not claim `(epsilon, delta)`-DO for the block adaptation.

## 2. The adaptation boundary

```text
1 allocation item == 1 logical LETIndex block
atomic bucket capacity == 1 block
```

Everything is counted in **blocks**: `bin_capacity_blocks` (the derived `Z`), the sampled
bin loads, `real_count`, `dummy_count`, the noisy prefix sums and `additive_error_blocks`.

A block is an allocation item, but it is **not** an atomic sortable database record: two
levels' block key ranges may interleave (Decision 0002).  M2 therefore allocates only
contiguous logical block **ranks** of one already-sorted run, and it defines no cross-run
merge order and no representative key.

## 3. Public API

```python
from swat_m_block import (
    SwatBlockAllocationConfig, swat_reference_config,
    BlockBinPlan, BlockAllocationPlan,
    BlockAllocationError, InsufficientSampledCapacity,
    allocate_block_bins, compute_bin_capacity_blocks, compute_bin_count,
    compute_factor, compute_noisy_prefix_sums,
    compute_weight, geom_conv, GeometricLoadSampler, LaplaceSampler,
    derive_stream_seed, ATOMIC_BUCKET_CAPACITY_BLOCKS,
)

config = swat_reference_config(seed=20260920)      # lambda 512, eps 1.0, delta 1e-12
plan = allocate_block_bins(block_count=64, config=config)
```

### Configuration

```text
SwatBlockAllocationConfig(security_lambda, privacy_epsilon, privacy_delta, seed)
```

The privacy epsilon is deliberately **not** called `epsilon`: the LETIndex PGM epsilon is a
different quantity with a different role, and the two must not be confused at the API
boundary.  Validation: `security_lambda > 1` and `1 - log(lambda)^-2 > 0`;
`privacy_epsilon > 0`; `0 < privacy_delta < 1`; `seed` a plain non-negative integer;
booleans are never accepted as integers; `NaN`/`inf` are rejected.

### Plan types

`BlockBinPlan` (immutable): `bin_index`, `logical_rank_start`, `logical_rank_stop`,
`sampled_load`, `real_count`, `dummy_count`, `capacity` — the constructor refuses any bin
that violates `real_count + dummy_count == capacity`, `real_count <= sampled_load`,
`sampled_load <= capacity` or `logical_rank_stop - logical_rank_start == real_count`.

`BlockAllocationPlan` (immutable): `block_count`, `bin_capacity_blocks`, `bin_count`,
`factor`, `bins`, `sampled_loads`, `noisy_prefix_sums`, `additive_error_blocks`, `config`,
plus `sampled_capacity`, `planned_slots`, `dummy_count`, `is_empty`, `covered_ranks()`.

No physical `SlotId`, storage handle, key, record, PGM or trace event appears in either
type, and no dummy item is materialised: padding is a count and real content is a rank
interval, so a plan of `10^6` blocks costs O(bins) objects, not O(slots).

## 4. Pipeline

```text
block_count n, config(security_lambda, privacy_epsilon, privacy_delta, seed)
   |
   v
compute_bin_capacity_blocks(...)        pinned sizing search            -> Z (even)
   |
   v
factor = 2 * 1 / (Z * (1 - log(lambda)^-2))
   |
   v
bin_count B = ceil(factor * n)          (n == 0 -> B == 0, empty plan)
   |
   v
sample B raw loads in [0, Z]            pinned truncated-geometric law, planner-owned RNG
   |
   v
noisy prefix sums over the loads        pinned DPPrefixSum adaptation  -> prefix, additive error
   |
   v
consume ranks left to right             real_i = min(load_i, n - consumed)
                                        dummy_i = Z - real_i
   |
   v
BlockAllocationPlan                     (or raise InsufficientSampledCapacity)
```

## 5. The distribution kernel (pinned semantics)

`compute_weight(Z, privacy_epsilon)` reproduces `Dist.hpp::Geom::computeWeight` verbatim:

```text
alpha = exp(-epsilon);  half = Z / 2
w[half] = (1 - alpha) / (1 + alpha - 2 * alpha ** (half + 1))
multiplier = alpha
for i in 1..half:
    w[half + i] = w[half - i] = w[half + i - 1] * multiplier
    multiplier *= alpha
```

The recurrence multiplies by `alpha`, then `alpha**2`, then `alpha**3`, … so the weight at
distance `i` from the centre is `w[half] * alpha**(i*(i+1)/2)` — a **linear-geometric**
(triangular-exponent) decay, *not* the textbook `alpha**i` geometric.  The pinned
normalising constant is kept, which is why the raw vector need not sum to one; the sampler
normalises, exactly as `std::discrete_distribution` does upstream.  The M2 tests
independently rebuild both the recurrence and the (naive, O(n·m)) convolution and compare.

`compute_bin_capacity_blocks(...)` reproduces the `DOMerger.hpp` constructor search in block
units (`bucket_capacity_blocks == 1`, so the tail test is `pmf[0] <= privacy_delta`), and
**not** the pinned AES/datum byte-alignment rounding — M2 plans abstract block items, not
ciphertext byte addresses.  The search keeps the pinned variable updates and exit
condition; only the indeterminate-member case is defined explicitly (Decision 0004 §D1–D3).

## 6. Randomness contract

* A fixed config and seed give the same plan, in-process and **across processes**.
* The streams are planner-owned: no module-global mutable RNG state exists, and no
  `random.seed()` is called at import or module scope.
* Two domains are separated (`STREAM_DOMAIN_LOADS`, `STREAM_DOMAIN_LAPLACE`), each seeded
  by `derive_stream_seed(seed, domain)` — SHA-256 based, so it is stable across processes
  and platforms.  Drawing from one stream can never perturb the other.
* No bit-for-bit identity with `std::mt19937` + `std::discrete_distribution` is claimed.

## 7. Block-unit noisy prefix sums

`compute_noisy_prefix_sums(loads, *, privacy_epsilon, bin_capacity_blocks, laplace)` is the
block-unit analogue of `DPPrefixSum.hpp`: a Fenwick-style block-sum/noisy-block-sum
structure, the pinned monotonicity step, and the pinned bounded-window clipping:

```text
prefix[0] == 0
prefix[t] = max(prefix[t-1], cast(noisy_estimate[t]))          # monotone
prefix[t] = min(prefix[t], true_prefix[t] + bin_capacity)      # upper window
prefix[t] = max(prefix[t], true_prefix[t] - bin_capacity)      # lower window
additive_error_blocks = max_t |true_prefix[t] - prefix[t]|
```

so every entry stays inside `[true_prefix - Z, true_prefix + Z]`, the sequence never
decreases, and the reported additive error is exactly the maximum deviation.  The pinned
`(uint32_t)` cast of a possibly-negative double is replaced by
`pinned_uint32_cast`, which is identical to C++ inside the well-defined range and
explicitly defined outside it; the pinned function-local `static Laplace` becomes an
explicit planner-owned stream; the pinned unused `epsilon_ = epsilon / log2(binCnt)` is
omitted (the effective scale is `1 / privacy_epsilon`).  All documented in Decision 0004.

## 8. Allocation semantics and explicit failure

For bin `i`, `real_count_i = min(sampled_load_i, n - consumed)`, the bin owns the
contiguous logical ranks `[consumed, consumed + real_count_i)`, and
`dummy_count_i = Z - real_count_i`, so every bin has exactly `Z` planned slots.  When the
run is exhausted, later bins keep their sampled load but fill fewer (or zero) real items —
`sampled_load != real_count` is a normal, observable outcome, not an error.

If `sum(sampled_loads) < n`, the allocator raises
`InsufficientSampledCapacity(BlockAllocationError)` with `block_count`, `sampled_capacity`,
`bin_count`, `bin_capacity_blocks`, `seed` and `shortfall_blocks`.  It never appends a bin,
inflates a load, resamples until it fits, or changes the distribution: one call samples
exactly one load per bin, and repeating the call reproduces exactly the same failure.

## 9. Reference geometry (pinned by the tests)

```text
lambda = 512, privacy_epsilon = 1.0, privacy_delta = 1e-12
1 - log(lambda)^-2 = 0.9743040866542517
bin_capacity_blocks Z = 16        factor = 0.12829670090910578
bin_count = ceil(0.12829670090910578 * n)
   n = 1 -> 1     n = 8 -> 2      n = 16 -> 3
   n = 64 -> 9    n = 512 -> 66
```

Both numbers are re-derived independently inside the test suite (separate weight,
convolution and search implementations) before the pin is compared, so the pin is verified,
not guessed.

## 10. Testing seam

`allocate_block_bins(..., load_sampler=...)` accepts any object with a `sample() -> int`
method returning a value in `[0, Z]`.  That is the whole injection surface: the
deterministic-fixture tests, the out-of-range refusal and the controlled
insufficient-capacity failure all use it, and the production path uses the planner-owned
`GeometricLoadSampler`.  No mocking framework is involved and no large test API is exposed.

## 11. Scope guard

M2 performs zero `UntrustedStorage` operations, emits zero `TraceEvent`s, schedules zero
physical `SlotId`s, and contains no `DOAllocate` data path, no `DOMerge`, no dummy/cover
physical I/O, no ciphertext/SGX/AES handling, no interior point or key binding, no output
blocks, no output shuffle, no `PRP` writeback, no de-amortisation, no attack code and no
benchmarks.  The suite enforces this statically (token and import scans) and operationally
(every `UntrustedStorage` entry point replaced by a failure, `TraceCollector` instrumented,
plans built — neither is called).

Note: the M0/M1 absence guards used to refuse the tokens `bin_allocator`, `noisy_allocat`
and `padded_bin` by name; M2 authorises exactly that mechanism, so those tokens were removed
from the guards while every still-excluded mechanism stayed refused (Decision 0004,
"Consequences for the existing guards").  M1's oracle-specific guarantees are now asserted
against the oracle module, with M2 guarded by its own suite.

## 12. Milestones

| Milestone | Content | State |
|---|---|---|
| M0 | repository / bootstrap freeze | **accepted** (Issue #1, CLOSED) |
| M1 | functional oracle | **accepted** (Issue #2, CLOSED, PR #3 merged) |
| **M2** | block-bin allocation planner (this document) | **current** (Issue #4) |
| M3 | bind block-bin plans to trusted contents; build the SWAT-M-Block merge/read schedule | not authorised, **not started** |
| later | physical execution, cover I/O, output shuffle, de-amortisation, attacks, experiments | out of scope |
