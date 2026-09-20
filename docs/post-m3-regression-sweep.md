# Post-M3 regression sweep (QA-1)

`codes/tools/post_m3_regression_sweep.py` is a **stabilization harness**.  It freezes no new
semantics and implements no mechanism: it re-derives the already accepted M1/M2/M3 contracts
from independently written oracles and invariants, and checks the shipped implementations
against them over deterministic generated cases.  It was added by Issue #8 (`[QA-1]`) and is
deliberately **not** part of the production package.

## Why it exists

M1, M2 and M3 are each covered by their own focused suites.  Those suites assert the contract
on curated fixtures; this harness re-asserts the same contract on a generated sweep, with the
expected answers produced by code that does not share the implementation's logic:

- the exact merged result is recomputed by a dict-based evaluator written from scratch;
- the canonical packing is recomputed arithmetically;
- the pinned `DOMerge` fetch order is reproduced by an independent simulator of the accepted
  pinned loop.

A PASS across thousands of cases therefore means the accepted behaviour is stable, not that a
second copy of the same reasoning was compared with itself.

## Architecture

```text
codes/tools/post_m3_regression_sweep.py
    independent_newer_wins                  dict-based exact newer-wins evaluator
    independent_expected_block_sizes        canonical packing, computed arithmetically
    pinned_projected_fetches                pinned DOMerge loop, projected to bin fetches
    physical_io_guard                       storage/trace entry points -> hard failure
    generate_m1_cases / m2_cases / m3_cases deterministic case generation
    check_m1_case / check_m2_case / check_m3_case   invariant checkers
    check_static_separation                 module-level separation sweep
    run_sweep / main                        --quick / --full / --seed
codes/tests/test_post_m3_regression_sweep.py
    focused QA tests over the harness itself (evaluators, teeth, determinism, CLI)
```

The harness is importable by tests (`importlib` from its path) but no module under
`codes/src/swat_m_block/` references it: `check_static_separation` fails the sweep if one ever
does, and a focused test asserts the same property directly.

## M1 checks

For every generated case (adjacent runs only, `source = L` → `target = L + 1`):

1. the exact result equals the independent evaluator — keys **and** values;
2. duplicate keys: the newer source value wins, both inputs are consumed, the key is emitted
   once, and `duplicate_count` equals the true intersection size;
3. the output key stream is strictly increasing and equals the exact merged key stream;
4. canonical packing: rank order `0..B-1`, every non-final block full, one final partial block,
   sizes equal to `independent_expected_block_sizes`;
5. several `advance(q)` schedules — the **whole** frozen `FunctionalMergeOracleResult` is
   equal across schedules (`None`, `(1,)`, `(2,)`, `(7,)`, `(1,1,1)`, `(5,3,2,9)`, `(1000,)`
   and, for some cases, a random valid segmentation);
6. the canonical PGM equals `build_batch_pgm(exact output key stream, epsilon)`;
7. the frozen incremental job drains to the same flattened record stream;
8. no `UntrustedStorage` operation and no trace event (instrumented subset);
9. no `SlotId` in the module's executable code (static check).

Case families: source-only, target-only, both-empty, disjoint (either side first), interleaved,
duplicate-heavy, identical keys, mixed-sign keys, sparse wide keys, single record and
many-record runs, across `items_per_block ∈ {1,2,3,4,8,16}`, `epsilon ∈ {0,1,2,4,8,16,64}`
and several adjacent level pairs.

## M2 checks

Generated over block counts (0 … 4096), seeds, `privacy_epsilon`, `privacy_delta` and valid
`security_lambda` values.  For every successful plan:

- the same configuration and seed reproduce the identical plan;
- every sampled load is a plain int inside the pinned support `[0, Z]`;
- `bin_capacity_blocks` is an even positive integer and equals the configuration's `Z`;
- `bin_count == ceil(factor * n)` and the plan materialises exactly `bin_count` bins
  (no appended bin);
- each bin has the fixed capacity, `real_count + dummy_count == capacity`, the rank interval
  width equals `real_count`, the bin load equals the corresponding sampled load, and
  `real_count ≤ sampled_load`;
- the rank intervals are contiguous, ordered, non-overlapping and cover `0..n-1` exactly once;
- the noisy prefix starts at 0, is monotone non-decreasing, stays inside the permitted window
  `|true_prefix − prefix[t]| ≤ Z`, and `additive_error_blocks` is exactly the maximum deviation;
- no physical or observational field exists on the plan.

For failures, three families are swept: a natural shortfall, and two injected-sampler
shortfalls (`load_sampler` returning 0 or a small constant) — the documented testing seam.  In
every case the failure must be the typed `InsufficientSampledCapacity`, must carry
`block_count / sampled_capacity / bin_count / bin_capacity_blocks / seed / shortfall_blocks`,
must be reproducible, and must never be repaired: the harness asserts that exactly one sample
is drawn per planned bin (no resample-until-success), that the reported capacity equals
independently re-derived sampled loads, that no plan is produced, and that a load outside
`[0, Z]` is refused as a different typed error rather than silently clamped.

## M3 checks

Each case builds canonical **mutable** `Block` inputs and a `LogicalBlockRunView` over them:

1. **snapshot immutability** — the view holds `LogicalBlockSnapshot` objects, not the caller's
   blocks; after the checkers take a baseline binding and schedule, every block with spare
   capacity is mutated in place (`Block.add_record(...)`, injecting the *preceding* block's
   maximum key, which would break the run-level cross-block order), and the records, keys,
   binding, interior points and schedule structure must all be unchanged;
2. **exact reconstruction** — flattening the bound bins (records and keys) reproduces each
   input run exactly, and the bound bins hold snapshots;
3. **interior points** — one point per bin, drawn from that bin's actual key set, with
   `real_keys()[record_index] == key`;
4. **zero-real bins** — `DUMMY_POS_INF`, carrying no record index;
5. **values-only invariance** — the same keys with different record payloads produce the same
   schedule structure;
6. **determinism** — identical plan/config/content reproduce the identical schedule;
7. **domain separation** — the source/target domains are the fixed side domains and differ
   from each other, and the derived stream seeds differ;
8. **signed-tag tie order** — on equal interior points the target's negative tag sorts before
   the source's positive tag;
9. **coverage** — every planned bin of each side is read exactly once, no read is out of range,
   and the two preloads are the source's bin 0 then the target's bin 0;
10. **zero surface** — no `SlotId`, no storage operation, no trace event (static + instrumented
    subset), and sampling returns a new allocation instead of mutating the binding.

## Pinned `DOMerge` projection

`pinned_projected_fetches` is an independent simulator of the accepted pinned skeleton:
preload bin 0 of each non-empty side, then one iteration per sorted tagged interior point,
where a source tag fetches the source's next unfetched bin (`j0`) and **otherwise** the
target's next unfetched bin is fetched (`j1`).  That `else if` is the real fallback: a source
tag arriving after the source is exhausted fetches from the target side.  The simulator never
imports or calls the M3 schedule builder — a focused test asserts that.

For every two-sided generated schedule the projected fetch sequence must equal the M3
`AbstractBinRead` sequence exactly, and the number of cases in which the fallback actually
fired is counted and reported.  What is *not* claimed: the pinned loop's iteration count and
its deferred safe-output/frontier work are not modelled, so the comparison is an equivalence of
**projected bin fetches**, not of merge-loop iterations.  Decision 0005 also records that M4
must replay the pinned per-tag loop and fallback rather than assuming one M3 read is one merge
iteration.

## Budgets, CLI and determinism

```text
python tools/post_m3_regression_sweep.py --quick     # seconds, suitable for local review
python tools/post_m3_regression_sweep.py --full      # thousands of combined cases
python tools/post_m3_regression_sweep.py --quick --seed 7
```

`--quick` generates 48 M1 + 150 M2 + 60 M3 cases; `--full` generates 640 + 2400 + 2000.  All
generation flows through a single `random.Random(master_seed)`, so a fixed seed is
deterministic in-process and across processes (a focused test compares two subprocess runs
line by line, ignoring the elapsed line).  Generation is bounded: there is no unbounded random
search and no time-based input.

Elapsed time is printed as **operational information only**.  This harness is not a benchmark
and makes no performance claim.

## Output

```text
post-M3 regression sweep (qa-1) — mode=full master_seed=20260920
  total cases                     : 4996
  M1 cases                        : 640
  M2 successful plans             : 836
  M2 explicit failures            : 1564
  M3 schedules                    : 1956 (two-sided 1296, one-sided 660, skipped 44)
  pinned merge-loop comparisons    : 1296
  pinned fallback-triggered cases : 372
  mutation-checked M3 cases       : 1197
  instrumented (zero-I/O) cases   : 164
  elapsed                         : 18.24 s (operational only; no performance claim)
  PASS
```

`skipped` counts M3 cases whose generated configuration raised the accepted
`InsufficientSampledCapacity` while planning a side; they are reported rather than hidden, and
they are the same explicit failure the M2 sweep checks.

On the first mismatch the tool prints a compact reproduction block — master seed, case index,
subsystem, violated invariant, a truncated description of the inputs and the command to
reproduce — and exits non-zero.  Whole objects are never dumped:

```text
FAILURE
  master seed : <seed>
  case index  : <index>
  subsystem   : M1 | M2 | M3 | static
  invariant   : <the violated contract>
  detail      : <what was observed>
  inputs      : family=... source_keys=[...]...+N more (n=...) ...
  reproduce   : python tools/post_m3_regression_sweep.py --quick --seed <seed>
```

## What this sweep deliberately does not do

- no M4 mechanism of any kind: no physical slot scheduling, no padded-bin staging, no dummy or
  cover I/O, no safe-output frontier, no bounded merge executor, no output publication or
  oblivious shuffle, no PRP, no de-amortisation;
- no ciphertext / SGX / AES;
- no attack and no benchmark;
- no privacy theorem claim;
- no change to M1/M2/M3 production semantics: if a generated case exposed a defect, the
  harness stops and reports rather than repairing anything;
- no external dependency: it uses the standard library plus the repository's existing
  dependencies only.

The independent evaluators live under `codes/tools/` and are never imported by the production
package; the frozen 35-file substrate is untouched and still verified by
`provenance/verify_manifest.py`.
