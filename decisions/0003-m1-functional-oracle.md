# Decision 0003 — M1 functional oracle

```text
Status:   ACCEPTED
Date:     2026-09-20
Milestone: M1 (functional SWAT-M-Block oracle, Issue #2)
```

## Context

M0 froze the LETIndex common substrate and the SWAT-M-Block contract
(`decisions/0001`, `decisions/0002`).  The frozen contract fixes a nine-step pipeline
whose first real content step is a SWAT-style noisy/padded block-bin allocation, and
whose *logical* result is the exact record-level newer-wins sorted union of the two input
levels.

There is an ordering problem: a privacy mechanism can only be judged against the exact
logical answer it is supposed to preserve.  If the observable schedule and the logical
result were built in the same step, a difference between two implementations could not be
attributed to the mechanism or to a semantics bug.  M1 therefore deliberately implements
the second half first.

## Decision

> **M1 is a logical correctness oracle only.  It computes the exact logical result of a
> legal merge — the canonical output blocks and the canonical PGM of the newer-wins sorted
> union — by driving the frozen common logical machinery.  It makes no privacy or
> security claim, provides no physical schedule, and performs no physical I/O.**

Frozen by this decision:

1. **The oracle is a correctness oracle.**  Its whole content is
   `C = exact sorted union(source, target)` under the frozen newer-wins rule (source is
   newer; on an equal key the source value wins and exactly one record is emitted), packed
   canonically and indexed by the canonical PGM.

2. **No privacy or security claim.**  M1 says nothing about differential obliviousness,
   `(epsilon, delta)` or any other privacy notion, and nothing here may be described as a
   protected execution.  Those questions belong to the milestone that actually introduces
   an observable schedule (M2 onward), and any claim remains unsupported until a separate
   proof exists (`decisions/0002` §4).

3. **No physical schedule.**  M1 defines no READ/WRITE order over blocks, no bin geometry,
   no padding, no dummy/cover access, no output permutation and no publication.  It is
   downstream of nothing physical and upstream of nothing physical.

4. **Zero physical surface.**  The M1 implementation performs zero `UntrustedStorage`
   access, emits zero `TraceEvent`, addresses zero physical `SlotId`, allocates no dummy or
   cover element, and uses no randomness.  This is asserted by tests, not merely intended.

5. **Reuse, not re-implementation.**  The merge, the canonical block packing and the
   canonical PGM come from the frozen common substrate
   (`enhanced_letindex.incremental_merge.IncrementalMergeJob`, which drives
   `enhanced_letindex.pgm`) — M1 contains no second newer-wins loop and no second packing
   or PGM rule.  Allowing the oracle to retain the full logical output blocks is
   legitimate precisely because M1 makes no bounded-memory claim.

6. **The oracle is the reference for later milestones.**  Any future SWAT-M-Block
   execution must reproduce the oracle's logical result exactly; the oracle may not be
   relaxed, padded or randomized to accommodate a mechanism.

7. **M2 is not authorized here.**  The next milestone, *M2 — SWAT-style noisy/padded
   block-bin allocation*, is **not** started, not scoped and not implied by this decision.
   It requires its own issue and its own decision record.

## Consequences

- The M1 module lives in `codes/src/swat_m_block/functional_oracle.py`, never in the
  frozen `enhanced_letindex` package, which stays byte-identical to the M0 snapshot.
- Later milestones compare against M1 rather than against each other: the oracle is the
  arbiter of logical correctness, which is what makes "the mechanism preserves the
  logical result" a *checkable* statement.
- Because M1 is exact and deterministic, its tests can assert equality — of records,
  block packing and PGM — instead of weaker properties.
- Nothing in this decision licenses any change to `decisions/0002`'s output-write rule:
  when the physical output path is built, rank-ordered `PRP(rank)` writes remain forbidden
  and an oblivious output shuffle remains required.
