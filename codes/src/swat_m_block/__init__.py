"""SWAT-M-Block: the block-granular SWAT-style merge baseline.

Substrate: ``codes/src/enhanced_letindex/`` is a one-time, byte-identical, SHA-256
manifest-gated snapshot of the LETIndex common substrate taken from
``Russell-H-0/EnhancedLETIndex@4e68be75`` (see ``PROVENANCE.md``); it is frozen and must
not be edited here.

Milestones:

* **M0** (Issue #1, CLOSED) — repository / bootstrap freeze: the frozen substrate, the
  provenance gate and the SWAT-M-Block contract (``decisions/0001``, ``decisions/0002``).
* **M1** (Issue #2, CLOSED) — the **functional / logical oracle** only
  (:mod:`swat_m_block.functional_oracle`, ``decisions/0003``).
* **M2** (Issue #4) — the **SWAT-style stochastic block-bin allocation planner**
  (:mod:`swat_m_block.distribution` + :mod:`swat_m_block.bin_allocator`,
  ``decisions/0004``).

M2 freezes the block-granular adaptation boundary: one allocation item is one logical
LETIndex block and the atomic bucket capacity is one block, so the derived bin capacity,
the bin loads and every prefix/error quantity are measured in blocks.  A block is an
allocation item but **not** an atomic sortable database record — block key ranges may
interleave, so M2 allocates contiguous logical block ranks of one already-sorted run and
defines no cross-run merge order.

M2 is a planning kernel, not an execution path.  There is still **no** physical I/O
(``UntrustedStorage``), **no** ``TraceEvent``, **no** ``SlotId`` scheduling, no
``DOAllocate`` data path, no ``DOMerge``, no dummy/cover physical I/O, no interior point or
key binding, no output blocks, no output shuffle, no ``PRP`` writeback, no de-amortisation,
no attack code and no benchmarks.  Seeded, planner-owned randomness *is* used — to build
the plan, never to touch storage — and no ``(epsilon, delta)`` privacy theorem is claimed
for the block adaptation.  M3 is not authorised.
"""

from .bin_allocator import (
    ATOMIC_BUCKET_CAPACITY_BLOCKS,
    SWAT_REFERENCE_PRIVACY_DELTA,
    SWAT_REFERENCE_PRIVACY_EPSILON,
    SWAT_REFERENCE_SECURITY_LAMBDA,
    BlockAllocationError,
    BlockAllocationPlan,
    BlockBinPlan,
    InsufficientSampledCapacity,
    SwatBlockAllocationConfig,
    allocate_block_bins,
    compute_bin_capacity_blocks,
    compute_bin_count,
    compute_factor,
    compute_noisy_prefix_sums,
    swat_reference_config,
)
from .distribution import (
    STREAM_DOMAIN_LAPLACE,
    STREAM_DOMAIN_LOADS,
    GeometricLoadSampler,
    LaplaceSampler,
    compute_weight,
    derive_stream_seed,
    geom_conv,
    pinned_uint32_cast,
)
from .functional_oracle import (
    DEFAULT_SCHEDULE_POLICY,
    FunctionalMergeOracleResult,
    FunctionalOracleError,
    collect_output_blocks,
    flatten_blocks,
    functional_merge_oracle,
    output_key_stream,
)

__all__ = [
    # M1: the functional / logical oracle
    "DEFAULT_SCHEDULE_POLICY",
    "FunctionalMergeOracleResult",
    "FunctionalOracleError",
    "collect_output_blocks",
    "flatten_blocks",
    "functional_merge_oracle",
    "output_key_stream",
    # M2: the SWAT-style block-bin allocation planner
    "ATOMIC_BUCKET_CAPACITY_BLOCKS",
    "STREAM_DOMAIN_LAPLACE",
    "STREAM_DOMAIN_LOADS",
    "SWAT_REFERENCE_PRIVACY_DELTA",
    "SWAT_REFERENCE_PRIVACY_EPSILON",
    "SWAT_REFERENCE_SECURITY_LAMBDA",
    "BlockAllocationError",
    "BlockAllocationPlan",
    "BlockBinPlan",
    "GeometricLoadSampler",
    "InsufficientSampledCapacity",
    "LaplaceSampler",
    "SwatBlockAllocationConfig",
    "allocate_block_bins",
    "compute_bin_capacity_blocks",
    "compute_bin_count",
    "compute_factor",
    "compute_noisy_prefix_sums",
    "compute_weight",
    "derive_stream_seed",
    "geom_conv",
    "pinned_uint32_cast",
    "swat_reference_config",
]
