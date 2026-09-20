"""M2: the SWAT-style **block-bin allocation planner**.

This is the first stochastic mechanism of SWAT-M-Block, and it is a *planning* layer
only: given one sorted logical run of ``n`` LETIndex blocks and SWAT-style privacy
parameters, it plans how many fixed-capacity bins are needed, how many real block items
each bin carries, how much dummy padding each bin carries, and what noisy prefix-sum
evidence is produced.

```text
1 allocation item == 1 logical LETIndex block
atomic bucket capacity == 1 block
```

and every quantity below — ``bin_capacity_blocks``, bin loads, ``real_count``,
``dummy_count``, the noisy prefix sums and ``additive_error_blocks`` — is expressed in
**blocks**.  This is the block-granular adaptation boundary of Decision 0004; nothing here
silently falls back to record-level allocation.

A block is an *allocation item*, but it is **not** an atomic sortable database record:
two levels' block key ranges may interleave (Decision 0002 stays binding).  M2 therefore
allocates only contiguous logical block **ranks** of one already-sorted run.  It computes
no representative key — no first/last/midpoint key and no rank-as-key surrogate — and it
defines no cross-run merge order.  The DP interior point of the pinned ``DOAllocate``
belongs to M3, where actual bin contents exist in the trusted path.

What M2 does **not** do: no ``UntrustedStorage`` READ or WRITE, no ``TraceEvent``, no
``SlotId`` scheduling, no physical publication, no dummy/cover I/O, no ``DOMerge``, no
output blocks, no output shuffle, no ``PRP`` writeback, no de-amortisation, no attack code,
no benchmarks, and no ``(epsilon, delta)`` privacy claim.  Seeded randomness *is* used, but
only to construct the plan: the plan is not an REE trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .distribution import (
    ATOMIC_BUCKET_CAPACITY_BLOCKS,
    STREAM_DOMAIN_LAPLACE,
    BlockAllocationError,
    GeometricLoadSampler,
    LaplaceSampler,
    compute_bin_capacity_blocks,
    derive_stream_seed,
    pinned_uint32_cast,
)

__all__ = [
    "ATOMIC_BUCKET_CAPACITY_BLOCKS",
    "SWAT_REFERENCE_PRIVACY_DELTA",
    "SWAT_REFERENCE_PRIVACY_EPSILON",
    "SWAT_REFERENCE_SECURITY_LAMBDA",
    "BlockAllocationError",
    "BlockAllocationPlan",
    "BlockBinPlan",
    "InsufficientSampledCapacity",
    "SwatBlockAllocationConfig",
    "allocate_block_bins",
    "compute_bin_capacity_blocks",
    "compute_bin_count",
    "compute_factor",
    "compute_noisy_prefix_sums",
    "swat_reference_config",
]

#: Pinned SWAT reference experiment values (``lambda``, ``epsilon``, ``delta``).
#: The seed is never defaulted: it must be explicit, so a plan is reproducible on purpose.
SWAT_REFERENCE_SECURITY_LAMBDA = 512
SWAT_REFERENCE_PRIVACY_EPSILON = 1.0
SWAT_REFERENCE_PRIVACY_DELTA = 1e-12


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _require_plain_int(value: object, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BlockAllocationError(
            f"{name} must be a plain int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise BlockAllocationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_real_in_open_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlockAllocationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise BlockAllocationError(
            f"{name} must be a finite value in the open interval (0, 1), got {value!r}"
        )
    return number


def _require_positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlockAllocationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BlockAllocationError(f"{name} must be a finite value > 0, got {value!r}")
    return number


@dataclass(frozen=True)
class SwatBlockAllocationConfig:
    """The SWAT privacy parameters of one block-bin allocation plan.

    The privacy epsilon is deliberately **not** called ``epsilon``: the LETIndex PGM
    epsilon is a different quantity with a different role, and the two must never be
    confused at the API boundary.
    """

    security_lambda: int
    privacy_epsilon: float
    privacy_delta: float
    seed: int

    def __post_init__(self) -> None:
        lam = _require_plain_int(self.security_lambda, "security_lambda", minimum=2)
        epsilon = _require_positive_real(self.privacy_epsilon, "privacy_epsilon")
        delta = _require_real_in_open_unit(self.privacy_delta, "privacy_delta")
        seed = _require_plain_int(self.seed, "seed", minimum=0)
        if not math.log(lam) ** 2 > 1.0:
            raise BlockAllocationError(
                f"security_lambda = {lam} must satisfy 1 - log(lambda)^-2 > 0 "
                f"(equivalently log(lambda)^2 > 1)"
            )
        object.__setattr__(self, "security_lambda", lam)
        object.__setattr__(self, "privacy_epsilon", epsilon)
        object.__setattr__(self, "privacy_delta", delta)
        object.__setattr__(self, "seed", seed)

    @property
    def decay(self) -> float:
        """The pinned ``1 - log(lambda) ** -2`` factor."""
        return 1.0 - math.log(self.security_lambda) ** -2


def swat_reference_config(seed: int) -> SwatBlockAllocationConfig:
    """The pinned SWAT reference parameters (``lambda = 512``, ``epsilon = 1.0``,
    ``delta = 1e-12``) with an explicit seed."""
    return SwatBlockAllocationConfig(
        security_lambda=SWAT_REFERENCE_SECURITY_LAMBDA,
        privacy_epsilon=SWAT_REFERENCE_PRIVACY_EPSILON,
        privacy_delta=SWAT_REFERENCE_PRIVACY_DELTA,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class InsufficientSampledCapacity(BlockAllocationError):
    """The sampled bin loads cannot cover every real block of the run.

    This is an explicit, typed allocation failure — never silently repaired.  The
    allocator does **not** append an extra bin, inflate the last sampled load, resample
    until the loads happen to suffice, or otherwise change the sampled distribution; a
    caller that wants a plan must supply a different config.

    The failure carries the evidence a later empirical failure-rate measurement needs.
    """

    def __init__(
        self,
        *,
        block_count: int,
        sampled_capacity: int,
        bin_count: int,
        bin_capacity_blocks: int,
        seed: int,
    ) -> None:
        self.block_count = block_count
        self.sampled_capacity = sampled_capacity
        self.bin_count = bin_count
        self.bin_capacity_blocks = bin_capacity_blocks
        self.seed = seed
        shortfall = block_count - sampled_capacity
        super().__init__(
            "the sampled bin loads cannot cover the run: "
            f"sampled_capacity = {sampled_capacity} block(s) < block_count = {block_count} "
            f"({shortfall} block(s) short) over {bin_count} bin(s) of capacity "
            f"{bin_capacity_blocks} block(s), seed = {seed}"
        )

    @property
    def shortfall_blocks(self) -> int:
        """How many blocks the sampled loads are short by."""
        return self.block_count - self.sampled_capacity


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def compute_factor(
    security_lambda: int,
    bin_capacity_blocks: int,
) -> float:
    """The pinned ``factor = 2 * bucketCapacity / (binCapacity * (1 - log(lambda)^-2))``.

    With the frozen ``bucket_capacity_blocks == 1`` this is
    ``2 / (bin_capacity_blocks * (1 - log(lambda)^-2))``.  The computation is done in
    floating point on purpose — the pinned expression is a ``double`` expression, and C/C++
    integer division must not be introduced accidentally.
    """
    lam = _require_plain_int(security_lambda, "security_lambda", minimum=2)
    capacity = _require_plain_int(bin_capacity_blocks, "bin_capacity_blocks", minimum=2)
    decay = 1.0 - math.log(lam) ** -2
    if not decay > 0.0:
        raise BlockAllocationError(
            f"security_lambda = {lam} does not satisfy 1 - log(lambda)^-2 > 0"
        )
    return 2.0 * ATOMIC_BUCKET_CAPACITY_BLOCKS / (capacity * decay)


def compute_bin_count(factor: float, block_count: int) -> int:
    """``bin_count = ceil(factor * block_count)`` for a run of ``block_count`` blocks.

    A run of zero blocks needs no bin at all — the pinned expression is an integer
    division ``dataCnt / bucketCapacity``, exact here because one bucket is one block.
    """
    rate = _require_positive_real(factor, "factor")
    blocks = _require_plain_int(block_count, "block_count", minimum=0)
    if blocks == 0:
        return 0
    return int(math.ceil(rate * blocks))


# ---------------------------------------------------------------------------
# noisy prefix sums (block-unit DPPrefixSum)
# ---------------------------------------------------------------------------


def _trailing_zeros(value: int) -> int:
    """The pinned ``while ((t_ & t) == 0) { t_ <<= 1; ++i; }`` loop."""
    count = 0
    bit = 1
    while (bit & value) == 0:
        bit <<= 1
        count += 1
    return count


def compute_noisy_prefix_sums(
    bin_loads: Sequence[int],
    *,
    privacy_epsilon: float,
    bin_capacity_blocks: int,
    laplace: LaplaceSampler,
) -> Tuple[Tuple[int, ...], int]:
    """The M2 block-unit analogue of pinned ``DPPrefixSum.hpp``.

    Input: the sampled bin loads in blocks, the privacy epsilon, the bin capacity in
    blocks and a planner-owned Laplace stream.  Output: ``(noisy_prefix_sums,
    additive_error_blocks)`` where the prefix vector has length ``bin_count + 1``.

    Structural properties required of the result (all asserted by the M2 tests):

    * ``prefix[0] == 0``;
    * the sequence is monotone non-decreasing — the pinned monotonicity step;
    * every entry is clipped into the pinned bounded window
      ``[true_prefix - bin_capacity, true_prefix + bin_capacity]``;
    * ``additive_error_blocks == max_t |true_prefix[t] - prefix[t]|``;
    * deterministic under a fixed seed (the Laplace stream is planner-owned).

    Documented adaptations of the pinned code, all recorded in Decision 0004:

    * the pinned function uses a *function-local static* Laplace object, whose stream
      lifetime spans calls and therefore depends on call order.  M2 takes the stream as an
      explicit argument, so the result depends only on the supplied plan and seed;
    * the pinned code computes ``epsilon_ = epsilon / log2(binCnt)`` and never uses it;
      the effective Laplace scale is ``1 / epsilon``, which is what M2 implements;
    * the pinned code casts a possibly-negative ``double`` to ``uint32_t`` (undefined
      behaviour).  :func:`swat_m_block.distribution.pinned_uint32_cast` defines the
      intended integer semantics explicitly instead of emulating UB.
    """
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    capacity = _require_plain_int(bin_capacity_blocks, "bin_capacity_blocks", minimum=2)
    if not isinstance(laplace, LaplaceSampler):
        raise BlockAllocationError(
            f"a planner-owned LaplaceSampler is required, got {type(laplace).__name__}"
        )

    loads: List[int] = []
    for load in bin_loads:
        value = _require_plain_int(load, "a sampled bin load", minimum=0)
        if value > capacity:
            raise BlockAllocationError(
                f"a sampled bin load must be within [0, {capacity}] block(s), got {value}"
            )
        loads.append(value)

    bin_count = len(loads)
    prefix: List[int] = [0] * (bin_count + 1)
    if bin_count == 0:
        return tuple(prefix), 0

    levels = int(math.ceil(math.log2(bin_count + 1)))
    block_sums: List[int] = [0] * levels
    noisy_sums: List[float] = [0.0] * levels
    current_sum = 0
    additive_error = 0

    for t in range(1, bin_count + 1):
        index = _trailing_zeros(t)
        target = loads[t - 1]
        current_sum += loads[t - 1]
        for lower in range(index):
            target += block_sums[lower]
            block_sums[lower] = 0
            noisy_sums[lower] = 0.0
        block_sums[index] = target
        noisy_sums[index] = target + laplace.sample(1.0 / epsilon)

        running = 0.0
        bit = 1
        level = 0
        while bit <= t:
            if t & bit:
                running += noisy_sums[level]
            bit <<= 1
            level += 1

        estimate = pinned_uint32_cast(running)
        value = prefix[t - 1] if estimate < prefix[t - 1] else estimate
        if value > current_sum + capacity:
            value = current_sum + capacity
        if value + capacity < current_sum:
            value = current_sum - capacity
        prefix[t] = value
        error = abs(current_sum - value)
        if error > additive_error:
            additive_error = error

    return tuple(prefix), additive_error


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockBinPlan:
    """One planned bin: a contiguous rank interval plus its padding.

    Only counts and rank ranges are stored — a plan never materialises one Python object
    per dummy slot.
    """

    bin_index: int
    logical_rank_start: int
    logical_rank_stop: int
    sampled_load: int
    real_count: int
    dummy_count: int
    capacity: int

    def __post_init__(self) -> None:
        index = _require_plain_int(self.bin_index, "bin_index", minimum=0)
        start = _require_plain_int(self.logical_rank_start, "logical_rank_start", minimum=0)
        stop = _require_plain_int(self.logical_rank_stop, "logical_rank_stop", minimum=0)
        load = _require_plain_int(self.sampled_load, "sampled_load", minimum=0)
        real = _require_plain_int(self.real_count, "real_count", minimum=0)
        dummy = _require_plain_int(self.dummy_count, "dummy_count", minimum=0)
        capacity = _require_plain_int(self.capacity, "capacity", minimum=2)
        if load > capacity:
            raise BlockAllocationError(
                f"bin {index}: sampled_load {load} exceeds capacity {capacity}"
            )
        if real > load:
            raise BlockAllocationError(
                f"bin {index}: real_count {real} cannot exceed sampled_load {load}"
            )
        if real + dummy != capacity:
            raise BlockAllocationError(
                f"bin {index}: real_count {real} + dummy_count {dummy} != capacity {capacity}"
            )
        if stop - start != real:
            raise BlockAllocationError(
                f"bin {index}: the rank interval [{start}, {stop}) does not hold "
                f"real_count = {real} block(s)"
            )
        object.__setattr__(self, "bin_index", index)
        object.__setattr__(self, "logical_rank_start", start)
        object.__setattr__(self, "logical_rank_stop", stop)
        object.__setattr__(self, "sampled_load", load)
        object.__setattr__(self, "real_count", real)
        object.__setattr__(self, "dummy_count", dummy)
        object.__setattr__(self, "capacity", capacity)

    def rank_range(self) -> range:
        """The contiguous logical block ranks of this bin (no materialised items)."""
        return range(self.logical_rank_start, self.logical_rank_stop)


@dataclass(frozen=True)
class BlockAllocationPlan:
    """The immutable M2 plan for one logical run of ``block_count`` blocks.

    The plan describes *planned* block items only: no physical ``SlotId``, storage handle,
    key, record, PGM or trace event appears anywhere in it, and no dummy item is
    materialised — ``dummy_count`` is a count, and the real content is a rank interval.
    """

    block_count: int
    bin_capacity_blocks: int
    bin_count: int
    factor: float
    bins: Tuple[BlockBinPlan, ...]
    sampled_loads: Tuple[int, ...]
    noisy_prefix_sums: Tuple[int, ...]
    additive_error_blocks: int
    config: SwatBlockAllocationConfig

    def __post_init__(self) -> None:
        blocks = _require_plain_int(self.block_count, "block_count", minimum=0)
        capacity = _require_plain_int(self.bin_capacity_blocks, "bin_capacity_blocks",
                                      minimum=2)
        count = _require_plain_int(self.bin_count, "bin_count", minimum=0)
        bins = tuple(self.bins)
        loads = tuple(self.sampled_loads)
        prefix = tuple(self.noisy_prefix_sums)
        error = _require_plain_int(self.additive_error_blocks, "additive_error_blocks",
                                   minimum=0)
        if not isinstance(self.config, SwatBlockAllocationConfig):
            raise BlockAllocationError(
                f"the plan needs a SwatBlockAllocationConfig, got {type(self.config).__name__}"
            )
        if len(bins) != count:
            raise BlockAllocationError(
                f"the plan reports {count} bin(s) but holds {len(bins)}"
            )
        if len(loads) != count:
            raise BlockAllocationError(
                f"the plan reports {count} bin(s) but holds {len(loads)} sampled load(s)"
            )
        if len(prefix) != count + 1:
            raise BlockAllocationError(
                f"the noisy prefix vector must hold bin_count + 1 = {count + 1} entries, "
                f"got {len(prefix)}"
            )
        if sum(bin_.real_count for bin_ in bins) != blocks:
            raise BlockAllocationError(
                "the planned bins do not cover exactly the run's real blocks"
            )
        object.__setattr__(self, "block_count", blocks)
        object.__setattr__(self, "bin_capacity_blocks", capacity)
        object.__setattr__(self, "bin_count", count)
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "sampled_loads", loads)
        object.__setattr__(self, "noisy_prefix_sums", prefix)
        object.__setattr__(self, "additive_error_blocks", error)

    @property
    def sampled_capacity(self) -> int:
        """Total sampled block capacity (real + dummy slots planned)."""
        return sum(self.sampled_loads)

    @property
    def planned_slots(self) -> int:
        """Total planned slots, ``bin_count * bin_capacity_blocks``."""
        return self.bin_count * self.bin_capacity_blocks

    @property
    def dummy_count(self) -> int:
        """Total dummy block items padded across all bins."""
        return sum(bin_.dummy_count for bin_ in self.bins)

    @property
    def is_empty(self) -> bool:
        return self.block_count == 0

    def covered_ranks(self) -> Tuple[int, ...]:
        """Every logical block rank the plan covers, in order."""
        return tuple(
            rank for bin_ in self.bins for rank in range(bin_.logical_rank_start,
                                                          bin_.logical_rank_stop)
        )


# ---------------------------------------------------------------------------
# the allocator
# ---------------------------------------------------------------------------


def _require_load(value: object, capacity: int) -> int:
    load = _require_plain_int(value, "a sampled bin load", minimum=0)
    if load > capacity:
        raise BlockAllocationError(
            f"a load sampler returned {load}, outside the permitted [0, {capacity}] "
            "block range"
        )
    return load


def allocate_block_bins(
    block_count: int,
    config: SwatBlockAllocationConfig,
    *,
    load_sampler: Optional[object] = None,
) -> BlockAllocationPlan:
    """Plan the SWAT-style padded bins of one logical run of ``block_count`` blocks.

    Steps: derive the block-unit bin capacity ``Z``, compute ``factor`` and
    ``bin_count = ceil(factor * block_count)``, sample ``bin_count`` raw loads from the
    pinned truncated-geometric law, compute the noisy prefix sums over those loads, then
    hand out contiguous logical ranks left to right with
    ``real_count_i = min(sampled_load_i, n - consumed)`` and ``dummy_count_i = Z -
    real_count_i``.

    ``load_sampler`` is the narrow testing seam: any object with a ``sample() -> int``
    method (returning a value in ``[0, Z]``) may be injected.  The normal path uses the
    planner-owned :class:`~swat_m_block.distribution.GeometricLoadSampler`.

    Raises :class:`InsufficientSampledCapacity` when ``sum(sampled_loads) < block_count``:
    the allocator never appends a bin, inflates a load, resamples until it fits, or
    otherwise alters the sampled distribution.
    """
    blocks = _require_plain_int(block_count, "block_count", minimum=0)
    if not isinstance(config, SwatBlockAllocationConfig):
        raise BlockAllocationError(
            f"allocate_block_bins needs a SwatBlockAllocationConfig, got "
            f"{type(config).__name__}"
        )

    capacity = compute_bin_capacity_blocks(
        security_lambda=config.security_lambda,
        privacy_epsilon=config.privacy_epsilon,
        privacy_delta=config.privacy_delta,
        bucket_capacity_blocks=ATOMIC_BUCKET_CAPACITY_BLOCKS,
    )
    factor = compute_factor(config.security_lambda, capacity)
    bin_count = compute_bin_count(factor, blocks)

    if blocks == 0:
        return BlockAllocationPlan(
            block_count=0,
            bin_capacity_blocks=capacity,
            bin_count=0,
            factor=factor,
            bins=(),
            sampled_loads=(),
            noisy_prefix_sums=(0,),
            additive_error_blocks=0,
            config=config,
        )

    sampler = load_sampler
    if sampler is None:
        sampler = GeometricLoadSampler(
            capacity, config.privacy_epsilon, seed=config.seed
        )
    if not callable(getattr(sampler, "sample", None)):
        raise BlockAllocationError(
            "a load sampler must expose a callable sample() method, got "
            f"{type(sampler).__name__}"
        )

    sampled_loads = tuple(
        _require_load(sampler.sample(), capacity) for _ in range(bin_count)
    )
    sampled_capacity = sum(sampled_loads)
    if sampled_capacity < blocks:
        raise InsufficientSampledCapacity(
            block_count=blocks,
            sampled_capacity=sampled_capacity,
            bin_count=bin_count,
            bin_capacity_blocks=capacity,
            seed=config.seed,
        )

    prefix, additive_error = compute_noisy_prefix_sums(
        sampled_loads,
        privacy_epsilon=config.privacy_epsilon,
        bin_capacity_blocks=capacity,
        laplace=LaplaceSampler(seed=config.seed, domain=STREAM_DOMAIN_LAPLACE),
    )

    bins: List[BlockBinPlan] = []
    consumed = 0
    for index, load in enumerate(sampled_loads):
        real = min(load, blocks - consumed)
        bins.append(
            BlockBinPlan(
                bin_index=index,
                logical_rank_start=consumed,
                logical_rank_stop=consumed + real,
                sampled_load=load,
                real_count=real,
                dummy_count=capacity - real,
                capacity=capacity,
            )
        )
        consumed += real

    return BlockAllocationPlan(
        block_count=blocks,
        bin_capacity_blocks=capacity,
        bin_count=bin_count,
        factor=factor,
        bins=tuple(bins),
        sampled_loads=sampled_loads,
        noisy_prefix_sums=prefix,
        additive_error_blocks=additive_error,
        config=config,
    )
