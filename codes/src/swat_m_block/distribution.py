"""M2: the pinned SWAT distribution kernel, adapted to LETIndex **block** units.

Provenance — this module re-implements, in Python, the distribution and sizing kernel of
the pinned SWAT reference implementation
(``CongGroup/SWAT`` @ ``b33646061ec1899ccf75c9ba8fe43b2653c44a6b``):

| Pinned source | What is ported here |
|---|---|
| ``include/enclave/Dist.hpp`` — ``fastPower`` | :func:`_fast_power` (exponentiation by squaring) |
| ``include/enclave/Dist.hpp`` — ``Geom::computeWeight`` | :func:`compute_weight`, verbatim recurrence |
| ``include/enclave/Dist.hpp`` — ``Geom::geomConv`` / ``multiply`` | :func:`geom_conv` (deterministic pure convolution) |
| ``include/enclave/Dist.hpp`` — ``Geom`` sampler | :class:`GeometricLoadSampler` (seeded discrete distribution) |
| ``include/enclave/Dist.hpp`` — ``Laplace`` | :class:`LaplaceSampler` |
| ``include/enclave/DOMerger.hpp`` — constructor sizing | :func:`compute_bin_capacity_blocks` |

No C++ source text is copied; the semantics are re-implemented and documented in
``decisions/0004-m2-block-bin-allocation.md``.

**Block-unit adaptation.**  One allocation item is one logical LETIndex block and the
atomic bucket capacity is one block (``ATOMIC_BUCKET_CAPACITY_BLOCKS == 1``), so every
quantity below — the derived bin capacity ``Z``, bin loads, and every prefix/error value —
is measured in **blocks**.  The pinned AES/datum byte-alignment rounding is deliberately
**not** ported (§ ``compute_bin_capacity_blocks``).

**Randomness.**  The streams here are planner-owned: no module-global mutable RNG state
exists, and the load stream and the Laplace stream are domain-separated.  A fixed config
and seed give a deterministic plan, but *no* claim is made that these are the same random
numbers as ``std::mt19937`` + ``std::discrete_distribution``.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "ATOMIC_BUCKET_CAPACITY_BLOCKS",
    "STREAM_DOMAIN_LAPLACE",
    "STREAM_DOMAIN_LOADS",
    "BlockAllocationError",
    "GeometricLoadSampler",
    "LaplaceSampler",
    "compute_bin_capacity_blocks",
    "compute_weight",
    "derive_stream_seed",
    "geom_conv",
]

#: One allocation item is one logical LETIndex block, and a bucket holds exactly one of
#: them.  This is the block-granular adaptation boundary (Decision 0004).
ATOMIC_BUCKET_CAPACITY_BLOCKS = 1

#: Domain-separation labels for the two planner-owned RNG streams.
STREAM_DOMAIN_LOADS = "swat-m-block/geometric-loads"
STREAM_DOMAIN_LAPLACE = "swat-m-block/laplace-prefix"

#: The pinned sizing search's upper-candidate exponent: ``log(lambda) ** 5``.
_UPPER_CANDIDATE_EXPONENT = 5

#: The smallest strictly positive value ``random.Random.random()`` can return (2**-53).
_SMALLEST_POSITIVE_UNIFORM = 2.0 ** -53

_UINT32_MAX = 2 ** 32 - 1


class BlockAllocationError(Exception):
    """A SWAT-M-Block block-bin allocation request violates the frozen M2 contract.

    Defined here because the shared distribution/sizing kernel validates its own
    parameters; :class:`swat_m_block.bin_allocator.InsufficientSampledCapacity` subclasses
    it, so callers have a single error vocabulary.
    """


# ---------------------------------------------------------------------------
# small numeric helpers (pinned semantics, explicit Python behaviour)
# ---------------------------------------------------------------------------


def _require_plain_int(value: object, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BlockAllocationError(
            f"{name} must be a plain int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise BlockAllocationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlockAllocationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BlockAllocationError(f"{name} must be a finite value > 0, got {value!r}")
    return number


def _require_even_positive(value: object, name: str) -> int:
    number = _require_plain_int(value, name, minimum=2)
    if number % 2 != 0:
        raise BlockAllocationError(
            f"{name} must be even (the pinned Geom constructor asserts an even "
            f"capacity), got {number}"
        )
    return number


def _fast_power(base: float, exponent: int) -> float:
    """Pinned ``Dist.hpp::fastPower<double>`` — exponentiation by squaring."""
    result = 1.0
    exp = int(exponent)
    value = float(base)
    while exp > 0:
        if exp & 1:
            result *= value
        exp >>= 1
        value *= value
    return result


def to_even_up(value: int) -> int:
    """Pinned ``DOMerger.hpp`` ``toEvenUB``: ``x + (x & 1)`` for ``x >= 0``."""
    number = _require_plain_int(value, "value", minimum=0)
    return number + (number & 1)


def _decay(security_lambda: int) -> float:
    """Pinned ``1 - log(lambda) ** -2``, required to be strictly positive."""
    logarithm = math.log(security_lambda)
    return 1.0 - logarithm ** -2


# ---------------------------------------------------------------------------
# pinned Geom weights and convolution
# ---------------------------------------------------------------------------


def compute_weight(bin_capacity_blocks: int, privacy_epsilon: float) -> Tuple[float, ...]:
    """Pinned ``Dist.hpp::Geom::computeWeight`` for an even ``Z`` (in blocks).

    The pinned source is::

        alpha = exp(-epsilon);  half = Z / 2
        weights[half] = (1 - alpha) / (1 + alpha - 2 * alpha ** (half + 1))
        multiplier = alpha
        for i in 1..half:
            weights[half + i] = weights[half - i] = weights[half + i - 1] * multiplier
            multiplier *= alpha

    The recurrence multiplies by ``alpha``, then ``alpha**2``, then ``alpha**3``, ... so
    the two-sided weight at distance ``i`` from the centre is
    ``weights[half] * alpha**(i * (i + 1) / 2)`` — a *linear-geometric* (triangular
    exponent) decay, **not** the textbook ``alpha**i`` geometric.  The pinned recurrence
    is reproduced verbatim, including the pinned normalising constant, and it is
    deliberately not "corrected" into a textbook geometric distribution.

    The returned vector is the pinned raw weight vector: it is normalised implicitly by
    the sampler (``std::discrete_distribution`` does the same upstream) and it is not
    renormalised here, because the sizing search consumes these raw weights.
    """
    capacity = _require_even_positive(bin_capacity_blocks, "bin_capacity_blocks")
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    alpha = math.exp(-epsilon)
    half = capacity // 2
    weights: List[float] = [0.0] * (capacity + 1)
    weights[half] = (1.0 - alpha) / (1.0 + alpha - 2.0 * _fast_power(alpha, half + 1))
    multiplier = alpha
    for index in range(1, half + 1):
        weight = weights[half + index - 1] * multiplier
        weights[half + index] = weight
        weights[half - index] = weight
        multiplier *= alpha
    return tuple(weights)


def _convolve(left: Sequence[float], right: Sequence[float]) -> Tuple[float, ...]:
    """Pinned ``Dist.hpp::Geom::multiply`` — a plain discrete convolution.

    The pinned implementation is the naive O(n*m) double loop (its FFT variant is
    commented out upstream), which is what is reproduced here.  Skipping zero terms is
    exact: adding ``0.0`` cannot change a sum.
    """
    if not left or not right:
        return ()
    result: List[float] = [0.0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        if left_value == 0.0:
            continue
        for j, right_value in enumerate(right):
            result[i + j] += left_value * right_value
    return tuple(result)


def geom_conv(
    bin_capacity_blocks: int,
    privacy_epsilon: float,
    bin_count: int,
) -> Tuple[float, ...]:
    """Pinned ``Dist.hpp::Geom::geomConv`` — the pmf of ``bin_count`` summed loads.

    A deterministic pure function: the ``bin_count``-fold convolution of the pinned
    weight vector, of length ``bin_capacity_blocks * bin_count + 1`` (the pinned ``assert``).
    """
    capacity = _require_even_positive(bin_capacity_blocks, "bin_capacity_blocks")
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    count = _require_plain_int(bin_count, "bin_count", minimum=1)
    geom = compute_weight(capacity, epsilon)
    current = geom
    for _ in range(1, count):
        current = _convolve(current, geom)
    if len(current) != capacity * count + 1:  # pragma: no cover - pinned invariant
        raise BlockAllocationError(
            "the convolutional pmf has length "
            f"{len(current)}, expected {capacity * count + 1}"
        )
    return current


# ---------------------------------------------------------------------------
# pinned bin-capacity sizing (block units, no AES byte alignment)
# ---------------------------------------------------------------------------


def compute_bin_capacity_blocks(
    *,
    security_lambda: int,
    privacy_epsilon: float,
    privacy_delta: float,
    bucket_capacity_blocks: int = ATOMIC_BUCKET_CAPACITY_BLOCKS,
) -> int:
    """Pinned ``DOMerger.hpp`` constructor sizing, in **blocks**.

    The pinned search is::

        upper = toEvenUB(floor(log(lambda)**5 / epsilon));  lower = 2
        while (lower + 2 < upper):
            Z    = toEvenUB((lower + upper) / 2)
            minB = ceil(2 * bucketCapacity / (Z * (1 - log(lambda)**-2)))
            pmf  = Geom::geomConv(Z, epsilon, minB)
            if sum(pmf[0 : bucketCapacity]) > delta: lower = Z + 2
            else:                                    upper = Z

    with ``bucket_capacity_blocks == 1`` (one block per bucket) the tail test is
    ``pmf[0] <= delta``, i.e. the convolution must assign at most ``delta`` weight to a
    total of fewer than one bucket of load.

    Two documented divergences from the pinned code (Decision 0004):

    1. the pinned ``upperBound(datumSize, binCapacity)`` AES/datum byte-alignment rounding
       is **not** ported — M2 plans abstract block items, not ciphertext byte addresses,
       so the byte-alignment step has no meaning here;
    2. the pinned member ``binCapacity`` is left *uninitialised* when the loop body never
       executes, and the loop can also exit holding a probe that failed the tail test.
       The first is an indeterminate read, which M2 does not emulate: the degenerate
       range is resolved explicitly as ``max(2, upper)``.  The second is preserved
       faithfully (the search keeps the pinned variable updates and returns the last
       probe), because the request was to preserve the convolutional sizing mathematics.
    """
    lam = _require_plain_int(security_lambda, "security_lambda", minimum=2)
    epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
    delta = _require_positive_real(privacy_delta, "privacy_delta")
    if not delta < 1.0:
        raise BlockAllocationError(f"privacy_delta must be < 1, got {delta}")
    bucket_capacity = _require_plain_int(
        bucket_capacity_blocks, "bucket_capacity_blocks", minimum=1
    )
    if bucket_capacity != ATOMIC_BUCKET_CAPACITY_BLOCKS:
        raise BlockAllocationError(
            "the SWAT-M-Block adaptation fixes one bucket to exactly "
            f"{ATOMIC_BUCKET_CAPACITY_BLOCKS} block (one allocation item), got "
            f"bucket_capacity_blocks = {bucket_capacity}"
        )
    decay = _decay(lam)
    if not decay > 0.0:
        raise BlockAllocationError(
            f"security_lambda = {lam} does not satisfy 1 - log(lambda)^-2 > 0"
        )

    upper = to_even_up(math.floor(_fast_power(math.log(lam), _UPPER_CANDIDATE_EXPONENT)
                                  / epsilon))
    lower = 2
    if lower + 2 >= upper:
        # the pinned loop body would never run, leaving the member uninitialised
        return max(lower, upper)

    capacity = upper
    while lower + 2 < upper:
        capacity = to_even_up((lower + upper) // 2)
        min_bins = math.ceil(2.0 * bucket_capacity / (capacity * decay))
        pmf = geom_conv(capacity, epsilon, min_bins)
        tail = math.fsum(pmf[:bucket_capacity])
        if tail > delta:
            lower = capacity + 2
        else:
            upper = capacity
    return capacity


# ---------------------------------------------------------------------------
# planner-owned, domain-separated streams
# ---------------------------------------------------------------------------


def derive_stream_seed(seed: int, domain: str) -> int:
    """A deterministic sub-seed for one RNG domain.

    Derived with SHA-256 rather than ``hash()`` so the value is stable across processes
    and platforms.  The point is that the load stream and the Laplace stream are
    *independent*: drawing from one can never perturb the other.
    """
    base = _require_plain_int(seed, "seed", minimum=0)
    if not isinstance(domain, str) or not domain:
        raise BlockAllocationError(f"domain must be a non-empty str, got {domain!r}")
    digest = hashlib.sha256(f"{base}|{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class GeometricLoadSampler:
    """Pinned ``Dist.hpp::Geom`` — a seeded discrete distribution over ``[0, Z]``.

    ``std::discrete_distribution`` normalises the pinned raw weights; this sampler draws
    with the same weights by scaling a uniform draw over the cumulative weight.  The RNG
    is owned by the sampler (planner-owned stream, no module-global state).
    """

    __slots__ = ("_bin_capacity_blocks", "_privacy_epsilon", "_weights", "_cumulative",
                 "_total", "_stream")

    def __init__(
        self,
        bin_capacity_blocks: int,
        privacy_epsilon: float,
        *,
        seed: int,
        domain: str = STREAM_DOMAIN_LOADS,
    ) -> None:
        self._bin_capacity_blocks = _require_even_positive(
            bin_capacity_blocks, "bin_capacity_blocks"
        )
        self._privacy_epsilon = _require_positive_real(privacy_epsilon, "privacy_epsilon")
        self._weights = compute_weight(self._bin_capacity_blocks, self._privacy_epsilon)
        cumulative: List[float] = []
        running = 0.0
        for weight in self._weights:
            running += weight
            cumulative.append(running)
        self._cumulative = tuple(cumulative)
        self._total = running
        if not self._total > 0.0:  # pragma: no cover - alpha < 1 keeps this positive
            raise BlockAllocationError("the pinned Geom weight vector sums to zero")
        self._stream = random.Random(derive_stream_seed(seed, domain))

    @property
    def bin_capacity_blocks(self) -> int:
        return self._bin_capacity_blocks

    @property
    def weights(self) -> Tuple[float, ...]:
        return self._weights

    def sample(self) -> int:
        """Draw one truncated-geometric load, in ``[0, bin_capacity_blocks]``."""
        draw = self._stream.random() * self._total
        index = bisect.bisect_right(self._cumulative, draw)
        if index >= len(self._weights):
            index = len(self._weights) - 1
        return index


class LaplaceSampler:
    """Pinned ``Dist.hpp::Laplace`` — the noise stream of ``DPPrefixSum``.

    The pinned sampler maps a uniform draw ``u`` to
    ``-b * (u > 0.5 ? 1 : -1) * log(1 - 2 * |u - 0.5|)``, i.e. ``b`` is the Laplace
    scale.  The enclosing code calls it with ``b = 1 / epsilon``.

    One documented divergence: the pinned formula evaluates ``log(0)`` when the uniform
    draw is exactly ``0.0`` (which ``uniform_real_distribution`` on ``[0, 1)`` may return),
    producing a non-finite sample that the pinned caller then casts to ``uint32_t`` —
    undefined behaviour.  M2 keeps the draw on the open interval by substituting the
    smallest value the stream can produce instead of emulating that corner.
    """

    __slots__ = ("_stream",)

    def __init__(self, *, seed: int, domain: str = STREAM_DOMAIN_LAPLACE) -> None:
        self._stream = random.Random(derive_stream_seed(seed, domain))

    def sample(self, scale: float) -> float:
        """Draw one Laplace variate with the given positive ``scale``."""
        magnitude = _require_positive_real(scale, "scale")
        draw = self._stream.random()
        if draw <= 0.0:
            draw = _SMALLEST_POSITIVE_UNIFORM
        sign = 1.0 if draw > 0.5 else -1.0
        return -magnitude * sign * math.log(1.0 - 2.0 * abs(draw - 0.5))


def pinned_uint32_cast(value: float) -> int:
    """The intended integer semantics of the pinned ``(uint32_t) curNoise`` cast.

    Inside the pinned well-defined range ``[0, 2**32)`` this is exactly C++'s truncation
    toward zero.  Outside it the pinned expression is undefined behaviour, which M2 does
    not emulate: a non-finite estimate is refused outright, and a value outside the
    unsigned range is clamped to it before truncation.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BlockAllocationError(
            f"the noisy prefix estimate must be real, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise BlockAllocationError(
            "the noisy prefix estimate is not finite; the pinned cast is undefined here"
        )
    if number <= 0.0:
        return 0
    if number >= _UINT32_MAX:
        return _UINT32_MAX
    return int(number)
