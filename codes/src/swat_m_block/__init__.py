"""SWAT-M-Block: the block-granular SWAT-style merge baseline.

Substrate: ``codes/src/enhanced_letindex/`` is a one-time, byte-identical, SHA-256
manifest-gated snapshot of the LETIndex common substrate taken from
``Russell-H-0/EnhancedLETIndex@4e68be75`` (see ``PROVENANCE.md``); it is frozen and must
not be edited here.

Milestones:

* **M0** (Issue #1, CLOSED) — repository / bootstrap freeze: the frozen substrate, the
  provenance gate and the SWAT-M-Block contract
  (``decisions/0001``, ``decisions/0002``).
* **M1** (Issue #2) — the **functional / logical oracle only**
  (:mod:`swat_m_block.functional_oracle`, ``decisions/0003``).

M1 exposes the exact logical merged result of a legal merge — the canonical output blocks
and the canonical PGM — by driving the frozen incremental logical merge.  It implements
**no** SWAT privacy mechanism and **no** physical I/O: no ``DOAllocate`` / ``DOMerge``, no
noisy or padded bin allocation, no dummy or cover I/O, no output blinding, no ``PRP``
writeback, no physical publication, no de-amortisation, no attack and no performance
experiment.  None of that may be added before milestone M2 is explicitly authorised.
"""

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
    "DEFAULT_SCHEDULE_POLICY",
    "FunctionalMergeOracleResult",
    "FunctionalOracleError",
    "collect_output_blocks",
    "flatten_blocks",
    "functional_merge_oracle",
    "output_key_stream",
]
