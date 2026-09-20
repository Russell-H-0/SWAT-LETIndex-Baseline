"""SWAT-M-Block: the block-granular SWAT-style merge baseline (M0: package marker only).

This package is the future home of the SWAT-M-Block merge adaptation frozen in
``decisions/0002-swat-m-block-contract.md``.

M0 deliberately implements **nothing** here.  There is no noisy/padded bin allocator,
no differential-oblivious merge, no ``DOAllocate`` / ``DOMerge`` port, no dummy or
cover block I/O, no oblivious output shuffle and no de-amortisation in this package,
and none of it may be added before milestone M1 is explicitly authorised.

The frozen common LETIndex substrate lives in the sibling package
``enhanced_letindex`` (one-time source snapshot; see ``PROVENANCE.md``).
"""

__all__: list[str] = []
