"""Configuration parameters for the EnhancedLETIndex engine."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

__all__ = ["Config", "ConfigError"]


class ConfigError(Exception):
    """Raised for invalid configuration."""


@dataclass(frozen=True)
class Config:
    """Engine configuration.

    G0-operational parameters:

    * ``block_capacity``: maximum number of records per block;
    * ``random_seed``: seed for the engine's deterministic RNG.

    G1-B1-operational parameter:

    * ``pgm_epsilon``: PGM error bound in record/item positions. ``None``
      means "PGM not explicitly configured" (building a PGM then fails
      loudly); any non-``None`` value must be a non-negative integer.
      ``0`` is legal.

    Reserved (stored, not yet operational):

    * ``lsm_level_ratio``: LSM level size ratio (future merge milestones).

    Security parameters (K/W/S/D/E) are intentionally absent in G0; they are
    added as explicit later layers rather than being guessed here.
    """

    block_capacity: int = 64
    random_seed: int = 0
    lsm_level_ratio: float = 10.0
    pgm_epsilon: Optional[int] = None

    def __post_init__(self) -> None:
        if self.block_capacity < 1:
            raise ConfigError(
                f"block_capacity must be >= 1, got {self.block_capacity}"
            )
        if self.pgm_epsilon is not None:
            # bool is an int subclass; reject it explicitly so that
            # `pgm_epsilon=True` is not silently accepted as 1.
            if isinstance(self.pgm_epsilon, bool):
                raise ConfigError(
                    f"pgm_epsilon must be an int (got bool {self.pgm_epsilon}); "
                    "None or a non-negative int is required"
                )
            if not isinstance(self.pgm_epsilon, int):
                raise ConfigError(
                    f"pgm_epsilon must be an int (got "
                    f"{type(self.pgm_epsilon).__name__} {self.pgm_epsilon!r})"
                )
            if self.pgm_epsilon < 0:
                raise ConfigError(
                    f"pgm_epsilon must be >= 0, got {self.pgm_epsilon}"
                )

    @property
    def rng(self) -> random.Random:
        """A deterministic RNG seeded from ``random_seed``.

        Currently unused for block placement (G0 allocation is sequential);
        provided for future randomized reshuffle/re-randomization milestones,
        which must derive all randomness from this seed for reproducibility.
        """
        return random.Random(self.random_seed)
