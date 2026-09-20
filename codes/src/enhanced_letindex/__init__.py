"""EnhancedLETIndex: security-enhanced LETIndex research prototype.

G0 bootstrap: clean logical/physical separation plus a minimal vertical
slice (create block -> place -> read -> trace).  No security mechanisms are
implemented yet by design.
"""

from .block import Block
from .builder import LevelBuilder, LevelValidationError, validate_level
from .config import Config, ConfigError
from .dataset import Dataset, DuplicateKeyError
from .engine import (
    DuplicateLevelInPlanError,
    MergeResult,
    MultiLevelLookupResult,
    RetiredLevelError,
    SingleLevelLookupResult,
    TrustedEngine,
    UnknownLevelError,
)
from .identifiers import BlockId, LevelId, RecordKey, SlotId
from .level import Level
from .mapping import LogicalPhysicalMapping
from .merge import EmptyUpdateBatchError, LevelMerger, MergeInputError
from .pgm import (
    BatchPgmIndex,
    PgmConfigurationError,
    PgmError,
    PgmSegment,
    SearchResult,
)
from .record import Record
from .storage import UntrustedStorage
from .trace import TraceCollector, TraceEvent, TraceOperation

__version__ = "0.1.0"

__all__ = [
    "BatchPgmIndex",
    "Block",
    "BlockId",
    "Config",
    "ConfigError",
    "Dataset",
    "DuplicateKeyError",
    "DuplicateLevelInPlanError",
    "EmptyUpdateBatchError",
    "Level",
    "LevelBuilder",
    "LevelId",
    "LevelMerger",
    "LevelValidationError",
    "LogicalPhysicalMapping",
    "MergeInputError",
    "MergeResult",
    "MultiLevelLookupResult",
    "PgmConfigurationError",
    "PgmError",
    "PgmSegment",
    "Record",
    "RecordKey",
    "RetiredLevelError",
    "SearchResult",
    "SingleLevelLookupResult",
    "SlotId",
    "TraceCollector",
    "TraceEvent",
    "TraceOperation",
    "TrustedEngine",
    "UnknownLevelError",
    "UntrustedStorage",
    "validate_level",
]
