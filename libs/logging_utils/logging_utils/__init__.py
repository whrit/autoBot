"""
Structured Logging Utilities for autoBot Trading Engine.

Provides comprehensive logging with:
- structlog with rich console output
- Progress tracking with rich progress bars
- Metrics logging with timing decorators
- Context binding for symbols, services, and task IDs
"""

from logging_utils.metrics import (
    CounterTracker,
    MemoryTracker,
    StatsEmitter,
    log_duration,
    log_memory,
    track_counter,
)
from logging_utils.progress import (
    BackfillProgress,
    MultiTaskProgress,
    ProgressTracker,
)
from logging_utils.setup import (
    LogConfig,
    bind_context,
    configure_logging,
    get_logger,
)

__all__ = [
    # Setup
    "LogConfig",
    "configure_logging",
    "get_logger",
    "bind_context",
    # Progress
    "ProgressTracker",
    "BackfillProgress",
    "MultiTaskProgress",
    # Metrics
    "log_duration",
    "log_memory",
    "track_counter",
    "CounterTracker",
    "MemoryTracker",
    "StatsEmitter",
]

__version__ = "0.1.0"
