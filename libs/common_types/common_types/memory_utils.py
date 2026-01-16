"""
Memory Utilities Module for autoBot Data Pipelines.

Provides shared utilities for memory management across services:
- Memory usage tracking and logging
- Batch processing helpers
- Generator-based iteration patterns
- Garbage collection utilities

Example usage:
    >>> from common_types.memory_utils import (
    ...     get_memory_usage_mb,
    ...     iter_batches,
    ...     MemoryTracker,
    ...     BatchConfig,
    ... )
    >>>
    >>> # Track memory during processing
    >>> with MemoryTracker("my_operation") as tracker:
    ...     process_large_dataset()
    >>> print(f"Peak memory: {tracker.peak_mb:.1f} MB")
    >>>
    >>> # Process data in batches
    >>> for batch in iter_batches(large_list, batch_size=1000):
    ...     process_batch(batch)
"""

from __future__ import annotations

import gc
import logging
import sys
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

# Type variable for generic sequences
T = TypeVar("T")

# Module logger
logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in megabytes.

    Uses the resource module on Unix systems and falls back to
    sys.getsizeof for basic estimation on other platforms.

    Returns:
        Memory usage in megabytes. Returns 0.0 if unavailable.
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes to MB
        return usage.ru_maxrss / 1024  # KB to MB
    except (ImportError, AttributeError):
        return 0.0


def get_memory_usage_bytes() -> int:
    """Get current process memory usage in bytes.

    Returns:
        Memory usage in bytes. Returns 0 if unavailable.
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss  # Already in bytes
        return usage.ru_maxrss * 1024  # KB to bytes
    except (ImportError, AttributeError):
        return 0


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB", "256 MB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        batch_size: Number of items to process in each batch.
        enable_gc: Run garbage collection after each batch.
        log_progress: Log progress after each batch.
        log_memory: Log memory usage after each batch.
    """
    batch_size: int = 10_000
    enable_gc: bool = False
    log_progress: bool = False
    log_memory: bool = False


@dataclass
class MemoryStats:
    """Statistics for memory usage tracking.

    Attributes:
        start_memory_mb: Memory at start of tracking.
        peak_memory_mb: Peak memory during tracking.
        current_memory_mb: Current memory usage.
        items_processed: Number of items processed.
        batches_processed: Number of batches processed.
        start_time: When tracking started.
    """
    start_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    items_processed: int = 0
    batches_processed: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    def update(self) -> None:
        """Update current and peak memory measurements."""
        self.current_memory_mb = get_memory_usage_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, self.current_memory_mb)

    @property
    def memory_delta_mb(self) -> float:
        """Memory change since start."""
        return self.current_memory_mb - self.start_memory_mb

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "start_memory_mb": self.start_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.current_memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "items_processed": self.items_processed,
            "batches_processed": self.batches_processed,
        }


class MemoryTracker:
    """Context manager for tracking memory usage during operations.

    Example:
        >>> with MemoryTracker("data_processing") as tracker:
        ...     process_large_dataset()
        >>> print(f"Peak: {tracker.peak_mb:.1f} MB")

    Attributes:
        name: Name of the operation being tracked.
        stats: Memory statistics.
    """

    def __init__(
        self,
        name: str = "operation",
        log_on_exit: bool = True,
        gc_on_exit: bool = False,
    ) -> None:
        """Initialize memory tracker.

        Args:
            name: Name of the operation for logging.
            log_on_exit: Log stats when exiting context.
            gc_on_exit: Run garbage collection on exit.
        """
        self.name = name
        self.log_on_exit = log_on_exit
        self.gc_on_exit = gc_on_exit
        self.stats = MemoryStats()

    def __enter__(self) -> "MemoryTracker":
        """Enter context and record start memory."""
        self.stats = MemoryStats()
        self.stats.start_memory_mb = get_memory_usage_mb()
        self.stats.peak_memory_mb = self.stats.start_memory_mb
        self.stats.current_memory_mb = self.stats.start_memory_mb
        self.stats.start_time = datetime.now()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and optionally log stats."""
        self.stats.update()

        if self.gc_on_exit:
            gc.collect()
            self.stats.update()

        if self.log_on_exit:
            logger.info(
                f"Memory tracking for '{self.name}': "
                f"peak={self.stats.peak_memory_mb:.1f}MB, "
                f"delta={self.stats.memory_delta_mb:+.1f}MB"
            )

    @property
    def peak_mb(self) -> float:
        """Get peak memory in MB."""
        return self.stats.peak_memory_mb

    @property
    def current_mb(self) -> float:
        """Get current memory in MB."""
        self.stats.update()
        return self.stats.current_memory_mb

    def checkpoint(self, label: str = "") -> None:
        """Record a memory checkpoint with optional label.

        Args:
            label: Optional label for the checkpoint.
        """
        self.stats.update()
        logger.debug(
            f"Memory checkpoint [{self.name}] {label}: "
            f"{self.stats.current_memory_mb:.1f}MB "
            f"(peak: {self.stats.peak_memory_mb:.1f}MB)"
        )


def iter_batches(
    items: Sequence[T],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """Iterate over items in batches.

    Args:
        items: Sequence of items to batch.
        batch_size: Maximum items per batch.

    Yields:
        List of items for each batch.

    Example:
        >>> data = list(range(100))
        >>> for batch in iter_batches(data, batch_size=30):
        ...     print(f"Processing {len(batch)} items")
        Processing 30 items
        Processing 30 items
        Processing 30 items
        Processing 10 items
    """
    total = len(items)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield list(items[start:end])


def iter_batches_with_progress(
    items: Sequence[T],
    batch_size: int,
    config: BatchConfig | None = None,
) -> Generator[tuple[int, list[T]], None, None]:
    """Iterate over items in batches with progress info.

    Args:
        items: Sequence of items to batch.
        batch_size: Maximum items per batch.
        config: Optional batch configuration.

    Yields:
        Tuple of (batch_index, batch_items) for each batch.
    """
    config = config or BatchConfig(batch_size=batch_size)
    total = len(items)
    batch_idx = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = list(items[start:end])

        yield batch_idx, batch

        if config.log_progress:
            progress_pct = (end / total) * 100
            logger.info(
                f"Batch {batch_idx + 1}: processed {end}/{total} "
                f"({progress_pct:.1f}%)"
            )

        if config.log_memory:
            mem = get_memory_usage_mb()
            logger.debug(f"Memory after batch {batch_idx + 1}: {mem:.1f}MB")

        if config.enable_gc:
            gc.collect()

        batch_idx += 1


def process_in_batches(
    items: Sequence[T],
    process_fn: Any,  # Callable[[list[T]], Any]
    batch_size: int = 10_000,
    config: BatchConfig | None = None,
) -> Generator[Any, None, None]:
    """Process items in batches using a provided function.

    Args:
        items: Sequence of items to process.
        process_fn: Function to apply to each batch.
        batch_size: Maximum items per batch.
        config: Optional batch configuration.

    Yields:
        Result of process_fn for each batch.
    """
    config = config or BatchConfig(batch_size=batch_size)

    for batch_idx, batch in iter_batches_with_progress(
        items, batch_size, config
    ):
        result = process_fn(batch)
        yield result


@contextmanager
def gc_context(
    collect_before: bool = False,
    collect_after: bool = True,
) -> Generator[None, None, None]:
    """Context manager for garbage collection.

    Args:
        collect_before: Run GC before entering context.
        collect_after: Run GC after exiting context.

    Example:
        >>> with gc_context():
        ...     # Do memory-intensive work
        ...     pass  # GC runs after this block
    """
    if collect_before:
        gc.collect()

    try:
        yield
    finally:
        if collect_after:
            gc.collect()


def chunked_iterator(
    iterator: Iterator[T],
    chunk_size: int,
) -> Generator[list[T], None, None]:
    """Convert an iterator into chunks.

    Unlike iter_batches which requires a sequence, this works
    with any iterator and doesn't need to know the length upfront.

    Args:
        iterator: Iterator to chunk.
        chunk_size: Maximum items per chunk.

    Yields:
        List of items for each chunk.

    Example:
        >>> def generate_data():
        ...     for i in range(100):
        ...         yield i
        >>> for chunk in chunked_iterator(generate_data(), 30):
        ...     print(f"Got chunk of {len(chunk)} items")
    """
    chunk: list[T] = []

    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


class LimitedSizeList(list[T]):
    """A list that automatically drops oldest items when size limit is reached.

    Useful for keeping only the most recent N items in memory.

    Example:
        >>> lst = LimitedSizeList(max_size=3)
        >>> for i in range(5):
        ...     lst.append(i)
        >>> print(list(lst))
        [2, 3, 4]
    """

    def __init__(
        self,
        max_size: int,
        initial: Sequence[T] | None = None,
    ) -> None:
        """Initialize limited size list.

        Args:
            max_size: Maximum number of items to keep.
            initial: Optional initial items.
        """
        super().__init__()
        self.max_size = max_size
        if initial:
            self.extend(initial)

    def append(self, item: T) -> None:
        """Append item, dropping oldest if at capacity."""
        super().append(item)
        self._trim()

    def extend(self, items: Sequence[T]) -> None:  # type: ignore[override]
        """Extend list, dropping oldest if needed."""
        super().extend(items)
        self._trim()

    def _trim(self) -> None:
        """Remove oldest items if over capacity."""
        while len(self) > self.max_size:
            self.pop(0)


__all__ = [
    # Memory functions
    "get_memory_usage_mb",
    "get_memory_usage_bytes",
    "format_bytes",
    # Dataclasses
    "BatchConfig",
    "MemoryStats",
    # Classes
    "MemoryTracker",
    "LimitedSizeList",
    # Iterators and generators
    "iter_batches",
    "iter_batches_with_progress",
    "process_in_batches",
    "chunked_iterator",
    # Context managers
    "gc_context",
]
