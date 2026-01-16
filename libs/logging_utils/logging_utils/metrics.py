"""
Metrics Logging Module.

Provides:
- Timing decorators (@log_duration)
- Memory usage tracking
- Counter tracking
- Periodic stats emission
"""

from __future__ import annotations

import asyncio
import functools
import gc
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generator, ParamSpec, TypeVar

import structlog

from logging_utils.setup import get_logger


P = ParamSpec("P")
R = TypeVar("R")


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux
        return usage.ru_maxrss / 1024
    except ImportError:
        # Fallback for non-Unix systems
        return 0.0


def _get_object_size_mb(obj: Any) -> float:
    """Get approximate size of an object in MB."""
    return sys.getsizeof(obj) / (1024 * 1024)


def log_duration(
    logger: structlog.stdlib.BoundLogger | None = None,
    level: str = "info",
    message: str | None = None,
    include_args: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function execution duration.

    Args:
        logger: Logger to use. If None, creates one from function module.
        level: Log level (debug, info, warning, error).
        message: Custom log message. Defaults to "Function {name} completed".
        include_args: Whether to include function arguments in log.

    Returns:
        Decorated function.

    Example:
        @log_duration()
        def process_data(data: list) -> None:
            ...

        @log_duration(level="debug", include_args=True)
        async def fetch_data(symbol: str) -> Data:
            ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_async = asyncio.iscoroutinefunction(func)
        log_msg = message or f"Function {func.__name__} completed"

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            log = logger or get_logger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                log_kwargs: dict[str, Any] = {
                    "function": func.__name__,
                    "duration_ms": round(duration * 1000, 2),
                }

                if include_args:
                    log_kwargs["args"] = str(args)[:100]
                    log_kwargs["kwargs"] = str(kwargs)[:100]

                getattr(log, level)(log_msg, **log_kwargs)
                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                log.error(
                    f"Function {func.__name__} failed",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            log = logger or get_logger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                log_kwargs: dict[str, Any] = {
                    "function": func.__name__,
                    "duration_ms": round(duration * 1000, 2),
                }

                if include_args:
                    log_kwargs["args"] = str(args)[:100]
                    log_kwargs["kwargs"] = str(kwargs)[:100]

                getattr(log, level)(log_msg, **log_kwargs)
                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                log.error(
                    f"Function {func.__name__} failed",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return async_wrapper if is_async else sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def log_memory(
    operation: str,
    logger: structlog.stdlib.BoundLogger | None = None,
    log_gc: bool = False,
) -> Generator[dict[str, float], None, None]:
    """Context manager to track memory usage during an operation.

    Args:
        operation: Name of the operation being tracked.
        logger: Logger to use.
        log_gc: Whether to run garbage collection and log object counts.

    Yields:
        Dict that will be populated with memory stats.

    Example:
        with log_memory("load_dataset") as stats:
            df = pd.read_parquet("data.parquet")
        print(f"Memory delta: {stats['delta_mb']:.2f} MB")
    """
    log = logger or get_logger(__name__)
    stats: dict[str, float] = {}

    # Get initial memory
    if log_gc:
        gc.collect()

    initial_memory = _get_memory_usage_mb()
    stats["initial_mb"] = initial_memory

    try:
        yield stats
    finally:
        # Get final memory
        if log_gc:
            gc.collect()

        final_memory = _get_memory_usage_mb()
        delta = final_memory - initial_memory

        stats["final_mb"] = final_memory
        stats["delta_mb"] = delta

        log.info(
            f"Memory usage for {operation}",
            operation=operation,
            initial_mb=round(initial_memory, 2),
            final_mb=round(final_memory, 2),
            delta_mb=round(delta, 2),
        )


@dataclass
class CounterTracker:
    """Track and log counter values.

    Example:
        counter = CounterTracker("records_processed")
        for record in records:
            process(record)
            counter.increment()
        counter.log_final()
    """

    name: str
    initial_value: int = 0
    log_interval: int = 0  # 0 = don't log until final
    logger: structlog.stdlib.BoundLogger | None = None

    _value: int = field(default=0, init=False)
    _last_log_value: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._value = self.initial_value
        self._last_log_value = self.initial_value

    def increment(self, amount: int = 1) -> int:
        """Increment the counter.

        Args:
            amount: Amount to increment by.

        Returns:
            New counter value.
        """
        self._value += amount

        # Log at interval if configured
        if self.log_interval > 0:
            if self._value - self._last_log_value >= self.log_interval:
                self._log_progress()
                self._last_log_value = self._value

        return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement the counter."""
        self._value -= amount
        return self._value

    def set(self, value: int) -> None:
        """Set the counter to a specific value."""
        self._value = value

    @property
    def value(self) -> int:
        """Get current counter value."""
        return self._value

    @property
    def rate(self) -> float:
        """Get rate (increments per second)."""
        elapsed = time.time() - self._start_time
        return self._value / elapsed if elapsed > 0 else 0.0

    def _log_progress(self) -> None:
        """Log current progress."""
        log = self.logger or get_logger(__name__)
        log.info(
            f"Counter progress: {self.name}",
            counter=self.name,
            value=self._value,
            rate=round(self.rate, 2),
        )

    def log_final(self) -> None:
        """Log final counter value and stats."""
        log = self.logger or get_logger(__name__)
        elapsed = time.time() - self._start_time

        log.info(
            f"Counter final: {self.name}",
            counter=self.name,
            final_value=self._value,
            elapsed_seconds=round(elapsed, 2),
            rate_per_second=round(self.rate, 2),
        )


def track_counter(name: str, log_interval: int = 0) -> CounterTracker:
    """Create a counter tracker.

    Args:
        name: Counter name.
        log_interval: Log progress every N increments (0 = final only).

    Returns:
        CounterTracker instance.
    """
    return CounterTracker(name=name, log_interval=log_interval)


@dataclass
class MemoryTracker:
    """Track memory usage over time.

    Example:
        tracker = MemoryTracker("data_loading")
        tracker.checkpoint("loaded_raw")
        process_data()
        tracker.checkpoint("processed")
        tracker.log_summary()
    """

    name: str
    logger: structlog.stdlib.BoundLogger | None = None

    _checkpoints: list[tuple[str, float, float]] = field(
        default_factory=list, init=False
    )
    _start_memory: float = field(default=0.0, init=False)
    _start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._start_memory = _get_memory_usage_mb()
        self._start_time = time.time()
        self._checkpoints = [("start", self._start_memory, 0.0)]

    def checkpoint(self, name: str) -> float:
        """Record a memory checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Current memory usage in MB.
        """
        current = _get_memory_usage_mb()
        elapsed = time.time() - self._start_time
        self._checkpoints.append((name, current, elapsed))
        return current

    def log_summary(self) -> None:
        """Log summary of all checkpoints."""
        log = self.logger or get_logger(__name__)

        # Build checkpoint deltas
        checkpoints_info = []
        for i, (name, memory, elapsed) in enumerate(self._checkpoints):
            if i > 0:
                prev_memory = self._checkpoints[i - 1][1]
                delta = memory - prev_memory
            else:
                delta = 0.0

            checkpoints_info.append({
                "name": name,
                "memory_mb": round(memory, 2),
                "delta_mb": round(delta, 2),
                "elapsed_s": round(elapsed, 2),
            })

        total_delta = self._checkpoints[-1][1] - self._start_memory

        log.info(
            f"Memory tracking summary: {self.name}",
            tracker=self.name,
            total_delta_mb=round(total_delta, 2),
            checkpoints=checkpoints_info,
        )


class StatsEmitter:
    """Periodically emit statistics.

    Collects stats from registered providers and logs them at intervals.

    Example:
        emitter = StatsEmitter(interval_seconds=60)
        emitter.register("trades", lambda: trade_counter.value)
        emitter.register("memory", _get_memory_usage_mb)
        emitter.start()
        # ... do work ...
        emitter.stop()
    """

    def __init__(
        self,
        interval_seconds: float = 60.0,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        self.interval = interval_seconds
        self.logger = logger
        self._providers: dict[str, Callable[[], Any]] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._history: list[dict[str, Any]] = []

    def register(self, name: str, provider: Callable[[], Any]) -> None:
        """Register a stats provider.

        Args:
            name: Stat name.
            provider: Callable that returns the stat value.
        """
        self._providers[name] = provider

    def unregister(self, name: str) -> None:
        """Unregister a stats provider."""
        self._providers.pop(name, None)

    def _emit_stats(self) -> None:
        """Emit current stats."""
        log = self.logger or get_logger(__name__)
        stats: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for name, provider in self._providers.items():
            try:
                stats[name] = provider()
            except Exception as e:
                stats[name] = f"ERROR: {e}"

        self._history.append(stats)
        log.info("Periodic stats", **stats)

    def _run_loop(self) -> None:
        """Background thread loop."""
        while not self._stop_event.wait(self.interval):
            if self._running:
                self._emit_stats()

    def start(self) -> None:
        """Start periodic stats emission."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic stats emission."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_history(self) -> list[dict[str, Any]]:
        """Get historical stats."""
        return self._history.copy()

    def emit_now(self) -> None:
        """Emit stats immediately."""
        self._emit_stats()


@dataclass
class TimingStats:
    """Collect timing statistics for operations."""

    name: str
    _times: list[float] = field(default_factory=list, init=False)

    def record(self, duration_ms: float) -> None:
        """Record a timing measurement."""
        self._times.append(duration_ms)

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager to measure and record timing."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000
            self.record(duration)

    @property
    def count(self) -> int:
        """Number of measurements."""
        return len(self._times)

    @property
    def total_ms(self) -> float:
        """Total time in milliseconds."""
        return sum(self._times)

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self._times) if self._times else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self._times) if self._times else 0.0

    @property
    def p50_ms(self) -> float:
        """50th percentile (median) in milliseconds."""
        if not self._times:
            return 0.0
        sorted_times = sorted(self._times)
        idx = len(sorted_times) // 2
        return sorted_times[idx]

    @property
    def p95_ms(self) -> float:
        """95th percentile in milliseconds."""
        if not self._times:
            return 0.0
        sorted_times = sorted(self._times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_ms(self) -> float:
        """99th percentile in milliseconds."""
        if not self._times:
            return 0.0
        sorted_times = sorted(self._times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def summary(self) -> dict[str, float]:
        """Get summary statistics."""
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }

    def log_summary(self, logger: structlog.stdlib.BoundLogger | None = None) -> None:
        """Log timing summary."""
        log = logger or get_logger(__name__)
        log.info(f"Timing stats: {self.name}", **self.summary())
