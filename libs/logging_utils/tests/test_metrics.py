"""Tests for metrics logging module."""

import asyncio
import time

import pytest

from logging_utils.metrics import (
    CounterTracker,
    MemoryTracker,
    StatsEmitter,
    TimingStats,
    log_duration,
    log_memory,
    track_counter,
)


class TestLogDuration:
    """Tests for log_duration decorator."""

    def test_sync_function(self) -> None:
        """Test decorating synchronous function."""

        @log_duration()
        def slow_function() -> str:
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_sync_function_with_args(self) -> None:
        """Test decorating function with arguments."""

        @log_duration(include_args=True)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_sync_function_exception(self) -> None:
        """Test decorator handles exceptions."""

        @log_duration()
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        """Test decorating async function."""

        @log_duration()
        async def async_function() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await async_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_async_function_exception(self) -> None:
        """Test decorator handles async exceptions."""

        @log_duration()
        async def failing_async() -> None:
            raise ValueError("Async error")

        with pytest.raises(ValueError, match="Async error"):
            await failing_async()

    def test_custom_message(self) -> None:
        """Test decorator with custom message."""

        @log_duration(message="Custom operation complete")
        def operation() -> str:
            return "result"

        result = operation()
        assert result == "result"

    def test_debug_level(self) -> None:
        """Test decorator with debug level."""

        @log_duration(level="debug")
        def debug_operation() -> int:
            return 42

        result = debug_operation()
        assert result == 42


class TestLogMemory:
    """Tests for log_memory context manager."""

    def test_basic_usage(self) -> None:
        """Test basic memory tracking."""
        with log_memory("test_operation") as stats:
            _ = [i for i in range(1000)]

        assert "initial_mb" in stats
        assert "final_mb" in stats
        assert "delta_mb" in stats

    def test_with_gc(self) -> None:
        """Test memory tracking with garbage collection."""
        with log_memory("gc_operation", log_gc=True) as stats:
            _ = [i for i in range(1000)]

        assert "initial_mb" in stats
        assert "final_mb" in stats

    def test_stats_populated(self) -> None:
        """Test that stats dict is populated."""
        with log_memory("test") as stats:
            pass

        assert isinstance(stats["initial_mb"], float)
        assert isinstance(stats["final_mb"], float)
        assert isinstance(stats["delta_mb"], float)


class TestCounterTracker:
    """Tests for CounterTracker class."""

    def test_initialization(self) -> None:
        """Test counter initialization."""
        counter = CounterTracker(name="test")
        assert counter.value == 0

    def test_initial_value(self) -> None:
        """Test counter with initial value."""
        counter = CounterTracker(name="test", initial_value=100)
        assert counter.value == 100

    def test_increment(self) -> None:
        """Test incrementing counter."""
        counter = CounterTracker(name="test")
        counter.increment()
        assert counter.value == 1

    def test_increment_by_amount(self) -> None:
        """Test incrementing by specific amount."""
        counter = CounterTracker(name="test")
        counter.increment(10)
        assert counter.value == 10

    def test_decrement(self) -> None:
        """Test decrementing counter."""
        counter = CounterTracker(name="test", initial_value=10)
        counter.decrement()
        assert counter.value == 9

    def test_decrement_by_amount(self) -> None:
        """Test decrementing by specific amount."""
        counter = CounterTracker(name="test", initial_value=10)
        counter.decrement(5)
        assert counter.value == 5

    def test_set_value(self) -> None:
        """Test setting counter value."""
        counter = CounterTracker(name="test")
        counter.set(42)
        assert counter.value == 42

    def test_rate_calculation(self) -> None:
        """Test rate calculation."""
        counter = CounterTracker(name="test")
        counter.increment(100)
        time.sleep(0.1)
        rate = counter.rate
        assert rate > 0

    def test_log_final(self) -> None:
        """Test log_final method."""
        counter = CounterTracker(name="test")
        counter.increment(100)
        # Should not raise
        counter.log_final()

    def test_log_interval(self) -> None:
        """Test logging at intervals."""
        counter = CounterTracker(name="test", log_interval=10)
        for _ in range(25):
            counter.increment()
        assert counter.value == 25


class TestTrackCounter:
    """Tests for track_counter convenience function."""

    def test_creates_counter(self) -> None:
        """Test that track_counter creates CounterTracker."""
        counter = track_counter("test")
        assert isinstance(counter, CounterTracker)
        assert counter.name == "test"

    def test_with_log_interval(self) -> None:
        """Test with log interval."""
        counter = track_counter("test", log_interval=10)
        assert counter.log_interval == 10


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_initialization(self) -> None:
        """Test memory tracker initialization."""
        tracker = MemoryTracker(name="test")
        assert tracker.name == "test"

    def test_checkpoint(self) -> None:
        """Test creating checkpoints."""
        tracker = MemoryTracker(name="test")
        memory = tracker.checkpoint("first")
        assert isinstance(memory, float)
        assert memory >= 0

    def test_multiple_checkpoints(self) -> None:
        """Test multiple checkpoints."""
        tracker = MemoryTracker(name="test")
        tracker.checkpoint("step1")
        _ = [i for i in range(10000)]
        tracker.checkpoint("step2")
        tracker.checkpoint("step3")

    def test_log_summary(self) -> None:
        """Test logging summary."""
        tracker = MemoryTracker(name="test")
        tracker.checkpoint("step1")
        tracker.checkpoint("step2")
        # Should not raise
        tracker.log_summary()


class TestStatsEmitter:
    """Tests for StatsEmitter class."""

    def test_initialization(self) -> None:
        """Test emitter initialization."""
        emitter = StatsEmitter(interval_seconds=60.0)
        assert emitter.interval == 60.0

    def test_register_provider(self) -> None:
        """Test registering stat provider."""
        emitter = StatsEmitter()
        emitter.register("test", lambda: 42)

    def test_unregister_provider(self) -> None:
        """Test unregistering stat provider."""
        emitter = StatsEmitter()
        emitter.register("test", lambda: 42)
        emitter.unregister("test")

    def test_emit_now(self) -> None:
        """Test immediate emission."""
        emitter = StatsEmitter()
        emitter.register("test_value", lambda: 123)
        emitter.emit_now()
        history = emitter.get_history()
        assert len(history) == 1
        assert history[0]["test_value"] == 123

    def test_start_stop(self) -> None:
        """Test starting and stopping emitter."""
        emitter = StatsEmitter(interval_seconds=0.1)
        emitter.register("value", lambda: 1)
        emitter.start()
        time.sleep(0.05)
        emitter.stop()

    def test_history(self) -> None:
        """Test getting emission history."""
        emitter = StatsEmitter()
        emitter.register("counter", lambda: 10)
        emitter.emit_now()
        emitter.emit_now()
        history = emitter.get_history()
        assert len(history) == 2

    def test_error_handling(self) -> None:
        """Test that errors in providers are handled."""
        emitter = StatsEmitter()
        emitter.register("failing", lambda: 1 / 0)
        emitter.emit_now()
        history = emitter.get_history()
        assert "ERROR" in str(history[0]["failing"])


class TestTimingStats:
    """Tests for TimingStats class."""

    def test_initialization(self) -> None:
        """Test timing stats initialization."""
        stats = TimingStats(name="test")
        assert stats.name == "test"
        assert stats.count == 0

    def test_record_timing(self) -> None:
        """Test recording timing."""
        stats = TimingStats(name="test")
        stats.record(10.5)
        stats.record(20.5)
        assert stats.count == 2

    def test_measure_context_manager(self) -> None:
        """Test measure context manager."""
        stats = TimingStats(name="test")
        with stats.measure():
            time.sleep(0.01)
        assert stats.count == 1
        assert stats.total_ms >= 10

    def test_total_ms(self) -> None:
        """Test total milliseconds."""
        stats = TimingStats(name="test")
        stats.record(10.0)
        stats.record(20.0)
        assert stats.total_ms == 30.0

    def test_mean_ms(self) -> None:
        """Test mean calculation."""
        stats = TimingStats(name="test")
        stats.record(10.0)
        stats.record(20.0)
        stats.record(30.0)
        assert stats.mean_ms == 20.0

    def test_mean_ms_empty(self) -> None:
        """Test mean with no data."""
        stats = TimingStats(name="test")
        assert stats.mean_ms == 0.0

    def test_min_max(self) -> None:
        """Test min and max."""
        stats = TimingStats(name="test")
        stats.record(5.0)
        stats.record(15.0)
        stats.record(10.0)
        assert stats.min_ms == 5.0
        assert stats.max_ms == 15.0

    def test_min_max_empty(self) -> None:
        """Test min/max with no data."""
        stats = TimingStats(name="test")
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        stats = TimingStats(name="test")
        for i in range(100):
            stats.record(float(i))
        assert stats.p50_ms == 50.0
        assert stats.p95_ms >= 94.0
        assert stats.p99_ms >= 98.0

    def test_percentiles_empty(self) -> None:
        """Test percentiles with no data."""
        stats = TimingStats(name="test")
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0

    def test_summary(self) -> None:
        """Test summary dictionary."""
        stats = TimingStats(name="test")
        stats.record(10.0)
        stats.record(20.0)
        summary = stats.summary()
        assert "count" in summary
        assert "total_ms" in summary
        assert "mean_ms" in summary
        assert "min_ms" in summary
        assert "max_ms" in summary
        assert "p50_ms" in summary
        assert "p95_ms" in summary
        assert "p99_ms" in summary

    def test_log_summary(self) -> None:
        """Test logging summary."""
        stats = TimingStats(name="test")
        stats.record(10.0)
        # Should not raise
        stats.log_summary()
