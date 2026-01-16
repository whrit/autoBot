"""Tests for progress tracking module."""

import time

import pytest

from logging_utils.progress import (
    BackfillProgress,
    MultiTaskProgress,
    ProgressTracker,
    SymbolProgress,
    _format_bytes,
    _format_count,
    track_progress,
)


class TestFormatCount:
    """Tests for _format_count helper function."""

    def test_small_numbers(self) -> None:
        """Test formatting small numbers."""
        assert _format_count(0) == "0"
        assert _format_count(1) == "1"
        assert _format_count(999) == "999"

    def test_thousands(self) -> None:
        """Test formatting thousands."""
        assert _format_count(1000) == "1.0K"
        assert _format_count(1500) == "1.5K"
        assert _format_count(999999) == "1000.0K"

    def test_millions(self) -> None:
        """Test formatting millions."""
        assert _format_count(1_000_000) == "1.0M"
        assert _format_count(1_500_000) == "1.5M"
        assert _format_count(999_999_999) == "1000.0M"

    def test_billions(self) -> None:
        """Test formatting billions."""
        assert _format_count(1_000_000_000) == "1.0B"
        assert _format_count(2_500_000_000) == "2.5B"


class TestFormatBytes:
    """Tests for _format_bytes helper function."""

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert _format_bytes(0) == "0 B"
        assert _format_bytes(512) == "512 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(1536) == "1.5 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert _format_bytes(1_048_576) == "1.0 MB"
        assert _format_bytes(1_572_864) == "1.5 MB"

    def test_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert _format_bytes(1_073_741_824) == "1.0 GB"
        assert _format_bytes(2_147_483_648) == "2.0 GB"


class TestSymbolProgress:
    """Tests for SymbolProgress dataclass."""

    def test_initialization(self) -> None:
        """Test SymbolProgress initialization."""
        prog = SymbolProgress(symbol="SPY", total_days=90)
        assert prog.symbol == "SPY"
        assert prog.total_days == 90
        assert prog.completed_days == 0
        assert prog.total_records == 0

    def test_progress_percentage(self) -> None:
        """Test progress percentage calculation."""
        prog = SymbolProgress(symbol="SPY", total_days=100)
        prog.completed_days = 50
        assert prog.progress_pct == 50.0

    def test_progress_percentage_zero_days(self) -> None:
        """Test progress percentage with zero total days."""
        prog = SymbolProgress(symbol="SPY", total_days=0)
        assert prog.progress_pct == 0.0

    def test_eta_calculation(self) -> None:
        """Test ETA calculation."""
        prog = SymbolProgress(symbol="SPY", total_days=100)
        prog.start_time = time.time() - 10  # 10 seconds ago
        prog.completed_days = 10  # 1 day per second
        eta = prog.eta_seconds
        assert eta is not None
        assert 85 <= eta <= 95  # Should be around 90 seconds

    def test_eta_with_no_progress(self) -> None:
        """Test ETA with no progress."""
        prog = SymbolProgress(symbol="SPY", total_days=100)
        assert prog.eta_seconds is None


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_context_manager(self) -> None:
        """Test ProgressTracker as context manager."""
        with ProgressTracker("Test", total=10) as tracker:
            assert tracker is not None
            tracker.advance(5)

    def test_advance(self) -> None:
        """Test advancing progress."""
        with ProgressTracker("Test", total=10) as tracker:
            tracker.advance()
            tracker.advance(3)

    def test_update(self) -> None:
        """Test updating progress."""
        with ProgressTracker("Test", total=10) as tracker:
            tracker.update(completed=5)
            tracker.update(description="Updated")

    def test_elapsed_time(self) -> None:
        """Test elapsed time tracking."""
        with ProgressTracker("Test", total=10) as tracker:
            time.sleep(0.1)
            assert tracker.elapsed >= 0.1

    def test_throughput(self) -> None:
        """Test throughput calculation."""
        with ProgressTracker("Test", total=100) as tracker:
            tracker.advance(50)
            time.sleep(0.1)
            # Should have some throughput
            assert tracker.throughput >= 0


class TestMultiTaskProgress:
    """Tests for MultiTaskProgress class."""

    def test_context_manager(self) -> None:
        """Test MultiTaskProgress as context manager."""
        with MultiTaskProgress() as progress:
            assert progress is not None

    def test_add_task(self) -> None:
        """Test adding tasks."""
        with MultiTaskProgress() as progress:
            task_id = progress.add_task("Task 1", total=100)
            assert task_id is not None

    def test_add_multiple_tasks(self) -> None:
        """Test adding multiple tasks."""
        with MultiTaskProgress() as progress:
            task1 = progress.add_task("Task 1", total=100)
            task2 = progress.add_task("Task 2", total=50)
            assert task1 != task2

    def test_update_task(self) -> None:
        """Test updating task progress."""
        with MultiTaskProgress() as progress:
            task_id = progress.add_task("Task", total=100)
            progress.update(task_id, advance=10)
            progress.update(task_id, records=1000)

    def test_is_finished(self) -> None:
        """Test task completion check."""
        with MultiTaskProgress() as progress:
            task_id = progress.add_task("Task", total=10)
            assert not progress.is_finished(task_id)
            progress.update(task_id, completed=10)
            assert progress.is_finished(task_id)

    def test_all_finished(self) -> None:
        """Test all tasks completion check."""
        with MultiTaskProgress() as progress:
            task1 = progress.add_task("Task 1", total=10)
            task2 = progress.add_task("Task 2", total=10)
            assert not progress.all_finished()
            progress.update(task1, completed=10)
            assert not progress.all_finished()
            progress.update(task2, completed=10)
            assert progress.all_finished()

    def test_add_task_without_context(self) -> None:
        """Test adding task without context raises error."""
        progress = MultiTaskProgress()
        with pytest.raises(RuntimeError):
            progress.add_task("Task", total=10)


class TestBackfillProgress:
    """Tests for BackfillProgress class."""

    def test_context_manager(self) -> None:
        """Test BackfillProgress as context manager."""
        with BackfillProgress(symbols=["SPY"], days=10) as progress:
            assert progress is not None

    def test_single_symbol(self) -> None:
        """Test progress with single symbol."""
        with BackfillProgress(symbols=["SPY"], days=10) as progress:
            progress.update("SPY", records=100)

    def test_multiple_symbols(self) -> None:
        """Test progress with multiple symbols."""
        with BackfillProgress(symbols=["SPY", "QQQ"], days=10) as progress:
            progress.update("SPY", records=100)
            progress.update("QQQ", records=50)

    def test_update_with_days(self) -> None:
        """Test updating with day count."""
        with BackfillProgress(symbols=["SPY"], days=10) as progress:
            progress.update("SPY", records=100, advance_days=2)

    def test_get_summary(self) -> None:
        """Test getting summary statistics."""
        with BackfillProgress(symbols=["SPY", "QQQ"], days=10) as progress:
            progress.update("SPY", records=100, advance_days=5)
            progress.update("QQQ", records=50, advance_days=3)
            summary = progress.get_summary()

            assert summary["symbols"] == 2
            assert summary["total_days_processed"] == 8
            assert summary["total_records"] == 150
            assert "SPY" in summary["per_symbol"]
            assert "QQQ" in summary["per_symbol"]

    def test_update_unknown_symbol(self) -> None:
        """Test updating unknown symbol is ignored."""
        with BackfillProgress(symbols=["SPY"], days=10) as progress:
            # Should not raise
            progress.update("UNKNOWN", records=100)


class TestTrackProgressContextManager:
    """Tests for track_progress context manager."""

    def test_basic_usage(self) -> None:
        """Test basic usage of track_progress."""
        with track_progress("Processing", total=100) as tracker:
            for _ in range(10):
                tracker.advance()

    def test_indeterminate_progress(self) -> None:
        """Test indeterminate progress (no total)."""
        with track_progress("Processing") as tracker:
            tracker.advance()

    def test_custom_unit(self) -> None:
        """Test custom unit."""
        with track_progress("Processing", total=100, unit="bytes") as tracker:
            tracker.advance(50)
