"""
Progress Tracking with Rich Progress Bars.

Provides:
- Rich progress bars for long-running operations
- Multi-task progress display
- ETA calculation
- Throughput metrics (records/sec, MB/sec)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Generator, Iterator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from logging_utils.setup import get_console


def _format_count(count: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_bytes(bytes_count: int) -> str:
    """Format bytes with KB/MB/GB suffixes."""
    if bytes_count >= 1_073_741_824:  # 1 GB
        return f"{bytes_count / 1_073_741_824:.1f} GB"
    elif bytes_count >= 1_048_576:  # 1 MB
        return f"{bytes_count / 1_048_576:.1f} MB"
    elif bytes_count >= 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count} B"


class ThroughputColumn(TextColumn):
    """Custom column showing throughput metrics."""

    def __init__(self, unit: str = "records") -> None:
        super().__init__("")
        self.unit = unit

    def render(self, task: Any) -> Text:
        """Render throughput information."""
        if task.finished:
            elapsed = task.finished_time - task.start_time if task.finished_time else 0
        else:
            elapsed = time.time() - task.start_time if task.start_time else 0

        if elapsed > 0 and task.completed > 0:
            rate = task.completed / elapsed
            if self.unit == "bytes":
                rate_str = f"{_format_bytes(int(rate))}/s"
            else:
                rate_str = f"{_format_count(int(rate))}/s"
            return Text(f"{rate_str}", style="cyan")
        return Text("--/s", style="dim")


class RecordCountColumn(TextColumn):
    """Custom column showing formatted record counts."""

    def __init__(self) -> None:
        super().__init__("")

    def render(self, task: Any) -> Text:
        """Render the record count."""
        count = task.fields.get("records", task.completed)
        return Text(f"{_format_count(count)} trades", style="magenta")


@dataclass
class ProgressTracker:
    """Track progress for a single operation with throughput metrics.

    Example:
        with ProgressTracker("Processing data", total=1000) as tracker:
            for item in items:
                process(item)
                tracker.advance()
    """

    description: str
    total: int | None = None
    unit: str = "records"
    show_speed: bool = True
    transient: bool = False

    _progress: Progress | None = field(default=None, init=False, repr=False)
    _task_id: TaskID | None = field(default=None, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _console: Console | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> ProgressTracker:
        self._console = get_console()
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
        ]

        if self.total:
            columns.append(MofNCompleteColumn())

        if self.show_speed:
            columns.append(ThroughputColumn(self.unit))

        columns.extend([
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ])

        self._progress = Progress(
            *columns,
            console=self._console,
            transient=self.transient,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(self.description, total=self.total)
        self._start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._progress:
            self._progress.stop()

    def advance(self, amount: int = 1) -> None:
        """Advance the progress by the given amount."""
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def update(
        self,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
        **fields: Any,
    ) -> None:
        """Update progress with new values."""
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=completed,
                total=total,
                description=description,
                **fields,
            )

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self._start_time

    @property
    def throughput(self) -> float:
        """Get current throughput (items per second)."""
        if self._progress and self._task_id is not None:
            task = self._progress.tasks[self._task_id]
            elapsed = self.elapsed
            if elapsed > 0:
                return task.completed / elapsed
        return 0.0


@dataclass
class TaskInfo:
    """Information about a task in multi-task progress."""

    task_id: TaskID
    description: str
    total: int
    completed: int = 0
    records: int = 0
    start_time: float = field(default_factory=time.time)


class MultiTaskProgress:
    """Track progress for multiple concurrent tasks.

    Example:
        with MultiTaskProgress() as progress:
            spy_task = progress.add_task("Backfilling SPY", total=90)
            qqq_task = progress.add_task("Backfilling QQQ", total=90)

            for day in range(90):
                # Process SPY
                records = fetch_spy_data(day)
                progress.update(spy_task, advance=1, records=len(records))
    """

    def __init__(
        self,
        title: str = "",
        console: Console | None = None,
        transient: bool = False,
    ) -> None:
        self._console = console or get_console()
        self._title = title
        self._transient = transient
        self._tasks: dict[TaskID, TaskInfo] = {}
        self._progress: Progress | None = None

    def __enter__(self) -> MultiTaskProgress:
        columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("[magenta]{task.fields[records_str]}"),
            TextColumn("[cyan]{task.fields[rate_str]}"),
            TimeRemainingColumn(),
        ]

        self._progress = Progress(
            *columns,
            console=self._console,
            transient=self._transient,
        )
        self._progress.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._progress:
            self._progress.stop()

    def add_task(self, description: str, total: int) -> TaskID:
        """Add a new task to track.

        Args:
            description: Task description (e.g., "Backfilling SPY").
            total: Total number of steps.

        Returns:
            Task ID for subsequent updates.
        """
        if self._progress is None:
            raise RuntimeError("Progress not started. Use context manager.")

        task_id = self._progress.add_task(
            description,
            total=total,
            records_str="0 trades",
            rate_str="--/s",
        )
        self._tasks[task_id] = TaskInfo(
            task_id=task_id,
            description=description,
            total=total,
        )
        return task_id

    def update(
        self,
        task_id: TaskID,
        advance: int = 0,
        completed: int | None = None,
        records: int | None = None,
        description: str | None = None,
    ) -> None:
        """Update task progress.

        Args:
            task_id: Task ID from add_task.
            advance: Steps to advance.
            completed: Set completed count directly.
            records: Total records processed (for display).
            description: Update task description.
        """
        if self._progress is None or task_id not in self._tasks:
            return

        task_info = self._tasks[task_id]

        if completed is not None:
            task_info.completed = completed
        else:
            task_info.completed += advance

        if records is not None:
            task_info.records = records

        # Calculate rate
        elapsed = time.time() - task_info.start_time
        rate_str = "--/s"
        if elapsed > 0 and task_info.records > 0:
            rate = task_info.records / elapsed
            rate_str = f"{_format_count(int(rate))}/s"

        self._progress.update(
            task_id,
            advance=advance if completed is None else 0,
            completed=completed,
            description=description,
            records_str=f"{_format_count(task_info.records)} trades",
            rate_str=rate_str,
        )

    def is_finished(self, task_id: TaskID) -> bool:
        """Check if a task is finished."""
        if task_id in self._tasks:
            info = self._tasks[task_id]
            return info.completed >= info.total
        return False

    def all_finished(self) -> bool:
        """Check if all tasks are finished."""
        return all(self.is_finished(tid) for tid in self._tasks)


@dataclass
class SymbolProgress:
    """Track backfill progress for a single symbol."""

    symbol: str
    total_days: int
    completed_days: int = 0
    total_records: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_days == 0:
            return 0.0
        return (self.completed_days / self.total_days) * 100

    @property
    def eta_seconds(self) -> float | None:
        """Estimate time remaining in seconds."""
        if self.completed_days == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.completed_days / elapsed
        remaining = self.total_days - self.completed_days
        return remaining / rate if rate > 0 else None


class BackfillProgress:
    """Specialized progress tracker for backfill operations.

    Provides per-symbol progress tracking with:
    - Days completed
    - Total records fetched
    - ETA calculation
    - Throughput metrics

    Example:
        with BackfillProgress(symbols=["SPY", "QQQ"], days=90) as progress:
            for symbol in symbols:
                for day in range(90):
                    records = fetch_data(symbol, day)
                    progress.update(symbol, records=len(records))
    """

    def __init__(
        self,
        symbols: list[str],
        days: int,
        console: Console | None = None,
    ) -> None:
        self._console = console or get_console()
        self._symbols = symbols
        self._days = days
        self._progress: dict[str, SymbolProgress] = {}
        self._multi_progress: MultiTaskProgress | None = None
        self._task_ids: dict[str, TaskID] = {}

    def __enter__(self) -> BackfillProgress:
        self._multi_progress = MultiTaskProgress(
            title="Backfill Progress",
            console=self._console,
        )
        self._multi_progress.__enter__()

        for symbol in self._symbols:
            self._progress[symbol] = SymbolProgress(
                symbol=symbol,
                total_days=self._days,
            )
            task_id = self._multi_progress.add_task(
                f"Backfilling {symbol}",
                total=self._days,
            )
            self._task_ids[symbol] = task_id

        return self

    def __exit__(self, *args: Any) -> None:
        if self._multi_progress:
            self._multi_progress.__exit__(*args)
        self._print_summary()

    def update(
        self,
        symbol: str,
        records: int = 0,
        advance_days: int = 1,
    ) -> None:
        """Update progress for a symbol.

        Args:
            symbol: Symbol being processed.
            records: Number of records fetched in this update.
            advance_days: Number of days completed.
        """
        if symbol not in self._progress:
            return

        prog = self._progress[symbol]
        prog.completed_days += advance_days
        prog.total_records += records

        if self._multi_progress and symbol in self._task_ids:
            self._multi_progress.update(
                self._task_ids[symbol],
                advance=advance_days,
                records=prog.total_records,
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all symbols."""
        total_records = sum(p.total_records for p in self._progress.values())
        total_days = sum(p.completed_days for p in self._progress.values())

        return {
            "symbols": len(self._symbols),
            "total_days_processed": total_days,
            "total_records": total_records,
            "per_symbol": {
                symbol: {
                    "days": prog.completed_days,
                    "records": prog.total_records,
                }
                for symbol, prog in self._progress.items()
            },
        }

    def _print_summary(self) -> None:
        """Print final summary table."""
        summary = self.get_summary()

        table = Table(title="Backfill Summary", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Days", justify="right")
        table.add_column("Records", justify="right", style="magenta")
        table.add_column("Rate", justify="right", style="green")

        for symbol, stats in summary["per_symbol"].items():
            prog = self._progress[symbol]
            elapsed = time.time() - prog.start_time
            rate = prog.total_records / elapsed if elapsed > 0 else 0

            table.add_row(
                symbol,
                str(stats["days"]),
                _format_count(stats["records"]),
                f"{_format_count(int(rate))}/s",
            )

        # Add totals row
        total_elapsed = max(
            time.time() - p.start_time for p in self._progress.values()
        ) if self._progress else 0
        total_rate = summary["total_records"] / total_elapsed if total_elapsed > 0 else 0

        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{summary['total_days_processed']}[/bold]",
            f"[bold]{_format_count(summary['total_records'])}[/bold]",
            f"[bold]{_format_count(int(total_rate))}/s[/bold]",
        )

        self._console.print()
        self._console.print(table)


@contextmanager
def track_progress(
    description: str,
    total: int | None = None,
    unit: str = "items",
) -> Generator[ProgressTracker, None, None]:
    """Context manager for simple progress tracking.

    Args:
        description: Progress bar description.
        total: Total items to process (None for indeterminate).
        unit: Unit name for throughput display.

    Yields:
        ProgressTracker instance.
    """
    tracker = ProgressTracker(
        description=description,
        total=total,
        unit=unit,
    )
    with tracker:
        yield tracker
