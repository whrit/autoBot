"""
Logging Configuration - Structlog and Rich console setup.

Thin wrapper around logging_utils for backward compatibility.
Provides centralized logging configuration with structured logging
and beautiful console output.

Supports file-friendly mode for background processes where Rich
Live displays don't work (stdout redirected to file).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import TaskID

from logging_utils import (
    LogConfig,
    configure_logging as configure_logging_utils,
    get_logger as get_logger_utils,
)
from logging_utils.setup import TRADING_THEME


def is_tty() -> bool:
    """Check if stdout is connected to a TTY (interactive terminal).

    Returns:
        True if running in an interactive terminal, False if redirected to file.
    """
    return sys.stdout.isatty()


def is_file_mode() -> bool:
    """Check if we should use file-friendly output (no cursor movement).

    File mode is enabled when:
    - stdout is not a TTY (redirected to file)
    - INGESTOR_FILE_MODE=1 environment variable is set

    Returns:
        True if file-friendly mode should be used.
    """
    return not is_tty() or os.environ.get("INGESTOR_FILE_MODE", "0") == "1"


# Re-export console for backward compatibility
# Use force_terminal when FORCE_COLOR is set to enable colors in file output
_force_terminal = os.environ.get("FORCE_COLOR", "0") == "1"
console = Console(theme=TRADING_THEME, stderr=True, force_terminal=_force_terminal)


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with Rich console output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    configure_logging_utils(LogConfig(
        level=level,
        console_output=True,
        json_output=False,
        tree_format=True,
        service_name="ingestor",
    ))


def get_logger(name: str) -> Any:
    """Get a structured logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound structured logger (structlog.stdlib.BoundLogger).
    """
    return get_logger_utils(name)


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = int(size_bytes / 1024.0)
    return f"{size_bytes:.1f} PB"


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        num: Number to format.

    Returns:
        Formatted string (e.g., "1.5M").
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration (e.g., "1m 23s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


@dataclass
class FileProgressTask:
    """Track progress for a single task in file mode."""

    task_id: TaskID
    description: str
    total: int
    completed: int = 0
    trades: int = 0
    quotes: int = 0
    bars: int = 0
    start_time: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)


class FileProgress:
    """File-friendly progress tracker that prints line-by-line updates.

    Unlike Rich's Progress which uses cursor movement, this prints
    regular log lines that work correctly when stdout is redirected
    to a file (e.g., via nohup).

    Updates are throttled to avoid flooding the log with too many lines.
    """

    def __init__(
        self,
        log_interval: float = 10.0,
        console: Console | None = None,
    ) -> None:
        """Initialize file progress tracker.

        Args:
            log_interval: Minimum seconds between progress log lines.
            console: Optional Rich console for formatted output.
        """
        self._tasks: dict[TaskID, FileProgressTask] = {}
        self._log_interval = log_interval
        self._console = console or globals()["console"]
        self._started = False
        self._next_task_id = 0

    def __enter__(self) -> "FileProgress":
        self._started = True
        return self

    def __exit__(self, *args: Any) -> None:
        # Print final summary for all tasks
        for task in self._tasks.values():
            self._log_progress(task, final=True)
        self._started = False

    def add_task(
        self,
        description: str,
        total: int,
        **fields: Any,
    ) -> TaskID:
        """Add a new task to track.

        Args:
            description: Task description.
            total: Total steps to complete.
            **fields: Additional task fields.

        Returns:
            Task ID for progress tracking.
        """
        task_id = TaskID(self._next_task_id)
        self._next_task_id += 1
        self._tasks[task_id] = FileProgressTask(
            task_id=task_id,
            description=description,
            total=total,
        )
        # Log initial state
        self._console.print(
            f"[bold blue]Started:[/bold blue] {description} (0/{total})"
        )
        return task_id

    def update(
        self,
        task_id: TaskID,
        trades: str | None = None,
        quotes: str | None = None,
        bars: str | None = None,
        **fields: Any,
    ) -> None:
        """Update task fields (trades, quotes counts).

        Args:
            task_id: Task ID from add_task.
            trades: Formatted trades count string.
            quotes: Formatted quotes count string.
            bars: Formatted bars count string.
        """
        if task_id not in self._tasks:
            return
        task = self._tasks[task_id]
        # Store raw counts for logging (strip formatting)
        if trades is not None:
            task.trades = self._parse_count(trades)
        if quotes is not None:
            task.quotes = self._parse_count(quotes)

    def _parse_count(self, formatted: str) -> int:
        """Parse a formatted count back to int."""
        try:
            # Handle K, M, B suffixes
            s = formatted.strip().upper()
            if s.endswith("B"):
                return int(float(s[:-1]) * 1_000_000_000)
            elif s.endswith("M"):
                return int(float(s[:-1]) * 1_000_000)
            elif s.endswith("K"):
                return int(float(s[:-1]) * 1_000)
            return int(s)
        except (ValueError, IndexError):
            return 0

    def advance(self, task_id: TaskID, amount: int = 1) -> None:
        """Advance task progress.

        Args:
            task_id: Task ID from add_task.
            amount: Steps to advance (default 1).
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.completed += amount

        # Throttle logging to avoid flooding
        now = time.time()
        if now - task.last_log_time >= self._log_interval:
            self._log_progress(task)
            task.last_log_time = now

    def _log_progress(self, task: FileProgressTask, final: bool = False) -> None:
        """Log progress line for a task.

        Args:
            task: Task to log.
            final: Whether this is the final log for this task.
        """
        elapsed = time.time() - task.start_time
        pct = (task.completed / task.total * 100) if task.total > 0 else 0
        rate = (task.trades + task.quotes) / elapsed if elapsed > 0 else 0

        status = "[green]Complete[/green]" if final and task.completed >= task.total else "[yellow]Running[/yellow]"

        self._console.print(
            f"[bold]{task.description}[/bold] "
            f"[{status}] "
            f"{task.completed}/{task.total} ({pct:.0f}%) "
            f"| [green]{format_number(task.trades)}[/green] trades "
            f"| [cyan]{format_number(task.quotes)}[/cyan] quotes "
            f"| {format_number(int(rate))}/s "
            f"| {format_duration(elapsed)}"
        )
