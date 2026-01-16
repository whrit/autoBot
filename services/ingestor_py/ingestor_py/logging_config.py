"""
Logging Configuration - Structlog and Rich console setup.

Thin wrapper around logging_utils for backward compatibility.
Provides centralized logging configuration with structured logging
and beautiful console output.
"""

from rich.console import Console

from logging_utils import (
    LogConfig,
    configure_logging as configure_logging_utils,
    get_logger as get_logger_utils,
)
from logging_utils.setup import TRADING_THEME

# Re-export console for backward compatibility
console = Console(theme=TRADING_THEME, stderr=True)


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


def get_logger(name: str) -> "structlog.stdlib.BoundLogger":  # noqa: F821
    """Get a structured logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound structured logger.
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
