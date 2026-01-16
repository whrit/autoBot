"""
Logging configuration for optimizer_py service.

Provides structured logging with memory-efficient output for optimization runs.
Uses structlog for structured logging and rich for console formatting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from rich.console import Console
from rich.theme import Theme


# Custom theme for optimization context
OPTIMIZER_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "best": "bold magenta",
    "improvement": "green",
    "no_improvement": "dim",
    "header": "bold magenta",
    "metric": "bold cyan",
    "gpu": "bold yellow",
    "cpu": "dim cyan",
})

# Global console instance
console = Console(theme=OPTIMIZER_THEME)


class OptimizationTracker:
    """
    Track optimization progress and best results.

    Memory-efficient tracking of parameter combinations tested
    and their performance metrics.
    """

    def __init__(self, max_history: int = 1000) -> None:
        """
        Initialize optimization tracker.

        Args:
            max_history: Maximum number of results to keep in history
        """
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history
        self._best_score: float = float("-inf")
        self._best_params: dict[str, Any] = {}
        self._total_evaluated = 0
        self._improvements = 0

    def record_result(
        self,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        """
        Record an optimization result.

        Args:
            params: Parameter combination tested
            score: Score achieved
            metrics: Additional metrics

        Returns:
            True if this is a new best result
        """
        is_best = score > self._best_score

        if is_best:
            self._best_score = score
            self._best_params = params.copy()
            self._improvements += 1

        self._total_evaluated += 1

        # Store in history with memory limit
        result = {
            "params": params,
            "score": score,
            "metrics": metrics or {},
            "is_best": is_best,
            "eval_number": self._total_evaluated,
        }

        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append(result)

        return is_best

    def get_best(self) -> tuple[dict[str, Any], float]:
        """Get best parameters and score."""
        return self._best_params, self._best_score

    def get_top_n(self, n: int = 10) -> list[dict[str, Any]]:
        """Get top N results by score."""
        sorted_results = sorted(
            self._history,
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_results[:n]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        scores = [r["score"] for r in self._history]
        return {
            "total_evaluated": self._total_evaluated,
            "improvements": self._improvements,
            "best_score": self._best_score,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }

    def clear(self) -> None:
        """Reset the tracker."""
        self._history.clear()
        self._best_score = float("-inf")
        self._best_params = {}
        self._total_evaluated = 0
        self._improvements = 0


# Global tracker instance
_tracker = OptimizationTracker()


def get_tracker() -> OptimizationTracker:
    """Get the global optimization tracker."""
    return _tracker


def reset_tracker(max_history: int = 1000) -> None:
    """Reset the global tracker."""
    global _tracker
    _tracker = OptimizationTracker(max_history=max_history)


def add_timestamp(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def add_service_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service context to log entries."""
    event_dict["service"] = "optimizer_py"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = False,
) -> None:
    """
    Configure structured logging for the optimizer service.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON format; otherwise human-readable
    """
    processors: list[Callable[..., Any]] = [
        structlog.processors.add_log_level,
        add_timestamp,
        add_service_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Filter by log level in processors instead of wrapper
    min_level = getattr(logging, log_level.upper(), logging.INFO)

    def level_filter(
        logger: structlog.types.WrappedLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Filter events below minimum log level."""
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        event_level = level_map.get(method_name, logging.INFO)
        if event_level < min_level:
            raise structlog.DropEvent
        return event_dict

    # Insert level filter at the beginning
    processors.insert(0, level_filter)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "optimizer") -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name for context

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Configure logging on module import
configure_logging()


__all__ = [
    "console",
    "OPTIMIZER_THEME",
    "OptimizationTracker",
    "get_tracker",
    "reset_tracker",
    "configure_logging",
    "get_logger",
]
