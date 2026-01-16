"""
Logging configuration for backtester_py service.

Provides structured logging with memory-efficient output for large backtests.
Uses structlog for structured logging and rich for console formatting.
"""

from __future__ import annotations

import io
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from rich.console import Console
from rich.theme import Theme


# Custom theme for trading context
TRADING_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "profit": "green",
    "loss": "red",
    "neutral": "dim",
    "header": "bold magenta",
    "metric": "bold cyan",
    "trade_buy": "green",
    "trade_sell": "red",
})

# Global console instance
console = Console(theme=TRADING_THEME)


class RingBuffer:
    """
    Memory-efficient ring buffer for storing recent log entries.

    Keeps only the last N entries to prevent memory issues with large backtests.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize ring buffer.

        Args:
            max_size: Maximum number of entries to keep
        """
        self._buffer: list[dict[str, Any]] = []
        self._max_size = max_size
        self._total_count = 0

    def append(self, entry: dict[str, Any]) -> None:
        """Add an entry to the buffer."""
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(entry)
        self._total_count += 1

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the most recent N entries."""
        return self._buffer[-n:]

    def get_all(self) -> list[dict[str, Any]]:
        """Get all entries in buffer."""
        return self._buffer.copy()

    @property
    def total_count(self) -> int:
        """Total number of entries that have passed through."""
        return self._total_count

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_count = 0


class TradeLogBuffer:
    """
    Specialized buffer for trade logging with memory efficiency.

    Stores trades in a compact format and provides summary statistics.
    """

    def __init__(self, max_trades: int = 10000) -> None:
        """
        Initialize trade buffer.

        Args:
            max_trades: Maximum trades to keep in memory
        """
        self._trades: list[dict[str, Any]] = []
        self._max_trades = max_trades
        self._summary = {
            "total_trades": 0,
            "total_buy": 0,
            "total_sell": 0,
            "total_notional": 0.0,
            "total_slippage_bps": 0.0,
            "total_commission": 0.0,
        }

    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        notional: float,
        slippage_bps: float,
        commission: float,
    ) -> None:
        """Log a trade execution."""
        trade = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "notional": notional,
            "slippage_bps": slippage_bps,
            "commission": commission,
        }

        # Update summary
        self._summary["total_trades"] += 1
        if side == "buy":
            self._summary["total_buy"] += 1
        else:
            self._summary["total_sell"] += 1
        self._summary["total_notional"] += notional
        self._summary["total_slippage_bps"] += slippage_bps
        self._summary["total_commission"] += commission

        # Store trade with memory limit
        if len(self._trades) >= self._max_trades:
            self._trades.pop(0)
        self._trades.append(trade)

    def get_summary(self) -> dict[str, Any]:
        """Get trade summary statistics."""
        total = self._summary["total_trades"]
        return {
            **self._summary,
            "avg_slippage_bps": (
                self._summary["total_slippage_bps"] / total if total > 0 else 0.0
            ),
            "avg_commission": (
                self._summary["total_commission"] / total if total > 0 else 0.0
            ),
        }

    def get_recent_trades(self, n: int = 10) -> list[dict[str, Any]]:
        """Get most recent trades."""
        return self._trades[-n:]

    def clear(self) -> None:
        """Clear all trade data."""
        self._trades.clear()
        for key in self._summary:
            if isinstance(self._summary[key], float):
                self._summary[key] = 0.0
            else:
                self._summary[key] = 0


# Global trade buffer for backtest logging
_trade_buffer = TradeLogBuffer()


def get_trade_buffer() -> TradeLogBuffer:
    """Get the global trade buffer."""
    return _trade_buffer


def reset_trade_buffer(max_trades: int = 10000) -> None:
    """Reset the global trade buffer."""
    global _trade_buffer
    _trade_buffer = TradeLogBuffer(max_trades=max_trades)


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
    event_dict["service"] = "backtester_py"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = False,
    memory_efficient: bool = True,
) -> None:
    """
    Configure structured logging for the backtester service.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON format; otherwise human-readable
        memory_efficient: If True, use ring buffer for log storage
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


def get_logger(name: str = "backtester") -> structlog.BoundLogger:
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
    "TRADING_THEME",
    "RingBuffer",
    "TradeLogBuffer",
    "get_trade_buffer",
    "reset_trade_buffer",
    "configure_logging",
    "get_logger",
]
