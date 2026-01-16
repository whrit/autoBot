"""
Fill Confirmation Stream - WebSocket-based fill confirmations (T6.04).

Uses Alpaca TradingStream to receive real-time order status updates
and fill confirmations for paper trading.

CRITICAL: Always uses paper=True to connect to paper trading WebSocket.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from alpaca.trading.stream import TradingStream

from runner_py.paper import PaperFill

logger = structlog.get_logger(__name__)


@dataclass
class FillStreamConfig:
    """
    Configuration for fill confirmation stream.

    Attributes:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: Must be True for paper trading (enforced)
    """

    api_key: str
    api_secret: str
    paper: bool = True


@dataclass
class LatencyStats:
    """Track latency statistics for the fill stream."""

    samples: list[float] = field(default_factory=list)
    max_samples: int = 1000

    def add_sample(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def min_latency_ms(self) -> float:
        """Get minimum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return min(self.samples)

    @property
    def max_latency_ms(self) -> float:
        """Get maximum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return max(self.samples)

    @property
    def p95_latency_ms(self) -> float:
        """Get 95th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_count": len(self.samples),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


@dataclass
class ConnectionStats:
    """Track WebSocket connection statistics."""

    connected_at: datetime | None = None
    disconnected_at: datetime | None = None
    reconnect_count: int = 0
    total_messages: int = 0
    error_count: int = 0
    last_message_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        uptime_seconds = 0.0
        if self.connected_at:
            end_time = self.disconnected_at or datetime.now(UTC)
            uptime_seconds = (end_time - self.connected_at).total_seconds()

        return {
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "uptime_seconds": round(uptime_seconds, 2),
            "reconnect_count": self.reconnect_count,
            "total_messages": self.total_messages,
            "error_count": self.error_count,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }


class FillConfirmationStream:
    """
    Stream fill confirmations via Alpaca WebSocket.

    Provides real-time notifications when paper trading orders are filled.
    Supports callback registration for fill notifications and maintains
    a history of recent fills.

    CRITICAL: Always creates TradingStream with paper=True.
    """

    # Maximum fill history entries to retain
    MAX_HISTORY_SIZE = 1000

    def __init__(self, config: FillStreamConfig) -> None:
        """
        Initialize the fill confirmation stream.

        Args:
            config: Stream configuration
        """
        self.config = config

        # CRITICAL: Always use paper=True
        self._stream = TradingStream(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=True,  # NEVER change this to False
        )

        self._callbacks: list[Callable[[PaperFill], None]] = []
        self._fill_history: list[PaperFill] = []
        self._running = False
        self._latency_stats = LatencyStats()
        self._connection_stats = ConnectionStats()
        self._order_timestamps: dict[str, float] = {}  # order_id -> submission timestamp
        self._logger = logger.bind(component="fill_stream")

        self._logger.info(
            "fill_stream_initialized",
            paper_mode=True,
            max_history_size=self.MAX_HISTORY_SIZE,
        )

    def register_callback(self, callback: Callable[[PaperFill], None]) -> None:
        """
        Register callback for fill notifications.

        Args:
            callback: Function to call when fill is received
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            self._logger.info(
                "callback_registered",
                callback_name=callback.__name__ if hasattr(callback, "__name__") else str(callback),
                total_callbacks=len(self._callbacks),
            )

    def unregister_callback(self, callback: Callable[[PaperFill], None]) -> None:
        """
        Unregister a fill callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            self._logger.info(
                "callback_unregistered",
                callback_name=callback.__name__ if hasattr(callback, "__name__") else str(callback),
                remaining_callbacks=len(self._callbacks),
            )

    def track_order_submission(self, order_id: str) -> None:
        """
        Track when an order was submitted for latency calculation.

        Args:
            order_id: The order ID being submitted
        """
        self._order_timestamps[order_id] = time.time()
        self._logger.debug(
            "order_tracked_for_latency",
            order_id=order_id,
        )

    async def _handle_trade_update(self, data: Any) -> None:
        """
        Handle trade update from WebSocket.

        Parses fill data and notifies registered callbacks.

        Args:
            data: Trade update event from Alpaca WebSocket
        """
        receive_time = time.time()
        self._connection_stats.total_messages += 1
        self._connection_stats.last_message_at = datetime.now(UTC)

        try:
            event_type = data.event if hasattr(data, "event") else data.get("event", "")

            # Log all trade events for debugging
            self._logger.debug(
                "trade_event_received",
                event_type=event_type,
                message_count=self._connection_stats.total_messages,
            )

            # Only process fill-related events
            if event_type not in ("fill", "partial_fill"):
                self._logger.debug(
                    "non_fill_event",
                    event_type=event_type,
                )
                return

            order = data.order if hasattr(data, "order") else data.get("order", {})

            # Extract fill information
            if hasattr(order, "id"):
                order_id = str(order.id)
                symbol = order.symbol
                side = order.side if isinstance(order.side, str) else str(order.side) if order.side else "unknown"
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                status = order.status if isinstance(order.status, str) else str(order.status) if order.status else "unknown"
            else:
                order_id = str(order.get("id", ""))
                symbol = order.get("symbol", "")
                side = order.get("side", "unknown")
                filled_qty = float(order.get("filled_qty", 0))
                filled_price = float(order.get("filled_avg_price", 0))
                status = order.get("status", "unknown")

            # Get timestamp
            if hasattr(data, "timestamp"):
                filled_at = (
                    data.timestamp
                    if isinstance(data.timestamp, datetime)
                    else datetime.now(UTC)
                )
            else:
                filled_at = datetime.now(UTC)

            # Calculate latency if we tracked this order
            latency_ms = None
            if order_id in self._order_timestamps:
                submission_time = self._order_timestamps.pop(order_id)
                latency_ms = (receive_time - submission_time) * 1000
                self._latency_stats.add_sample(latency_ms)

            # Create fill record
            fill = PaperFill(
                order_id=order_id,
                symbol=str(symbol),
                side=str(side),
                qty=filled_qty,
                filled_price=filled_price,
                filled_at=filled_at,
                status=str(status),
            )

            # Calculate slippage in basis points if we have the info
            notional = filled_qty * filled_price

            self._logger.info(
                "fill_confirmed",
                order_id=fill.order_id,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                filled_price=fill.filled_price,
                notional=notional,
                status=fill.status,
                filled_at=fill.filled_at.isoformat(),
                latency_ms=round(latency_ms, 2) if latency_ms else None,
                event_type=event_type,
            )

            # Store in history
            self._add_to_history(fill)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    self._connection_stats.error_count += 1
                    self._logger.error(
                        "callback_error",
                        callback_name=callback.__name__ if hasattr(callback, "__name__") else str(callback),
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        except Exception as e:
            self._connection_stats.error_count += 1
            self._logger.error(
                "trade_update_processing_error",
                error=str(e),
                error_type=type(e).__name__,
            )

    def _add_to_history(self, fill: PaperFill) -> None:
        """
        Add fill to history, maintaining max size.

        Args:
            fill: Fill to add
        """
        self._fill_history.append(fill)

        # Trim history if needed
        if len(self._fill_history) > self.MAX_HISTORY_SIZE:
            trimmed_count = len(self._fill_history) - self.MAX_HISTORY_SIZE
            self._fill_history = self._fill_history[-self.MAX_HISTORY_SIZE:]
            self._logger.debug(
                "fill_history_trimmed",
                trimmed_count=trimmed_count,
                current_size=len(self._fill_history),
            )

    async def start(self) -> None:
        """
        Start the fill confirmation stream.

        Subscribes to trade updates and runs the WebSocket connection.
        Blocks until stop() is called or connection fails.
        """
        self._logger.info(
            "stream_starting",
            callback_count=len(self._callbacks),
        )

        self._running = True
        self._connection_stats.connected_at = datetime.now(UTC)
        self._connection_stats.disconnected_at = None

        # Subscribe to trade updates
        self._stream.subscribe_trade_updates(self._handle_trade_update)

        self._logger.info(
            "websocket_connected",
            connected_at=self._connection_stats.connected_at.isoformat(),
        )

        try:
            # TradingStream.run() is synchronous, run in executor for async compat
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._stream.run)
        except Exception as e:
            self._connection_stats.error_count += 1
            if self._running:
                self._logger.error(
                    "websocket_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    uptime_seconds=(datetime.now(UTC) - self._connection_stats.connected_at).total_seconds()
                    if self._connection_stats.connected_at
                    else 0,
                )
                raise
        finally:
            self._connection_stats.disconnected_at = datetime.now(UTC)
            self._logger.info(
                "websocket_disconnected",
                disconnected_at=self._connection_stats.disconnected_at.isoformat(),
                total_messages=self._connection_stats.total_messages,
                error_count=self._connection_stats.error_count,
            )

    async def stop(self) -> None:
        """
        Stop the stream gracefully.

        Closes WebSocket connection and cleans up.
        """
        self._logger.info(
            "stream_stopping",
            total_messages=self._connection_stats.total_messages,
            fill_count=len(self._fill_history),
        )

        self._running = False

        try:
            await self._stream.close()
            self._connection_stats.disconnected_at = datetime.now(UTC)

            self._logger.info(
                "stream_stopped",
                graceful=True,
                uptime_seconds=(self._connection_stats.disconnected_at - self._connection_stats.connected_at).total_seconds()
                if self._connection_stats.connected_at
                else 0,
            )
        except Exception as e:
            self._connection_stats.error_count += 1
            self._logger.warning(
                "stream_stop_error",
                error=str(e),
                error_type=type(e).__name__,
            )

    def get_recent_fills(self, limit: int = 100) -> list[PaperFill]:
        """
        Get recent fill history.

        Args:
            limit: Maximum number of fills to return

        Returns:
            List of recent fills (most recent first)
        """
        # Return most recent first
        return list(reversed(self._fill_history[-limit:]))

    def get_fill_by_order_id(self, order_id: str) -> PaperFill | None:
        """
        Get fill for specific order.

        Args:
            order_id: Order ID to search for

        Returns:
            PaperFill if found, None otherwise
        """
        for fill in reversed(self._fill_history):
            if fill.order_id == order_id:
                return fill
        return None

    def get_fills_by_symbol(self, symbol: str) -> list[PaperFill]:
        """
        Get all fills for a specific symbol.

        Args:
            symbol: Trading symbol to filter by

        Returns:
            List of fills for the symbol
        """
        return [fill for fill in self._fill_history if fill.symbol == symbol]

    def clear_history(self) -> None:
        """Clear all fill history."""
        old_count = len(self._fill_history)
        self._fill_history.clear()
        self._logger.info(
            "fill_history_cleared",
            cleared_count=old_count,
        )

    def get_latency_stats(self) -> dict[str, Any]:
        """
        Get latency statistics.

        Returns:
            Dictionary with latency metrics
        """
        return self._latency_stats.to_dict()

    def get_connection_stats(self) -> dict[str, Any]:
        """
        Get WebSocket connection statistics.

        Returns:
            Dictionary with connection metrics
        """
        return self._connection_stats.to_dict()

    def get_stats(self) -> dict[str, Any]:
        """
        Get stream statistics.

        Returns:
            Dictionary with stream stats
        """
        total_notional = sum(f.qty * f.filled_price for f in self._fill_history)
        symbols = {f.symbol for f in self._fill_history}

        stats = {
            "total_fills": len(self._fill_history),
            "total_notional": total_notional,
            "symbols_traded": list(symbols),
            "callback_count": len(self._callbacks),
            "is_running": self._running,
            "latency": self._latency_stats.to_dict(),
            "connection": self._connection_stats.to_dict(),
        }

        self._logger.debug(
            "stats_retrieved",
            **stats,
        )

        return stats

    def log_status_summary(self) -> None:
        """Log a comprehensive status summary."""
        stats = self.get_stats()
        self._logger.info(
            "stream_status_summary",
            is_running=stats["is_running"],
            total_fills=stats["total_fills"],
            total_notional=stats["total_notional"],
            symbols_traded=stats["symbols_traded"],
            callback_count=stats["callback_count"],
            latency_stats=stats["latency"],
            connection_stats=stats["connection"],
        )
