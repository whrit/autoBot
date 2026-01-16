"""
Fill Confirmation Stream - WebSocket-based fill confirmations (T6.04).

Uses Alpaca TradingStream to receive real-time order status updates
and fill confirmations for paper trading.

CRITICAL: Always uses paper=True to connect to paper trading WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from alpaca.trading.stream import TradingStream

from runner_py.paper import PaperFill

logger = logging.getLogger(__name__)


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

        logger.info("FillConfirmationStream initialized with paper trading mode")

    def register_callback(self, callback: Callable[[PaperFill], None]) -> None:
        """
        Register callback for fill notifications.

        Args:
            callback: Function to call when fill is received
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered fill callback: {callback}")

    def unregister_callback(self, callback: Callable[[PaperFill], None]) -> None:
        """
        Unregister a fill callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered fill callback: {callback}")

    async def _handle_trade_update(self, data: Any) -> None:
        """
        Handle trade update from WebSocket.

        Parses fill data and notifies registered callbacks.

        Args:
            data: Trade update event from Alpaca WebSocket
        """
        try:
            event_type = data.event if hasattr(data, "event") else data.get("event", "")

            # Only process fill-related events
            if event_type not in ("fill", "partial_fill"):
                logger.debug(f"Ignoring non-fill event: {event_type}")
                return

            order = data.order if hasattr(data, "order") else data.get("order", {})

            # Extract fill information
            if hasattr(order, "id"):
                order_id = order.id
                symbol = order.symbol
                side = order.side if isinstance(order.side, str) else order.side
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                status = order.status if isinstance(order.status, str) else order.status
            else:
                order_id = order.get("id", "")
                symbol = order.get("symbol", "")
                side = order.get("side", "")
                filled_qty = float(order.get("filled_qty", 0))
                filled_price = float(order.get("filled_avg_price", 0))
                status = order.get("status", "")

            # Get timestamp
            if hasattr(data, "timestamp"):
                filled_at = (
                    data.timestamp
                    if isinstance(data.timestamp, datetime)
                    else datetime.now(UTC)
                )
            else:
                filled_at = datetime.now(UTC)

            # Create fill record
            fill = PaperFill(
                order_id=str(order_id),
                symbol=str(symbol),
                side=str(side),
                qty=filled_qty,
                filled_price=filled_price,
                filled_at=filled_at,
                status=str(status),
            )

            logger.info(
                f"Fill received: {fill.symbol} {fill.side} {fill.qty}@{fill.filled_price} "
                f"order_id={fill.order_id}"
            )

            # Store in history
            self._add_to_history(fill)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error handling trade update: {e}", exc_info=True)

    def _add_to_history(self, fill: PaperFill) -> None:
        """
        Add fill to history, maintaining max size.

        Args:
            fill: Fill to add
        """
        self._fill_history.append(fill)

        # Trim history if needed
        if len(self._fill_history) > self.MAX_HISTORY_SIZE:
            self._fill_history = self._fill_history[-self.MAX_HISTORY_SIZE:]

    async def start(self) -> None:
        """
        Start the fill confirmation stream.

        Subscribes to trade updates and runs the WebSocket connection.
        Blocks until stop() is called or connection fails.
        """
        logger.info("Starting fill confirmation stream...")
        self._running = True

        # Subscribe to trade updates
        self._stream.subscribe_trade_updates(self._handle_trade_update)

        try:
            # TradingStream.run() is synchronous, run in executor for async compat
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._stream.run)
        except Exception as e:
            if self._running:
                logger.error(f"Stream error: {e}", exc_info=True)
                raise

    async def stop(self) -> None:
        """
        Stop the stream gracefully.

        Closes WebSocket connection and cleans up.
        """
        logger.info("Stopping fill confirmation stream...")
        self._running = False

        try:
            await self._stream.close()
        except Exception as e:
            logger.warning(f"Error closing stream: {e}")

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
        self._fill_history.clear()
        logger.info("Fill history cleared")

    def get_stats(self) -> dict[str, Any]:
        """
        Get stream statistics.

        Returns:
            Dictionary with stream stats
        """
        total_notional = sum(f.qty * f.filled_price for f in self._fill_history)
        symbols = {f.symbol for f in self._fill_history}

        return {
            "total_fills": len(self._fill_history),
            "total_notional": total_notional,
            "symbols_traded": symbols,
            "callback_count": len(self._callbacks),
            "is_running": self._running,
        }
