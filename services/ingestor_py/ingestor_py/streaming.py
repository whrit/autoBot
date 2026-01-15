"""
Realtime Streamer - WebSocket streaming via Alpaca.

Uses alpaca-py StockDataStream for real-time trades and quotes.
Includes reconnection logic with exponential backoff.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpaca.data.live import StockDataStream

from ingestor_py.writer import ParquetWriter


logger = logging.getLogger(__name__)


class RealtimeStreamer:
    """Real-time market data streaming via WebSocket.

    Streams trades and quotes for configured symbols and writes
    them to partitioned Parquet files in batches.

    Attributes:
        symbols: List of symbols to stream.
        feed: Data feed ('iex' or 'sip').
        reconnect_attempts: Number of reconnection attempts made.
        last_trade: Most recent trade record (for testing).
        last_quote: Most recent quote record (for testing).
    """

    # Batch size for writing to parquet
    BATCH_SIZE = 1000

    # Maximum reconnection attempts
    MAX_RECONNECT_ATTEMPTS = 10

    # Base delay for exponential backoff (seconds)
    RECONNECT_BASE_DELAY = 1.0

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lake_path: Path | str,
        symbols: list[str],
        feed: str = "iex",
    ) -> None:
        """Initialize the realtime streamer.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            lake_path: Path to the data lake directory.
            symbols: List of symbols to stream.
            feed: Data feed ('iex' or 'sip').
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.lake_path = Path(lake_path)
        self.symbols = symbols
        self.feed = feed

        self._stream = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
            feed=feed,  # type: ignore[arg-type]
        )

        self.writer = ParquetWriter(base_path=lake_path)

        # State tracking
        self._running = False
        self._trade_buffer: list[dict[str, object]] = []
        self._quote_buffer: list[dict[str, object]] = []
        self.reconnect_attempts = 0
        self.last_trade: dict[str, object] | None = None
        self.last_quote: dict[str, object] | None = None

        # Register handlers
        self._stream.subscribe_trades(self._on_trade, *self.symbols)
        self._stream.subscribe_quotes(self._on_quote, *self.symbols)

    def subscribe(self) -> None:
        """Subscribe to trades and quotes for configured symbols.

        This is called automatically during initialization but can
        be called again to re-subscribe after connection issues.
        """
        self._stream.subscribe_trades(self._on_trade, *self.symbols)
        self._stream.subscribe_quotes(self._on_quote, *self.symbols)

    def _on_trade(self, trade: Any) -> None:
        """Handle incoming trade message.

        Captures ts_recv immediately upon receipt for latency measurement.

        Args:
            trade: Trade object from Alpaca stream.
        """
        ts_recv = datetime.now(timezone.utc)

        trade_record: dict[str, object] = {
            "symbol": trade.symbol,
            "ts_event": trade.timestamp,
            "ts_recv": ts_recv,
            "price": float(trade.price),
            "size": float(trade.size),
            "exchange": trade.exchange,
            "conditions": (
                " ".join(trade.conditions) if trade.conditions else ""
            ),
        }

        self.last_trade = trade_record
        self._trade_buffer.append(trade_record)

        # Flush buffer if full
        if len(self._trade_buffer) >= self.BATCH_SIZE:
            self._flush_trades()

    def _on_quote(self, quote: Any) -> None:
        """Handle incoming quote message.

        Captures ts_recv immediately upon receipt for latency measurement.

        Args:
            quote: Quote object from Alpaca stream.
        """
        ts_recv = datetime.now(timezone.utc)

        quote_record: dict[str, object] = {
            "symbol": quote.symbol,
            "ts_event": quote.timestamp,
            "ts_recv": ts_recv,
            "bid_price": float(quote.bid_price),
            "bid_size": float(quote.bid_size),
            "ask_price": float(quote.ask_price),
            "ask_size": float(quote.ask_size),
            "bid_exchange": quote.bid_exchange,
            "ask_exchange": quote.ask_exchange,
            "conditions": (
                " ".join(quote.conditions) if quote.conditions else ""
            ),
        }

        self.last_quote = quote_record
        self._quote_buffer.append(quote_record)

        # Flush buffer if full
        if len(self._quote_buffer) >= self.BATCH_SIZE:
            self._flush_quotes()

    def _flush_trades(self) -> None:
        """Write buffered trades to Parquet."""
        if self._trade_buffer:
            try:
                self.writer.write_trades(self._trade_buffer)
                logger.info(f"Flushed {len(self._trade_buffer)} trades to parquet")
            except Exception as e:
                logger.error(f"Failed to flush trades: {e}")
            finally:
                self._trade_buffer = []

    def _flush_quotes(self) -> None:
        """Write buffered quotes to Parquet."""
        if self._quote_buffer:
            try:
                self.writer.write_quotes(self._quote_buffer)
                logger.info(f"Flushed {len(self._quote_buffer)} quotes to parquet")
            except Exception as e:
                logger.error(f"Failed to flush quotes: {e}")
            finally:
                self._quote_buffer = []

    def _flush_all(self) -> None:
        """Flush all buffers."""
        self._flush_trades()
        self._flush_quotes()

    async def _run_with_reconnect(self) -> None:
        """Run the stream with reconnection logic."""
        while self._running and self.reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            try:
                logger.info("Starting Alpaca WebSocket stream...")
                self._stream.run()
            except Exception as e:
                if not self._running:
                    break

                self.reconnect_attempts += 1
                delay = self.RECONNECT_BASE_DELAY * (2 ** (self.reconnect_attempts - 1))
                logger.warning(
                    f"Stream disconnected: {e}. "
                    f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})"
                )
                await asyncio.sleep(delay)

        if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error("Max reconnection attempts reached. Stopping stream.")

    async def start(self) -> None:
        """Start streaming and processing.

        This method runs until stop() is called or max reconnection
        attempts are reached.
        """
        self._running = True
        self.reconnect_attempts = 0

        try:
            await self._run_with_reconnect()
        finally:
            self._flush_all()

    async def stop(self) -> None:
        """Stop streaming gracefully.

        Flushes remaining buffers and closes the WebSocket connection.
        """
        logger.info("Stopping Alpaca WebSocket stream...")
        self._running = False

        # Flush remaining data
        self._flush_all()

        # Close the stream
        try:
            await self._stream.close()
        except Exception as e:
            logger.warning(f"Error closing stream: {e}")
