"""
Realtime Streamer - WebSocket streaming via Alpaca.

Uses alpaca-py StockDataStream for real-time trades and quotes.
Includes reconnection logic with exponential backoff and
comprehensive structured logging.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpaca.data.live import StockDataStream

from ingestor_py.logging_config import (
    console,
    format_bytes,
    format_duration,
    format_number,
    get_logger,
)
from ingestor_py.writer import ParquetWriter

logger = get_logger(__name__)


@dataclass
class StreamStats:
    """Statistics for streaming session."""

    start_time: float = field(default_factory=time.time)
    trades_received: int = 0
    quotes_received: int = 0
    trades_flushed: int = 0
    quotes_flushed: int = 0
    bytes_written: int = 0
    reconnect_attempts: int = 0
    last_trade_time: float | None = None
    last_quote_time: float | None = None
    last_stats_emission: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def trades_per_sec(self) -> float:
        """Get trades per second rate."""
        if self.elapsed == 0:
            return 0.0
        return self.trades_received / self.elapsed

    @property
    def quotes_per_sec(self) -> float:
        """Get quotes per second rate."""
        if self.elapsed == 0:
            return 0.0
        return self.quotes_received / self.elapsed

    @property
    def messages_per_sec(self) -> float:
        """Get total messages per second rate."""
        return self.trades_per_sec + self.quotes_per_sec


class RealtimeStreamer:
    """Real-time market data streaming via WebSocket.

    Streams trades and quotes for configured symbols and writes
    them to partitioned Parquet files in batches.

    Includes comprehensive structured logging for:
    - Connection status and WebSocket events
    - Message rates and throughput
    - Reconnection attempts with context
    - Periodic statistics emission

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

    # Stats emission interval (seconds)
    STATS_INTERVAL = 60.0

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

        # Statistics
        self.stats = StreamStats()

        # Register handlers
        self._stream.subscribe_trades(self._on_trade, *self.symbols)
        self._stream.subscribe_quotes(self._on_quote, *self.symbols)

        logger.info(
            "streamer_initialized",
            symbols=symbols,
            feed=feed,
            lake_path=str(lake_path),
            batch_size=self.BATCH_SIZE,
        )

    def subscribe(self) -> None:
        """Subscribe to trades and quotes for configured symbols.

        This is called automatically during initialization but can
        be called again to re-subscribe after connection issues.
        """
        self._stream.subscribe_trades(self._on_trade, *self.symbols)
        self._stream.subscribe_quotes(self._on_quote, *self.symbols)
        logger.info("subscribed_to_symbols", symbols=self.symbols)

    async def _on_trade(self, trade: Any) -> None:
        """Handle incoming trade message.

        Captures ts_recv immediately upon receipt for latency measurement.

        Args:
            trade: Trade object from Alpaca stream.
        """
        ts_recv = datetime.now(UTC)

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
        self.stats.trades_received += 1
        self.stats.last_trade_time = time.time()

        # Check if we should emit periodic stats
        self._maybe_emit_stats()

        # Flush buffer if full
        if len(self._trade_buffer) >= self.BATCH_SIZE:
            self._flush_trades()

    async def _on_quote(self, quote: Any) -> None:
        """Handle incoming quote message.

        Captures ts_recv immediately upon receipt for latency measurement.

        Args:
            quote: Quote object from Alpaca stream.
        """
        ts_recv = datetime.now(UTC)

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
        self.stats.quotes_received += 1
        self.stats.last_quote_time = time.time()

        # Check if we should emit periodic stats
        self._maybe_emit_stats()

        # Flush buffer if full
        if len(self._quote_buffer) >= self.BATCH_SIZE:
            self._flush_quotes()

    def _maybe_emit_stats(self) -> None:
        """Emit periodic statistics if interval has elapsed."""
        now = time.time()
        if now - self.stats.last_stats_emission >= self.STATS_INTERVAL:
            self._emit_stats()
            self.stats.last_stats_emission = now

    def _emit_stats(self) -> None:
        """Emit current streaming statistics."""
        logger.info(
            "streaming_stats",
            elapsed=format_duration(self.stats.elapsed),
            trades_received=self.stats.trades_received,
            quotes_received=self.stats.quotes_received,
            trades_flushed=self.stats.trades_flushed,
            quotes_flushed=self.stats.quotes_flushed,
            bytes_written=format_bytes(self.stats.bytes_written),
            trades_per_sec=f"{self.stats.trades_per_sec:.1f}",
            quotes_per_sec=f"{self.stats.quotes_per_sec:.1f}",
            messages_per_sec=f"{self.stats.messages_per_sec:.1f}",
            buffer_trades=len(self._trade_buffer),
            buffer_quotes=len(self._quote_buffer),
        )

    def _flush_trades(self) -> None:
        """Write buffered trades to Parquet."""
        if self._trade_buffer:
            count = len(self._trade_buffer)
            try:
                bytes_written = self.writer.write_trades(self._trade_buffer)
                self.stats.trades_flushed += count
                self.stats.bytes_written += bytes_written

                logger.info(
                    "trades_flushed",
                    count=count,
                    bytes_written=format_bytes(bytes_written),
                    total_flushed=self.stats.trades_flushed,
                )
            except Exception as e:
                logger.error(
                    "trades_flush_error",
                    error=str(e),
                    buffer_size=count,
                    exc_info=True,
                )
            finally:
                self._trade_buffer = []

    def _flush_quotes(self) -> None:
        """Write buffered quotes to Parquet."""
        if self._quote_buffer:
            count = len(self._quote_buffer)
            try:
                bytes_written = self.writer.write_quotes(self._quote_buffer)
                self.stats.quotes_flushed += count
                self.stats.bytes_written += bytes_written

                logger.info(
                    "quotes_flushed",
                    count=count,
                    bytes_written=format_bytes(bytes_written),
                    total_flushed=self.stats.quotes_flushed,
                )
            except Exception as e:
                logger.error(
                    "quotes_flush_error",
                    error=str(e),
                    buffer_size=count,
                    exc_info=True,
                )
            finally:
                self._quote_buffer = []

    def _flush_all(self) -> None:
        """Flush all buffers."""
        self._flush_trades()
        self._flush_quotes()

    async def _run_with_reconnect(self) -> None:
        """Run the stream with reconnection logic.

        The Alpaca StockDataStream.run() is blocking, so we run it in
        a thread executor to avoid blocking the async event loop.
        """
        loop = asyncio.get_event_loop()

        while self._running and self.reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            try:
                logger.info(
                    "websocket_connecting",
                    symbols=self.symbols,
                    feed=self.feed,
                    attempt=self.reconnect_attempts + 1,
                )

                # Run blocking stream in thread executor
                await loop.run_in_executor(None, self._stream.run)

                # If we get here normally, connection was closed gracefully
                if self._running:
                    logger.warning("websocket_closed_unexpectedly")

            except Exception as e:
                if not self._running:
                    break

                self.reconnect_attempts += 1
                self.stats.reconnect_attempts = self.reconnect_attempts
                delay = self.RECONNECT_BASE_DELAY * (2 ** (self.reconnect_attempts - 1))

                logger.warning(
                    "websocket_disconnected",
                    error=str(e),
                    reconnect_attempt=self.reconnect_attempts,
                    max_attempts=self.MAX_RECONNECT_ATTEMPTS,
                    retry_delay=f"{delay:.1f}s",
                )

                await asyncio.sleep(delay)

                # Re-subscribe after reconnection
                if self._running:
                    logger.info("websocket_resubscribing", symbols=self.symbols)
                    self.subscribe()

        if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "max_reconnects_exceeded",
                max_attempts=self.MAX_RECONNECT_ATTEMPTS,
                total_trades=self.stats.trades_received,
                total_quotes=self.stats.quotes_received,
            )

    async def start(self) -> None:
        """Start streaming and processing.

        This method runs until stop() is called or max reconnection
        attempts are reached.
        """
        self._running = True
        self.reconnect_attempts = 0
        self.stats = StreamStats()

        logger.info(
            "stream_starting",
            symbols=self.symbols,
            feed=self.feed,
            batch_size=self.BATCH_SIZE,
            stats_interval=f"{self.STATS_INTERVAL}s",
        )

        console.print(f"\n[bold green]Stream started[/bold green] for {', '.join(self.symbols)}")
        console.print(f"[dim]Stats will be logged every {int(self.STATS_INTERVAL)}s[/dim]\n")

        try:
            await self._run_with_reconnect()
        finally:
            self._flush_all()
            self._emit_final_stats()

    def _emit_final_stats(self) -> None:
        """Emit final statistics when stream ends."""
        logger.info(
            "stream_final_stats",
            elapsed=format_duration(self.stats.elapsed),
            total_trades=self.stats.trades_received,
            total_quotes=self.stats.quotes_received,
            trades_flushed=self.stats.trades_flushed,
            quotes_flushed=self.stats.quotes_flushed,
            bytes_written=format_bytes(self.stats.bytes_written),
            avg_trades_per_sec=f"{self.stats.trades_per_sec:.1f}",
            avg_quotes_per_sec=f"{self.stats.quotes_per_sec:.1f}",
            reconnect_attempts=self.stats.reconnect_attempts,
        )

        console.print("\n[bold]Stream Summary[/bold]")
        console.print(f"  Duration: {format_duration(self.stats.elapsed)}")
        console.print(f"  Trades: {format_number(self.stats.trades_received)}")
        console.print(f"  Quotes: {format_number(self.stats.quotes_received)}")
        console.print(f"  Data Written: {format_bytes(self.stats.bytes_written)}")

    async def stop(self) -> None:
        """Stop streaming gracefully.

        Flushes remaining buffers and closes the WebSocket connection.
        """
        logger.info(
            "stream_stopping",
            trades_in_buffer=len(self._trade_buffer),
            quotes_in_buffer=len(self._quote_buffer),
        )

        self._running = False

        # Flush remaining data
        self._flush_all()

        # Close the stream
        try:
            await self._stream.close()
            logger.info("websocket_closed")
        except Exception as e:
            logger.warning("websocket_close_error", error=str(e))
