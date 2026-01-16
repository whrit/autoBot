"""
Backfill Orchestrator - Coordinate historical data ingestion.

Handles downloading historical trades, quotes, and bars from Alpaca
and writing them to partitioned Parquet files with comprehensive
logging and progress tracking.

Optimized for streaming writes with day-by-day processing and
real-time progress updates showing throughput and current status.

Supports both sequential and parallel modes:
- Sequential: Traditional day-by-day processing (lower API usage).
- Parallel: Concurrent symbol fetching using asyncio (faster for many symbols).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ingestor_py.client import AlpacaDataClient, AsyncAlpacaDataClient, SymbolFetchResult
from ingestor_py.logging_config import (
    FileProgress,
    console,
    format_bytes,
    format_duration,
    format_number,
    get_logger,
    is_file_mode,
)
from ingestor_py.writer import ParquetWriter

logger = get_logger(__name__)

# Type alias for progress tracker (Rich Progress or FileProgress)
ProgressType = Progress | FileProgress


class DataTypeSummary(TypedDict):
    """Summary of ingested data for a single data type."""

    count: int
    start: str
    end: str


@dataclass
class SymbolStats:
    """Statistics for a single symbol's backfill."""

    symbol: str
    trades_count: int = 0
    quotes_count: int = 0
    bars_count: int = 0
    bytes_written: int = 0
    days_completed: int = 0
    total_days: int = 0
    current_day: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def total_records(self) -> int:
        """Get total records across all data types."""
        return self.trades_count + self.quotes_count + self.bars_count

    @property
    def throughput(self) -> float:
        """Get records per second."""
        if self.elapsed == 0:
            return 0.0
        return self.total_records / self.elapsed

    @property
    def days_progress(self) -> str:
        """Get days progress as string."""
        return f"{self.days_completed}/{self.total_days}"


@dataclass
class BackfillStats:
    """Aggregate statistics for entire backfill operation."""

    symbols: dict[str, SymbolStats] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def total_trades(self) -> int:
        """Get total trades across all symbols."""
        return sum(s.trades_count for s in self.symbols.values())

    @property
    def total_quotes(self) -> int:
        """Get total quotes across all symbols."""
        return sum(s.quotes_count for s in self.symbols.values())

    @property
    def total_bars(self) -> int:
        """Get total bars across all symbols."""
        return sum(s.bars_count for s in self.symbols.values())

    @property
    def total_records(self) -> int:
        """Get total records across all symbols and data types."""
        return self.total_trades + self.total_quotes + self.total_bars

    @property
    def total_bytes(self) -> int:
        """Get total bytes written."""
        return sum(s.bytes_written for s in self.symbols.values())

    @property
    def avg_throughput(self) -> float:
        """Get average records per second."""
        if self.elapsed == 0:
            return 0.0
        return self.total_records / self.elapsed


class BackfillOrchestrator:
    """Orchestrate historical data backfill.

    Coordinates between AlpacaDataClient for fetching data
    and ParquetWriter for persisting to the data lake.
    Includes comprehensive logging and progress tracking.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lake_path: Path | str,
        feed: str = "iex",
    ) -> None:
        """Initialize the backfill orchestrator.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            lake_path: Path to the data lake directory.
            feed: Data feed ('iex' or 'sip').
        """
        self.client = AlpacaDataClient(
            api_key=api_key,
            api_secret=api_secret,
            feed=feed,
        )
        self.writer = ParquetWriter(base_path=lake_path)
        self.lake_path = Path(lake_path)
        self.stats = BackfillStats()

        logger.info(
            "backfill_orchestrator_initialized",
            lake_path=str(lake_path),
            feed=feed,
        )

    def _iter_days(
        self,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Generate day-by-day date ranges for streaming processing.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of (day_start, day_end) tuples for each day in range.
        """
        days: list[tuple[datetime, datetime]] = []
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)

        while current < end:
            day_start = current
            day_end = min(current + timedelta(days=1), end)
            days.append((day_start, day_end))
            current = day_end

        return days

    def _create_progress(self) -> ProgressType:
        """Create a progress tracker for backfill tracking.

        Returns Rich Progress for interactive terminals, or FileProgress
        for file-redirected output (e.g., nohup).
        """
        if is_file_mode():
            logger.debug("using_file_progress_mode")
            return FileProgress(log_interval=10.0, console=console)

        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[symbol]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]|"),
            TextColumn("[yellow]{task.fields[current_day]}[/]"),
            TextColumn("[dim]|"),
            TextColumn("[green]{task.fields[records]}[/] rec"),
            TextColumn("[dim]|"),
            TextColumn("[magenta]{task.fields[throughput]}[/]/s"),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )

    def _backfill_symbol_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        progress: ProgressType | None = None,
        task_id: TaskID | None = None,
    ) -> int:
        """Backfill trades for a single symbol.

        Args:
            symbol: Stock symbol.
            start: Start datetime.
            end: End datetime.
            progress: Optional progress bar.
            task_id: Optional task ID for progress updates.

        Returns:
            Number of trades ingested.
        """
        log = logger.bind(symbol=symbol, data_type="trades")
        log.debug("fetching_trades", start=start.isoformat(), end=end.isoformat())

        fetch_start = time.time()
        trades = list(self.client.get_trades(symbol, start, end))
        fetch_elapsed = time.time() - fetch_start

        if trades:
            write_start = time.time()
            bytes_written = self.writer.write_trades(trades)
            write_elapsed = time.time() - write_start

            # Update stats
            self.stats.symbols[symbol].trades_count += len(trades)
            self.stats.symbols[symbol].bytes_written += bytes_written

            log.info(
                "trades_ingested",
                count=len(trades),
                bytes_written=bytes_written,
                fetch_time=f"{fetch_elapsed:.2f}s",
                write_time=f"{write_elapsed:.2f}s",
            )

            # Update progress
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    trades=format_number(self.stats.symbols[symbol].trades_count),
                )

        return len(trades)

    def _backfill_symbol_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        progress: ProgressType | None = None,
        task_id: TaskID | None = None,
    ) -> int:
        """Backfill quotes for a single symbol.

        Args:
            symbol: Stock symbol.
            start: Start datetime.
            end: End datetime.
            progress: Optional progress bar.
            task_id: Optional task ID for progress updates.

        Returns:
            Number of quotes ingested.
        """
        log = logger.bind(symbol=symbol, data_type="quotes")
        log.debug("fetching_quotes", start=start.isoformat(), end=end.isoformat())

        fetch_start = time.time()
        quotes = list(self.client.get_quotes(symbol, start, end))
        fetch_elapsed = time.time() - fetch_start

        if quotes:
            write_start = time.time()
            bytes_written = self.writer.write_quotes(quotes)
            write_elapsed = time.time() - write_start

            # Update stats
            self.stats.symbols[symbol].quotes_count += len(quotes)
            self.stats.symbols[symbol].bytes_written += bytes_written

            log.info(
                "quotes_ingested",
                count=len(quotes),
                bytes_written=bytes_written,
                fetch_time=f"{fetch_elapsed:.2f}s",
                write_time=f"{write_elapsed:.2f}s",
            )

            # Update progress
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    quotes=format_number(self.stats.symbols[symbol].quotes_count),
                )

        return len(quotes)

    def _backfill_symbol_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """Backfill bars for a single symbol.

        Args:
            symbol: Stock symbol.
            start: Start datetime.
            end: End datetime.

        Returns:
            Number of bars ingested.
        """
        log = logger.bind(symbol=symbol, data_type="bars")
        log.debug("fetching_bars", start=start.isoformat(), end=end.isoformat())

        fetch_start = time.time()
        bars = list(self.client.get_bars(symbol, start, end))
        fetch_elapsed = time.time() - fetch_start

        if bars:
            write_start = time.time()
            bytes_written = self.writer.write_bars(bars)
            write_elapsed = time.time() - write_start

            # Update stats
            self.stats.symbols[symbol].bars_count += len(bars)
            self.stats.symbols[symbol].bytes_written += bytes_written

            log.info(
                "bars_ingested",
                count=len(bars),
                bytes_written=bytes_written,
                fetch_time=f"{fetch_elapsed:.2f}s",
                write_time=f"{write_elapsed:.2f}s",
            )

        return len(bars)

    def _backfill_trades(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> DataTypeSummary:
        """Backfill trades for given symbols and date range.

        Args:
            symbols: List of stock symbols.
            start: Start datetime.
            end: End datetime.

        Returns:
            Summary of ingested trades.
        """
        total_count = 0

        for symbol in symbols:
            trades = list(self.client.get_trades(symbol, start, end))
            if trades:
                self.writer.write_trades(trades)
                total_count += len(trades)
                logger.info(
                    "trades_batch_complete",
                    symbol=symbol,
                    count=len(trades),
                )

        return DataTypeSummary(
            count=total_count,
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def _backfill_quotes(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> DataTypeSummary:
        """Backfill quotes for given symbols and date range.

        Args:
            symbols: List of stock symbols.
            start: Start datetime.
            end: End datetime.

        Returns:
            Summary of ingested quotes.
        """
        total_count = 0

        for symbol in symbols:
            quotes = list(self.client.get_quotes(symbol, start, end))
            if quotes:
                self.writer.write_quotes(quotes)
                total_count += len(quotes)
                logger.info(
                    "quotes_batch_complete",
                    symbol=symbol,
                    count=len(quotes),
                )

        return DataTypeSummary(
            count=total_count,
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def _backfill_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> DataTypeSummary:
        """Backfill bars for given symbols and date range.

        Args:
            symbols: List of stock symbols.
            start: Start datetime.
            end: End datetime.

        Returns:
            Summary of ingested bars.
        """
        total_count = 0

        for symbol in symbols:
            bars = list(self.client.get_bars(symbol, start, end))
            if bars:
                self.writer.write_bars(bars)
                total_count += len(bars)
                logger.info(
                    "bars_batch_complete",
                    symbol=symbol,
                    count=len(bars),
                )

        return DataTypeSummary(
            count=total_count,
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def backfill(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> dict[str, DataTypeSummary]:
        """Backfill historical data for symbols.

        Args:
            symbols: List of stock symbols to backfill.
            start: Start datetime.
            end: End datetime.
            data_types: List of data types to backfill.
                       Options: 'trades', 'quotes', 'bars'.
                       Defaults to all types.

        Returns:
            Summary dict with counts and date ranges for each data type.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        result: dict[str, DataTypeSummary] = {}

        if "trades" in data_types:
            result["trades"] = self._backfill_trades(symbols, start, end)

        if "quotes" in data_types:
            result["quotes"] = self._backfill_quotes(symbols, start, end)

        if "bars" in data_types:
            result["bars"] = self._backfill_bars(symbols, start, end)

        return result

    def backfill_with_progress(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> BackfillStats:
        """Backfill historical data with Rich progress display.

        Processes data DAY BY DAY with streaming writes and real-time
        progress updates showing current day, records fetched, and throughput.

        Args:
            symbols: List of stock symbols to backfill.
            start: Start datetime.
            end: End datetime.
            data_types: List of data types to backfill.

        Returns:
            BackfillStats with comprehensive statistics.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        self.stats = BackfillStats()

        # Calculate total days for progress tracking
        days = self._iter_days(start, end)
        total_days = len(days)
        # Progress is tracked per day per data type per symbol
        total_steps = total_days * len(data_types)

        # Initialize symbol stats with day tracking
        for symbol in symbols:
            self.stats.symbols[symbol] = SymbolStats(
                symbol=symbol,
                total_days=total_days,
            )

        logger.info(
            "backfill_started",
            symbols=symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            data_types=data_types,
            total_days=total_days,
        )

        console.print(
            f"\n[bold]Backfill Progress[/bold] "
            f"({total_days} days x {len(data_types)} types x {len(symbols)} symbols)"
        )

        progress = self._create_progress()

        with progress:
            for symbol in symbols:
                symbol_stats = self.stats.symbols[symbol]

                task_id = progress.add_task(
                    f"Processing {symbol}",
                    total=total_steps,
                    symbol=symbol,
                    current_day="starting...",
                    records="0",
                    throughput="0",
                )

                try:
                    # Process each day sequentially for streaming writes
                    for day_idx, (day_start, day_end) in enumerate(days):
                        day_str = day_start.strftime("%Y-%m-%d")
                        symbol_stats.current_day = day_str

                        # Update progress with current day
                        progress.update(
                            task_id,
                            current_day=day_str,
                            records=format_number(symbol_stats.total_records),
                            throughput=format_number(int(symbol_stats.throughput)),
                        )

                        log = logger.bind(symbol=symbol, day=day_str)

                        # Process each data type for this day
                        if "trades" in data_types:
                            day_trades = self._fetch_and_write_day_trades(
                                symbol, day_start, day_end, log
                            )
                            progress.update(
                                task_id,
                                records=format_number(symbol_stats.total_records),
                                throughput=format_number(int(symbol_stats.throughput)),
                            )
                            progress.advance(task_id)

                        if "quotes" in data_types:
                            day_quotes = self._fetch_and_write_day_quotes(
                                symbol, day_start, day_end, log
                            )
                            progress.update(
                                task_id,
                                records=format_number(symbol_stats.total_records),
                                throughput=format_number(int(symbol_stats.throughput)),
                            )
                            progress.advance(task_id)

                        if "bars" in data_types:
                            day_bars = self._fetch_and_write_day_bars(
                                symbol, day_start, day_end, log
                            )
                            progress.update(
                                task_id,
                                records=format_number(symbol_stats.total_records),
                                throughput=format_number(int(symbol_stats.throughput)),
                            )
                            progress.advance(task_id)

                        # Track days completed
                        symbol_stats.days_completed = day_idx + 1

                        log.debug(
                            "day_complete",
                            day=day_str,
                            days_progress=symbol_stats.days_progress,
                            total_records=symbol_stats.total_records,
                            throughput=f"{symbol_stats.throughput:.0f}/s",
                        )

                    symbol_stats.end_time = time.time()

                    logger.info(
                        "symbol_backfill_complete",
                        symbol=symbol,
                        trades=symbol_stats.trades_count,
                        quotes=symbol_stats.quotes_count,
                        bars=symbol_stats.bars_count,
                        bytes_written=format_bytes(symbol_stats.bytes_written),
                        elapsed=format_duration(symbol_stats.elapsed),
                        throughput=f"{symbol_stats.throughput:.0f}/s",
                    )

                except Exception as e:
                    logger.error(
                        "backfill_symbol_error",
                        symbol=symbol,
                        current_day=symbol_stats.current_day,
                        days_completed=symbol_stats.days_completed,
                        error=str(e),
                        exc_info=True,
                    )
                    raise

        self.stats.end_time = time.time()

        logger.info(
            "backfill_complete",
            total_trades=self.stats.total_trades,
            total_quotes=self.stats.total_quotes,
            total_bars=self.stats.total_bars,
            total_bytes=format_bytes(self.stats.total_bytes),
            elapsed=format_duration(self.stats.elapsed),
            avg_throughput=f"{self.stats.avg_throughput:.0f} records/sec",
        )

        return self.stats

    def _fetch_and_write_day_trades(
        self,
        symbol: str,
        day_start: datetime,
        day_end: datetime,
        log: Any,
    ) -> int:
        """Fetch and write trades for a single day (streaming).

        Args:
            symbol: Stock symbol.
            day_start: Day start datetime.
            day_end: Day end datetime.
            log: Bound logger with context.

        Returns:
            Number of trades written.
        """
        fetch_start = time.time()
        trades = list(self.client.get_trades(symbol, day_start, day_end))
        fetch_elapsed = time.time() - fetch_start

        if not trades:
            return 0

        write_start = time.time()
        bytes_written = self.writer.write_trades(trades)
        write_elapsed = time.time() - write_start

        # Update stats
        self.stats.symbols[symbol].trades_count += len(trades)
        self.stats.symbols[symbol].bytes_written += bytes_written

        log.debug(
            "day_trades_written",
            count=len(trades),
            bytes=format_bytes(bytes_written),
            fetch_time=f"{fetch_elapsed:.2f}s",
            write_time=f"{write_elapsed:.2f}s",
        )

        return len(trades)

    def _fetch_and_write_day_quotes(
        self,
        symbol: str,
        day_start: datetime,
        day_end: datetime,
        log: Any,
    ) -> int:
        """Fetch and write quotes for a single day (streaming).

        Args:
            symbol: Stock symbol.
            day_start: Day start datetime.
            day_end: Day end datetime.
            log: Bound logger with context.

        Returns:
            Number of quotes written.
        """
        fetch_start = time.time()
        quotes = list(self.client.get_quotes(symbol, day_start, day_end))
        fetch_elapsed = time.time() - fetch_start

        if not quotes:
            return 0

        write_start = time.time()
        bytes_written = self.writer.write_quotes(quotes)
        write_elapsed = time.time() - write_start

        # Update stats
        self.stats.symbols[symbol].quotes_count += len(quotes)
        self.stats.symbols[symbol].bytes_written += bytes_written

        log.debug(
            "day_quotes_written",
            count=len(quotes),
            bytes=format_bytes(bytes_written),
            fetch_time=f"{fetch_elapsed:.2f}s",
            write_time=f"{write_elapsed:.2f}s",
        )

        return len(quotes)

    def _fetch_and_write_day_bars(
        self,
        symbol: str,
        day_start: datetime,
        day_end: datetime,
        log: Any,
    ) -> int:
        """Fetch and write bars for a single day (streaming).

        Args:
            symbol: Stock symbol.
            day_start: Day start datetime.
            day_end: Day end datetime.
            log: Bound logger with context.

        Returns:
            Number of bars written.
        """
        fetch_start = time.time()
        bars = list(self.client.get_bars(symbol, day_start, day_end))
        fetch_elapsed = time.time() - fetch_start

        if not bars:
            return 0

        write_start = time.time()
        bytes_written = self.writer.write_bars(bars)
        write_elapsed = time.time() - write_start

        # Update stats
        self.stats.symbols[symbol].bars_count += len(bars)
        self.stats.symbols[symbol].bytes_written += bytes_written

        log.debug(
            "day_bars_written",
            count=len(bars),
            bytes=format_bytes(bytes_written),
            fetch_time=f"{fetch_elapsed:.2f}s",
            write_time=f"{write_elapsed:.2f}s",
        )

        return len(bars)

    def backfill_date_range(
        self,
        symbols: list[str],
        days: int = 30,
        data_types: list[str] | None = None,
    ) -> dict[str, DataTypeSummary]:
        """Backfill historical data for the last N days.

        Args:
            symbols: List of stock symbols.
            days: Number of days to backfill (default: 30).
            data_types: List of data types to backfill.

        Returns:
            Summary of ingested data.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        return self.backfill(symbols, start, end, data_types)

    def backfill_date_range_with_progress(
        self,
        symbols: list[str],
        days: int = 30,
        data_types: list[str] | None = None,
    ) -> BackfillStats:
        """Backfill historical data for the last N days with progress display.

        Args:
            symbols: List of stock symbols.
            days: Number of days to backfill (default: 30).
            data_types: List of data types to backfill.

        Returns:
            BackfillStats with comprehensive statistics.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        return self.backfill_with_progress(symbols, start, end, data_types)


class AsyncBackfillOrchestrator:
    """Async orchestrator for parallel historical data backfill.

    Uses AsyncAlpacaDataClient for concurrent fetching of multiple symbols
    and data types. Includes rate limiting and error handling.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lake_path: Path | str,
        feed: str = "iex",
        requests_per_minute: int = 200,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the async backfill orchestrator.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            lake_path: Path to the data lake directory.
            feed: Data feed ('iex' or 'sip').
            requests_per_minute: Rate limit (default: 200 for Alpaca).
            max_concurrent: Maximum concurrent requests (default: 10).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self.writer = ParquetWriter(base_path=lake_path)
        self.lake_path = Path(lake_path)
        self.stats = BackfillStats()

        logger.info(
            "async_backfill_orchestrator_initialized",
            lake_path=str(lake_path),
            feed=feed,
            requests_per_minute=requests_per_minute,
            max_concurrent=max_concurrent,
        )

    def _iter_days(
        self,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Generate day-by-day date ranges for streaming processing.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of (day_start, day_end) tuples for each day in range.
        """
        days: list[tuple[datetime, datetime]] = []
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)

        while current < end:
            day_start = current
            day_end = min(current + timedelta(days=1), end)
            days.append((day_start, day_end))
            current = day_end

        return days

    def _create_progress(self) -> ProgressType:
        """Create a progress tracker for backfill tracking.

        Returns Rich Progress for interactive terminals, or FileProgress
        for file-redirected output (e.g., nohup).
        """
        if is_file_mode():
            logger.debug("using_file_progress_mode")
            return FileProgress(log_interval=10.0, console=console)

        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[symbol]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]|"),
            TextColumn("[yellow]{task.fields[current_day]}[/]"),
            TextColumn("[dim]|"),
            TextColumn("[green]{task.fields[records]}[/] rec"),
            TextColumn("[dim]|"),
            TextColumn("[magenta]{task.fields[throughput]}[/]/s"),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )

    async def _write_results(
        self,
        results: dict[str, SymbolFetchResult],
        data_type: str,
    ) -> int:
        """Write fetched results to Parquet files.

        Args:
            results: Dictionary mapping symbol to fetch results.
            data_type: Type of data ('trades', 'quotes', 'bars').

        Returns:
            Total bytes written.
        """
        total_bytes = 0

        for symbol, result in results.items():
            if not result.success or not result.data:
                if not result.success:
                    logger.warning(
                        f"skipping_failed_{data_type}",
                        symbol=symbol,
                        error=result.error,
                    )
                continue

            write_start = time.time()

            if data_type == "trades":
                bytes_written = self.writer.write_trades(result.data)
            elif data_type == "quotes":
                bytes_written = self.writer.write_quotes(result.data)
            elif data_type == "bars":
                bytes_written = self.writer.write_bars(result.data)
            else:
                continue

            write_elapsed = time.time() - write_start
            total_bytes += bytes_written

            # Update stats
            if symbol in self.stats.symbols:
                if data_type == "trades":
                    self.stats.symbols[symbol].trades_count += len(result.data)
                elif data_type == "quotes":
                    self.stats.symbols[symbol].quotes_count += len(result.data)
                elif data_type == "bars":
                    self.stats.symbols[symbol].bars_count += len(result.data)
                self.stats.symbols[symbol].bytes_written += bytes_written

            logger.debug(
                f"{data_type}_written",
                symbol=symbol,
                count=len(result.data),
                bytes_written=format_bytes(bytes_written),
                write_time=f"{write_elapsed:.2f}s",
            )

        return total_bytes

    async def backfill_parallel(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> BackfillStats:
        """Backfill historical data for multiple symbols in parallel.

        Fetches all symbols concurrently for each day, respecting rate limits.
        If one symbol fails, continues with others.

        Args:
            symbols: List of stock symbols to backfill.
            start: Start datetime.
            end: End datetime.
            data_types: List of data types to backfill.

        Returns:
            BackfillStats with comprehensive statistics.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        self.stats = BackfillStats()
        days = self._iter_days(start, end)
        total_days = len(days)

        # Initialize symbol stats
        for symbol in symbols:
            self.stats.symbols[symbol] = SymbolStats(
                symbol=symbol,
                total_days=total_days,
            )

        logger.info(
            "parallel_backfill_started",
            symbols=symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            data_types=data_types,
            total_days=total_days,
            mode="parallel",
        )

        console.print(
            f"\n[bold]Parallel Backfill[/bold] "
            f"({len(symbols)} symbols x {total_days} days x {len(data_types)} types)"
        )

        progress = self._create_progress()

        async with AsyncAlpacaDataClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            feed=self.feed,
            requests_per_minute=self.requests_per_minute,
            max_concurrent=self.max_concurrent,
        ) as client:
            with progress:
                # Create overall task
                overall_task = progress.add_task(
                    "Overall",
                    total=total_days * len(data_types),
                    symbol="ALL",
                    current_day="starting...",
                    records="0",
                    throughput="0",
                )

                for day_idx, (day_start, day_end) in enumerate(days):
                    day_str = day_start.strftime("%Y-%m-%d")

                    # Update all symbol stats with current day
                    for symbol in symbols:
                        self.stats.symbols[symbol].current_day = day_str

                    progress.update(
                        overall_task,
                        current_day=day_str,
                        records=format_number(self.stats.total_records),
                        throughput=format_number(int(self.stats.avg_throughput)),
                    )

                    # Fetch each data type for all symbols in parallel
                    for dtype in data_types:
                        fetch_start = time.time()

                        try:
                            if dtype == "trades":
                                results = await client.get_trades_multi(
                                    symbols, day_start, day_end
                                )
                            elif dtype == "quotes":
                                results = await client.get_quotes_multi(
                                    symbols, day_start, day_end
                                )
                            elif dtype == "bars":
                                results = await client.get_bars_multi(
                                    symbols, day_start, day_end
                                )
                            else:
                                continue

                            # Write results
                            await self._write_results(results, dtype)

                            fetch_elapsed = time.time() - fetch_start
                            total_count = sum(
                                len(r.data) for r in results.values() if r.success
                            )
                            failed_count = sum(
                                1 for r in results.values() if not r.success
                            )

                            logger.info(
                                f"parallel_{dtype}_batch_complete",
                                day=day_str,
                                symbols_count=len(symbols),
                                records_count=total_count,
                                failed_count=failed_count,
                                elapsed=f"{fetch_elapsed:.2f}s",
                            )

                        except Exception as e:
                            logger.error(
                                f"parallel_{dtype}_batch_failed",
                                day=day_str,
                                error=str(e),
                                exc_info=True,
                            )
                            # Continue with next data type

                        progress.advance(overall_task)
                        progress.update(
                            overall_task,
                            records=format_number(self.stats.total_records),
                            throughput=format_number(int(self.stats.avg_throughput)),
                        )

                    # Update days completed for all symbols
                    for symbol in symbols:
                        self.stats.symbols[symbol].days_completed = day_idx + 1

        # Finalize stats
        self.stats.end_time = time.time()
        for symbol in symbols:
            self.stats.symbols[symbol].end_time = time.time()

        logger.info(
            "parallel_backfill_complete",
            total_trades=self.stats.total_trades,
            total_quotes=self.stats.total_quotes,
            total_bars=self.stats.total_bars,
            total_bytes=format_bytes(self.stats.total_bytes),
            elapsed=format_duration(self.stats.elapsed),
            avg_throughput=f"{self.stats.avg_throughput:.0f} records/sec",
            mode="parallel",
        )

        return self.stats

    async def backfill_date_range_parallel(
        self,
        symbols: list[str],
        days: int = 30,
        data_types: list[str] | None = None,
    ) -> BackfillStats:
        """Backfill historical data for the last N days in parallel.

        Args:
            symbols: List of stock symbols.
            days: Number of days to backfill (default: 30).
            data_types: List of data types to backfill.

        Returns:
            BackfillStats with comprehensive statistics.
        """
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        return await self.backfill_parallel(symbols, start, end, data_types)

    async def backfill_symbol_all_types(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
    ) -> SymbolStats:
        """Backfill all data types for a single symbol concurrently.

        Fetches trades, quotes, and bars in parallel for each day.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime.
            end: End datetime.
            data_types: List of data types to backfill.

        Returns:
            SymbolStats for the symbol.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        days = self._iter_days(start, end)
        stats = SymbolStats(symbol=symbol, total_days=len(days))

        logger.info(
            "parallel_symbol_backfill_started",
            symbol=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            data_types=data_types,
        )

        async with AsyncAlpacaDataClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            feed=self.feed,
            requests_per_minute=self.requests_per_minute,
            max_concurrent=self.max_concurrent,
        ) as client:
            for day_idx, (day_start, day_end) in enumerate(days):
                day_str = day_start.strftime("%Y-%m-%d")
                stats.current_day = day_str

                try:
                    # Fetch all data types concurrently
                    results = await client.get_all_data_for_symbol(
                        symbol, day_start, day_end, data_types
                    )

                    # Write results
                    for dtype, result in results.items():
                        if not result.success or not result.data:
                            continue

                        if dtype == "trades":
                            bytes_written = self.writer.write_trades(result.data)
                            stats.trades_count += len(result.data)
                        elif dtype == "quotes":
                            bytes_written = self.writer.write_quotes(result.data)
                            stats.quotes_count += len(result.data)
                        elif dtype == "bars":
                            bytes_written = self.writer.write_bars(result.data)
                            stats.bars_count += len(result.data)
                        else:
                            continue

                        stats.bytes_written += bytes_written

                except Exception as e:
                    logger.error(
                        "parallel_symbol_day_failed",
                        symbol=symbol,
                        day=day_str,
                        error=str(e),
                    )
                    # Continue with next day

                stats.days_completed = day_idx + 1

        stats.end_time = time.time()

        logger.info(
            "parallel_symbol_backfill_complete",
            symbol=symbol,
            trades=stats.trades_count,
            quotes=stats.quotes_count,
            bars=stats.bars_count,
            bytes_written=format_bytes(stats.bytes_written),
            elapsed=format_duration(stats.elapsed),
        )

        return stats
