"""
Backfill Orchestrator - Coordinate historical data ingestion.

Handles downloading historical trades, quotes, and bars from Alpaca
and writing them to partitioned Parquet files with comprehensive
logging and progress tracking.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from ingestor_py.client import AlpacaDataClient
from ingestor_py.logging_config import (
    console,
    format_bytes,
    format_duration,
    format_number,
    get_logger,
)
from ingestor_py.writer import ParquetWriter

logger = get_logger(__name__)


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

    def _create_progress(self) -> Progress:
        """Create a Rich progress bar for backfill tracking."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[symbol]}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]|"),
            TextColumn("[green]{task.fields[trades]}[/] trades"),
            TextColumn("[dim]|"),
            TextColumn("[cyan]{task.fields[quotes]}[/] quotes"),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )

    def _backfill_symbol_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        progress: Progress | None = None,
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
        progress: Progress | None = None,
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
        num_data_types = len(data_types)

        # Initialize symbol stats
        for symbol in symbols:
            self.stats.symbols[symbol] = SymbolStats(symbol=symbol)

        logger.info(
            "backfill_started",
            symbols=symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            data_types=data_types,
        )

        console.print("\n[bold]Backfill Progress[/bold]")

        progress = self._create_progress()

        with progress:
            for symbol in symbols:
                task_id = progress.add_task(
                    f"Processing {symbol}",
                    total=num_data_types,
                    symbol=symbol,
                    trades="0",
                    quotes="0",
                )

                try:
                    if "trades" in data_types:
                        self._backfill_symbol_trades(
                            symbol, start, end, progress, task_id
                        )
                        progress.advance(task_id)

                    if "quotes" in data_types:
                        self._backfill_symbol_quotes(
                            symbol, start, end, progress, task_id
                        )
                        progress.advance(task_id)

                    if "bars" in data_types:
                        self._backfill_symbol_bars(symbol, start, end)
                        progress.advance(task_id)

                    self.stats.symbols[symbol].end_time = time.time()

                except Exception as e:
                    logger.error(
                        "backfill_symbol_error",
                        symbol=symbol,
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
            total_bytes=self.stats.total_bytes,
            elapsed=format_duration(self.stats.elapsed),
            avg_throughput=f"{self.stats.avg_throughput:.0f} records/sec",
        )

        return self.stats

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
