"""
Backfill Orchestrator - Coordinate historical data ingestion.

Handles downloading historical trades, quotes, and bars from Alpaca
and writing them to partitioned Parquet files.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from ingestor_py.client import AlpacaDataClient
from ingestor_py.writer import ParquetWriter


class DataTypeSummary(TypedDict):
    """Summary of ingested data for a single data type."""

    count: int
    start: str
    end: str


class BackfillOrchestrator:
    """Orchestrate historical data backfill.

    Coordinates between AlpacaDataClient for fetching data
    and ParquetWriter for persisting to the data lake.
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
