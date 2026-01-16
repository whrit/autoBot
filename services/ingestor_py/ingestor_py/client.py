"""
Alpaca Data Client - Historical data fetching via REST API.

Uses alpaca-py SDK for market data access.
Supports trades, quotes, and bars with configurable feeds (IEX or SIP).

Data is fetched in daily chunks to maintain constant memory usage
instead of loading entire date ranges at once.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame

from ingestor_py.logging_config import get_logger

logger = get_logger(__name__)


class AlpacaDataClient:
    """Client for fetching historical market data from Alpaca.

    Attributes:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        feed: Data feed to use ('iex' for free, 'sip' for paid).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        feed: str = "iex",
    ) -> None:
        """Initialize the Alpaca data client.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            feed: Data feed ('iex' or 'sip').
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed

        self._client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

    def _chunk_date_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[tuple[datetime, datetime]]:
        """Split a date range into daily chunks.

        Yields consecutive daily (start, end) tuples from the overall start
        to the overall end. Each chunk covers one calendar day, which helps
        maintain constant memory usage when fetching large date ranges.

        Args:
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Yields:
            Tuples of (chunk_start, chunk_end) representing daily intervals.
        """
        current = start
        while current < end:
            # End of current day or overall end, whichever comes first
            next_day = current.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            chunk_end = min(next_day, end)
            yield current, chunk_end
            current = chunk_end

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[dict[str, object]]:
        """Fetch historical trades for a symbol.

        Data is fetched in daily chunks to maintain constant memory usage.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Yields:
            Trade records with ts_event and ts_recv timestamps.
        """
        for chunk_start, chunk_end in self._chunk_date_range(start, end):
            logger.debug(
                "fetching_trades_chunk",
                symbol=symbol,
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
            )

            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=chunk_start,
                end=chunk_end,
                feed=self.feed,  # type: ignore[arg-type]
            )

            response = self._client.get_stock_trades(request)

            if not response or symbol not in response:
                continue

            ts_recv = datetime.now(UTC)

            for trade in response[symbol]:
                yield {
                    "symbol": symbol,
                    "ts_event": trade.timestamp,
                    "ts_recv": ts_recv,
                    "price": float(trade.price),
                    "size": float(trade.size),
                    "exchange": trade.exchange,
                    "conditions": (
                        " ".join(trade.conditions) if trade.conditions else ""
                    ),
                }

    def get_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[dict[str, object]]:
        """Fetch historical quotes for a symbol.

        Data is fetched in daily chunks to maintain constant memory usage.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Yields:
            Quote records with ts_event and ts_recv timestamps.
        """
        for chunk_start, chunk_end in self._chunk_date_range(start, end):
            logger.debug(
                "fetching_quotes_chunk",
                symbol=symbol,
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
            )

            request = StockQuotesRequest(
                symbol_or_symbols=symbol,
                start=chunk_start,
                end=chunk_end,
                feed=self.feed,  # type: ignore[arg-type]
            )

            response = self._client.get_stock_quotes(request)

            if not response or symbol not in response:
                continue

            ts_recv = datetime.now(UTC)

            for quote in response[symbol]:
                yield {
                    "symbol": symbol,
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

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
    ) -> Iterator[dict[str, object]]:
        """Fetch historical bars for a symbol.

        Data is fetched in daily chunks to maintain constant memory usage.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            timeframe: Bar timeframe (default: 1 minute).

        Yields:
            Bar records with ts_event and ts_recv timestamps.
        """
        for chunk_start, chunk_end in self._chunk_date_range(start, end):
            logger.debug(
                "fetching_bars_chunk",
                symbol=symbol,
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
                timeframe=str(timeframe),
            )

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=chunk_start,
                end=chunk_end,
                timeframe=timeframe,
                feed=self.feed,  # type: ignore[arg-type]
            )

            response = self._client.get_stock_bars(request)

            if not response or symbol not in response:
                continue

            ts_recv = datetime.now(UTC)

            for bar in response[symbol]:
                yield {
                    "symbol": symbol,
                    "ts_event": bar.timestamp,
                    "ts_recv": ts_recv,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "trade_count": int(bar.trade_count) if bar.trade_count else 0,
                    "vwap": float(bar.vwap) if bar.vwap else 0.0,
                }
