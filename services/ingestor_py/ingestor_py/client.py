"""
Alpaca Data Client - Historical data fetching via REST API.

Uses alpaca-py SDK for market data access.
Supports trades, quotes, and bars with configurable feeds (IEX or SIP).
"""

from collections.abc import Iterator
from datetime import UTC, datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame


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

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[dict[str, object]]:
        """Fetch historical trades for a symbol.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Yields:
            Trade records with ts_event and ts_recv timestamps.
        """
        request = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            feed=self.feed,  # type: ignore[arg-type]
        )

        response = self._client.get_stock_trades(request)

        if not response or symbol not in response:
            return

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

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Yields:
            Quote records with ts_event and ts_recv timestamps.
        """
        request = StockQuotesRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            feed=self.feed,  # type: ignore[arg-type]
        )

        response = self._client.get_stock_quotes(request)

        if not response or symbol not in response:
            return

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

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            timeframe: Bar timeframe (default: 1 minute).

        Yields:
            Bar records with ts_event and ts_recv timestamps.
        """
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            feed=self.feed,  # type: ignore[arg-type]
        )

        response = self._client.get_stock_bars(request)

        if not response or symbol not in response:
            return

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
