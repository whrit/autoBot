"""
Alpaca Data Client - Historical data fetching via REST API.

Uses alpaca-py SDK for market data access.
Supports trades, quotes, and bars with configurable feeds (IEX or SIP).

Data is fetched in daily chunks to maintain constant memory usage
instead of loading entire date ranges at once.

Also includes AsyncAlpacaDataClient for concurrent fetching using httpx.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame

from ingestor_py.logging_config import get_logger

logger = get_logger(__name__)

# Alpaca API base URL for market data
ALPACA_DATA_API_BASE = "https://data.alpaca.markets/v2"


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API calls.

    Alpaca has a rate limit of 200 requests per minute.
    This limiter uses a token bucket algorithm to smooth out requests.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute.
        tokens: Current available tokens.
        last_refill: Timestamp of last token refill.
    """

    requests_per_minute: int = 200
    tokens: float = field(default=200.0)
    last_refill: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        Uses token bucket algorithm to rate limit requests.
        """
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
            self.last_refill = now

            if self.tokens < 1:
                # Wait for token to become available
                wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)
                logger.debug("rate_limit_wait", wait_seconds=f"{wait_time:.2f}")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


@dataclass
class SymbolFetchResult:
    """Result of fetching data for a single symbol.

    Attributes:
        symbol: Stock ticker symbol.
        data: List of data records.
        success: Whether the fetch was successful.
        error: Error message if fetch failed.
        elapsed: Time taken in seconds.
    """

    symbol: str
    data: list[dict[str, Any]]
    success: bool = True
    error: str | None = None
    elapsed: float = 0.0


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

            if not response:
                continue

            # Note: alpaca-py BarSet/TradeSet/QuoteSet have broken __contains__
            # so we use try/except instead of "symbol in response"
            try:
                trades_data = response[symbol]
                if not trades_data:
                    continue
            except (KeyError, TypeError):
                continue

            ts_recv = datetime.now(UTC)

            for trade in trades_data:
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

            if not response:
                continue

            # Note: alpaca-py QuoteSet has broken __contains__
            try:
                quotes_data = response[symbol]
                if not quotes_data:
                    continue
            except (KeyError, TypeError):
                continue

            ts_recv = datetime.now(UTC)

            for quote in quotes_data:
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

            if not response:
                continue

            # Note: alpaca-py BarSet has broken __contains__
            try:
                bars_data = response[symbol]
                if not bars_data:
                    continue
            except (KeyError, TypeError):
                continue

            ts_recv = datetime.now(UTC)

            for bar in bars_data:
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


class AsyncAlpacaDataClient:
    """Async client for concurrent fetching of historical market data from Alpaca.

    Uses httpx for async HTTP requests and includes built-in rate limiting.
    Designed for parallel processing of multiple symbols.

    Attributes:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        feed: Data feed to use ('iex' for free, 'sip' for paid).
        rate_limiter: Rate limiter instance.
        max_concurrent: Maximum concurrent requests per batch.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        feed: str = "iex",
        requests_per_minute: int = 200,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the async Alpaca data client.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            feed: Data feed ('iex' or 'sip').
            requests_per_minute: Rate limit (default: 200 for Alpaca).
            max_concurrent: Maximum concurrent requests (default: 10).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=ALPACA_DATA_API_BASE,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncAlpacaDataClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for Alpaca API.

        Args:
            dt: Datetime to format.

        Returns:
            RFC3339 formatted string.
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def _fetch_with_retry(
        self,
        endpoint: str,
        params: dict[str, Any],
        retries: int = 3,
    ) -> dict[str, Any]:
        """Fetch data with rate limiting and retry logic.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            retries: Number of retry attempts.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.HTTPError: If request fails after all retries.
        """
        client = await self._get_client()

        for attempt in range(retries):
            await self.rate_limiter.acquire()
            async with self._semaphore:
                try:
                    response = await client.get(endpoint, params=params)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(
                            "rate_limited",
                            attempt=attempt + 1,
                            wait_seconds=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    if attempt == retries - 1:
                        raise
                except httpx.TimeoutException:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(1)

        # This shouldn't be reached, but satisfy type checker
        return {}

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch historical trades for a symbol asynchronously.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Returns:
            List of trade records.
        """
        all_trades: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "start": self._format_datetime(start),
                "end": self._format_datetime(end),
                "feed": self.feed,
                "limit": 10000,
            }
            if next_page_token:
                params["page_token"] = next_page_token

            logger.debug(
                "async_fetching_trades",
                symbol=symbol,
                start=params["start"],
                end=params["end"],
            )

            data = await self._fetch_with_retry(f"/stocks/{symbol}/trades", params)

            trades = data.get("trades", [])
            ts_recv = datetime.now(UTC)

            for trade in trades:
                all_trades.append({
                    "symbol": symbol,
                    "ts_event": datetime.fromisoformat(
                        trade["t"].replace("Z", "+00:00")
                    ),
                    "ts_recv": ts_recv,
                    "price": float(trade["p"]),
                    "size": float(trade["s"]),
                    "exchange": trade.get("x", ""),
                    "conditions": " ".join(trade.get("c", [])),
                })

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        return all_trades

    async def get_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch historical quotes for a symbol asynchronously.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Returns:
            List of quote records.
        """
        all_quotes: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "start": self._format_datetime(start),
                "end": self._format_datetime(end),
                "feed": self.feed,
                "limit": 10000,
            }
            if next_page_token:
                params["page_token"] = next_page_token

            logger.debug(
                "async_fetching_quotes",
                symbol=symbol,
                start=params["start"],
                end=params["end"],
            )

            data = await self._fetch_with_retry(f"/stocks/{symbol}/quotes", params)

            quotes = data.get("quotes", [])
            ts_recv = datetime.now(UTC)

            for quote in quotes:
                all_quotes.append({
                    "symbol": symbol,
                    "ts_event": datetime.fromisoformat(
                        quote["t"].replace("Z", "+00:00")
                    ),
                    "ts_recv": ts_recv,
                    "bid_price": float(quote.get("bp", 0)),
                    "bid_size": float(quote.get("bs", 0)),
                    "ask_price": float(quote.get("ap", 0)),
                    "ask_size": float(quote.get("as", 0)),
                    "bid_exchange": quote.get("bx", ""),
                    "ask_exchange": quote.get("ax", ""),
                    "conditions": " ".join(quote.get("c", [])),
                })

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        return all_quotes

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> list[dict[str, Any]]:
        """Fetch historical bars for a symbol asynchronously.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            timeframe: Bar timeframe (default: '1Min').

        Returns:
            List of bar records.
        """
        all_bars: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "start": self._format_datetime(start),
                "end": self._format_datetime(end),
                "feed": self.feed,
                "timeframe": timeframe,
                "limit": 10000,
            }
            if next_page_token:
                params["page_token"] = next_page_token

            logger.debug(
                "async_fetching_bars",
                symbol=symbol,
                start=params["start"],
                end=params["end"],
                timeframe=timeframe,
            )

            data = await self._fetch_with_retry(f"/stocks/{symbol}/bars", params)

            bars = data.get("bars", [])
            ts_recv = datetime.now(UTC)

            for bar in bars:
                all_bars.append({
                    "symbol": symbol,
                    "ts_event": datetime.fromisoformat(
                        bar["t"].replace("Z", "+00:00")
                    ),
                    "ts_recv": ts_recv,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                    "trade_count": int(bar.get("n", 0)),
                    "vwap": float(bar.get("vw", 0)),
                })

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        return all_bars

    async def get_trades_multi(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, SymbolFetchResult]:
        """Fetch trades for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Returns:
            Dictionary mapping symbol to SymbolFetchResult.
        """
        return await self._fetch_multi(
            symbols,
            start,
            end,
            data_type="trades",
        )

    async def get_quotes_multi(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, SymbolFetchResult]:
        """Fetch quotes for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).

        Returns:
            Dictionary mapping symbol to SymbolFetchResult.
        """
        return await self._fetch_multi(
            symbols,
            start,
            end,
            data_type="quotes",
        )

    async def get_bars_multi(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> dict[str, SymbolFetchResult]:
        """Fetch bars for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            timeframe: Bar timeframe (default: '1Min').

        Returns:
            Dictionary mapping symbol to SymbolFetchResult.
        """
        return await self._fetch_multi(
            symbols,
            start,
            end,
            data_type="bars",
            timeframe=timeframe,
        )

    async def _fetch_multi(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        data_type: Literal["trades", "quotes", "bars"],
        timeframe: str = "1Min",
    ) -> dict[str, SymbolFetchResult]:
        """Fetch data for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            data_type: Type of data to fetch.
            timeframe: Bar timeframe (for bars only).

        Returns:
            Dictionary mapping symbol to SymbolFetchResult.
        """
        async def fetch_one(symbol: str) -> SymbolFetchResult:
            fetch_start = time.time()
            try:
                if data_type == "trades":
                    data = await self.get_trades(symbol, start, end)
                elif data_type == "quotes":
                    data = await self.get_quotes(symbol, start, end)
                else:
                    data = await self.get_bars(symbol, start, end, timeframe)

                elapsed = time.time() - fetch_start
                logger.debug(
                    f"async_{data_type}_fetched",
                    symbol=symbol,
                    count=len(data),
                    elapsed=f"{elapsed:.2f}s",
                )
                return SymbolFetchResult(
                    symbol=symbol,
                    data=data,
                    success=True,
                    elapsed=elapsed,
                )
            except Exception as e:
                elapsed = time.time() - fetch_start
                logger.error(
                    f"async_{data_type}_failed",
                    symbol=symbol,
                    error=str(e),
                    elapsed=f"{elapsed:.2f}s",
                )
                return SymbolFetchResult(
                    symbol=symbol,
                    data=[],
                    success=False,
                    error=str(e),
                    elapsed=elapsed,
                )

        # Use TaskGroup for structured concurrency (Python 3.11+)
        results: dict[str, SymbolFetchResult] = {}
        async with asyncio.TaskGroup() as tg:
            tasks = {symbol: tg.create_task(fetch_one(symbol)) for symbol in symbols}

        for symbol, task in tasks.items():
            results[symbol] = task.result()

        return results

    async def get_all_data_for_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_types: list[str] | None = None,
        timeframe: str = "1Min",
    ) -> dict[str, SymbolFetchResult]:
        """Fetch all data types for a single symbol concurrently.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime (timezone-aware).
            end: End datetime (timezone-aware).
            data_types: List of data types ('trades', 'quotes', 'bars').
            timeframe: Bar timeframe (for bars).

        Returns:
            Dictionary mapping data type to SymbolFetchResult.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        results: dict[str, SymbolFetchResult] = {}

        async def fetch_type(dtype: str) -> tuple[str, SymbolFetchResult]:
            fetch_start = time.time()
            try:
                if dtype == "trades":
                    data = await self.get_trades(symbol, start, end)
                elif dtype == "quotes":
                    data = await self.get_quotes(symbol, start, end)
                elif dtype == "bars":
                    data = await self.get_bars(symbol, start, end, timeframe)
                else:
                    raise ValueError(f"Unknown data type: {dtype}")

                elapsed = time.time() - fetch_start
                return dtype, SymbolFetchResult(
                    symbol=symbol,
                    data=data,
                    success=True,
                    elapsed=elapsed,
                )
            except Exception as e:
                elapsed = time.time() - fetch_start
                return dtype, SymbolFetchResult(
                    symbol=symbol,
                    data=[],
                    success=False,
                    error=str(e),
                    elapsed=elapsed,
                )

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch_type(dtype)) for dtype in data_types]

        for task in tasks:
            dtype, result = task.result()
            results[dtype] = result

        return results
