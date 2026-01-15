"""
Incremental Processor (T2.05).

Supports incremental updates without full recomputation.
Maintains state for efficient processing of streaming data.
"""

from datetime import datetime
from typing import Any

import polars as pl

from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.micro_bars import MicrostructureBarBuilder


class IncrementalProcessor:
    """
    Process trade and quote data incrementally.

    Maintains state for each symbol to enable efficient updates
    without full recomputation.
    """

    def __init__(self) -> None:
        """Initialize the incremental processor."""
        self._state: dict[str, dict[str, Any]] = {}
        self._bar_builder = StandardBarBuilder(granularity="1m")
        self._micro_builder = MicrostructureBarBuilder(granularity="30s")
        # Buffers for partial bar data
        self._trade_buffers: dict[str, list[dict[str, Any]]] = {}
        self._quote_buffers: dict[str, list[dict[str, Any]]] = {}

    def process_trades(
        self, trades: pl.DataFrame, symbol: str
    ) -> pl.DataFrame | None:
        """
        Process trades incrementally.

        Args:
            trades: DataFrame with new trade data.
            symbol: Stock symbol.

        Returns:
            DataFrame with newly computed bars, or None if no complete bars.
        """
        if trades.is_empty():
            return None

        # Initialize state if needed
        if symbol not in self._state:
            self._initialize_state(symbol)

        # Get watermark (last processed timestamp)
        watermark = self._state[symbol].get("last_processed_ts")

        # Filter out already processed data
        if watermark is not None:
            new_trades = trades.filter(pl.col("ts_event") > watermark)
        else:
            new_trades = trades

        if new_trades.is_empty():
            return None

        # Update watermark
        max_ts = new_trades["ts_event"].max()
        self._state[symbol]["last_processed_ts"] = max_ts

        # Add to buffer
        if symbol not in self._trade_buffers:
            self._trade_buffers[symbol] = []

        # Convert to list of dicts and add to buffer
        for row in new_trades.iter_rows(named=True):
            self._trade_buffers[symbol].append(row)

        # Build bars from buffered data
        buffer_df = pl.DataFrame(self._trade_buffers[symbol])
        if buffer_df.is_empty():
            return None

        bars = self._bar_builder.build(buffer_df, symbol=symbol)

        if bars.is_empty():
            return None

        # Keep only incomplete bar data in buffer
        self._prune_trade_buffer(symbol, bars)

        # Update state with computed bars
        self._state[symbol]["bars"] = bars
        self._state[symbol]["symbol"] = symbol

        return bars

    def process_quotes(
        self, quotes: pl.DataFrame, symbol: str
    ) -> pl.DataFrame | None:
        """
        Process quotes incrementally.

        Args:
            quotes: DataFrame with new quote data.
            symbol: Stock symbol.

        Returns:
            DataFrame with newly computed micro bars, or None if no complete bars.
        """
        if quotes.is_empty():
            return None

        # Initialize state if needed
        if symbol not in self._state:
            self._initialize_state(symbol)

        # Get watermark
        watermark = self._state[symbol].get("last_quote_ts")

        # Filter out already processed data
        if watermark is not None:
            new_quotes = quotes.filter(pl.col("ts_event") > watermark)
        else:
            new_quotes = quotes

        if new_quotes.is_empty():
            return None

        # Update watermark
        max_ts = new_quotes["ts_event"].max()
        self._state[symbol]["last_quote_ts"] = max_ts

        # Add to buffer
        if symbol not in self._quote_buffers:
            self._quote_buffers[symbol] = []

        for row in new_quotes.iter_rows(named=True):
            self._quote_buffers[symbol].append(row)

        # Get trade buffer for micro bar computation
        trade_buffer = self._trade_buffers.get(symbol, [])
        trade_df = pl.DataFrame(trade_buffer) if trade_buffer else pl.DataFrame()

        quote_df = pl.DataFrame(self._quote_buffers[symbol])
        if quote_df.is_empty():
            return None

        micro_bars = self._micro_builder.build(trade_df, quote_df, symbol=symbol)

        if micro_bars.is_empty():
            return None

        # Prune quote buffer
        self._prune_quote_buffer(symbol, micro_bars)

        # Update state
        self._state[symbol]["micro_bars"] = micro_bars

        return micro_bars

    def get_state(self, symbol: str) -> dict[str, Any] | None:
        """
        Get the current state for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            State dictionary or None if no state exists.
        """
        return self._state.get(symbol)

    def get_watermark(self, symbol: str) -> datetime | None:
        """
        Get the last processed timestamp for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Last processed timestamp or None.
        """
        state = self._state.get(symbol)
        if state is None:
            return None
        return state.get("last_processed_ts")

    def reset(self, symbol: str) -> None:
        """
        Reset the state for a symbol.

        Args:
            symbol: Stock symbol.
        """
        if symbol in self._state:
            self._state[symbol] = {}
        if symbol in self._trade_buffers:
            self._trade_buffers[symbol] = []
        if symbol in self._quote_buffers:
            self._quote_buffers[symbol] = []

    def reset_all(self) -> None:
        """Reset all state."""
        self._state.clear()
        self._trade_buffers.clear()
        self._quote_buffers.clear()

    def _initialize_state(self, symbol: str) -> None:
        """Initialize state for a new symbol."""
        self._state[symbol] = {
            "symbol": symbol,
            "last_processed_ts": None,
            "last_quote_ts": None,
            "bars": None,
            "micro_bars": None,
        }

    def _prune_trade_buffer(
        self, symbol: str, completed_bars: pl.DataFrame
    ) -> None:
        """Remove completed bar data from trade buffer."""
        if completed_bars.is_empty():
            return

        # Get the end time of the last completed bar
        last_bar_end = completed_bars["bar_end"].max()

        # Keep only trades after the last completed bar
        if symbol in self._trade_buffers:
            self._trade_buffers[symbol] = [
                t for t in self._trade_buffers[symbol]
                if t["ts_event"] >= last_bar_end
            ]

    def _prune_quote_buffer(
        self, symbol: str, completed_bars: pl.DataFrame
    ) -> None:
        """Remove completed bar data from quote buffer."""
        if completed_bars.is_empty():
            return

        # Get the end time of the last completed bar
        last_bar_end = completed_bars["bar_end"].max()

        # Keep only quotes after the last completed bar
        if symbol in self._quote_buffers:
            self._quote_buffers[symbol] = [
                q for q in self._quote_buffers[symbol]
                if q["ts_event"] >= last_bar_end
            ]

    def get_buffered_trades(self, symbol: str) -> pl.DataFrame:
        """Get buffered trades for a symbol."""
        buffer = self._trade_buffers.get(symbol, [])
        if not buffer:
            return pl.DataFrame()
        return pl.DataFrame(buffer)

    def get_buffered_quotes(self, symbol: str) -> pl.DataFrame:
        """Get buffered quotes for a symbol."""
        buffer = self._quote_buffers.get(symbol, [])
        if not buffer:
            return pl.DataFrame()
        return pl.DataFrame(buffer)
