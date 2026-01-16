"""
Microstructure Bar Builder (T2.02).

Builds high-frequency microstructure features at 5s, 15s, and 30s intervals.
Features: spread, midprice, microprice, quote_imbalance, trade_imbalance, realized_vol.
"""

import sys
import time
from datetime import timedelta
from typing import Callable

import polars as pl
import structlog

# Valid granularities for microstructure bars
VALID_GRANULARITIES = {"5s", "15s", "30s"}

# Granularity to timedelta mapping
GRANULARITY_MAP = {
    "5s": timedelta(seconds=5),
    "15s": timedelta(seconds=15),
    "30s": timedelta(seconds=30),
}

logger = structlog.get_logger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


class MicrostructureBarBuilder:
    """
    Build microstructure bars from trade and quote data.

    Supports 5s, 15s, and 30s granularities.
    Computes: vwap, midprice, microprice, spread, quote_imbalance, trade_imbalance, realized_vol.

    The trade_imbalance feature measures the imbalance between buyer-initiated
    and seller-initiated trades using the tick rule:
        trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    where buy_volume is the volume of uptick trades and sell_volume is the volume
    of downtick trades. Values range from -1 (all sells) to 1 (all buys).
    """

    def __init__(self, granularity: str = "30s") -> None:
        """
        Initialize the microstructure bar builder.

        Args:
            granularity: Bar interval ('5s', '15s', '30s').

        Raises:
            ValueError: If granularity is not valid.
        """
        if granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid granularity '{granularity}'. "
                f"Must be one of: {VALID_GRANULARITIES}"
            )
        self.granularity = granularity
        self.interval = GRANULARITY_MAP[granularity]
        self._log = logger.bind(
            builder="MicrostructureBarBuilder",
            granularity=granularity,
        )
        self._log.info("initialized")

    def build(
        self,
        trades: pl.DataFrame,
        quotes: pl.DataFrame,
        symbol: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """
        Build microstructure bars from trade and quote data.

        Args:
            trades: DataFrame with trade data.
            quotes: DataFrame with quote data.
            symbol: Stock symbol.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            DataFrame with microstructure bar data.
        """
        start_time = time.perf_counter()
        start_memory = get_memory_usage_mb()
        trade_rows = len(trades)
        quote_rows = len(quotes)
        total_input_rows = trade_rows + quote_rows

        self._log.info(
            "build_started",
            symbol=symbol,
            trade_rows=trade_rows,
            quote_rows=quote_rows,
            timeframe=self.granularity,
        )

        if trades.is_empty() and quotes.is_empty():
            self._log.warning("empty_input", symbol=symbol)
            return self._empty_micro_bars_df()

        # Ensure timestamps are timezone-aware
        trades = self._ensure_utc(trades, "ts_event")
        quotes = self._ensure_utc(quotes, "ts_event")
        if progress_callback:
            progress_callback(1, 4)

        # Get interval string for polars
        interval_str = self._get_interval_str()

        # Compute quote-based features
        self._log.debug("building_quote_bars", quote_rows=quote_rows)
        quote_bars = self._build_quote_bars(quotes, interval_str)
        if progress_callback:
            progress_callback(2, 4)

        # Compute trade-based features
        self._log.debug("building_trade_bars", trade_rows=trade_rows)
        trade_bars = self._build_trade_bars(trades, interval_str)
        if progress_callback:
            progress_callback(3, 4)

        # Join quote and trade bars
        if quote_bars.is_empty():
            return self._empty_micro_bars_df()

        if not trade_bars.is_empty():
            bars = quote_bars.join(
                trade_bars,
                on="bar_start",
                how="left",
            )
        else:
            bars = quote_bars.with_columns([
                pl.lit(0.0).alias("vwap"),
                pl.lit(0.0).alias("trade_volume"),
                pl.lit(0.0).alias("realized_vol"),
                pl.lit(0.0).alias("trade_imbalance"),
            ])

        # Add symbol and bar_end
        bars = bars.with_columns([
            pl.lit(symbol).alias("symbol"),
            (pl.col("bar_start") + self.interval).alias("bar_end"),
        ])

        # Fill nulls with defaults
        bars = bars.with_columns([
            pl.col("vwap").fill_null(pl.col("midprice")),
            pl.col("trade_volume").fill_null(0.0),
            pl.col("realized_vol").fill_null(0.0),
            pl.col("trade_imbalance").fill_null(0.0),
        ])
        if progress_callback:
            progress_callback(4, 4)

        # Reorder columns to match schema
        result = bars.select([
            "symbol",
            "bar_start",
            "bar_end",
            "vwap",
            "midprice",
            "microprice",
            "spread",
            "bid_size",
            "ask_size",
            "quote_imbalance",
            "trade_imbalance",
            "trade_volume",
            "realized_vol",
        ])

        elapsed = time.perf_counter() - start_time
        end_memory = get_memory_usage_mb()
        output_bars = len(result)
        throughput = total_input_rows / elapsed if elapsed > 0 else 0

        features_computed = [
            "vwap", "midprice", "microprice", "spread",
            "bid_size", "ask_size", "quote_imbalance",
            "trade_imbalance", "trade_volume", "realized_vol",
        ]

        self._log.info(
            "build_completed",
            symbol=symbol,
            timeframe=self.granularity,
            trade_rows=trade_rows,
            quote_rows=quote_rows,
            output_bars=output_bars,
            features_computed=len(features_computed),
            processing_time_sec=round(elapsed, 3),
            throughput_rows_per_sec=round(throughput, 1),
            memory_start_mb=round(start_memory, 2),
            memory_end_mb=round(end_memory, 2),
            memory_delta_mb=round(end_memory - start_memory, 2),
        )

        return result

    def _build_quote_bars(
        self, quotes: pl.DataFrame, interval_str: str
    ) -> pl.DataFrame:
        """Build quote-based features."""
        if quotes.is_empty():
            return pl.DataFrame()

        # Group quotes by time interval
        quote_bars = (
            quotes.sort("ts_event")
            .group_by_dynamic(
                "ts_event",
                every=interval_str,
                period=interval_str,
                label="left",
                closed="left",
            )
            .agg([
                # Midprice: average of last bid/ask in interval
                ((pl.col("bid_price").last() + pl.col("ask_price").last()) / 2)
                .alias("midprice"),
                # Spread: average spread in interval
                (pl.col("ask_price") - pl.col("bid_price")).mean().alias("spread"),
                # Average bid/ask sizes
                pl.col("bid_size").mean().alias("bid_size"),
                pl.col("ask_size").mean().alias("ask_size"),
                # Microprice: size-weighted midprice (last values)
                pl.col("bid_price").last().alias("last_bid"),
                pl.col("ask_price").last().alias("last_ask"),
                pl.col("bid_size").last().alias("last_bid_size"),
                pl.col("ask_size").last().alias("last_ask_size"),
            ])
            .rename({"ts_event": "bar_start"})
        )

        # Calculate microprice
        quote_bars = quote_bars.with_columns([
            pl.when(pl.col("last_bid_size") + pl.col("last_ask_size") > 0)
            .then(
                (
                    pl.col("last_bid") * pl.col("last_ask_size")
                    + pl.col("last_ask") * pl.col("last_bid_size")
                )
                / (pl.col("last_bid_size") + pl.col("last_ask_size"))
            )
            .otherwise(pl.col("midprice"))
            .alias("microprice")
        ])

        # Calculate quote imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        quote_bars = quote_bars.with_columns([
            pl.when(pl.col("bid_size") + pl.col("ask_size") > 0)
            .then(
                (pl.col("bid_size") - pl.col("ask_size"))
                / (pl.col("bid_size") + pl.col("ask_size"))
            )
            .otherwise(0.0)
            .alias("quote_imbalance")
        ])

        # Drop temporary columns
        return quote_bars.drop([
            "last_bid", "last_ask", "last_bid_size", "last_ask_size"
        ])

    def _build_trade_bars(
        self, trades: pl.DataFrame, interval_str: str
    ) -> pl.DataFrame:
        """Build trade-based features."""
        if trades.is_empty():
            return pl.DataFrame()

        # Add tick direction using tick rule (compare to previous price)
        # uptick (price > prev_price) = buy, downtick (price < prev_price) = sell
        trades_with_direction = trades.sort("ts_event").with_columns([
            pl.col("price").shift(1).alias("prev_price")
        ]).with_columns([
            pl.when(pl.col("price") > pl.col("prev_price"))
            .then(pl.lit(1))  # Buy
            .when(pl.col("price") < pl.col("prev_price"))
            .then(pl.lit(-1))  # Sell
            .otherwise(pl.lit(0))  # No change
            .alias("direction")
        ])

        # Group trades by time interval
        trade_bars = (
            trades_with_direction
            .group_by_dynamic(
                "ts_event",
                every=interval_str,
                period=interval_str,
                label="left",
                closed="left",
            )
            .agg([
                # VWAP
                (
                    (pl.col("price") * pl.col("size")).sum()
                    / pl.col("size").sum()
                ).alias("vwap"),
                # Total volume
                pl.col("size").sum().alias("trade_volume"),
                # Buy volume (upticks)
                pl.when(pl.col("direction") == 1)
                .then(pl.col("size"))
                .otherwise(0.0)
                .sum()
                .alias("buy_volume"),
                # Sell volume (downticks)
                pl.when(pl.col("direction") == -1)
                .then(pl.col("size"))
                .otherwise(0.0)
                .sum()
                .alias("sell_volume"),
                # Returns for realized vol
                pl.col("price").alias("prices"),
            ])
            .rename({"ts_event": "bar_start"})
        )

        # Calculate realized volatility from intra-bar returns
        trade_bars = trade_bars.with_columns([
            pl.col("prices")
            .map_elements(
                lambda prices: self._calc_realized_vol(prices),
                return_dtype=pl.Float64,
            )
            .alias("realized_vol")
        ])

        # Calculate trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        trade_bars = trade_bars.with_columns([
            pl.when(pl.col("buy_volume") + pl.col("sell_volume") > 0)
            .then(
                (pl.col("buy_volume") - pl.col("sell_volume"))
                / (pl.col("buy_volume") + pl.col("sell_volume"))
            )
            .otherwise(0.0)
            .alias("trade_imbalance")
        ])

        return trade_bars.drop(["prices", "buy_volume", "sell_volume"])

    @staticmethod
    def _calc_realized_vol(prices: pl.Series) -> float:
        """Calculate realized volatility from prices."""
        if prices is None or len(prices) < 2:
            return 0.0
        # Convert to list for computation
        price_list = prices.to_list()
        returns = []
        for i in range(1, len(price_list)):
            if price_list[i - 1] != 0:
                ret = (price_list[i] - price_list[i - 1]) / price_list[i - 1]
                returns.append(ret)
        if len(returns) < 2:
            return 0.0
        # Standard deviation of returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return float(variance ** 0.5)

    def _ensure_utc(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Ensure timestamp column is UTC timezone-aware."""
        if df.is_empty():
            return df
        ts_dtype = df.schema.get(col)
        if ts_dtype is not None and isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
            df = df.with_columns([
                pl.col(col).dt.replace_time_zone("UTC"),
            ])
        return df

    def _get_interval_str(self) -> str:
        """Get polars interval string for the granularity."""
        if self.granularity == "5s":
            return "5s"
        elif self.granularity == "15s":
            return "15s"
        elif self.granularity == "30s":
            return "30s"
        return "30s"

    def _empty_micro_bars_df(self) -> pl.DataFrame:
        """Return an empty DataFrame with correct schema."""
        return pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.String),
            "bar_start": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "bar_end": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "vwap": pl.Series([], dtype=pl.Float64),
            "midprice": pl.Series([], dtype=pl.Float64),
            "microprice": pl.Series([], dtype=pl.Float64),
            "spread": pl.Series([], dtype=pl.Float64),
            "bid_size": pl.Series([], dtype=pl.Float64),
            "ask_size": pl.Series([], dtype=pl.Float64),
            "quote_imbalance": pl.Series([], dtype=pl.Float64),
            "trade_imbalance": pl.Series([], dtype=pl.Float64),
            "trade_volume": pl.Series([], dtype=pl.Float64),
            "realized_vol": pl.Series([], dtype=pl.Float64),
        })
