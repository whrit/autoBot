"""
Standard Bar Builder (T2.01).

Builds OHLCV bars from raw trades at 1-minute, 5-minute, and 15-minute intervals.
Features: open, high, low, close, volume, returns, ATR, volatility.
"""

import time
from datetime import timedelta
from typing import Callable

import polars as pl
import structlog

# Valid granularities for standard bars
VALID_GRANULARITIES = {"1m", "5m", "15m"}

# Granularity to timedelta mapping
GRANULARITY_MAP = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
}

logger = structlog.get_logger(__name__)


class StandardBarBuilder:
    """
    Build standard OHLCV bars from trade data.

    Supports 1m, 5m, and 15m granularities.
    Computes: open, high, low, close, volume, returns, ATR, realized volatility.
    """

    def __init__(self, granularity: str = "1m", atr_period: int = 14) -> None:
        """
        Initialize the bar builder.

        Args:
            granularity: Bar interval ('1m', '5m', '15m').
            atr_period: Period for ATR calculation (default 14).

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
        self.atr_period = atr_period
        self._log = logger.bind(
            builder="StandardBarBuilder",
            granularity=granularity,
            atr_period=atr_period,
        )
        self._log.info("initialized")

    def build(
        self,
        trades: pl.DataFrame,
        symbol: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """
        Build bars from trade data.

        Args:
            trades: DataFrame with columns: ts_event, ts_recv, price, size, exchange, conditions.
            symbol: Stock symbol.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            DataFrame with bar data including OHLCV and derived features.
        """
        start_time = time.perf_counter()
        input_rows = len(trades)

        self._log.info(
            "build_started",
            symbol=symbol,
            input_rows=input_rows,
        )

        if trades.is_empty():
            self._log.warning("empty_input", symbol=symbol)
            return self._empty_bars_df()

        # Ensure timestamps are timezone-aware
        trades = self._ensure_utc(trades)
        if progress_callback:
            progress_callback(1, 5)

        # Get interval string for polars groupby_dynamic
        interval_str = self._get_interval_str()

        # Group by time interval and compute OHLCV
        bars = (
            trades.sort("ts_event")
            .group_by_dynamic(
                "ts_event",
                every=interval_str,
                period=interval_str,
                label="left",
                closed="left",
            )
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
            ])
            .rename({"ts_event": "bar_start"})
        )
        if progress_callback:
            progress_callback(2, 5)

        # Add bar_end column
        bars = bars.with_columns([
            (pl.col("bar_start") + self.interval).alias("bar_end"),
            pl.lit(symbol).alias("symbol"),
        ])

        # Calculate returns (close-to-close)
        bars = bars.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")
        ])
        if progress_callback:
            progress_callback(3, 5)

        # Calculate ATR (Average True Range)
        bars = self._calculate_atr(bars)
        if progress_callback:
            progress_callback(4, 5)

        # Calculate realized volatility
        bars = self._calculate_realized_vol(bars)
        if progress_callback:
            progress_callback(5, 5)

        # Reorder columns to match schema
        result = bars.select([
            "symbol",
            "bar_start",
            "bar_end",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "atr",
            "realized_vol",
        ])

        elapsed = time.perf_counter() - start_time
        output_bars = len(result)
        throughput = input_rows / elapsed if elapsed > 0 else 0

        self._log.info(
            "build_completed",
            symbol=symbol,
            input_rows=input_rows,
            output_bars=output_bars,
            processing_time_sec=round(elapsed, 3),
            throughput_rows_per_sec=round(throughput, 1),
        )

        return result

    def _ensure_utc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure timestamps are UTC timezone-aware."""
        ts_dtype = df.schema.get("ts_event")
        if ts_dtype is not None and isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
            df = df.with_columns([
                pl.col("ts_event").dt.replace_time_zone("UTC"),
            ])
        return df

    def _get_interval_str(self) -> str:
        """Get polars interval string for the granularity."""
        if self.granularity == "1m":
            return "1m"
        elif self.granularity == "5m":
            return "5m"
        elif self.granularity == "15m":
            return "15m"
        return "1m"

    def _calculate_atr(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Average True Range (ATR).

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = rolling mean of True Range.
        """
        bars = bars.with_columns([
            pl.col("close").shift(1).alias("prev_close")
        ])

        bars = bars.with_columns([
            pl.max_horizontal([
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("prev_close")).abs(),
                (pl.col("low") - pl.col("prev_close")).abs(),
            ]).alias("true_range")
        ])

        bars = bars.with_columns([
            pl.col("true_range")
            .rolling_mean(window_size=self.atr_period, min_samples=1)
            .alias("atr")
        ])

        return bars.drop(["prev_close", "true_range"])

    def _calculate_realized_vol(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate realized volatility (rolling std of returns).
        """
        bars = bars.with_columns([
            pl.col("returns")
            .rolling_std(window_size=self.atr_period, min_samples=2)
            .alias("realized_vol")
        ])
        return bars

    def _empty_bars_df(self) -> pl.DataFrame:
        """Return an empty DataFrame with correct schema."""
        return pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.String),
            "bar_start": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "bar_end": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
            "returns": pl.Series([], dtype=pl.Float64),
            "atr": pl.Series([], dtype=pl.Float64),
            "realized_vol": pl.Series([], dtype=pl.Float64),
        })
