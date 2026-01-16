"""
Standard Bar Builder (T2.01).

Builds OHLCV bars from raw trades at 1-minute, 5-minute, and 15-minute intervals.
Features: open, high, low, close, volume, returns, ATR, volatility.

Optimized for Large Dataset Processing:
- Lazy frame support with scan_parquet() for memory efficiency
- Batch processing for datasets larger than available memory
- Streaming collection for large files
- Progress callbacks for long-running operations
"""

import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Generator

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

    def build_lazy(
        self,
        trades_lf: pl.LazyFrame,
        symbol: str,
    ) -> pl.LazyFrame:
        """
        Build bars from a LazyFrame for memory-efficient processing.

        Uses lazy evaluation - no computation happens until .collect() is called.
        This enables query optimization and predicate pushdown.

        Args:
            trades_lf: LazyFrame with trade data.
            symbol: Stock symbol.

        Returns:
            LazyFrame with bar data (call .collect() to materialize).
        """
        self._log.info(
            "build_lazy_started",
            symbol=symbol,
        )

        # Get interval string for polars groupby_dynamic
        interval_str = self._get_interval_str()

        # Build the lazy computation graph
        bars_lf = (
            trades_lf.sort("ts_event")
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
            .with_columns([
                (pl.col("bar_start") + self.interval).alias("bar_end"),
                pl.lit(symbol).alias("symbol"),
            ])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")
            ])
        )

        # ATR calculation (simplified for lazy evaluation)
        bars_lf = bars_lf.with_columns([
            pl.col("close").shift(1).alias("prev_close")
        ]).with_columns([
            pl.max_horizontal([
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("prev_close")).abs(),
                (pl.col("low") - pl.col("prev_close")).abs(),
            ]).alias("true_range")
        ]).with_columns([
            pl.col("true_range")
            .rolling_mean(window_size=self.atr_period, min_samples=1)
            .alias("atr")
        ]).with_columns([
            pl.col("returns")
            .rolling_std(window_size=self.atr_period, min_samples=2)
            .alias("realized_vol")
        ]).drop(["prev_close", "true_range"])

        # Reorder columns
        result_lf = bars_lf.select([
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

        self._log.info(
            "build_lazy_plan_created",
            symbol=symbol,
        )

        return result_lf

    def build_from_parquet(
        self,
        path: Path | str,
        symbol: str,
        streaming: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """
        Build bars directly from a parquet file using lazy evaluation.

        Uses scan_parquet() for memory efficiency and streaming collection
        for large files.

        Args:
            path: Path to parquet file with trade data.
            symbol: Stock symbol.
            streaming: Use streaming collection for large files.
            progress_callback: Optional callback for progress updates.

        Returns:
            DataFrame with bar data.
        """
        path = Path(path)
        start_time = time.perf_counter()

        self._log.info(
            "build_from_parquet_started",
            symbol=symbol,
            path=str(path),
            streaming=streaming,
        )

        # Scan parquet lazily with memory mapping
        trades_lf = pl.scan_parquet(
            path,
            memory_map=True,
            low_memory=False,
        )

        if progress_callback:
            progress_callback(1, 3)

        # Build lazy computation graph
        bars_lf = self.build_lazy(trades_lf, symbol)

        if progress_callback:
            progress_callback(2, 3)

        # Collect with streaming if requested
        try:
            if streaming:
                result = bars_lf.collect(streaming=True)
            else:
                result = bars_lf.collect()
        except Exception as e:
            self._log.warning(
                "streaming_fallback",
                error=str(e),
            )
            result = bars_lf.collect()

        if progress_callback:
            progress_callback(3, 3)

        elapsed = time.perf_counter() - start_time

        self._log.info(
            "build_from_parquet_completed",
            symbol=symbol,
            output_bars=len(result),
            processing_time_sec=round(elapsed, 3),
        )

        return result

    def build_batched(
        self,
        path: Path | str,
        symbol: str,
        batch_size: int = 1_000_000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Build bars in batches for very large datasets.

        Processes the input file in chunks, yielding bar DataFrames
        for each batch. Useful when the dataset is too large to fit
        in memory.

        Args:
            path: Path to parquet file with trade data.
            symbol: Stock symbol.
            batch_size: Number of rows per batch.
            progress_callback: Optional callback for progress updates.

        Yields:
            DataFrame with bars for each batch.
        """
        path = Path(path)

        self._log.info(
            "build_batched_started",
            symbol=symbol,
            path=str(path),
            batch_size=batch_size,
        )

        # Get total row count
        total_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        num_batches = (total_rows + batch_size - 1) // batch_size

        self._log.info(
            "batch_plan",
            total_rows=total_rows,
            batch_size=batch_size,
            num_batches=num_batches,
        )

        offset = 0
        batch_idx = 0

        while offset < total_rows:
            # Scan with offset and limit
            batch_lf = pl.scan_parquet(path, memory_map=True)
            batch_lf = batch_lf.sort("ts_event").slice(offset, batch_size)

            # Build bars for this batch
            bars_lf = self.build_lazy(batch_lf, symbol)
            batch_bars = bars_lf.collect()

            if progress_callback:
                progress_callback(batch_idx + 1, num_batches)

            self._log.debug(
                "batch_completed",
                batch_idx=batch_idx,
                input_rows=min(batch_size, total_rows - offset),
                output_bars=len(batch_bars),
            )

            yield batch_bars

            offset += batch_size
            batch_idx += 1

        self._log.info(
            "build_batched_completed",
            symbol=symbol,
            batches_processed=batch_idx,
        )

    def build_and_write(
        self,
        input_path: Path | str,
        output_path: Path | str,
        symbol: str,
        batch_size: int = 1_000_000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Build bars and write directly to parquet file.

        Processes in batches and writes incrementally to avoid
        holding all data in memory.

        Args:
            input_path: Path to input parquet file with trade data.
            output_path: Path to output parquet file.
            symbol: Stock symbol.
            batch_size: Number of rows per batch.
            progress_callback: Optional callback for progress updates.

        Returns:
            Total number of bars written.
        """
        output_path = Path(output_path)
        total_bars = 0
        first_batch = True

        for batch_bars in self.build_batched(
            input_path, symbol, batch_size, progress_callback
        ):
            if first_batch:
                # Write first batch (creates file)
                batch_bars.write_parquet(output_path)
                first_batch = False
            else:
                # Append subsequent batches
                existing = pl.read_parquet(output_path)
                combined = pl.concat([existing, batch_bars])
                combined.write_parquet(output_path)

            total_bars += len(batch_bars)

        self._log.info(
            "build_and_write_completed",
            symbol=symbol,
            output_path=str(output_path),
            total_bars=total_bars,
        )

        return total_bars
