"""
Decision Frame Builder (T2.04).

Combines multi-timeframe features into a single decision matrix.
Each row = one decision point with all available features.
"""

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from feature_builder_py.joins import AsOfJoiner


def get_schema_version(source: pl.DataFrame | Path | str) -> str:
    """
    Get schema version from a DataFrame or Parquet file.

    Args:
        source: Either a DataFrame with schema_version column,
                or a path to a Parquet file.

    Returns:
        The schema version string, or "unknown" if not found.
    """
    if isinstance(source, pl.DataFrame):
        if "schema_version" in source.columns and len(source) > 0:
            return str(source["schema_version"][0])
        return "unknown"

    # It's a path - read from Parquet metadata
    path = Path(source)
    if not path.exists():
        return "unknown"

    try:
        # First try to read from Parquet metadata
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.schema_arrow.metadata
        if metadata and b"schema_version" in metadata:
            return str(metadata[b"schema_version"].decode("utf-8"))

        # Fallback: read the schema_version column
        df = pl.read_parquet(path, columns=["schema_version"])
        if len(df) > 0:
            return str(df["schema_version"][0])
    except Exception:
        pass

    return "unknown"


def validate_schema_version(
    source: pl.DataFrame | Path | str,
    expected_version: str,
) -> bool:
    """
    Validate that the schema version matches the expected version.

    Args:
        source: Either a DataFrame or path to a Parquet file.
        expected_version: The expected schema version string.

    Returns:
        True if versions match, False otherwise.
    """
    actual_version = get_schema_version(source)
    return actual_version == expected_version


class DecisionFrameBuilder:
    """
    Build decision frames from multiple timeframe features.

    Combines 30s microstructure, 1m, 5m, and 15m features into
    a single feature matrix for each decision point.
    """

    def __init__(self, schema_version: str = "1.0.0") -> None:
        """
        Initialize the decision frame builder.

        Args:
            schema_version: Version string for the output schema.
        """
        self.schema_version = schema_version
        self.joiner = AsOfJoiner()

    def build(
        self,
        decision_times: pl.DataFrame,
        micro_bars_30s: pl.DataFrame,
        bars_1m: pl.DataFrame,
        bars_5m: pl.DataFrame,
        bars_15m: pl.DataFrame | None = None,
        symbol: str = "",
    ) -> pl.DataFrame:
        """
        Build a decision frame from multiple timeframe features.

        Args:
            decision_times: DataFrame with 'decision_ts' column.
            micro_bars_30s: 30-second microstructure bars.
            bars_1m: 1-minute OHLCV bars.
            bars_5m: 5-minute OHLCV bars.
            bars_15m: 15-minute OHLCV bars (optional).
            symbol: Stock symbol.

        Returns:
            DataFrame with all features for each decision point.
        """
        if decision_times.is_empty():
            return self._empty_decision_frame()

        # Start with decision times
        frame = decision_times.clone()

        # Add symbol
        frame = frame.with_columns(pl.lit(symbol).alias("symbol"))

        # Join 30s micro features
        frame = self._join_micro_features(frame, micro_bars_30s)

        # Join 1m features
        frame = self._join_1m_features(frame, bars_1m)

        # Join 5m features
        frame = self._join_5m_features(frame, bars_5m)

        # Join 15m features (optional)
        frame = self._join_15m_features(frame, bars_15m)

        # Ensure all expected columns exist
        frame = self._ensure_columns(frame)

        # Add schema version column
        frame = frame.with_columns(
            pl.lit(self.schema_version).alias("schema_version")
        )

        # Reorder columns to match schema
        return self._reorder_columns(frame)

    def _join_micro_features(
        self, frame: pl.DataFrame, micro_bars: pl.DataFrame
    ) -> pl.DataFrame:
        """Join 30s microstructure features."""
        if micro_bars.is_empty():
            return frame.with_columns([
                pl.lit(None).cast(pl.Float64).alias("spread_30s"),
                pl.lit(None).cast(pl.Float64).alias("microprice_30s"),
                pl.lit(None).cast(pl.Float64).alias("quote_imbalance_30s"),
                pl.lit(None).cast(pl.Float64).alias("vol_30s"),
            ])

        # Select relevant columns
        micro_features = micro_bars.select([
            "bar_end",
            pl.col("spread").alias("spread_30s"),
            pl.col("microprice").alias("microprice_30s"),
            pl.col("quote_imbalance").alias("quote_imbalance_30s"),
            pl.col("realized_vol").alias("vol_30s"),
        ])

        return self.joiner.join(
            frame, micro_features,
            on="decision_ts", by="bar_end"
        )

    def _join_1m_features(
        self, frame: pl.DataFrame, bars_1m: pl.DataFrame
    ) -> pl.DataFrame:
        """Join 1-minute features."""
        if bars_1m.is_empty():
            return frame.with_columns([
                pl.lit(None).cast(pl.Float64).alias("ret_1m"),
                pl.lit(None).cast(pl.Float64).alias("vol_1m"),
                pl.lit(None).cast(pl.Float64).alias("atr_1m"),
            ])

        # Select relevant columns
        bar_features = bars_1m.select([
            "bar_end",
            pl.col("returns").alias("ret_1m"),
            pl.col("realized_vol").alias("vol_1m"),
            pl.col("atr").alias("atr_1m"),
        ])

        return self.joiner.join(
            frame, bar_features,
            on="decision_ts", by="bar_end"
        )

    def _join_5m_features(
        self, frame: pl.DataFrame, bars_5m: pl.DataFrame
    ) -> pl.DataFrame:
        """Join 5-minute features."""
        if bars_5m.is_empty():
            return frame.with_columns([
                pl.lit(None).cast(pl.Float64).alias("ret_5m"),
                pl.lit(None).cast(pl.Float64).alias("trend_5m"),
                pl.lit(None).cast(pl.Float64).alias("vol_5m"),
            ])

        # Calculate trend as close - open (price direction)
        bars_with_trend = bars_5m.with_columns([
            (pl.col("close") - pl.col("open")).alias("trend"),
        ])

        # Select relevant columns
        bar_features = bars_with_trend.select([
            "bar_end",
            pl.col("returns").alias("ret_5m"),
            pl.col("trend").alias("trend_5m"),
            pl.col("realized_vol").alias("vol_5m"),
        ])

        return self.joiner.join(
            frame, bar_features,
            on="decision_ts", by="bar_end"
        )

    def _join_15m_features(
        self, frame: pl.DataFrame, bars_15m: pl.DataFrame | None
    ) -> pl.DataFrame:
        """Join 15-minute features."""
        if bars_15m is None or bars_15m.is_empty():
            return frame.with_columns([
                pl.lit(None).cast(pl.Float64).alias("ret_15m"),
                pl.lit(None).cast(pl.Float64).alias("trend_15m"),
                pl.lit(None).cast(pl.Float64).alias("vol_15m"),
            ])

        # Calculate trend
        bars_with_trend = bars_15m.with_columns([
            (pl.col("close") - pl.col("open")).alias("trend"),
        ])

        # Select relevant columns
        bar_features = bars_with_trend.select([
            "bar_end",
            pl.col("returns").alias("ret_15m"),
            pl.col("trend").alias("trend_15m"),
            pl.col("realized_vol").alias("vol_15m"),
        ])

        return self.joiner.join(
            frame, bar_features,
            on="decision_ts", by="bar_end"
        )

    def _ensure_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Ensure all expected columns exist in the frame."""
        expected_columns: dict[str, pl.DataType] = {
            "symbol": pl.String(),
            "decision_ts": pl.Datetime("us", "UTC"),
            "spread_30s": pl.Float64(),
            "microprice_30s": pl.Float64(),
            "quote_imbalance_30s": pl.Float64(),
            "vol_30s": pl.Float64(),
            "ret_1m": pl.Float64(),
            "vol_1m": pl.Float64(),
            "atr_1m": pl.Float64(),
            "ret_5m": pl.Float64(),
            "trend_5m": pl.Float64(),
            "vol_5m": pl.Float64(),
            "ret_15m": pl.Float64(),
            "trend_15m": pl.Float64(),
            "vol_15m": pl.Float64(),
        }

        for col, dtype in expected_columns.items():
            if col not in frame.columns:
                frame = frame.with_columns(
                    pl.lit(None).cast(dtype).alias(col)
                )

        return frame

    def _reorder_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Reorder columns to match the DecisionFrame schema."""
        column_order = [
            "symbol",
            "decision_ts",
            # 30s micro features
            "spread_30s",
            "microprice_30s",
            "quote_imbalance_30s",
            "vol_30s",
            # 1m features
            "ret_1m",
            "vol_1m",
            "atr_1m",
            # 5m features
            "ret_5m",
            "trend_5m",
            "vol_5m",
            # 15m features
            "ret_15m",
            "trend_15m",
            "vol_15m",
            # Schema version (always last)
            "schema_version",
        ]

        # Select only columns that exist
        available_columns = [c for c in column_order if c in frame.columns]
        return frame.select(available_columns)

    def write_parquet(
        self,
        frame: pl.DataFrame,
        path: Path | str,
    ) -> None:
        """
        Write decision frame to Parquet with schema version metadata.

        Args:
            frame: DataFrame to write.
            path: Path to output Parquet file.
        """

        path = Path(path)

        # Convert to Arrow table
        table = frame.to_arrow()

        # Add schema_version to Arrow metadata
        existing_metadata = table.schema.metadata or {}
        new_metadata = {
            **existing_metadata,
            b"schema_version": self.schema_version.encode("utf-8"),
        }
        table = table.replace_schema_metadata(new_metadata)

        # Write to Parquet
        pq.write_table(table, path)

    def _empty_decision_frame(self) -> pl.DataFrame:
        """Return an empty DataFrame with correct schema."""
        return pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.String),
            "decision_ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "spread_30s": pl.Series([], dtype=pl.Float64),
            "microprice_30s": pl.Series([], dtype=pl.Float64),
            "quote_imbalance_30s": pl.Series([], dtype=pl.Float64),
            "vol_30s": pl.Series([], dtype=pl.Float64),
            "ret_1m": pl.Series([], dtype=pl.Float64),
            "vol_1m": pl.Series([], dtype=pl.Float64),
            "atr_1m": pl.Series([], dtype=pl.Float64),
            "ret_5m": pl.Series([], dtype=pl.Float64),
            "trend_5m": pl.Series([], dtype=pl.Float64),
            "vol_5m": pl.Series([], dtype=pl.Float64),
            "ret_15m": pl.Series([], dtype=pl.Float64),
            "trend_15m": pl.Series([], dtype=pl.Float64),
            "vol_15m": pl.Series([], dtype=pl.Float64),
            "schema_version": pl.Series([], dtype=pl.String),
        })
