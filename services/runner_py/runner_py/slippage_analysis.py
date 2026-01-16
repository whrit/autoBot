"""
Paper Slippage Analysis (T6.08).

This module provides analysis of actual vs expected slippage in paper trading
to validate execution quality and identify issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class SlippageAnalysisConfig:
    """
    Configuration for slippage analysis.

    Attributes:
        expected_slippage_bps: Expected slippage in basis points
        max_acceptable_ratio: Maximum acceptable ratio to expected (2.0 = 2x expected)
    """

    expected_slippage_bps: float = 2.0
    max_acceptable_ratio: float = 2.0


@dataclass
class SlippageReport:
    """
    Report of paper trading slippage analysis.

    Attributes:
        total_fills: Total number of fills analyzed
        avg_slippage_bps: Average slippage in basis points
        max_slippage_bps: Maximum slippage in basis points
        min_slippage_bps: Minimum slippage in basis points
        std_slippage_bps: Standard deviation of slippage
        ratio_to_expected: Ratio of actual to expected slippage
        within_acceptable: Whether slippage is within acceptable range
        by_symbol: Breakdown by trading symbol
        by_hour: Breakdown by hour of day
        outliers: List of outlier fills
    """

    total_fills: int
    avg_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    std_slippage_bps: float
    ratio_to_expected: float
    within_acceptable: bool
    by_symbol: dict[str, dict[str, float | int]] = field(default_factory=dict)
    by_hour: dict[int, dict[str, float | int]] = field(default_factory=dict)
    outliers: list[dict[str, str | float]] = field(default_factory=list)


class SlippageAnalyzer:
    """
    Analyze paper trading slippage vs expectations.

    Provides comprehensive analysis of fill quality including breakdowns
    by symbol and time of day, and identification of outliers.
    """

    def __init__(self, config: SlippageAnalysisConfig | None = None) -> None:
        """
        Initialize the SlippageAnalyzer.

        Args:
            config: Analysis configuration. Uses defaults if not provided.
        """
        self.config = config or SlippageAnalysisConfig()

    def analyze(self, fills: pl.DataFrame) -> SlippageReport:
        """
        Analyze slippage from paper fills.

        Args:
            fills: DataFrame with fill data.
                Expected columns: timestamp, symbol, side, quantity,
                                 expected_price, actual_price, slippage_bps, order_id

        Returns:
            SlippageReport with comprehensive analysis.
        """

        # Handle empty DataFrame
        if len(fills) == 0:
            return SlippageReport(
                total_fills=0,
                avg_slippage_bps=0.0,
                max_slippage_bps=0.0,
                min_slippage_bps=0.0,
                std_slippage_bps=0.0,
                ratio_to_expected=0.0,
                within_acceptable=True,
                by_symbol={},
                by_hour={},
                outliers=[],
            )

        # Extract slippage values
        slippage_values = fills["slippage_bps"].to_numpy()

        # Calculate basic statistics
        total_fills = len(fills)
        avg_slippage = float(np.mean(slippage_values))
        max_slippage = float(np.max(slippage_values))
        min_slippage = float(np.min(slippage_values))

        # Handle single value case for std
        std_slippage = (
            float(np.std(slippage_values, ddof=1))
            if len(slippage_values) > 1
            else 0.0
        )

        # Calculate ratio to expected
        if self.config.expected_slippage_bps > 0:
            ratio_to_expected = abs(avg_slippage) / self.config.expected_slippage_bps
        else:
            ratio_to_expected = 0.0 if avg_slippage == 0 else float("inf")

        # Determine if within acceptable range
        within_acceptable = ratio_to_expected <= self.config.max_acceptable_ratio

        # Analyze by symbol
        by_symbol = self.analyze_by_symbol(fills)

        # Analyze by time
        by_hour = self.analyze_by_time(fills)

        # Identify outliers (default threshold: 10 bps)
        outliers = self.identify_outliers(fills, threshold_bps=10.0)

        return SlippageReport(
            total_fills=total_fills,
            avg_slippage_bps=avg_slippage,
            max_slippage_bps=max_slippage,
            min_slippage_bps=min_slippage,
            std_slippage_bps=std_slippage,
            ratio_to_expected=ratio_to_expected,
            within_acceptable=within_acceptable,
            by_symbol=by_symbol,
            by_hour=by_hour,
            outliers=outliers,
        )

    def identify_outliers(
        self, fills: pl.DataFrame, threshold_bps: float = 10.0
    ) -> list[dict[str, str | float]]:
        """
        Identify fills with abnormal slippage.

        Args:
            fills: DataFrame with fill data
            threshold_bps: Slippage threshold for outlier detection

        Returns:
            List of outlier fill records
        """
        import polars as pl

        if len(fills) == 0:
            return []

        # Filter fills exceeding threshold
        outlier_df = fills.filter(pl.col("slippage_bps").abs() > threshold_bps)

        if len(outlier_df) == 0:
            return []

        # Convert to list of dicts
        outliers: list[dict[str, str | float]] = []
        for row in outlier_df.iter_rows(named=True):
            outliers.append({
                "order_id": str(row.get("order_id", "")),
                "symbol": str(row.get("symbol", "")),
                "slippage_bps": float(row.get("slippage_bps", 0.0)),
                "side": str(row.get("side", "")),
                "quantity": float(row.get("quantity", 0.0)),
            })

        return outliers

    def analyze_by_symbol(self, fills: pl.DataFrame) -> dict[str, dict[str, float | int]]:
        """
        Break down slippage by symbol.

        Args:
            fills: DataFrame with fill data

        Returns:
            Dictionary mapping symbol to slippage statistics
        """
        import polars as pl

        if len(fills) == 0:
            return {}

        # Group by symbol and calculate stats
        grouped = fills.group_by("symbol").agg([
            pl.col("slippage_bps").mean().alias("avg_slippage_bps"),
            pl.col("slippage_bps").max().alias("max_slippage_bps"),
            pl.col("slippage_bps").min().alias("min_slippage_bps"),
            pl.col("slippage_bps").count().alias("fill_count"),
        ])

        result: dict[str, dict[str, float | int]] = {}
        for row in grouped.iter_rows(named=True):
            symbol = str(row["symbol"])
            result[symbol] = {
                "avg_slippage_bps": float(row["avg_slippage_bps"]),
                "max_slippage_bps": float(row["max_slippage_bps"]),
                "min_slippage_bps": float(row["min_slippage_bps"]),
                "fill_count": int(row["fill_count"]),
            }

        return result

    def analyze_by_time(self, fills: pl.DataFrame) -> dict[int, dict[str, float | int]]:
        """
        Break down slippage by hour of day.

        Args:
            fills: DataFrame with fill data

        Returns:
            Dictionary mapping hour (0-23) to slippage statistics
        """
        import polars as pl

        if len(fills) == 0:
            return {}

        # Extract hour from timestamp and group
        fills_with_hour = fills.with_columns([
            pl.col("timestamp").dt.hour().alias("hour")
        ])

        grouped = fills_with_hour.group_by("hour").agg([
            pl.col("slippage_bps").mean().alias("avg_slippage_bps"),
            pl.col("slippage_bps").max().alias("max_slippage_bps"),
            pl.col("slippage_bps").count().alias("fill_count"),
        ])

        result: dict[int, dict[str, float | int]] = {}
        for row in grouped.iter_rows(named=True):
            hour = int(row["hour"])
            result[hour] = {
                "avg_slippage_bps": float(row["avg_slippage_bps"]),
                "max_slippage_bps": float(row["max_slippage_bps"]),
                "fill_count": int(row["fill_count"]),
            }

        return result


__all__ = ["SlippageAnalysisConfig", "SlippageAnalyzer", "SlippageReport"]
