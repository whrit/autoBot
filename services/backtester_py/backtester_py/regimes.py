"""
Regime analysis module for market condition classification.

This module provides regime-based performance evaluation:
- Volatility regimes (low, medium, high based on realized vol)
- Liquidity regimes (based on spread and volume)
- Trend regimes (based on price momentum)

Understanding strategy performance across regimes helps identify
when strategies work best and when to avoid trading.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


class RegimeType(str, Enum):
    """Regime classification levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RegimeAnalyzer:
    """
    Market regime analyzer and classifier.

    Classifies market data into volatility, liquidity, and trend
    regimes based on quantile thresholds.

    Attributes:
        vol_quantiles: Tuple of (low_threshold, high_threshold) for volatility
        liquidity_quantiles: Tuple of (low_threshold, high_threshold) for liquidity
        trend_quantiles: Tuple of (low_threshold, high_threshold) for trend
        vol_lookback: Lookback period for volatility calculation
    """

    def __init__(
        self,
        vol_quantiles: tuple[float, float] = (0.33, 0.67),
        liquidity_quantiles: tuple[float, float] = (0.33, 0.67),
        trend_quantiles: tuple[float, float] = (0.33, 0.67),
        vol_lookback: int = 20,
        trend_lookback: int = 20,
    ) -> None:
        """
        Initialize RegimeAnalyzer.

        Args:
            vol_quantiles: Quantile thresholds for volatility classification
            liquidity_quantiles: Quantile thresholds for liquidity classification
            trend_quantiles: Quantile thresholds for trend classification
            vol_lookback: Rolling window for volatility calculation
            trend_lookback: Rolling window for trend calculation
        """
        self.vol_quantiles = vol_quantiles
        self.liquidity_quantiles = liquidity_quantiles
        self.trend_quantiles = trend_quantiles
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback

    def classify_regimes(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Classify market data into volatility, liquidity, and trend regimes.

        Args:
            data: DataFrame with columns: timestamp, close, volume, spread, returns

        Returns:
            DataFrame with original columns plus:
            - vol_regime: Volatility regime (low, medium, high)
            - liquidity_regime: Liquidity regime (low, medium, high)
            - trend_regime: Trend regime (low, medium, high)
            - realized_vol: Rolling realized volatility
            - liquidity_score: Combined liquidity score
            - trend_score: Rolling momentum score
        """
        result = data.clone()

        # Calculate volatility regime
        result = self._add_volatility_regime(result)

        # Calculate liquidity regime
        result = self._add_liquidity_regime(result)

        # Calculate trend regime
        result = self._add_trend_regime(result)

        return result

    def _add_volatility_regime(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Add volatility regime classification.

        Volatility is measured as rolling standard deviation of returns.

        Args:
            data: DataFrame with 'returns' column

        Returns:
            DataFrame with 'realized_vol' and 'vol_regime' columns
        """
        # Calculate rolling volatility
        if "returns" in data.columns:
            returns = data["returns"].to_numpy().astype(np.float64)
        elif "close" in data.columns:
            close = data["close"].to_numpy().astype(np.float64)
            returns = np.diff(np.log(close), prepend=0.0)
        else:
            # Default to zero volatility if no returns available
            data = data.with_columns(
                pl.lit(0.0).alias("realized_vol"),
                pl.lit(RegimeType.MEDIUM.value).alias("vol_regime"),
            )
            return data

        # Calculate rolling standard deviation
        realized_vol = self._rolling_std(returns, self.vol_lookback)

        # Classify into regimes based on quantiles
        vol_regime = self._classify_by_quantile(
            realized_vol, self.vol_quantiles
        )

        return data.with_columns(
            pl.Series("realized_vol", realized_vol),
            pl.Series("vol_regime", vol_regime),
        )

    def _add_liquidity_regime(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Add liquidity regime classification.

        Liquidity is measured as inverse of spread (normalized by volume).

        Args:
            data: DataFrame with 'spread' and 'volume' columns

        Returns:
            DataFrame with 'liquidity_score' and 'liquidity_regime' columns
        """
        if "spread" in data.columns and "volume" in data.columns:
            spread = data["spread"].to_numpy().astype(np.float64)
            volume = data["volume"].to_numpy().astype(np.float64)

            # Liquidity score: higher volume / lower spread = more liquid
            # Normalize both components
            spread_normalized = self._normalize_to_range(spread)
            volume_normalized = self._normalize_to_range(volume)

            # Combine: high volume + low spread = high liquidity
            liquidity_score = volume_normalized - spread_normalized

        elif "spread" in data.columns:
            spread = data["spread"].to_numpy().astype(np.float64)
            # Lower spread = higher liquidity
            liquidity_score = -self._normalize_to_range(spread)

        elif "volume" in data.columns:
            volume = data["volume"].to_numpy().astype(np.float64)
            # Higher volume = higher liquidity
            liquidity_score = self._normalize_to_range(volume)

        else:
            # Default to medium liquidity
            data = data.with_columns(
                pl.lit(0.0).alias("liquidity_score"),
                pl.lit(RegimeType.MEDIUM.value).alias("liquidity_regime"),
            )
            return data

        # Classify into regimes
        liquidity_regime = self._classify_by_quantile(
            liquidity_score, self.liquidity_quantiles
        )

        return data.with_columns(
            pl.Series("liquidity_score", liquidity_score),
            pl.Series("liquidity_regime", liquidity_regime),
        )

    def _add_trend_regime(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Add trend regime classification.

        Trend is measured as rolling momentum (cumulative return over lookback).

        Args:
            data: DataFrame with 'returns' or 'close' column

        Returns:
            DataFrame with 'trend_score' and 'trend_regime' columns
        """
        if "returns" in data.columns:
            returns = data["returns"].to_numpy().astype(np.float64)
        elif "close" in data.columns:
            close = data["close"].to_numpy().astype(np.float64)
            returns = np.diff(np.log(close), prepend=0.0)
        else:
            data = data.with_columns(
                pl.lit(0.0).alias("trend_score"),
                pl.lit(RegimeType.MEDIUM.value).alias("trend_regime"),
            )
            return data

        # Calculate rolling momentum (cumulative return over lookback)
        trend_score = self._rolling_sum(returns, self.trend_lookback)

        # Classify into regimes
        trend_regime = self._classify_by_quantile(
            trend_score, self.trend_quantiles
        )

        return data.with_columns(
            pl.Series("trend_score", trend_score),
            pl.Series("trend_regime", trend_regime),
        )

    def _rolling_std(self, values: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling standard deviation.

        Args:
            values: Input array
            window: Rolling window size

        Returns:
            Array of rolling standard deviations
        """
        n = len(values)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_vals = values[i - window + 1 : i + 1]
            result[i] = np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0.0

        # Fill initial values with first valid value
        first_valid_idx = window - 1
        if first_valid_idx < n:
            result[:first_valid_idx] = result[first_valid_idx]

        return result

    def _rolling_sum(self, values: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate rolling sum.

        Args:
            values: Input array
            window: Rolling window size

        Returns:
            Array of rolling sums
        """
        n = len(values)
        result = np.full(n, np.nan)

        cumsum = np.cumsum(values)
        result[window - 1 :] = cumsum[window - 1 :] - np.concatenate(
            [[0], cumsum[: -window]]
        )

        # Fill initial values with partial sums
        for i in range(window - 1):
            result[i] = np.sum(values[: i + 1])

        return result

    def _normalize_to_range(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values to [0, 1] range.

        Args:
            values: Input array

        Returns:
            Normalized array
        """
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)

        if max_val == min_val:
            return np.full_like(values, 0.5)

        result: np.ndarray = (values - min_val) / (max_val - min_val)
        return result

    def _classify_by_quantile(
        self,
        values: np.ndarray,
        quantiles: tuple[float, float],
    ) -> list[str]:
        """
        Classify values into low/medium/high based on quantile thresholds.

        Args:
            values: Input array
            quantiles: Tuple of (low_threshold, high_threshold)

        Returns:
            List of regime labels (low, medium, high)
        """
        # Calculate quantile thresholds
        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return [RegimeType.MEDIUM.value] * len(values)

        low_thresh = np.percentile(valid_values, quantiles[0] * 100)
        high_thresh = np.percentile(valid_values, quantiles[1] * 100)

        # Classify each value
        result: list[str] = []
        for v in values:
            if np.isnan(v):
                result.append(RegimeType.MEDIUM.value)
            elif v <= low_thresh:
                result.append(RegimeType.LOW.value)
            elif v >= high_thresh:
                result.append(RegimeType.HIGH.value)
            else:
                result.append(RegimeType.MEDIUM.value)

        return result


__all__ = ["RegimeAnalyzer", "RegimeType"]
