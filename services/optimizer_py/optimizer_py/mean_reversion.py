"""
Mean Reversion Strategy Implementation (T4.03).

Implements mean-reversion trading signals using:
- Bollinger Bands reversion
- Z-score based mean reversion
- RSI overbought/oversold

Mean reversion strategies aim to profit from:
- Buying when price is oversold (expecting bounce)
- Selling when price is overbought (expecting pullback)
- Exploiting temporary deviations from fair value
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from optimizer_py.strategy_family import StrategyFamily, StrategySignal


@dataclass
class MeanReversionParameters:
    """Parameters for mean-reversion strategy."""

    bb_period: int = 20
    bb_std: float = 2.0
    zscore_threshold: float = 2.0
    rsi_overbought: int = 70
    rsi_oversold: int = 30


class MeanReversionStrategy(StrategyFamily):
    """
    Mean-reversion strategy implementation.

    Generates long signals when:
    - Price touches/breaks lower Bollinger Band
    - Z-score is below -threshold (oversold)
    - RSI is below oversold threshold

    Generates short signals when:
    - Price touches/breaks upper Bollinger Band
    - Z-score is above +threshold (overbought)
    - RSI is above overbought threshold

    Confidence is based on:
    - Distance from bands
    - Z-score magnitude
    - RSI extremity

    Parameters:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Bollinger Bands standard deviation multiplier (default: 2.0)
        zscore_threshold: Z-score threshold for signals (default: 2.0)
        rsi_overbought: RSI overbought level (default: 70)
        rsi_oversold: RSI oversold level (default: 30)
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        zscore_threshold: float = 2.0,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
    ) -> None:
        """Initialize mean-reversion strategy with parameters."""
        self._params = MeanReversionParameters(
            bb_period=bb_period,
            bb_std=bb_std,
            zscore_threshold=zscore_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
        )

    @property
    def name(self) -> str:
        """Strategy family name."""
        return "mean_reversion"

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Current strategy parameters."""
        return {
            "bb_period": self._params.bb_period,
            "bb_std": self._params.bb_std,
            "zscore_threshold": self._params.zscore_threshold,
            "rsi_overbought": self._params.rsi_overbought,
            "rsi_oversold": self._params.rsi_oversold,
        }

    def get_parameter_grid(self) -> dict[str, list[float | int]]:
        """Get parameter grid for optimization."""
        return {
            "bb_period": [10, 20, 30, 50],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
            "zscore_threshold": [1.0, 1.5, 2.0, 2.5],
            "rsi_overbought": [65, 70, 75, 80],
            "rsi_oversold": [20, 25, 30, 35],
        }

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """
        Generate mean-reversion signals from decision frame.

        Args:
            decision_frame: DataFrame with timestamp, symbol, close columns

        Returns:
            List of StrategySignal objects
        """
        if decision_frame.is_empty():
            return []

        # Validate required columns
        required = {"timestamp", "symbol", "close"}
        if not required.issubset(set(decision_frame.columns)):
            missing = required - set(decision_frame.columns)
            raise ValueError(f"Missing required columns: {missing}")

        signals: list[StrategySignal] = []

        # Process each symbol separately
        symbols = decision_frame["symbol"].unique().to_list()

        for symbol in symbols:
            symbol_df = decision_frame.filter(pl.col("symbol") == symbol).sort("timestamp")

            if len(symbol_df) < self._params.bb_period:
                # Not enough data for indicators
                continue

            symbol_signals = self._generate_symbol_signals(symbol_df, symbol)
            signals.extend(symbol_signals)

        return signals

    def _generate_symbol_signals(
        self, df: pl.DataFrame, symbol: str
    ) -> list[StrategySignal]:
        """Generate signals for a single symbol."""
        signals: list[StrategySignal] = []

        # Calculate indicators
        df_with_indicators = self._calculate_indicators(df)

        if df_with_indicators is None:
            return signals

        # Generate signals from indicators
        for row in df_with_indicators.iter_rows(named=True):
            signal = self._evaluate_signal(row, symbol)
            if signal is not None:
                signals.append(signal)

        return signals

    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Calculate all mean-reversion indicators."""
        try:
            close = df["close"]

            # Bollinger Bands
            bb_middle = close.rolling_mean(window_size=self._params.bb_period)
            bb_std = close.rolling_std(window_size=self._params.bb_period)

            # Handle null std (need at least 2 values)
            bb_std_filled = bb_std.fill_null(0.0)

            bb_upper = bb_middle + (self._params.bb_std * bb_std_filled)
            bb_lower = bb_middle - (self._params.bb_std * bb_std_filled)

            # Z-score
            zscore = self._calculate_zscore(close)

            # RSI
            rsi = self._calculate_rsi(close)

            # Percent B (position within Bollinger Bands)
            percent_b = self._calculate_percent_b(close, bb_upper, bb_lower)

            # Add indicators to dataframe
            df_indicators = df.with_columns([
                bb_middle.alias("bb_middle"),
                bb_upper.alias("bb_upper"),
                bb_lower.alias("bb_lower"),
                pl.Series("zscore", zscore),
                pl.Series("rsi", rsi),
                pl.Series("percent_b", percent_b),
            ])

            # Drop rows with nulls (warmup period)
            return df_indicators.drop_nulls(subset=["bb_middle", "bb_upper", "bb_lower"])

        except Exception:
            return None

    def _calculate_zscore(self, close: pl.Series) -> np.ndarray:
        """Calculate rolling Z-score."""
        close_arr = close.to_numpy()
        n = len(close_arr)
        zscore = np.zeros(n)

        for i in range(self._params.bb_period, n):
            window = close_arr[i - self._params.bb_period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                zscore[i] = (close_arr[i] - mean) / std

        return zscore

    def _calculate_rsi(self, close: pl.Series) -> np.ndarray:
        """Calculate RSI indicator."""
        close_arr = close.to_numpy()
        n = len(close_arr)
        rsi_period = min(14, self._params.bb_period)  # Use reasonable RSI period
        rsi = np.full(n, 50.0)  # Default to neutral

        if n < rsi_period + 1:
            return rsi

        # Calculate price changes
        delta = np.diff(close_arr, prepend=close_arr[0])

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Calculate average gain/loss
        for i in range(rsi_period, n):
            avg_gain = np.mean(gains[i - rsi_period + 1:i + 1])
            avg_loss = np.mean(losses[i - rsi_period + 1:i + 1])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0

        return rsi

    def _calculate_percent_b(
        self, close: pl.Series, bb_upper: pl.Series, bb_lower: pl.Series
    ) -> np.ndarray:
        """Calculate Percent B (position within bands)."""
        close_arr = close.to_numpy()
        upper_arr = bb_upper.to_numpy()
        lower_arr = bb_lower.to_numpy()

        n = len(close_arr)
        percent_b = np.full(n, 0.5)  # Default to middle

        for i in range(n):
            if upper_arr[i] is not None and lower_arr[i] is not None:
                band_width = upper_arr[i] - lower_arr[i]
                if band_width > 0:
                    percent_b[i] = (close_arr[i] - lower_arr[i]) / band_width

        return percent_b

    def _evaluate_signal(
        self, row: dict[str, object], symbol: str
    ) -> StrategySignal | None:
        """Evaluate signal from indicator values."""
        # Extract values with type safety
        timestamp = row["timestamp"]
        close = float(row["close"])  # type: ignore[arg-type]
        bb_upper = float(row["bb_upper"])  # type: ignore[arg-type]
        bb_lower = float(row["bb_lower"])  # type: ignore[arg-type]
        bb_middle = float(row["bb_middle"])  # type: ignore[arg-type]
        zscore = float(row["zscore"])  # type: ignore[arg-type]
        rsi = float(row["rsi"])  # type: ignore[arg-type]
        percent_b = float(row["percent_b"])  # type: ignore[arg-type]

        # Ensure timestamp is datetime
        if not isinstance(timestamp, datetime):
            return None

        # Calculate signal components
        bb_signal = self._bollinger_signal(close, bb_upper, bb_lower)
        zscore_signal = self._zscore_signal(zscore)
        rsi_signal = self._rsi_signal(rsi)

        # Combine signals (mean reversion is contrarian)
        combined_signal = self._combine_signals(bb_signal, zscore_signal, rsi_signal)

        if combined_signal == 0:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            close, bb_upper, bb_lower, bb_middle, zscore, rsi, percent_b
        )

        return StrategySignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=combined_signal,
            confidence=confidence,
            metadata={
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_middle": bb_middle,
                "zscore": zscore,
                "rsi": rsi,
                "percent_b": percent_b,
            },
        )

    def _bollinger_signal(
        self, close: float, upper: float, lower: float
    ) -> int:
        """Generate signal from Bollinger Bands."""
        # Price at or below lower band -> buy signal (expect reversion up)
        if close <= lower:
            return 1

        # Price at or above upper band -> sell signal (expect reversion down)
        if close >= upper:
            return -1

        return 0

    def _zscore_signal(self, zscore: float) -> int:
        """Generate signal from Z-score."""
        # Z-score below negative threshold -> oversold -> buy
        if zscore <= -self._params.zscore_threshold:
            return 1

        # Z-score above positive threshold -> overbought -> sell
        if zscore >= self._params.zscore_threshold:
            return -1

        return 0

    def _rsi_signal(self, rsi: float) -> int:
        """Generate signal from RSI."""
        # RSI below oversold -> buy signal
        if rsi <= self._params.rsi_oversold:
            return 1

        # RSI above overbought -> sell signal
        if rsi >= self._params.rsi_overbought:
            return -1

        return 0

    def _combine_signals(
        self, bb_signal: int, zscore_signal: int, rsi_signal: int
    ) -> int:
        """Combine individual signals into final signal."""
        # Count agreement
        signals = [bb_signal, zscore_signal, rsi_signal]
        non_zero = [s for s in signals if s != 0]

        if not non_zero:
            return 0

        # Need majority agreement for signal
        long_count = sum(1 for s in non_zero if s == 1)
        short_count = sum(1 for s in non_zero if s == -1)

        # At least 2 indicators must agree
        if long_count >= 2:
            return 1
        elif short_count >= 2:
            return -1

        # Only one indicator has opinion - need it to be BB (primary)
        if bb_signal != 0 and zscore_signal == 0 and rsi_signal == 0:
            return bb_signal

        return 0

    def _calculate_confidence(
        self,
        close: float,
        bb_upper: float,
        bb_lower: float,
        bb_middle: float,
        zscore: float,
        rsi: float,
        percent_b: float,
    ) -> float:
        """Calculate signal confidence based on indicator strength."""
        confidence = 0.5  # Base confidence

        # Band penetration strength
        if close <= bb_lower:
            band_width = bb_middle - bb_lower
            if band_width > 0:
                penetration = (bb_lower - close) / band_width
                confidence += min(penetration * 0.3, 0.15)
        elif close >= bb_upper:
            band_width = bb_upper - bb_middle
            if band_width > 0:
                penetration = (close - bb_upper) / band_width
                confidence += min(penetration * 0.3, 0.15)

        # Z-score strength
        zscore_excess = abs(zscore) - self._params.zscore_threshold
        if zscore_excess > 0:
            confidence += min(zscore_excess * 0.1, 0.15)

        # RSI strength
        if rsi <= self._params.rsi_oversold:
            rsi_excess = self._params.rsi_oversold - rsi
            confidence += min(rsi_excess / 100, 0.1)
        elif rsi >= self._params.rsi_overbought:
            rsi_excess = rsi - self._params.rsi_overbought
            confidence += min(rsi_excess / 100, 0.1)

        # Percent B extremity
        if percent_b < 0 or percent_b > 1:
            confidence += 0.05

        return min(max(confidence, 0.0), 1.0)
