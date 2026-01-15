"""
Trend Following Strategy Implementation (T4.02).

Implements trend-following trading signals using:
- Moving Average Crossover (fast/slow MA)
- Momentum Indicators (RSI, Rate of Change)
- Breakout Detection (channel breakouts)

Trend strategies aim to capture directional moves by:
- Going long when uptrend is detected
- Going short when downtrend is detected
- Staying flat in choppy/unclear markets
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from optimizer_py.strategy_family import StrategyFamily, StrategySignal


@dataclass
class TrendParameters:
    """Parameters for trend-following strategy."""

    ma_fast: int = 10
    ma_slow: int = 50
    rsi_period: int = 14
    rsi_threshold: int = 30
    breakout_lookback: int = 20


class TrendStrategy(StrategyFamily):
    """
    Trend-following strategy implementation.

    Generates long signals when:
    - Fast MA crosses above slow MA
    - RSI is not overbought (below 100 - rsi_threshold)
    - Price breaks above recent high channel

    Generates short signals when:
    - Fast MA crosses below slow MA
    - RSI is not oversold (above rsi_threshold)
    - Price breaks below recent low channel

    Confidence is based on:
    - Strength of MA crossover
    - RSI extremity
    - Breakout magnitude

    Parameters:
        ma_fast: Fast moving average period (default: 10)
        ma_slow: Slow moving average period (default: 50)
        rsi_period: RSI calculation period (default: 14)
        rsi_threshold: RSI threshold for overbought/oversold (default: 30)
        breakout_lookback: Lookback for channel breakout (default: 20)
    """

    def __init__(
        self,
        ma_fast: int = 10,
        ma_slow: int = 50,
        rsi_period: int = 14,
        rsi_threshold: int = 30,
        breakout_lookback: int = 20,
    ) -> None:
        """Initialize trend strategy with parameters."""
        self._params = TrendParameters(
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            rsi_period=rsi_period,
            rsi_threshold=rsi_threshold,
            breakout_lookback=breakout_lookback,
        )

    @property
    def name(self) -> str:
        """Strategy family name."""
        return "trend"

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Current strategy parameters."""
        return {
            "ma_fast": self._params.ma_fast,
            "ma_slow": self._params.ma_slow,
            "rsi_period": self._params.rsi_period,
            "rsi_threshold": self._params.rsi_threshold,
            "breakout_lookback": self._params.breakout_lookback,
        }

    def get_parameter_grid(self) -> dict[str, list[float | int]]:
        """Get parameter grid for optimization."""
        return {
            "ma_fast": [5, 10, 20],
            "ma_slow": [20, 50, 100, 200],
            "rsi_period": [7, 14, 21],
            "rsi_threshold": [20, 30, 40],
            "breakout_lookback": [10, 20, 40],
        }

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """
        Generate trend-following signals from decision frame.

        Args:
            decision_frame: DataFrame with timestamp, symbol, close, high, low columns

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

            if len(symbol_df) < self._params.ma_slow:
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
        """Calculate all trend indicators."""
        try:
            close = df["close"]

            # Moving averages
            ma_fast = close.rolling_mean(window_size=self._params.ma_fast)
            ma_slow = close.rolling_mean(window_size=self._params.ma_slow)

            # RSI calculation
            rsi = self._calculate_rsi(close)

            # Rate of change (momentum)
            roc = self._calculate_roc(close, period=self._params.ma_fast)

            # Channel breakout levels
            high_col = df["high"] if "high" in df.columns else close
            low_col = df["low"] if "low" in df.columns else close

            upper_channel = high_col.rolling_max(window_size=self._params.breakout_lookback)
            lower_channel = low_col.rolling_min(window_size=self._params.breakout_lookback)

            # Add indicators to dataframe
            df_indicators = df.with_columns([
                ma_fast.alias("ma_fast"),
                ma_slow.alias("ma_slow"),
                pl.Series("rsi", rsi),
                pl.Series("roc", roc),
                upper_channel.alias("upper_channel"),
                lower_channel.alias("lower_channel"),
            ])

            # Drop rows with nulls (warmup period)
            return df_indicators.drop_nulls(subset=["ma_fast", "ma_slow"])

        except Exception:
            return None

    def _calculate_rsi(self, close: pl.Series) -> np.ndarray:
        """Calculate RSI indicator."""
        close_arr = close.to_numpy()
        n = len(close_arr)
        rsi = np.full(n, 50.0)  # Default to neutral

        if n < self._params.rsi_period + 1:
            return rsi

        # Calculate price changes
        delta = np.diff(close_arr, prepend=close_arr[0])

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Calculate average gain/loss using EMA-style smoothing
        for i in range(self._params.rsi_period, n):
            avg_gain = np.mean(gains[i - self._params.rsi_period + 1:i + 1])
            avg_loss = np.mean(losses[i - self._params.rsi_period + 1:i + 1])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0

        return rsi

    def _calculate_roc(self, close: pl.Series, period: int) -> np.ndarray:
        """Calculate Rate of Change indicator."""
        close_arr = close.to_numpy()
        n = len(close_arr)
        roc = np.zeros(n)

        for i in range(period, n):
            if close_arr[i - period] != 0:
                roc[i] = (close_arr[i] - close_arr[i - period]) / close_arr[i - period] * 100

        return roc

    def _evaluate_signal(
        self, row: dict[str, object], symbol: str
    ) -> StrategySignal | None:
        """Evaluate signal from indicator values."""
        # Extract values with type safety
        timestamp = row["timestamp"]
        close_val = row["close"]
        ma_fast_val = row["ma_fast"]
        ma_slow_val = row["ma_slow"]
        rsi_val = row["rsi"]
        roc_val = row["roc"]
        upper_channel_val = row["upper_channel"]
        lower_channel_val = row["lower_channel"]

        # Check for None values
        if any(
            v is None
            for v in [close_val, ma_fast_val, ma_slow_val, rsi_val, roc_val]
        ):
            return None

        close = float(close_val)  # type: ignore[arg-type]
        ma_fast = float(ma_fast_val)  # type: ignore[arg-type]
        ma_slow = float(ma_slow_val)  # type: ignore[arg-type]
        rsi = float(rsi_val)  # type: ignore[arg-type]
        roc = float(roc_val)  # type: ignore[arg-type]

        # Handle optional channel values
        upper_channel = (
            float(upper_channel_val)  # type: ignore[arg-type]
            if upper_channel_val is not None
            else close * 1.1
        )
        lower_channel = (
            float(lower_channel_val)  # type: ignore[arg-type]
            if lower_channel_val is not None
            else close * 0.9
        )

        # Ensure timestamp is datetime
        from datetime import datetime

        if not isinstance(timestamp, datetime):
            return None

        # Calculate signal components
        ma_signal = self._ma_crossover_signal(ma_fast, ma_slow)
        rsi_filter = self._rsi_filter(rsi, ma_signal)
        breakout_signal, breakout_type = self._breakout_signal(close, upper_channel, lower_channel)

        # Combine signals
        combined_signal = self._combine_signals(ma_signal, rsi_filter, breakout_signal)

        if combined_signal == 0:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            ma_fast, ma_slow, rsi, roc, breakout_signal
        )

        return StrategySignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=combined_signal,
            confidence=confidence,
            metadata={
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "rsi": rsi,
                "roc": roc,
                "breakout": breakout_type,
            },
        )

    def _ma_crossover_signal(self, ma_fast: float, ma_slow: float) -> int:
        """Generate signal from MA crossover."""
        if ma_slow == 0:
            return 0

        ratio = ma_fast / ma_slow

        if ratio > 1.001:  # Fast above slow with margin
            return 1
        elif ratio < 0.999:  # Fast below slow with margin
            return -1
        return 0

    def _rsi_filter(self, rsi: float, ma_signal: int) -> bool:
        """Filter signal based on RSI extremes."""
        overbought = 100 - self._params.rsi_threshold
        oversold = self._params.rsi_threshold

        # Don't go long if overbought
        if ma_signal == 1 and rsi > overbought:
            return False
        # Don't go short if oversold
        return not (ma_signal == -1 and rsi < oversold)

    def _breakout_signal(
        self, close: float, upper: float, lower: float
    ) -> tuple[int, str | None]:
        """Detect channel breakout."""
        if close >= upper * 0.999:  # At or above upper channel
            return 1, "upper"
        elif close <= lower * 1.001:  # At or below lower channel
            return -1, "lower"
        return 0, None

    def _combine_signals(
        self, ma_signal: int, rsi_filter: bool, breakout: int
    ) -> int:
        """Combine individual signals into final signal."""
        if not rsi_filter:
            return 0

        # MA signal is primary, breakout confirms
        if ma_signal != 0:
            if breakout == ma_signal:
                return ma_signal  # Confirmed
            elif breakout == 0:
                return ma_signal  # Unconfirmed but not contradicted
            # Contradicted by breakout - be cautious
            return 0

        # No MA signal but strong breakout
        if breakout != 0:
            return breakout

        return 0

    def _calculate_confidence(
        self,
        ma_fast: float,
        ma_slow: float,
        rsi: float,
        roc: float,
        breakout: int,
    ) -> float:
        """Calculate signal confidence based on indicator strength."""
        confidence = 0.5  # Base confidence

        # MA strength (how far apart are MAs)
        if ma_slow > 0:
            ma_spread = abs(ma_fast - ma_slow) / ma_slow
            confidence += min(ma_spread * 5, 0.2)  # Up to +0.2

        # RSI strength (how extreme)
        rsi_strength = (rsi - 50) / 50 if rsi > 50 else (50 - rsi) / 50
        confidence += rsi_strength * 0.15  # Up to +0.15

        # ROC strength
        roc_strength = min(abs(roc) / 5, 0.1)  # Up to +0.1
        confidence += roc_strength

        # Breakout confirmation
        if breakout != 0:
            confidence += 0.05

        return min(max(confidence, 0.0), 1.0)
