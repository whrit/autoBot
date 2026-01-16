"""
Volatility Strategy Implementation (T4.04).

Implements volatility-based trading signals using:
- ATR (Average True Range) breakout detection
- Volatility targeting for position sizing
- Volatility mean-reversion signals

Volatility strategies aim to:
- Enter positions when price breaks ATR bands (momentum/breakout)
- Scale position sizes inversely to realized volatility (risk management)
- Trade volatility mean-reversion (bet on vol returning to average)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from optimizer_py.strategy_family import StrategyFamily, StrategySignal


@dataclass
class VolatilityParameters:
    """Parameters for volatility strategy."""

    atr_period: int = 14
    atr_multiplier: float = 2.0
    vol_lookback: int = 20
    vol_target: float = 0.15  # Target annualized volatility (15%)
    vol_reversion_threshold: float = 2.0  # Z-score threshold for vol reversion


class VolatilityStrategy(StrategyFamily):
    """
    Volatility-based strategy implementation.

    Generates signals based on:
    - ATR Breakout: Long when price breaks above ATR upper band,
      short when price breaks below ATR lower band
    - Volatility Targeting: Scales position size inversely to realized vol
    - Volatility Mean-Reversion: Signals when vol is extreme vs historical

    Parameters:
        atr_period: Period for ATR calculation (default: 14)
        atr_multiplier: Multiplier for ATR bands (default: 2.0)
        vol_lookback: Lookback for volatility calculation (default: 20)
        vol_target: Target annualized volatility for sizing (default: 0.15)
        vol_reversion_threshold: Z-score threshold for vol signals (default: 2.0)
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        vol_lookback: int = 20,
        vol_target: float = 0.15,
        vol_reversion_threshold: float = 2.0,
    ) -> None:
        """Initialize volatility strategy with parameters."""
        # Validate parameters
        if atr_period <= 0:
            raise ValueError(f"atr_period must be positive, got {atr_period}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")
        if vol_lookback <= 0:
            raise ValueError(f"vol_lookback must be positive, got {vol_lookback}")
        if vol_target <= 0:
            raise ValueError(f"vol_target must be positive, got {vol_target}")

        self._params = VolatilityParameters(
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            vol_lookback=vol_lookback,
            vol_target=vol_target,
            vol_reversion_threshold=vol_reversion_threshold,
        )

    @property
    def name(self) -> str:
        """Strategy family name."""
        return "volatility"

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Current strategy parameters."""
        return {
            "atr_period": self._params.atr_period,
            "atr_multiplier": self._params.atr_multiplier,
            "vol_lookback": self._params.vol_lookback,
            "vol_target": self._params.vol_target,
            "vol_reversion_threshold": self._params.vol_reversion_threshold,
        }

    def get_parameter_grid(self) -> dict[str, list[float | int]]:
        """Get parameter grid for optimization."""
        return {
            "atr_period": [7, 14, 21, 28],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
            "vol_lookback": [10, 20, 30, 50],
            "vol_target": [0.10, 0.15, 0.20, 0.25],
            "vol_reversion_threshold": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """
        Generate volatility-based signals from decision frame.

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

        # Check for high/low columns (required for ATR)
        if "high" not in decision_frame.columns or "low" not in decision_frame.columns:
            raise ValueError(
                "Volatility strategy requires 'high' and 'low' columns for ATR calculation"
            )

        signals: list[StrategySignal] = []

        # Process each symbol separately
        symbols = decision_frame["symbol"].unique().to_list()

        for symbol in symbols:
            symbol_df = decision_frame.filter(pl.col("symbol") == symbol).sort("timestamp")

            min_rows = max(self._params.atr_period, self._params.vol_lookback)
            if len(symbol_df) < min_rows:
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
        """Calculate all volatility indicators."""
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # Calculate ATR
            atr = self._calculate_atr(high, low, close)

            # Calculate ATR bands (using close as center)
            close.to_numpy()
            atr_middle = close.rolling_mean(window_size=self._params.atr_period)

            # ATR bands
            atr_upper = atr_middle.to_numpy() + (self._params.atr_multiplier * atr)
            atr_lower = atr_middle.to_numpy() - (self._params.atr_multiplier * atr)

            # Calculate realized volatility
            realized_vol = self._calculate_realized_volatility(close)

            # Calculate volatility z-score
            vol_zscore = self._calculate_volatility_zscore(realized_vol)

            # Calculate volatility target position size
            vol_target_size = self._calculate_vol_target_size(realized_vol)

            # Add indicators to dataframe
            df_indicators = df.with_columns([
                pl.Series("atr", atr),
                pl.Series("atr_middle", atr_middle),
                pl.Series("atr_upper", atr_upper),
                pl.Series("atr_lower", atr_lower),
                pl.Series("realized_vol", realized_vol),
                pl.Series("vol_zscore", vol_zscore),
                pl.Series("vol_target_size", vol_target_size),
            ])

            # Drop rows with nulls (warmup period)
            return df_indicators.drop_nulls(subset=["atr", "atr_middle"])

        except Exception:
            return None

    def _calculate_atr(
        self, high: pl.Series, low: pl.Series, close: pl.Series
    ) -> np.ndarray:
        """Calculate Average True Range."""
        high_arr = high.to_numpy()
        low_arr = low.to_numpy()
        close_arr = close.to_numpy()
        n = len(close_arr)

        # True Range components
        tr = np.zeros(n)
        tr[0] = high_arr[0] - low_arr[0]

        for i in range(1, n):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i - 1])
            lc = abs(low_arr[i] - close_arr[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR as exponential moving average of TR
        atr = np.zeros(n)
        atr[: self._params.atr_period] = np.nan

        # Initial ATR is simple average
        if n >= self._params.atr_period:
            atr[self._params.atr_period - 1] = np.mean(tr[: self._params.atr_period])

            # Subsequent ATR uses EMA-style smoothing
            multiplier = 2 / (self._params.atr_period + 1)
            for i in range(self._params.atr_period, n):
                atr[i] = (tr[i] - atr[i - 1]) * multiplier + atr[i - 1]

        return atr

    def _calculate_realized_volatility(self, close: pl.Series) -> np.ndarray:
        """Calculate realized volatility (annualized)."""
        close_arr = close.to_numpy()
        n = len(close_arr)
        realized_vol = np.zeros(n)

        # Calculate log returns
        log_returns = np.zeros(n)
        for i in range(1, n):
            if close_arr[i - 1] > 0:
                log_returns[i] = np.log(close_arr[i] / close_arr[i - 1])

        # Rolling standard deviation of returns, annualized
        # Assuming minute data, ~390 minutes per day, ~252 days per year
        annualization_factor = np.sqrt(390 * 252)

        for i in range(self._params.vol_lookback, n):
            window = log_returns[i - self._params.vol_lookback + 1 : i + 1]
            realized_vol[i] = np.std(window) * annualization_factor

        return realized_vol

    def _calculate_volatility_zscore(self, realized_vol: np.ndarray) -> np.ndarray:
        """Calculate z-score of current volatility vs historical."""
        n = len(realized_vol)
        vol_zscore = np.zeros(n)

        # Longer lookback for z-score comparison
        zscore_lookback = self._params.vol_lookback * 2

        for i in range(zscore_lookback, n):
            window = realized_vol[i - zscore_lookback + 1 : i]  # Exclude current
            mean_vol = np.mean(window)
            std_vol = np.std(window)

            if std_vol > 0:
                vol_zscore[i] = (realized_vol[i] - mean_vol) / std_vol

        return vol_zscore

    def _calculate_vol_target_size(self, realized_vol: np.ndarray) -> np.ndarray:
        """Calculate position size factor based on volatility targeting."""
        n = len(realized_vol)
        size_factor = np.ones(n)

        for i in range(n):
            if realized_vol[i] > 0:
                # Size = target_vol / realized_vol
                raw_size = self._params.vol_target / realized_vol[i]
                # Cap position size between 0.25 and 3.0
                size_factor[i] = min(max(raw_size, 0.25), 3.0)

        return size_factor

    def _evaluate_signal(
        self, row: dict[str, object], symbol: str
    ) -> StrategySignal | None:
        """Evaluate signal from indicator values."""
        # Extract values with type safety
        timestamp = row["timestamp"]
        close_val = row["close"]
        atr_val = row["atr"]
        atr_upper_val = row["atr_upper"]
        atr_lower_val = row["atr_lower"]
        realized_vol_val = row["realized_vol"]
        vol_zscore_val = row["vol_zscore"]
        vol_target_size_val = row["vol_target_size"]

        # Check for None values
        if any(
            v is None for v in [close_val, atr_val, atr_upper_val, atr_lower_val]
        ):
            return None

        # Ensure timestamp is datetime
        if not isinstance(timestamp, datetime):
            return None

        close = float(close_val)  # type: ignore[arg-type]
        atr = float(atr_val)  # type: ignore[arg-type]
        atr_upper = float(atr_upper_val)  # type: ignore[arg-type]
        atr_lower = float(atr_lower_val)  # type: ignore[arg-type]
        realized_vol = float(realized_vol_val) if realized_vol_val else 0.0  # type: ignore[arg-type]
        vol_zscore = float(vol_zscore_val) if vol_zscore_val else 0.0  # type: ignore[arg-type]
        vol_target_size = float(vol_target_size_val) if vol_target_size_val else 1.0  # type: ignore[arg-type]

        # Calculate signal components
        breakout_signal, breakout_type = self._atr_breakout_signal(
            close, atr_upper, atr_lower
        )
        vol_reversion_signal, vol_reversion_trigger = self._vol_reversion_signal(
            vol_zscore
        )

        # Combine signals
        combined_signal = self._combine_signals(breakout_signal, vol_reversion_signal)

        if combined_signal == 0:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            close, atr, atr_upper, atr_lower, vol_zscore, breakout_signal
        )

        return StrategySignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=combined_signal,
            confidence=confidence,
            metadata={
                "atr": atr,
                "atr_upper": atr_upper,
                "atr_lower": atr_lower,
                "realized_vol": realized_vol,
                "vol_zscore": vol_zscore,
                "vol_target_size": vol_target_size,
                "breakout_type": breakout_type,
                "vol_reversion_trigger": vol_reversion_trigger,
            },
        )

    def _atr_breakout_signal(
        self, close: float, atr_upper: float, atr_lower: float
    ) -> tuple[int, str | None]:
        """Generate signal from ATR breakout."""
        # Price above upper band -> long breakout
        if close > atr_upper:
            return 1, "upper"
        # Price below lower band -> short breakout
        elif close < atr_lower:
            return -1, "lower"
        return 0, None

    def _vol_reversion_signal(self, vol_zscore: float) -> tuple[int, bool]:
        """
        Generate signal from volatility mean-reversion.

        High vol (positive z-score) suggests selling vol / reducing risk
        Low vol (negative z-score) suggests buying vol / increasing risk

        For price direction:
        - Extremely high vol often precedes reversals (contrarian)
        - This is a secondary signal, not primary
        """
        if vol_zscore >= self._params.vol_reversion_threshold:
            # Very high vol - expect mean reversion (bearish/cautious)
            return -1, True
        elif vol_zscore <= -self._params.vol_reversion_threshold:
            # Very low vol - expect vol expansion (could be either direction)
            # Generally neutral for direction, but can indicate breakout coming
            return 0, True
        return 0, False

    def _combine_signals(self, breakout_signal: int, vol_reversion_signal: int) -> int:
        """Combine individual signals into final signal."""
        # ATR breakout is primary signal
        if breakout_signal != 0:
            # If vol reversion contradicts, reduce conviction but keep direction
            if vol_reversion_signal != 0 and vol_reversion_signal != breakout_signal:
                # Contradiction - still follow breakout but lower confidence
                return breakout_signal
            return breakout_signal

        # No breakout - vol reversion alone is weak signal
        # Only use vol reversion as standalone in extreme cases
        if vol_reversion_signal != 0:
            return vol_reversion_signal

        return 0

    def _calculate_confidence(
        self,
        close: float,
        atr: float,
        atr_upper: float,
        atr_lower: float,
        vol_zscore: float,
        breakout_signal: int,
    ) -> float:
        """Calculate signal confidence based on indicator strength."""
        confidence = 0.5  # Base confidence

        # Breakout strength (how far beyond bands)
        if breakout_signal == 1 and close > atr_upper:
            breakout_excess = (close - atr_upper) / atr if atr > 0 else 0
            confidence += min(breakout_excess * 0.2, 0.2)
        elif breakout_signal == -1 and close < atr_lower:
            breakout_excess = (atr_lower - close) / atr if atr > 0 else 0
            confidence += min(breakout_excess * 0.2, 0.2)

        # Vol z-score strength
        vol_strength = abs(vol_zscore) / 3  # Normalize by typical range
        confidence += min(vol_strength * 0.1, 0.15)

        # ATR magnitude (higher ATR = more volatility = potentially stronger moves)
        if atr > 0 and close > 0:
            atr_pct = atr / close
            if atr_pct > 0.02:  # ATR > 2% of price
                confidence += 0.05

        return min(max(confidence, 0.0), 1.0)
