"""
Tests for GPU-accelerated technical indicators.

TDD tests written BEFORE implementation.
Verifies GPU results match CPU calculations.
"""

from __future__ import annotations

import numpy as np
import pytest


def reference_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """CPU reference implementation of SMA."""
    result = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        result[i] = np.mean(prices[i - window + 1 : i + 1])
    return result


def reference_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """CPU reference implementation of EMA."""
    result = np.zeros(len(prices))
    alpha = 2.0 / (window + 1)

    # First value is SMA
    result[window - 1] = np.mean(prices[:window])

    # Fill earlier values with NaN
    result[:window - 1] = np.nan

    # Calculate EMA
    for i in range(window, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

    return result


def reference_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """CPU reference implementation of RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(prices), np.nan)

    # Use SMA for initial average
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    if avg_loss == 0:
        result[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[window] = 100.0 - (100.0 / (1.0 + rs))

    # Use EMA for subsequent values
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


class TestSMA:
    """Tests for Simple Moving Average (SMA) indicator."""

    def test_sma_basic(self) -> None:
        """SMA should compute correctly for simple case."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma(prices, window=3)

        # Window 3: [nan, nan, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = reference_sma(prices, 3)

        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)], expected[~np.isnan(expected)]
        )

    def test_sma_matches_cpu_reference(self) -> None:
        """GPU SMA should match CPU reference implementation."""
        from backtester_py.gpu.indicators import sma

        np.random.seed(42)
        prices = np.random.randn(1000).cumsum() + 100

        for window in [5, 10, 20, 50]:
            result = sma(prices, window=window)
            expected = reference_sma(prices, window)

            # Compare non-NaN values
            valid_mask = ~np.isnan(expected)
            np.testing.assert_array_almost_equal(
                result[valid_mask], expected[valid_mask], decimal=6
            )

    def test_sma_window_validation(self) -> None:
        """SMA should raise for invalid window sizes."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            sma(prices, window=0)

        with pytest.raises(ValueError):
            sma(prices, window=-1)

    def test_sma_empty_input(self) -> None:
        """SMA should handle empty input gracefully."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([])
        result = sma(prices, window=5)

        assert len(result) == 0

    def test_sma_returns_correct_length(self) -> None:
        """SMA should return array of same length as input."""
        from backtester_py.gpu.indicators import sma

        prices = np.random.randn(100)
        result = sma(prices, window=10)

        assert len(result) == len(prices)


class TestEMA:
    """Tests for Exponential Moving Average (EMA) indicator."""

    def test_ema_basic(self) -> None:
        """EMA should compute correctly for simple case."""
        from backtester_py.gpu.indicators import ema

        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(prices, window=3)

        expected = reference_ema(prices, 3)

        # Compare non-NaN values
        valid_mask = ~np.isnan(expected)
        np.testing.assert_array_almost_equal(
            result[valid_mask], expected[valid_mask], decimal=5
        )

    def test_ema_matches_cpu_reference(self) -> None:
        """GPU EMA should match CPU reference implementation."""
        from backtester_py.gpu.indicators import ema

        np.random.seed(42)
        prices = np.random.randn(1000).cumsum() + 100

        for window in [5, 10, 20, 50]:
            result = ema(prices, window=window)
            expected = reference_ema(prices, window)

            # Compare non-NaN values
            valid_mask = ~np.isnan(expected)
            np.testing.assert_array_almost_equal(
                result[valid_mask], expected[valid_mask], decimal=5
            )

    def test_ema_window_validation(self) -> None:
        """EMA should raise for invalid window sizes."""
        from backtester_py.gpu.indicators import ema

        prices = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            ema(prices, window=0)

    def test_ema_returns_correct_length(self) -> None:
        """EMA should return array of same length as input."""
        from backtester_py.gpu.indicators import ema

        prices = np.random.randn(100)
        result = ema(prices, window=10)

        assert len(result) == len(prices)


class TestRSI:
    """Tests for Relative Strength Index (RSI) indicator."""

    def test_rsi_basic(self) -> None:
        """RSI should compute correctly for simple case."""
        from backtester_py.gpu.indicators import rsi

        prices = np.array([44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5,
                          43.25, 43.5, 44.0, 44.25, 44.5, 44.75, 45.0, 45.25])
        result = rsi(prices, window=14)

        # RSI should be between 0 and 100 for non-NaN values
        valid_values = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid_values)

    def test_rsi_range(self) -> None:
        """RSI should always be between 0 and 100."""
        from backtester_py.gpu.indicators import rsi

        np.random.seed(42)
        prices = np.random.randn(500).cumsum() + 100

        result = rsi(prices, window=14)
        valid_values = result[~np.isnan(result)]

        assert all(0 <= v <= 100 for v in valid_values)

    def test_rsi_matches_cpu_reference(self) -> None:
        """GPU RSI should match CPU reference implementation."""
        from backtester_py.gpu.indicators import rsi

        np.random.seed(42)
        prices = np.random.randn(200).cumsum() + 100

        result = rsi(prices, window=14)
        expected = reference_rsi(prices, 14)

        # Compare non-NaN values
        valid_mask = ~np.isnan(expected)
        np.testing.assert_array_almost_equal(
            result[valid_mask], expected[valid_mask], decimal=4
        )

    def test_rsi_window_validation(self) -> None:
        """RSI should raise for invalid window sizes."""
        from backtester_py.gpu.indicators import rsi

        prices = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            rsi(prices, window=0)

    def test_rsi_all_gains(self) -> None:
        """RSI should be 100 for continuously rising prices."""
        from backtester_py.gpu.indicators import rsi

        # Prices that only go up
        prices = np.arange(1.0, 50.0)
        result = rsi(prices, window=14)

        # After warmup, RSI should approach 100
        valid_values = result[~np.isnan(result)]
        assert all(v > 90 for v in valid_values[-10:])

    def test_rsi_all_losses(self) -> None:
        """RSI should be near 0 for continuously falling prices."""
        from backtester_py.gpu.indicators import rsi

        # Prices that only go down
        prices = np.arange(50.0, 1.0, -1.0)
        result = rsi(prices, window=14)

        # After warmup, RSI should approach 0
        valid_values = result[~np.isnan(result)]
        assert all(v < 10 for v in valid_values[-10:])


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_basic(self) -> None:
        """MACD should compute correctly for simple case."""
        from backtester_py.gpu.indicators import macd

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        macd_line, signal_line, histogram = macd(prices)

        # Check shapes
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_macd_histogram_equals_difference(self) -> None:
        """MACD histogram should equal MACD line minus signal line."""
        from backtester_py.gpu.indicators import macd

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        macd_line, signal_line, histogram = macd(prices)

        # Histogram = MACD - Signal
        valid_mask = ~(np.isnan(macd_line) | np.isnan(signal_line))
        expected_hist = macd_line[valid_mask] - signal_line[valid_mask]

        np.testing.assert_array_almost_equal(
            histogram[valid_mask], expected_hist, decimal=6
        )

    def test_macd_custom_periods(self) -> None:
        """MACD should accept custom periods."""
        from backtester_py.gpu.indicators import macd

        np.random.seed(42)
        prices = np.random.randn(200).cumsum() + 100

        # Custom periods: fast=8, slow=21, signal=9
        macd_line, signal_line, histogram = macd(
            prices, fast_period=8, slow_period=21, signal_period=9
        )

        assert len(macd_line) == len(prices)


class TestBollingerBands:
    """Tests for Bollinger Bands indicator."""

    def test_bollinger_basic(self) -> None:
        """Bollinger Bands should compute correctly."""
        from backtester_py.gpu.indicators import bollinger_bands

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        # Check shapes
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_bollinger_middle_equals_sma(self) -> None:
        """Bollinger middle band should equal SMA."""
        from backtester_py.gpu.indicators import bollinger_bands, sma

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)
        sma_result = sma(prices, window=20)

        valid_mask = ~np.isnan(middle)
        np.testing.assert_array_almost_equal(
            middle[valid_mask], sma_result[valid_mask], decimal=6
        )

    def test_bollinger_symmetry(self) -> None:
        """Bollinger upper and lower bands should be symmetric around middle."""
        from backtester_py.gpu.indicators import bollinger_bands

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        valid_mask = ~np.isnan(middle)

        # Upper - Middle should equal Middle - Lower
        upper_diff = upper[valid_mask] - middle[valid_mask]
        lower_diff = middle[valid_mask] - lower[valid_mask]

        np.testing.assert_array_almost_equal(upper_diff, lower_diff, decimal=6)

    def test_bollinger_upper_greater_than_lower(self) -> None:
        """Bollinger upper band should always be >= lower band."""
        from backtester_py.gpu.indicators import bollinger_bands

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        valid_mask = ~np.isnan(upper)
        assert all(upper[valid_mask] >= lower[valid_mask])


class TestATR:
    """Tests for Average True Range (ATR) indicator."""

    def test_atr_basic(self) -> None:
        """ATR should compute correctly for OHLC data."""
        from backtester_py.gpu.indicators import atr

        np.random.seed(42)
        n = 100

        # Generate OHLC data
        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        result = atr(high, low, close, window=14)

        assert len(result) == n

    def test_atr_positive(self) -> None:
        """ATR should always be positive."""
        from backtester_py.gpu.indicators import atr

        np.random.seed(42)
        n = 100

        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        result = atr(high, low, close, window=14)

        valid_values = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid_values)

    def test_atr_window_validation(self) -> None:
        """ATR should raise for invalid window sizes."""
        from backtester_py.gpu.indicators import atr

        high = np.array([1.0, 2.0, 3.0])
        low = np.array([0.5, 1.5, 2.5])
        close = np.array([0.8, 1.8, 2.8])

        with pytest.raises(ValueError):
            atr(high, low, close, window=0)


class TestStandardDeviation:
    """Tests for rolling standard deviation indicator."""

    def test_std_basic(self) -> None:
        """Rolling std should compute correctly."""
        from backtester_py.gpu.indicators import rolling_std

        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = rolling_std(prices, window=3)

        assert len(result) == len(prices)

    def test_std_matches_numpy(self) -> None:
        """Rolling std should match numpy calculation."""
        from backtester_py.gpu.indicators import rolling_std

        np.random.seed(42)
        prices = np.random.randn(100).cumsum() + 100
        window = 20

        result = rolling_std(prices, window=window)

        # Calculate reference using numpy
        reference = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            reference[i] = np.std(prices[i - window + 1 : i + 1], ddof=1)

        valid_mask = ~np.isnan(reference)
        np.testing.assert_array_almost_equal(
            result[valid_mask], reference[valid_mask], decimal=6
        )


class TestIndicatorBatchProcessing:
    """Tests for batch processing multiple indicators."""

    def test_compute_all_indicators(self) -> None:
        """compute_all_indicators should compute all supported indicators."""
        from backtester_py.gpu.indicators import compute_all_indicators

        np.random.seed(42)
        n = 200

        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        result = compute_all_indicators(
            close=close,
            high=high,
            low=low,
            sma_windows=[10, 20],
            ema_windows=[12, 26],
            rsi_window=14,
            macd_params=(12, 26, 9),
            bollinger_params=(20, 2.0),
            atr_window=14,
        )

        # Check all indicators are present
        assert "sma_10" in result
        assert "sma_20" in result
        assert "ema_12" in result
        assert "ema_26" in result
        assert "rsi_14" in result
        assert "macd" in result
        assert "macd_signal" in result
        assert "macd_histogram" in result
        assert "bollinger_upper" in result
        assert "bollinger_middle" in result
        assert "bollinger_lower" in result
        assert "atr_14" in result

        # Check shapes
        for key, value in result.items():
            assert len(value) == n, f"{key} has wrong length"


class TestIndicatorEdgeCases:
    """Tests for edge cases in indicator calculations."""

    def test_sma_with_nan_values(self) -> None:
        """SMA should handle NaN values appropriately."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma(prices, window=3)

        # SMA with NaN in window should produce NaN
        # After NaN passes out of window, should be valid
        assert len(result) == len(prices)

    def test_ema_with_nan_values(self) -> None:
        """EMA should handle NaN values appropriately."""
        from backtester_py.gpu.indicators import ema

        prices = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(prices, window=3)

        assert len(result) == len(prices)

    def test_rsi_with_constant_prices(self) -> None:
        """RSI should handle constant prices (no gains/losses)."""
        from backtester_py.gpu.indicators import rsi

        prices = np.full(50, 100.0)
        result = rsi(prices, window=14)

        # RSI should be 50 or NaN for constant prices (no movement)
        valid_values = result[~np.isnan(result)]
        # With no gains or losses, RSI is undefined or 50
        if len(valid_values) > 0:
            assert all(40 <= v <= 60 for v in valid_values) or all(np.isnan(valid_values))

    def test_sma_window_larger_than_data(self) -> None:
        """SMA should handle window larger than data length."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([1.0, 2.0, 3.0])
        result = sma(prices, window=5)

        # All values should be NaN
        assert all(np.isnan(result))

    def test_ema_window_larger_than_data(self) -> None:
        """EMA should handle window larger than data length."""
        from backtester_py.gpu.indicators import ema

        prices = np.array([1.0, 2.0, 3.0])
        result = ema(prices, window=5)

        # All values should be NaN
        assert all(np.isnan(result))

    def test_bollinger_with_zero_std(self) -> None:
        """Bollinger bands should handle zero standard deviation."""
        from backtester_py.gpu.indicators import bollinger_bands

        # Constant prices = zero std
        prices = np.full(50, 100.0)
        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        # With zero std, upper = middle = lower
        valid_mask = ~np.isnan(middle)
        if np.any(valid_mask):
            np.testing.assert_array_almost_equal(
                upper[valid_mask], middle[valid_mask], decimal=10
            )
            np.testing.assert_array_almost_equal(
                lower[valid_mask], middle[valid_mask], decimal=10
            )


class TestIndicatorMarketPatterns:
    """Tests for indicators with specific market patterns."""

    def test_sma_trending_up(self) -> None:
        """SMA should lag behind in uptrend."""
        from backtester_py.gpu.indicators import sma

        # Strong uptrend
        prices = np.linspace(100, 150, 100)
        result = sma(prices, window=10)

        # SMA should be below price in uptrend
        valid_mask = ~np.isnan(result)
        sma_values = result[valid_mask]
        price_values = prices[valid_mask]

        # Most SMA values should be below price
        assert np.sum(sma_values < price_values) > len(sma_values) * 0.9

    def test_sma_trending_down(self) -> None:
        """SMA should lag behind in downtrend."""
        from backtester_py.gpu.indicators import sma

        # Strong downtrend
        prices = np.linspace(150, 100, 100)
        result = sma(prices, window=10)

        # SMA should be above price in downtrend
        valid_mask = ~np.isnan(result)
        sma_values = result[valid_mask]
        price_values = prices[valid_mask]

        # Most SMA values should be above price
        assert np.sum(sma_values > price_values) > len(sma_values) * 0.9

    def test_ema_more_responsive_than_sma(self) -> None:
        """EMA should respond faster to price changes than SMA."""
        from backtester_py.gpu.indicators import ema, sma

        # Create price with sudden jump
        prices = np.concatenate([
            np.full(50, 100.0),
            np.full(50, 110.0)
        ])

        window = 10
        sma_result = sma(prices, window=window)
        ema_result = ema(prices, window=window)

        # After the jump, EMA should reach new level faster
        # Check at index 55 (5 periods after jump)
        idx = 55
        assert ema_result[idx] > sma_result[idx]

    def test_rsi_overbought_in_strong_uptrend(self) -> None:
        """RSI should be overbought (>70) in strong uptrend."""
        from backtester_py.gpu.indicators import rsi

        prices = np.linspace(100, 200, 100)  # Strong uptrend
        result = rsi(prices, window=14)

        valid_values = result[~np.isnan(result)]
        # Most values should be overbought
        assert np.mean(valid_values > 70) > 0.8

    def test_rsi_oversold_in_strong_downtrend(self) -> None:
        """RSI should be oversold (<30) in strong downtrend."""
        from backtester_py.gpu.indicators import rsi

        prices = np.linspace(200, 100, 100)  # Strong downtrend
        result = rsi(prices, window=14)

        valid_values = result[~np.isnan(result)]
        # Most values should be oversold
        assert np.mean(valid_values < 30) > 0.8

    def test_bollinger_bandwidth_increases_with_volatility(self) -> None:
        """Bollinger bandwidth should increase during high volatility."""
        from backtester_py.gpu.indicators import bollinger_bands

        # Low volatility period followed by high volatility
        np.random.seed(42)
        low_vol = 100 + np.random.randn(50) * 0.5
        high_vol = 100 + np.random.randn(50) * 5.0
        prices = np.concatenate([low_vol, high_vol])

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        # Calculate bandwidth
        bandwidth = upper - lower

        # Average bandwidth in high vol period should be greater
        valid_mask = ~np.isnan(bandwidth)
        if np.sum(valid_mask[50:]) > 10:  # Need enough valid points
            low_vol_bw = np.nanmean(bandwidth[30:50])
            high_vol_bw = np.nanmean(bandwidth[70:])
            assert high_vol_bw > low_vol_bw

    def test_atr_increases_with_volatility(self) -> None:
        """ATR should increase during high volatility periods."""
        from backtester_py.gpu.indicators import atr

        np.random.seed(42)
        n = 100

        # Low volatility followed by high volatility
        close = np.concatenate([
            100 + np.cumsum(np.random.randn(50) * 0.1),
            100 + np.cumsum(np.random.randn(50) * 2.0)
        ])
        high = close + np.abs(np.random.randn(n)) * np.concatenate([
            np.full(50, 0.5),
            np.full(50, 3.0)
        ])
        low = close - np.abs(np.random.randn(n)) * np.concatenate([
            np.full(50, 0.5),
            np.full(50, 3.0)
        ])

        result = atr(high, low, close, window=14)

        # Average ATR in high vol period should be greater
        valid_mask = ~np.isnan(result)
        if np.sum(valid_mask[50:]) > 10:
            low_vol_atr = np.nanmean(result[30:50])
            high_vol_atr = np.nanmean(result[70:])
            assert high_vol_atr > low_vol_atr


class TestIndicatorCpuGpuConsistency:
    """Tests verifying GPU indicators match CPU calculations."""

    def test_sma_gpu_cpu_match_large_dataset(self) -> None:
        """GPU SMA should match CPU on large dataset."""
        from backtester_py.gpu.indicators import sma

        np.random.seed(42)
        prices = np.random.randn(10000).cumsum() + 100

        result = sma(prices, window=50)
        expected = reference_sma(prices, 50)

        valid_mask = ~np.isnan(expected)
        np.testing.assert_allclose(
            result[valid_mask], expected[valid_mask], rtol=1e-6
        )

    def test_ema_gpu_cpu_match_large_dataset(self) -> None:
        """GPU EMA should match CPU on large dataset."""
        from backtester_py.gpu.indicators import ema

        np.random.seed(42)
        prices = np.random.randn(10000).cumsum() + 100

        result = ema(prices, window=50)
        expected = reference_ema(prices, 50)

        valid_mask = ~np.isnan(expected)
        np.testing.assert_allclose(
            result[valid_mask], expected[valid_mask], rtol=1e-5
        )

    def test_rsi_gpu_cpu_match_large_dataset(self) -> None:
        """GPU RSI should match CPU on large dataset."""
        from backtester_py.gpu.indicators import rsi

        np.random.seed(42)
        prices = np.random.randn(5000).cumsum() + 100

        result = rsi(prices, window=14)
        expected = reference_rsi(prices, 14)

        valid_mask = ~np.isnan(expected)
        np.testing.assert_allclose(
            result[valid_mask], expected[valid_mask], rtol=1e-4
        )

    def test_all_indicators_deterministic(self) -> None:
        """Indicators should produce identical results on repeated calls."""
        from backtester_py.gpu.indicators import bollinger_bands, ema, rsi, sma

        np.random.seed(42)
        prices = np.random.randn(1000).cumsum() + 100

        # Run twice
        sma_1 = sma(prices, window=20)
        sma_2 = sma(prices, window=20)
        np.testing.assert_array_equal(sma_1, sma_2)

        ema_1 = ema(prices, window=20)
        ema_2 = ema(prices, window=20)
        np.testing.assert_array_equal(ema_1, ema_2)

        rsi_1 = rsi(prices, window=14)
        rsi_2 = rsi(prices, window=14)
        np.testing.assert_array_equal(rsi_1, rsi_2)

        bb_1 = bollinger_bands(prices, window=20, num_std=2.0)
        bb_2 = bollinger_bands(prices, window=20, num_std=2.0)
        for b1, b2 in zip(bb_1, bb_2):
            np.testing.assert_array_equal(b1, b2)


class TestIndicatorPerformance:
    """Tests for indicator performance with large datasets."""

    def test_sma_large_dataset_performance(self) -> None:
        """SMA should handle large datasets efficiently."""
        from backtester_py.gpu.indicators import sma

        np.random.seed(42)
        prices = np.random.randn(100_000).cumsum() + 100

        # Should complete without timeout
        result = sma(prices, window=50)
        assert len(result) == len(prices)

    def test_ema_large_dataset_performance(self) -> None:
        """EMA should handle large datasets efficiently."""
        from backtester_py.gpu.indicators import ema

        np.random.seed(42)
        prices = np.random.randn(100_000).cumsum() + 100

        result = ema(prices, window=50)
        assert len(result) == len(prices)

    def test_rsi_large_dataset_performance(self) -> None:
        """RSI should handle large datasets efficiently."""
        from backtester_py.gpu.indicators import rsi

        np.random.seed(42)
        prices = np.random.randn(100_000).cumsum() + 100

        result = rsi(prices, window=14)
        assert len(result) == len(prices)

    def test_bollinger_large_dataset_performance(self) -> None:
        """Bollinger bands should handle large datasets efficiently."""
        from backtester_py.gpu.indicators import bollinger_bands

        np.random.seed(42)
        prices = np.random.randn(100_000).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_batch_indicators_large_dataset(self) -> None:
        """Batch indicator computation should handle large datasets."""
        from backtester_py.gpu.indicators import compute_all_indicators

        np.random.seed(42)
        n = 50_000

        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        result = compute_all_indicators(
            close=close,
            high=high,
            low=low,
            sma_windows=[10, 20, 50],
            ema_windows=[12, 26],
            rsi_window=14,
            macd_params=(12, 26, 9),
            bollinger_params=(20, 2.0),
            atr_window=14,
        )

        assert len(result) > 0
        for key, value in result.items():
            assert len(value) == n, f"{key} has wrong length"


class TestIndicatorFallback:
    """Tests for graceful fallback when GPU unavailable."""

    def test_sma_fallback(self) -> None:
        """SMA should work regardless of GPU availability."""
        from backtester_py.gpu.indicators import sma

        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma(prices, window=3)

        # Should produce valid output
        assert len(result) == len(prices)
        # First (window-1) values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_ema_fallback(self) -> None:
        """EMA should work regardless of GPU availability."""
        from backtester_py.gpu.indicators import ema

        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(prices, window=3)

        assert len(result) == len(prices)

    def test_rsi_fallback(self) -> None:
        """RSI should work regardless of GPU availability."""
        from backtester_py.gpu.indicators import rsi

        prices = np.array([44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5,
                          43.25, 43.5, 44.0, 44.25, 44.5, 44.75, 45.0, 45.25])
        result = rsi(prices, window=14)

        assert len(result) == len(prices)
        valid_values = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid_values)

    def test_bollinger_fallback(self) -> None:
        """Bollinger bands should work regardless of GPU availability."""
        from backtester_py.gpu.indicators import bollinger_bands

        np.random.seed(42)
        prices = np.random.randn(50).cumsum() + 100

        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
