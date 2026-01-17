"""
GPU-accelerated technical indicators.

Provides high-performance implementations of common technical indicators
using GPU acceleration when available, with automatic CPU fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from backtester_py.gpu.device import get_array_module, is_cuda_available, to_cpu, to_gpu

if TYPE_CHECKING:
    pass


def _validate_window(window: int, name: str = "window") -> None:
    """Validate window parameter."""
    if window <= 0:
        raise ValueError(f"{name} must be positive, got {window}")


def _get_xp(data: np.ndarray) -> Any:
    """Get array module for data, transferring to GPU if available."""
    if is_cuda_available():
        gpu_data = to_gpu(data)
        return get_array_module(gpu_data), gpu_data
    return np, data


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).

    SMA is the unweighted mean of the previous `window` data points.

    Args:
        prices: Array of price data.
        window: Number of periods for the moving average.

    Returns:
        Array of SMA values. First (window-1) values are NaN.

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = sma(prices, window=3)
        >>> result[2]  # First valid value
        2.0
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    xp, data = _get_xp(prices.astype(np.float64))

    result = xp.full(len(data), xp.nan, dtype=np.float64)

    if len(data) < window:
        return to_cpu(result) if is_cuda_available() else result

    # Calculate cumulative sum for efficient SMA
    cumsum = xp.cumsum(data)

    # First SMA value
    result[window - 1] = cumsum[window - 1] / window

    # Subsequent values using cumsum difference
    if len(data) > window:
        result[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return to_cpu(result) if is_cuda_available() else result


def ema(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive
    to new information than SMA.

    Args:
        prices: Array of price data.
        window: Number of periods for the moving average.

    Returns:
        Array of EMA values. First (window-1) values are NaN.

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = ema(prices, window=3)
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    # EMA calculation requires sequential dependency, use CPU
    data = prices.astype(np.float64)
    result = np.full(len(data), np.nan, dtype=np.float64)

    if len(data) < window:
        return result

    alpha = 2.0 / (window + 1)

    # First EMA value is SMA of first window
    result[window - 1] = np.mean(data[:window])

    # Calculate EMA for remaining values
    for i in range(window, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        prices: Array of price data.
        window: Number of periods for RSI calculation (default 14).

    Returns:
        Array of RSI values (0-100). First window values are NaN.

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> prices = np.array([44, 44.25, 44.5, 43.75, 44.5, ...])
        >>> result = rsi(prices, window=14)
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    data = prices.astype(np.float64)
    result = np.full(len(data), np.nan, dtype=np.float64)

    if len(data) <= window:
        return result

    # Calculate price changes
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial averages using SMA
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    # First RSI value
    if avg_loss == 0:
        if avg_gain > 0:
            result[window] = 100.0
        else:
            result[window] = 50.0  # No movement
    else:
        rs = avg_gain / avg_loss
        result[window] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI for remaining values using Wilder's smoothing
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        if avg_loss == 0:
            if avg_gain > 0:
                result[i + 1] = 100.0
            else:
                result[i + 1] = 50.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD shows the relationship between two moving averages of prices.

    Args:
        prices: Array of price data.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).

    Raises:
        ValueError: If any period is not positive.

    Example:
        >>> macd_line, signal_line, histogram = macd(prices)
    """
    _validate_window(fast_period, "fast_period")
    _validate_window(slow_period, "slow_period")
    _validate_window(signal_period, "signal_period")

    if len(prices) == 0:
        empty = np.array([])
        return empty, empty, empty

    # Calculate fast and slow EMAs
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)

    # MACD line = fast EMA - slow EMA
    macd_line = fast_ema - slow_ema

    # Signal line = EMA of MACD line
    # Need to handle NaN values in MACD line
    signal_line = np.full(len(prices), np.nan, dtype=np.float64)

    # Find first valid MACD value
    valid_start = slow_period - 1
    if valid_start < len(macd_line):
        # Calculate EMA of MACD values starting from valid_start
        macd_valid = macd_line[valid_start:]
        if len(macd_valid) >= signal_period:
            signal_ema = ema(macd_valid, signal_period)
            signal_line[valid_start:] = signal_ema

    # Histogram = MACD - Signal
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: np.ndarray,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) with upper and lower
    bands at a specified number of standard deviations.

    Args:
        prices: Array of price data.
        window: Number of periods for the moving average (default 20).
        num_std: Number of standard deviations for bands (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)
    """
    _validate_window(window)

    if len(prices) == 0:
        empty = np.array([])
        return empty, empty, empty

    xp, data = _get_xp(prices.astype(np.float64))

    # Middle band is SMA
    middle_band = sma(prices, window)

    # Calculate rolling standard deviation
    std = rolling_std(prices, window)

    # Upper and lower bands
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std

    return upper_band, middle_band, lower_band


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of an asset price for a period.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        window: Number of periods for ATR calculation (default 14).

    Returns:
        Array of ATR values.

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> result = atr(high, low, close, window=14)
    """
    _validate_window(window)

    if len(close) == 0:
        return np.array([])

    high_arr = high.astype(np.float64)
    low_arr = low.astype(np.float64)
    close_arr = close.astype(np.float64)

    n = len(close_arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return result

    # Calculate True Range
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high_arr[0] - low_arr[0]

    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hpc = abs(high_arr[i] - close_arr[i - 1])
        lpc = abs(low_arr[i] - close_arr[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # ATR is the EMA/SMA of True Range
    # Use Wilder's smoothing (similar to EMA)
    if n <= window:
        return result

    # First ATR is average of first window TRs
    result[window - 1] = np.mean(tr[:window])

    # Subsequent values using Wilder's smoothing
    for i in range(window, n):
        result[i] = (result[i - 1] * (window - 1) + tr[i]) / window

    return result


def rolling_std(prices: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """
    Calculate rolling standard deviation.

    Args:
        prices: Array of price data.
        window: Number of periods for the rolling window.
        ddof: Delta degrees of freedom (default 1 for sample std).

    Returns:
        Array of rolling standard deviation values.

    Raises:
        ValueError: If window is not positive.

    Example:
        >>> result = rolling_std(prices, window=20)
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    xp, data = _get_xp(prices.astype(np.float64))

    n = len(data)
    result = xp.full(n, xp.nan, dtype=np.float64)

    if n < window:
        return to_cpu(result) if is_cuda_available() else result

    # Calculate rolling variance efficiently
    # Var = E[X^2] - E[X]^2
    cumsum = xp.cumsum(data)
    cumsum_sq = xp.cumsum(data ** 2)

    # First window
    mean = cumsum[window - 1] / window
    mean_sq = cumsum_sq[window - 1] / window
    var = mean_sq - mean ** 2
    # Apply Bessel's correction
    if ddof > 0 and window > ddof:
        var = var * window / (window - ddof)
    result[window - 1] = xp.sqrt(xp.maximum(var, 0))

    # Remaining windows
    if n > window:
        sum_window = cumsum[window:] - cumsum[:-window]
        sum_sq_window = cumsum_sq[window:] - cumsum_sq[:-window]

        mean = sum_window / window
        mean_sq = sum_sq_window / window
        var = mean_sq - mean ** 2

        if ddof > 0 and window > ddof:
            var = var * window / (window - ddof)

        result[window:] = xp.sqrt(xp.maximum(var, 0))

    return to_cpu(result) if is_cuda_available() else result


def rolling_mean(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling mean (same as SMA).

    Args:
        prices: Array of price data.
        window: Number of periods for the rolling window.

    Returns:
        Array of rolling mean values.

    Raises:
        ValueError: If window is not positive.
    """
    return sma(prices, window)


def rolling_max(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling maximum.

    Args:
        prices: Array of price data.
        window: Number of periods for the rolling window.

    Returns:
        Array of rolling maximum values.

    Raises:
        ValueError: If window is not positive.
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    data = prices.astype(np.float64)
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        result[i] = np.max(data[i - window + 1 : i + 1])

    return result


def rolling_min(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling minimum.

    Args:
        prices: Array of price data.
        window: Number of periods for the rolling window.

    Returns:
        Array of rolling minimum values.

    Raises:
        ValueError: If window is not positive.
    """
    _validate_window(window)

    if len(prices) == 0:
        return np.array([])

    data = prices.astype(np.float64)
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result

    for i in range(window - 1, n):
        result[i] = np.min(data[i - window + 1 : i + 1])

    return result


def compute_all_indicators(
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    rsi_window: int = 14,
    macd_params: tuple[int, int, int] | None = None,
    bollinger_params: tuple[int, float] | None = None,
    atr_window: int = 14,
) -> dict[str, np.ndarray]:
    """
    Compute multiple technical indicators efficiently.

    Args:
        close: Array of closing prices.
        high: Array of high prices (optional, needed for ATR).
        low: Array of low prices (optional, needed for ATR).
        sma_windows: List of SMA window sizes.
        ema_windows: List of EMA window sizes.
        rsi_window: RSI window size.
        macd_params: Tuple of (fast, slow, signal) periods for MACD.
        bollinger_params: Tuple of (window, num_std) for Bollinger Bands.
        atr_window: ATR window size.

    Returns:
        Dictionary mapping indicator names to arrays.

    Example:
        >>> result = compute_all_indicators(
        ...     close=close_prices,
        ...     high=high_prices,
        ...     low=low_prices,
        ...     sma_windows=[10, 20, 50],
        ...     ema_windows=[12, 26],
        ... )
    """
    result: dict[str, np.ndarray] = {}

    # Default windows if not specified
    if sma_windows is None:
        sma_windows = []
    if ema_windows is None:
        ema_windows = []
    if macd_params is None:
        macd_params = (12, 26, 9)
    if bollinger_params is None:
        bollinger_params = (20, 2.0)

    # SMAs
    for w in sma_windows:
        result[f"sma_{w}"] = sma(close, w)

    # EMAs
    for w in ema_windows:
        result[f"ema_{w}"] = ema(close, w)

    # RSI
    result[f"rsi_{rsi_window}"] = rsi(close, rsi_window)

    # MACD
    fast, slow, signal = macd_params
    macd_line, signal_line, histogram = macd(close, fast, slow, signal)
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram

    # Bollinger Bands
    bb_window, bb_std = bollinger_params
    upper, middle, lower = bollinger_bands(close, bb_window, bb_std)
    result["bollinger_upper"] = upper
    result["bollinger_middle"] = middle
    result["bollinger_lower"] = lower

    # ATR (requires high, low)
    if high is not None and low is not None:
        result[f"atr_{atr_window}"] = atr(high, low, close, atr_window)

    return result


__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "rolling_std",
    "rolling_mean",
    "rolling_max",
    "rolling_min",
    "compute_all_indicators",
]
