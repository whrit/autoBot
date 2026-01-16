"""Pytest fixtures for viz_utils tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from viz_utils import BacktestData, TradeData


@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Generate sample timestamps for 252 trading days."""
    base = datetime(2023, 1, 3)
    timestamps = []
    current = base
    for _ in range(252):
        timestamps.append(current)
        # Skip weekends
        current += timedelta(days=1)
        while current.weekday() >= 5:  # Saturday or Sunday
            current += timedelta(days=1)
    return timestamps


@pytest.fixture
def sample_equity(sample_timestamps: list[datetime]) -> list[float]:
    """Generate sample equity curve with realistic returns."""
    np.random.seed(42)
    n = len(sample_timestamps)

    # Generate daily returns with slight positive drift
    daily_returns = np.random.normal(0.0005, 0.015, n)

    # Create equity curve
    equity = [100_000.0]
    for ret in daily_returns[1:]:
        equity.append(equity[-1] * (1 + ret))

    return equity


@pytest.fixture
def sample_returns(sample_equity: list[float]) -> list[float]:
    """Calculate returns from equity curve."""
    equity_arr = np.array(sample_equity, dtype=np.float64)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    return [0.0] + returns.tolist()


@pytest.fixture
def sample_trades(sample_timestamps: list[datetime]) -> list[TradeData]:
    """Generate sample trades."""
    np.random.seed(42)
    trades = []
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    # Generate 50 trades over the period
    n_trades = 50
    trade_indices = sorted(np.random.choice(len(sample_timestamps) - 5, n_trades, replace=False))

    for idx in trade_indices:
        entry_time = sample_timestamps[idx]
        duration = np.random.randint(1, 5)
        exit_time = sample_timestamps[min(idx + duration, len(sample_timestamps) - 1)]

        symbol = np.random.choice(symbols)
        side = np.random.choice(["long", "short"])
        notional = np.random.uniform(5000, 20000)

        # Random P&L with slight positive skew
        pnl = np.random.normal(100, 500)
        return_pct = pnl / notional

        trades.append(
            TradeData(
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=symbol,
                side=side,
                pnl=pnl,
                return_pct=return_pct,
                notional=notional,
            )
        )

    return trades


@pytest.fixture
def sample_benchmark_equity(sample_timestamps: list[datetime]) -> list[float]:
    """Generate benchmark equity curve (like S&P 500)."""
    np.random.seed(123)
    n = len(sample_timestamps)

    # Generate daily returns with lower volatility
    daily_returns = np.random.normal(0.0003, 0.01, n)

    # Create equity curve starting at same value
    equity = [100_000.0]
    for ret in daily_returns[1:]:
        equity.append(equity[-1] * (1 + ret))

    return equity


@pytest.fixture
def sample_polars_equity_curve(
    sample_timestamps: list[datetime],
    sample_equity: list[float],
    sample_returns: list[float],
) -> pl.DataFrame:
    """Create sample Polars DataFrame for equity curve."""
    return pl.DataFrame(
        {
            "timestamp": sample_timestamps,
            "equity": sample_equity,
            "returns": sample_returns,
        }
    )


@pytest.fixture
def sample_polars_benchmark(
    sample_timestamps: list[datetime],
    sample_benchmark_equity: list[float],
) -> pl.DataFrame:
    """Create sample Polars DataFrame for benchmark."""
    return pl.DataFrame(
        {
            "timestamp": sample_timestamps,
            "equity": sample_benchmark_equity,
        }
    )


@pytest.fixture
def sample_backtest_data(
    sample_timestamps: list[datetime],
    sample_equity: list[float],
    sample_returns: list[float],
    sample_trades: list[TradeData],
    sample_benchmark_equity: list[float],
) -> BacktestData:
    """Create complete BacktestData fixture."""
    return BacktestData(
        timestamps=sample_timestamps,
        equity=sample_equity,
        returns=sample_returns,
        trades=sample_trades,
        benchmark_equity=sample_benchmark_equity,
        initial_capital=100_000.0,
    )


@pytest.fixture
def minimal_backtest_data() -> BacktestData:
    """Create minimal BacktestData for edge case testing."""
    timestamps = [
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
        datetime(2023, 1, 5),
    ]
    return BacktestData(
        timestamps=timestamps,
        equity=[100_000.0, 100_500.0, 100_200.0],
        returns=[0.0, 0.005, -0.003],
        trades=[],
        initial_capital=100_000.0,
    )


@pytest.fixture
def empty_backtest_data() -> BacktestData:
    """Create empty BacktestData for edge case testing."""
    return BacktestData(
        timestamps=[datetime(2023, 1, 3)],
        equity=[100_000.0],
        returns=[0.0],
        trades=[],
        initial_capital=100_000.0,
    )


@pytest.fixture
def sample_regimes(sample_timestamps: list[datetime]) -> list[tuple[datetime, datetime, str]]:
    """Create sample market regimes."""
    n = len(sample_timestamps)
    quarter = n // 4

    return [
        (sample_timestamps[0], sample_timestamps[quarter - 1], "bull"),
        (sample_timestamps[quarter], sample_timestamps[quarter * 2 - 1], "bear"),
        (sample_timestamps[quarter * 2], sample_timestamps[quarter * 3 - 1], "neutral"),
        (sample_timestamps[quarter * 3], sample_timestamps[-1], "high_vol"),
    ]


@pytest.fixture
def sample_positions(
    sample_timestamps: list[datetime],
) -> list[tuple[datetime, dict[str, float]]]:
    """Create sample position snapshots."""
    np.random.seed(42)
    positions = []
    symbols = ["AAPL", "GOOGL", "MSFT"]

    for ts in sample_timestamps[::5]:  # Every 5th timestamp
        pos_dict = {}
        for symbol in symbols:
            if np.random.random() > 0.3:  # 70% chance of having a position
                pos_dict[symbol] = np.random.uniform(-10000, 20000)
        positions.append((ts, pos_dict))

    return positions


@pytest.fixture
def sample_exposures(
    sample_timestamps: list[datetime],
) -> list[tuple[datetime, dict[str, float]]]:
    """Create sample exposure snapshots."""
    np.random.seed(42)
    exposures = []

    for ts in sample_timestamps[::5]:  # Every 5th timestamp
        exp_dict = {
            "gross": np.random.uniform(0.5, 1.5),
            "net": np.random.uniform(-0.3, 0.5),
            "long": np.random.uniform(0.3, 1.0),
            "short": np.random.uniform(0.0, 0.5),
        }
        exposures.append((ts, exp_dict))

    return exposures
