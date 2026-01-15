"""Pytest configuration and shared fixtures for optimizer_py tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_decision_frame() -> pl.DataFrame:
    """
    Create a sample decision frame for testing strategies.

    Includes:
    - timestamp: datetime column
    - symbol: str column
    - close: price column
    - high, low, open: OHLC columns
    - volume: volume column
    - bid_price, ask_price: quote columns
    - spread_bps: spread in basis points
    """
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    # Generate synthetic price data with trend and noise
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(n_rows) * 0.002  # 0.2% std per bar
    # Add a trend component
    trend = np.linspace(0, 0.05, n_rows)  # 5% trend over period
    prices = base_price * np.cumprod(1 + returns + trend / n_rows)

    # Generate OHLC
    highs = prices * (1 + np.abs(np.random.randn(n_rows)) * 0.001)
    lows = prices * (1 - np.abs(np.random.randn(n_rows)) * 0.001)
    opens = np.roll(prices, 1)
    opens[0] = base_price

    # Generate volume
    volumes = np.random.randint(10000, 100000, size=n_rows).astype(float)

    # Generate bid/ask (spread around close)
    spread_bps = np.random.uniform(1, 5, size=n_rows)
    spread_pct = spread_bps / 10000
    bid_prices = prices * (1 - spread_pct / 2)
    ask_prices = prices * (1 + spread_pct / 2)

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["AAPL"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": bid_prices,
        "ask_price": ask_prices,
        "spread_bps": spread_bps,
    })


@pytest.fixture
def trending_decision_frame() -> pl.DataFrame:
    """Create a decision frame with clear uptrend for testing trend strategies."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(123)
    base_price = 100.0
    # Strong uptrend with minimal noise
    trend = np.linspace(0, 0.20, n_rows)  # 20% uptrend
    noise = np.random.randn(n_rows) * 0.001
    prices = base_price * (1 + trend + noise)

    highs = prices * 1.002
    lows = prices * 0.998
    opens = np.roll(prices, 1)
    opens[0] = base_price
    volumes = np.random.randint(50000, 150000, size=n_rows).astype(float)
    spread_bps = np.ones(n_rows) * 2.0

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["SPY"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": prices * 0.9999,
        "ask_price": prices * 1.0001,
        "spread_bps": spread_bps,
    })


@pytest.fixture
def mean_reverting_decision_frame() -> pl.DataFrame:
    """Create a decision frame with mean-reverting behavior for testing."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(456)
    base_price = 100.0
    # Mean-reverting around base price
    oscillation = np.sin(np.linspace(0, 4 * np.pi, n_rows)) * 0.03
    noise = np.random.randn(n_rows) * 0.002
    prices = base_price * (1 + oscillation + noise)

    highs = prices * 1.003
    lows = prices * 0.997
    opens = np.roll(prices, 1)
    opens[0] = base_price
    volumes = np.random.randint(30000, 80000, size=n_rows).astype(float)
    spread_bps = np.ones(n_rows) * 3.0

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["XLF"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": prices * 0.99985,
        "ask_price": prices * 1.00015,
        "spread_bps": spread_bps,
    })


@pytest.fixture
def multi_symbol_decision_frame() -> pl.DataFrame:
    """Create a decision frame with multiple symbols."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows_per_symbol = 50
    symbols = ["AAPL", "MSFT", "GOOGL"]

    np.random.seed(789)
    frames = []

    for symbol in symbols:
        base_price = {"AAPL": 180.0, "MSFT": 400.0, "GOOGL": 150.0}[symbol]
        returns = np.random.randn(n_rows_per_symbol) * 0.003
        prices = base_price * np.cumprod(1 + returns)

        frames.append(pl.DataFrame({
            "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows_per_symbol)],
            "symbol": [symbol] * n_rows_per_symbol,
            "open": np.roll(prices, 1),
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.randint(20000, 100000, size=n_rows_per_symbol).astype(float),
            "bid_price": prices * 0.9999,
            "ask_price": prices * 1.0001,
            "spread_bps": np.ones(n_rows_per_symbol) * 2.5,
        }))

    return pl.concat(frames).sort(["timestamp", "symbol"])


@pytest.fixture
def sample_labeled_frame(sample_decision_frame: pl.DataFrame) -> pl.DataFrame:
    """Create a sample labeled decision frame with targets for ML training."""
    np.random.seed(42)
    n_rows = len(sample_decision_frame)
    prices = sample_decision_frame["close"].to_numpy()

    # Generate forward returns
    forward_returns = np.diff(prices, append=prices[-1]) / prices
    # Shift forward to simulate actual forward return
    forward_returns = np.roll(forward_returns, -1)
    forward_returns[-1] = 0.0

    # Direction labels based on returns
    direction = np.zeros(n_rows, dtype=int)
    direction[forward_returns > 0.001] = 1  # Long signal if return > 10 bps
    direction[forward_returns < -0.001] = -1  # Short signal if return < -10 bps

    # Add features for ML
    returns_1 = np.diff(prices, prepend=prices[0]) / np.roll(prices, 1)
    returns_1[0] = 0
    returns_5 = np.convolve(returns_1, np.ones(5) / 5, mode="same")

    # RSI-like feature
    gains = np.where(returns_1 > 0, returns_1, 0)
    losses = np.where(returns_1 < 0, -returns_1, 0)
    avg_gain = np.convolve(gains, np.ones(14) / 14, mode="same")
    avg_loss = np.convolve(losses, np.ones(14) / 14, mode="same")
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))

    # Volatility features
    vol_5 = np.array([np.std(returns_1[max(0, i - 5):i + 1]) if i >= 5 else 0.001 for i in range(n_rows)])
    vol_20 = np.array([np.std(returns_1[max(0, i - 20):i + 1]) if i >= 20 else 0.001 for i in range(n_rows)])

    return sample_decision_frame.with_columns([
        pl.Series("forward_return_60s", forward_returns),
        pl.Series("direction_60s", direction),
        pl.Series("return_1", returns_1),
        pl.Series("return_5", returns_5),
        pl.Series("rsi_14", rsi_14),
        pl.Series("vol_5", vol_5),
        pl.Series("vol_20", vol_20),
        pl.Series("volume_imbalance", np.random.uniform(-0.5, 0.5, n_rows)),
    ])


@pytest.fixture
def small_parameter_grid() -> dict[str, list]:
    """Small parameter grid for testing sweep functions."""
    return {
        "lookback": [5, 10],
        "threshold": [0.5, 1.0],
    }


@pytest.fixture
def ml_feature_columns() -> list[str]:
    """Feature columns for ML model testing."""
    return [
        "return_1",
        "return_5",
        "rsi_14",
        "vol_5",
        "vol_20",
        "volume_imbalance",
        "spread_bps",
    ]


@pytest.fixture
def large_labeled_frame() -> pl.DataFrame:
    """Create a larger labeled frame for ML training with more samples."""
    np.random.seed(42)
    n_rows = 1000
    base_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=UTC)

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n_rows) * 0.002
    trend = np.linspace(0, 0.1, n_rows)
    prices = base_price * np.cumprod(1 + returns + trend / n_rows)

    # Forward returns for labels
    forward_returns = np.diff(prices, append=prices[-1]) / prices
    forward_returns = np.roll(forward_returns, -1)
    forward_returns[-5:] = 0.0  # Zero out last few

    # Direction labels
    direction = np.zeros(n_rows, dtype=int)
    direction[forward_returns > 0.0008] = 1
    direction[forward_returns < -0.0008] = -1

    # Features
    returns_1 = np.diff(prices, prepend=prices[0]) / np.roll(prices, 1)
    returns_1[0] = 0
    returns_5 = np.convolve(returns_1, np.ones(5) / 5, mode="same")
    returns_10 = np.convolve(returns_1, np.ones(10) / 10, mode="same")

    # RSI
    gains = np.where(returns_1 > 0, returns_1, 0)
    losses = np.where(returns_1 < 0, -returns_1, 0)
    avg_gain = np.convolve(gains, np.ones(14) / 14, mode="same")
    avg_loss = np.convolve(losses, np.ones(14) / 14, mode="same")
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))

    # Volatility
    vol_5 = np.array([np.std(returns_1[max(0, i - 5):i + 1]) if i >= 5 else 0.001 for i in range(n_rows)])
    vol_20 = np.array([np.std(returns_1[max(0, i - 20):i + 1]) if i >= 20 else 0.001 for i in range(n_rows)])

    spread_bps = np.random.uniform(1, 5, size=n_rows)

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["SPY"] * n_rows,
        "open": np.roll(prices, 1),
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": np.random.randint(10000, 100000, size=n_rows).astype(float),
        "bid_price": prices * (1 - spread_bps / 20000),
        "ask_price": prices * (1 + spread_bps / 20000),
        "spread_bps": spread_bps,
        "forward_return_60s": forward_returns,
        "direction_60s": direction,
        "return_1": returns_1,
        "return_5": returns_5,
        "return_10": returns_10,
        "rsi_14": rsi_14,
        "vol_5": vol_5,
        "vol_20": vol_20,
        "volume_imbalance": np.random.uniform(-0.5, 0.5, n_rows),
    })


# --- Fixtures for Ranking Tests (T4.09) ---


@pytest.fixture
def sample_candidate_scores() -> list[dict]:
    """Create sample candidate scores for ranking tests."""
    return [
        {
            "strategy_id": "trend_ma_20",
            "family": "trend",
            "params": {"lookback": 20, "threshold": 0.02},
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        },
        {
            "strategy_id": "trend_ma_50",
            "family": "trend",
            "params": {"lookback": 50, "threshold": 0.03},
            "sharpe": 1.2,
            "sortino": 1.6,
            "max_drawdown": 0.08,
            "profit_factor": 1.5,
            "win_rate": 0.52,
            "num_trades": 75,
        },
        {
            "strategy_id": "meanrev_bb_20",
            "family": "mean_reversion",
            "params": {"lookback": 20, "bands": 2.0},
            "sharpe": 1.8,
            "sortino": 2.3,
            "max_drawdown": 0.12,
            "profit_factor": 2.0,
            "win_rate": 0.58,
            "num_trades": 120,
        },
        {
            "strategy_id": "meanrev_bb_30",
            "family": "mean_reversion",
            "params": {"lookback": 30, "bands": 2.5},
            "sharpe": 1.3,
            "sortino": 1.7,
            "max_drawdown": 0.15,
            "profit_factor": 1.6,
            "win_rate": 0.54,
            "num_trades": 90,
        },
        {
            "strategy_id": "vol_breakout",
            "family": "volatility",
            "params": {"atr_mult": 1.5, "lookback": 14},
            "sharpe": 0.9,
            "sortino": 1.1,
            "max_drawdown": 0.18,
            "profit_factor": 1.3,
            "win_rate": 0.48,
            "num_trades": 60,
        },
    ]


@pytest.fixture
def low_trade_candidate() -> dict:
    """Create a candidate with insufficient trade count."""
    return {
        "strategy_id": "low_trades",
        "family": "test",
        "params": {},
        "sharpe": 2.5,
        "sortino": 3.0,
        "max_drawdown": 0.05,
        "profit_factor": 2.5,
        "win_rate": 0.70,
        "num_trades": 10,  # Below typical minimum threshold
    }


@pytest.fixture
def high_drawdown_candidate() -> dict:
    """Create a candidate with excessive drawdown."""
    return {
        "strategy_id": "high_dd",
        "family": "test",
        "params": {},
        "sharpe": 1.5,
        "sortino": 2.0,
        "max_drawdown": 0.35,  # Above typical threshold
        "profit_factor": 1.8,
        "win_rate": 0.55,
        "num_trades": 100,
    }


@pytest.fixture
def multiple_families_candidates() -> list[dict]:
    """Create candidates across multiple strategy families."""
    np.random.seed(42)
    candidates = []

    # Trend family candidates
    for i, params in enumerate([(20, 0.02), (50, 0.03), (100, 0.05)]):
        candidates.append(
            {
                "strategy_id": f"trend_{i}",
                "family": "trend",
                "params": {"lookback": params[0], "threshold": params[1]},
                "sharpe": 1.2 + np.random.rand() * 0.5,
                "sortino": 1.5 + np.random.rand() * 0.5,
                "max_drawdown": 0.08 + np.random.rand() * 0.05,
                "profit_factor": 1.4 + np.random.rand() * 0.4,
                "win_rate": 0.50 + np.random.rand() * 0.10,
                "num_trades": 50 + int(np.random.rand() * 50),
            }
        )

    # Mean reversion family candidates
    for i, params in enumerate([(20, 2.0), (30, 2.5)]):
        candidates.append(
            {
                "strategy_id": f"meanrev_{i}",
                "family": "mean_reversion",
                "params": {"lookback": params[0], "bands": params[1]},
                "sharpe": 1.0 + np.random.rand() * 0.8,
                "sortino": 1.3 + np.random.rand() * 0.7,
                "max_drawdown": 0.10 + np.random.rand() * 0.08,
                "profit_factor": 1.3 + np.random.rand() * 0.5,
                "win_rate": 0.52 + np.random.rand() * 0.08,
                "num_trades": 60 + int(np.random.rand() * 40),
            }
        )

    # ML family candidates
    for i in range(2):
        candidates.append(
            {
                "strategy_id": f"ml_{i}",
                "family": "ml",
                "params": {"model_type": "xgboost", "features": 20 + i * 10},
                "sharpe": 1.4 + np.random.rand() * 0.6,
                "sortino": 1.8 + np.random.rand() * 0.6,
                "max_drawdown": 0.07 + np.random.rand() * 0.06,
                "profit_factor": 1.5 + np.random.rand() * 0.5,
                "win_rate": 0.54 + np.random.rand() * 0.08,
                "num_trades": 70 + int(np.random.rand() * 30),
            }
        )

    return candidates


# --- Fixtures for Cost Sensitivity Tests (T4.10) ---


@pytest.fixture
def mock_quotes() -> pl.DataFrame:
    """Create mock quotes data for backtesting."""
    base_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=UTC)
    n_quotes = 100

    timestamps = [base_time + timedelta(minutes=i * 5) for i in range(n_quotes)]

    np.random.seed(42)
    midprice = 100.0
    bid_prices = []
    ask_prices = []

    for _ in range(n_quotes):
        midprice *= 1 + np.random.randn() * 0.001
        spread = midprice * 0.0001 * (1 + np.random.rand())
        bid_prices.append(midprice - spread / 2)
        ask_prices.append(midprice + spread / 2)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * n_quotes,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": [1000.0 + np.random.rand() * 500 for _ in range(n_quotes)],
            "ask_size": [1000.0 + np.random.rand() * 500 for _ in range(n_quotes)],
            "vol": [0.001 + np.random.rand() * 0.0005 for _ in range(n_quotes)],
        }
    )


@pytest.fixture
def mock_backtest_result() -> dict:
    """Create a mock backtest result for testing."""
    np.random.seed(42)
    n_periods = 50

    # Generate equity curve with some volatility
    equity = [100000.0]
    for _ in range(n_periods - 1):
        change = equity[-1] * np.random.randn() * 0.01
        equity.append(equity[-1] + change + 50)  # Slight positive drift

    returns = np.diff(equity) / np.array(equity[:-1])

    return {
        "total_pnl": equity[-1] - equity[0],
        "total_trades": 45,
        "fills": [
            {"pnl": np.random.randn() * 100 + 20, "symbol": "SPY"}
            for _ in range(45)
        ],
        "equity_curve": pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
                    for i in range(n_periods)
                ],
                "equity": equity,
                "returns": [0.0] + list(returns),
            }
        ),
    }


@pytest.fixture
def cost_scenarios() -> list[tuple[float, float]]:
    """Cost scenarios as (spread_multiplier, slippage_multiplier) tuples."""
    return [
        (0.5, 0.5),   # Low cost scenario
        (1.0, 1.0),   # Base case
        (1.5, 1.5),   # Moderate stress
        (2.0, 2.0),   # High stress
        (3.0, 2.5),   # Extreme stress
    ]
