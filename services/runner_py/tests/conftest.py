"""
Pytest configuration and fixtures for runner_py tests.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from runner_py.orchestrator import OrchestrationConfig, Orchestrator
    from runner_py.promotion import AutoPromoter, PromotionCriteria


@pytest.fixture
def sample_shadow_results() -> pl.DataFrame:
    """Create sample shadow execution results."""
    base_time = datetime(2024, 1, 15, 9, 30, 0)
    n_rows = 100

    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]
    np.random.seed(42)

    returns = np.random.normal(0.0001, 0.002, n_rows)
    equity = 100000 * np.cumprod(1 + returns)

    return pl.DataFrame({
        "timestamp": timestamps,
        "equity": equity.tolist(),
        "returns": returns.tolist(),
        "symbol": ["SPY"] * n_rows,
        "strategy_id": ["strategy_001"] * n_rows,
        "fill_count": [1] * n_rows,
        "slippage_bps": np.random.uniform(0.5, 3.0, n_rows).tolist(),
    })


@pytest.fixture
def sample_backtest_results() -> pl.DataFrame:
    """Create sample backtest results matching shadow execution."""
    base_time = datetime(2024, 1, 15, 9, 30, 0)
    n_rows = 100

    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]
    np.random.seed(42)

    base_returns = np.random.normal(0.0001, 0.002, n_rows)
    returns = base_returns * (1 + np.random.uniform(-0.05, 0.05, n_rows))
    equity = 100000 * np.cumprod(1 + returns)

    return pl.DataFrame({
        "timestamp": timestamps,
        "equity": equity.tolist(),
        "returns": returns.tolist(),
        "symbol": ["SPY"] * n_rows,
        "strategy_id": ["strategy_001"] * n_rows,
        "fill_count": [1] * n_rows,
        "slippage_bps": np.random.uniform(0.5, 3.0, n_rows).tolist(),
    })


@pytest.fixture
def sample_paper_fills() -> pl.DataFrame:
    """Create sample paper trading fills."""
    base_time = datetime(2024, 1, 15, 9, 30, 0)
    n_fills = 50

    np.random.seed(42)
    timestamps = [base_time + timedelta(minutes=i * 2) for i in range(n_fills)]

    symbols = np.random.choice(["SPY", "QQQ", "AAPL"], n_fills)
    sides = np.random.choice(["buy", "sell"], n_fills)
    quantities = np.random.uniform(10, 100, n_fills)
    expected_prices = np.random.uniform(400, 500, n_fills)

    slippage_bps = np.random.uniform(0.5, 5.0, n_fills)
    actual_prices = expected_prices * (
        1 + slippage_bps / 10000 * np.where(sides == "buy", 1, -1)
    )

    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": symbols.tolist(),
        "side": sides.tolist(),
        "quantity": quantities.tolist(),
        "expected_price": expected_prices.tolist(),
        "actual_price": actual_prices.tolist(),
        "slippage_bps": slippage_bps.tolist(),
        "order_id": [f"order_{i:04d}" for i in range(n_fills)],
    })


@pytest.fixture
def divergent_shadow_results(sample_shadow_results: pl.DataFrame) -> pl.DataFrame:
    """Create shadow results that significantly diverge from backtest."""
    modified = sample_shadow_results.with_columns([
        (pl.col("returns") * 1.5).alias("returns"),
    ])

    returns_arr = modified["returns"].to_numpy()
    equity = 100000 * np.cumprod(1 + returns_arr)

    return modified.with_columns([pl.Series("equity", equity.tolist())])


@pytest.fixture
def high_slippage_fills(sample_paper_fills: pl.DataFrame) -> pl.DataFrame:
    """Create fills with abnormally high slippage."""
    return sample_paper_fills.with_columns([
        (pl.col("slippage_bps") * 5).alias("slippage_bps"),
        (pl.col("actual_price") * 1.005).alias("actual_price"),
    ])


@pytest.fixture
def empty_fills() -> pl.DataFrame:
    """Create empty fills DataFrame with correct schema."""
    return pl.DataFrame({
        "timestamp": [],
        "symbol": [],
        "side": [],
        "quantity": [],
        "expected_price": [],
        "actual_price": [],
        "slippage_bps": [],
        "order_id": [],
    }).cast({
        "timestamp": pl.Datetime,
        "symbol": pl.Utf8,
        "side": pl.Utf8,
        "quantity": pl.Float64,
        "expected_price": pl.Float64,
        "actual_price": pl.Float64,
        "slippage_bps": pl.Float64,
        "order_id": pl.Utf8,
    })


# =============================================================================
# Orchestrator Fixtures (T6.05)
# =============================================================================


@pytest.fixture
def default_orchestration_config() -> OrchestrationConfig:
    """Create default orchestration config for tests."""
    from runner_py.orchestrator import OrchestrationConfig

    return OrchestrationConfig()


@pytest.fixture
def test_orchestration_config() -> OrchestrationConfig:
    """Create test-friendly orchestration config with no delays."""
    from runner_py.orchestrator import OrchestrationConfig, OrchestrationMode

    return OrchestrationConfig(
        mode=OrchestrationMode.DAILY,
        market_open=time(9, 30),
        market_close=time(16, 0),
        pre_market_minutes=0,
        post_market_minutes=0,
        feature_refresh_interval_seconds=1,
        signal_check_interval_seconds=1,
        registry_api_url="http://localhost:8080",
    )


@pytest.fixture
def orchestrator(test_orchestration_config: OrchestrationConfig) -> Orchestrator:
    """Create orchestrator instance for tests."""
    from runner_py.orchestrator import Orchestrator

    return Orchestrator(test_orchestration_config)


# =============================================================================
# Promotion Fixtures (T6.06)
# =============================================================================


@pytest.fixture
def default_promotion_criteria() -> PromotionCriteria:
    """Create default promotion criteria."""
    from runner_py.promotion import PromotionCriteria

    return PromotionCriteria()


@pytest.fixture
def lenient_promotion_criteria() -> PromotionCriteria:
    """Create lenient promotion criteria for testing."""
    from runner_py.promotion import PromotionCriteria

    return PromotionCriteria(
        min_shadow_days=1,
        min_shadow_sharpe=0.0,
        max_shadow_drawdown=1.0,
        min_shadow_trades=1,
        no_alerts_required=False,
    )


@pytest.fixture
def auto_promoter(default_promotion_criteria: PromotionCriteria) -> AutoPromoter:
    """Create auto promoter instance for tests."""
    from runner_py.promotion import AutoPromoter

    return AutoPromoter(criteria=default_promotion_criteria)


@pytest.fixture
def mock_strategy_data() -> dict[str, Any]:
    """Create mock strategy data."""
    return {
        "id": 1,
        "name": "test_strategy",
        "family": "trend",
        "version": "1.0.0",
        "state": "shadow",
        "parameters": {"lookback": 20},
    }


@pytest.fixture
def mock_shadow_performance_data() -> dict[str, Any]:
    """Create mock shadow performance data."""
    return {
        "strategy_id": "1",
        "state": "shadow",
        "shadow_days": 10,
        "sharpe": 0.75,
        "sortino": 0.90,
        "max_drawdown": 0.05,
        "num_trades": 50,
        "win_rate": 0.55,
        "profit_factor": 1.5,
        "started_at": datetime(2024, 1, 1).isoformat(),
    }


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create mock HTTP client for API testing."""
    client = AsyncMock()

    mock_response = AsyncMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()
    mock_response.status_code = 200

    client.get.return_value = mock_response
    client.post.return_value = mock_response

    return client
