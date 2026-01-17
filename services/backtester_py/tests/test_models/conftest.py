"""
Pytest fixtures for cost models and risk checker tests.

Provides common test fixtures for order data, portfolios, and model instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest

from cost_models import SlippageModel, TransactionCostModel
from risk_models import RiskChecker, RiskLimits, RiskState


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_order() -> dict:
    """Basic order data for testing."""
    return {
        "symbol": "SPY",
        "side": "buy",
        "price": Decimal("450.00"),
        "quantity": Decimal("100"),
        "volume": Decimal("1000000"),
        "volatility": 0.02,
    }


@pytest.fixture
def sample_portfolio() -> dict:
    """Basic portfolio data for testing."""
    return {
        "value": Decimal("100000"),
        "drawdown": 0.05,
    }


@pytest.fixture
def sample_quote() -> dict:
    """Basic quote data for testing."""
    return {
        "bid_price": 100.00,
        "ask_price": 100.02,
        "bid_size": 1000,
        "ask_size": 800,
        "timestamp": datetime.now(UTC),
    }


# ============================================================================
# Cost Model Fixtures
# ============================================================================


@pytest.fixture
def default_slippage_model() -> SlippageModel:
    """Default slippage model with standard coefficients."""
    return SlippageModel(
        spread_coef=1.0,
        size_coef=1.0,
        vol_coef=1.0,
        max_size_impact_bps=50.0,
    )


@pytest.fixture
def aggressive_slippage_model() -> SlippageModel:
    """Aggressive slippage model with higher coefficients."""
    return SlippageModel(
        spread_coef=1.5,
        size_coef=2.0,
        vol_coef=3.0,
        max_size_impact_bps=100.0,
    )


@pytest.fixture
def zero_slippage_model() -> SlippageModel:
    """Zero slippage model for baseline testing."""
    return SlippageModel(
        spread_coef=0.0,
        size_coef=0.0,
        vol_coef=0.0,
        max_size_impact_bps=0.0,
    )


@pytest.fixture
def default_cost_model(default_slippage_model: SlippageModel) -> TransactionCostModel:
    """Default transaction cost model with fixed costs."""
    return TransactionCostModel(
        slippage_model=default_slippage_model,
        fixed_cost_bps=0.35,
    )


@pytest.fixture
def zero_cost_model(zero_slippage_model: SlippageModel) -> TransactionCostModel:
    """Zero cost model for passthrough testing."""
    return TransactionCostModel(
        slippage_model=zero_slippage_model,
        fixed_cost_bps=0.0,
    )


# ============================================================================
# Risk Model Fixtures
# ============================================================================


@pytest.fixture
def default_risk_limits() -> RiskLimits:
    """Default risk limits for testing."""
    return RiskLimits(
        max_position_notional=100_000,
        max_gross_exposure=500_000,
        max_net_exposure=250_000,
        max_daily_loss=10_000,
        max_drawdown_pct=0.20,
    )


@pytest.fixture
def tight_risk_limits() -> RiskLimits:
    """Tight risk limits for boundary testing."""
    return RiskLimits(
        max_position_notional=10_000,
        max_gross_exposure=50_000,
        max_net_exposure=25_000,
        max_daily_loss=1_000,
        max_drawdown_pct=0.05,
    )


@pytest.fixture
def empty_risk_state() -> RiskState:
    """Empty risk state with no positions."""
    return RiskState(
        positions={},
        daily_pnl=0.0,
        peak_equity=100_000,
        current_equity=100_000,
    )


@pytest.fixture
def risk_state_with_positions() -> RiskState:
    """Risk state with existing positions."""
    return RiskState(
        positions={
            "AAPL": 50_000,
            "GOOGL": -30_000,
            "MSFT": 20_000,
        },
        daily_pnl=500.0,
        peak_equity=100_000,
        current_equity=100_500,
    )


@pytest.fixture
def risk_state_near_limits(tight_risk_limits: RiskLimits) -> RiskState:
    """Risk state approaching limits."""
    return RiskState(
        positions={
            "AAPL": 8_000,  # Near 10k position limit
            "GOOGL": -7_000,
        },
        daily_pnl=-800.0,  # Near 1k daily loss limit
        peak_equity=100_000,
        current_equity=96_000,  # 4% drawdown, near 5% limit
    )


@pytest.fixture
def default_risk_checker(default_risk_limits: RiskLimits) -> RiskChecker:
    """Default risk checker for testing."""
    return RiskChecker(limits=default_risk_limits)


@pytest.fixture
def tight_risk_checker(tight_risk_limits: RiskLimits) -> RiskChecker:
    """Risk checker with tight limits for boundary testing."""
    return RiskChecker(limits=tight_risk_limits)


# ============================================================================
# Time Fixtures
# ============================================================================


@pytest.fixture
def current_time() -> datetime:
    """Current timestamp in UTC."""
    return datetime.now(UTC)


@pytest.fixture
def stale_quote_time(current_time: datetime) -> datetime:
    """Timestamp 10 seconds in the past (stale quote)."""
    return current_time - timedelta(seconds=10)


@pytest.fixture
def fresh_quote_time(current_time: datetime) -> datetime:
    """Timestamp 100ms in the past (fresh quote)."""
    return current_time - timedelta(milliseconds=100)


# ============================================================================
# Helper Fixtures
# ============================================================================


@dataclass
class MarketQuote:
    """Helper dataclass for market quote data."""

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime

    @property
    def midprice(self) -> float:
        """Calculate midprice."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Calculate spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        return self.spread / self.midprice * 10000


@pytest.fixture
def market_quote(fresh_quote_time: datetime) -> MarketQuote:
    """Standard market quote fixture."""
    return MarketQuote(
        bid_price=100.00,
        ask_price=100.02,
        bid_size=1000,
        ask_size=800,
        timestamp=fresh_quote_time,
    )


@pytest.fixture
def wide_spread_quote(fresh_quote_time: datetime) -> MarketQuote:
    """Market quote with wide spread."""
    return MarketQuote(
        bid_price=100.00,
        ask_price=100.10,
        bid_size=500,
        ask_size=300,
        timestamp=fresh_quote_time,
    )


@pytest.fixture
def crossed_market_quote(fresh_quote_time: datetime) -> MarketQuote:
    """Crossed market quote (bid > ask)."""
    return MarketQuote(
        bid_price=100.05,
        ask_price=100.00,
        bid_size=1000,
        ask_size=800,
        timestamp=fresh_quote_time,
    )


@pytest.fixture
def no_liquidity_quote(fresh_quote_time: datetime) -> MarketQuote:
    """Quote with no liquidity."""
    return MarketQuote(
        bid_price=100.00,
        ask_price=100.02,
        bid_size=0,
        ask_size=0,
        timestamp=fresh_quote_time,
    )
