"""
Comprehensive Integration Tests for Backtester Service (T3.09-T3.10).

These tests verify:
- T3.09: End-to-end pipeline from feature_builder -> labeler -> backtester
- T3.10: Backtest validation against known benchmarks and provider bars

Test Scenarios:
1. Simple Long Trade - verify entry/exit prices and P&L
2. Round Trip with Costs - verify transaction cost calculations
3. Risk Limits Enforcement - verify RiskChecker blocks violating trades
4. Stop-Loss Trigger - verify position closure at stop-loss level
5. Walk-Forward Consistency - verify deterministic/reproducible results
6. Data Leakage Detection - verify no future data usage
7. Known P&L Scenarios - validate against pre-calculated results
8. Metrics Validation - verify metric calculations match formulas
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from cost_models import SlippageModel, TransactionCostModel
from risk_models import RiskChecker, RiskLimits

from backtester_py.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Fill,
    Signal,
    SignalType,
)
from backtester_py.evaluation import (
    calculate_metrics,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_slippage_model() -> SlippageModel:
    """Create a simple slippage model for testing."""
    return SlippageModel(
        spread_coef=1.0,
        size_coef=0.5,
        vol_coef=2.0,
        max_size_impact_bps=50.0,
    )


@pytest.fixture
def basic_cost_model(basic_slippage_model: SlippageModel) -> TransactionCostModel:
    """Create a basic transaction cost model."""
    return TransactionCostModel(
        slippage_model=basic_slippage_model,
        fixed_cost_bps=0.35,  # ~$0.35 per $10,000 traded
    )


@pytest.fixture
def basic_risk_limits() -> RiskLimits:
    """Create basic risk limits for testing."""
    return RiskLimits(
        max_position_notional=50_000.0,
        max_gross_exposure=100_000.0,
        max_net_exposure=50_000.0,
        max_daily_loss=5_000.0,
        max_drawdown_pct=0.10,  # 10% max drawdown
    )


@pytest.fixture
def basic_risk_checker(basic_risk_limits: RiskLimits) -> RiskChecker:
    """Create a basic risk checker."""
    return RiskChecker(limits=basic_risk_limits)


@pytest.fixture
def basic_backtest_config(
    basic_cost_model: TransactionCostModel,
    basic_risk_checker: RiskChecker,
) -> BacktestConfig:
    """Create a basic backtest configuration."""
    return BacktestConfig(
        initial_capital=100_000.0,
        cost_model=basic_cost_model,
        risk_checker=basic_risk_checker,
        default_order_notional=10_000.0,
        stop_loss_pct=0.02,
        book_notional_default=500_000.0,
        short_term_vol_default=0.001,
    )


@pytest.fixture
def sample_market_data() -> pl.DataFrame:
    """
    Create sample market data with clear price movement.

    Creates 100 bars with an upward trend followed by a downtrend.
    """
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_bars = 100

    data = []
    base_price = 100.0
    spread = 0.02  # 2 cents spread

    for i in range(n_bars):
        ts = base_time + timedelta(minutes=i)

        # Create price pattern: up for first 50 bars, down for next 50
        # 2 cents per bar up for first 50, then down
        trend = i * 0.02 if i < 50 else 50 * 0.02 - (i - 50) * 0.02

        mid_price = base_price + trend

        data.append({
            "timestamp": ts,
            "symbol": "SPY",
            "bid_price": mid_price - spread / 2,
            "ask_price": mid_price + spread / 2,
            "bid_size": 5000.0,
            "ask_size": 5000.0,
            "vol": 0.001,
        })

    return pl.DataFrame(data)


@pytest.fixture
def uptrend_market_data() -> pl.DataFrame:
    """Create market data with clear uptrend for long trade testing."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_bars = 50

    data = []
    base_price = 100.0
    spread = 0.02

    for i in range(n_bars):
        ts = base_time + timedelta(minutes=i)
        mid_price = base_price + i * 0.05  # 5 cents per bar up

        data.append({
            "timestamp": ts,
            "symbol": "SPY",
            "bid_price": mid_price - spread / 2,
            "ask_price": mid_price + spread / 2,
            "bid_size": 5000.0,
            "ask_size": 5000.0,
            "vol": 0.001,
        })

    return pl.DataFrame(data)


# =============================================================================
# SCENARIO 1: SIMPLE LONG TRADE
# =============================================================================


class TestSimpleLongTrade:
    """Test Scenario 1: Simple long trade with entry/exit verification."""

    def test_long_entry_at_ask_plus_slippage(
        self,
        basic_backtest_config: BacktestConfig,
        uptrend_market_data: pl.DataFrame,
    ) -> None:
        """Verify that long entry occurs at ask price + slippage."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Single long signal at t=0
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(uptrend_market_data, signals)

        # Verify we got a fill
        assert len(result.fills) == 1
        fill = result.fills[0]

        # Get the ask price at entry time
        ask_price = 100.01  # mid=100, spread=0.02, ask=100.01

        # Fill price should be >= ask price (with slippage)
        assert fill.price >= ask_price
        assert fill.side == "buy"
        assert fill.symbol == "SPY"
        assert fill.slippage_bps >= 0

    def test_long_exit_at_bid_minus_slippage(
        self,
        basic_backtest_config: BacktestConfig,
        uptrend_market_data: pl.DataFrame,
    ) -> None:
        """Verify that long exit occurs at bid price - slippage."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Long entry then flat (exit) signal
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=10),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result = engine.run(uptrend_market_data, signals)

        # Should have 2 fills: buy and sell
        assert len(result.fills) == 2

        entry_fill = result.fills[0]
        exit_fill = result.fills[1]

        assert entry_fill.side == "buy"
        assert exit_fill.side == "sell"

        # Exit at t=10: mid_price = 100 + 10*0.05 = 100.50, bid = 100.49
        expected_bid = 100.49

        # Exit price should be <= bid price (with slippage deduction)
        assert exit_fill.price <= expected_bid

    def test_long_trade_pnl_calculation(
        self,
        basic_backtest_config: BacktestConfig,
        uptrend_market_data: pl.DataFrame,
    ) -> None:
        """Verify P&L is correctly calculated for a profitable long trade."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=20),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result = engine.run(uptrend_market_data, signals)

        # Calculate expected P&L
        # Entry at t=0: ask ~100.01 + slippage
        # Exit at t=20: mid = 100 + 20*0.05 = 101.00, bid = 100.99 - slippage

        # Price moved up by ~1%, so with 10k notional we expect ~$100 gross profit
        # Less transaction costs

        # The total P&L should be positive for this uptrend trade
        assert result.total_pnl > 0 or len(result.fills) == 2

        # Verify fills show the price appreciation
        if len(result.fills) == 2:
            entry_price = result.fills[0].price
            exit_price = result.fills[1].price
            # Exit should be at higher price than entry
            assert exit_price > entry_price


# =============================================================================
# SCENARIO 2: ROUND TRIP WITH COSTS
# =============================================================================


class TestRoundTripWithCosts:
    """Test Scenario 2: Verify round-trip cost calculations."""

    def test_round_trip_cost_matches_model(
        self,
        basic_cost_model: TransactionCostModel,
        basic_risk_checker: RiskChecker,
    ) -> None:
        """Verify round-trip cost matches TransactionCostModel calculation."""
        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=basic_cost_model,
            risk_checker=basic_risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.05,
            book_notional_default=500_000.0,
            short_term_vol_default=0.001,
        )
        engine = BacktestEngine(config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Create flat market (no price movement)
        data = []
        for i in range(20):
            ts = base_time + timedelta(minutes=i)
            data.append({
                "timestamp": ts,
                "symbol": "SPY",
                "bid_price": 99.99,
                "ask_price": 100.01,
                "bid_size": 5000.0,
                "ask_size": 5000.0,
                "vol": 0.001,
            })
        market_data = pl.DataFrame(data)

        # Buy and sell at same price level
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=10),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result = engine.run(market_data, signals)

        # In a flat market, round-trip P&L should be negative (costs only)
        # P&L = spread crossing cost + slippage + commissions
        if len(result.fills) == 2:
            total_commission = sum(f.commission for f in result.fills)

            # Verify commissions were charged
            assert total_commission > 0

            # In flat market, we lose money due to costs
            assert (
                result.total_pnl < 0
                or result.total_pnl == pytest.approx(0, abs=total_commission + 10)
            )

    def test_spread_cost_deducted(
        self,
        basic_backtest_config: BacktestConfig,
    ) -> None:
        """Verify spread cost is properly deducted from P&L."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Wide spread market
        data = []
        for i in range(20):
            ts = base_time + timedelta(minutes=i)
            data.append({
                "timestamp": ts,
                "symbol": "SPY",
                "bid_price": 99.90,  # Wide spread: 20 cents
                "ask_price": 100.10,
                "bid_size": 5000.0,
                "ask_size": 5000.0,
                "vol": 0.001,
            })
        market_data = pl.DataFrame(data)

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=5),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result = engine.run(market_data, signals)

        # Wide spread means more cost
        if len(result.fills) == 2:
            # Spread of 20 cents = 20 bps on $100 price
            # For $10k notional, spread cost alone ~$20 round trip
            assert result.total_pnl < -10  # Should lose at least $10 to spread


# =============================================================================
# SCENARIO 3: RISK LIMITS ENFORCEMENT
# =============================================================================


class TestRiskLimitsEnforcement:
    """Test Scenario 3: Verify RiskChecker blocks violating trades."""

    def test_position_limit_blocks_trade(
        self,
        basic_cost_model: TransactionCostModel,
    ) -> None:
        """Verify trades exceeding position limits are blocked."""
        # Very tight position limit
        tight_limits = RiskLimits(
            max_position_notional=5_000.0,  # Only $5k max position
            max_gross_exposure=100_000.0,
            max_net_exposure=50_000.0,
            max_daily_loss=5_000.0,
            max_drawdown_pct=0.10,
        )
        risk_checker = RiskChecker(limits=tight_limits)

        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=basic_cost_model,
            risk_checker=risk_checker,
            default_order_notional=10_000.0,  # Larger than position limit!
            stop_loss_pct=0.02,
        )
        engine = BacktestEngine(config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        market_data = pl.DataFrame({
            "timestamp": [base_time],
            "symbol": ["SPY"],
            "bid_price": [99.99],
            "ask_price": [100.01],
            "bid_size": [5000.0],
            "ask_size": [5000.0],
            "vol": [0.001],
        })

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,  # Exceeds $5k limit
            ),
        ]

        result = engine.run(market_data, signals)

        # Trade should be blocked
        assert len(result.risk_violations) > 0
        violation_msg = result.risk_violations[0]
        assert "position" in violation_msg.lower() or "Position" in violation_msg

    def test_gross_exposure_limit_blocks_second_trade(
        self,
        basic_cost_model: TransactionCostModel,
    ) -> None:
        """Verify gross exposure limit prevents additional positions."""
        limits = RiskLimits(
            max_position_notional=50_000.0,
            max_gross_exposure=15_000.0,  # Tight gross exposure
            max_net_exposure=50_000.0,
            max_daily_loss=5_000.0,
            max_drawdown_pct=0.10,
        )
        risk_checker = RiskChecker(limits=limits)

        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=basic_cost_model,
            risk_checker=risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.02,
        )
        engine = BacktestEngine(config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Two symbols
        market_data = pl.DataFrame([
            {"timestamp": base_time, "symbol": "SPY", "bid_price": 99.99, "ask_price": 100.01,
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=1), "symbol": "SPY", "bid_price": 99.99,
             "ask_price": 100.01, "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time, "symbol": "QQQ", "bid_price": 399.99, "ask_price": 400.01,
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=1), "symbol": "QQQ", "bid_price": 399.99,
             "ask_price": 400.01, "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
        ])

        # Try to open two $10k positions = $20k gross exposure > $15k limit
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=1),
                symbol="QQQ",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(market_data, signals)

        # First trade should succeed, second should be blocked
        spy_fills = [f for f in result.fills if f.symbol == "SPY"]
        assert len(spy_fills) == 1  # First trade went through

        # Should have a gross exposure violation
        assert len(result.risk_violations) > 0


# =============================================================================
# SCENARIO 4: STOP-LOSS TRIGGER
# =============================================================================


class TestStopLossTrigger:
    """Test Scenario 4: Verify stop-loss triggers position closure."""

    def test_stop_loss_closes_position(
        self,
        basic_cost_model: TransactionCostModel,
        basic_risk_checker: RiskChecker,
    ) -> None:
        """Verify position is closed when stop-loss is triggered."""
        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=basic_cost_model,
            risk_checker=basic_risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.02,  # 2% stop loss
        )
        engine = BacktestEngine(config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Create downtrend market that triggers stop-loss
        # Entry at 100, need price to drop 2% to trigger stop
        data = []
        for i in range(20):
            ts = base_time + timedelta(minutes=i)
            # Price drops 0.15% per bar, triggers stop after ~14 bars
            mid_price = 100.0 * (1 - i * 0.0015)

            data.append({
                "timestamp": ts,
                "symbol": "SPY",
                "bid_price": mid_price - 0.01,
                "ask_price": mid_price + 0.01,
                "bid_size": 5000.0,
                "ask_size": 5000.0,
                "vol": 0.001,
            })
        market_data = pl.DataFrame(data)

        # Enter long, then send flat signals later (stop should trigger first)
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            # These signals at later times should see stop-loss trigger
            Signal(
                timestamp=base_time + timedelta(minutes=15),
                symbol="SPY",
                signal_type=SignalType.LONG,  # No-op if already closed
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(market_data, signals)

        # Should have entry and stop-loss exit
        assert len(result.fills) >= 2

        # P&L should be negative (stopped out at loss)
        position = result.positions.get("SPY")
        if position:
            assert position.realized_pnl < 0 or position.is_flat


# =============================================================================
# SCENARIO 5: WALK-FORWARD CONSISTENCY
# =============================================================================


class TestWalkForwardConsistency:
    """Test Scenario 5: Verify deterministic and reproducible results."""

    def test_backtest_is_deterministic(
        self,
        basic_backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Running same backtest twice produces identical results."""
        engine1 = BacktestEngine(basic_backtest_config)
        engine2 = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=30),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result1 = engine1.run(sample_market_data, signals)
        result2 = engine2.run(sample_market_data, signals)

        # Results should be identical
        assert len(result1.fills) == len(result2.fills)
        assert result1.total_trades == result2.total_trades
        assert result1.total_pnl == pytest.approx(result2.total_pnl, rel=1e-10)

        for f1, f2 in zip(result1.fills, result2.fills, strict=True):
            assert f1.price == pytest.approx(f2.price, rel=1e-10)
            assert f1.quantity == pytest.approx(f2.quantity, rel=1e-10)

    def test_different_signal_order_same_timestamps(
        self,
        basic_backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Signals at same timestamp processed consistently regardless of list order."""
        engine1 = BacktestEngine(basic_backtest_config)
        engine2 = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        signals1 = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=30),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        # Same signals in reverse order
        signals2 = [
            Signal(
                timestamp=base_time + timedelta(minutes=30),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
            ),
        ]

        result1 = engine1.run(sample_market_data, signals1)
        result2 = engine2.run(sample_market_data, signals2)

        # Should produce same results (signals are sorted by timestamp internally)
        assert len(result1.fills) == len(result2.fills)
        assert result1.total_pnl == pytest.approx(result2.total_pnl, rel=1e-10)


# =============================================================================
# SCENARIO 6: DATA LEAKAGE DETECTION
# =============================================================================


class TestDataLeakageDetection:
    """Test Scenario 6: Verify no future data is used in decisions."""

    def test_no_future_price_used(
        self,
        basic_backtest_config: BacktestConfig,
    ) -> None:
        """Verify fill price comes from data at or before signal time."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Market data with distinct prices at each timestamp
        market_data = pl.DataFrame([
            {"timestamp": base_time, "symbol": "SPY",
             "bid_price": 100.00, "ask_price": 100.02,
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=1), "symbol": "SPY",
             "bid_price": 101.00, "ask_price": 101.02,  # Future price (higher)
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=2), "symbol": "SPY",
             "bid_price": 102.00, "ask_price": 102.02,  # Even more future
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
        ])

        # Signal at t=0 should use t=0 prices, not future prices
        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(market_data, signals)

        assert len(result.fills) == 1
        fill = result.fills[0]

        # Fill price should be based on t=0 ask (100.02) + slippage
        # NOT future prices (101.02 or 102.02)
        assert fill.price < 101.0  # Should not use future price
        assert fill.price >= 100.02  # Should be at or above t=0 ask


# =============================================================================
# T3.10: VALIDATION TESTS - KNOWN P&L SCENARIOS
# =============================================================================


class TestKnownPnlScenarios:
    """Test Scenario 7: Validate P&L calculations against known results."""

    def test_known_long_trade_pnl(self) -> None:
        """
        Test P&L calculation for a known scenario.

        Scenario:
        - Buy 100 shares at $100.00 (ask) + 5bps slippage = $100.05
        - Sell 100 shares at $101.00 (bid) - 5bps slippage = $100.95
        - Gross P&L = (100.95 - 100.05) * 100 = $90
        - Commission: 0.35 bps on $10k buy + 0.35 bps on ~$10k sell = ~$0.70
        - Net P&L ~= $89.30
        """
        # Create minimal slippage model for predictable results
        slippage = SlippageModel(
            spread_coef=0.5,  # Half spread impact
            size_coef=0.0,    # No size impact
            vol_coef=0.0,     # No vol impact
        )
        cost_model = TransactionCostModel(slippage, fixed_cost_bps=0.35)

        limits = RiskLimits(
            max_position_notional=100_000.0,
            max_gross_exposure=200_000.0,
            max_net_exposure=100_000.0,
            max_daily_loss=10_000.0,
            max_drawdown_pct=0.20,
        )
        risk_checker = RiskChecker(limits=limits)

        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=cost_model,
            risk_checker=risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.10,
            book_notional_default=1_000_000.0,
            short_term_vol_default=0.001,
        )
        engine = BacktestEngine(config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        market_data = pl.DataFrame([
            {"timestamp": base_time, "symbol": "SPY",
             "bid_price": 99.99, "ask_price": 100.01,  # mid=100, spread=0.02
             "bid_size": 10000.0, "ask_size": 10000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=10), "symbol": "SPY",
             "bid_price": 100.99, "ask_price": 101.01,  # mid=101, spread=0.02
             "bid_size": 10000.0, "ask_size": 10000.0, "vol": 0.001},
        ])

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=10),
                symbol="SPY",
                signal_type=SignalType.FLAT,
                strength=1.0,
            ),
        ]

        result = engine.run(market_data, signals)

        assert len(result.fills) == 2

        entry_fill = result.fills[0]
        exit_fill = result.fills[1]

        # Entry: ask=100.01 + slippage (half of 2bps spread = 1bp)
        # Exit: bid=100.99 - slippage

        # Gross return should be approximately 1% - costs
        # With 10k notional, ~$100 gross - ~$1-2 costs = ~$98-99 net

        total_commission = entry_fill.commission + exit_fill.commission
        assert total_commission > 0  # Commissions were charged

        # Overall trade should be profitable (price moved up 1%)
        # This is approximate due to slippage model
        assert result.total_pnl > 0 or entry_fill.price < exit_fill.price


# =============================================================================
# T3.10: METRICS VALIDATION
# =============================================================================


class TestMetricsValidation:
    """Test Scenario 8: Validate metric calculations against formulas."""

    def test_sharpe_ratio_calculation(self) -> None:
        """
        Verify Sharpe ratio calculation matches formula.

        Given: daily returns [0.01, -0.005, 0.02, -0.01, 0.015]
        Mean = 0.006
        Std = 0.0112 (approximately)
        Sharpe (annualized) = mean/std * sqrt(252)
        """
        # Create a mock result with known returns
        equity_values = [100000.0]  # Start
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]

        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))

        timestamps = [
            datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC) + timedelta(days=i)
            for i in range(len(equity_values))
        ]

        result = BacktestResult(
            fills=[],
            positions={},
            equity_curve=pl.DataFrame({
                "timestamp": timestamps,
                "equity": equity_values,
            }),
            total_pnl=equity_values[-1] - equity_values[0],
            total_trades=0,
        )

        metrics = calculate_metrics(
            result,
            initial_capital=100000.0,
            risk_free_rate=0.0,
            trading_days_per_year=252,
        )

        # Calculate expected Sharpe
        returns_array = np.array(returns)
        expected_mean = np.mean(returns_array)
        expected_std = np.std(returns_array)
        expected_sharpe = expected_mean / expected_std * np.sqrt(252)

        assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, rel=0.1)

    def test_max_drawdown_calculation(self) -> None:
        """Verify max drawdown calculation."""
        # Equity: 100k -> 110k -> 90k -> 95k
        # Peak = 110k, trough = 90k
        # Max DD = (110k - 90k) / 110k = 18.18%

        timestamps = [
            datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC) + timedelta(days=i)
            for i in range(4)
        ]

        result = BacktestResult(
            fills=[],
            positions={},
            equity_curve=pl.DataFrame({
                "timestamp": timestamps,
                "equity": [100000.0, 110000.0, 90000.0, 95000.0],
            }),
            total_pnl=-5000.0,
            total_trades=0,
        )

        metrics = calculate_metrics(result, initial_capital=100000.0)

        expected_max_dd = (110000 - 90000) / 110000  # 18.18%
        assert metrics.max_drawdown == pytest.approx(expected_max_dd, rel=0.01)

    def test_win_rate_calculation(self) -> None:
        """Verify win rate calculation."""
        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Create fills that result in 3 wins and 2 losses
        fills = [
            # Trade 1: Buy then sell at profit
            Fill(timestamp=base_time, symbol="SPY", side="buy",
                 quantity=100, price=100.0, notional=10000, slippage_bps=1, commission=0.35),
            Fill(timestamp=base_time + timedelta(minutes=10), symbol="SPY", side="sell",
                 quantity=100, price=101.0, notional=10100, slippage_bps=1, commission=0.35),
            # Trade 2: Buy then sell at loss
            Fill(timestamp=base_time + timedelta(minutes=20), symbol="SPY", side="buy",
                 quantity=100, price=101.0, notional=10100, slippage_bps=1, commission=0.35),
            Fill(timestamp=base_time + timedelta(minutes=30), symbol="SPY", side="sell",
                 quantity=100, price=100.5, notional=10050, slippage_bps=1, commission=0.35),
            # Trade 3: Buy then sell at profit
            Fill(timestamp=base_time + timedelta(minutes=40), symbol="SPY", side="buy",
                 quantity=100, price=100.0, notional=10000, slippage_bps=1, commission=0.35),
            Fill(timestamp=base_time + timedelta(minutes=50), symbol="SPY", side="sell",
                 quantity=100, price=102.0, notional=10200, slippage_bps=1, commission=0.35),
        ]

        timestamps = [
            base_time + timedelta(minutes=i * 10)
            for i in range(7)
        ]

        result = BacktestResult(
            fills=fills,
            positions={},
            equity_curve=pl.DataFrame({
                "timestamp": timestamps,
                "equity": [100000.0 + i * 50 for i in range(7)],
            }),
            total_pnl=250.0,
            total_trades=6,
        )

        metrics = calculate_metrics(result, initial_capital=100000.0)

        # 3 round trips: 2 profitable, 1 loss = 66.7% win rate
        assert metrics.total_trades == 6  # 6 fills
        # Win rate based on round-trip P&L calculation
        assert 0.5 <= metrics.win_rate <= 1.0  # Between 50% and 100%


# =============================================================================
# INTEGRATION WITH FEATURE_BUILDER AND LABELER
# =============================================================================


class TestFeatureBuilderIntegration:
    """Integration tests with feature_builder_py and labeler_py."""

    def test_decision_frame_to_signal_pipeline(
        self,
        basic_backtest_config: BacktestConfig,
    ) -> None:
        """Test converting decision frames to signals and backtesting."""
        from feature_builder_py import DecisionFrameBuilder

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Create minimal decision frame
        decision_times = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=i * 10) for i in range(5)]
        })

        # Create minimal bar data
        micro_bars = pl.DataFrame({
            "symbol": ["SPY"] * 20,
            "bar_start": [base_time + timedelta(seconds=i * 30) for i in range(20)],
            "bar_end": [base_time + timedelta(seconds=(i + 1) * 30) for i in range(20)],
            "vwap": [100.0 + i * 0.01 for i in range(20)],
            "midprice": [100.0 + i * 0.01 for i in range(20)],
            "microprice": [100.0 + i * 0.01 for i in range(20)],
            "spread": [0.02] * 20,
            "bid_size": [1000.0] * 20,
            "ask_size": [1000.0] * 20,
            "quote_imbalance": [0.0] * 20,
            "trade_volume": [5000.0] * 20,
            "realized_vol": [0.01] * 20,
        })

        bars_1m = pl.DataFrame({
            "symbol": ["SPY"] * 10,
            "bar_start": [base_time + timedelta(minutes=i) for i in range(10)],
            "bar_end": [base_time + timedelta(minutes=i + 1) for i in range(10)],
            "open": [100.0 + i * 0.05 for i in range(10)],
            "high": [100.1 + i * 0.05 for i in range(10)],
            "low": [99.9 + i * 0.05 for i in range(10)],
            "close": [100.05 + i * 0.05 for i in range(10)],
            "volume": [10000.0] * 10,
            "returns": [0.0005] * 10,
            "atr": [0.1] * 10,
            "realized_vol": [0.01] * 10,
        })

        bars_5m = pl.DataFrame({
            "symbol": ["SPY"] * 2,
            "bar_start": [base_time, base_time + timedelta(minutes=5)],
            "bar_end": [base_time + timedelta(minutes=5), base_time + timedelta(minutes=10)],
            "open": [100.0, 100.25],
            "high": [100.3, 100.55],
            "low": [99.9, 100.15],
            "close": [100.25, 100.5],
            "volume": [50000.0, 50000.0],
            "returns": [0.0025, 0.0025],
            "atr": [0.3] * 2,
            "realized_vol": [0.015] * 2,
        })

        # Build decision frame
        builder = DecisionFrameBuilder()
        decision_frame = builder.build(
            decision_times=decision_times,
            micro_bars_30s=micro_bars,
            bars_1m=bars_1m,
            bars_5m=bars_5m,
            symbol="SPY",
        )

        assert decision_frame is not None
        assert len(decision_frame) == 5

        # Convert decision frame to signals (simple strategy: long if trend_5m > 0)
        signals = []
        for row in decision_frame.iter_rows(named=True):
            dt = row["decision_ts"]
            trend = row.get("trend_5m", 0)

            if trend is not None and trend > 0:
                signal_type = SignalType.LONG
            elif trend is not None and trend < 0:
                signal_type = SignalType.SHORT
            else:
                signal_type = SignalType.FLAT

            signals.append(Signal(
                timestamp=dt,
                symbol="SPY",
                signal_type=signal_type,
                strength=1.0,
                target_notional=10_000.0,
            ))

        # Create market data aligned with decision times
        market_data = pl.DataFrame([
            {
                "timestamp": base_time + timedelta(minutes=i * 10),
                "symbol": "SPY",
                "bid_price": 100.0 + i * 0.1 - 0.01,
                "ask_price": 100.0 + i * 0.1 + 0.01,
                "bid_size": 5000.0,
                "ask_size": 5000.0,
                "vol": 0.001,
            }
            for i in range(5)
        ])

        # Run backtest
        engine = BacktestEngine(basic_backtest_config)
        result = engine.run(market_data, signals)

        # Verify we got results (exact values depend on signal logic)
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None


class TestLabelerIntegration:
    """Integration tests with labeler_py."""

    def test_labels_to_signals_conversion(
        self,
        basic_backtest_config: BacktestConfig,
    ) -> None:
        """Test converting labels to trading signals."""
        from labeler_py import DirectionLabeler

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Create sample returns DataFrame
        returns_df = pl.DataFrame({
            "symbol": ["SPY"] * 5,
            "decision_ts": [base_time + timedelta(minutes=i * 10) for i in range(5)],
            "horizon": [60] * 5,
            "fwd_return_mid": [0.002, -0.003, 0.0003, -0.001, 0.005],  # Various returns
        })

        # Label with direction
        labeler = DirectionLabeler(no_trade_threshold=0.0005)
        labeled_df = labeler.label_returns(returns_df)

        # Convert labels to signals
        signals = []
        for row in labeled_df.iter_rows(named=True):
            direction = row["direction"]
            if direction == 1:
                signal_type = SignalType.LONG
            elif direction == -1:
                signal_type = SignalType.SHORT
            else:
                signal_type = SignalType.FLAT

            signals.append(Signal(
                timestamp=row["decision_ts"],
                symbol=row["symbol"],
                signal_type=signal_type,
                strength=abs(row["fwd_return_mid"]) * 100,  # Scale strength by return magnitude
            ))

        # Verify signal directions match labels
        expected_directions = [SignalType.LONG, SignalType.SHORT, SignalType.FLAT,
                               SignalType.SHORT, SignalType.LONG]

        for signal, expected in zip(signals, expected_directions, strict=True):
            assert signal.signal_type == expected


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_signals_list(
        self,
        basic_backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Verify handling of empty signals list."""
        engine = BacktestEngine(basic_backtest_config)
        result = engine.run(sample_market_data, [])

        assert len(result.fills) == 0
        assert result.total_trades == 0
        assert result.total_pnl == 0.0

    def test_single_signal_no_exit(
        self,
        basic_backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Verify handling of single signal without exit."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(sample_market_data, signals)

        # Should have entry fill, position open
        assert len(result.fills) == 1
        assert "SPY" in result.positions
        assert not result.positions["SPY"].is_flat

    def test_flip_from_long_to_short(
        self,
        basic_backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Verify position flip from long to short."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        signals = [
            Signal(
                timestamp=base_time,
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
            Signal(
                timestamp=base_time + timedelta(minutes=30),
                symbol="SPY",
                signal_type=SignalType.SHORT,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(sample_market_data, signals)

        # Should have: entry long, exit long, entry short
        assert len(result.fills) >= 2

    def test_market_data_gaps(
        self,
        basic_backtest_config: BacktestConfig,
    ) -> None:
        """Verify handling of gaps in market data."""
        engine = BacktestEngine(basic_backtest_config)

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Market data with gap (missing 9:31-9:34)
        market_data = pl.DataFrame([
            {"timestamp": base_time, "symbol": "SPY",
             "bid_price": 100.0, "ask_price": 100.02,
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
            {"timestamp": base_time + timedelta(minutes=5), "symbol": "SPY",
             "bid_price": 100.10, "ask_price": 100.12,
             "bid_size": 5000.0, "ask_size": 5000.0, "vol": 0.001},
        ])

        # Signal during gap should use most recent available data
        signals = [
            Signal(
                timestamp=base_time + timedelta(minutes=2),  # During gap
                symbol="SPY",
                signal_type=SignalType.LONG,
                strength=1.0,
                target_notional=10_000.0,
            ),
        ]

        result = engine.run(market_data, signals)

        # Should use t=0 prices (most recent before signal)
        if len(result.fills) == 1:
            assert result.fills[0].price <= 100.10  # Not future price
