"""
Comprehensive Tests for Backtester Engine (T3.01-T3.04).

Tests cover:
- T3.01: Core simulation loop
- T3.02: Quote-based fill logic
- T3.03: Slippage model integration
- T3.04: Position limits and stops
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pytest
from cost_models import SlippageModel, TransactionCostModel
from risk_models import RiskChecker, RiskLimits, RiskState

from backtester_py.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Fill,
    Position,
    Signal,
    SignalType,
)
from backtester_py.fills import FillSimulator
from backtester_py.risk import RiskManager, StopConfig
from backtester_py.slippage import SlippageIntegrator

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_market_data(utc_now: datetime) -> pl.DataFrame:
    """Create sample market data for testing."""
    timestamps = [utc_now + timedelta(minutes=i) for i in range(5)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 5,
        "bid_price": [150.00, 150.10, 150.20, 150.15, 150.25],
        "ask_price": [150.02, 150.12, 150.22, 150.17, 150.27],
        "bid_size": [1000.0, 1200.0, 800.0, 1100.0, 900.0],
        "ask_size": [900.0, 1000.0, 1100.0, 950.0, 1000.0],
        "vol": [0.001, 0.0012, 0.0015, 0.0011, 0.0013],
    })


@pytest.fixture
def sample_signals(utc_now: datetime) -> list[Signal]:
    """Create sample signals for testing."""
    return [
        Signal(
            timestamp=utc_now,
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=1.0,
        ),
        Signal(
            timestamp=utc_now + timedelta(minutes=2),
            symbol="AAPL",
            signal_type=SignalType.FLAT,
            strength=1.0,
        ),
    ]


@pytest.fixture
def multi_symbol_data(utc_now: datetime) -> pl.DataFrame:
    """Create market data for multiple symbols."""
    timestamps = [utc_now + timedelta(minutes=i) for i in range(4)]
    return pl.DataFrame({
        "timestamp": [timestamps[0], timestamps[0], timestamps[1], timestamps[1]],
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "bid_price": [150.00, 400.00, 150.10, 400.20],
        "ask_price": [150.02, 400.05, 150.12, 400.25],
        "bid_size": [1000.0, 500.0, 1200.0, 600.0],
        "ask_size": [900.0, 450.0, 1000.0, 550.0],
        "vol": [0.001, 0.0008, 0.0012, 0.0009],
    })


@pytest.fixture
def multi_symbol_signals(utc_now: datetime) -> list[Signal]:
    """Create signals for multiple symbols."""
    return [
        Signal(
            timestamp=utc_now,
            symbol="AAPL",
            signal_type=SignalType.LONG,
        ),
        Signal(
            timestamp=utc_now,
            symbol="MSFT",
            signal_type=SignalType.LONG,
        ),
        Signal(
            timestamp=utc_now + timedelta(minutes=1),
            symbol="AAPL",
            signal_type=SignalType.FLAT,
        ),
        Signal(
            timestamp=utc_now + timedelta(minutes=1),
            symbol="MSFT",
            signal_type=SignalType.SHORT,
        ),
    ]


@pytest.fixture
def slippage_model() -> SlippageModel:
    """Create a standard slippage model."""
    return SlippageModel(
        spread_coef=1.0,
        size_coef=0.5,
        vol_coef=2.0,
        max_size_impact_bps=50.0,
    )


@pytest.fixture
def cost_model(slippage_model: SlippageModel) -> TransactionCostModel:
    """Create a transaction cost model."""
    return TransactionCostModel(
        slippage_model=slippage_model,
        fixed_cost_bps=0.35,
    )


@pytest.fixture
def risk_limits() -> RiskLimits:
    """Create standard risk limits."""
    return RiskLimits(
        max_position_notional=100_000.0,
        max_gross_exposure=500_000.0,
        max_net_exposure=250_000.0,
        max_daily_loss=10_000.0,
        max_drawdown_pct=0.05,
    )


@pytest.fixture
def risk_checker(risk_limits: RiskLimits) -> RiskChecker:
    """Create a risk checker."""
    return RiskChecker(limits=risk_limits)


@pytest.fixture
def backtest_config(
    cost_model: TransactionCostModel, risk_checker: RiskChecker
) -> BacktestConfig:
    """Create a backtest configuration."""
    return BacktestConfig(
        initial_capital=100_000.0,
        cost_model=cost_model,
        risk_checker=risk_checker,
        default_order_notional=10_000.0,
        stop_loss_pct=0.02,
    )


# =============================================================================
# T3.01: Core Simulation Loop Tests
# =============================================================================


class TestBacktestEngine:
    """Tests for the core backtesting engine (T3.01)."""

    def test_engine_init(self, backtest_config: BacktestConfig) -> None:
        """Test engine initialization."""
        engine = BacktestEngine(config=backtest_config)
        assert engine.config == backtest_config

    def test_run_returns_backtest_result(
        self,
        backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
        sample_signals: list[Signal],
    ) -> None:
        """Test that run returns a BacktestResult."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(sample_market_data, sample_signals)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.fills, list)
        assert isinstance(result.positions, dict)
        assert isinstance(result.equity_curve, pl.DataFrame)

    def test_signal_generates_fill(
        self,
        backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
        sample_signals: list[Signal],
    ) -> None:
        """Test that a signal generates a fill."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(sample_market_data, sample_signals)

        # Should have fills from signals
        assert len(result.fills) > 0

    def test_long_signal_generates_buy(
        self,
        backtest_config: BacktestConfig,
        utc_now: datetime,
    ) -> None:
        """Test that a long signal generates a buy fill."""
        data = pl.DataFrame({
            "timestamp": [utc_now],
            "symbol": ["AAPL"],
            "bid_price": [150.00],
            "ask_price": [150.02],
            "bid_size": [1000.0],
            "ask_size": [900.0],
            "vol": [0.001],
        })
        signals = [Signal(timestamp=utc_now, symbol="AAPL", signal_type=SignalType.LONG)]

        engine = BacktestEngine(config=backtest_config)
        result = engine.run(data, signals)

        assert len(result.fills) >= 1
        assert result.fills[0].side == "buy"

    def test_flat_signal_closes_position(
        self,
        backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
        sample_signals: list[Signal],
    ) -> None:
        """Test that a flat signal closes an existing position."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(sample_market_data, sample_signals)

        # Should have entry and exit fills
        assert len(result.fills) >= 2

    def test_multi_symbol_support(
        self,
        backtest_config: BacktestConfig,
        multi_symbol_data: pl.DataFrame,
        multi_symbol_signals: list[Signal],
    ) -> None:
        """Test engine handles multiple symbols correctly."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(multi_symbol_data, multi_symbol_signals)

        # Should have fills for both symbols
        symbols_traded = {f.symbol for f in result.fills}
        assert "AAPL" in symbols_traded
        assert "MSFT" in symbols_traded

    def test_empty_signals_returns_empty_result(
        self,
        backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
    ) -> None:
        """Test engine handles empty signals gracefully."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(sample_market_data, [])

        assert len(result.fills) == 0
        assert result.total_trades == 0


# =============================================================================
# T3.02: Quote-Based Fill Logic Tests
# =============================================================================


class TestFillSimulator:
    """Tests for quote-based fill simulation (T3.02)."""

    def test_buy_fills_at_ask_plus_slippage(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test buy orders fill at ask price plus slippage."""
        simulator = FillSimulator(cost_model=cost_model)

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=datetime.now(UTC),
            quote_time=datetime.now(UTC),
        )

        assert result.filled
        assert result.fill_price is not None
        assert result.fill_price >= 150.02  # At least ask price

    def test_sell_fills_at_bid_minus_slippage(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test sell orders fill at bid price minus slippage."""
        simulator = FillSimulator(cost_model=cost_model)

        result = simulator.simulate_fill(
            side="sell",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=datetime.now(UTC),
            quote_time=datetime.now(UTC),
        )

        assert result.filled
        assert result.fill_price is not None
        assert result.fill_price <= 150.00  # At most bid price

    def test_stale_quote_no_fill(self, cost_model: TransactionCostModel) -> None:
        """Test stale quotes result in no fill."""
        simulator = FillSimulator(
            cost_model=cost_model,
            max_quote_age_seconds=5.0,
        )

        order_time = datetime.now(UTC)
        quote_time = order_time - timedelta(seconds=10)  # Quote is 10s old

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=order_time,
            quote_time=quote_time,
        )

        assert not result.filled
        assert result.reject_reason == "stale_quote"

    def test_crossed_market_no_fill(self, cost_model: TransactionCostModel) -> None:
        """Test crossed markets (bid > ask) result in no fill."""
        simulator = FillSimulator(cost_model=cost_model)

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.05,  # Bid > Ask (crossed)
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=datetime.now(UTC),
            quote_time=datetime.now(UTC),
        )

        assert not result.filled
        assert result.reject_reason == "crossed_market"

    def test_fill_tracks_latency(self, cost_model: TransactionCostModel) -> None:
        """Test fill result tracks latency."""
        simulator = FillSimulator(cost_model=cost_model)

        order_time = datetime.now(UTC)
        quote_time = order_time - timedelta(milliseconds=100)

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=order_time,
            quote_time=quote_time,
        )

        assert result.filled
        assert result.latency_ms is not None
        assert result.latency_ms >= 100.0

    def test_fill_result_has_slippage_bps(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test fill result includes slippage in basis points."""
        simulator = FillSimulator(cost_model=cost_model)

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=900.0,
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=datetime.now(UTC),
            quote_time=datetime.now(UTC),
        )

        assert result.filled
        assert result.slippage_bps is not None
        assert result.slippage_bps >= 0

    def test_zero_size_quote_no_fill(self, cost_model: TransactionCostModel) -> None:
        """Test zero size on the relevant side results in no fill."""
        simulator = FillSimulator(cost_model=cost_model)

        result = simulator.simulate_fill(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            bid_size=1000.0,
            ask_size=0.0,  # No liquidity on ask side
            order_notional=10_000.0,
            short_term_vol=0.001,
            order_time=datetime.now(UTC),
            quote_time=datetime.now(UTC),
        )

        assert not result.filled
        assert result.reject_reason == "no_liquidity"


# =============================================================================
# T3.03: Slippage Model Integration Tests
# =============================================================================


class TestSlippageIntegrator:
    """Tests for slippage model integration (T3.03)."""

    def test_integrator_wraps_cost_model(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test integrator properly wraps the cost model."""
        integrator = SlippageIntegrator(cost_model=cost_model)
        assert integrator.cost_model == cost_model

    def test_calculate_slippage_for_buy(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test slippage calculation for buy orders."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        slippage_bps = integrator.calculate_slippage(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=10_000.0,
            book_notional=150_000.0,
            short_term_vol=0.001,
        )

        assert slippage_bps >= 0

    def test_calculate_slippage_for_sell(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test slippage calculation for sell orders."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        slippage_bps = integrator.calculate_slippage(
            side="sell",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=10_000.0,
            book_notional=150_000.0,
            short_term_vol=0.001,
        )

        assert slippage_bps >= 0

    def test_slippage_increases_with_order_size(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test that slippage increases with larger order sizes."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        small_order_slippage = integrator.calculate_slippage(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=5_000.0,
            book_notional=150_000.0,
            short_term_vol=0.001,
        )

        large_order_slippage = integrator.calculate_slippage(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=50_000.0,
            book_notional=150_000.0,
            short_term_vol=0.001,
        )

        assert large_order_slippage > small_order_slippage

    def test_slippage_increases_with_volatility(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test that slippage increases with higher volatility."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        low_vol_slippage = integrator.calculate_slippage(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=10_000.0,
            book_notional=150_000.0,
            short_term_vol=0.001,
        )

        high_vol_slippage = integrator.calculate_slippage(
            side="buy",
            bid_price=150.00,
            ask_price=150.02,
            order_notional=10_000.0,
            book_notional=150_000.0,
            short_term_vol=0.01,  # 10x higher vol
        )

        assert high_vol_slippage > low_vol_slippage

    def test_apply_slippage_to_price_buy(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test applying slippage to buy price."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        fill_price = integrator.apply_slippage(
            side="buy",
            base_price=150.02,  # Ask price
            slippage_bps=10.0,
        )

        # Price should be higher due to slippage
        expected = 150.02 * (1 + 10.0 / 10000)
        assert abs(fill_price - expected) < 0.0001

    def test_apply_slippage_to_price_sell(
        self, cost_model: TransactionCostModel
    ) -> None:
        """Test applying slippage to sell price."""
        integrator = SlippageIntegrator(cost_model=cost_model)

        fill_price = integrator.apply_slippage(
            side="sell",
            base_price=150.00,  # Bid price
            slippage_bps=10.0,
        )

        # Price should be lower due to slippage
        expected = 150.00 * (1 - 10.0 / 10000)
        assert abs(fill_price - expected) < 0.0001


# =============================================================================
# T3.04: Position Limits and Stops Tests
# =============================================================================


class TestRiskManager:
    """Tests for position limits and stops (T3.04)."""

    def test_risk_manager_init(self, risk_limits: RiskLimits) -> None:
        """Test risk manager initialization."""
        manager = RiskManager(risk_limits=risk_limits)
        assert manager.risk_limits == risk_limits
        assert manager.risk_checker is not None

    def test_max_position_size_enforced(self, risk_limits: RiskLimits) -> None:
        """Test maximum position size is enforced."""
        manager = RiskManager(risk_limits=risk_limits)

        # Try to create a position larger than max
        allowed, adjusted_size, violations = manager.check_order(
            symbol="AAPL",
            side="buy",
            notional=150_000.0,  # Over max_position_notional of 100k
            current_state=RiskState(),
        )

        # Should either reject or adjust
        assert not allowed or adjusted_size <= risk_limits.max_position_notional

    def test_max_gross_exposure_enforced(self, risk_limits: RiskLimits) -> None:
        """Test maximum gross exposure is enforced."""
        manager = RiskManager(risk_limits=risk_limits)

        # Create state with existing positions near limit
        state = RiskState(
            positions={"AAPL": 200_000.0, "MSFT": 200_000.0},  # 400k gross
        )

        # Try to add more exposure
        allowed, adjusted_size, violations = manager.check_order(
            symbol="GOOG",
            side="buy",
            notional=150_000.0,  # Would push gross to 550k > 500k limit
            current_state=state,
        )

        assert not allowed or adjusted_size <= 100_000.0

    def test_stop_loss_price_trigger(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test stop-loss triggers on price movement."""
        stop_config = StopConfig(
            stop_loss_pct=0.02,  # 2% stop loss
            take_profit_pct=0.05,
        )
        manager = RiskManager(risk_limits=risk_limits, stop_config=stop_config)

        # Enter long position at $150
        manager.record_entry(
            symbol="AAPL",
            side="buy",
            entry_price=150.00,
            entry_time=utc_now,
            notional=10_000.0,
        )

        # Check if stop triggered at $147 (2% below entry)
        should_exit = manager.check_stop(
            symbol="AAPL",
            current_price=146.90,  # Below stop level
            current_time=utc_now + timedelta(minutes=5),
        )

        assert should_exit
        assert manager.get_exit_reason("AAPL") == "stop_loss"

    def test_stop_loss_time_trigger(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test time-based stop triggers."""
        stop_config = StopConfig(
            max_hold_time_seconds=300,  # 5 minute max hold
        )
        manager = RiskManager(risk_limits=risk_limits, stop_config=stop_config)

        # Enter position
        manager.record_entry(
            symbol="AAPL",
            side="buy",
            entry_price=150.00,
            entry_time=utc_now,
            notional=10_000.0,
        )

        # Check after max hold time
        should_exit = manager.check_stop(
            symbol="AAPL",
            current_price=150.10,  # Price is fine
            current_time=utc_now + timedelta(seconds=350),  # Past max hold
        )

        assert should_exit
        assert manager.get_exit_reason("AAPL") == "time_stop"

    def test_take_profit_trigger(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test take-profit triggers on price movement."""
        stop_config = StopConfig(
            stop_loss_pct=0.02,
            take_profit_pct=0.03,  # 3% take profit
        )
        manager = RiskManager(risk_limits=risk_limits, stop_config=stop_config)

        # Enter long position
        manager.record_entry(
            symbol="AAPL",
            side="buy",
            entry_price=150.00,
            entry_time=utc_now,
            notional=10_000.0,
        )

        # Check if take-profit triggers at $154.50 (3% above entry)
        should_exit = manager.check_stop(
            symbol="AAPL",
            current_price=154.60,  # Above take profit level
            current_time=utc_now + timedelta(minutes=2),
        )

        assert should_exit
        assert manager.get_exit_reason("AAPL") == "take_profit"

    def test_short_position_stop_logic(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test stop loss works correctly for short positions."""
        stop_config = StopConfig(stop_loss_pct=0.02)
        manager = RiskManager(risk_limits=risk_limits, stop_config=stop_config)

        # Enter short position
        manager.record_entry(
            symbol="AAPL",
            side="sell",
            entry_price=150.00,
            entry_time=utc_now,
            notional=10_000.0,
        )

        # For shorts, stop triggers when price goes UP
        should_exit = manager.check_stop(
            symbol="AAPL",
            current_price=153.10,  # 2%+ above entry (bad for short)
            current_time=utc_now + timedelta(minutes=2),
        )

        assert should_exit

    def test_position_tracking(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test position entry/exit tracking."""
        manager = RiskManager(risk_limits=risk_limits)

        # Record entry
        manager.record_entry(
            symbol="AAPL",
            side="buy",
            entry_price=150.00,
            entry_time=utc_now,
            notional=10_000.0,
        )

        assert manager.has_position("AAPL")
        assert manager.get_position_side("AAPL") == "buy"

        # Record exit
        manager.record_exit(
            symbol="AAPL",
            exit_price=151.00,
            exit_time=utc_now + timedelta(minutes=5),
        )

        assert not manager.has_position("AAPL")

    def test_pnl_calculation(
        self, risk_limits: RiskLimits, utc_now: datetime
    ) -> None:
        """Test P&L calculation for positions."""
        manager = RiskManager(risk_limits=risk_limits)

        # Enter long position
        manager.record_entry(
            symbol="AAPL",
            side="buy",
            entry_price=150.00,
            entry_time=utc_now,
            notional=15_000.0,  # 100 shares
        )

        # Calculate unrealized P&L
        unrealized_pnl = manager.calculate_unrealized_pnl(
            symbol="AAPL",
            current_price=151.50,
        )

        # 100 shares * $1.50 gain = $150 (1% of 15000 = 150)
        expected_pnl = (151.50 - 150.00) / 150.00 * 15_000.0
        assert abs(unrealized_pnl - expected_pnl) < 0.01

    def test_daily_loss_kill_switch(self, risk_limits: RiskLimits) -> None:
        """Test kill switch triggers on daily loss limit."""
        manager = RiskManager(risk_limits=risk_limits)

        # Simulate losses
        state = RiskState(
            daily_pnl=-11_000.0,  # Over max_daily_loss of 10k
            peak_equity=100_000.0,
            current_equity=89_000.0,
        )

        should_halt = manager.check_kill_switch(state)
        assert should_halt


# =============================================================================
# Integration Tests
# =============================================================================


class TestBacktestIntegration:
    """Integration tests for the complete backtesting pipeline."""

    def test_backtest_with_stop_loss(
        self,
        cost_model: TransactionCostModel,
        risk_checker: RiskChecker,
        utc_now: datetime,
    ) -> None:
        """Test backtest with stop-loss logic."""
        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=cost_model,
            risk_checker=risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.02,  # 2% stop
        )

        # Create frames where price drops enough to trigger stop
        timestamps = [utc_now + timedelta(minutes=i) for i in range(5)]
        data = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["AAPL"] * 5,
            "bid_price": [150.00, 149.00, 148.00, 147.00, 146.00],  # Dropping
            "ask_price": [150.02, 149.02, 148.02, 147.02, 146.02],
            "bid_size": [1000.0] * 5,
            "ask_size": [900.0] * 5,
            "vol": [0.001] * 5,
        })

        signals = [
            Signal(timestamp=utc_now, symbol="AAPL", signal_type=SignalType.LONG),
            Signal(
                timestamp=utc_now + timedelta(minutes=2),
                symbol="AAPL",
                signal_type=SignalType.LONG,
            ),
            Signal(
                timestamp=utc_now + timedelta(minutes=3),
                symbol="AAPL",
                signal_type=SignalType.LONG,
            ),
        ]

        engine = BacktestEngine(config=config)
        result = engine.run(data, signals)

        # Position should have been entered and potentially stopped out
        assert len(result.fills) >= 1

    def test_backtest_respects_risk_limits(
        self,
        cost_model: TransactionCostModel,
        utc_now: datetime,
    ) -> None:
        """Test backtest respects position and exposure limits."""
        # Very restrictive limits
        strict_limits = RiskLimits(
            max_position_notional=5_000.0,
            max_gross_exposure=10_000.0,
            max_net_exposure=5_000.0,
            max_daily_loss=1_000.0,
            max_drawdown_pct=0.01,
        )
        strict_checker = RiskChecker(limits=strict_limits)

        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=cost_model,
            risk_checker=strict_checker,
            default_order_notional=10_000.0,  # Larger than allowed
        )

        timestamps = [utc_now + timedelta(minutes=i) for i in range(3)]
        data = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "bid_price": [150.00, 400.00, 170.00],
            "ask_price": [150.02, 400.05, 170.02],
            "bid_size": [1000.0, 500.0, 800.0],
            "ask_size": [900.0, 450.0, 750.0],
            "vol": [0.001, 0.0008, 0.0012],
        })

        signals = [
            Signal(timestamp=timestamps[0], symbol="AAPL", signal_type=SignalType.LONG),
            Signal(timestamp=timestamps[1], symbol="MSFT", signal_type=SignalType.LONG),
            Signal(timestamp=timestamps[2], symbol="GOOG", signal_type=SignalType.LONG),
        ]

        engine = BacktestEngine(config=config)
        result = engine.run(data, signals)

        # Should have risk violations recorded
        assert len(result.risk_violations) > 0 or all(
            f.notional <= strict_limits.max_position_notional for f in result.fills
        )

    def test_backtest_equity_curve(
        self,
        backtest_config: BacktestConfig,
        sample_market_data: pl.DataFrame,
        sample_signals: list[Signal],
    ) -> None:
        """Test backtest generates valid equity curve."""
        engine = BacktestEngine(config=backtest_config)
        result = engine.run(sample_market_data, sample_signals)

        # Equity curve should have data
        assert not result.equity_curve.is_empty()
        assert "timestamp" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_is_long(self) -> None:
        """Test is_long property."""
        pos = Position(symbol="AAPL", quantity=100.0, avg_entry_price=150.0)
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat

    def test_position_is_short(self) -> None:
        """Test is_short property."""
        pos = Position(symbol="AAPL", quantity=-100.0, avg_entry_price=150.0)
        assert not pos.is_long
        assert pos.is_short
        assert not pos.is_flat

    def test_position_is_flat(self) -> None:
        """Test is_flat property."""
        pos = Position(symbol="AAPL", quantity=0.0)
        assert not pos.is_long
        assert not pos.is_short
        assert pos.is_flat

    def test_update_mtm(self) -> None:
        """Test mark-to-market update."""
        pos = Position(symbol="AAPL", quantity=100.0, avg_entry_price=150.0)
        pos.update_mtm(155.0)

        assert pos.notional == 15500.0  # 100 * 155
        assert pos.unrealized_pnl == 500.0  # 100 * (155 - 150)


class TestFill:
    """Tests for Fill dataclass."""

    def test_fill_creation(self, utc_now: datetime) -> None:
        """Test Fill object creation."""
        fill = Fill(
            timestamp=utc_now,
            symbol="AAPL",
            side="buy",
            quantity=100.0,
            price=150.05,
            notional=15005.0,
            slippage_bps=1.5,
            commission=0.35,
        )

        assert fill.symbol == "AAPL"
        assert fill.side == "buy"
        assert fill.quantity == 100.0
        assert fill.price == 150.05
        assert fill.slippage_bps == 1.5
