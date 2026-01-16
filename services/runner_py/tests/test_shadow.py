"""
Tests for Shadow Executor (T6.01).

Tests shadow execution mode that simulates signal execution
without placing real orders.
"""

from datetime import UTC, datetime

import pytest

from runner_py.shadow import ShadowConfig, ShadowExecutor
from runner_py.types import (
    ExecutionMode,
    MarketData,
    ShadowExecutionResult,
    ShadowFill,
    Signal,
    SignalDirection,
)


@pytest.fixture
def shadow_config() -> ShadowConfig:
    """Create test shadow config."""
    return ShadowConfig(
        strategy_id="test-strategy",
        symbols=["SPY", "QQQ", "AAPL"],
        max_position_size=10000.0,
        slippage_model_bps=2.0,
        check_risk_limits=True,
    )


@pytest.fixture
def sample_market_data() -> MarketData:
    """Create sample market data for SPY."""
    return MarketData(
        symbol="SPY",
        timestamp=datetime.now(UTC),
        bid=450.00,
        ask=450.10,
        last=450.05,
        bid_size=1000.0,
        ask_size=800.0,
    )


@pytest.fixture
def sample_long_signal() -> Signal:
    """Create a sample LONG signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="SPY",
        signal_type=SignalDirection.LONG,
        strength=0.8,
        target_notional=5000.0,
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_short_signal() -> Signal:
    """Create a sample SHORT signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="SPY",
        signal_type=SignalDirection.SHORT,
        strength=0.7,
        target_notional=3000.0,
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_flat_signal() -> Signal:
    """Create a sample FLAT signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="SPY",
        signal_type=SignalDirection.FLAT,
        strength=1.0,
        target_notional=0.0,
        strategy_id="test-strategy",
    )


class TestShadowConfig:
    """Tests for ShadowConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = ShadowConfig(
            strategy_id="test",
            symbols=["SPY"],
        )
        assert config.max_position_size == 10000.0
        assert config.slippage_model_bps == 2.0
        assert config.check_risk_limits is True

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = ShadowConfig(
            strategy_id="test",
            symbols=["SPY", "QQQ"],
            max_position_size=50000.0,
            slippage_model_bps=5.0,
            check_risk_limits=False,
        )
        assert config.max_position_size == 50000.0
        assert config.slippage_model_bps == 5.0
        assert config.check_risk_limits is False

    def test_multiple_symbols(self) -> None:
        """Test config with multiple symbols."""
        config = ShadowConfig(
            strategy_id="multi-symbol",
            symbols=["SPY", "QQQ", "AAPL", "MSFT"],
        )
        assert len(config.symbols) == 4
        assert "SPY" in config.symbols


class TestShadowExecutorInit:
    """Tests for ShadowExecutor initialization."""

    def test_init_creates_executor(self, shadow_config: ShadowConfig) -> None:
        """Test executor initialization."""
        executor = ShadowExecutor(shadow_config)
        assert executor.config == shadow_config
        assert executor._positions == {}
        assert executor._trade_history == []

    def test_init_with_symbols(self, shadow_config: ShadowConfig) -> None:
        """Test executor tracks configured symbols."""
        executor = ShadowExecutor(shadow_config)
        assert "SPY" in executor.config.symbols
        assert "QQQ" in executor.config.symbols


class TestShadowExecutorExecute:
    """Tests for executing signals."""

    def test_execute_long_signal(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test executing a LONG signal."""
        executor = ShadowExecutor(shadow_config)
        result = executor.execute(sample_long_signal, sample_market_data)

        assert isinstance(result, ShadowExecutionResult)
        assert result.mode == ExecutionMode.SHADOW
        assert result.signal == sample_long_signal
        assert result.success is True
        assert result.fill is not None
        assert result.fill.would_have_executed is True

    def test_execute_short_signal(
        self,
        shadow_config: ShadowConfig,
        sample_short_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test executing a SHORT signal."""
        executor = ShadowExecutor(shadow_config)
        result = executor.execute(sample_short_signal, sample_market_data)

        assert result.success is True
        assert result.fill is not None
        assert result.fill.would_have_executed is True

    def test_execute_flat_signal(
        self,
        shadow_config: ShadowConfig,
        sample_flat_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test executing a FLAT signal."""
        executor = ShadowExecutor(shadow_config)
        result = executor.execute(sample_flat_signal, sample_market_data)

        assert result.success is True
        # FLAT signal with no position should be a no-op
        assert result.message == "No position to close"

    def test_execute_flat_closes_position(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_flat_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test FLAT signal closes existing position."""
        executor = ShadowExecutor(shadow_config)

        # First open a position
        executor.execute(sample_long_signal, sample_market_data)
        assert executor.get_position("SPY") != 0

        # Then close it
        result = executor.execute(sample_flat_signal, sample_market_data)

        assert result.success is True
        assert executor.get_position("SPY") == 0


class TestShadowExecutorSimulateFill:
    """Tests for fill simulation."""

    def test_simulate_fill_buy_at_ask(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test buy orders fill at ask price plus slippage."""
        executor = ShadowExecutor(shadow_config)
        fill = executor.simulate_fill(sample_long_signal, sample_market_data)

        # Buy should be at ask price + slippage
        expected_base = sample_market_data.ask
        expected_slippage = expected_base * (shadow_config.slippage_model_bps / 10000)
        expected_price = expected_base + expected_slippage

        assert fill.simulated_price == pytest.approx(expected_price, rel=1e-4)

    def test_simulate_fill_sell_at_bid(
        self,
        shadow_config: ShadowConfig,
        sample_short_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test sell orders fill at bid price minus slippage."""
        executor = ShadowExecutor(shadow_config)
        fill = executor.simulate_fill(sample_short_signal, sample_market_data)

        # Sell should be at bid price - slippage
        expected_base = sample_market_data.bid
        expected_slippage = expected_base * (shadow_config.slippage_model_bps / 10000)
        expected_price = expected_base - expected_slippage

        assert fill.simulated_price == pytest.approx(expected_price, rel=1e-4)

    def test_slippage_applied(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test slippage is correctly applied to fills."""
        executor = ShadowExecutor(shadow_config)
        fill = executor.simulate_fill(sample_long_signal, sample_market_data)

        assert fill.simulated_slippage_bps == shadow_config.slippage_model_bps
        assert fill.simulated_price > sample_market_data.ask  # Slippage for buy


class TestShadowExecutorRiskLimits:
    """Tests for risk limit checking."""

    def test_risk_limit_check_passes(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
    ) -> None:
        """Test risk limit check passes for normal signal."""
        executor = ShadowExecutor(shadow_config)
        passed, reason = executor.check_risk_limits(sample_long_signal)

        assert passed is True
        assert reason is None

    def test_risk_limit_check_exceeds_max_position(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test risk limit check fails when exceeding max position."""
        executor = ShadowExecutor(shadow_config)

        big_signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal_type=SignalDirection.LONG,
            strength=1.0,
            target_notional=50000.0,  # Exceeds max of 10000
            strategy_id="test-strategy",
        )

        passed, reason = executor.check_risk_limits(big_signal)

        assert passed is False
        assert "exceeds maximum" in reason.lower()

    def test_risk_limit_check_invalid_symbol(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test risk limit check fails for unconfigured symbol."""
        executor = ShadowExecutor(shadow_config)

        invalid_signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="INVALID",  # Not in configured symbols
            signal_type=SignalDirection.LONG,
            strength=0.5,
            target_notional=1000.0,
            strategy_id="test-strategy",
        )

        passed, reason = executor.check_risk_limits(invalid_signal)

        assert passed is False
        assert "not in allowed symbols" in reason.lower()

    def test_risk_limit_check_disabled(self) -> None:
        """Test risk limit check can be disabled."""
        config = ShadowConfig(
            strategy_id="test",
            symbols=["SPY"],
            max_position_size=1000.0,
            check_risk_limits=False,
        )
        executor = ShadowExecutor(config)

        big_signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal_type=SignalDirection.LONG,
            strength=1.0,
            target_notional=50000.0,
            strategy_id="test-strategy",
        )

        passed, reason = executor.check_risk_limits(big_signal)
        assert passed is True  # Risk limits disabled


class TestShadowExecutorPositionTracking:
    """Tests for position tracking."""

    def test_position_tracking_initial_flat(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test initial positions are flat."""
        executor = ShadowExecutor(shadow_config)
        assert executor.get_position("SPY") == 0
        assert executor.get_position("QQQ") == 0

    def test_position_tracking_after_long(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test position tracking after long signal."""
        executor = ShadowExecutor(shadow_config)
        executor.execute(sample_long_signal, sample_market_data)

        position = executor.get_position("SPY")
        assert position > 0  # Long position

    def test_position_tracking_after_short(
        self,
        shadow_config: ShadowConfig,
        sample_short_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test position tracking after short signal."""
        executor = ShadowExecutor(shadow_config)
        executor.execute(sample_short_signal, sample_market_data)

        position = executor.get_position("SPY")
        assert position < 0  # Short position

    def test_update_position_from_fill(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test position updates from fill."""
        executor = ShadowExecutor(shadow_config)
        fill = executor.simulate_fill(sample_long_signal, sample_market_data)
        fill = ShadowFill(
            signal=fill.signal,
            simulated_price=fill.simulated_price,
            simulated_slippage_bps=fill.simulated_slippage_bps,
            timestamp=fill.timestamp,
            would_have_executed=True,
        )

        executor.update_position(fill)
        assert executor.get_position("SPY") != 0


class TestShadowExecutorPnL:
    """Tests for P&L calculation."""

    def test_pnl_initial_zero(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test initial P&L is zero."""
        executor = ShadowExecutor(shadow_config)
        pnl = executor.get_pnl()

        assert pnl.get("SPY", 0) == 0
        assert pnl.get("QQQ", 0) == 0

    def test_pnl_calculation_long_profit(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test P&L calculation for profitable long position."""
        executor = ShadowExecutor(shadow_config)

        # Open long position
        executor.execute(sample_long_signal, sample_market_data)

        # Simulate price increase
        higher_market = MarketData(
            symbol="SPY",
            timestamp=datetime.now(UTC),
            bid=455.00,  # Price went up
            ask=455.10,
            last=455.05,
        )
        executor.mark_to_market(higher_market)

        pnl = executor.get_pnl()
        assert pnl.get("SPY", 0) > 0  # Profitable

    def test_pnl_calculation_short_profit(
        self,
        shadow_config: ShadowConfig,
        sample_short_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test P&L calculation for profitable short position."""
        executor = ShadowExecutor(shadow_config)

        # Open short position
        executor.execute(sample_short_signal, sample_market_data)

        # Simulate price decrease
        lower_market = MarketData(
            symbol="SPY",
            timestamp=datetime.now(UTC),
            bid=445.00,  # Price went down
            ask=445.10,
            last=445.05,
        )
        executor.mark_to_market(lower_market)

        pnl = executor.get_pnl()
        assert pnl.get("SPY", 0) > 0  # Profitable

    def test_get_total_pnl(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test getting total P&L across all symbols."""
        executor = ShadowExecutor(shadow_config)

        # Simulate some trades
        now = datetime.now(UTC)

        spy_signal = Signal(
            timestamp=now,
            symbol="SPY",
            signal_type=SignalDirection.LONG,
            strength=0.8,
            target_notional=5000.0,
            strategy_id="test-strategy",
        )
        spy_market = MarketData(
            symbol="SPY", timestamp=now, bid=450.0, ask=450.1, last=450.05
        )

        qqq_signal = Signal(
            timestamp=now,
            symbol="QQQ",
            signal_type=SignalDirection.SHORT,
            strength=0.7,
            target_notional=3000.0,
            strategy_id="test-strategy",
        )
        qqq_market = MarketData(
            symbol="QQQ", timestamp=now, bid=380.0, ask=380.1, last=380.05
        )

        executor.execute(spy_signal, spy_market)
        executor.execute(qqq_signal, qqq_market)

        total_pnl = executor.get_total_pnl()
        assert isinstance(total_pnl, float)


class TestShadowExecutorTradeHistory:
    """Tests for trade history tracking."""

    def test_trade_history_empty_initially(
        self,
        shadow_config: ShadowConfig,
    ) -> None:
        """Test trade history is empty initially."""
        executor = ShadowExecutor(shadow_config)
        history = executor.get_trade_history()

        assert len(history) == 0

    def test_trade_history_records_trades(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test trade history records executed trades."""
        executor = ShadowExecutor(shadow_config)
        executor.execute(sample_long_signal, sample_market_data)

        history = executor.get_trade_history()
        assert len(history) == 1
        assert history[0].signal == sample_long_signal

    def test_trade_history_preserves_order(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_short_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test trade history preserves chronological order."""
        executor = ShadowExecutor(shadow_config)

        executor.execute(sample_long_signal, sample_market_data)
        executor.execute(sample_short_signal, sample_market_data)

        history = executor.get_trade_history()
        assert len(history) == 2
        assert history[0].signal == sample_long_signal
        assert history[1].signal == sample_short_signal


class TestShadowExecutorMetrics:
    """Tests for executor metrics."""

    def test_get_metrics(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test getting executor metrics."""
        executor = ShadowExecutor(shadow_config)
        executor.execute(sample_long_signal, sample_market_data)

        metrics = executor.get_metrics()

        assert "total_trades" in metrics
        assert "total_pnl" in metrics
        assert "positions" in metrics
        assert metrics["total_trades"] == 1

    def test_get_position_summary(
        self,
        shadow_config: ShadowConfig,
        sample_long_signal: Signal,
        sample_market_data: MarketData,
    ) -> None:
        """Test getting position summary."""
        executor = ShadowExecutor(shadow_config)
        executor.execute(sample_long_signal, sample_market_data)

        summary = executor.get_position_summary()

        assert "SPY" in summary
        assert summary["SPY"]["quantity"] != 0
