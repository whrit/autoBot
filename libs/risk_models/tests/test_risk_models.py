"""
Test suite for risk_models library.

Tests written FIRST following TDD London School approach.
Focus on behavior verification and mock-driven development.
"""

import pytest

from risk_models import (
    RiskChecker,
    RiskLimits,
    RiskState,
    RiskViolation,
    RiskViolationType,
)


class TestRiskViolationType:
    """Test risk violation type enumeration."""

    def test_violation_types_exist(self) -> None:
        """All expected violation types should exist."""
        assert RiskViolationType.MAX_POSITION == "max_position"
        assert RiskViolationType.MAX_GROSS_EXPOSURE == "max_gross_exposure"
        assert RiskViolationType.MAX_NET_EXPOSURE == "max_net_exposure"
        assert RiskViolationType.MAX_DAILY_LOSS == "max_daily_loss"
        assert RiskViolationType.MAX_DRAWDOWN == "max_drawdown"


class TestRiskViolation:
    """Test RiskViolation dataclass."""

    def test_create_violation(self) -> None:
        """Should create a violation with all fields."""
        violation = RiskViolation(
            type=RiskViolationType.MAX_POSITION,
            limit=100000.0,
            current=150000.0,
            message="Position exceeds limit",
        )
        assert violation.type == RiskViolationType.MAX_POSITION
        assert violation.limit == 100000.0
        assert violation.current == 150000.0
        assert violation.message == "Position exceeds limit"


class TestRiskLimits:
    """Test RiskLimits configuration dataclass."""

    def test_create_limits(self) -> None:
        """Should create limits with all parameters."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.05,
        )
        assert limits.max_position_notional == 100000.0
        assert limits.max_gross_exposure == 500000.0
        assert limits.max_net_exposure == 200000.0
        assert limits.max_daily_loss == 10000.0
        assert limits.max_drawdown_pct == 0.05

    def test_limits_validation_negative_position(self) -> None:
        """max_position_notional must be positive."""
        with pytest.raises(ValueError, match="max_position_notional must be positive"):
            RiskLimits(
                max_position_notional=-1000.0,
                max_gross_exposure=500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=0.05,
            )

    def test_limits_validation_zero_position(self) -> None:
        """max_position_notional must be positive (not zero)."""
        with pytest.raises(ValueError, match="max_position_notional must be positive"):
            RiskLimits(
                max_position_notional=0.0,
                max_gross_exposure=500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=0.05,
            )

    def test_limits_validation_negative_gross_exposure(self) -> None:
        """max_gross_exposure must be positive."""
        with pytest.raises(ValueError, match="max_gross_exposure must be positive"):
            RiskLimits(
                max_position_notional=100000.0,
                max_gross_exposure=-500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=0.05,
            )

    def test_limits_validation_negative_net_exposure(self) -> None:
        """max_net_exposure must be positive."""
        with pytest.raises(ValueError, match="max_net_exposure must be positive"):
            RiskLimits(
                max_position_notional=100000.0,
                max_gross_exposure=500000.0,
                max_net_exposure=-200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=0.05,
            )

    def test_limits_validation_negative_daily_loss(self) -> None:
        """max_daily_loss must be positive."""
        with pytest.raises(ValueError, match="max_daily_loss must be positive"):
            RiskLimits(
                max_position_notional=100000.0,
                max_gross_exposure=500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=-10000.0,
                max_drawdown_pct=0.05,
            )

    def test_limits_validation_drawdown_pct_zero(self) -> None:
        """max_drawdown_pct must be between 0 and 1 (exclusive of 0)."""
        with pytest.raises(ValueError, match="max_drawdown_pct must be between 0 and 1"):
            RiskLimits(
                max_position_notional=100000.0,
                max_gross_exposure=500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=0.0,
            )

    def test_limits_validation_drawdown_pct_above_one(self) -> None:
        """max_drawdown_pct must be between 0 and 1."""
        with pytest.raises(ValueError, match="max_drawdown_pct must be between 0 and 1"):
            RiskLimits(
                max_position_notional=100000.0,
                max_gross_exposure=500000.0,
                max_net_exposure=200000.0,
                max_daily_loss=10000.0,
                max_drawdown_pct=1.5,
            )

    def test_limits_validation_drawdown_pct_exactly_one(self) -> None:
        """max_drawdown_pct of exactly 1.0 should be valid (100% drawdown)."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=1.0,
        )
        assert limits.max_drawdown_pct == 1.0


class TestRiskState:
    """Test RiskState tracking dataclass."""

    def test_initial_state(self) -> None:
        """Initial state should have zero exposure."""
        state = RiskState()
        assert state.current_gross_exposure == 0.0
        assert state.current_net_exposure == 0.0
        assert state.daily_pnl == 0.0
        assert state.peak_equity == 0.0
        assert state.current_equity == 0.0

    def test_update_position_long(self) -> None:
        """Should track long position correctly."""
        state = RiskState()
        state.update_position("SPY", 10000.0)
        assert state.get_position("SPY") == 10000.0
        assert state.current_gross_exposure == 10000.0
        assert state.current_net_exposure == 10000.0

    def test_update_position_short(self) -> None:
        """Should track short position correctly."""
        state = RiskState()
        state.update_position("SPY", -5000.0)
        assert state.get_position("SPY") == -5000.0
        assert state.current_gross_exposure == 5000.0
        assert state.current_net_exposure == 5000.0

    def test_update_position_multiple_symbols(self) -> None:
        """Should calculate exposure across multiple symbols."""
        state = RiskState()
        state.update_position("SPY", 10000.0)  # Long $10k
        state.update_position("QQQ", -5000.0)  # Short $5k
        assert state.current_gross_exposure == 15000.0
        assert state.current_net_exposure == 5000.0  # Net long $5k

    def test_update_position_accumulate(self) -> None:
        """Should accumulate position on same symbol."""
        state = RiskState()
        state.update_position("SPY", 10000.0)
        state.update_position("SPY", 5000.0)
        assert state.get_position("SPY") == 15000.0

    def test_update_position_reduce_to_zero(self) -> None:
        """Should remove position when reduced to zero."""
        state = RiskState()
        state.update_position("SPY", 10000.0)
        state.update_position("SPY", -10000.0)
        assert state.get_position("SPY") == 0.0
        assert "SPY" not in state.positions

    def test_get_position_nonexistent(self) -> None:
        """Should return 0 for nonexistent position."""
        state = RiskState()
        assert state.get_position("AAPL") == 0.0

    def test_current_drawdown_pct_zero_peak(self) -> None:
        """Should return 0 drawdown when peak is zero."""
        state = RiskState()
        state.peak_equity = 0.0
        state.current_equity = 0.0
        assert state.current_drawdown_pct == 0.0

    def test_current_drawdown_pct_calculation(self) -> None:
        """Should calculate drawdown percentage correctly."""
        state = RiskState()
        state.peak_equity = 100000.0
        state.current_equity = 90000.0
        assert state.current_drawdown_pct == 0.1  # 10% drawdown

    def test_current_drawdown_pct_no_drawdown(self) -> None:
        """Should return 0 when at peak."""
        state = RiskState()
        state.peak_equity = 100000.0
        state.current_equity = 100000.0
        assert state.current_drawdown_pct == 0.0


class TestRiskChecker:
    """Test RiskChecker pre-trade and state checks."""

    @pytest.fixture
    def default_limits(self) -> RiskLimits:
        """Standard risk limits for testing."""
        return RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.05,
        )

    def test_check_order_within_limits(self, default_limits: RiskLimits) -> None:
        """Small order should pass all checks."""
        checker = RiskChecker(default_limits)
        state = RiskState()

        violations = checker.check_order(
            symbol="SPY",
            side="buy",
            notional=10000.0,
            state=state,
        )
        assert len(violations) == 0

    def test_check_order_exceeds_position_limit(self) -> None:
        """Order exceeding position limit should be flagged."""
        limits = RiskLimits(
            max_position_notional=10000.0,  # Small limit
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.05,
        )
        checker = RiskChecker(limits)
        state = RiskState()

        violations = checker.check_order(
            symbol="SPY",
            side="buy",
            notional=50000.0,
            state=state,
        )
        assert len(violations) > 0
        assert any(v.type == RiskViolationType.MAX_POSITION for v in violations)

    def test_check_order_exceeds_gross_exposure(self) -> None:
        """Order exceeding gross exposure should be flagged."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=20000.0,  # Small limit
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.05,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.update_position("QQQ", 15000.0)  # Existing position

        violations = checker.check_order(
            symbol="SPY",
            side="buy",
            notional=10000.0,  # Would make gross = 25000 > 20000
            state=state,
        )
        assert any(v.type == RiskViolationType.MAX_GROSS_EXPOSURE for v in violations)

    def test_check_order_exceeds_net_exposure(self) -> None:
        """Order exceeding net exposure should be flagged."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=10000.0,  # Small limit
            max_daily_loss=10000.0,
            max_drawdown_pct=0.05,
        )
        checker = RiskChecker(limits)
        state = RiskState()

        violations = checker.check_order(
            symbol="SPY",
            side="buy",
            notional=20000.0,  # Would make net = 20000 > 10000
            state=state,
        )
        assert any(v.type == RiskViolationType.MAX_NET_EXPOSURE for v in violations)

    def test_check_order_sell_side(self, default_limits: RiskLimits) -> None:
        """Sell order should decrease position correctly."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.update_position("SPY", 50000.0)  # Existing long

        violations = checker.check_order(
            symbol="SPY",
            side="sell",
            notional=30000.0,  # Reduce to 20000
            state=state,
        )
        assert len(violations) == 0

    def test_check_order_sell_creates_short(self, default_limits: RiskLimits) -> None:
        """Sell order can flip to short position."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.update_position("SPY", 10000.0)  # Small long

        violations = checker.check_order(
            symbol="SPY",
            side="sell",
            notional=50000.0,  # Flip to -40000 short
            state=state,
        )
        assert len(violations) == 0

    def test_check_state_no_violations(self, default_limits: RiskLimits) -> None:
        """Healthy state should have no violations."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.daily_pnl = 5000.0
        state.peak_equity = 100000.0
        state.current_equity = 99000.0

        violations = checker.check_state(state)
        assert len(violations) == 0

    def test_check_state_daily_loss_violation(self) -> None:
        """Excessive daily loss should be flagged."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=5000.0,  # $5k max daily loss
            max_drawdown_pct=0.10,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.daily_pnl = -6000.0  # Lost $6k today

        violations = checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_DAILY_LOSS for v in violations)

    def test_check_state_drawdown_violation(self, default_limits: RiskLimits) -> None:
        """Excessive drawdown should be flagged."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.peak_equity = 100000.0
        state.current_equity = 90000.0  # 10% drawdown > 5% limit

        violations = checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_DRAWDOWN for v in violations)

    def test_check_state_gross_exposure_violation(self) -> None:
        """Excessive gross exposure in state should be flagged."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=50000.0,  # Small limit
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.10,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.update_position("SPY", 30000.0)
        state.update_position("QQQ", 30000.0)  # Gross = 60000 > 50000

        violations = checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_GROSS_EXPOSURE for v in violations)

    def test_is_kill_switch_triggered_false(self, default_limits: RiskLimits) -> None:
        """Kill switch should not trigger on healthy state."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.daily_pnl = 1000.0
        state.peak_equity = 100000.0
        state.current_equity = 99000.0

        assert checker.is_kill_switch_triggered(state) is False

    def test_is_kill_switch_triggered_daily_loss(self) -> None:
        """Kill switch should trigger on daily loss violation."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=500000.0,
            max_net_exposure=200000.0,
            max_daily_loss=5000.0,
            max_drawdown_pct=0.10,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.daily_pnl = -6000.0

        assert checker.is_kill_switch_triggered(state) is True

    def test_is_kill_switch_triggered_drawdown(self, default_limits: RiskLimits) -> None:
        """Kill switch should trigger on drawdown violation."""
        checker = RiskChecker(default_limits)
        state = RiskState()
        state.peak_equity = 100000.0
        state.current_equity = 80000.0  # 20% drawdown

        assert checker.is_kill_switch_triggered(state) is True

    def test_is_kill_switch_not_triggered_gross_exposure(self) -> None:
        """Kill switch should NOT trigger on gross exposure alone."""
        limits = RiskLimits(
            max_position_notional=100000.0,
            max_gross_exposure=50000.0,  # Will be violated
            max_net_exposure=200000.0,
            max_daily_loss=10000.0,
            max_drawdown_pct=0.10,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.update_position("SPY", 60000.0)  # Gross > limit
        state.peak_equity = 100000.0
        state.current_equity = 100000.0
        state.daily_pnl = 0.0

        # Gross exposure violation exists but kill switch not triggered
        violations = checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_GROSS_EXPOSURE for v in violations)
        assert checker.is_kill_switch_triggered(state) is False


class TestRiskModelsIntegration:
    """Integration tests for risk models workflow."""

    def test_full_trading_workflow(self) -> None:
        """Simulate a full trading day workflow."""
        limits = RiskLimits(
            max_position_notional=50000.0,
            max_gross_exposure=100000.0,
            max_net_exposure=75000.0,
            max_daily_loss=5000.0,
            max_drawdown_pct=0.03,
        )
        checker = RiskChecker(limits)
        state = RiskState()
        state.peak_equity = 100000.0
        state.current_equity = 100000.0

        # Trade 1: Buy SPY - should pass
        v1 = checker.check_order("SPY", "buy", 30000.0, state)
        assert len(v1) == 0
        state.update_position("SPY", 30000.0)

        # Trade 2: Buy QQQ - should pass
        v2 = checker.check_order("QQQ", "buy", 25000.0, state)
        assert len(v2) == 0
        state.update_position("QQQ", 25000.0)

        # Trade 3: Buy more SPY - should fail (position limit)
        v3 = checker.check_order("SPY", "buy", 25000.0, state)
        assert any(v.type == RiskViolationType.MAX_POSITION for v in v3)

        # Trade 4: Sell some SPY - should pass
        v4 = checker.check_order("SPY", "sell", 10000.0, state)
        assert len(v4) == 0
        state.update_position("SPY", -10000.0)

        # Record some P&L - stay within 3% drawdown limit
        state.daily_pnl = -2000.0
        state.current_equity = 98000.0  # 2% drawdown < 3% limit

        # Check state - should be OK
        sv1 = checker.check_state(state)
        assert len(sv1) == 0

        # More losses - exceed daily loss limit
        state.daily_pnl = -6000.0
        state.current_equity = 94000.0  # 6% drawdown > 3% limit

        # Check state - should trigger both daily loss and drawdown
        sv2 = checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_DAILY_LOSS for v in sv2)
        assert checker.is_kill_switch_triggered(state)
