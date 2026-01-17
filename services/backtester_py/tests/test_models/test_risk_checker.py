"""
Test suite for backtester_py.models.risk_checker module.

Tests written FIRST following TDD London School approach.
Focus on behavior verification with comprehensive edge case coverage.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from backtester_py.models.risk_checker import (
    PositionSizer,
    RiskChecker,
    RiskManager,
    RiskSeverity,
    RiskViolation,
)


# ============================================================================
# RiskSeverity Tests
# ============================================================================


class TestRiskSeverity:
    """Test RiskSeverity enumeration."""

    def test_warning_value(self) -> None:
        """WARNING severity should have correct value."""
        assert RiskSeverity.WARNING.value == "warning"

    def test_block_value(self) -> None:
        """BLOCK severity should have correct value."""
        assert RiskSeverity.BLOCK.value == "block"

    def test_enum_members(self) -> None:
        """Should have exactly two severity levels."""
        assert len(RiskSeverity) == 2
        assert RiskSeverity.WARNING in RiskSeverity
        assert RiskSeverity.BLOCK in RiskSeverity


# ============================================================================
# RiskViolation Tests
# ============================================================================


class TestRiskViolation:
    """Test RiskViolation dataclass."""

    def test_create_violation(self) -> None:
        """Should create a violation with all fields."""
        violation = RiskViolation(
            rule="max_position_pct",
            message="Position size 15.00% exceeds limit of 10.00%",
            severity=RiskSeverity.BLOCK,
        )
        assert violation.rule == "max_position_pct"
        assert "15.00%" in violation.message
        assert violation.severity == RiskSeverity.BLOCK

    def test_create_warning_violation(self) -> None:
        """Should create a warning violation."""
        violation = RiskViolation(
            rule="drawdown_warning",
            message="Current drawdown 16.00% is approaching limit of 20.00%",
            severity=RiskSeverity.WARNING,
        )
        assert violation.rule == "drawdown_warning"
        assert violation.severity == RiskSeverity.WARNING

    def test_violation_equality(self) -> None:
        """Two identical violations should be equal."""
        v1 = RiskViolation(
            rule="max_order_value",
            message="Order value exceeds limit",
            severity=RiskSeverity.BLOCK,
        )
        v2 = RiskViolation(
            rule="max_order_value",
            message="Order value exceeds limit",
            severity=RiskSeverity.BLOCK,
        )
        assert v1 == v2

    def test_violation_immutability(self) -> None:
        """RiskViolation should be immutable (frozen dataclass)."""
        violation = RiskViolation(
            rule="max_position_pct",
            message="Test",
            severity=RiskSeverity.BLOCK,
        )
        with pytest.raises(AttributeError):
            violation.rule = "modified"  # type: ignore[misc]


# ============================================================================
# RiskChecker Tests
# ============================================================================


class TestRiskCheckerCreation:
    """Test RiskChecker instantiation."""

    def test_default_parameters(self) -> None:
        """Should create checker with default parameters."""
        checker = RiskChecker()
        assert checker.max_position_pct == 0.10
        assert checker.max_order_value == Decimal("100000")
        assert checker.max_daily_trades == 100
        assert checker.max_drawdown_pct == 0.20

    def test_custom_parameters(self) -> None:
        """Should create checker with custom parameters."""
        checker = RiskChecker(
            max_position_pct=0.05,
            max_order_value=Decimal("50000"),
            max_daily_trades=50,
            max_drawdown_pct=0.15,
        )
        assert checker.max_position_pct == 0.05
        assert checker.max_order_value == Decimal("50000")
        assert checker.max_daily_trades == 50
        assert checker.max_drawdown_pct == 0.15


class TestRiskCheckerOrderValidation:
    """Test RiskChecker check_order method."""

    @pytest.fixture
    def checker(self) -> RiskChecker:
        """Standard risk checker for tests."""
        return RiskChecker(
            max_position_pct=0.10,
            max_order_value=Decimal("100000"),
            max_daily_trades=100,
            max_drawdown_pct=0.20,
        )

    def test_valid_order_no_violations(self, checker: RiskChecker) -> None:
        """Order within all limits should return no violations."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.05,
        )
        assert len(violations) == 0

    def test_exceeds_position_pct_limit(self, checker: RiskChecker) -> None:
        """Order exceeding position size limit should return BLOCK violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("15000"),  # 15% of portfolio
            portfolio_value=Decimal("100000"),
            current_drawdown=0.05,
        )
        assert len(violations) >= 1
        position_violations = [v for v in violations if v.rule == "max_position_pct"]
        assert len(position_violations) == 1
        assert position_violations[0].severity == RiskSeverity.BLOCK

    def test_exactly_at_position_limit(self, checker: RiskChecker) -> None:
        """Order exactly at position limit should pass."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("10000"),  # Exactly 10%
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        position_violations = [v for v in violations if v.rule == "max_position_pct"]
        assert len(position_violations) == 0

    def test_exceeds_max_order_value(self, checker: RiskChecker) -> None:
        """Order exceeding max order value should return BLOCK violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("150000"),  # Exceeds 100k limit
            portfolio_value=Decimal("2000000"),  # Large portfolio so position pct is OK
            current_drawdown=0.0,
        )
        order_violations = [v for v in violations if v.rule == "max_order_value"]
        assert len(order_violations) == 1
        assert order_violations[0].severity == RiskSeverity.BLOCK

    def test_exactly_at_max_order_value(self, checker: RiskChecker) -> None:
        """Order exactly at max order value should pass."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("100000"),  # Exactly at limit
            portfolio_value=Decimal("2000000"),
            current_drawdown=0.0,
        )
        order_violations = [v for v in violations if v.rule == "max_order_value"]
        assert len(order_violations) == 0

    def test_exceeds_daily_trades(self, checker: RiskChecker) -> None:
        """Exceeding daily trade limit should return BLOCK violation."""
        # Record 100 trades to hit the limit
        for _ in range(100):
            checker.record_trade()

        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("1000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        daily_violations = [v for v in violations if v.rule == "max_daily_trades"]
        assert len(daily_violations) == 1
        assert daily_violations[0].severity == RiskSeverity.BLOCK

    def test_exceeds_drawdown_limit(self, checker: RiskChecker) -> None:
        """Order with excessive drawdown should return BLOCK violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.25,  # 25% > 20% limit
        )
        drawdown_violations = [v for v in violations if v.rule == "max_drawdown"]
        assert len(drawdown_violations) == 1
        assert drawdown_violations[0].severity == RiskSeverity.BLOCK

    def test_drawdown_warning_when_approaching_limit(self, checker: RiskChecker) -> None:
        """Order near drawdown limit should return WARNING violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.17,  # 17% is 85% of 20% limit (>80%)
        )
        warning_violations = [v for v in violations if v.rule == "drawdown_warning"]
        assert len(warning_violations) == 1
        assert warning_violations[0].severity == RiskSeverity.WARNING

    def test_multiple_violations(self, checker: RiskChecker) -> None:
        """Order can trigger multiple violations."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("150000"),  # Exceeds max order value
            portfolio_value=Decimal("100000"),  # 150% position size
            current_drawdown=0.25,  # Exceeds drawdown
        )
        assert len(violations) >= 3
        rules = {v.rule for v in violations}
        assert "max_position_pct" in rules
        assert "max_order_value" in rules
        assert "max_drawdown" in rules

    def test_zero_notional_passes(self, checker: RiskChecker) -> None:
        """Zero notional order should pass (no-op)."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("0"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        assert len(violations) == 0

    def test_zero_portfolio_value_blocks(self, checker: RiskChecker) -> None:
        """Zero portfolio value should return BLOCK violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("0"),
            current_drawdown=0.0,
        )
        assert len(violations) >= 1
        zero_portfolio = [v for v in violations if v.rule == "zero_portfolio"]
        assert len(zero_portfolio) == 1
        assert zero_portfolio[0].severity == RiskSeverity.BLOCK

    def test_negative_portfolio_value_blocks(self, checker: RiskChecker) -> None:
        """Negative portfolio value should return BLOCK violation."""
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("-10000"),
            current_drawdown=0.0,
        )
        assert len(violations) >= 1
        invalid_portfolio = [v for v in violations if v.rule == "invalid_portfolio"]
        assert len(invalid_portfolio) == 1


class TestRiskCheckerTradeTracking:
    """Test RiskChecker trade counting and reset."""

    def test_record_trade_increments_count(self) -> None:
        """record_trade should increment daily trade count."""
        checker = RiskChecker(max_daily_trades=10)
        assert checker._daily_trade_count == 0
        checker.record_trade()
        assert checker._daily_trade_count == 1
        checker.record_trade()
        assert checker._daily_trade_count == 2

    def test_reset_daily_clears_count(self) -> None:
        """reset_daily should reset trade count to zero."""
        checker = RiskChecker(max_daily_trades=10)
        checker.record_trade()
        checker.record_trade()
        checker.record_trade()
        assert checker._daily_trade_count == 3
        checker.reset_daily()
        assert checker._daily_trade_count == 0


# ============================================================================
# PositionSizer Tests
# ============================================================================


class TestPositionSizerCreation:
    """Test PositionSizer instantiation."""

    def test_default_parameters(self) -> None:
        """Should create sizer with default parameters."""
        sizer = PositionSizer()
        assert sizer.risk_per_trade_pct == 0.02
        assert sizer.max_position_pct == 0.10

    def test_custom_parameters(self) -> None:
        """Should create sizer with custom parameters."""
        sizer = PositionSizer(
            risk_per_trade_pct=0.01,
            max_position_pct=0.05,
        )
        assert sizer.risk_per_trade_pct == 0.01
        assert sizer.max_position_pct == 0.05


class TestPositionSizerCalculation:
    """Test PositionSizer calculate_size method."""

    @pytest.fixture
    def sizer(self) -> PositionSizer:
        """Standard position sizer for tests."""
        return PositionSizer(
            risk_per_trade_pct=0.02,
            max_position_pct=0.10,
        )

    def test_basic_sizing(self, sizer: PositionSizer) -> None:
        """Should calculate position size based on stop distance."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),  # $5 stop distance
        )
        # Risk amount = 100000 * 0.02 = 2000
        # Shares = 2000 / 5 = 400
        # Notional = 400 * 100 = 40000
        # But capped at 10% = 10000
        assert size == Decimal("10000.00")

    def test_size_within_cap(self) -> None:
        """Should return uncapped size when below max position pct."""
        sizer = PositionSizer(risk_per_trade_pct=0.02, max_position_pct=0.50)
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),  # $5 stop distance
        )
        # Risk amount = 2000, shares = 400, notional = 40000
        # 40% of 100k, below 50% cap
        assert size == Decimal("40000.00")

    def test_respects_max_position_cap(self, sizer: PositionSizer) -> None:
        """Should cap position at max_position_pct."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("99.90"),  # Very tight stop, $0.10
        )
        # Risk amount = 2000, shares = 2000 / 0.10 = 20000
        # Notional would be 2,000,000 but capped at 10% = 10000
        assert size == Decimal("10000.00")

    def test_zero_account_balance(self, sizer: PositionSizer) -> None:
        """Zero account balance should return zero."""
        size = sizer.calculate_size(
            account_balance=Decimal("0"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
        )
        assert size == Decimal("0")

    def test_negative_account_balance(self, sizer: PositionSizer) -> None:
        """Negative account balance should return zero."""
        size = sizer.calculate_size(
            account_balance=Decimal("-10000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
        )
        assert size == Decimal("0")

    def test_zero_entry_price(self, sizer: PositionSizer) -> None:
        """Zero entry price should return zero."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("0"),
            stop_loss_price=Decimal("95.00"),
        )
        assert size == Decimal("0")

    def test_zero_stop_loss_price(self, sizer: PositionSizer) -> None:
        """Zero stop loss price should return zero."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("0"),
        )
        assert size == Decimal("0")

    def test_same_entry_and_stop(self, sizer: PositionSizer) -> None:
        """Same entry and stop price (zero risk) should return zero."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("100.00"),
        )
        assert size == Decimal("0")

    def test_short_position_sizing(self, sizer: PositionSizer) -> None:
        """Should handle short positions (stop above entry)."""
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("105.00"),  # Stop above for short
        )
        # Same calculation, absolute distance is $5
        # Risk = 2000, shares = 400, notional = 40000, capped at 10% = 10000
        assert size == Decimal("10000.00")

    def test_result_rounded_to_cents(self, sizer: PositionSizer) -> None:
        """Result should be rounded to 2 decimal places."""
        sizer = PositionSizer(risk_per_trade_pct=0.01, max_position_pct=1.0)
        size = sizer.calculate_size(
            account_balance=Decimal("10000"),
            entry_price=Decimal("33.33"),
            stop_loss_price=Decimal("32.00"),
        )
        # Risk = 100, distance = 1.33, shares = 75.188...
        # Notional = 75.188 * 33.33 = 2506.02...
        assert size == size.quantize(Decimal("0.01"))


# ============================================================================
# RiskManager Tests
# ============================================================================


class TestRiskManagerCreation:
    """Test RiskManager instantiation."""

    def test_creation_with_components(self) -> None:
        """Should create manager with checker and sizer."""
        checker = RiskChecker()
        sizer = PositionSizer()
        manager = RiskManager(checker, sizer)
        assert manager.checker is checker
        assert manager.sizer is sizer


class TestRiskManagerGetSizedOrder:
    """Test RiskManager get_sized_order method."""

    @pytest.fixture
    def manager(self) -> RiskManager:
        """Standard risk manager for tests."""
        checker = RiskChecker(
            max_position_pct=0.10,
            max_order_value=Decimal("100000"),
            max_daily_trades=100,
            max_drawdown_pct=0.20,
        )
        sizer = PositionSizer(
            risk_per_trade_pct=0.02,
            max_position_pct=0.10,
        )
        return RiskManager(checker, sizer)

    def test_sized_order_within_limits(self, manager: RiskManager) -> None:
        """Should return size and no violations for valid order."""
        size, violations = manager.get_sized_order(
            symbol="AAPL",
            side="buy",
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
            account_balance=Decimal("100000"),
            current_drawdown=0.05,
        )
        # Size should be calculated: risk=2000, shares=400, notional=40000
        # But capped at 10% = 10000
        assert size == Decimal("10000.00")
        assert len(violations) == 0

    def test_sized_order_with_violations(self, manager: RiskManager) -> None:
        """Should return size and violations when limits exceeded."""
        size, violations = manager.get_sized_order(
            symbol="AAPL",
            side="buy",
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
            account_balance=Decimal("100000"),
            current_drawdown=0.25,  # Exceeds drawdown limit
        )
        assert size == Decimal("10000.00")
        assert len(violations) >= 1
        assert any(v.rule == "max_drawdown" for v in violations)

    def test_sized_order_zero_result(self, manager: RiskManager) -> None:
        """Zero size order should return no violations."""
        size, violations = manager.get_sized_order(
            symbol="AAPL",
            side="buy",
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("100.00"),  # Zero risk
            account_balance=Decimal("100000"),
            current_drawdown=0.05,
        )
        assert size == Decimal("0")
        assert len(violations) == 0

    def test_record_trade_delegates(self, manager: RiskManager) -> None:
        """record_trade should delegate to checker."""
        assert manager.checker._daily_trade_count == 0
        manager.record_trade()
        assert manager.checker._daily_trade_count == 1

    def test_reset_daily_delegates(self, manager: RiskManager) -> None:
        """reset_daily should delegate to checker."""
        manager.record_trade()
        manager.record_trade()
        assert manager.checker._daily_trade_count == 2
        manager.reset_daily()
        assert manager.checker._daily_trade_count == 0


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases across all classes."""

    def test_very_small_position(self) -> None:
        """Very small position size should work."""
        sizer = PositionSizer(risk_per_trade_pct=0.001, max_position_pct=1.0)
        size = sizer.calculate_size(
            account_balance=Decimal("1000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("99.00"),
        )
        # Risk = 1, distance = 1, shares = 1, notional = 100
        assert size == Decimal("100.00")

    def test_very_large_account(self) -> None:
        """Very large account should work without overflow."""
        checker = RiskChecker(max_order_value=Decimal("1000000000"))
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("50000000"),
            portfolio_value=Decimal("1000000000"),  # $1B
            current_drawdown=0.0,
        )
        assert len(violations) == 0

    def test_very_small_stop_distance(self) -> None:
        """Very small stop distance should cap at max position."""
        sizer = PositionSizer(risk_per_trade_pct=0.02, max_position_pct=0.10)
        size = sizer.calculate_size(
            account_balance=Decimal("100000"),
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("99.99"),  # Only $0.01 stop
        )
        # Would be huge but capped at 10%
        assert size == Decimal("10000.00")

    def test_sell_side_order_validation(self) -> None:
        """Sell orders should be validated same as buy orders."""
        checker = RiskChecker(max_position_pct=0.10)
        violations = checker.check_order(
            symbol="AAPL",
            side="sell",
            notional=Decimal("15000"),  # 15% position
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        assert any(v.rule == "max_position_pct" for v in violations)


class TestIntegrationScenarios:
    """Integration tests for realistic trading scenarios."""

    def test_full_trading_workflow(self) -> None:
        """Test complete trading workflow."""
        checker = RiskChecker(
            max_position_pct=0.10,
            max_order_value=Decimal("50000"),
            max_daily_trades=5,
            max_drawdown_pct=0.15,
        )
        sizer = PositionSizer(risk_per_trade_pct=0.02, max_position_pct=0.10)
        manager = RiskManager(checker, sizer)

        # Trade 1: Valid
        size, violations = manager.get_sized_order(
            symbol="AAPL",
            side="buy",
            entry_price=Decimal("150.00"),
            stop_loss_price=Decimal("145.00"),
            account_balance=Decimal("100000"),
            current_drawdown=0.02,
        )
        assert size > Decimal("0")
        assert len(violations) == 0
        manager.record_trade()

        # Trade 2-5: More trades
        for _ in range(4):
            manager.record_trade()

        # Trade 6: Should hit daily limit
        _, violations = manager.get_sized_order(
            symbol="GOOGL",
            side="buy",
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
            account_balance=Decimal("100000"),
            current_drawdown=0.02,
        )
        assert any(v.rule == "max_daily_trades" for v in violations)

        # Reset daily and trade again
        manager.reset_daily()
        _, violations = manager.get_sized_order(
            symbol="GOOGL",
            side="buy",
            entry_price=Decimal("100.00"),
            stop_loss_price=Decimal("95.00"),
            account_balance=Decimal("100000"),
            current_drawdown=0.02,
        )
        daily_violations = [v for v in violations if v.rule == "max_daily_trades"]
        assert len(daily_violations) == 0

    def test_drawdown_progression(self) -> None:
        """Test drawdown warning and block progression."""
        checker = RiskChecker(max_drawdown_pct=0.20)

        # No warning at low drawdown
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.10,  # 10%
        )
        assert not any(v.rule in ("drawdown_warning", "max_drawdown") for v in violations)

        # Warning when approaching limit (>80% of max)
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.17,  # 85% of 20% limit
        )
        assert any(v.rule == "drawdown_warning" for v in violations)
        assert not any(v.rule == "max_drawdown" for v in violations)

        # Block when over limit
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("5000"),
            portfolio_value=Decimal("100000"),
            current_drawdown=0.25,  # Over 20% limit
        )
        assert any(v.rule == "max_drawdown" for v in violations)


class TestMathematicalCorrectness:
    """Tests verifying mathematical formulas."""

    def test_position_size_formula(self) -> None:
        """Verify position sizing formula: risk_amount / stop_distance * entry_price."""
        sizer = PositionSizer(risk_per_trade_pct=0.02, max_position_pct=1.0)

        account = Decimal("50000")
        entry = Decimal("200.00")
        stop = Decimal("190.00")

        size = sizer.calculate_size(
            account_balance=account,
            entry_price=entry,
            stop_loss_price=stop,
        )

        # Manual calculation:
        # risk_amount = 50000 * 0.02 = 1000
        # stop_distance = |200 - 190| = 10
        # shares = 1000 / 10 = 100
        # notional = 100 * 200 = 20000
        expected = Decimal("20000.00")
        assert size == expected

    def test_position_pct_formula(self) -> None:
        """Verify position percentage calculation."""
        checker = RiskChecker(max_position_pct=0.10)

        # 10% should pass, 11% should fail
        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("10000"),  # Exactly 10%
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        position_violations = [v for v in violations if v.rule == "max_position_pct"]
        assert len(position_violations) == 0

        violations = checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=Decimal("10001"),  # 10.001% > 10%
            portfolio_value=Decimal("100000"),
            current_drawdown=0.0,
        )
        position_violations = [v for v in violations if v.rule == "max_position_pct"]
        assert len(position_violations) == 1
