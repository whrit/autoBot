"""
Integration tests for cost_models and risk_models working together.

Tests cover:
- CostModel + RiskChecker combined workflows
- Realistic trading scenarios
- Order flow: check risk -> calculate fill -> update state
- End-to-end trading simulations
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cost_models import SlippageModel, TransactionCostModel
from risk_models import RiskChecker, RiskLimits, RiskState, RiskViolationType


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture
def realistic_slippage_model() -> SlippageModel:
    """Realistic slippage model for equity trading."""
    return SlippageModel(
        spread_coef=1.0,
        size_coef=0.5,
        vol_coef=2.0,
        max_size_impact_bps=100.0,
    )


@pytest.fixture
def realistic_cost_model(realistic_slippage_model: SlippageModel) -> TransactionCostModel:
    """Realistic transaction cost model for equity trading."""
    return TransactionCostModel(
        slippage_model=realistic_slippage_model,
        fixed_cost_bps=0.35,  # Commission + fees
    )


@pytest.fixture
def realistic_risk_limits() -> RiskLimits:
    """Realistic risk limits for a $1M account."""
    return RiskLimits(
        max_position_notional=100_000,  # 10% max per position
        max_gross_exposure=500_000,  # 50% max gross
        max_net_exposure=300_000,  # 30% max net directional
        max_daily_loss=20_000,  # 2% daily loss limit
        max_drawdown_pct=0.10,  # 10% max drawdown
    )


@pytest.fixture
def realistic_risk_checker(realistic_risk_limits: RiskLimits) -> RiskChecker:
    """Risk checker with realistic limits."""
    return RiskChecker(limits=realistic_risk_limits)


@pytest.fixture
def account_state() -> RiskState:
    """Fresh account state for testing."""
    return RiskState(
        positions={},
        daily_pnl=0.0,
        peak_equity=1_000_000,
        current_equity=1_000_000,
    )


# ============================================================================
# Order Flow Integration Tests
# ============================================================================


class TestOrderFlowIntegration:
    """Test complete order flow from risk check to fill."""

    def test_successful_buy_order_flow(
        self,
        realistic_cost_model: TransactionCostModel,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test successful buy order from risk check to fill."""
        symbol = "AAPL"
        order_notional = 50_000
        bid_price = 150.00
        ask_price = 150.05

        # Step 1: Check risk
        violations = realistic_risk_checker.check_order(
            symbol=symbol,
            side="buy",
            notional=order_notional,
            state=account_state,
        )
        assert len(violations) == 0, "Order should pass risk checks"

        # Step 2: Calculate fill price
        book_notional = ask_price * 5000  # Assume 5000 shares on ask
        fill_price = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=0.02,
        )

        # Step 3: Verify fill is reasonable
        assert fill_price > ask_price, "Buy fill should be above ask"
        assert fill_price < ask_price * 1.01, "Fill should be within 1% of ask"

        # Step 4: Update state
        account_state.update_position(symbol, order_notional)
        assert account_state.get_position(symbol) == order_notional
        assert account_state.current_gross_exposure == order_notional

    def test_successful_sell_order_flow(
        self,
        realistic_cost_model: TransactionCostModel,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test successful sell order (short) from risk check to fill."""
        symbol = "GOOGL"
        order_notional = 40_000
        bid_price = 140.00
        ask_price = 140.10

        # Step 1: Check risk
        violations = realistic_risk_checker.check_order(
            symbol=symbol,
            side="sell",
            notional=order_notional,
            state=account_state,
        )
        assert len(violations) == 0, "Short order should pass risk checks"

        # Step 2: Calculate fill price
        book_notional = bid_price * 3000
        fill_price = realistic_cost_model.calculate_fill_price(
            side="sell",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=0.015,
        )

        # Step 3: Verify fill is reasonable
        assert fill_price < bid_price, "Sell fill should be below bid"
        assert fill_price > bid_price * 0.99, "Fill should be within 1% of bid"

        # Step 4: Update state (negative for short)
        account_state.update_position(symbol, -order_notional)
        assert account_state.get_position(symbol) == -order_notional

    def test_order_rejected_by_risk_check(
        self,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test order that gets rejected by risk check."""
        # Order exceeds position limit (100k max, ordering 150k)
        violations = realistic_risk_checker.check_order(
            symbol="AAPL",
            side="buy",
            notional=150_000,
            state=account_state,
        )

        assert len(violations) > 0, "Order should be rejected"
        assert any(v.type == RiskViolationType.MAX_POSITION for v in violations)

        # Verify state unchanged
        assert account_state.get_position("AAPL") == 0

    def test_order_rejected_after_drawdown(
        self,
        realistic_risk_checker: RiskChecker,
    ) -> None:
        """Test order rejection when drawdown limit exceeded."""
        # Account with significant drawdown
        state = RiskState(
            positions={},
            daily_pnl=-50_000,
            peak_equity=1_000_000,
            current_equity=850_000,  # 15% drawdown > 10% limit
        )

        # Check state violations
        state_violations = realistic_risk_checker.check_state(state)
        assert any(v.type == RiskViolationType.MAX_DRAWDOWN for v in state_violations)

        # Kill switch should be triggered
        assert realistic_risk_checker.is_kill_switch_triggered(state) is True


# ============================================================================
# Realistic Trading Scenarios
# ============================================================================


class TestRealisticTradingScenarios:
    """Test realistic multi-trade scenarios."""

    def test_multiple_positions_workflow(
        self,
        realistic_cost_model: TransactionCostModel,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test building a portfolio with multiple positions."""
        trades = [
            ("AAPL", "buy", 80_000, 150.00, 150.05),
            ("GOOGL", "buy", 70_000, 140.00, 140.10),
            ("MSFT", "buy", 60_000, 380.00, 380.20),
        ]

        total_fill_cost = 0.0

        for symbol, side, notional, bid, ask in trades:
            # Check risk
            violations = realistic_risk_checker.check_order(
                symbol=symbol,
                side=side,
                notional=notional,
                state=account_state,
            )

            if not violations:
                # Calculate fill
                book_notional = ask * 2000
                fill_price = realistic_cost_model.calculate_fill_price(
                    side=side,
                    ask_price=ask,
                    bid_price=bid,
                    order_notional=notional,
                    book_notional=book_notional,
                    short_term_vol=0.02,
                )

                # Calculate slippage cost
                slippage_cost = (fill_price - ask) / ask * notional
                total_fill_cost += slippage_cost

                # Update state
                account_state.update_position(symbol, notional)

        # Verify final state
        assert account_state.current_gross_exposure == 210_000
        assert account_state.current_net_exposure == 210_000
        assert total_fill_cost > 0, "Should have accumulated some slippage"

    def test_position_sizing_and_closing(
        self,
        realistic_cost_model: TransactionCostModel,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test building and closing a position."""
        symbol = "AAPL"

        # Open position
        account_state.update_position(symbol, 80_000)
        assert account_state.get_position(symbol) == 80_000

        # Check if we can close it (selling)
        violations = realistic_risk_checker.check_order(
            symbol=symbol,
            side="sell",
            notional=80_000,  # Close entire position
            state=account_state,
        )

        # Closing should be allowed
        # Note: Selling reduces position from 80k to 0, not creating new short
        assert len(violations) == 0, "Closing position should be allowed"

        # Close position
        account_state.update_position(symbol, -80_000)
        assert account_state.get_position(symbol) == 0
        assert symbol not in account_state.positions

    def test_hedged_position_scenario(
        self,
        realistic_cost_model: TransactionCostModel,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test building a hedged portfolio."""
        # Long equity
        account_state.update_position("SPY", 100_000)

        # Short hedge
        account_state.update_position("VXX", -50_000)

        # Verify exposures
        assert account_state.current_gross_exposure == 150_000
        assert account_state.current_net_exposure == 50_000  # Net long

        # Gross exposure check
        state_violations = realistic_risk_checker.check_state(account_state)
        gross_violations = [
            v for v in state_violations if v.type == RiskViolationType.MAX_GROSS_EXPOSURE
        ]
        assert len(gross_violations) == 0, "Gross exposure within limits"

    def test_high_volatility_increases_costs(
        self,
        realistic_cost_model: TransactionCostModel,
    ) -> None:
        """Test that high volatility increases fill costs."""
        low_vol_fill = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.95,
            order_notional=50_000,
            book_notional=500_000,
            short_term_vol=0.005,  # 0.5% volatility
        )

        high_vol_fill = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.95,
            order_notional=50_000,
            book_notional=500_000,
            short_term_vol=0.05,  # 5% volatility
        )

        assert high_vol_fill > low_vol_fill
        # High vol fill should be noticeably worse
        vol_impact = (high_vol_fill - low_vol_fill) / 100.00
        assert vol_impact > 0.0005  # At least 5 bps difference

    def test_large_order_impact(
        self,
        realistic_cost_model: TransactionCostModel,
    ) -> None:
        """Test that large orders have more market impact."""
        small_order_fill = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.95,
            order_notional=10_000,  # Small order
            book_notional=100_000,  # 10% of book
            short_term_vol=0.02,
        )

        large_order_fill = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.95,
            order_notional=80_000,  # Large order
            book_notional=100_000,  # 80% of book
            short_term_vol=0.02,
        )

        assert large_order_fill > small_order_fill
        # Large order should have significantly more impact
        impact_diff = (large_order_fill - small_order_fill) / 100.00
        assert impact_diff > 0.002  # At least 20 bps difference


# ============================================================================
# Round-Trip Trade Tests
# ============================================================================


class TestRoundTripTrades:
    """Test complete round-trip trade scenarios."""

    def test_profitable_round_trip_requires_edge(
        self,
        realistic_cost_model: TransactionCostModel,
    ) -> None:
        """Test that round-trip requires sufficient edge to be profitable."""
        # Calculate round-trip cost
        spread_bps = 5.0  # 5 bps spread
        round_trip_cost = realistic_cost_model.calculate_round_trip_cost_bps(
            spread_bps=spread_bps,
            order_notional=50_000,
            book_notional=500_000,
            short_term_vol=0.02,
        )

        # To be profitable, move must exceed round-trip cost
        assert round_trip_cost > 0
        # Typical equity trade should cost 10-50 bps round-trip
        assert round_trip_cost > 5, "Round-trip should cost more than just spread"
        assert round_trip_cost < 100, "Round-trip shouldn't exceed 1%"

    def test_round_trip_pnl_calculation(
        self,
        realistic_cost_model: TransactionCostModel,
        account_state: RiskState,
    ) -> None:
        """Test P&L calculation for a round-trip trade."""
        # Entry
        entry_notional = 50_000
        entry_ask = 100.05
        entry_bid = 100.00
        entry_book = 500_000

        entry_fill = realistic_cost_model.calculate_fill_price(
            side="buy",
            ask_price=entry_ask,
            bid_price=entry_bid,
            order_notional=entry_notional,
            book_notional=entry_book,
            short_term_vol=0.02,
        )

        # Assume price moved up 50 bps
        price_move = 0.005  # 50 bps
        exit_bid = entry_bid * (1 + price_move)
        exit_ask = entry_ask * (1 + price_move)

        exit_fill = realistic_cost_model.calculate_fill_price(
            side="sell",
            ask_price=exit_ask,
            bid_price=exit_bid,
            order_notional=entry_notional,
            book_notional=entry_book,
            short_term_vol=0.02,
        )

        # Calculate P&L
        shares = entry_notional / entry_fill
        gross_pnl = (exit_fill - entry_fill) * shares

        # Verify P&L is reasonable given move and costs
        gross_pnl_bps = gross_pnl / entry_notional * 10000
        # Move was 50 bps, minus round-trip costs
        assert gross_pnl_bps < 50, "Shouldn't capture entire move"
        assert gross_pnl_bps > 0, "Should be profitable with 50 bps move"


# ============================================================================
# Risk Limit Progression Tests
# ============================================================================


class TestRiskLimitProgression:
    """Test behavior as positions approach limits."""

    def test_approaching_position_limit(
        self,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test orders as position approaches limit."""
        symbol = "AAPL"

        # First order: 80k (within 100k limit)
        violations_1 = realistic_risk_checker.check_order(
            symbol=symbol,
            side="buy",
            notional=80_000,
            state=account_state,
        )
        assert len(violations_1) == 0
        account_state.update_position(symbol, 80_000)

        # Second order: 25k (would exceed 100k limit: 80k + 25k = 105k)
        violations_2 = realistic_risk_checker.check_order(
            symbol=symbol,
            side="buy",
            notional=25_000,
            state=account_state,
        )
        assert any(v.type == RiskViolationType.MAX_POSITION for v in violations_2)

        # Third order: 20k (exactly at limit)
        violations_3 = realistic_risk_checker.check_order(
            symbol=symbol,
            side="buy",
            notional=20_000,
            state=account_state,
        )
        position_violations = [
            v for v in violations_3 if v.type == RiskViolationType.MAX_POSITION
        ]
        assert len(position_violations) == 0, "Order to exactly 100k should pass"

    def test_approaching_gross_exposure_limit(
        self,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test orders as gross exposure approaches limit."""
        # Build up exposure across multiple symbols
        account_state.update_position("AAPL", 90_000)
        account_state.update_position("GOOGL", 90_000)
        account_state.update_position("MSFT", 90_000)
        account_state.update_position("AMZN", 90_000)
        # Total: 360k gross, limit is 500k

        # Order that stays within limit
        violations_ok = realistic_risk_checker.check_order(
            symbol="META",
            side="buy",
            notional=90_000,  # Would be 450k, still under 500k
            state=account_state,
        )
        gross_violations_ok = [
            v for v in violations_ok if v.type == RiskViolationType.MAX_GROSS_EXPOSURE
        ]
        assert len(gross_violations_ok) == 0

        # Order that exceeds limit
        violations_exceed = realistic_risk_checker.check_order(
            symbol="META",
            side="buy",
            notional=150_000,  # Would be 510k, over 500k limit
            state=account_state,
        )
        assert any(v.type == RiskViolationType.MAX_GROSS_EXPOSURE for v in violations_exceed)


# ============================================================================
# Kill Switch Integration Tests
# ============================================================================


class TestKillSwitchIntegration:
    """Test kill switch in realistic scenarios."""

    def test_kill_switch_after_losing_streak(
        self,
        realistic_risk_checker: RiskChecker,
    ) -> None:
        """Test kill switch triggers after losing streak."""
        # Simulate losing streak
        state = RiskState(
            positions={"AAPL": 50_000},
            daily_pnl=-25_000,  # Lost 2.5%, exceeds 2% daily limit
            peak_equity=1_000_000,
            current_equity=975_000,
        )

        # Kill switch should trigger
        assert realistic_risk_checker.is_kill_switch_triggered(state) is True

        # No new orders should be allowed (conceptually)
        violations = realistic_risk_checker.check_order(
            symbol="GOOGL",
            side="buy",
            notional=50_000,
            state=state,
        )
        # Order might pass risk check, but trading should be halted
        # This is a business logic decision, not enforced by check_order
        assert realistic_risk_checker.is_kill_switch_triggered(state) is True

    def test_kill_switch_from_drawdown(
        self,
        realistic_risk_checker: RiskChecker,
    ) -> None:
        """Test kill switch from accumulated drawdown."""
        state = RiskState(
            positions={"AAPL": 50_000},
            daily_pnl=-5_000,  # Within daily limit
            peak_equity=1_000_000,
            current_equity=850_000,  # 15% drawdown > 10% limit
        )

        assert realistic_risk_checker.is_kill_switch_triggered(state) is True


# ============================================================================
# Stress Testing
# ============================================================================


class TestStressScenarios:
    """Test behavior under stress conditions."""

    def test_flash_crash_scenario(
        self,
        realistic_cost_model: TransactionCostModel,
    ) -> None:
        """Test fill calculation during flash crash (wide spread, high vol)."""
        fill_price = realistic_cost_model.calculate_fill_price(
            side="sell",  # Panic selling
            ask_price=100.00,
            bid_price=95.00,  # 5% spread (extremely wide)
            order_notional=100_000,
            book_notional=50_000,  # Thin liquidity
            short_term_vol=0.20,  # 20% short-term volatility
        )

        # Fill should be significantly worse than bid
        assert fill_price < 95.00
        # But should still be calculable
        assert fill_price > 0

    def test_many_small_orders(
        self,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test many small orders don't cause issues."""
        for i in range(100):
            symbol = f"SYM{i}"
            violations = realistic_risk_checker.check_order(
                symbol=symbol,
                side="buy",
                notional=1_000,  # Small orders
                state=account_state,
            )

            if not violations:
                account_state.update_position(symbol, 1_000)

        # Should have many small positions
        assert len(account_state.positions) == 100
        assert account_state.current_gross_exposure == 100_000

    def test_rapid_position_changes(
        self,
        realistic_risk_checker: RiskChecker,
        account_state: RiskState,
    ) -> None:
        """Test rapid position changes (HFT-like)."""
        symbol = "AAPL"

        for _ in range(100):
            # Buy
            violations = realistic_risk_checker.check_order(
                symbol=symbol,
                side="buy",
                notional=10_000,
                state=account_state,
            )
            if not violations:
                account_state.update_position(symbol, 10_000)

            # Sell back
            violations = realistic_risk_checker.check_order(
                symbol=symbol,
                side="sell",
                notional=10_000,
                state=account_state,
            )
            if not violations:
                account_state.update_position(symbol, -10_000)

        # Should end flat
        assert account_state.get_position(symbol) == 0
