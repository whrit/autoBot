"""
Comprehensive tests for the cost_models library.

Tests cover:
- SlippageModel: slippage calculation formula, coefficients, edge cases
- TransactionCostModel: fill price calculation, round-trip costs
- Mathematical correctness and numerical stability
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest
from cost_models import SlippageModel, TransactionCostModel

from backtester_py.models.cost_model import (
    CostModel as BacktesterCostModel,
)
from backtester_py.models.cost_model import (
    CostModelType,
    FillResult,
    FixedCostModel,
    TieredCostModel,
    VolumeImpactCostModel,
    ZeroCostModel,
    create_cost_model,
)

# ============================================================================
# SlippageModel Tests
# ============================================================================


class TestSlippageModelCreation:
    """Test SlippageModel dataclass creation and default values."""

    def test_default_coefficients(self) -> None:
        """Slippage model should have default coefficient values of 1.0."""
        model = SlippageModel()
        assert model.spread_coef == 1.0
        assert model.size_coef == 1.0
        assert model.vol_coef == 1.0
        assert model.max_size_impact_bps == 50.0

    def test_custom_coefficients(self) -> None:
        """Slippage model should accept custom coefficient values."""
        model = SlippageModel(
            spread_coef=1.5,
            size_coef=0.8,
            vol_coef=2.0,
            max_size_impact_bps=100.0,
        )
        assert model.spread_coef == 1.5
        assert model.size_coef == 0.8
        assert model.vol_coef == 2.0
        assert model.max_size_impact_bps == 100.0

    def test_zero_coefficients_allowed(self) -> None:
        """Zero coefficients should be allowed for disabled components."""
        model = SlippageModel(
            spread_coef=0.0,
            size_coef=0.0,
            vol_coef=0.0,
        )
        assert model.spread_coef == 0.0
        assert model.size_coef == 0.0
        assert model.vol_coef == 0.0

    def test_negative_coefficients_allowed(self) -> None:
        """Negative coefficients should be allowed (edge case)."""
        model = SlippageModel(spread_coef=-1.0)
        assert model.spread_coef == -1.0


class TestSlippageCalculation:
    """Test slippage calculation formula correctness."""

    def test_spread_component_only(self) -> None:
        """Spread component should be proportional to spread_bps * spread_coef."""
        model = SlippageModel(spread_coef=1.0, size_coef=0.0, vol_coef=0.0)
        spread_bps = 5.0  # 5 basis points spread

        slippage = model.calculate_slippage_bps(
            spread_bps=spread_bps,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        assert slippage == pytest.approx(5.0, rel=1e-9)

    def test_spread_component_with_coefficient(self) -> None:
        """Spread component should scale with coefficient."""
        model = SlippageModel(spread_coef=1.5, size_coef=0.0, vol_coef=0.0)
        spread_bps = 5.0

        slippage = model.calculate_slippage_bps(
            spread_bps=spread_bps,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Expected: 1.5 * 5.0 = 7.5
        assert slippage == pytest.approx(7.5, rel=1e-9)

    def test_size_component_formula(self) -> None:
        """Size component should be size_coef * (order/book) * 100."""
        model = SlippageModel(spread_coef=0.0, size_coef=1.0, vol_coef=0.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=10_000,  # 10% of book
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Expected: 1.0 * (10000/100000) * 100 = 10.0 bps
        assert slippage == pytest.approx(10.0, rel=1e-9)

    def test_size_component_with_coefficient(self) -> None:
        """Size component should scale with coefficient."""
        model = SlippageModel(spread_coef=0.0, size_coef=0.5, vol_coef=0.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=20_000,  # 20% of book
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Expected: 0.5 * (20000/100000) * 100 = 10.0 bps
        assert slippage == pytest.approx(10.0, rel=1e-9)

    def test_size_component_max_cap(self) -> None:
        """Size impact should be capped at max_size_impact_bps."""
        model = SlippageModel(
            spread_coef=0.0,
            size_coef=1.0,
            vol_coef=0.0,
            max_size_impact_bps=50.0,
        )

        # Order 10x book - should cap at 50 bps
        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=1_000_000,  # 1000% of book
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Expected: capped at 50.0 (not 1000.0)
        assert slippage == pytest.approx(50.0, rel=1e-9)

    def test_vol_component_formula(self) -> None:
        """Volatility component should be vol_coef * vol * 100."""
        model = SlippageModel(spread_coef=0.0, size_coef=0.0, vol_coef=1.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,  # 1% volatility
        )

        # Expected: 1.0 * 0.01 * 100 = 1.0 bps
        assert slippage == pytest.approx(1.0, rel=1e-9)

    def test_vol_component_with_coefficient(self) -> None:
        """Volatility component should scale with coefficient."""
        model = SlippageModel(spread_coef=0.0, size_coef=0.0, vol_coef=2.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.02,  # 2% volatility
        )

        # Expected: 2.0 * 0.02 * 100 = 4.0 bps
        assert slippage == pytest.approx(4.0, rel=1e-9)

    def test_combined_components(self) -> None:
        """All components should add together correctly."""
        model = SlippageModel(
            spread_coef=1.0,
            size_coef=1.0,
            vol_coef=1.0,
            max_size_impact_bps=100.0,
        )

        slippage = model.calculate_slippage_bps(
            spread_bps=5.0,  # 5 bps spread contribution
            order_notional=10_000,  # 10% of book = 10 bps size contribution
            book_notional=100_000,
            short_term_vol=0.02,  # 2% vol = 2 bps vol contribution
        )

        # Expected: 5.0 + 10.0 + 2.0 = 17.0 bps
        assert slippage == pytest.approx(17.0, rel=1e-9)

    def test_zero_book_notional_uses_max_impact(self) -> None:
        """Zero book notional should use max_size_impact_bps."""
        model = SlippageModel(
            spread_coef=0.0,
            size_coef=1.0,
            vol_coef=0.0,
            max_size_impact_bps=50.0,
        )

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=10_000,
            book_notional=0.0,  # Empty book
            short_term_vol=0.0,
        )

        assert slippage == pytest.approx(50.0, rel=1e-9)

    def test_negative_book_notional_uses_max_impact(self) -> None:
        """Negative book notional should use max_size_impact_bps."""
        model = SlippageModel(
            spread_coef=0.0,
            size_coef=1.0,
            vol_coef=0.0,
            max_size_impact_bps=50.0,
        )

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=10_000,
            book_notional=-100.0,  # Invalid negative book
            short_term_vol=0.0,
        )

        assert slippage == pytest.approx(50.0, rel=1e-9)


class TestSlippageModelEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_all_inputs(self) -> None:
        """Zero inputs should return zero slippage."""
        model = SlippageModel()

        slippage = model.calculate_slippage_bps(
            spread_bps=0.0,
            order_notional=0.0,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        assert slippage == pytest.approx(0.0, abs=1e-9)

    def test_very_small_values(self) -> None:
        """Very small values should compute without numerical issues."""
        model = SlippageModel()

        slippage = model.calculate_slippage_bps(
            spread_bps=0.001,  # 0.001 bps
            order_notional=1.0,
            book_notional=1_000_000,
            short_term_vol=0.0001,
        )

        # Should be a small positive number
        assert slippage > 0
        assert slippage < 1.0
        assert math.isfinite(slippage)

    def test_very_large_values(self) -> None:
        """Very large values should compute without overflow."""
        model = SlippageModel(max_size_impact_bps=1000.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=100.0,  # Wide spread
            order_notional=1e9,  # 1 billion
            book_notional=1e6,  # 1 million
            short_term_vol=0.5,  # 50% volatility
        )

        # Should be finite and capped
        assert math.isfinite(slippage)
        # Size component should be capped at 1000
        assert slippage <= 100.0 + 1000.0 + 50.0

    def test_negative_spread_bps(self) -> None:
        """Negative spread (crossed market) should reduce slippage."""
        model = SlippageModel(spread_coef=1.0, size_coef=0.0, vol_coef=0.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=-5.0,  # Crossed market
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Negative spread contribution
        assert slippage == pytest.approx(-5.0, rel=1e-9)


# ============================================================================
# TransactionCostModel Tests
# ============================================================================


class TestTransactionCostModelCreation:
    """Test TransactionCostModel dataclass creation."""

    def test_creation_with_defaults(self, default_slippage_model: SlippageModel) -> None:
        """Transaction cost model should accept default fixed_cost_bps."""
        model = TransactionCostModel(slippage_model=default_slippage_model)
        assert model.fixed_cost_bps == 0.0

    def test_creation_with_fixed_costs(self, default_slippage_model: SlippageModel) -> None:
        """Transaction cost model should accept custom fixed_cost_bps."""
        model = TransactionCostModel(
            slippage_model=default_slippage_model,
            fixed_cost_bps=0.35,
        )
        assert model.fixed_cost_bps == 0.35


class TestBuyFillCalculation:
    """Test fill price calculation for buy orders."""

    def test_buy_fill_basic(self, default_cost_model: TransactionCostModel) -> None:
        """Buy fill price should be ask + costs."""
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Fill should be higher than ask price
        assert fill_price > 100.02

    def test_buy_fill_zero_slippage(self, zero_cost_model: TransactionCostModel) -> None:
        """Buy fill with zero costs should equal ask price."""
        fill_price = zero_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        assert fill_price == pytest.approx(100.02, rel=1e-9)

    def test_buy_fill_formula(self) -> None:
        """Buy fill should follow: ask * (1 + total_cost_bps / 10000)."""
        slippage_model = SlippageModel(spread_coef=0.0, size_coef=0.0, vol_coef=0.0)
        model = TransactionCostModel(slippage_model=slippage_model, fixed_cost_bps=10.0)

        fill_price = model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.98,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Expected: 100.00 * (1 + 10.0/10000) = 100.00 * 1.001 = 100.10
        assert fill_price == pytest.approx(100.10, rel=1e-9)


class TestSellFillCalculation:
    """Test fill price calculation for sell orders."""

    def test_sell_fill_basic(self, default_cost_model: TransactionCostModel) -> None:
        """Sell fill price should be bid - costs."""
        fill_price = default_cost_model.calculate_fill_price(
            side="sell",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Fill should be lower than bid price
        assert fill_price < 100.00

    def test_sell_fill_zero_slippage(self, zero_cost_model: TransactionCostModel) -> None:
        """Sell fill with zero costs should equal bid price."""
        fill_price = zero_cost_model.calculate_fill_price(
            side="sell",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        assert fill_price == pytest.approx(100.00, rel=1e-9)

    def test_sell_fill_formula(self) -> None:
        """Sell fill should follow: bid * (1 - total_cost_bps / 10000)."""
        slippage_model = SlippageModel(spread_coef=0.0, size_coef=0.0, vol_coef=0.0)
        model = TransactionCostModel(slippage_model=slippage_model, fixed_cost_bps=10.0)

        fill_price = model.calculate_fill_price(
            side="sell",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Expected: 100.00 * (1 - 10.0/10000) = 100.00 * 0.999 = 99.90
        assert fill_price == pytest.approx(99.90, rel=1e-9)


class TestRoundTripCost:
    """Test round-trip cost calculation."""

    def test_round_trip_is_double_one_way(self) -> None:
        """Round-trip cost should be 2x one-way cost."""
        slippage_model = SlippageModel(spread_coef=1.0, size_coef=0.0, vol_coef=0.0)
        model = TransactionCostModel(slippage_model=slippage_model, fixed_cost_bps=5.0)

        round_trip = model.calculate_round_trip_cost_bps(
            spread_bps=10.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # One way = 10.0 (spread) + 5.0 (fixed) = 15.0
        # Round trip = 2 * 15.0 = 30.0
        assert round_trip == pytest.approx(30.0, rel=1e-9)

    def test_round_trip_zero_cost(self, zero_cost_model: TransactionCostModel) -> None:
        """Round-trip with zero costs should be zero."""
        round_trip = zero_cost_model.calculate_round_trip_cost_bps(
            spread_bps=0.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        assert round_trip == pytest.approx(0.0, abs=1e-9)


class TestCostModelEdgeCases:
    """Test edge cases for TransactionCostModel."""

    def test_zero_volume_order(self, default_cost_model: TransactionCostModel) -> None:
        """Zero volume order should still compute fill price."""
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.02,
            bid_price=100.00,
            order_notional=0.0,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Should be ask + some slippage (spread + vol components)
        assert fill_price >= 100.02
        assert math.isfinite(fill_price)

    def test_very_large_order(self, default_cost_model: TransactionCostModel) -> None:
        """Very large order relative to book should have max size impact."""
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.98,
            order_notional=10_000_000,  # 10M order
            book_notional=100_000,  # 100k book
            short_term_vol=0.01,
        )

        # Fill should be significantly higher than ask
        assert fill_price > 100.00 * 1.005  # At least 50 bps impact
        assert math.isfinite(fill_price)

    def test_extreme_volatility(self, default_cost_model: TransactionCostModel) -> None:
        """Extreme volatility should increase slippage significantly."""
        low_vol_fill = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.98,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.001,  # 0.1% vol
        )

        high_vol_fill = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=99.98,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.10,  # 10% vol
        )

        # High vol fill should be worse (higher price for buy)
        assert high_vol_fill > low_vol_fill

    def test_negative_prices_handled(self, default_cost_model: TransactionCostModel) -> None:
        """Model should compute even with negative prices (edge case)."""
        # This tests numerical stability, not real-world validity
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=-100.00,
            bid_price=-100.02,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Should compute without error
        assert math.isfinite(fill_price)

    def test_equal_bid_ask(self, default_cost_model: TransactionCostModel) -> None:
        """Equal bid/ask (zero spread) should still compute."""
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=100.00,
            bid_price=100.00,  # Zero spread
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        # Should be price + non-spread costs
        assert fill_price >= 100.00
        assert math.isfinite(fill_price)


# ============================================================================
# Property-Based Testing
# ============================================================================


class TestSlippageModelProperties:
    """Property-based tests for slippage model invariants."""

    @pytest.mark.parametrize(
        "spread_bps,order_notional,book_notional,vol",
        [
            (1.0, 10_000, 100_000, 0.01),
            (5.0, 50_000, 100_000, 0.02),
            (10.0, 100_000, 100_000, 0.05),
            (0.0, 10_000, 100_000, 0.0),
        ],
    )
    def test_non_negative_coefficients_give_non_negative_slippage(
        self,
        spread_bps: float,
        order_notional: float,
        book_notional: float,
        vol: float,
    ) -> None:
        """With non-negative coefficients, slippage should be non-negative."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)

        slippage = model.calculate_slippage_bps(
            spread_bps=spread_bps,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=vol,
        )

        assert slippage >= 0

    @pytest.mark.parametrize("coef", [0.5, 1.0, 1.5, 2.0])
    def test_slippage_scales_linearly_with_spread_coef(self, coef: float) -> None:
        """Slippage should scale linearly with spread_coef."""
        base_model = SlippageModel(spread_coef=1.0, size_coef=0.0, vol_coef=0.0)
        scaled_model = SlippageModel(spread_coef=coef, size_coef=0.0, vol_coef=0.0)

        base_slippage = base_model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        scaled_slippage = scaled_model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        assert scaled_slippage == pytest.approx(base_slippage * coef, rel=1e-9)


class TestTransactionCostModelProperties:
    """Property-based tests for transaction cost model invariants."""

    @pytest.mark.parametrize(
        "ask_price,bid_price",
        [
            (100.02, 100.00),
            (150.05, 150.00),
            (50.01, 50.00),
        ],
    )
    def test_buy_fill_always_at_or_above_ask(
        self,
        default_cost_model: TransactionCostModel,
        ask_price: float,
        bid_price: float,
    ) -> None:
        """Buy fill price should always be >= ask price (with non-negative costs)."""
        fill_price = default_cost_model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        assert fill_price >= ask_price

    @pytest.mark.parametrize(
        "ask_price,bid_price",
        [
            (100.02, 100.00),
            (150.05, 150.00),
            (50.01, 50.00),
        ],
    )
    def test_sell_fill_always_at_or_below_bid(
        self,
        default_cost_model: TransactionCostModel,
        ask_price: float,
        bid_price: float,
    ) -> None:
        """Sell fill price should always be <= bid price (with non-negative costs)."""
        fill_price = default_cost_model.calculate_fill_price(
            side="sell",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.01,
        )

        assert fill_price <= bid_price

    def test_round_trip_cost_symmetric(
        self, default_cost_model: TransactionCostModel
    ) -> None:
        """Round-trip cost should be same regardless of direction."""
        params = {
            "spread_bps": 5.0,
            "order_notional": 10_000,
            "book_notional": 100_000,
            "short_term_vol": 0.01,
        }

        round_trip = default_cost_model.calculate_round_trip_cost_bps(**params)

        # Cost should be positive
        assert round_trip > 0


# ============================================================================
# Mathematical Correctness Tests
# ============================================================================


class TestMathematicalCorrectness:
    """Tests verifying exact mathematical formulas."""

    def test_slippage_formula_exact(self) -> None:
        """
        Verify exact slippage formula:
        slippage_bps = a*spread + b*(order/book)*100 + c*vol*100
        """
        a, b, c = 1.5, 0.8, 2.0
        model = SlippageModel(spread_coef=a, size_coef=b, vol_coef=c)

        spread_bps = 5.0
        order_notional = 20_000
        book_notional = 100_000
        vol = 0.015

        expected = a * spread_bps + b * (order_notional / book_notional) * 100 + c * vol * 100
        # = 1.5 * 5.0 + 0.8 * 0.2 * 100 + 2.0 * 0.015 * 100
        # = 7.5 + 16.0 + 3.0 = 26.5

        actual = model.calculate_slippage_bps(
            spread_bps=spread_bps,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=vol,
        )

        assert actual == pytest.approx(expected, rel=1e-9)
        assert actual == pytest.approx(26.5, rel=1e-9)

    def test_fill_price_formula_exact(self) -> None:
        """
        Verify exact fill price formulas:
        Buy: ask * (1 + total_cost_bps / 10000)
        Sell: bid * (1 - total_cost_bps / 10000)
        """
        # Use zero slippage to isolate fixed cost
        slippage_model = SlippageModel(spread_coef=0.0, size_coef=0.0, vol_coef=0.0)
        fixed_cost_bps = 5.0
        model = TransactionCostModel(
            slippage_model=slippage_model, fixed_cost_bps=fixed_cost_bps
        )

        ask_price = 100.00
        bid_price = 99.98

        buy_fill = model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        sell_fill = model.calculate_fill_price(
            side="sell",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        expected_buy = ask_price * (1 + fixed_cost_bps / 10000)  # 100.00 * 1.0005 = 100.05
        expected_sell = bid_price * (1 - fixed_cost_bps / 10000)  # 99.98 * 0.9995 = 99.93001

        assert buy_fill == pytest.approx(expected_buy, rel=1e-9)
        assert sell_fill == pytest.approx(expected_sell, rel=1e-9)

    def test_spread_bps_calculation_in_model(self) -> None:
        """Verify spread is calculated correctly from prices."""
        # The model internally calculates spread_bps from prices
        slippage_model = SlippageModel(spread_coef=1.0, size_coef=0.0, vol_coef=0.0)
        model = TransactionCostModel(slippage_model=slippage_model, fixed_cost_bps=0.0)

        ask_price = 100.10
        bid_price = 100.00
        midprice = (ask_price + bid_price) / 2  # 100.05
        spread_bps = (ask_price - bid_price) / midprice * 10000  # ~9.995 bps

        fill_price = model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=10_000,
            book_notional=100_000,
            short_term_vol=0.0,
        )

        # Buy fill = ask * (1 + spread_bps / 10000)
        expected = ask_price * (1 + spread_bps / 10000)
        assert fill_price == pytest.approx(expected, rel=1e-6)


# ============================================================================
# Backtester Cost Model Wrapper Tests (Decimal API)
# ============================================================================


class TestFillResultDecimalAPI:
    """Tests for FillResult dataclass with Decimal API."""

    def test_fill_result_creation_decimal(self) -> None:
        """Test FillResult can be created with Decimal fields."""
        result = FillResult(
            fill_price=Decimal("100.50"),
            commission=Decimal("0.10"),
            slippage=Decimal("0.05"),
        )
        assert result.fill_price == Decimal("100.50")
        assert result.commission == Decimal("0.10")
        assert result.slippage == Decimal("0.05")

    def test_fill_result_total_cost(self) -> None:
        """Test that commission and slippage are separate from fill price."""
        result = FillResult(
            fill_price=Decimal("100.00"),
            commission=Decimal("0.25"),
            slippage=Decimal("0.10"),
        )
        total_cost = result.commission + result.slippage
        assert total_cost == Decimal("0.35")

    def test_fill_result_zero_costs(self) -> None:
        """Test FillResult with zero costs."""
        result = FillResult(
            fill_price=Decimal("50.00"),
            commission=Decimal("0"),
            slippage=Decimal("0"),
        )
        assert result.commission == Decimal("0")
        assert result.slippage == Decimal("0")


class TestBacktesterCostModelABC:
    """Tests for Backtester CostModel abstract base class."""

    def test_cost_model_is_abstract(self) -> None:
        """Test that CostModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BacktesterCostModel()  # type: ignore[abstract]


class TestZeroCostModelDecimalAPI:
    """Tests for ZeroCostModel with Decimal API."""

    @pytest.fixture
    def model(self) -> ZeroCostModel:
        """Create a ZeroCostModel instance."""
        return ZeroCostModel()

    def test_buy_order_no_cost(self, model: ZeroCostModel) -> None:
        """Test buy order returns price with no costs."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        assert result.fill_price == Decimal("100.00")
        assert result.commission == Decimal("0")
        assert result.slippage == Decimal("0")

    def test_sell_order_no_cost(self, model: ZeroCostModel) -> None:
        """Test sell order returns price with no costs."""
        result = model.calculate_fill(
            side="sell",
            price=Decimal("50.00"),
            quantity=Decimal("5"),
            volume=Decimal("500"),
            volatility=0.01,
        )
        assert result.fill_price == Decimal("50.00")
        assert result.commission == Decimal("0")
        assert result.slippage == Decimal("0")

    def test_zero_volume(self, model: ZeroCostModel) -> None:
        """Test with zero volume still works."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("0"),
            volatility=0.0,
        )
        assert result.fill_price == Decimal("100.00")


class TestFixedCostModelDecimalAPI:
    """Tests for FixedCostModel with Decimal API."""

    @pytest.fixture
    def model(self) -> FixedCostModel:
        """Create a FixedCostModel with 0.1% commission and 0.05% slippage."""
        return FixedCostModel(
            commission_rate=Decimal("0.001"),  # 0.1%
            slippage_rate=Decimal("0.0005"),  # 0.05%
        )

    def test_buy_order_increases_price(self, model: FixedCostModel) -> None:
        """Test buy order fill price increases due to slippage."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # Buy: price + slippage (pay more)
        expected_fill = Decimal("100.00") * (1 + Decimal("0.0005"))
        assert result.fill_price == expected_fill
        # Commission on notional (price * quantity)
        expected_commission = Decimal("100.00") * Decimal("10") * Decimal("0.001")
        assert result.commission == expected_commission
        expected_slippage = Decimal("100.00") * Decimal("0.0005")
        assert result.slippage == expected_slippage

    def test_sell_order_decreases_price(self, model: FixedCostModel) -> None:
        """Test sell order fill price decreases due to slippage."""
        result = model.calculate_fill(
            side="sell",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # Sell: price - slippage (receive less)
        expected_fill = Decimal("100.00") * (1 - Decimal("0.0005"))
        assert result.fill_price == expected_fill

    def test_zero_commission_rate(self) -> None:
        """Test with zero commission rate."""
        model = FixedCostModel(
            commission_rate=Decimal("0"),
            slippage_rate=Decimal("0.001"),
        )
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        assert result.commission == Decimal("0")

    def test_large_order(self, model: FixedCostModel) -> None:
        """Test with a large order."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("50000.00"),
            quantity=Decimal("1000"),
            volume=Decimal("100000"),
            volatility=0.05,
        )
        # Commission on 50,000,000 notional at 0.1%
        expected_commission = Decimal("50000.00") * Decimal("1000") * Decimal("0.001")
        assert result.commission == expected_commission

    def test_default_rates(self) -> None:
        """Test default commission and slippage rates."""
        model = FixedCostModel()
        # Check defaults are sensible
        assert model.commission_rate >= Decimal("0")
        assert model.slippage_rate >= Decimal("0")


class TestVolumeImpactCostModelDecimalAPI:
    """Tests for VolumeImpactCostModel with Decimal API."""

    @pytest.fixture
    def model(self) -> VolumeImpactCostModel:
        """Create a VolumeImpactCostModel."""
        return VolumeImpactCostModel(
            base_commission=Decimal("0.001"),  # 0.1%
            impact_coefficient=Decimal("0.1"),  # 10% of participation rate
            volatility_coefficient=Decimal("0.5"),  # Vol impact multiplier
        )

    def test_buy_order_with_market_impact(self, model: VolumeImpactCostModel) -> None:
        """Test buy order includes volume-based market impact."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),  # 10% of volume
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # Buy: price increases by slippage
        assert result.fill_price > Decimal("100.00")
        assert result.slippage > Decimal("0")

    def test_sell_order_with_market_impact(self, model: VolumeImpactCostModel) -> None:
        """Test sell order includes volume-based market impact."""
        result = model.calculate_fill(
            side="sell",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # Sell: price decreases by slippage
        assert result.fill_price < Decimal("100.00")

    def test_small_order_minimal_impact(self, model: VolumeImpactCostModel) -> None:
        """Test small order (low participation) has minimal impact."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("1"),  # 0.1% of volume
            volume=Decimal("1000"),
            volatility=0.01,
        )
        # Very small impact for tiny order
        assert result.slippage < Decimal("1.00")  # Less than $1.00 slippage

    def test_large_order_high_impact(self, model: VolumeImpactCostModel) -> None:
        """Test large order (high participation) has significant impact."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("500"),  # 50% of volume
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # High participation should have higher impact
        assert result.slippage > Decimal("1.00")

    def test_zero_volume_max_impact(self, model: VolumeImpactCostModel) -> None:
        """Test zero volume applies maximum impact cap."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("0"),  # Zero volume
            volatility=0.02,
        )
        # Should handle zero volume gracefully (max impact cap)
        assert result.fill_price > Decimal("100.00")
        assert result.slippage <= Decimal("100.00") * Decimal("0.05")  # Max 5% cap

    def test_high_volatility_increases_impact(
        self, model: VolumeImpactCostModel
    ) -> None:
        """Test high volatility increases slippage."""
        low_vol_result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            volume=Decimal("1000"),
            volatility=0.01,  # 1% vol
        )
        high_vol_result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            volume=Decimal("1000"),
            volatility=0.10,  # 10% vol
        )
        assert high_vol_result.slippage > low_vol_result.slippage

    def test_extreme_volatility_capped(self, model: VolumeImpactCostModel) -> None:
        """Test extreme volatility is capped."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            volume=Decimal("1000"),
            volatility=1.0,  # 100% vol (extreme)
        )
        # Impact should be capped at reasonable maximum
        max_impact = Decimal("100.00") * Decimal("0.05")  # 5% cap
        assert result.slippage <= max_impact


class TestTieredCostModelDecimalAPI:
    """Tests for TieredCostModel with Decimal API."""

    @pytest.fixture
    def model(self) -> TieredCostModel:
        """Create a TieredCostModel with exchange-like tiers."""
        tiers = [
            # (volume_threshold, commission_rate)
            (Decimal("0"), Decimal("0.001")),  # 0.1% for volume < 10k
            (Decimal("10000"), Decimal("0.0008")),  # 0.08% for 10k-100k
            (Decimal("100000"), Decimal("0.0005")),  # 0.05% for 100k-1M
            (Decimal("1000000"), Decimal("0.0002")),  # 0.02% for >= 1M
        ]
        return TieredCostModel(
            tiers=tiers,
            slippage_rate=Decimal("0.0003"),  # Fixed 0.03% slippage
        )

    def test_lowest_tier_highest_fee(self, model: TieredCostModel) -> None:
        """Test lowest volume tier has highest fee."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),  # 1,000 notional
            volume=Decimal("1000"),
            volatility=0.02,
        )
        # Notional = 1,000 (lowest tier: 0.1%)
        expected_commission = Decimal("1000") * Decimal("0.001")
        assert result.commission == expected_commission

    def test_middle_tier(self, model: TieredCostModel) -> None:
        """Test middle volume tier."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("500"),  # 50,000 notional
            volume=Decimal("10000"),
            volatility=0.02,
        )
        # Notional = 50,000 (tier 2: 0.08%)
        expected_commission = Decimal("50000") * Decimal("0.0008")
        assert result.commission == expected_commission

    def test_highest_tier_lowest_fee(self, model: TieredCostModel) -> None:
        """Test highest volume tier has lowest fee."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("1000.00"),
            quantity=Decimal("2000"),  # 2,000,000 notional
            volume=Decimal("100000"),
            volatility=0.02,
        )
        # Notional = 2,000,000 (highest tier: 0.02%)
        expected_commission = Decimal("2000000") * Decimal("0.0002")
        assert result.commission == expected_commission

    def test_tier_boundary_exact(self, model: TieredCostModel) -> None:
        """Test exact tier boundary uses lower rate."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),  # Exactly 10,000 notional
            volume=Decimal("10000"),
            volatility=0.02,
        )
        # Notional = 10,000 (exactly at tier 2 boundary: 0.08%)
        expected_commission = Decimal("10000") * Decimal("0.0008")
        assert result.commission == expected_commission

    def test_slippage_applied(self, model: TieredCostModel) -> None:
        """Test slippage is applied regardless of tier."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        expected_slippage = Decimal("100.00") * Decimal("0.0003")
        assert result.slippage == expected_slippage

    def test_buy_increases_price(self, model: TieredCostModel) -> None:
        """Test buy order fill price increases due to slippage."""
        result = model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        expected_fill = Decimal("100.00") * (1 + Decimal("0.0003"))
        assert result.fill_price == expected_fill

    def test_sell_decreases_price(self, model: TieredCostModel) -> None:
        """Test sell order fill price decreases due to slippage."""
        result = model.calculate_fill(
            side="sell",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.02,
        )
        expected_fill = Decimal("100.00") * (1 - Decimal("0.0003"))
        assert result.fill_price == expected_fill

    def test_empty_tiers_raises(self) -> None:
        """Test empty tiers raises ValueError."""
        with pytest.raises(ValueError, match="At least one tier"):
            TieredCostModel(tiers=[], slippage_rate=Decimal("0.0003"))


class TestDecimalAPIEdgeCases:
    """Tests for edge cases across all models with Decimal API."""

    @pytest.fixture
    def fixed_model(self) -> FixedCostModel:
        """Create a FixedCostModel."""
        return FixedCostModel(
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )

    @pytest.fixture
    def volume_model(self) -> VolumeImpactCostModel:
        """Create a VolumeImpactCostModel."""
        return VolumeImpactCostModel(
            base_commission=Decimal("0.001"),
            impact_coefficient=Decimal("0.1"),
            volatility_coefficient=Decimal("0.5"),
        )

    def test_negative_price_raises(self, fixed_model: FixedCostModel) -> None:
        """Test negative price raises ValueError."""
        with pytest.raises(ValueError, match="[Pp]rice"):
            fixed_model.calculate_fill(
                side="buy",
                price=Decimal("-100.00"),
                quantity=Decimal("10"),
                volume=Decimal("1000"),
                volatility=0.02,
            )

    def test_zero_price_raises(self, fixed_model: FixedCostModel) -> None:
        """Test zero price raises ValueError."""
        with pytest.raises(ValueError, match="[Pp]rice"):
            fixed_model.calculate_fill(
                side="buy",
                price=Decimal("0"),
                quantity=Decimal("10"),
                volume=Decimal("1000"),
                volatility=0.02,
            )

    def test_negative_quantity_raises(self, fixed_model: FixedCostModel) -> None:
        """Test negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="[Qq]uantity"):
            fixed_model.calculate_fill(
                side="buy",
                price=Decimal("100.00"),
                quantity=Decimal("-10"),
                volume=Decimal("1000"),
                volatility=0.02,
            )

    def test_zero_quantity_raises(self, fixed_model: FixedCostModel) -> None:
        """Test zero quantity raises ValueError."""
        with pytest.raises(ValueError, match="[Qq]uantity"):
            fixed_model.calculate_fill(
                side="buy",
                price=Decimal("100.00"),
                quantity=Decimal("0"),
                volume=Decimal("1000"),
                volatility=0.02,
            )

    def test_negative_volatility_raises(self, volume_model: VolumeImpactCostModel) -> None:
        """Test negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="[Vv]olatility"):
            volume_model.calculate_fill(
                side="buy",
                price=Decimal("100.00"),
                quantity=Decimal("10"),
                volume=Decimal("1000"),
                volatility=-0.02,
            )

    def test_negative_volume_raises(self, fixed_model: FixedCostModel) -> None:
        """Test negative volume raises ValueError."""
        with pytest.raises(ValueError, match="[Vv]olume"):
            fixed_model.calculate_fill(
                side="buy",
                price=Decimal("100.00"),
                quantity=Decimal("10"),
                volume=Decimal("-1000"),
                volatility=0.02,
            )

    def test_invalid_side_raises(self, fixed_model: FixedCostModel) -> None:
        """Test invalid side raises ValueError."""
        with pytest.raises(ValueError, match="[Ss]ide"):
            fixed_model.calculate_fill(
                side="invalid",  # type: ignore[arg-type]
                price=Decimal("100.00"),
                quantity=Decimal("10"),
                volume=Decimal("1000"),
                volatility=0.02,
            )

    def test_very_small_price(self, fixed_model: FixedCostModel) -> None:
        """Test with very small price (penny stock)."""
        result = fixed_model.calculate_fill(
            side="buy",
            price=Decimal("0.0001"),
            quantity=Decimal("1000000"),
            volume=Decimal("10000000"),
            volatility=0.05,
        )
        assert result.fill_price > Decimal("0")
        assert result.commission > Decimal("0")

    def test_very_large_price(self, fixed_model: FixedCostModel) -> None:
        """Test with very large price (high-value asset)."""
        result = fixed_model.calculate_fill(
            side="buy",
            price=Decimal("100000.00"),
            quantity=Decimal("1"),
            volume=Decimal("100"),
            volatility=0.01,
        )
        assert result.fill_price > Decimal("100000.00")

    def test_fractional_quantity(self, fixed_model: FixedCostModel) -> None:
        """Test with fractional quantity (crypto-like)."""
        result = fixed_model.calculate_fill(
            side="buy",
            price=Decimal("50000.00"),
            quantity=Decimal("0.00001"),  # Very small BTC amount
            volume=Decimal("100"),
            volatility=0.02,
        )
        assert result.fill_price > Decimal("50000.00")
        assert result.commission > Decimal("0")

    def test_zero_volatility(self, volume_model: VolumeImpactCostModel) -> None:
        """Test zero volatility is valid."""
        result = volume_model.calculate_fill(
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("10"),
            volume=Decimal("1000"),
            volatility=0.0,
        )
        assert result.fill_price >= Decimal("100.00")


class TestCreateCostModelFactory:
    """Tests for create_cost_model factory function."""

    def test_create_zero_model(self) -> None:
        """Test creating a ZeroCostModel."""
        model = create_cost_model(CostModelType.ZERO)
        assert isinstance(model, ZeroCostModel)

    def test_create_fixed_model(self) -> None:
        """Test creating a FixedCostModel."""
        model = create_cost_model(CostModelType.FIXED, fee_rate=0.002, slippage_rate=0.001)
        assert isinstance(model, FixedCostModel)

    def test_create_volume_model(self) -> None:
        """Test creating a VolumeImpactCostModel."""
        model = create_cost_model(CostModelType.VOLUME)
        assert isinstance(model, VolumeImpactCostModel)

    def test_create_tiered_model(self) -> None:
        """Test creating a TieredCostModel."""
        model = create_cost_model(CostModelType.TIERED)
        assert isinstance(model, TieredCostModel)

    def test_create_with_string_type(self) -> None:
        """Test creating models with string type."""
        model = create_cost_model("fixed")
        assert isinstance(model, FixedCostModel)

    def test_create_tiered_with_custom_tiers(self) -> None:
        """Test creating TieredCostModel with custom tiers."""
        tiers = [
            (Decimal("0"), Decimal("0.002")),
            (Decimal("50000"), Decimal("0.001")),
        ]
        model = create_cost_model(CostModelType.TIERED, tiers=tiers)
        assert isinstance(model, TieredCostModel)
