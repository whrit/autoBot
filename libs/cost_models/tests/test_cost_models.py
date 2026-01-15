"""
TDD Tests for Cost Models Library.

Following London School TDD - testing behavior and interactions.
Tests written FIRST before implementation.
"""

import json
from pathlib import Path

import pytest

from cost_models import SlippageModel, TransactionCostModel, load_cost_model


class TestSlippageModel:
    """Tests for SlippageModel - slippage calculation based on market conditions."""

    def test_slippage_calculation_basic(self) -> None:
        """Test: slippage_bps = a*spread + b*(order/book)*100 + c*vol*100."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.02,
        )
        # Expected: 1.0*10 + 1.0*(10000/100000)*100 + 1.0*0.02*100
        expected = 10.0 + 10.0 + 2.0
        assert abs(slippage - expected) < 0.01

    def test_slippage_zero_book_size(self) -> None:
        """Handle edge case of zero book size - should use max slippage."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=0,  # Edge case
            short_term_vol=0.02,
        )
        # Should use max slippage for size impact (default 50 bps)
        # spread_impact + max_size_impact + vol_impact = 10 + 50 + 2 = 62
        assert slippage > 10.0  # At least spread component
        assert abs(slippage - 62.0) < 0.01

    def test_slippage_negative_book_size(self) -> None:
        """Handle edge case of negative book size - should use max slippage."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=-100,  # Invalid edge case
            short_term_vol=0.02,
        )
        # Should use max slippage for size impact
        assert slippage > 10.0

    def test_slippage_coefficients(self) -> None:
        """Test different coefficient values scale correctly."""
        model = SlippageModel(spread_coef=2.0, size_coef=0.5, vol_coef=3.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=5.0,
            order_notional=5000,
            book_notional=50000,
            short_term_vol=0.01,
        )
        # 2.0*5 + 0.5*(5000/50000)*100 + 3.0*0.01*100
        # = 10.0 + 0.5*0.1*100 + 3.0
        # = 10.0 + 5.0 + 3.0 = 18.0
        expected = 10.0 + 5.0 + 3.0
        assert abs(slippage - expected) < 0.01

    def test_slippage_size_impact_capped(self) -> None:
        """Test that size impact is capped at max_size_impact_bps."""
        model = SlippageModel(
            spread_coef=1.0,
            size_coef=1.0,
            vol_coef=1.0,
            max_size_impact_bps=25.0,
        )
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=100000,  # Large order
            book_notional=10000,  # Small book = 1000% ratio
            short_term_vol=0.01,
        )
        # Size impact would be 1.0 * (100000/10000) * 100 = 1000 bps
        # But capped at 25 bps
        # Total: 10 + 25 + 1 = 36
        expected = 10.0 + 25.0 + 1.0
        assert abs(slippage - expected) < 0.01

    def test_slippage_zero_volatility(self) -> None:
        """Test slippage with zero volatility."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.0,
        )
        # No volatility component
        expected = 10.0 + 10.0 + 0.0
        assert abs(slippage - expected) < 0.01

    def test_slippage_high_volatility(self) -> None:
        """Test slippage with high volatility."""
        model = SlippageModel(spread_coef=1.0, size_coef=1.0, vol_coef=1.0)
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.10,  # 10% volatility
        )
        # vol_impact = 1.0 * 0.10 * 100 = 10 bps
        expected = 10.0 + 10.0 + 10.0
        assert abs(slippage - expected) < 0.01

    def test_slippage_default_coefficients(self) -> None:
        """Test that default coefficients are applied."""
        model = SlippageModel()  # Use defaults
        slippage = model.calculate_slippage_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.02,
        )
        # With default coefficients of 1.0
        expected = 10.0 + 10.0 + 2.0
        assert abs(slippage - expected) < 0.01


class TestTransactionCostModel:
    """Tests for TransactionCostModel - full transaction cost calculation."""

    def test_total_cost_buy(self) -> None:
        """Buy order: pay ask + slippage + fixed costs."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=1.0,
        )
        fill_price = model.calculate_fill_price(
            side="buy",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        # Buy at ask + slippage
        assert fill_price > 100.10

    def test_total_cost_sell(self) -> None:
        """Sell order: receive bid - slippage - fixed costs."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=1.0,
        )
        fill_price = model.calculate_fill_price(
            side="sell",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        # Sell at bid - slippage
        assert fill_price < 100.00

    def test_buy_fill_price_calculation(self) -> None:
        """Test precise buy fill price calculation."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=0.0,  # No fixed costs for precise calculation
        )
        fill_price = model.calculate_fill_price(
            side="buy",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        # Spread = (100.10 - 100.00) / 100.05 * 10000 ~ 10 bps
        # Slippage = spread_bps + size_bps + vol_bps ~ 10 + 10 + 1 = 21 bps
        # Fill = ask * (1 + slippage/10000) = 100.10 * 1.0021
        assert fill_price > 100.10
        assert fill_price < 100.50  # Reasonable upper bound

    def test_sell_fill_price_calculation(self) -> None:
        """Test precise sell fill price calculation."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=0.0,
        )
        fill_price = model.calculate_fill_price(
            side="sell",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        # Sell at bid - slippage
        assert fill_price < 100.00
        assert fill_price > 99.50  # Reasonable lower bound

    def test_fixed_cost_added_to_buy(self) -> None:
        """Test that fixed costs increase buy price."""
        model_no_fixed = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=0.0,
        )
        model_with_fixed = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=5.0,
        )
        fill_no_fixed = model_no_fixed.calculate_fill_price(
            side="buy",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        fill_with_fixed = model_with_fixed.calculate_fill_price(
            side="buy",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        assert fill_with_fixed > fill_no_fixed

    def test_fixed_cost_reduces_sell(self) -> None:
        """Test that fixed costs decrease sell price (worse for seller)."""
        model_no_fixed = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=0.0,
        )
        model_with_fixed = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
            fixed_cost_bps=5.0,
        )
        fill_no_fixed = model_no_fixed.calculate_fill_price(
            side="sell",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        fill_with_fixed = model_with_fixed.calculate_fill_price(
            side="sell",
            ask_price=100.10,
            bid_price=100.00,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        assert fill_with_fixed < fill_no_fixed

    def test_round_trip_cost_calculation(self) -> None:
        """Test round-trip cost is double one-way plus both fixed costs."""
        slippage_model = SlippageModel(1.0, 1.0, 1.0)
        model = TransactionCostModel(
            slippage_model=slippage_model,
            fixed_cost_bps=2.0,
        )
        round_trip = model.calculate_round_trip_cost_bps(
            spread_bps=10.0,
            order_notional=10000,
            book_notional=100000,
            short_term_vol=0.01,
        )
        # One way slippage = 10 + 10 + 1 = 21 bps
        # Round trip = 2 * (21 + 2) = 46 bps
        expected = 2 * (21.0 + 2.0)
        assert abs(round_trip - expected) < 0.01

    def test_default_fixed_cost(self) -> None:
        """Test that default fixed cost is zero."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(1.0, 1.0, 1.0),
        )
        assert model.fixed_cost_bps == 0.0


class TestLoadCostModel:
    """Tests for load_cost_model - loading from JSON configuration."""

    def test_load_from_json(self, tmp_path: Path) -> None:
        """Load cost model from JSON config file."""
        config_file = tmp_path / "cost_model.json"
        config_file.write_text(
            json.dumps(
                {
                    "version": "cm_v1",
                    "spread_coef": 1.5,
                    "size_coef": 0.8,
                    "vol_coef": 2.0,
                    "fixed_cost_bps": 0.5,
                }
            )
        )
        model = load_cost_model(str(config_file))
        assert model.slippage_model.spread_coef == 1.5
        assert model.slippage_model.size_coef == 0.8
        assert model.slippage_model.vol_coef == 2.0
        assert model.fixed_cost_bps == 0.5

    def test_load_from_path_object(self, tmp_path: Path) -> None:
        """Load cost model using Path object."""
        config_file = tmp_path / "cost_model.json"
        config_file.write_text(
            json.dumps(
                {
                    "spread_coef": 1.0,
                    "size_coef": 1.0,
                    "vol_coef": 1.0,
                    "fixed_cost_bps": 1.0,
                }
            )
        )
        model = load_cost_model(config_file)  # Pass Path object
        assert model.slippage_model.spread_coef == 1.0
        assert model.fixed_cost_bps == 1.0

    def test_load_with_defaults(self, tmp_path: Path) -> None:
        """Load cost model with missing fields uses defaults."""
        config_file = tmp_path / "cost_model.json"
        config_file.write_text(
            json.dumps(
                {
                    "spread_coef": 2.0,
                    # Missing other fields - should use defaults
                }
            )
        )
        model = load_cost_model(config_file)
        assert model.slippage_model.spread_coef == 2.0
        assert model.slippage_model.size_coef == 1.0  # Default
        assert model.slippage_model.vol_coef == 1.0  # Default
        assert model.fixed_cost_bps == 0.0  # Default

    def test_load_empty_config(self, tmp_path: Path) -> None:
        """Load cost model from empty config uses all defaults."""
        config_file = tmp_path / "cost_model.json"
        config_file.write_text("{}")
        model = load_cost_model(config_file)
        assert model.slippage_model.spread_coef == 1.0
        assert model.slippage_model.size_coef == 1.0
        assert model.slippage_model.vol_coef == 1.0
        assert model.fixed_cost_bps == 0.0

    def test_load_with_max_size_impact(self, tmp_path: Path) -> None:
        """Load cost model with custom max_size_impact_bps."""
        config_file = tmp_path / "cost_model.json"
        config_file.write_text(
            json.dumps(
                {
                    "spread_coef": 1.0,
                    "size_coef": 1.0,
                    "vol_coef": 1.0,
                    "max_size_impact_bps": 100.0,
                    "fixed_cost_bps": 0.0,
                }
            )
        )
        model = load_cost_model(config_file)
        assert model.slippage_model.max_size_impact_bps == 100.0

    def test_load_file_not_found(self) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_cost_model("/nonexistent/path/config.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test that loading invalid JSON raises error."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("not valid json {")
        with pytest.raises(json.JSONDecodeError):
            load_cost_model(config_file)


class TestCostModelIntegration:
    """Integration tests for cost model components working together."""

    def test_realistic_trading_scenario(self) -> None:
        """Test a realistic trading scenario with typical market conditions."""
        # Typical SPY-like conditions
        model = TransactionCostModel(
            slippage_model=SlippageModel(
                spread_coef=1.0,
                size_coef=0.5,
                vol_coef=2.0,
                max_size_impact_bps=30.0,
            ),
            fixed_cost_bps=0.35,  # ~$0.0035 per share for institutional
        )

        # SPY at ~$500, spread ~1 cent = ~2 bps
        ask_price = 500.01
        bid_price = 500.00
        order_notional = 50000  # $50k order
        book_notional = 500000  # $500k at top of book
        short_term_vol = 0.001  # 0.1% 30-second vol (typical)

        buy_fill = model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        sell_fill = model.calculate_fill_price(
            side="sell",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        # Buy should be above ask
        assert buy_fill > ask_price
        # Sell should be below bid
        assert sell_fill < bid_price
        # Round trip cost should be reasonable (< 50 bps for liquid instrument)
        round_trip_pct = (buy_fill - sell_fill) / ((ask_price + bid_price) / 2) * 100
        assert round_trip_pct < 0.50  # Less than 50 bps round trip

    def test_illiquid_scenario(self) -> None:
        """Test cost model with illiquid conditions."""
        model = TransactionCostModel(
            slippage_model=SlippageModel(
                spread_coef=1.0,
                size_coef=1.0,
                vol_coef=1.0,
                max_size_impact_bps=100.0,
            ),
            fixed_cost_bps=1.0,
        )

        # Wide spread, thin book
        ask_price = 50.10
        bid_price = 50.00
        order_notional = 50000  # Large order relative to book
        book_notional = 10000  # Thin book
        short_term_vol = 0.05  # High volatility

        buy_fill = model.calculate_fill_price(
            side="buy",
            ask_price=ask_price,
            bid_price=bid_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        # Should have significant slippage
        assert buy_fill > ask_price * 1.005  # At least 50 bps above ask
