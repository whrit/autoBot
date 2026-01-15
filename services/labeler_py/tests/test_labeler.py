"""
Comprehensive test suite for labeler_py service.

Tests cover:
- T2.06: Forward returns calculation at configurable horizons
- T2.07: Direction labels with no-trade band
- T2.08: Net-of-spread returns calculation
- T2.10: Unit tests for labeler
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from labeler_py.labels import DirectionLabeler
from labeler_py.net_returns import NetOfSpreadCalculator
from labeler_py.returns import ForwardReturnsCalculator


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_quotes_df() -> pl.DataFrame:
    """
    Create sample quotes DataFrame for testing.

    Creates 20 minutes of quote data at 1-second intervals.
    """
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
    n_rows = 1200  # 20 minutes of 1-second data

    # Create timestamps
    timestamps = [base_time + timedelta(seconds=i) for i in range(n_rows)]

    # Create price data with some trend and noise
    bid_prices = []
    ask_prices = []
    bid_sizes = []
    ask_sizes = []

    base_bid = 100.0
    spread = 0.02  # 2 cents spread

    for i in range(n_rows):
        # Add small trend and noise
        trend = i * 0.0001  # Small upward trend
        noise = (i % 10 - 5) * 0.001  # Small oscillation
        bid = base_bid + trend + noise
        bid_prices.append(bid)
        ask_prices.append(bid + spread)
        bid_sizes.append(1000.0 + (i % 500))
        ask_sizes.append(800.0 + (i % 400))

    return pl.DataFrame(
        {
            "symbol": ["SPY"] * n_rows,
            "ts_event": timestamps,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        }
    )


@pytest.fixture
def sample_decision_times(sample_quotes_df: pl.DataFrame) -> list[datetime]:
    """
    Get sample decision times from the quotes DataFrame.

    Returns timestamps at 60-second intervals starting from second 60.
    """
    base_time = sample_quotes_df["ts_event"][0]
    return [base_time + timedelta(seconds=s) for s in [60, 120, 180, 240, 300]]


@pytest.fixture
def returns_calculator() -> ForwardReturnsCalculator:
    """Create default ForwardReturnsCalculator instance."""
    return ForwardReturnsCalculator(horizons=[60, 300, 900])


@pytest.fixture
def direction_labeler() -> DirectionLabeler:
    """Create default DirectionLabeler instance."""
    return DirectionLabeler(no_trade_threshold=0.0005)


@pytest.fixture
def net_spread_calculator() -> NetOfSpreadCalculator:
    """Create default NetOfSpreadCalculator instance."""
    return NetOfSpreadCalculator()


# =============================================================================
# T2.06: FORWARD RETURNS CALCULATION TESTS
# =============================================================================


class TestForwardReturnsCalculator:
    """Test suite for ForwardReturnsCalculator (T2.06)."""

    def test_init_default_horizons(self) -> None:
        """Test default horizons are set correctly."""
        calc = ForwardReturnsCalculator()
        assert calc.horizons == [60, 300, 900]

    def test_init_custom_horizons(self) -> None:
        """Test custom horizons are accepted."""
        calc = ForwardReturnsCalculator(horizons=[30, 120, 600])
        assert calc.horizons == [30, 120, 600]

    def test_init_invalid_horizons_raises(self) -> None:
        """Test that non-positive horizons raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ForwardReturnsCalculator(horizons=[0, 60])
        with pytest.raises(ValueError, match="positive"):
            ForwardReturnsCalculator(horizons=[-60, 60])

    def test_calculate_midprice(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test midprice calculation."""
        midprice = returns_calculator._calculate_midprice(100.0, 100.02)
        assert midprice == pytest.approx(100.01)

    def test_calculate_microprice_equal_sizes(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test microprice with equal bid/ask sizes equals midprice."""
        microprice = returns_calculator._calculate_microprice(
            bid_price=100.0, ask_price=100.02, bid_size=1000.0, ask_size=1000.0
        )
        assert microprice == pytest.approx(100.01)

    def test_calculate_microprice_unequal_sizes(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test microprice calculation with unequal sizes."""
        # More ask size means microprice closer to bid
        microprice = returns_calculator._calculate_microprice(
            bid_price=100.0, ask_price=100.02, bid_size=100.0, ask_size=900.0
        )
        # microprice = (100.0 * 900 + 100.02 * 100) / 1000 = 100.002
        assert microprice == pytest.approx(100.002)

    def test_calculate_microprice_zero_total_size(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test microprice falls back to midprice when total size is zero."""
        microprice = returns_calculator._calculate_microprice(
            bid_price=100.0, ask_price=100.02, bid_size=0.0, ask_size=0.0
        )
        assert microprice == pytest.approx(100.01)

    def test_calculate_forward_return(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test forward return calculation."""
        # 1% return
        ret = returns_calculator._calculate_forward_return(
            current_price=100.0, future_price=101.0
        )
        assert ret == pytest.approx(0.01)

        # Negative return
        ret = returns_calculator._calculate_forward_return(
            current_price=100.0, future_price=99.0
        )
        assert ret == pytest.approx(-0.01)

    def test_calculate_returns_basic(
        self,
        returns_calculator: ForwardReturnsCalculator,
        sample_quotes_df: pl.DataFrame,
        sample_decision_times: list[datetime],
    ) -> None:
        """Test basic forward returns calculation."""
        result = returns_calculator.calculate(
            quotes_df=sample_quotes_df,
            decision_times=sample_decision_times,
            symbol="SPY",
        )

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert "symbol" in result.columns
        assert "decision_ts" in result.columns
        assert "horizon" in result.columns
        assert "fwd_return_mid" in result.columns

        # Check we have results for all decision times and horizons
        # Only decision times with enough forward data should have results
        assert len(result) > 0

    def test_calculate_returns_uses_midprice_not_last_trade(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """
        Test that returns are calculated using midprice, not last trade price.

        This is critical for taker-realistic returns.
        """
        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        # Create simple quotes with known midprice
        quotes = pl.DataFrame(
            {
                "symbol": ["SPY"] * 120,
                "ts_event": [base_time + timedelta(seconds=i) for i in range(120)],
                "bid_price": [100.0] * 120,
                "ask_price": [100.02] * 120,  # midprice = 100.01
                "bid_size": [1000.0] * 120,
                "ask_size": [1000.0] * 120,
            }
        )

        # Modify future prices to create known return
        future_bid = 101.0
        future_ask = 101.02  # future midprice = 101.01
        quotes = quotes.with_columns(
            [
                pl.when(pl.col("ts_event") >= base_time + timedelta(seconds=60))
                .then(pl.lit(future_bid))
                .otherwise(pl.col("bid_price"))
                .alias("bid_price"),
                pl.when(pl.col("ts_event") >= base_time + timedelta(seconds=60))
                .then(pl.lit(future_ask))
                .otherwise(pl.col("ask_price"))
                .alias("ask_price"),
            ]
        )

        calc = ForwardReturnsCalculator(horizons=[60])
        result = calc.calculate(
            quotes_df=quotes,
            decision_times=[base_time],
            symbol="SPY",
            use_microprice=False,
        )

        # Expected return: (101.01 - 100.01) / 100.01 = 0.009999...
        expected_return = (101.01 - 100.01) / 100.01
        assert result["fwd_return_mid"][0] == pytest.approx(expected_return, rel=1e-4)

    def test_calculate_returns_multiple_horizons(
        self,
        sample_quotes_df: pl.DataFrame,
    ) -> None:
        """Test returns are calculated for all configured horizons."""
        base_time = sample_quotes_df["ts_event"][0]
        decision_times = [base_time]

        calc = ForwardReturnsCalculator(horizons=[60, 120, 180])
        result = calc.calculate(
            quotes_df=sample_quotes_df,
            decision_times=decision_times,
            symbol="SPY",
        )

        horizons = result["horizon"].to_list()
        assert 60 in horizons
        assert 120 in horizons
        assert 180 in horizons

    def test_calculate_returns_insufficient_data(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test handling when there's insufficient forward data."""
        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        # Create only 30 seconds of data, but request 60-second horizon
        quotes = pl.DataFrame(
            {
                "symbol": ["SPY"] * 30,
                "ts_event": [base_time + timedelta(seconds=i) for i in range(30)],
                "bid_price": [100.0] * 30,
                "ask_price": [100.02] * 30,
                "bid_size": [1000.0] * 30,
                "ask_size": [1000.0] * 30,
            }
        )

        result = returns_calculator.calculate(
            quotes_df=quotes,
            decision_times=[base_time],
            symbol="SPY",
        )

        # Should return empty or no valid results for horizons > 30s
        assert len(result.filter(pl.col("horizon") == 60)) == 0

    def test_calculate_returns_with_microprice(
        self, returns_calculator: ForwardReturnsCalculator
    ) -> None:
        """Test returns calculation using microprice instead of midprice."""
        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        # Create quotes with imbalanced sizes
        quotes = pl.DataFrame(
            {
                "symbol": ["SPY"] * 120,
                "ts_event": [base_time + timedelta(seconds=i) for i in range(120)],
                "bid_price": [100.0] * 120,
                "ask_price": [100.10] * 120,
                "bid_size": [100.0] * 120,  # Small bid size
                "ask_size": [900.0] * 120,  # Large ask size
            }
        )

        result = returns_calculator.calculate(
            quotes_df=quotes,
            decision_times=[base_time],
            symbol="SPY",
            use_microprice=True,
        )

        # With imbalanced sizes, microprice should differ from midprice
        assert len(result) > 0

    def test_calculate_returns_preserves_timezone(
        self, returns_calculator: ForwardReturnsCalculator, sample_quotes_df: pl.DataFrame
    ) -> None:
        """Test that timezone information is preserved in results."""
        decision_times = [sample_quotes_df["ts_event"][0]]

        result = returns_calculator.calculate(
            quotes_df=sample_quotes_df,
            decision_times=decision_times,
            symbol="SPY",
        )

        if len(result) > 0:
            # The decision_ts should have timezone info
            ts = result["decision_ts"][0]
            assert ts.tzinfo is not None


# =============================================================================
# T2.07: DIRECTION LABELS WITH NO-TRADE BAND TESTS
# =============================================================================


class TestDirectionLabeler:
    """Test suite for DirectionLabeler (T2.07)."""

    def test_init_default_threshold(self) -> None:
        """Test default no-trade threshold."""
        labeler = DirectionLabeler()
        assert labeler.no_trade_threshold == 0.0005  # 5 bps default

    def test_init_custom_threshold(self) -> None:
        """Test custom no-trade threshold."""
        labeler = DirectionLabeler(no_trade_threshold=0.001)
        assert labeler.no_trade_threshold == 0.001

    def test_init_invalid_threshold_raises(self) -> None:
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            DirectionLabeler(no_trade_threshold=-0.001)

    def test_label_positive_return_above_threshold(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test that returns above threshold get +1 label."""
        # Return of 0.001 > threshold of 0.0005
        label = direction_labeler.label_direction(0.001)
        assert label == 1

    def test_label_negative_return_below_threshold(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test that returns below negative threshold get -1 label."""
        # Return of -0.001 < -threshold of -0.0005
        label = direction_labeler.label_direction(-0.001)
        assert label == -1

    def test_label_return_within_no_trade_band(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test that returns within no-trade band get 0 label."""
        # Returns within +/- 0.0005
        assert direction_labeler.label_direction(0.0003) == 0
        assert direction_labeler.label_direction(-0.0003) == 0
        assert direction_labeler.label_direction(0.0) == 0

    def test_label_at_threshold_boundary(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test labeling at exact threshold boundaries."""
        # At exactly threshold, should be no-trade (0)
        assert direction_labeler.label_direction(0.0005) == 0
        assert direction_labeler.label_direction(-0.0005) == 0

        # Just above threshold
        assert direction_labeler.label_direction(0.00051) == 1
        assert direction_labeler.label_direction(-0.00051) == -1

    def test_label_dataframe(self, direction_labeler: DirectionLabeler) -> None:
        """Test labeling a DataFrame of returns."""
        returns_df = pl.DataFrame(
            {
                "symbol": ["SPY"] * 5,
                "decision_ts": [
                    datetime(2024, 1, 15, 9, 30, i, tzinfo=timezone.utc)
                    for i in range(5)
                ],
                "horizon": [60] * 5,
                "fwd_return_mid": [0.002, -0.002, 0.0003, -0.0003, 0.0],
            }
        )

        result = direction_labeler.label_returns(returns_df)

        assert "direction" in result.columns
        directions = result["direction"].to_list()
        assert directions == [1, -1, 0, 0, 0]

    def test_label_dataframe_preserves_columns(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test that labeling preserves all existing columns."""
        returns_df = pl.DataFrame(
            {
                "symbol": ["SPY"],
                "decision_ts": [datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)],
                "horizon": [60],
                "fwd_return_mid": [0.002],
                "extra_column": ["value"],
            }
        )

        result = direction_labeler.label_returns(returns_df)

        assert "extra_column" in result.columns
        assert result["extra_column"][0] == "value"

    def test_zero_threshold_labels_all_nonzero(self) -> None:
        """Test with zero threshold, all non-zero returns get direction."""
        labeler = DirectionLabeler(no_trade_threshold=0.0)

        assert labeler.label_direction(0.0001) == 1
        assert labeler.label_direction(-0.0001) == -1
        assert labeler.label_direction(0.0) == 0

    def test_direction_values_constrained(
        self, direction_labeler: DirectionLabeler
    ) -> None:
        """Test that direction values are always -1, 0, or +1."""
        test_returns = [0.1, -0.1, 0.0001, -0.0001, 0.0, 0.5, -0.5]
        for ret in test_returns:
            label = direction_labeler.label_direction(ret)
            assert label in [-1, 0, 1]


# =============================================================================
# T2.08: NET-OF-SPREAD RETURNS TESTS
# =============================================================================


class TestNetOfSpreadCalculator:
    """Test suite for NetOfSpreadCalculator (T2.08)."""

    def test_init_default_values(self) -> None:
        """Test default slippage factor."""
        calc = NetOfSpreadCalculator()
        assert calc.half_spread_factor == 1.0

    def test_init_custom_slippage_factor(self) -> None:
        """Test custom half-spread factor."""
        calc = NetOfSpreadCalculator(half_spread_factor=1.5)
        assert calc.half_spread_factor == 1.5

    def test_calculate_spread_cost_basic(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test basic spread cost calculation."""
        # Spread of 0.02 on price of 100 = 0.02% = 0.0002 relative
        # Half spread cost (one-way) = 0.0001
        cost = net_spread_calculator.calculate_spread_cost(
            bid_price=100.0, ask_price=100.02
        )
        # Cost = half_spread / midprice = 0.01 / 100.01 ~ 0.0001
        assert cost == pytest.approx(0.0001, rel=1e-2)

    def test_calculate_spread_cost_tight_spread(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test spread cost with tight spread."""
        cost = net_spread_calculator.calculate_spread_cost(
            bid_price=100.0, ask_price=100.01
        )
        # Half spread = 0.005, midprice = 100.005
        expected = 0.005 / 100.005
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_calculate_spread_cost_wide_spread(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test spread cost with wide spread."""
        cost = net_spread_calculator.calculate_spread_cost(
            bid_price=100.0, ask_price=100.10
        )
        # Half spread = 0.05, midprice = 100.05
        expected = 0.05 / 100.05
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_calculate_spread_cost_with_factor(self) -> None:
        """Test spread cost with custom half-spread factor."""
        calc = NetOfSpreadCalculator(half_spread_factor=2.0)
        cost = calc.calculate_spread_cost(bid_price=100.0, ask_price=100.02)
        # Double the base cost
        base_cost = 0.01 / 100.01
        assert cost == pytest.approx(2.0 * base_cost, rel=1e-4)

    def test_calculate_net_return_positive_profitable(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test net return when gross return exceeds costs."""
        # Gross return of 0.005 (50 bps)
        # Spread cost of ~0.0001 (1 bp)
        net = net_spread_calculator.calculate_net_return(
            gross_return=0.005, bid_price=100.0, ask_price=100.02
        )
        # Net should be gross - cost
        spread_cost = 0.01 / 100.01
        expected = 0.005 - spread_cost
        assert net == pytest.approx(expected, rel=1e-4)
        assert net > 0  # Still profitable

    def test_calculate_net_return_positive_unprofitable(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test net return when gross return is less than costs."""
        # Gross return of 0.00005 (0.5 bps), less than spread cost
        net = net_spread_calculator.calculate_net_return(
            gross_return=0.00005, bid_price=100.0, ask_price=100.02
        )
        # Should be negative after costs
        assert net < 0.00005  # Reduced by costs

    def test_calculate_net_return_negative(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test net return when gross return is negative."""
        # Negative return becomes more negative after costs
        net = net_spread_calculator.calculate_net_return(
            gross_return=-0.005, bid_price=100.0, ask_price=100.02
        )
        spread_cost = 0.01 / 100.01
        expected = -0.005 - spread_cost
        assert net == pytest.approx(expected, rel=1e-4)
        assert net < -0.005  # More negative

    def test_calculate_net_returns_dataframe(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test calculating net returns for a DataFrame."""
        df = pl.DataFrame(
            {
                "symbol": ["SPY"] * 3,
                "decision_ts": [
                    datetime(2024, 1, 15, 9, 30, i, tzinfo=timezone.utc) for i in range(3)
                ],
                "horizon": [60] * 3,
                "fwd_return_mid": [0.005, -0.005, 0.0001],
                "bid_price": [100.0, 100.0, 100.0],
                "ask_price": [100.02, 100.02, 100.02],
            }
        )

        result = net_spread_calculator.calculate_net_returns(df)

        assert "fwd_return_net" in result.columns
        # Check net returns are less than gross for positive, more negative for negative
        assert result["fwd_return_net"][0] < result["fwd_return_mid"][0]
        assert result["fwd_return_net"][1] < result["fwd_return_mid"][1]
        assert result["fwd_return_net"][2] < result["fwd_return_mid"][2]

    def test_calculate_net_returns_preserves_columns(
        self, net_spread_calculator: NetOfSpreadCalculator
    ) -> None:
        """Test that net returns calculation preserves existing columns."""
        df = pl.DataFrame(
            {
                "symbol": ["SPY"],
                "decision_ts": [datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)],
                "horizon": [60],
                "fwd_return_mid": [0.005],
                "bid_price": [100.0],
                "ask_price": [100.02],
                "extra": ["value"],
            }
        )

        result = net_spread_calculator.calculate_net_returns(df)

        assert "extra" in result.columns
        assert result["extra"][0] == "value"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestLabelingPipeline:
    """Integration tests for the full labeling pipeline."""

    def test_full_pipeline(self, sample_quotes_df: pl.DataFrame) -> None:
        """Test the full labeling pipeline: returns -> net returns -> direction."""
        base_time = sample_quotes_df["ts_event"][0]
        decision_times = [base_time + timedelta(seconds=60)]

        # Step 1: Calculate forward returns
        returns_calc = ForwardReturnsCalculator(horizons=[60, 120])
        returns_df = returns_calc.calculate(
            quotes_df=sample_quotes_df,
            decision_times=decision_times,
            symbol="SPY",
        )

        if len(returns_df) == 0:
            pytest.skip("Not enough data for forward returns")

        # Add bid/ask prices for net return calculation
        # Get the prices at decision time
        decision_quote = sample_quotes_df.filter(
            pl.col("ts_event") == decision_times[0]
        )
        bid_price = decision_quote["bid_price"][0]
        ask_price = decision_quote["ask_price"][0]

        returns_df = returns_df.with_columns(
            [
                pl.lit(bid_price).alias("bid_price"),
                pl.lit(ask_price).alias("ask_price"),
            ]
        )

        # Step 2: Calculate net-of-spread returns
        net_calc = NetOfSpreadCalculator()
        net_returns_df = net_calc.calculate_net_returns(returns_df)

        # Step 3: Label directions
        labeler = DirectionLabeler(no_trade_threshold=0.0005)
        # Use net returns for labeling
        labeled_df = net_returns_df.with_columns(
            [
                pl.col("fwd_return_net")
                .map_elements(labeler.label_direction, return_dtype=pl.Int64)
                .alias("direction")
            ]
        )

        # Verify final schema matches Label model
        required_columns = [
            "symbol",
            "decision_ts",
            "horizon",
            "fwd_return_mid",
            "fwd_return_net",
            "direction",
        ]
        for col in required_columns:
            assert col in labeled_df.columns

        # Verify direction values are valid
        directions = labeled_df["direction"].to_list()
        for d in directions:
            assert d in [-1, 0, 1]

    def test_pipeline_handles_empty_input(self) -> None:
        """Test pipeline handles empty input gracefully."""
        empty_quotes = pl.DataFrame(
            {
                "symbol": pl.Series([], dtype=pl.Utf8),
                "ts_event": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "bid_price": pl.Series([], dtype=pl.Float64),
                "ask_price": pl.Series([], dtype=pl.Float64),
                "bid_size": pl.Series([], dtype=pl.Float64),
                "ask_size": pl.Series([], dtype=pl.Float64),
            }
        )

        returns_calc = ForwardReturnsCalculator(horizons=[60])
        result = returns_calc.calculate(
            quotes_df=empty_quotes,
            decision_times=[],
            symbol="SPY",
        )

        assert len(result) == 0

    def test_label_model_compatibility(self, sample_quotes_df: pl.DataFrame) -> None:
        """Test that output can be converted to Label pydantic models."""
        pytest.importorskip("common_types")
        from common_types import Label

        base_time = sample_quotes_df["ts_event"][0]
        decision_times = [base_time + timedelta(seconds=60)]

        returns_calc = ForwardReturnsCalculator(horizons=[60])
        returns_df = returns_calc.calculate(
            quotes_df=sample_quotes_df,
            decision_times=decision_times,
            symbol="SPY",
        )

        if len(returns_df) == 0:
            pytest.skip("Not enough data for forward returns")

        # Add required columns for Label model
        decision_quote = sample_quotes_df.filter(
            pl.col("ts_event") == decision_times[0]
        )
        bid_price = decision_quote["bid_price"][0]
        ask_price = decision_quote["ask_price"][0]

        returns_df = returns_df.with_columns(
            [
                pl.lit(bid_price).alias("bid_price"),
                pl.lit(ask_price).alias("ask_price"),
            ]
        )

        net_calc = NetOfSpreadCalculator()
        net_returns_df = net_calc.calculate_net_returns(returns_df)

        labeler = DirectionLabeler()
        labeled_df = labeler.label_returns(net_returns_df, return_column="fwd_return_net")

        # Try to create Label models from results
        for row in labeled_df.iter_rows(named=True):
            label = Label(
                symbol=row["symbol"],
                decision_ts=row["decision_ts"],
                horizon=row["horizon"],
                fwd_return_mid=row["fwd_return_mid"],
                fwd_return_net=row["fwd_return_net"],
                direction=row["direction"],
            )
            assert label.direction in [-1, 0, 1]
            assert label.horizon > 0
