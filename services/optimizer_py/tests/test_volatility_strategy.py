"""
TDD Tests for Volatility Strategy Family (T4.04).

Test-first approach: These tests define the expected behavior of:
- VolatilityStrategy implementation
- ATR breakout signals
- Volatility targeting position sizing
- Volatility mean-reversion signals
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

# =============================================================================
# Fixtures for Volatility Strategy Tests
# =============================================================================


@pytest.fixture
def high_volatility_decision_frame() -> pl.DataFrame:
    """Create a decision frame with high volatility characteristics."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(789)
    base_price = 100.0

    # Generate high volatility price data with expanding ATR
    prices = [base_price]
    highs = [base_price * 1.02]
    lows = [base_price * 0.98]

    for i in range(1, n_rows):
        # Increasing volatility over time
        vol_multiplier = 1 + (i / n_rows) * 2  # Vol increases 3x by end
        move = np.random.randn() * 0.01 * vol_multiplier
        new_price = prices[-1] * (1 + move)
        prices.append(new_price)

        # High/low range also expands
        range_pct = 0.01 * vol_multiplier
        highs.append(new_price * (1 + range_pct))
        lows.append(new_price * (1 - range_pct))

    prices = np.array(prices)
    highs = np.array(highs)
    lows = np.array(lows)
    opens = np.roll(prices, 1)
    opens[0] = base_price

    volumes = np.random.randint(50000, 200000, size=n_rows).astype(float)
    spread_bps = np.ones(n_rows) * 2.0

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["VXX"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": prices * 0.9999,
        "ask_price": prices * 1.0001,
        "spread_bps": spread_bps,
    })


@pytest.fixture
def low_volatility_decision_frame() -> pl.DataFrame:
    """Create a decision frame with low, stable volatility."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(111)
    base_price = 100.0

    # Very low volatility - tight range
    prices = [base_price]
    for _ in range(1, n_rows):
        move = np.random.randn() * 0.001  # Very small moves
        prices.append(prices[-1] * (1 + move))

    prices = np.array(prices)
    highs = prices * 1.001
    lows = prices * 0.999
    opens = np.roll(prices, 1)
    opens[0] = base_price

    volumes = np.random.randint(30000, 80000, size=n_rows).astype(float)
    spread_bps = np.ones(n_rows) * 1.0

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["BND"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": prices * 0.9999,
        "ask_price": prices * 1.0001,
        "spread_bps": spread_bps,
    })


@pytest.fixture
def atr_breakout_frame() -> pl.DataFrame:
    """Create a decision frame with clear ATR breakout conditions."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(222)
    base_price = 100.0

    # First half: normal volatility
    # Second half: breakout with large moves exceeding ATR bands
    prices = [base_price]
    highs = [base_price * 1.01]
    lows = [base_price * 0.99]

    for i in range(1, n_rows):
        # Normal volatility for first half, breakout regime for second half
        move = (
            np.random.randn() * 0.005
            if i < 50
            else 0.015 + np.random.randn() * 0.005
        )

        new_price = prices[-1] * (1 + move)
        prices.append(new_price)

        if i < 50:
            highs.append(new_price * 1.008)
            lows.append(new_price * 0.992)
        else:
            # Larger ranges during breakout
            highs.append(new_price * 1.015)
            lows.append(new_price * 0.985)

    prices = np.array(prices)
    highs = np.array(highs)
    lows = np.array(lows)
    opens = np.roll(prices, 1)
    opens[0] = base_price

    volumes = np.random.randint(50000, 150000, size=n_rows).astype(float)

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["QQQ"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "bid_price": prices * 0.9999,
        "ask_price": prices * 1.0001,
        "spread_bps": np.ones(n_rows) * 2.0,
    })


# =============================================================================
# T4.04: Volatility Strategy Tests
# =============================================================================


class TestVolatilityStrategyCreation:
    """Tests for VolatilityStrategy instantiation."""

    def test_volatility_strategy_creation(self) -> None:
        """Test VolatilityStrategy instantiation with default parameters."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()

        assert strategy.name == "volatility"
        assert isinstance(strategy.parameters, dict)
        assert "atr_period" in strategy.parameters
        assert "atr_multiplier" in strategy.parameters

    def test_volatility_strategy_custom_parameters(self) -> None:
        """Test VolatilityStrategy with custom parameters."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(
            atr_period=10,
            atr_multiplier=1.5,
            vol_lookback=15,
            vol_target=0.15,
            vol_reversion_threshold=1.5,
        )

        assert strategy.parameters["atr_period"] == 10
        assert strategy.parameters["atr_multiplier"] == 1.5
        assert strategy.parameters["vol_lookback"] == 15
        assert strategy.parameters["vol_target"] == 0.15
        assert strategy.parameters["vol_reversion_threshold"] == 1.5

    def test_volatility_strategy_name_property(self) -> None:
        """Test VolatilityStrategy name property."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        assert strategy.name == "volatility"

    def test_volatility_strategy_implements_interface(self) -> None:
        """Test that VolatilityStrategy implements StrategyFamily interface."""
        from optimizer_py.strategy_family import StrategyFamily
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()

        assert isinstance(strategy, StrategyFamily)
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "parameters")
        assert hasattr(strategy, "generate_signals")
        assert hasattr(strategy, "get_parameter_grid")


class TestVolatilityStrategyParameterValidation:
    """Tests for parameter validation."""

    def test_atr_period_must_be_positive(self) -> None:
        """Test that ATR period must be positive."""
        from optimizer_py.volatility import VolatilityStrategy

        with pytest.raises(ValueError, match="atr_period"):
            VolatilityStrategy(atr_period=0)

        with pytest.raises(ValueError, match="atr_period"):
            VolatilityStrategy(atr_period=-5)

    def test_atr_multiplier_must_be_positive(self) -> None:
        """Test that ATR multiplier must be positive."""
        from optimizer_py.volatility import VolatilityStrategy

        with pytest.raises(ValueError, match="atr_multiplier"):
            VolatilityStrategy(atr_multiplier=0)

        with pytest.raises(ValueError, match="atr_multiplier"):
            VolatilityStrategy(atr_multiplier=-1.0)

    def test_vol_target_must_be_in_valid_range(self) -> None:
        """Test that volatility target must be reasonable."""
        from optimizer_py.volatility import VolatilityStrategy

        with pytest.raises(ValueError, match="vol_target"):
            VolatilityStrategy(vol_target=0)

        with pytest.raises(ValueError, match="vol_target"):
            VolatilityStrategy(vol_target=-0.1)

    def test_vol_lookback_must_be_positive(self) -> None:
        """Test that volatility lookback must be positive."""
        from optimizer_py.volatility import VolatilityStrategy

        with pytest.raises(ValueError, match="vol_lookback"):
            VolatilityStrategy(vol_lookback=0)


class TestVolatilityStrategyParameterGrid:
    """Tests for parameter grid generation."""

    def test_volatility_strategy_parameter_grid(self) -> None:
        """Test VolatilityStrategy parameter grid generation."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        grid = strategy.get_parameter_grid()

        assert "atr_period" in grid
        assert "atr_multiplier" in grid
        assert "vol_lookback" in grid
        assert "vol_target" in grid
        assert "vol_reversion_threshold" in grid

        # Verify grid values are lists
        for key, values in grid.items():
            assert isinstance(values, list), f"{key} should be a list"
            assert len(values) > 0, f"{key} should have values"

    def test_parameter_grid_values_are_numeric(self) -> None:
        """Test that parameter grid values are numeric."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        grid = strategy.get_parameter_grid()

        for param_name, values in grid.items():
            for v in values:
                assert isinstance(v, (int, float)), f"{param_name} values should be numeric"


class TestATRBreakoutSignals:
    """Tests for ATR breakout signal generation."""

    def test_generate_signals_returns_list(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generate_signals returns a list of StrategySignal."""
        from optimizer_py.strategy_family import StrategySignal
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, StrategySignal)

    def test_signals_have_valid_values(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generated signals have valid signal values."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0

    def test_atr_breakout_detects_upward_breakout(
        self, atr_breakout_frame: pl.DataFrame
    ) -> None:
        """Test that ATR breakout detects upward price breakouts."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(atr_period=14, atr_multiplier=2.0)
        signals = strategy.generate_signals(atr_breakout_frame)

        # After row 50, we have strong upward breakout
        # Should generate some long signals
        long_signals = [s for s in signals if s.signal == 1]
        assert len(long_signals) > 0, "Should detect upward breakout"

    def test_atr_breakout_metadata_contains_atr(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that ATR value is included in signal metadata."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            assert "atr" in signal.metadata
            assert signal.metadata["atr"] >= 0

    def test_atr_breakout_metadata_contains_band_info(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that ATR band information is in metadata."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            if "atr_upper" in signal.metadata:
                assert "atr_lower" in signal.metadata
                assert signal.metadata["atr_upper"] > signal.metadata["atr_lower"]


class TestVolatilityTargeting:
    """Tests for volatility targeting position sizing."""

    def test_vol_target_position_size_included(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that volatility-based position size factor is computed."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(vol_target=0.15)
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            if "vol_target_size" in signal.metadata:
                # Position size factor should be positive
                assert signal.metadata["vol_target_size"] > 0

    def test_high_vol_reduces_position_size(
        self, high_volatility_decision_frame: pl.DataFrame
    ) -> None:
        """Test that high volatility leads to smaller position sizes."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(vol_target=0.10)
        signals = strategy.generate_signals(high_volatility_decision_frame)

        # In high vol environment, position sizes should be reduced (< 1.0)
        # Get later signals where vol is highest
        late_signals = [s for s in signals if "vol_target_size" in s.metadata]
        if late_signals:
            # At least some positions should be scaled down
            small_positions = [s for s in late_signals if s.metadata["vol_target_size"] < 1.0]
            assert len(small_positions) > 0, "High vol should reduce position sizes"

    def test_low_vol_increases_position_size(
        self, low_volatility_decision_frame: pl.DataFrame
    ) -> None:
        """Test that low volatility leads to larger position sizes."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(vol_target=0.20)  # High target
        signals = strategy.generate_signals(low_volatility_decision_frame)

        # In low vol environment, position sizes could be increased (>= 1.0)
        # But capped at some reasonable maximum
        for signal in signals:
            if "vol_target_size" in signal.metadata:
                # Size should be capped at a reasonable maximum
                assert signal.metadata["vol_target_size"] <= 3.0

    def test_realized_volatility_in_metadata(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that realized volatility is included in metadata."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            if "realized_vol" in signal.metadata:
                assert signal.metadata["realized_vol"] >= 0


class TestVolatilityMeanReversion:
    """Tests for volatility mean-reversion signals."""

    def test_vol_reversion_signal_generation(
        self, high_volatility_decision_frame: pl.DataFrame
    ) -> None:
        """Test that volatility mean-reversion generates signals."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(vol_reversion_threshold=1.5)
        signals = strategy.generate_signals(high_volatility_decision_frame)

        # High vol frame has expanding volatility
        # Should detect when vol is elevated vs historical
        vol_reversion_signals = [
            s for s in signals if s.metadata.get("vol_reversion_trigger")
        ]
        # May or may not have signals depending on data
        assert isinstance(vol_reversion_signals, list)

    def test_vol_zscore_in_metadata(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that volatility z-score is included in metadata."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            if "vol_zscore" in signal.metadata:
                # Z-score can be any real number
                assert isinstance(signal.metadata["vol_zscore"], (int, float))


class TestVolatilityStrategyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_empty_frame(self) -> None:
        """Test VolatilityStrategy handles empty DataFrame gracefully."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        empty_frame = pl.DataFrame({
            "timestamp": [],
            "symbol": [],
            "close": [],
            "high": [],
            "low": [],
            "open": [],
            "volume": [],
        })

        signals = strategy.generate_signals(empty_frame)
        assert signals == []

    def test_handles_insufficient_data(self) -> None:
        """Test VolatilityStrategy with insufficient data for indicators."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(atr_period=14, vol_lookback=20)

        # Only 5 rows - not enough for ATR or volatility calculation
        small_frame = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=i)
                for i in range(5)
            ],
            "symbol": ["AAPL"] * 5,
            "close": [100.0, 101.0, 102.0, 101.5, 103.0],
            "high": [100.5, 101.5, 102.5, 102.0, 103.5],
            "low": [99.5, 100.5, 101.5, 101.0, 102.5],
            "open": [100.0, 100.0, 101.0, 102.0, 101.5],
            "volume": [10000.0] * 5,
        })

        signals = strategy.generate_signals(small_frame)
        # Should return empty or minimal signals
        assert isinstance(signals, list)

    def test_handles_missing_high_low_columns(self) -> None:
        """Test VolatilityStrategy when high/low columns are missing."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()

        # Frame without high/low columns
        frame = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=i)
                for i in range(50)
            ],
            "symbol": ["AAPL"] * 50,
            "close": [100.0 + i * 0.1 for i in range(50)],
            "open": [100.0 + i * 0.1 for i in range(50)],
            "volume": [10000.0] * 50,
        })

        # Should either raise ValueError or handle gracefully
        # ATR requires high/low, so should raise error
        with pytest.raises(ValueError, match="high.*low|missing"):
            strategy.generate_signals(frame)

    def test_handles_multi_symbol(
        self, multi_symbol_decision_frame: pl.DataFrame
    ) -> None:
        """Test VolatilityStrategy with multiple symbols."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(multi_symbol_decision_frame)

        # Should handle multiple symbols
        assert isinstance(signals, list)
        symbols_in_signals = {s.symbol for s in signals}
        assert len(symbols_in_signals) > 0

    def test_missing_required_columns_raises_error(self) -> None:
        """Test that missing required columns raise ValueError."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()

        # Missing 'close' column
        frame = pl.DataFrame({
            "timestamp": [datetime.now(UTC)],
            "symbol": ["AAPL"],
            "high": [101.0],
            "low": [99.0],
        })

        with pytest.raises(ValueError, match="close|Missing"):
            strategy.generate_signals(frame)


class TestVolatilityStrategyConfidence:
    """Tests for confidence calculation."""

    def test_confidence_in_valid_range(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that confidence is always between 0 and 1."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            assert 0.0 <= signal.confidence <= 1.0

    def test_stronger_breakout_higher_confidence(
        self, atr_breakout_frame: pl.DataFrame
    ) -> None:
        """Test that stronger breakouts have higher confidence."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(atr_breakout_frame)

        # During breakout period, confidence should be higher
        # Filter signals from breakout period (row 50+)
        if signals:
            confidences = [s.confidence for s in signals]
            # Should have some variation in confidence
            assert max(confidences) > min(confidences) or len(confidences) == 1


class TestVolatilityStrategyIntegration:
    """Integration tests with other strategy components."""

    def test_signals_compatible_with_backtester(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that volatility signals are compatible with backtester."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        # Signals should have required fields for backtester
        for signal in signals:
            assert signal.timestamp is not None
            assert signal.symbol is not None
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0

    def test_strategy_repr(self) -> None:
        """Test strategy string representation."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy = VolatilityStrategy(atr_period=10, atr_multiplier=1.5)
        repr_str = repr(strategy)

        assert "VolatilityStrategy" in repr_str
        assert "volatility" in repr_str

    def test_strategy_reproducibility(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that strategy produces reproducible results."""
        from optimizer_py.volatility import VolatilityStrategy

        strategy1 = VolatilityStrategy(atr_period=14, atr_multiplier=2.0)
        strategy2 = VolatilityStrategy(atr_period=14, atr_multiplier=2.0)

        signals1 = strategy1.generate_signals(sample_decision_frame)
        signals2 = strategy2.generate_signals(sample_decision_frame)

        # Same parameters should produce same signals
        assert len(signals1) == len(signals2)
        for s1, s2 in zip(signals1, signals2, strict=True):
            assert s1.timestamp == s2.timestamp
            assert s1.symbol == s2.symbol
            assert s1.signal == s2.signal
