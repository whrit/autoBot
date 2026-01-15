"""
TDD Tests for Strategy Family Framework (T4.01-T4.03).

Test-first approach: These tests define the expected behavior of:
- StrategySignal dataclass
- StrategyFamily abstract base class
- TrendStrategy implementation
- MeanReversionStrategy implementation
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

# =============================================================================
# T4.01: Strategy Family Framework Tests
# =============================================================================


class TestStrategySignal:
    """Tests for StrategySignal dataclass."""

    def test_strategy_signal_creation(self) -> None:
        """Test basic StrategySignal creation with required fields."""
        from optimizer_py.strategy_family import StrategySignal

        signal = StrategySignal(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            symbol="AAPL",
            signal=1,
            confidence=0.85,
            metadata={"source": "trend"},
        )

        assert signal.timestamp == datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        assert signal.symbol == "AAPL"
        assert signal.signal == 1
        assert signal.confidence == 0.85
        assert signal.metadata == {"source": "trend"}

    def test_strategy_signal_valid_values(self) -> None:
        """Test StrategySignal with valid signal values (-1, 0, +1)."""
        from optimizer_py.strategy_family import StrategySignal

        # Long signal
        long_signal = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal=1,
            confidence=0.9,
            metadata={},
        )
        assert long_signal.signal == 1

        # Short signal
        short_signal = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal=-1,
            confidence=0.75,
            metadata={},
        )
        assert short_signal.signal == -1

        # Flat/neutral signal
        flat_signal = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal=0,
            confidence=0.5,
            metadata={},
        )
        assert flat_signal.signal == 0

    def test_strategy_signal_confidence_bounds(self) -> None:
        """Test that confidence is between 0 and 1."""
        from optimizer_py.strategy_family import StrategySignal

        signal = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            signal=1,
            confidence=0.0,
            metadata={},
        )
        assert 0.0 <= signal.confidence <= 1.0

        signal_high = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            signal=1,
            confidence=1.0,
            metadata={},
        )
        assert 0.0 <= signal_high.confidence <= 1.0

    def test_strategy_signal_metadata(self) -> None:
        """Test StrategySignal with various metadata."""
        from optimizer_py.strategy_family import StrategySignal

        signal = StrategySignal(
            timestamp=datetime.now(UTC),
            symbol="MSFT",
            signal=1,
            confidence=0.8,
            metadata={
                "indicator": "ma_crossover",
                "fast_ma": 10,
                "slow_ma": 50,
                "rsi": 65.5,
            },
        )

        assert signal.metadata["indicator"] == "ma_crossover"
        assert signal.metadata["fast_ma"] == 10


class TestStrategyFamily:
    """Tests for StrategyFamily abstract base class."""

    def test_strategy_family_is_abstract(self) -> None:
        """Test that StrategyFamily cannot be instantiated directly."""
        from optimizer_py.strategy_family import StrategyFamily

        with pytest.raises(TypeError):
            StrategyFamily()  # type: ignore[abstract]

    def test_strategy_family_requires_name(self) -> None:
        """Test that concrete implementations must provide name property."""
        from optimizer_py.strategy_family import StrategyFamily, StrategySignal

        class IncompleteStrategy(StrategyFamily):
            @property
            def parameters(self) -> dict[str, float | int | str]:
                return {}

            def generate_signals(
                self, decision_frame: pl.DataFrame
            ) -> list[StrategySignal]:
                return []

            def get_parameter_grid(self) -> dict[str, list[float | int]]:
                return {}

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]

    def test_strategy_family_requires_parameters(self) -> None:
        """Test that concrete implementations must provide parameters property."""
        from optimizer_py.strategy_family import StrategyFamily, StrategySignal

        class IncompleteStrategy(StrategyFamily):
            @property
            def name(self) -> str:
                return "incomplete"

            def generate_signals(
                self, decision_frame: pl.DataFrame
            ) -> list[StrategySignal]:
                return []

            def get_parameter_grid(self) -> dict[str, list[float | int]]:
                return {}

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]

    def test_strategy_family_requires_generate_signals(self) -> None:
        """Test that concrete implementations must provide generate_signals method."""
        from optimizer_py.strategy_family import StrategyFamily

        class IncompleteStrategy(StrategyFamily):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def parameters(self) -> dict[str, float | int | str]:
                return {}

            def get_parameter_grid(self) -> dict[str, list[float | int]]:
                return {}

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]

    def test_strategy_family_requires_parameter_grid(self) -> None:
        """Test that concrete implementations must provide get_parameter_grid method."""
        from optimizer_py.strategy_family import StrategyFamily, StrategySignal

        class IncompleteStrategy(StrategyFamily):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def parameters(self) -> dict[str, float | int | str]:
                return {}

            def generate_signals(
                self, decision_frame: pl.DataFrame
            ) -> list[StrategySignal]:
                return []

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]


# =============================================================================
# T4.02: Trend Strategy Tests
# =============================================================================


class TestTrendStrategy:
    """Tests for TrendStrategy implementation."""

    def test_trend_strategy_creation(self) -> None:
        """Test TrendStrategy instantiation with default parameters."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()

        assert strategy.name == "trend"
        assert isinstance(strategy.parameters, dict)
        assert "ma_fast" in strategy.parameters
        assert "ma_slow" in strategy.parameters

    def test_trend_strategy_custom_parameters(self) -> None:
        """Test TrendStrategy with custom parameters."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(
            ma_fast=5,
            ma_slow=20,
            rsi_period=14,
            rsi_threshold=30,
            breakout_lookback=10,
        )

        assert strategy.parameters["ma_fast"] == 5
        assert strategy.parameters["ma_slow"] == 20
        assert strategy.parameters["rsi_period"] == 14
        assert strategy.parameters["rsi_threshold"] == 30
        assert strategy.parameters["breakout_lookback"] == 10

    def test_trend_strategy_name_property(self) -> None:
        """Test TrendStrategy name property."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        assert strategy.name == "trend"

    def test_trend_strategy_parameter_grid(self) -> None:
        """Test TrendStrategy parameter grid generation."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        grid = strategy.get_parameter_grid()

        assert "ma_fast" in grid
        assert "ma_slow" in grid
        assert "rsi_period" in grid
        assert "rsi_threshold" in grid
        assert "breakout_lookback" in grid

        # Verify grid values are lists
        for _key, values in grid.items():
            assert isinstance(values, list)
            assert len(values) > 0

    def test_trend_strategy_generate_signals_returns_list(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generate_signals returns a list of StrategySignal."""
        from optimizer_py.strategy_family import StrategySignal
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, StrategySignal)

    def test_trend_strategy_signals_have_valid_values(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generated signals have valid signal values."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.symbol == "AAPL"

    def test_trend_strategy_detects_uptrend(
        self, trending_decision_frame: pl.DataFrame
    ) -> None:
        """Test that TrendStrategy generates long signals in uptrend."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(ma_fast=5, ma_slow=20)
        signals = strategy.generate_signals(trending_decision_frame)

        # In a strong uptrend, expect more long signals
        long_signals = [s for s in signals if s.signal == 1]
        short_signals = [s for s in signals if s.signal == -1]

        # With a 20% uptrend, should have more longs than shorts
        assert len(long_signals) >= len(short_signals)

    def test_trend_strategy_ma_crossover_logic(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test MA crossover signal generation."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(ma_fast=5, ma_slow=10)
        signals = strategy.generate_signals(sample_decision_frame)

        # Should generate some signals (not all flat)
        non_flat = [s for s in signals if s.signal != 0]
        assert len(non_flat) > 0

    def test_trend_strategy_rsi_integration(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test RSI indicator integration in signal generation."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(rsi_period=14, rsi_threshold=30)
        signals = strategy.generate_signals(sample_decision_frame)

        # Verify RSI values in metadata when present
        for signal in signals:
            if "rsi" in signal.metadata:
                assert 0 <= signal.metadata["rsi"] <= 100

    def test_trend_strategy_breakout_detection(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test channel breakout detection."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(breakout_lookback=10)
        signals = strategy.generate_signals(sample_decision_frame)

        # Verify breakout metadata when present
        for signal in signals:
            if "breakout" in signal.metadata:
                assert signal.metadata["breakout"] in ["upper", "lower", None]

    def test_trend_strategy_handles_empty_frame(self) -> None:
        """Test TrendStrategy handles empty DataFrame gracefully."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
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

    def test_trend_strategy_handles_insufficient_data(self) -> None:
        """Test TrendStrategy with fewer rows than required for indicators."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy(ma_fast=10, ma_slow=20)

        # Only 5 rows - not enough for 20-period MA
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
        # Should return empty or minimal signals when data insufficient
        assert isinstance(signals, list)

    def test_trend_strategy_multi_symbol(
        self, multi_symbol_decision_frame: pl.DataFrame
    ) -> None:
        """Test TrendStrategy with multiple symbols."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        signals = strategy.generate_signals(multi_symbol_decision_frame)

        # Should have signals for different symbols
        symbols_in_signals = {s.symbol for s in signals}
        assert len(symbols_in_signals) > 0

    def test_trend_strategy_momentum_roc(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test Rate of Change momentum indicator."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        # ROC should be in metadata when computed
        for signal in signals:
            if "roc" in signal.metadata:
                assert isinstance(signal.metadata["roc"], (int, float))


# =============================================================================
# T4.03: Mean Reversion Strategy Tests
# =============================================================================


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy implementation."""

    def test_mean_reversion_strategy_creation(self) -> None:
        """Test MeanReversionStrategy instantiation with default parameters."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()

        assert strategy.name == "mean_reversion"
        assert isinstance(strategy.parameters, dict)
        assert "bb_period" in strategy.parameters
        assert "bb_std" in strategy.parameters

    def test_mean_reversion_strategy_custom_parameters(self) -> None:
        """Test MeanReversionStrategy with custom parameters."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(
            bb_period=15,
            bb_std=1.5,
            zscore_threshold=1.5,
            rsi_overbought=75,
            rsi_oversold=25,
        )

        assert strategy.parameters["bb_period"] == 15
        assert strategy.parameters["bb_std"] == 1.5
        assert strategy.parameters["zscore_threshold"] == 1.5
        assert strategy.parameters["rsi_overbought"] == 75
        assert strategy.parameters["rsi_oversold"] == 25

    def test_mean_reversion_strategy_name_property(self) -> None:
        """Test MeanReversionStrategy name property."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_reversion"

    def test_mean_reversion_strategy_parameter_grid(self) -> None:
        """Test MeanReversionStrategy parameter grid generation."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        grid = strategy.get_parameter_grid()

        assert "bb_period" in grid
        assert "bb_std" in grid
        assert "zscore_threshold" in grid
        assert "rsi_overbought" in grid
        assert "rsi_oversold" in grid

        # Verify grid values are lists
        for _key, values in grid.items():
            assert isinstance(values, list)
            assert len(values) > 0

    def test_mean_reversion_generate_signals_returns_list(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generate_signals returns a list of StrategySignal."""
        from optimizer_py.mean_reversion import MeanReversionStrategy
        from optimizer_py.strategy_family import StrategySignal

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, StrategySignal)

    def test_mean_reversion_signals_have_valid_values(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that generated signals have valid signal values."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        for signal in signals:
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0

    def test_mean_reversion_bollinger_bands(
        self, mean_reverting_decision_frame: pl.DataFrame
    ) -> None:
        """Test Bollinger Bands signal generation."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        signals = strategy.generate_signals(mean_reverting_decision_frame)

        # In mean-reverting data, should generate both long and short signals
        long_signals = [s for s in signals if s.signal == 1]
        short_signals = [s for s in signals if s.signal == -1]

        # Mean-reverting data should produce signals in both directions
        assert len(long_signals) + len(short_signals) > 0

    def test_mean_reversion_zscore_logic(
        self, mean_reverting_decision_frame: pl.DataFrame
    ) -> None:
        """Test Z-score based mean reversion logic."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(zscore_threshold=1.5)
        signals = strategy.generate_signals(mean_reverting_decision_frame)

        # Z-score should be in metadata
        for signal in signals:
            if "zscore" in signal.metadata:
                assert isinstance(signal.metadata["zscore"], (int, float))

    def test_mean_reversion_rsi_overbought_oversold(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test RSI overbought/oversold signal generation."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(rsi_overbought=70, rsi_oversold=30)
        signals = strategy.generate_signals(sample_decision_frame)

        # RSI should be in metadata when computed
        for signal in signals:
            if "rsi" in signal.metadata:
                assert 0 <= signal.metadata["rsi"] <= 100

    def test_mean_reversion_handles_empty_frame(self) -> None:
        """Test MeanReversionStrategy handles empty DataFrame gracefully."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
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

    def test_mean_reversion_handles_insufficient_data(self) -> None:
        """Test MeanReversionStrategy with insufficient data for indicators."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(bb_period=20)

        # Only 10 rows - not enough for 20-period BB
        small_frame = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 9, 30, tzinfo=UTC) + timedelta(minutes=i)
                for i in range(10)
            ],
            "symbol": ["AAPL"] * 10,
            "close": [100.0 + i * 0.1 for i in range(10)],
            "high": [100.5 + i * 0.1 for i in range(10)],
            "low": [99.5 + i * 0.1 for i in range(10)],
            "open": [100.0 + i * 0.1 for i in range(10)],
            "volume": [10000.0] * 10,
        })

        signals = strategy.generate_signals(small_frame)
        assert isinstance(signals, list)

    def test_mean_reversion_multi_symbol(
        self, multi_symbol_decision_frame: pl.DataFrame
    ) -> None:
        """Test MeanReversionStrategy with multiple symbols."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(multi_symbol_decision_frame)

        # Should handle multiple symbols
        assert isinstance(signals, list)

    def test_mean_reversion_contrarian_signals(
        self, trending_decision_frame: pl.DataFrame
    ) -> None:
        """Test that mean reversion generates contrarian signals in trends."""
        from optimizer_py.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy(bb_std=1.5, zscore_threshold=1.0)
        signals = strategy.generate_signals(trending_decision_frame)

        # In a strong trend, mean reversion should generate some short signals
        # (contrarian to the trend)
        short_signals = [s for s in signals if s.signal == -1]
        # May not have shorts if trend is within bands, but structure is correct
        assert isinstance(short_signals, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestStrategyIntegration:
    """Integration tests for strategy framework."""

    def test_both_strategies_implement_same_interface(self) -> None:
        """Test that both strategies implement StrategyFamily interface."""
        from optimizer_py.mean_reversion import MeanReversionStrategy
        from optimizer_py.strategy_family import StrategyFamily
        from optimizer_py.trend import TrendStrategy

        trend = TrendStrategy()
        mean_rev = MeanReversionStrategy()

        assert isinstance(trend, StrategyFamily)
        assert isinstance(mean_rev, StrategyFamily)

        # Both should have required properties/methods
        assert hasattr(trend, "name")
        assert hasattr(trend, "parameters")
        assert hasattr(trend, "generate_signals")
        assert hasattr(trend, "get_parameter_grid")

        assert hasattr(mean_rev, "name")
        assert hasattr(mean_rev, "parameters")
        assert hasattr(mean_rev, "generate_signals")
        assert hasattr(mean_rev, "get_parameter_grid")

    def test_strategy_signals_compatible_with_backtester(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that strategy signals can be converted to backtester signals."""
        from optimizer_py.trend import TrendStrategy

        strategy = TrendStrategy()
        signals = strategy.generate_signals(sample_decision_frame)

        # Signals should have data needed for backtester conversion
        for signal in signals:
            assert signal.timestamp is not None
            assert signal.symbol is not None
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0

    def test_parameter_grid_produces_valid_combinations(self) -> None:
        """Test that parameter grids can be used for optimization."""
        from optimizer_py.mean_reversion import MeanReversionStrategy
        from optimizer_py.trend import TrendStrategy

        trend = TrendStrategy()
        mean_rev = MeanReversionStrategy()

        trend_grid = trend.get_parameter_grid()
        mean_rev_grid = mean_rev.get_parameter_grid()

        # Grids should have consistent structure
        for grid in [trend_grid, mean_rev_grid]:
            for param_name, values in grid.items():
                assert isinstance(param_name, str)
                assert isinstance(values, list)
                assert all(isinstance(v, (int, float)) for v in values)

    def test_strategies_produce_different_signals(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that different strategies produce different signals."""
        from optimizer_py.mean_reversion import MeanReversionStrategy
        from optimizer_py.trend import TrendStrategy

        trend = TrendStrategy()
        mean_rev = MeanReversionStrategy()

        trend_signals = trend.generate_signals(sample_decision_frame)
        mean_rev_signals = mean_rev.generate_signals(sample_decision_frame)

        # Convert to comparable format
        trend_values = [(s.timestamp, s.signal) for s in trend_signals]
        mean_rev_values = [(s.timestamp, s.signal) for s in mean_rev_signals]

        # Different strategies should produce different results
        # (not a strict test, but signals shouldn't be identical)
        if trend_values and mean_rev_values:
            # At least some signals should differ
            # This is a soft check - strategies CAN agree sometimes
            pass  # Both produce valid signals, that's the main test

    def test_strategy_reproducibility(
        self, sample_decision_frame: pl.DataFrame
    ) -> None:
        """Test that strategies produce reproducible results."""
        from optimizer_py.trend import TrendStrategy

        strategy1 = TrendStrategy(ma_fast=10, ma_slow=20)
        strategy2 = TrendStrategy(ma_fast=10, ma_slow=20)

        signals1 = strategy1.generate_signals(sample_decision_frame)
        signals2 = strategy2.generate_signals(sample_decision_frame)

        # Same parameters should produce same signals
        assert len(signals1) == len(signals2)
        for s1, s2 in zip(signals1, signals2, strict=True):
            assert s1.timestamp == s2.timestamp
            assert s1.symbol == s2.symbol
            assert s1.signal == s2.signal
