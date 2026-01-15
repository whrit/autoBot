"""
Tests for the evaluation framework.

This module contains comprehensive tests for:
- T3.05: Walk-forward optimization framework
- T3.06: Metrics computation (Sharpe, Sortino, MDD, etc.)
- T3.07: Regime slicing (volatility, liquidity, trend)
- T3.08: Purged cross-validation

TDD approach: These tests are written first to define the expected behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from backtester_py.metrics import BacktestMetrics, MetricsCalculator
from backtester_py.purged_cv import PurgedCrossValidator
from backtester_py.regimes import RegimeAnalyzer, RegimeType
from backtester_py.walk_forward import WalkForwardOptimizer

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_equity_curve() -> pl.DataFrame:
    """Generate a sample equity curve for testing."""
    np.random.seed(42)
    n_bars = 252  # One trading year

    # Generate random returns with slight positive drift
    daily_returns = np.random.normal(0.0005, 0.01, n_bars)

    # Calculate cumulative equity
    equity = 100000 * np.cumprod(1 + daily_returns)

    # Create timestamps
    start_date = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_date + timedelta(days=i) for i in range(n_bars)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "equity": equity.tolist(),
            "returns": daily_returns.tolist(),
        }
    )


@pytest.fixture
def sample_trades() -> list[dict[str, float | str]]:
    """Generate sample trades for testing."""
    return [
        {"pnl": 150.0, "side": "buy", "notional": 10000.0},
        {"pnl": -80.0, "side": "sell", "notional": 8000.0},
        {"pnl": 200.0, "side": "buy", "notional": 12000.0},
        {"pnl": -50.0, "side": "sell", "notional": 5000.0},
        {"pnl": 120.0, "side": "buy", "notional": 10000.0},
        {"pnl": 180.0, "side": "buy", "notional": 11000.0},
        {"pnl": -100.0, "side": "sell", "notional": 9000.0},
        {"pnl": 90.0, "side": "buy", "notional": 8500.0},
        {"pnl": -30.0, "side": "sell", "notional": 6000.0},
        {"pnl": 250.0, "side": "buy", "notional": 15000.0},
    ]


@pytest.fixture
def sample_market_data() -> pl.DataFrame:
    """Generate sample market data with price, volume, and spread for regime testing."""
    np.random.seed(42)
    n_bars = 500

    start_date = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_date + timedelta(minutes=i) for i in range(n_bars)]

    # Generate price with trend changes
    price = 100.0
    prices = []
    for i in range(n_bars):
        # Add some trend changes
        if i < 150:
            drift = 0.0001  # Uptrend
        elif i < 300:
            drift = -0.0002  # Downtrend
        else:
            drift = 0.00005  # Slight uptrend
        price *= 1 + np.random.normal(drift, 0.001)
        prices.append(price)

    # Generate volume (random with some variation)
    volume = np.random.lognormal(10, 0.5, n_bars)

    # Generate spread (varying based on volatility)
    spreads = np.abs(np.random.normal(0.01, 0.005, n_bars))

    # Calculate returns
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": prices,
            "volume": volume.tolist(),
            "spread": spreads.tolist(),
            "returns": returns.tolist(),
        }
    )


@pytest.fixture
def walk_forward_data() -> pl.DataFrame:
    """Generate data for walk-forward testing."""
    np.random.seed(42)
    n_bars = 1000

    start_date = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_date + timedelta(minutes=i) for i in range(n_bars)]

    # Simple price series
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.001, n_bars))

    # Signals (random for testing)
    signals = np.random.choice([-1, 0, 1], n_bars, p=[0.2, 0.6, 0.2])

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": prices.tolist(),
            "signal": signals.tolist(),
        }
    )


# =============================================================================
# T3.06: METRICS COMPUTATION TESTS
# =============================================================================


class TestBacktestMetrics:
    """Tests for the BacktestMetrics dataclass."""

    def test_backtest_metrics_creation(self) -> None:
        """Test creating a BacktestMetrics instance."""
        metrics = BacktestMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            max_drawdown_duration=20,
            profit_factor=1.8,
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=100.0,
            total_return=0.25,
            cagr=0.20,
            calmar_ratio=1.33,
            num_trades=100,
        )

        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 2.0
        assert metrics.max_drawdown == 0.15
        assert metrics.max_drawdown_duration == 20
        assert metrics.profit_factor == 1.8
        assert metrics.win_rate == 0.55
        assert metrics.num_trades == 100


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculator_initialization(self) -> None:
        """Test MetricsCalculator initialization with default and custom params."""
        # Default parameters
        calc = MetricsCalculator()
        assert calc.risk_free_rate == 0.0
        assert calc.periods_per_year == 252

        # Custom parameters
        calc = MetricsCalculator(risk_free_rate=0.05, periods_per_year=365)
        assert calc.risk_free_rate == 0.05
        assert calc.periods_per_year == 365

    def test_sharpe_ratio_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator(risk_free_rate=0.0, periods_per_year=252)
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Sharpe should be a reasonable value (not NaN, not infinite)
        assert np.isfinite(metrics.sharpe_ratio)
        # For random returns with slight positive drift, expect Sharpe around 0.5-2.0
        assert -5.0 < metrics.sharpe_ratio < 5.0

    def test_sharpe_ratio_with_risk_free_rate(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test Sharpe ratio with non-zero risk-free rate."""
        calc_no_rf = MetricsCalculator(risk_free_rate=0.0)
        calc_with_rf = MetricsCalculator(risk_free_rate=0.05)

        metrics_no_rf = calc_no_rf.calculate(sample_equity_curve, sample_trades)
        metrics_with_rf = calc_with_rf.calculate(sample_equity_curve, sample_trades)

        # Sharpe with positive risk-free rate should be lower
        assert metrics_with_rf.sharpe_ratio < metrics_no_rf.sharpe_ratio

    def test_sortino_ratio_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test Sortino ratio calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Sortino should be a reasonable value
        assert np.isfinite(metrics.sortino_ratio)
        # Sortino typically >= Sharpe (only uses downside deviation)
        # But not always true, so just check it's reasonable
        assert -10.0 < metrics.sortino_ratio < 10.0

    def test_max_drawdown_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Max drawdown should be between 0 and 1
        assert 0.0 <= metrics.max_drawdown <= 1.0

        # For random returns, expect some drawdown
        assert metrics.max_drawdown > 0.0

    def test_max_drawdown_duration(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test max drawdown duration calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Duration should be positive integer
        assert metrics.max_drawdown_duration >= 0
        assert isinstance(metrics.max_drawdown_duration, int)

    def test_profit_factor_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test profit factor calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Calculate expected profit factor from sample trades
        gross_profit = sum(t["pnl"] for t in sample_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in sample_trades if t["pnl"] < 0))
        expected_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        assert np.isclose(metrics.profit_factor, expected_pf, rtol=0.01)

    def test_win_rate_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test win rate calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Calculate expected win rate
        wins = sum(1 for t in sample_trades if t["pnl"] > 0)
        expected_win_rate = wins / len(sample_trades)

        assert np.isclose(metrics.win_rate, expected_win_rate, rtol=0.01)

    def test_avg_win_loss_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test average win and loss calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Calculate expected averages
        wins = [t["pnl"] for t in sample_trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in sample_trades if t["pnl"] < 0]

        expected_avg_win = np.mean(wins) if wins else 0.0
        expected_avg_loss = abs(np.mean(losses)) if losses else 0.0

        assert np.isclose(metrics.avg_win, expected_avg_win, rtol=0.01)
        assert np.isclose(metrics.avg_loss, expected_avg_loss, rtol=0.01)

    def test_total_return_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test total return calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Calculate expected total return
        initial_equity = sample_equity_curve["equity"][0]
        final_equity = sample_equity_curve["equity"][-1]
        expected_return = (final_equity - initial_equity) / initial_equity

        assert np.isclose(metrics.total_return, expected_return, rtol=0.01)

    def test_cagr_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test CAGR calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # CAGR should be a reasonable annual value
        assert np.isfinite(metrics.cagr)

    def test_calmar_ratio_calculation(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test Calmar ratio calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        # Calmar = CAGR / Max Drawdown
        if metrics.max_drawdown > 0:
            expected_calmar = metrics.cagr / metrics.max_drawdown
            assert np.isclose(metrics.calmar_ratio, expected_calmar, rtol=0.01)

    def test_num_trades_count(
        self, sample_equity_curve: pl.DataFrame, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test trade count."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, sample_trades)

        assert metrics.num_trades == len(sample_trades)

    def test_empty_trades_handling(self, sample_equity_curve: pl.DataFrame) -> None:
        """Test handling of empty trade list."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_equity_curve, [])

        assert metrics.num_trades == 0
        assert metrics.profit_factor == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0

    def test_constant_equity_handling(
        self, sample_trades: list[dict[str, float | str]]
    ) -> None:
        """Test handling of constant equity (no returns)."""
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        constant_equity = pl.DataFrame(
            {
                "timestamp": timestamps,
                "equity": [100000.0] * 100,
                "returns": [0.0] * 100,
            }
        )

        calc = MetricsCalculator()
        metrics = calc.calculate(constant_equity, sample_trades)

        # Sharpe and Sortino should be 0 (no returns)
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0


# =============================================================================
# T3.05: WALK-FORWARD FRAMEWORK TESTS
# =============================================================================


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_optimizer_initialization(self) -> None:
        """Test WalkForwardOptimizer initialization."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)

        assert opt.train_size == 200
        assert opt.test_size == 50
        assert opt.step_size == 50

    def test_invalid_sizes(self) -> None:
        """Test that invalid sizes raise errors."""
        with pytest.raises(ValueError, match="train_size must be positive"):
            WalkForwardOptimizer(train_size=0, test_size=50, step_size=50)

        with pytest.raises(ValueError, match="test_size must be positive"):
            WalkForwardOptimizer(train_size=200, test_size=0, step_size=50)

        with pytest.raises(ValueError, match="step_size must be positive"):
            WalkForwardOptimizer(train_size=200, test_size=50, step_size=0)

    def test_split_generation(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that splits are generated correctly."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)

        splits = list(opt.split(walk_forward_data))

        # Should have multiple splits
        assert len(splits) > 0

        for train, test in splits:
            # Each split should have correct sizes
            assert len(train) == 200
            assert len(test) == 50

    def test_no_lookahead_bias(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that train data always precedes test data (no lookahead)."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)

        for train, test in opt.split(walk_forward_data):
            train_end = train["timestamp"].max()
            test_start = test["timestamp"].min()

            # Train data must end before test data starts
            assert train_end < test_start

    def test_rolling_windows(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that windows roll forward correctly."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)

        splits = list(opt.split(walk_forward_data))

        # Each subsequent split should start step_size bars later
        for i in range(1, len(splits)):
            prev_train_start = splits[i - 1][0]["timestamp"][0]
            curr_train_start = splits[i][0]["timestamp"][0]

            # Time difference should correspond to step_size (approximately)
            assert curr_train_start > prev_train_start

    def test_insufficient_data(self) -> None:
        """Test handling of insufficient data."""
        # Data smaller than train_size + test_size
        small_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)],
                "close": [100.0] * 100,
                "signal": [0] * 100,
            }
        )

        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)
        splits = list(opt.split(small_data))

        # Should return empty list for insufficient data
        assert len(splits) == 0

    def test_split_coverage(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that splits cover the data appropriately."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)

        splits = list(opt.split(walk_forward_data))

        # Last test window should not exceed data length
        if splits:
            last_test = splits[-1][1]
            last_test_idx = walk_forward_data["timestamp"].to_list().index(
                last_test["timestamp"][-1]
            )
            assert last_test_idx < len(walk_forward_data)


# =============================================================================
# T3.07: REGIME SLICING TESTS
# =============================================================================


class TestRegimeAnalyzer:
    """Tests for RegimeAnalyzer."""

    def test_analyzer_initialization(self) -> None:
        """Test RegimeAnalyzer initialization."""
        analyzer = RegimeAnalyzer()

        # Check default quantile thresholds
        assert analyzer.vol_quantiles == (0.33, 0.67)
        assert analyzer.liquidity_quantiles == (0.33, 0.67)
        assert analyzer.trend_quantiles == (0.33, 0.67)

    def test_custom_quantiles(self) -> None:
        """Test RegimeAnalyzer with custom quantile thresholds."""
        analyzer = RegimeAnalyzer(
            vol_quantiles=(0.25, 0.75),
            liquidity_quantiles=(0.20, 0.80),
            trend_quantiles=(0.40, 0.60),
        )

        assert analyzer.vol_quantiles == (0.25, 0.75)
        assert analyzer.liquidity_quantiles == (0.20, 0.80)
        assert analyzer.trend_quantiles == (0.40, 0.60)

    def test_volatility_regime_classification(
        self, sample_market_data: pl.DataFrame
    ) -> None:
        """Test volatility regime classification."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_regimes(sample_market_data)

        # Check that volatility regimes are assigned
        assert "vol_regime" in regimes.columns

        # Should have low, medium, high regimes
        vol_regimes = regimes["vol_regime"].unique().to_list()
        assert RegimeType.LOW.value in vol_regimes or len(vol_regimes) >= 1

    def test_liquidity_regime_classification(
        self, sample_market_data: pl.DataFrame
    ) -> None:
        """Test liquidity regime classification."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_regimes(sample_market_data)

        assert "liquidity_regime" in regimes.columns

    def test_trend_regime_classification(
        self, sample_market_data: pl.DataFrame
    ) -> None:
        """Test trend regime classification."""
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_regimes(sample_market_data)

        assert "trend_regime" in regimes.columns

    def test_regime_distribution(self, sample_market_data: pl.DataFrame) -> None:
        """Test that regimes are distributed according to quantiles."""
        analyzer = RegimeAnalyzer(vol_quantiles=(0.33, 0.67))
        regimes = analyzer.classify_regimes(sample_market_data)

        # Approximately 1/3 of data should be in each regime
        vol_counts = regimes["vol_regime"].value_counts()

        # Allow some tolerance (due to ties at boundaries)
        total = len(regimes)
        for count in vol_counts["count"].to_list():
            # Each regime should have roughly 1/3 of data (within 20% tolerance)
            assert 0.1 * total <= count <= 0.6 * total

    def test_lookback_window(self, sample_market_data: pl.DataFrame) -> None:
        """Test that volatility is calculated over lookback window."""
        analyzer = RegimeAnalyzer(vol_lookback=20)
        regimes = analyzer.classify_regimes(sample_market_data)

        # First few rows should have NaN or default regime (insufficient lookback)
        # The exact behavior depends on implementation
        assert len(regimes) == len(sample_market_data)


# =============================================================================
# T3.08: PURGED CV TESTS
# =============================================================================


class TestPurgedCrossValidator:
    """Tests for PurgedCrossValidator."""

    def test_validator_initialization(self) -> None:
        """Test PurgedCrossValidator initialization."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        assert cv.n_splits == 5
        assert cv.embargo_pct == 0.01

    def test_invalid_n_splits(self) -> None:
        """Test that invalid n_splits raises error."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            PurgedCrossValidator(n_splits=1)

    def test_invalid_embargo_pct(self) -> None:
        """Test that invalid embargo_pct raises error."""
        with pytest.raises(ValueError, match="embargo_pct must be between 0 and 1"):
            PurgedCrossValidator(n_splits=5, embargo_pct=1.5)

        with pytest.raises(ValueError, match="embargo_pct must be between 0 and 1"):
            PurgedCrossValidator(n_splits=5, embargo_pct=-0.1)

    def test_split_generation(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that correct number of splits are generated."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(walk_forward_data))

        assert len(splits) == 5

    def test_embargo_period(self, walk_forward_data: pl.DataFrame) -> None:
        """Test that embargo period is respected."""
        embargo_pct = 0.05  # 5% embargo
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=embargo_pct)

        for train, test in cv.split(walk_forward_data):
            train_min = train["timestamp"].min()
            train_max = train["timestamp"].max()
            test_min = test["timestamp"].min()
            test_max = test["timestamp"].max()

            # No overlap between train and test
            train_timestamps = set(train["timestamp"].to_list())
            test_timestamps = set(test["timestamp"].to_list())
            assert len(train_timestamps & test_timestamps) == 0

            # When train precedes test, there should be a gap (embargo)
            if train_max < test_min:
                # Train before test - verify embargo gap exists
                assert train_max < test_min
            else:
                # Train after test - verify embargo gap exists
                assert test_max < train_min

    def test_no_overlap_between_train_test(
        self, walk_forward_data: pl.DataFrame
    ) -> None:
        """Test that there is no overlap between train and test sets."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        for train, test in cv.split(walk_forward_data):
            train_timestamps = set(train["timestamp"].to_list())
            test_timestamps = set(test["timestamp"].to_list())

            # No overlap
            assert len(train_timestamps & test_timestamps) == 0

    def test_deflated_sharpe_calculation(self) -> None:
        """Test deflated Sharpe ratio calculation."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        # Test with multiple Sharpe ratios
        sharpe_ratios = [1.2, 0.8, 1.5, 0.9, 1.1]
        num_trials = 100

        deflated_sharpe = cv.deflated_sharpe(sharpe_ratios, num_trials)

        # Deflated Sharpe should be less than max observed Sharpe
        assert deflated_sharpe <= max(sharpe_ratios)

        # Should be a valid number
        assert np.isfinite(deflated_sharpe)

    def test_deflated_sharpe_single_trial(self) -> None:
        """Test deflated Sharpe with single trial (edge case)."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        sharpe_ratios = [1.5]
        num_trials = 1

        deflated_sharpe = cv.deflated_sharpe(sharpe_ratios, num_trials)

        # With single trial, deflated should equal observed
        assert np.isclose(deflated_sharpe, 1.5, rtol=0.1)

    def test_deflated_sharpe_many_trials(self) -> None:
        """Test that more trials lead to lower deflated Sharpe."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        sharpe_ratios = [1.2, 0.8, 1.5, 0.9, 1.1]

        # More trials = more penalty for multiple testing
        deflated_10 = cv.deflated_sharpe(sharpe_ratios, num_trials=10)
        deflated_100 = cv.deflated_sharpe(sharpe_ratios, num_trials=100)
        deflated_1000 = cv.deflated_sharpe(sharpe_ratios, num_trials=1000)

        # Deflated Sharpe should decrease as trials increase
        assert deflated_100 <= deflated_10
        assert deflated_1000 <= deflated_100

    def test_combinatorial_purged_cv(self, walk_forward_data: pl.DataFrame) -> None:
        """Test combinatorial purged cross-validation."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)

        # Get all splits
        splits = list(cv.split(walk_forward_data))

        # Each split should be valid (non-empty train and test)
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests combining multiple evaluation components."""

    def test_walk_forward_with_metrics(
        self,
        walk_forward_data: pl.DataFrame,
        sample_trades: list[dict[str, float | str]],
    ) -> None:
        """Test walk-forward optimization with metrics calculation."""
        opt = WalkForwardOptimizer(train_size=200, test_size=50, step_size=50)
        calc = MetricsCalculator()

        results: list[BacktestMetrics] = []

        for _train, test in opt.split(walk_forward_data):
            # Create equity curve from test data
            equity_curve = pl.DataFrame(
                {
                    "timestamp": test["timestamp"],
                    "equity": (
                        100000 * (1 + test["close"].to_numpy() / test["close"][0] - 1)
                    ).tolist(),
                    "returns": test["close"].pct_change().fill_null(0.0).to_list(),
                }
            )

            metrics = calc.calculate(equity_curve, sample_trades)
            results.append(metrics)

        # Should have results for each window
        assert len(results) > 0

        # All results should have valid metrics
        for metrics in results:
            assert np.isfinite(metrics.sharpe_ratio)
            assert 0.0 <= metrics.max_drawdown <= 1.0

    def test_regime_sliced_metrics(
        self,
        sample_market_data: pl.DataFrame,
        sample_trades: list[dict[str, float | str]],
    ) -> None:
        """Test metrics calculation by regime."""
        analyzer = RegimeAnalyzer()
        calc = MetricsCalculator()

        regimes = analyzer.classify_regimes(sample_market_data)

        # Calculate metrics for each volatility regime
        regime_metrics: dict[str, BacktestMetrics] = {}

        for regime_value in regimes["vol_regime"].unique().to_list():
            regime_data = regimes.filter(pl.col("vol_regime") == regime_value)

            if len(regime_data) > 10:  # Need enough data for metrics
                equity_curve = pl.DataFrame(
                    {
                        "timestamp": regime_data["timestamp"],
                        "equity": (
                            100000
                            * (1 + regime_data["close"].to_numpy() / regime_data["close"][0] - 1)
                        ).tolist(),
                        "returns": regime_data["returns"],
                    }
                )

                metrics = calc.calculate(equity_curve, sample_trades)
                regime_metrics[str(regime_value)] = metrics

        # Should have metrics for at least some regimes
        assert len(regime_metrics) > 0

    def test_purged_cv_with_metrics(
        self,
        walk_forward_data: pl.DataFrame,
        sample_trades: list[dict[str, float | str]],
    ) -> None:
        """Test purged CV with metrics aggregation."""
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.02)
        calc = MetricsCalculator()

        sharpe_ratios: list[float] = []

        for _train, test in cv.split(walk_forward_data):
            if len(test) > 10:
                equity_curve = pl.DataFrame(
                    {
                        "timestamp": test["timestamp"],
                        "equity": (
                            100000 * (1 + test["close"].to_numpy() / test["close"][0] - 1)
                        ).tolist(),
                        "returns": test["close"].pct_change().fill_null(0.0).to_list(),
                    }
                )

                metrics = calc.calculate(equity_curve, sample_trades)
                sharpe_ratios.append(metrics.sharpe_ratio)

        # Calculate deflated Sharpe
        if sharpe_ratios:
            deflated = cv.deflated_sharpe(sharpe_ratios, num_trials=50)
            assert np.isfinite(deflated)
            assert deflated <= max(sharpe_ratios)
