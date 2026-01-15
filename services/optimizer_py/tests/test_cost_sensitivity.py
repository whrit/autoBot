"""
Tests for cost sensitivity module (T4.10).

Tests cover:
- Running sensitivity tests under various cost scenarios
- Computing breakeven cost levels
- Calculating robustness scores
- Strategy degradation under increased costs
- Configuration of cost multipliers
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from optimizer_py.cost_sensitivity import (
    CostScenario,
    CostSensitivityConfig,
    CostSensitivityTester,
    SensitivityResult,
)

# --- Mock Strategy Protocol for Testing ---


class MockBacktestResult:
    """Mock backtest result for testing."""

    def __init__(
        self,
        total_pnl: float,
        total_trades: int,
        sharpe: float = 1.0,
        fills: list[dict] | None = None,
    ) -> None:
        self.total_pnl = total_pnl
        self.total_trades = total_trades
        self.sharpe = sharpe
        self.fills = fills or []


class MockBacktester:
    """Mock backtester for cost sensitivity testing."""

    def __init__(
        self,
        base_pnl: float = 5000.0,
        cost_sensitivity: float = 0.5,
    ) -> None:
        """
        Initialize mock backtester.

        Args:
            base_pnl: Base P&L at 1x cost multiplier
            cost_sensitivity: How much P&L degrades per cost multiplier
                             (0.5 means 50% of extra cost eats into P&L)
        """
        self.base_pnl = base_pnl
        self.cost_sensitivity = cost_sensitivity
        self.run_count = 0

    def run(
        self,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
        spread_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
    ) -> MockBacktestResult:
        """
        Run mock backtest with cost adjustments.

        Returns degraded P&L based on cost multipliers.
        """
        self.run_count += 1

        # Calculate cost impact
        cost_factor = (spread_multiplier + slippage_multiplier) / 2
        pnl_degradation = (cost_factor - 1.0) * self.base_pnl * self.cost_sensitivity

        adjusted_pnl = self.base_pnl - pnl_degradation

        # Sharpe degrades proportionally
        base_sharpe = 1.5
        adjusted_sharpe = base_sharpe * (adjusted_pnl / self.base_pnl)

        return MockBacktestResult(
            total_pnl=adjusted_pnl,
            total_trades=50,
            sharpe=adjusted_sharpe,
            fills=[{"pnl": adjusted_pnl / 50}] * 50,
        )


@dataclass
class MockStrategy:
    """Mock strategy for testing."""

    strategy_id: str
    family: str
    params: dict = field(default_factory=dict)


# --- Test Classes ---


class TestCostScenario:
    """Tests for CostScenario dataclass."""

    def test_create_cost_scenario(self) -> None:
        """Test creating a CostScenario instance."""
        scenario = CostScenario(
            name="high_stress",
            spread_multiplier=2.0,
            slippage_multiplier=1.5,
        )

        assert scenario.name == "high_stress"
        assert scenario.spread_multiplier == 2.0
        assert scenario.slippage_multiplier == 1.5

    def test_default_scenario_values(self) -> None:
        """Test default scenario values."""
        scenario = CostScenario(name="default")

        assert scenario.spread_multiplier == 1.0
        assert scenario.slippage_multiplier == 1.0


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_create_sensitivity_result(self) -> None:
        """Test creating a SensitivityResult instance."""
        result = SensitivityResult(
            strategy_id="trend_ma_20",
            cost_scenario="high_stress",
            spread_multiplier=2.0,
            slippage_multiplier=1.5,
            sharpe=0.8,
            pnl=3000.0,
            pnl_change_pct=-0.40,
            still_profitable=True,
        )

        assert result.strategy_id == "trend_ma_20"
        assert result.cost_scenario == "high_stress"
        assert result.spread_multiplier == 2.0
        assert result.slippage_multiplier == 1.5
        assert result.sharpe == 0.8
        assert result.pnl == 3000.0
        assert result.pnl_change_pct == -0.40
        assert result.still_profitable is True

    def test_sensitivity_result_unprofitable(self) -> None:
        """Test result with negative P&L."""
        result = SensitivityResult(
            strategy_id="weak_strategy",
            cost_scenario="extreme",
            spread_multiplier=3.0,
            slippage_multiplier=2.5,
            sharpe=-0.5,
            pnl=-1000.0,
            pnl_change_pct=-1.20,
            still_profitable=False,
        )

        assert result.pnl < 0
        assert result.still_profitable is False


class TestCostSensitivityConfig:
    """Tests for CostSensitivityConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CostSensitivityConfig()

        assert config.base_spread_bps == 1.0
        assert 0.5 in config.spread_multipliers
        assert 1.0 in config.spread_multipliers
        assert 2.0 in config.spread_multipliers
        assert len(config.slippage_multipliers) > 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CostSensitivityConfig(
            base_spread_bps=2.0,
            spread_multipliers=[1.0, 1.5, 2.0],
            slippage_multipliers=[1.0, 1.5],
        )

        assert config.base_spread_bps == 2.0
        assert config.spread_multipliers == [1.0, 1.5, 2.0]
        assert config.slippage_multipliers == [1.0, 1.5]

    def test_generate_scenarios(self) -> None:
        """Test generating cost scenarios from config."""
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0],
            slippage_multipliers=[1.0, 1.5],
        )

        scenarios = config.generate_scenarios()

        # Should generate all combinations
        assert len(scenarios) == 4  # 2 x 2 combinations


class TestCostSensitivityTester:
    """Tests for CostSensitivityTester class."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default configuration."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        assert tester.backtester is backtester
        assert tester.config.base_spread_bps == 1.0

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        backtester = MockBacktester()
        config = CostSensitivityConfig(base_spread_bps=2.0)
        tester = CostSensitivityTester(backtester, config)

        assert tester.config.base_spread_bps == 2.0


class TestRunSensitivityTest:
    """Tests for running sensitivity tests."""

    def test_sensitivity_under_higher_costs(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that P&L degrades under higher costs."""
        backtester = MockBacktester(base_pnl=5000.0, cost_sensitivity=0.5)
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0],
            slippage_multipliers=[1.0],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        # Find results for 1x and 2x spread
        base_result = next(r for r in results if r.spread_multiplier == 1.0)
        high_result = next(r for r in results if r.spread_multiplier == 2.0)

        # Higher costs should reduce P&L
        assert high_result.pnl < base_result.pnl

    def test_sensitivity_returns_all_scenarios(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that all configured scenarios are tested."""
        backtester = MockBacktester()
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 1.5, 2.0],
            slippage_multipliers=[1.0, 1.5],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        # Should have one result per scenario combination
        assert len(results) == 6  # 3 x 2 combinations

    def test_sensitivity_captures_profitability(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that profitability status is correctly captured."""
        # Use high cost sensitivity to make strategy unprofitable at high costs
        backtester = MockBacktester(base_pnl=2000.0, cost_sensitivity=1.5)
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 3.0],
            slippage_multipliers=[1.0],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        base_result = next(r for r in results if r.spread_multiplier == 1.0)
        extreme_result = next(r for r in results if r.spread_multiplier == 3.0)

        # Base case should be profitable
        assert base_result.still_profitable is True
        # Extreme case might become unprofitable
        assert extreme_result.pnl < base_result.pnl

    def test_sensitivity_calculates_pnl_change_pct(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that P&L change percentage is correctly calculated."""
        backtester = MockBacktester(base_pnl=5000.0, cost_sensitivity=0.5)
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0],
            slippage_multipliers=[1.0],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        # Base case should have 0% change
        base_result = next(r for r in results if r.spread_multiplier == 1.0)
        assert abs(base_result.pnl_change_pct) < 0.01

        # Higher cost should have negative change
        high_result = next(r for r in results if r.spread_multiplier == 2.0)
        assert high_result.pnl_change_pct < 0


class TestBreakevenCostCalculation:
    """Tests for breakeven cost calculation."""

    def test_compute_breakeven_costs(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test computation of breakeven cost levels."""
        backtester = MockBacktester(base_pnl=5000.0, cost_sensitivity=0.5)
        tester = CostSensitivityTester(backtester)

        strategy = MockStrategy(strategy_id="test", family="trend")

        breakeven = tester.compute_breakeven_costs(
            strategy, sample_decision_frame, mock_quotes
        )

        # Should return breakeven multipliers
        assert "spread_breakeven" in breakeven
        assert "slippage_breakeven" in breakeven
        assert breakeven["spread_breakeven"] > 1.0  # Should be above base

    def test_breakeven_costs_sensitive_strategy(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test breakeven costs for a cost-sensitive strategy."""
        # Strategy that becomes unprofitable at 2x costs
        backtester = MockBacktester(base_pnl=2000.0, cost_sensitivity=1.0)
        tester = CostSensitivityTester(backtester)

        strategy = MockStrategy(strategy_id="sensitive", family="trend")

        breakeven = tester.compute_breakeven_costs(
            strategy, sample_decision_frame, mock_quotes
        )

        # Breakeven should be lower for more sensitive strategy
        assert breakeven["spread_breakeven"] < 5.0

    def test_breakeven_includes_combined_multiplier(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that combined breakeven multiplier is computed."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        strategy = MockStrategy(strategy_id="test", family="trend")

        breakeven = tester.compute_breakeven_costs(
            strategy, sample_decision_frame, mock_quotes
        )

        # Should have combined multiplier
        assert "combined_breakeven" in breakeven


class TestRobustnessScore:
    """Tests for robustness score calculation."""

    def test_robustness_score_calculation(self) -> None:
        """Test basic robustness score calculation."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        # Create results with varying profitability
        results = [
            SensitivityResult(
                strategy_id="test",
                cost_scenario="base",
                spread_multiplier=1.0,
                slippage_multiplier=1.0,
                sharpe=1.5,
                pnl=5000.0,
                pnl_change_pct=0.0,
                still_profitable=True,
            ),
            SensitivityResult(
                strategy_id="test",
                cost_scenario="high",
                spread_multiplier=2.0,
                slippage_multiplier=1.5,
                sharpe=0.8,
                pnl=2500.0,
                pnl_change_pct=-0.50,
                still_profitable=True,
            ),
            SensitivityResult(
                strategy_id="test",
                cost_scenario="extreme",
                spread_multiplier=3.0,
                slippage_multiplier=2.0,
                sharpe=-0.2,
                pnl=-500.0,
                pnl_change_pct=-1.10,
                still_profitable=False,
            ),
        ]

        score = tester.robustness_score(results)

        # Score should be between 0 and 1
        assert 0 <= score <= 1

    def test_robustness_score_all_profitable(self) -> None:
        """Test robustness score when all scenarios are profitable."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        results = [
            SensitivityResult(
                strategy_id="robust",
                cost_scenario=f"scenario_{i}",
                spread_multiplier=1.0 + i * 0.5,
                slippage_multiplier=1.0 + i * 0.25,
                sharpe=1.5 - i * 0.2,
                pnl=5000.0 - i * 500,
                pnl_change_pct=-i * 0.1,
                still_profitable=True,
            )
            for i in range(5)
        ]

        score = tester.robustness_score(results)

        # All profitable = high robustness
        assert score >= 0.7

    def test_robustness_score_none_profitable(self) -> None:
        """Test robustness score when no scenarios are profitable."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        results = [
            SensitivityResult(
                strategy_id="weak",
                cost_scenario=f"scenario_{i}",
                spread_multiplier=1.0 + i * 0.5,
                slippage_multiplier=1.0,
                sharpe=-0.5 - i * 0.1,
                pnl=-1000.0 - i * 500,
                pnl_change_pct=-0.2 - i * 0.1,
                still_profitable=False,
            )
            for i in range(5)
        ]

        score = tester.robustness_score(results)

        # None profitable = low robustness
        assert score <= 0.35

    def test_robustness_considers_pnl_degradation(self) -> None:
        """Test that robustness score considers P&L degradation rate."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        # Strategy A: Gradual degradation
        results_gradual = [
            SensitivityResult(
                strategy_id="gradual",
                cost_scenario=f"s{i}",
                spread_multiplier=1.0 + i * 0.5,
                slippage_multiplier=1.0,
                sharpe=1.5 - i * 0.1,
                pnl=5000.0 - i * 200,
                pnl_change_pct=-i * 0.04,
                still_profitable=True,
            )
            for i in range(5)
        ]

        # Strategy B: Sharp degradation
        results_sharp = [
            SensitivityResult(
                strategy_id="sharp",
                cost_scenario=f"s{i}",
                spread_multiplier=1.0 + i * 0.5,
                slippage_multiplier=1.0,
                sharpe=1.5 - i * 0.4,
                pnl=5000.0 - i * 1000,
                pnl_change_pct=-i * 0.20,
                still_profitable=5000.0 - i * 1000 > 0,
            )
            for i in range(5)
        ]

        score_gradual = tester.robustness_score(results_gradual)
        score_sharp = tester.robustness_score(results_sharp)

        # Gradual degradation should have higher robustness
        assert score_gradual > score_sharp


class TestStrategyDegradation:
    """Tests for strategy degradation under costs."""

    def test_strategy_degrades_gracefully(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that strategy performance degrades smoothly with costs."""
        backtester = MockBacktester(base_pnl=5000.0, cost_sensitivity=0.4)
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 1.5, 2.0, 2.5, 3.0],
            slippage_multipliers=[1.0],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        # Sort by spread multiplier
        sorted_results = sorted(results, key=lambda r: r.spread_multiplier)

        # Check monotonic degradation
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].pnl >= sorted_results[i + 1].pnl

    def test_sharpe_ratio_degradation(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that Sharpe ratio degrades with higher costs."""
        backtester = MockBacktester(base_pnl=5000.0, cost_sensitivity=0.5)
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0, 3.0],
            slippage_multipliers=[1.0],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        sorted_results = sorted(results, key=lambda r: r.spread_multiplier)

        # Sharpe should degrade with costs
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].sharpe >= sorted_results[i + 1].sharpe


class TestCostSensitivitySummary:
    """Tests for sensitivity summary methods."""

    def test_get_sensitivity_summary(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test getting a summary of sensitivity analysis."""
        backtester = MockBacktester()
        tester = CostSensitivityTester(backtester)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )
        summary = tester.get_sensitivity_summary(results)

        # Summary should include key metrics
        assert "strategy_id" in summary
        assert "base_pnl" in summary
        assert "worst_pnl" in summary
        assert "robustness_score" in summary
        assert "profitable_scenarios" in summary

    def test_get_cost_impact_matrix(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test getting a cost impact matrix."""
        backtester = MockBacktester()
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0],
            slippage_multipliers=[1.0, 1.5],
        )
        tester = CostSensitivityTester(backtester, config)

        strategy = MockStrategy(strategy_id="test", family="trend")

        results = tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )
        matrix = tester.get_cost_impact_matrix(results)

        # Matrix should be indexed by spread and slippage
        assert isinstance(matrix, dict)
        assert len(matrix) > 0
