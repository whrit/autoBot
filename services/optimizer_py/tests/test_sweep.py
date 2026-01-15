"""
Tests for Parameter Sweep Framework (T4.04).

Tests grid search and random search optimization for strategy parameters.
"""

from typing import Any

import polars as pl

from optimizer_py.strategies import StrategyFamily, StrategySignal
from optimizer_py.sweep import (
    ParameterSweep,
    SweepConfig,
    SweepResult,
    generate_parameter_combinations,
)


class MockStrategy(StrategyFamily):
    """Mock strategy for testing parameter sweep."""

    family_name: str = "mock"

    def __init__(self, lookback: int = 10, threshold: float = 0.5) -> None:
        """Initialize mock strategy."""
        self.lookback = lookback
        self.threshold = threshold

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        """Get default parameters."""
        return {"lookback": 10, "threshold": 0.5}

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """Generate mock signals based on parameters."""
        signals: list[StrategySignal] = []
        prices = decision_frame["close"].to_numpy()
        timestamps = decision_frame["timestamp"].to_list()
        symbol = decision_frame["symbol"][0]

        for i in range(self.lookback, len(prices)):
            # Simple momentum strategy for testing
            returns = (prices[i] - prices[i - self.lookback]) / prices[i - self.lookback]
            if returns > self.threshold / 100:
                signals.append(
                    StrategySignal(
                        timestamp=timestamps[i],
                        symbol=symbol,
                        direction=1,
                        strength=min(1.0, abs(returns) * 10),
                    )
                )
            elif returns < -self.threshold / 100:
                signals.append(
                    StrategySignal(
                        timestamp=timestamps[i],
                        symbol=symbol,
                        direction=-1,
                        strength=min(1.0, abs(returns) * 10),
                    )
                )

        return signals


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid={"lookback": [5, 10], "threshold": [0.5, 1.0]},
        )
        assert config.n_splits == 5
        assert config.metric == "sharpe"
        assert config.n_jobs == -1

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid={"lookback": [5, 10, 15]},
            n_splits=3,
            metric="sortino",
            n_jobs=4,
        )
        assert config.n_splits == 3
        assert config.metric == "sortino"
        assert config.n_jobs == 4


class TestSweepResult:
    """Tests for SweepResult dataclass."""

    def test_result_creation(self) -> None:
        """Test SweepResult creation."""
        result = SweepResult(
            best_params={"lookback": 10, "threshold": 0.5},
            best_score=1.5,
            all_results=[
                {"params": {"lookback": 5}, "score": 1.0},
                {"params": {"lookback": 10}, "score": 1.5},
            ],
            cv_scores=[1.2, 1.4, 1.6, 1.5, 1.3],
        )
        assert result.best_params["lookback"] == 10
        assert result.best_score == 1.5
        assert len(result.all_results) == 2
        assert len(result.cv_scores) == 5


class TestParameterCombinations:
    """Tests for parameter combination generation."""

    def test_generate_combinations_simple(self) -> None:
        """Test simple parameter grid expansion."""
        grid = {"a": [1, 2], "b": ["x", "y"]}
        combinations = generate_parameter_combinations(grid)
        assert len(combinations) == 4
        assert {"a": 1, "b": "x"} in combinations
        assert {"a": 2, "b": "y"} in combinations

    def test_generate_combinations_single_param(self) -> None:
        """Test single parameter grid."""
        grid = {"lookback": [5, 10, 15, 20]}
        combinations = generate_parameter_combinations(grid)
        assert len(combinations) == 4

    def test_generate_combinations_empty(self) -> None:
        """Test empty parameter grid."""
        grid: dict[str, list[Any]] = {}
        combinations = generate_parameter_combinations(grid)
        assert len(combinations) == 1
        assert combinations[0] == {}

    def test_generate_combinations_large(self) -> None:
        """Test larger parameter grid."""
        grid = {
            "lookback": [5, 10, 15],
            "threshold": [0.5, 1.0, 1.5],
            "multiplier": [1.0, 2.0],
        }
        combinations = generate_parameter_combinations(grid)
        assert len(combinations) == 3 * 3 * 2  # 18 combinations


class TestParameterSweep:
    """Tests for ParameterSweep class."""

    def test_sweep_initialization(
        self, sample_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test ParameterSweep initialization."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=small_parameter_grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        assert sweep.config.strategy_family == "mock"
        assert sweep.strategy_class == MockStrategy

    def test_grid_search_basic(
        self, trending_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test basic grid search execution."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=small_parameter_grid,
            n_splits=2,
            metric="sharpe",
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.grid_search(trending_decision_frame)

        assert isinstance(result, SweepResult)
        assert "lookback" in result.best_params
        assert "threshold" in result.best_params
        assert isinstance(result.best_score, float)
        assert len(result.all_results) == 4  # 2x2 grid

    def test_grid_search_returns_all_combinations(
        self, trending_decision_frame: pl.DataFrame
    ) -> None:
        """Test that grid search evaluates all parameter combinations."""
        grid = {
            "lookback": [5, 10, 15],
            "threshold": [0.5, 1.0],
        }
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.grid_search(trending_decision_frame)

        assert len(result.all_results) == 6  # 3x2 combinations

    def test_random_search_basic(
        self, trending_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test basic random search execution."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=small_parameter_grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.random_search(trending_decision_frame, n_iter=3)

        assert isinstance(result, SweepResult)
        assert "lookback" in result.best_params
        assert len(result.all_results) <= 3

    def test_random_search_respects_n_iter(
        self, trending_decision_frame: pl.DataFrame
    ) -> None:
        """Test that random search respects n_iter parameter."""
        grid = {
            "lookback": [5, 10, 15, 20, 25, 30],
            "threshold": [0.5, 1.0, 1.5, 2.0],
        }
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.random_search(trending_decision_frame, n_iter=5)

        assert len(result.all_results) <= 5

    def test_sweep_with_different_metrics(
        self, trending_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test sweep with different optimization metrics."""
        for metric in ["sharpe", "sortino", "total_return"]:
            config = SweepConfig(
                strategy_family="mock",
                parameter_grid=small_parameter_grid,
                n_splits=2,
                metric=metric,
            )
            sweep = ParameterSweep(
                strategy_class=MockStrategy,
                config=config,
            )
            result = sweep.grid_search(trending_decision_frame)
            assert isinstance(result.best_score, float)

    def test_sweep_cv_scores_populated(
        self, trending_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test that CV scores are populated for best parameters."""
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=small_parameter_grid,
            n_splits=3,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.grid_search(trending_decision_frame)

        assert len(result.cv_scores) > 0
        assert all(isinstance(s, (int, float)) for s in result.cv_scores)

    def test_sweep_result_ordered_by_score(
        self, trending_decision_frame: pl.DataFrame
    ) -> None:
        """Test that results are ordered by score (descending)."""
        grid = {
            "lookback": [5, 10, 15],
            "threshold": [0.5, 1.0],
        }
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.grid_search(trending_decision_frame)

        scores = [r["score"] for r in result.all_results]
        assert scores == sorted(scores, reverse=True)


class TestSweepEdgeCases:
    """Test edge cases for parameter sweep."""

    def test_single_parameter_value(self, trending_decision_frame: pl.DataFrame) -> None:
        """Test sweep with single parameter value."""
        grid = {"lookback": [10], "threshold": [0.5]}
        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=grid,
            n_splits=2,
        )
        sweep = ParameterSweep(
            strategy_class=MockStrategy,
            config=config,
        )
        result = sweep.grid_search(trending_decision_frame)

        assert len(result.all_results) == 1
        assert result.best_params == {"lookback": 10, "threshold": 0.5}

    def test_sweep_reproducibility(
        self, trending_decision_frame: pl.DataFrame, small_parameter_grid: dict[str, list[Any]]
    ) -> None:
        """Test that sweep results are reproducible with same seed."""
        import random

        config = SweepConfig(
            strategy_family="mock",
            parameter_grid=small_parameter_grid,
            n_splits=2,
        )

        random.seed(42)
        sweep1 = ParameterSweep(strategy_class=MockStrategy, config=config)
        result1 = sweep1.random_search(trending_decision_frame, n_iter=3)

        random.seed(42)
        sweep2 = ParameterSweep(strategy_class=MockStrategy, config=config)
        result2 = sweep2.random_search(trending_decision_frame, n_iter=3)

        assert result1.best_params == result2.best_params
