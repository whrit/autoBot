"""
Parameter Sweep Framework (T4.04).

Provides grid search and random search optimization for strategy parameters.

Key Components:
    - SweepConfig: Configuration for parameter sweep
    - SweepResult: Results from parameter sweep optimization
    - ParameterSweep: Main class for running parameter optimization

Example:
    >>> config = SweepConfig(
    ...     strategy_family="trend",
    ...     parameter_grid={"lookback": [10, 20, 50], "threshold": [0.5, 1.0]},
    ...     n_splits=5,
    ...     metric="sharpe"
    ... )
    >>> sweep = ParameterSweep(strategy_class=TrendStrategy, config=config)
    >>> result = sweep.grid_search(decision_frame)
    >>> print(f"Best params: {result.best_params}")
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from optimizer_py.strategies import StrategyFamily, StrategySignal


@dataclass
class SweepConfig:
    """
    Configuration for parameter sweep optimization.

    Attributes:
        strategy_family: Name of the strategy family being optimized
        parameter_grid: Dictionary mapping parameter names to lists of values
        n_splits: Number of cross-validation splits (default: 5)
        metric: Optimization metric ("sharpe", "sortino", "total_return")
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """

    strategy_family: str
    parameter_grid: dict[str, list[Any]]
    n_splits: int = 5
    metric: str = "sharpe"
    n_jobs: int = -1


@dataclass
class SweepResult:
    """
    Results from parameter sweep optimization.

    Attributes:
        best_params: Parameters that achieved the best score
        best_score: Best optimization metric score achieved
        all_results: All parameter combinations and their scores
        cv_scores: Cross-validation scores for best parameters
    """

    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]] = field(default_factory=list)
    cv_scores: list[float] = field(default_factory=list)


def generate_parameter_combinations(
    parameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """
    Generate all parameter combinations from a parameter grid.

    Args:
        parameter_grid: Dictionary mapping parameter names to lists of values

    Returns:
        List of dictionaries, each containing one parameter combination

    Example:
        >>> grid = {"a": [1, 2], "b": ["x", "y"]}
        >>> combinations = generate_parameter_combinations(grid)
        >>> len(combinations)
        4
    """
    if not parameter_grid:
        return [{}]

    keys = list(parameter_grid.keys())
    values = [parameter_grid[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=True)))

    return combinations


class ParameterSweep[T: StrategyFamily]:
    """
    Parameter sweep optimizer for strategy families.

    Provides grid search and random search for finding optimal
    strategy parameters based on backtested performance metrics.

    Example:
        >>> sweep = ParameterSweep(strategy_class=TrendStrategy, config=config)
        >>> result = sweep.grid_search(decision_frame)
        >>> print(f"Best Sharpe: {result.best_score:.2f}")
    """

    def __init__(
        self,
        strategy_class: type[T],
        config: SweepConfig,
    ) -> None:
        """
        Initialize parameter sweep.

        Args:
            strategy_class: Strategy class to instantiate with parameters
            config: Sweep configuration
        """
        self.strategy_class = strategy_class
        self.config = config

    def grid_search(self, decision_frame: pl.DataFrame) -> SweepResult:
        """
        Perform exhaustive grid search over all parameter combinations.

        Args:
            decision_frame: Market data for backtesting strategies

        Returns:
            SweepResult with best parameters and all results
        """
        combinations = generate_parameter_combinations(self.config.parameter_grid)
        return self._evaluate_combinations(decision_frame, combinations)

    def random_search(
        self,
        decision_frame: pl.DataFrame,
        n_iter: int,
    ) -> SweepResult:
        """
        Perform random search over parameter space.

        Args:
            decision_frame: Market data for backtesting strategies
            n_iter: Number of random parameter combinations to try

        Returns:
            SweepResult with best parameters and all results
        """
        all_combinations = generate_parameter_combinations(self.config.parameter_grid)

        # Sample up to n_iter combinations
        if len(all_combinations) <= n_iter:
            combinations = all_combinations
        else:
            combinations = random.sample(all_combinations, n_iter)

        return self._evaluate_combinations(decision_frame, combinations)

    def _evaluate_combinations(
        self,
        decision_frame: pl.DataFrame,
        combinations: list[dict[str, Any]],
    ) -> SweepResult:
        """
        Evaluate a list of parameter combinations.

        Args:
            decision_frame: Market data for backtesting
            combinations: List of parameter dictionaries to evaluate

        Returns:
            SweepResult with all evaluations and best result
        """
        results: list[dict[str, Any]] = []

        for params in combinations:
            cv_scores = self._cross_validate(decision_frame, params)
            mean_score = float(np.mean(cv_scores)) if cv_scores else 0.0

            results.append({
                "params": params,
                "score": mean_score,
                "cv_scores": cv_scores,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            return SweepResult(
                best_params={},
                best_score=0.0,
                all_results=[],
                cv_scores=[],
            )

        best = results[0]
        return SweepResult(
            best_params=best["params"],
            best_score=best["score"],
            all_results=results,
            cv_scores=best.get("cv_scores", []),
        )

    def _cross_validate(
        self,
        decision_frame: pl.DataFrame,
        params: dict[str, Any],
    ) -> list[float]:
        """
        Perform time-series cross-validation for given parameters.

        Uses walk-forward validation to avoid look-ahead bias.

        Args:
            decision_frame: Full decision frame for cross-validation
            params: Strategy parameters to evaluate

        Returns:
            List of metric scores for each fold
        """
        n_rows = len(decision_frame)
        if n_rows < self.config.n_splits + 1:
            # Not enough data for cross-validation
            return [self._evaluate_fold(decision_frame, params)]

        # Calculate fold sizes for walk-forward CV
        fold_size = n_rows // (self.config.n_splits + 1)
        scores: list[float] = []

        for fold_idx in range(self.config.n_splits):
            # Train on data up to fold, test on fold
            train_end = fold_size * (fold_idx + 1)
            test_end = train_end + fold_size

            if test_end > n_rows:
                break

            # Get test fold data
            test_data = decision_frame.slice(train_end, fold_size)

            if len(test_data) > 0:
                score = self._evaluate_fold(test_data, params)
                scores.append(score)

        return scores if scores else [0.0]

    def _evaluate_fold(
        self,
        fold_data: pl.DataFrame,
        params: dict[str, Any],
    ) -> float:
        """
        Evaluate strategy performance on a single data fold.

        Args:
            fold_data: Decision frame for evaluation
            params: Strategy parameters

        Returns:
            Performance metric score
        """
        try:
            # Instantiate strategy with parameters
            strategy = self.strategy_class(**params)
            signals = strategy.generate_signals(fold_data)

            # Calculate performance metric from signals
            return self._calculate_metric(fold_data, signals)
        except Exception:
            return 0.0

    def _calculate_metric(
        self,
        decision_frame: pl.DataFrame,
        signals: list[StrategySignal],
    ) -> float:
        """
        Calculate performance metric from signals.

        This is a simplified metric calculation that computes
        returns based on signal directions and price changes.

        Args:
            decision_frame: Market data
            signals: Generated trading signals

        Returns:
            Performance metric score
        """
        if not signals or len(decision_frame) < 2:
            return 0.0

        # Get price data
        prices = decision_frame["close"].to_numpy()
        timestamps = decision_frame["timestamp"].to_list()

        # Create timestamp to index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # Calculate returns based on signals
        returns: list[float] = []
        position: float = 0.0

        for signal in signals:
            if signal.timestamp not in ts_to_idx:
                continue

            idx = ts_to_idx[signal.timestamp]
            if idx + 1 >= len(prices):
                continue

            # Calculate return from this position
            price_return = (prices[idx + 1] - prices[idx]) / prices[idx]
            signal_return = position * price_return
            returns.append(signal_return)

            # Update position
            position = signal.direction * signal.strength

        if not returns:
            return 0.0

        returns_array = np.array(returns)

        if self.config.metric == "sharpe":
            return self._calculate_sharpe(returns_array)
        elif self.config.metric == "sortino":
            return self._calculate_sortino(returns_array)
        elif self.config.metric == "total_return":
            return float(np.sum(returns_array))
        else:
            return self._calculate_sharpe(returns_array)

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        # Annualize assuming daily returns
        return float(mean_return / std_return * np.sqrt(252))

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float("inf") if mean_return > 0 else 0.0
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        return float(mean_return / downside_std * np.sqrt(252))
