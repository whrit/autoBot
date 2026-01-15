"""
Cost Sensitivity Testing Module for Strategy Optimization (T4.10).

Tests strategy robustness to variations in transaction costs (spreads and slippage).
Helps identify strategies that maintain profitability under adverse cost conditions.

Key Features:
- Run strategies under various cost scenarios (spread/slippage multipliers)
- Compute breakeven cost levels where strategy becomes unprofitable
- Calculate robustness scores based on cost sensitivity
- Generate cost impact matrices and summaries

Example usage:
    >>> from optimizer_py.cost_sensitivity import CostSensitivityTester, CostSensitivityConfig
    >>> config = CostSensitivityConfig(spread_multipliers=[1.0, 1.5, 2.0, 3.0])
    >>> tester = CostSensitivityTester(backtester, config)
    >>> results = tester.run_sensitivity_test(strategy, decision_frame, quotes)
    >>> robustness = tester.robustness_score(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import polars as pl


class BacktesterProtocol(Protocol):
    """Protocol defining the interface expected from a backtester."""

    def run(
        self,
        decision_frame: Any,
        quotes: Any,
        spread_multiplier: float,
        slippage_multiplier: float,
    ) -> Any:
        """Run backtest with cost adjustments."""
        ...


class StrategyProtocol(Protocol):
    """Protocol defining the interface expected from a strategy."""

    strategy_id: str
    family: str


@dataclass
class CostScenario:
    """
    A cost scenario for sensitivity testing.

    Attributes:
        name: Descriptive name for the scenario
        spread_multiplier: Multiplier for base spread (1.0 = base)
        slippage_multiplier: Multiplier for base slippage (1.0 = base)
    """

    name: str
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0


@dataclass
class SensitivityResult:
    """
    Result from a single cost sensitivity scenario.

    Attributes:
        strategy_id: Identifier of the tested strategy
        cost_scenario: Name of the cost scenario
        spread_multiplier: Applied spread multiplier
        slippage_multiplier: Applied slippage multiplier
        sharpe: Sharpe ratio under this scenario
        pnl: Total P&L under this scenario
        pnl_change_pct: P&L change percentage from base case
        still_profitable: Whether strategy remains profitable
    """

    strategy_id: str
    cost_scenario: str
    spread_multiplier: float
    slippage_multiplier: float
    sharpe: float
    pnl: float
    pnl_change_pct: float
    still_profitable: bool


@dataclass
class CostSensitivityConfig:
    """
    Configuration for cost sensitivity testing.

    Attributes:
        base_spread_bps: Base spread in basis points (default: 1.0)
        spread_multipliers: List of spread multipliers to test
        slippage_multipliers: List of slippage multipliers to test
    """

    base_spread_bps: float = 1.0
    spread_multipliers: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0]
    )
    slippage_multipliers: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0]
    )

    def generate_scenarios(self) -> list[CostScenario]:
        """
        Generate all cost scenarios from configured multipliers.

        Returns:
            List of CostScenario instances for all combinations
        """
        scenarios = []
        for spread_mult in self.spread_multipliers:
            for slip_mult in self.slippage_multipliers:
                name = f"spread_{spread_mult}x_slip_{slip_mult}x"
                scenarios.append(
                    CostScenario(
                        name=name,
                        spread_multiplier=spread_mult,
                        slippage_multiplier=slip_mult,
                    )
                )
        return scenarios


class CostSensitivityTester:
    """
    Test strategy robustness to transaction cost variations.

    Provides methods for:
    - Running strategies under various cost scenarios
    - Computing breakeven cost levels
    - Calculating robustness scores
    - Generating cost impact summaries

    The tester uses a backtester instance to evaluate strategy performance
    under different cost assumptions.
    """

    def __init__(
        self,
        backtester: BacktesterProtocol,
        config: CostSensitivityConfig | None = None,
    ) -> None:
        """
        Initialize the CostSensitivityTester.

        Args:
            backtester: Backtester instance for running simulations
            config: Configuration for cost scenarios (uses defaults if not provided)
        """
        self.backtester = backtester
        self.config = config or CostSensitivityConfig()

    def run_sensitivity_test(
        self,
        strategy: StrategyProtocol,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
    ) -> list[SensitivityResult]:
        """
        Run strategy under various cost scenarios.

        Tests the strategy against all configured spread and slippage
        multiplier combinations, recording P&L and Sharpe ratio for each.

        Args:
            strategy: Strategy to test
            decision_frame: Feature DataFrame with timestamps and symbols
            quotes: Quote data for backtesting

        Returns:
            List of SensitivityResult for each cost scenario
        """
        results: list[SensitivityResult] = []
        base_pnl: float | None = None

        # Generate all scenarios
        scenarios = self.config.generate_scenarios()

        for scenario in scenarios:
            # Run backtest with adjusted costs
            backtest_result = self.backtester.run(
                decision_frame,
                quotes,
                spread_multiplier=scenario.spread_multiplier,
                slippage_multiplier=scenario.slippage_multiplier,
            )

            pnl = float(backtest_result.total_pnl)
            sharpe = float(backtest_result.sharpe)

            # Store base P&L for comparison
            if scenario.spread_multiplier == 1.0 and scenario.slippage_multiplier == 1.0:
                base_pnl = pnl

            # Calculate P&L change percentage
            if base_pnl is not None and base_pnl != 0:
                pnl_change_pct = (pnl - base_pnl) / abs(base_pnl)
            else:
                pnl_change_pct = 0.0

            results.append(
                SensitivityResult(
                    strategy_id=strategy.strategy_id,
                    cost_scenario=scenario.name,
                    spread_multiplier=scenario.spread_multiplier,
                    slippage_multiplier=scenario.slippage_multiplier,
                    sharpe=sharpe,
                    pnl=pnl,
                    pnl_change_pct=pnl_change_pct,
                    still_profitable=pnl > 0,
                )
            )

        # Update P&L change percentages now that we have base_pnl
        if base_pnl is not None and base_pnl != 0:
            for result in results:
                result.pnl_change_pct = (result.pnl - base_pnl) / abs(base_pnl)

        return results

    def compute_breakeven_costs(
        self,
        strategy: StrategyProtocol,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Find the cost level where strategy becomes unprofitable.

        Uses binary search to find the approximate breakeven point
        for both spread and slippage multipliers independently.

        Args:
            strategy: Strategy to test
            decision_frame: Feature DataFrame
            quotes: Quote data

        Returns:
            Dictionary with breakeven multipliers:
            - spread_breakeven: Spread multiplier at breakeven
            - slippage_breakeven: Slippage multiplier at breakeven
            - combined_breakeven: Combined multiplier at breakeven
        """
        # Find spread breakeven (holding slippage at 1.0)
        spread_breakeven = self._binary_search_breakeven(
            strategy, decision_frame, quotes, vary_spread=True
        )

        # Find slippage breakeven (holding spread at 1.0)
        slippage_breakeven = self._binary_search_breakeven(
            strategy, decision_frame, quotes, vary_spread=False
        )

        # Find combined breakeven (both vary equally)
        combined_breakeven = self._binary_search_breakeven(
            strategy, decision_frame, quotes, vary_both=True
        )

        return {
            "spread_breakeven": spread_breakeven,
            "slippage_breakeven": slippage_breakeven,
            "combined_breakeven": combined_breakeven,
        }

    def _binary_search_breakeven(
        self,
        strategy: StrategyProtocol,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
        vary_spread: bool = True,
        vary_both: bool = False,
        low: float = 1.0,
        high: float = 10.0,
        tolerance: float = 0.1,
        max_iterations: int = 20,
    ) -> float:
        """
        Binary search to find breakeven cost multiplier.

        Args:
            strategy: Strategy to test
            decision_frame: Feature DataFrame
            quotes: Quote data
            vary_spread: If True, vary spread (else vary slippage)
            vary_both: If True, vary both equally
            low: Lower bound of search
            high: Upper bound of search
            tolerance: Search precision
            max_iterations: Maximum search iterations

        Returns:
            Approximate breakeven multiplier
        """
        for _ in range(max_iterations):
            mid = (low + high) / 2

            if vary_both:
                spread_mult = mid
                slip_mult = mid
            elif vary_spread:
                spread_mult = mid
                slip_mult = 1.0
            else:
                spread_mult = 1.0
                slip_mult = mid

            result = self.backtester.run(
                decision_frame,
                quotes,
                spread_multiplier=spread_mult,
                slippage_multiplier=slip_mult,
            )

            if result.total_pnl > 0:
                low = mid
            else:
                high = mid

            if high - low < tolerance:
                break

        return (low + high) / 2

    def robustness_score(self, results: list[SensitivityResult]) -> float:
        """
        Compute robustness score (0-1) based on cost sensitivity.

        The score considers:
        - Fraction of scenarios remaining profitable
        - Average P&L degradation rate
        - Worst-case P&L preservation

        A score of 1.0 means the strategy is highly robust to cost changes.
        A score of 0.0 means the strategy is very sensitive to costs.

        Args:
            results: List of SensitivityResult from sensitivity test

        Returns:
            Robustness score between 0 and 1
        """
        if not results:
            return 0.0

        # Component 1: Fraction of scenarios that remain profitable
        profitable_count = sum(1 for r in results if r.still_profitable)
        profitability_ratio = profitable_count / len(results)

        # Component 2: Average P&L preservation (1 - avg degradation)
        pnl_changes = [r.pnl_change_pct for r in results]
        avg_degradation = float(abs(np.mean(pnl_changes)))
        # Cap degradation impact at 100%
        pnl_preservation = max(0.0, 1.0 - min(avg_degradation, 1.0))

        # Component 3: Worst case P&L preservation
        worst_change = float(min(pnl_changes))
        worst_preservation = max(0.0, 1.0 - min(abs(worst_change), 2.0) / 2.0)

        # Weighted combination
        score = (
            0.5 * profitability_ratio
            + 0.3 * pnl_preservation
            + 0.2 * worst_preservation
        )

        return float(np.clip(score, 0, 1))

    def get_sensitivity_summary(
        self, results: list[SensitivityResult]
    ) -> dict[str, Any]:
        """
        Get a summary of sensitivity analysis results.

        Args:
            results: List of SensitivityResult from sensitivity test

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        strategy_id = results[0].strategy_id
        pnls = [r.pnl for r in results]
        sharpes = [r.sharpe for r in results]

        # Find base case
        base_result = next(
            (
                r
                for r in results
                if r.spread_multiplier == 1.0 and r.slippage_multiplier == 1.0
            ),
            results[0],
        )

        profitable_scenarios = sum(1 for r in results if r.still_profitable)

        return {
            "strategy_id": strategy_id,
            "num_scenarios": len(results),
            "base_pnl": base_result.pnl,
            "base_sharpe": base_result.sharpe,
            "worst_pnl": min(pnls),
            "best_pnl": max(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "worst_sharpe": min(sharpes),
            "avg_sharpe": float(np.mean(sharpes)),
            "profitable_scenarios": profitable_scenarios,
            "unprofitable_scenarios": len(results) - profitable_scenarios,
            "robustness_score": self.robustness_score(results),
        }

    def get_cost_impact_matrix(
        self, results: list[SensitivityResult]
    ) -> dict[tuple[float, float], dict[str, float]]:
        """
        Get a cost impact matrix indexed by (spread_mult, slip_mult).

        Args:
            results: List of SensitivityResult from sensitivity test

        Returns:
            Dictionary mapping (spread, slippage) to result metrics
        """
        matrix: dict[tuple[float, float], dict[str, float]] = {}

        for r in results:
            key = (r.spread_multiplier, r.slippage_multiplier)
            matrix[key] = {
                "pnl": r.pnl,
                "sharpe": r.sharpe,
                "pnl_change_pct": r.pnl_change_pct,
                "profitable": 1.0 if r.still_profitable else 0.0,
            }

        return matrix


# Public API exports
__all__ = [
    "CostScenario",
    "SensitivityResult",
    "CostSensitivityConfig",
    "CostSensitivityTester",
]
