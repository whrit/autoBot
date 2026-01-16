"""
Shadow vs Backtest Behavior Validation (T6.07).

This module provides validation of shadow execution results against backtest results
to ensure consistency and detect behavioral drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration for shadow vs backtest behavior validation.

    Attributes:
        sharpe_tolerance: Maximum allowed Sharpe ratio deviation (0.10 = 10%)
        fill_rate_tolerance: Maximum allowed fill rate deviation (0.05 = 5%)
        slippage_tolerance_bps: Maximum allowed slippage deviation in basis points
    """

    sharpe_tolerance: float = 0.10
    fill_rate_tolerance: float = 0.05
    slippage_tolerance_bps: float = 1.0


@dataclass
class ValidationResult:
    """
    Result of shadow vs backtest validation.

    Attributes:
        strategy_id: Strategy identifier being validated
        shadow_sharpe: Sharpe ratio from shadow execution
        backtest_sharpe: Sharpe ratio from backtest
        sharpe_deviation: Absolute deviation between Sharpe ratios
        within_tolerance: Whether Sharpe deviation is within tolerance
        fill_rate_shadow: Fill rate from shadow execution
        fill_rate_backtest: Fill rate from backtest
        avg_slippage_deviation_bps: Average slippage deviation in bps
        issues: List of identified issues/warnings
        passed: Overall validation pass/fail status
    """

    strategy_id: str
    shadow_sharpe: float
    backtest_sharpe: float
    sharpe_deviation: float
    within_tolerance: bool
    fill_rate_shadow: float
    fill_rate_backtest: float
    avg_slippage_deviation_bps: float
    issues: list[str] = field(default_factory=list)
    passed: bool = True


class BehaviorValidator:
    """
    Validate shadow execution matches backtest behavior.

    Compares shadow execution results against backtest results to ensure
    the shadow execution is behaving as expected based on historical backtest.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """
        Initialize the BehaviorValidator.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        shadow_results: pl.DataFrame,
        backtest_results: pl.DataFrame,
    ) -> ValidationResult:
        """
        Compare shadow vs backtest results.

        Args:
            shadow_results: DataFrame with shadow execution results.
                Expected columns: timestamp, equity, returns, symbol, strategy_id,
                                 fill_count, slippage_bps
            backtest_results: DataFrame with backtest results.
                Same columns as shadow_results.

        Returns:
            ValidationResult with comparison metrics and pass/fail status.
        """
        issues: list[str] = []

        # Handle empty DataFrames
        if len(shadow_results) == 0 or len(backtest_results) == 0:
            return ValidationResult(
                strategy_id="unknown",
                shadow_sharpe=0.0,
                backtest_sharpe=0.0,
                sharpe_deviation=float("inf"),
                within_tolerance=False,
                fill_rate_shadow=0.0,
                fill_rate_backtest=0.0,
                avg_slippage_deviation_bps=0.0,
                issues=["Empty DataFrame provided - cannot validate"],
                passed=False,
            )

        # Extract strategy ID
        strategy_id = self._extract_strategy_id(shadow_results)

        # Calculate Sharpe ratios
        shadow_sharpe = self._calculate_sharpe(shadow_results)
        backtest_sharpe = self._calculate_sharpe(backtest_results)

        # Compare Sharpe ratios
        sharpe_deviation, sharpe_within = self.compare_sharpe(shadow_sharpe, backtest_sharpe)

        if not sharpe_within:
            issues.append(
                f"Sharpe ratio deviation ({sharpe_deviation:.2%}) exceeds "
                f"tolerance ({self.config.sharpe_tolerance:.2%})"
            )

        # Calculate fill rates
        fill_rate_shadow = self._calculate_fill_rate(shadow_results)
        fill_rate_backtest = self._calculate_fill_rate(backtest_results)

        # Compare fill rates
        fill_rate_deviation, fill_rate_within = self.compare_fill_rates(
            fill_rate_shadow, fill_rate_backtest
        )

        if not fill_rate_within:
            issues.append(
                f"Fill rate deviation ({fill_rate_deviation:.2%}) exceeds "
                f"tolerance ({self.config.fill_rate_tolerance:.2%})"
            )

        # Analyze slippage deviation
        shadow_slippage = self._extract_slippage(shadow_results)
        backtest_slippage = self._extract_slippage(backtest_results)
        avg_slippage_deviation = self.analyze_slippage_deviation(
            shadow_slippage, backtest_slippage
        )

        if avg_slippage_deviation > self.config.slippage_tolerance_bps:
            issues.append(
                f"Slippage deviation ({avg_slippage_deviation:.2f} bps) exceeds "
                f"tolerance ({self.config.slippage_tolerance_bps:.2f} bps)"
            )

        # Determine overall pass/fail
        passed = sharpe_within and fill_rate_within and len(issues) == 0

        return ValidationResult(
            strategy_id=strategy_id,
            shadow_sharpe=shadow_sharpe,
            backtest_sharpe=backtest_sharpe,
            sharpe_deviation=sharpe_deviation,
            within_tolerance=sharpe_within,
            fill_rate_shadow=fill_rate_shadow,
            fill_rate_backtest=fill_rate_backtest,
            avg_slippage_deviation_bps=avg_slippage_deviation,
            issues=issues,
            passed=passed,
        )

    def compare_sharpe(self, shadow: float, backtest: float) -> tuple[float, bool]:
        """
        Compare Sharpe ratios within tolerance.

        Args:
            shadow: Sharpe ratio from shadow execution
            backtest: Sharpe ratio from backtest

        Returns:
            Tuple of (deviation, within_tolerance)
        """
        if backtest == 0.0:
            # Handle zero backtest Sharpe
            if shadow == 0.0:
                return 0.0, True
            return float("inf"), False

        deviation = abs(shadow - backtest) / abs(backtest)
        within = deviation <= self.config.sharpe_tolerance

        return deviation, within

    def compare_fill_rates(self, shadow: float, backtest: float) -> tuple[float, bool]:
        """
        Compare fill rates within tolerance.

        Args:
            shadow: Fill rate from shadow execution
            backtest: Fill rate from backtest

        Returns:
            Tuple of (deviation, within_tolerance)
        """
        if backtest == 0.0:
            if shadow == 0.0:
                return 0.0, True
            return float("inf"), False

        deviation = abs(shadow - backtest) / backtest
        within = deviation <= self.config.fill_rate_tolerance

        return deviation, within

    def analyze_slippage_deviation(
        self,
        shadow_slippage: list[float],
        backtest_slippage: list[float],
    ) -> float:
        """
        Analyze slippage deviation in basis points.

        Args:
            shadow_slippage: List of slippage values from shadow execution
            backtest_slippage: List of slippage values from backtest

        Returns:
            Average absolute deviation in basis points
        """
        if not shadow_slippage or not backtest_slippage:
            return 0.0

        # Use minimum length for comparison
        min_len = min(len(shadow_slippage), len(backtest_slippage))
        if min_len == 0:
            return 0.0

        shadow_arr = np.array(shadow_slippage[:min_len])
        backtest_arr = np.array(backtest_slippage[:min_len])

        deviations = np.abs(shadow_arr - backtest_arr)
        return float(np.mean(deviations))

    def _extract_strategy_id(self, df: pl.DataFrame) -> str:
        """Extract strategy ID from DataFrame."""
        if "strategy_id" in df.columns and len(df) > 0:
            strategy_ids = df["strategy_id"].unique().to_list()
            return str(strategy_ids[0]) if strategy_ids else "unknown"
        return "unknown"

    def _calculate_sharpe(self, df: pl.DataFrame) -> float:
        """Calculate annualized Sharpe ratio from returns."""
        if "returns" not in df.columns or len(df) < 2:
            return 0.0

        returns = df["returns"].to_numpy().astype(np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        std = np.std(returns, ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0

        # Annualize assuming minute bars (252 days * 390 minutes)
        annualization_factor = np.sqrt(252 * 390)
        sharpe = (np.mean(returns) / std) * annualization_factor

        return float(sharpe)

    def _calculate_fill_rate(self, df: pl.DataFrame) -> float:
        """Calculate fill rate from DataFrame."""
        if "fill_count" not in df.columns or len(df) == 0:
            return 0.0

        # Fill rate = actual fills / expected fills
        # For simplicity, assume all rows are expected fills
        total_rows = len(df)
        filled = df["fill_count"].sum()

        if total_rows == 0:
            return 0.0

        return float(filled / total_rows)

    def _extract_slippage(self, df: pl.DataFrame) -> list[float]:
        """Extract slippage values from DataFrame."""
        if "slippage_bps" not in df.columns:
            return []
        return df["slippage_bps"].to_list()


__all__ = ["BehaviorValidator", "ValidationConfig", "ValidationResult"]
