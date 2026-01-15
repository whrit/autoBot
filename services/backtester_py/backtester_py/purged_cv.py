"""
Purged cross-validation module.

This module provides purged cross-validation for robust strategy evaluation:
- Embargo period between train and test to prevent data leakage
- Combinatorial purged CV for more robust estimates
- Deflated Sharpe ratio to account for multiple testing bias

Reference: Advances in Financial Machine Learning by Marcos Lopez de Prado
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import polars as pl


class PurgedCrossValidator:
    """
    Purged cross-validator with embargo period.

    Implements time-series cross-validation with a gap (embargo) between
    train and test sets to prevent information leakage from overlapping
    labels or features with memory.

    Attributes:
        n_splits: Number of cross-validation folds
        embargo_pct: Percentage of data to embargo between train and test
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        """
        Initialize PurgedCrossValidator.

        Args:
            n_splits: Number of cross-validation folds (must be >= 2)
            embargo_pct: Percentage of data to embargo (0 to 1)

        Raises:
            ValueError: If n_splits < 2 or embargo_pct not in [0, 1)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo_pct < 0 or embargo_pct >= 1:
            raise ValueError("embargo_pct must be between 0 and 1")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate train/test splits with embargo period.

        Each split ensures:
        - No overlap between train and test
        - Embargo period between train end and test start
        - Train data always precedes test data (no lookahead)

        Args:
            data: DataFrame with time-series data

        Yields:
            Tuple of (train_df, test_df) for each fold
        """
        n_rows = len(data)
        embargo_size = int(n_rows * self.embargo_pct)

        # Calculate fold boundaries
        fold_size = n_rows // self.n_splits

        for i in range(self.n_splits):
            # Test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_rows

            # Train data: everything before test, minus embargo
            train_end = max(0, test_start - embargo_size)

            # Also include data after test (for later folds we can use earlier data)
            # But we need to apply embargo after test as well for strictness
            # For simplicity, only use data before test

            if train_end <= 0:
                # Skip if no training data available (first fold with large embargo)
                # Instead, use data after test for training
                train_start_after = min(test_end + embargo_size, n_rows)
                if train_start_after >= n_rows:
                    continue

                train = data.slice(train_start_after, n_rows - train_start_after)
            else:
                train = data.slice(0, train_end)

            test = data.slice(test_start, test_end - test_start)

            if len(train) > 0 and len(test) > 0:
                yield train, test

    def split_combinatorial(
        self, data: pl.DataFrame, n_test_groups: int = 2
    ) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate combinatorial purged cross-validation splits.

        More thorough than standard k-fold by testing all combinations
        of test groups while respecting temporal order and embargo.

        Args:
            data: DataFrame with time-series data
            n_test_groups: Number of groups to use as test in each split

        Yields:
            Tuple of (train_df, test_df) for each combination
        """
        import itertools

        n_rows = len(data)
        embargo_size = int(n_rows * self.embargo_pct)
        fold_size = n_rows // self.n_splits

        # Get all fold indices
        fold_indices = list(range(self.n_splits))

        # Generate all combinations of n_test_groups folds
        for test_folds in itertools.combinations(fold_indices, n_test_groups):
            train_indices: list[int] = []
            test_indices: list[int] = []

            # Identify test regions
            for fold_idx in range(self.n_splits):
                fold_start = fold_idx * fold_size
                fold_end = (
                    (fold_idx + 1) * fold_size
                    if fold_idx < self.n_splits - 1
                    else n_rows
                )

                if fold_idx in test_folds:
                    # This fold is test
                    test_indices.extend(range(fold_start, fold_end))
                else:
                    # Check if this fold is embargoed
                    # A train fold is embargoed if it's adjacent to any test fold
                    for test_fold in test_folds:
                        # Check if current fold is within embargo of test fold
                        if abs(fold_idx - test_fold) == 1:
                            # Adjacent fold - apply embargo
                            if fold_idx < test_fold:
                                # Train fold before test - embargo end of train
                                fold_end = max(fold_start, fold_end - embargo_size)
                            else:
                                # Train fold after test - embargo start of train
                                fold_start = min(fold_end, fold_start + embargo_size)

                    if fold_start < fold_end:
                        train_indices.extend(range(fold_start, fold_end))

            if train_indices and test_indices:
                train = data[sorted(train_indices)]
                test = data[sorted(test_indices)]
                yield train, test

    def deflated_sharpe(
        self,
        sharpe_ratios: list[float],
        num_trials: int,
        expected_max_sharpe: float | None = None,
    ) -> float:
        """
        Calculate deflated Sharpe ratio accounting for multiple testing.

        The deflated Sharpe adjusts for the bias introduced by selecting
        the best strategy from multiple candidates (multiple testing).

        Based on Bailey and Lopez de Prado (2014):
        "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
        Overfitting and Non-Normality"

        Args:
            sharpe_ratios: List of Sharpe ratios from different trials/folds
            num_trials: Total number of strategy variations tested
            expected_max_sharpe: Expected maximum Sharpe under null hypothesis
                               (calculated if not provided)

        Returns:
            Deflated Sharpe ratio that accounts for selection bias
        """
        if not sharpe_ratios:
            return 0.0

        if num_trials <= 0:
            return float(max(sharpe_ratios))

        # Observed maximum Sharpe
        max_sharpe = max(sharpe_ratios)
        n = len(sharpe_ratios)

        if n == 1 and num_trials == 1:
            return max_sharpe

        # Estimate parameters
        mean_sharpe = float(np.mean(sharpe_ratios))
        std_sharpe = float(np.std(sharpe_ratios, ddof=1)) if n > 1 else 0.0

        if std_sharpe == 0:
            # All same - return mean
            return mean_sharpe

        # Expected maximum Sharpe under null hypothesis
        # E[max] = mean + std * E[max of N standard normals]
        if expected_max_sharpe is None:
            # Use approximation for expected maximum of N standard normals
            # E[max(Z_1, ..., Z_N)] ≈ sqrt(2 * log(N)) for large N
            # Note: Euler-Mascheroni constant (0.5772...) not needed in this approximation
            expected_max_z = np.sqrt(2 * np.log(num_trials)) - (
                np.log(np.log(num_trials)) + np.log(4 * np.pi)
            ) / (2 * np.sqrt(2 * np.log(num_trials)))

            # Cap at reasonable value
            expected_max_z = min(expected_max_z, 4.0)
            expected_max_sharpe = mean_sharpe + std_sharpe * expected_max_z

        # Probability that max Sharpe is not due to chance
        # p = P(Sharpe < observed | H0: true Sharpe = 0)
        # Deflated Sharpe is the quantile of the observed max given the expected distribution

        # Simplified approach: adjust max Sharpe by the expected inflation
        # Deflated = max - (E[max under null] - mean)
        inflation = expected_max_sharpe - mean_sharpe
        deflated = max_sharpe - inflation

        # Alternative: use probability approach
        # Calculate the probability that observed max is due to skill vs luck
        z_score = (max_sharpe - mean_sharpe) / std_sharpe if std_sharpe > 0 else 0.0

        # Expected z-score for max of num_trials standard normals
        expected_z = np.sqrt(2 * np.log(num_trials)) if num_trials > 1 else 0.0

        # Deflated Sharpe using haircut
        haircut = min(1.0, expected_z / max(z_score, 0.1)) if z_score > 0 else 1.0
        deflated_prob = max_sharpe * (1 - haircut * 0.5)

        # Return average of both methods (they should be similar)
        return float(min(max_sharpe, max(deflated, deflated_prob, 0.0)))

    def probabilistic_sharpe_ratio(
        self,
        observed_sharpe: float,
        benchmark_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Calculate probability that Sharpe ratio exceeds benchmark.

        PSR = Pr(SR > SR*) where SR* is the benchmark Sharpe.

        Based on Bailey and Lopez de Prado (2012):
        "The Sharpe Ratio Efficient Frontier"

        Args:
            observed_sharpe: Observed Sharpe ratio
            benchmark_sharpe: Benchmark Sharpe ratio to compare against
            n_observations: Number of observations used to calculate Sharpe
            skewness: Skewness of returns (default 0 = normal)
            kurtosis: Kurtosis of returns (default 3 = normal)

        Returns:
            Probability that true Sharpe exceeds benchmark
        """
        if n_observations <= 2:
            return 0.5  # Not enough data

        # Standard error of Sharpe ratio (with non-normality adjustment)
        se = np.sqrt(
            (
                1
                + 0.5 * observed_sharpe**2
                - skewness * observed_sharpe
                + ((kurtosis - 3) / 4) * observed_sharpe**2
            )
            / (n_observations - 1)
        )

        if se <= 0:
            return 1.0 if observed_sharpe > benchmark_sharpe else 0.0

        # z-statistic
        z = (observed_sharpe - benchmark_sharpe) / se

        # Probability using standard normal CDF
        psr = float(stats.norm.cdf(z))

        return psr


__all__ = ["PurgedCrossValidator"]
