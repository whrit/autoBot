"""
Walk-forward optimization framework.

This module provides walk-forward analysis for strategy evaluation:
- Rolling window splits with configurable train/test/step sizes
- No lookahead bias - train data always precedes test data
- Aggregation of results across windows

Walk-forward analysis helps identify strategies that generalize
to unseen data, rather than those that only fit historical data.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    Splits time-series data into rolling train/test windows for
    out-of-sample evaluation. Each window trains on historical data
    and tests on subsequent data, preventing lookahead bias.

    Attributes:
        train_size: Number of bars in each training window
        test_size: Number of bars in each test window
        step_size: Number of bars to advance between windows
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int,
    ) -> None:
        """
        Initialize WalkForwardOptimizer.

        Args:
            train_size: Number of bars in each training window
            test_size: Number of bars in each test window
            step_size: Number of bars to advance between windows

        Raises:
            ValueError: If any size parameter is not positive
        """
        if train_size <= 0:
            raise ValueError("train_size must be positive")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate train/test splits for walk-forward analysis.

        Each split contains:
        - train: `train_size` consecutive bars
        - test: `test_size` consecutive bars immediately following train

        Windows advance by `step_size` bars between splits.

        Args:
            data: DataFrame with time-series data (must have 'timestamp' column)

        Yields:
            Tuple of (train_df, test_df) for each window

        Note:
            Returns empty iterator if data is smaller than train_size + test_size
        """
        n_rows = len(data)
        min_required = self.train_size + self.test_size

        if n_rows < min_required:
            return

        # Calculate number of possible windows
        start_idx = 0

        while start_idx + min_required <= n_rows:
            train_end_idx = start_idx + self.train_size
            test_end_idx = train_end_idx + self.test_size

            # Ensure we don't exceed data bounds
            if test_end_idx > n_rows:
                break

            # Extract train and test windows
            train = data.slice(start_idx, self.train_size)
            test = data.slice(train_end_idx, self.test_size)

            yield train, test

            # Advance to next window
            start_idx += self.step_size

    def get_num_splits(self, n_rows: int) -> int:
        """
        Calculate the number of splits for a given data length.

        Args:
            n_rows: Total number of rows in the data

        Returns:
            Number of train/test splits that will be generated
        """
        min_required = self.train_size + self.test_size

        if n_rows < min_required:
            return 0

        # Count number of valid starting positions
        num_splits = 0
        start_idx = 0

        while start_idx + min_required <= n_rows:
            num_splits += 1
            start_idx += self.step_size

        return num_splits


__all__ = ["WalkForwardOptimizer"]
