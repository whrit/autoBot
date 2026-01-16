"""
Walk-forward optimization framework.

This module provides walk-forward analysis for strategy evaluation:
- Rolling window splits with configurable train/test/step sizes
- No lookahead bias - train data always precedes test data
- Aggregation of results across windows
- Progress tracking with Rich progress bars

Walk-forward analysis helps identify strategies that generalize
to unseen data, rather than those that only fit historical data.

Memory optimizations:
- Lazy data loading from Parquet files
- Generator-based splits to avoid loading all data
- Configurable memory limits
- Garbage collection between folds
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from backtester_py.logging_config import get_logger

if TYPE_CHECKING:
    import polars as pl

logger = get_logger("backtester.walk_forward")


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        return 0.0


@dataclass
class MemoryConfig:
    """Configuration for memory management in walk-forward optimization.

    Attributes:
        enable_gc: Run garbage collection after each fold.
        log_memory: Log memory usage during processing.
        max_memory_mb: Maximum memory usage before warning (0 = unlimited).
    """
    enable_gc: bool = True
    log_memory: bool = False
    max_memory_mb: float = 0.0


@dataclass
class FoldResult:
    """
    Result from a single walk-forward fold.

    Attributes:
        fold_index: Index of the fold (0-based)
        train_size: Number of rows in training data
        test_size: Number of rows in test data
        sharpe_ratio: Sharpe ratio on test data
        max_drawdown: Maximum drawdown on test data
        total_return: Total return on test data
        win_rate: Win rate on test data
        num_trades: Number of trades executed
        metrics: Additional metrics dictionary
    """

    fold_index: int
    train_size: int
    test_size: int
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """
    Aggregated result from walk-forward optimization.

    Attributes:
        fold_results: Results from each fold
        avg_sharpe: Average Sharpe ratio across folds
        avg_max_drawdown: Average max drawdown across folds
        avg_return: Average total return across folds
        std_sharpe: Standard deviation of Sharpe ratio
        total_folds: Total number of folds executed
    """

    fold_results: list[FoldResult] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_max_drawdown: float = 0.0
    avg_return: float = 0.0
    std_sharpe: float = 0.0
    total_folds: int = 0

    def summary_table(self, console: Console | None = None) -> Table:
        """Generate a Rich table summarizing walk-forward results."""
        table = Table(
            title="Walk-Forward Results Summary",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Fold", justify="center")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Trades", justify="right")

        for fr in self.fold_results:
            dd_style = "red" if fr.max_drawdown > 0.1 else "green"
            ret_style = "green" if fr.total_return > 0 else "red"

            table.add_row(
                str(fr.fold_index + 1),
                f"{fr.sharpe_ratio:.2f}",
                f"[{dd_style}]{fr.max_drawdown * 100:.1f}%[/{dd_style}]",
                f"[{ret_style}]{fr.total_return * 100:.1f}%[/{ret_style}]",
                f"{fr.win_rate * 100:.1f}%",
                str(fr.num_trades),
            )

        # Add summary row
        table.add_section()
        table.add_row(
            "AVG",
            f"{self.avg_sharpe:.2f}",
            f"{self.avg_max_drawdown * 100:.1f}%",
            f"{self.avg_return * 100:.1f}%",
            "-",
            "-",
            style="bold",
        )

        return table


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    Splits time-series data into rolling train/test windows for
    out-of-sample evaluation. Each window trains on historical data
    and tests on subsequent data, preventing lookahead bias.

    Includes Rich progress bar support for visualizing iteration progress.

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
        console: Console | None = None,
    ) -> None:
        """
        Initialize WalkForwardOptimizer.

        Args:
            train_size: Number of bars in each training window
            test_size: Number of bars in each test window
            step_size: Number of bars to advance between windows
            console: Optional Rich console for progress display

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
        self.console = console or Console()

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


    def run_with_progress(
        self,
        data: pl.DataFrame,
        evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], FoldResult],
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization with progress tracking.

        Args:
            data: Full dataset for walk-forward analysis
            evaluate_fn: Function that takes (train_df, test_df, fold_idx)
                and returns a FoldResult
            show_progress: Whether to display Rich progress bar

        Returns:
            WalkForwardResult with all fold results and aggregated metrics
        """
        import numpy as np

        n_splits = self.get_num_splits(len(data))

        if n_splits == 0:
            logger.warning(
                "insufficient_data_for_walk_forward",
                data_rows=len(data),
                required_rows=self.train_size + self.test_size,
            )
            return WalkForwardResult()

        logger.info(
            "walk_forward_started",
            total_folds=n_splits,
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
        )

        fold_results: list[FoldResult] = []

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]Sharpe: {task.fields[sharpe]:.2f}[/cyan]"),
                TextColumn("[red]MDD: {task.fields[mdd]:.1%}[/red]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )

            with progress:
                task = progress.add_task(
                    f"Walk-Forward [0/{n_splits}]",
                    total=n_splits,
                    sharpe=0.0,
                    mdd=0.0,
                )

                for fold_idx, (train, test) in enumerate(self.split(data)):
                    result = evaluate_fn(train, test, fold_idx)
                    fold_results.append(result)

                    # Update progress bar
                    progress.update(
                        task,
                        advance=1,
                        description=f"Fold [{fold_idx + 1}/{n_splits}]",
                        sharpe=result.sharpe_ratio,
                        mdd=result.max_drawdown,
                    )

                    logger.debug(
                        "fold_completed",
                        fold=fold_idx + 1,
                        sharpe=result.sharpe_ratio,
                        max_drawdown=result.max_drawdown,
                        trades=result.num_trades,
                    )
        else:
            for fold_idx, (train, test) in enumerate(self.split(data)):
                result = evaluate_fn(train, test, fold_idx)
                fold_results.append(result)

                logger.debug(
                    "fold_completed",
                    fold=fold_idx + 1,
                    sharpe=result.sharpe_ratio,
                    max_drawdown=result.max_drawdown,
                    trades=result.num_trades,
                )

        # Calculate aggregated metrics
        if fold_results:
            sharpes = [fr.sharpe_ratio for fr in fold_results]
            avg_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            avg_max_dd = float(np.mean([fr.max_drawdown for fr in fold_results]))
            avg_return = float(np.mean([fr.total_return for fr in fold_results]))
        else:
            avg_sharpe = std_sharpe = avg_max_dd = avg_return = 0.0

        result = WalkForwardResult(
            fold_results=fold_results,
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_max_dd,
            avg_return=avg_return,
            std_sharpe=std_sharpe,
            total_folds=len(fold_results),
        )

        logger.info(
            "walk_forward_completed",
            total_folds=result.total_folds,
            avg_sharpe=result.avg_sharpe,
            std_sharpe=result.std_sharpe,
            avg_max_drawdown=result.avg_max_drawdown,
            avg_return=result.avg_return,
        )

        return result

    def split_lazy(
        self,
        data_path: str | Path,
        timestamp_column: str = "timestamp",
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """
        Generate train/test splits with lazy loading from Parquet.

        Loads only the data needed for each fold rather than keeping
        the entire dataset in memory. Useful for multi-year backtests.

        Args:
            data_path: Path to Parquet file or directory
            timestamp_column: Name of timestamp column for sorting

        Yields:
            Tuple of (train_df, test_df) for each window
        """
        import polars as pl

        # Get row count without loading data
        lazy_df = pl.scan_parquet(data_path)
        n_rows = lazy_df.select(pl.len()).collect().item()

        min_required = self.train_size + self.test_size
        if n_rows < min_required:
            logger.warning(
                "insufficient_data_for_lazy_split",
                data_rows=n_rows,
                required_rows=min_required,
            )
            return

        start_idx = 0

        while start_idx + min_required <= n_rows:
            train_end_idx = start_idx + self.train_size
            test_end_idx = train_end_idx + self.test_size

            if test_end_idx > n_rows:
                break

            # Load only the required slices
            train = (
                pl.scan_parquet(data_path)
                .slice(start_idx, self.train_size)
                .collect()
            )
            test = (
                pl.scan_parquet(data_path)
                .slice(train_end_idx, self.test_size)
                .collect()
            )

            yield train, test

            # Explicitly delete to free memory
            del train, test
            gc.collect()

            start_idx += self.step_size

    def run_lazy(
        self,
        data_path: str | Path,
        evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], FoldResult],
        memory_config: MemoryConfig | None = None,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization with lazy data loading.

        Loads data incrementally from Parquet files rather than
        keeping the full dataset in memory.

        Args:
            data_path: Path to Parquet file or directory
            evaluate_fn: Function that takes (train_df, test_df, fold_idx)
                and returns a FoldResult
            memory_config: Optional memory management configuration
            show_progress: Whether to display Rich progress bar

        Returns:
            WalkForwardResult with all fold results and aggregated metrics
        """
        import numpy as np
        import polars as pl

        memory_config = memory_config or MemoryConfig()

        # Get total row count for progress calculation
        lazy_df = pl.scan_parquet(data_path)
        n_rows = lazy_df.select(pl.len()).collect().item()
        n_splits = self.get_num_splits(n_rows)

        if n_splits == 0:
            logger.warning(
                "insufficient_data_for_lazy_walk_forward",
                data_rows=n_rows,
                required_rows=self.train_size + self.test_size,
            )
            return WalkForwardResult()

        logger.info(
            "lazy_walk_forward_started",
            total_folds=n_splits,
            train_size=self.train_size,
            test_size=self.test_size,
            data_path=str(data_path),
        )

        fold_results: list[FoldResult] = []
        peak_memory_mb = 0.0

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]Sharpe: {task.fields[sharpe]:.2f}[/cyan]"),
                TextColumn("[yellow]Mem: {task.fields[mem]:.0f}MB[/yellow]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )

            with progress:
                task = progress.add_task(
                    f"Walk-Forward [0/{n_splits}]",
                    total=n_splits,
                    sharpe=0.0,
                    mem=0.0,
                )

                for fold_idx, (train, test) in enumerate(
                    self.split_lazy(data_path)
                ):
                    result = evaluate_fn(train, test, fold_idx)
                    fold_results.append(result)

                    # Memory tracking
                    current_mem = get_memory_usage_mb()
                    peak_memory_mb = max(peak_memory_mb, current_mem)

                    # Check memory limit
                    if memory_config.max_memory_mb > 0 and current_mem > memory_config.max_memory_mb:
                        logger.warning(
                            "memory_limit_exceeded",
                            current_mb=current_mem,
                            limit_mb=memory_config.max_memory_mb,
                            fold=fold_idx,
                        )

                    # Update progress
                    progress.update(
                        task,
                        advance=1,
                        description=f"Fold [{fold_idx + 1}/{n_splits}]",
                        sharpe=result.sharpe_ratio,
                        mem=current_mem,
                    )

                    # Cleanup
                    del train, test
                    if memory_config.enable_gc:
                        gc.collect()

                    if memory_config.log_memory:
                        logger.debug(
                            "lazy_fold_completed",
                            fold=fold_idx + 1,
                            memory_mb=current_mem,
                            peak_memory_mb=peak_memory_mb,
                        )
        else:
            for fold_idx, (train, test) in enumerate(self.split_lazy(data_path)):
                result = evaluate_fn(train, test, fold_idx)
                fold_results.append(result)

                current_mem = get_memory_usage_mb()
                peak_memory_mb = max(peak_memory_mb, current_mem)

                del train, test
                if memory_config.enable_gc:
                    gc.collect()

                if memory_config.log_memory:
                    logger.debug(
                        "lazy_fold_completed",
                        fold=fold_idx + 1,
                        memory_mb=current_mem,
                    )

        # Calculate aggregated metrics
        import numpy as np

        if fold_results:
            sharpes = [fr.sharpe_ratio for fr in fold_results]
            avg_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            avg_max_dd = float(np.mean([fr.max_drawdown for fr in fold_results]))
            avg_return = float(np.mean([fr.total_return for fr in fold_results]))
        else:
            avg_sharpe = std_sharpe = avg_max_dd = avg_return = 0.0

        result = WalkForwardResult(
            fold_results=fold_results,
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_max_dd,
            avg_return=avg_return,
            std_sharpe=std_sharpe,
            total_folds=len(fold_results),
        )

        logger.info(
            "lazy_walk_forward_completed",
            total_folds=result.total_folds,
            avg_sharpe=result.avg_sharpe,
            peak_memory_mb=peak_memory_mb,
        )

        return result

    def run_streaming(
        self,
        data_generator: Generator[pl.DataFrame, None, None],
        evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], FoldResult],
        memory_config: MemoryConfig | None = None,
    ) -> Generator[FoldResult, None, WalkForwardResult]:
        """
        Run walk-forward optimization with streaming data.

        Processes data as it arrives from a generator, useful for
        real-time or very large datasets.

        Args:
            data_generator: Generator yielding DataFrames
            evaluate_fn: Function to evaluate each fold
            memory_config: Optional memory configuration

        Yields:
            FoldResult after each fold

        Returns:
            WalkForwardResult with aggregated metrics
        """
        import numpy as np
        import polars as pl

        memory_config = memory_config or MemoryConfig()

        # Accumulate data until we have enough for first split
        accumulated: list[pl.DataFrame] = []
        total_rows = 0
        fold_idx = 0
        fold_results: list[FoldResult] = []

        min_required = self.train_size + self.test_size

        for chunk in data_generator:
            accumulated.append(chunk)
            total_rows += len(chunk)

            # Process splits when we have enough data
            while total_rows >= min_required:
                # Combine accumulated data
                combined = pl.concat(accumulated)

                # Extract train/test
                train = combined.slice(0, self.train_size)
                test = combined.slice(self.train_size, self.test_size)

                # Evaluate fold
                result = evaluate_fn(train, test, fold_idx)
                fold_results.append(result)
                yield result

                fold_idx += 1

                # Slide window by step_size
                if self.step_size >= len(combined):
                    accumulated = []
                    total_rows = 0
                else:
                    accumulated = [combined.slice(self.step_size)]
                    total_rows = len(accumulated[0])

                # Cleanup
                del combined, train, test
                if memory_config.enable_gc:
                    gc.collect()

        # Calculate final metrics
        if fold_results:
            sharpes = [fr.sharpe_ratio for fr in fold_results]
            avg_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            avg_max_dd = float(np.mean([fr.max_drawdown for fr in fold_results]))
            avg_return = float(np.mean([fr.total_return for fr in fold_results]))
        else:
            avg_sharpe = std_sharpe = avg_max_dd = avg_return = 0.0

        return WalkForwardResult(
            fold_results=fold_results,
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_max_dd,
            avg_return=avg_return,
            std_sharpe=std_sharpe,
            total_folds=len(fold_results),
        )


__all__ = [
    "WalkForwardOptimizer",
    "FoldResult",
    "WalkForwardResult",
    "MemoryConfig",
]
