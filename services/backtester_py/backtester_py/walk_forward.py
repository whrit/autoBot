"""
Walk-forward optimization framework.

This module provides walk-forward analysis for strategy evaluation:
- Rolling window splits with configurable train/test/step sizes
- No lookahead bias - train data always precedes test data
- Aggregation of results across windows
- Progress tracking with Rich progress bars
- Parallel processing with ProcessPoolExecutor for faster backtesting

Walk-forward analysis helps identify strategies that generalize
to unseen data, rather than those that only fit historical data.

Memory optimizations:
- Lazy data loading from Parquet files
- Generator-based splits to avoid loading all data
- Configurable memory limits
- Garbage collection between folds

Parallel processing:
- True parallelism via multiprocessing (bypasses GIL)
- Configurable max_workers (default: CPU count - 1)
- Per-worker memory monitoring
- Graceful shutdown on Ctrl+C
- Exception handling per fold with continuation
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import signal
import sys
import traceback
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
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

# Global flag for graceful shutdown
_shutdown_requested = False


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
class ParallelConfig:
    """Configuration for parallel walk-forward optimization.

    Attributes:
        max_workers: Maximum number of worker processes. Default is CPU count - 1.
                    Set to 1 to disable parallelism.
        timeout_per_fold: Timeout in seconds per fold (0 = no timeout).
        memory_config: Memory configuration per worker.
        enable_gc: Run garbage collection after each fold in workers.
        log_worker_memory: Log memory usage in each worker.
    """
    max_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    timeout_per_fold: float = 0.0
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    enable_gc: bool = True
    log_worker_memory: bool = False


@dataclass
class ParallelFoldInput:
    """Input data for a single fold to be processed in parallel.

    This dataclass is designed to be picklable for multiprocessing.

    Attributes:
        fold_index: Index of this fold (0-based).
        train_data_bytes: Serialized training data (Polars IPC format).
        test_data_bytes: Serialized test data (Polars IPC format).
    """
    fold_index: int
    train_data_bytes: bytes
    test_data_bytes: bytes


@dataclass
class ParallelFoldOutput:
    """Output from processing a single fold in parallel.

    Attributes:
        fold_index: Index of the fold.
        result: FoldResult if successful, None if failed.
        error: Error message if failed, None if successful.
        memory_usage_mb: Peak memory usage in MB during processing.
        processing_time_s: Time taken to process in seconds.
    """
    fold_index: int
    result: FoldResult | None
    error: str | None
    memory_usage_mb: float
    processing_time_s: float


class FoldEvaluator:
    """
    Picklable wrapper for fold evaluation functions.

    Use this class to wrap your evaluation logic for parallel execution.
    The class stores the necessary state and provides a __call__ method
    that can be pickled and sent to worker processes.

    Example:
        >>> class MyEvaluator(FoldEvaluator):
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def __call__(self, train_df, test_df, fold_idx):
        ...         engine = BacktestEngine(self.config)
        ...         result = engine.run(test_df, signals)
        ...         return FoldResult(...)
        >>>
        >>> evaluator = MyEvaluator(config)
        >>> optimizer.run_parallel(data, evaluator)
    """

    def __call__(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        fold_idx: int,
    ) -> "FoldResult":
        """
        Evaluate a single fold.

        Override this method in subclasses to implement custom evaluation logic.

        Args:
            train_df: Training data for this fold
            test_df: Test data for this fold
            fold_idx: Index of this fold (0-based)

        Returns:
            FoldResult with evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement __call__")


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("shutdown_requested", signal=signum)


def _process_fold_worker(
    fold_input: ParallelFoldInput,
    evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], "FoldResult"],
    enable_gc: bool = True,
    log_memory: bool = False,
) -> ParallelFoldOutput:
    """
    Worker function to process a single fold in a separate process.

    This function is designed to be called by ProcessPoolExecutor.
    It deserializes the data, runs the evaluation, and returns results.

    Args:
        fold_input: ParallelFoldInput with serialized data.
        evaluate_fn: Evaluation function to run on train/test data.
        enable_gc: Whether to run garbage collection after processing.
        log_memory: Whether to log memory usage.

    Returns:
        ParallelFoldOutput with results or error information.
    """
    import io
    import time
    import polars as pl

    start_time = time.perf_counter()
    peak_memory = get_memory_usage_mb()

    try:
        # Deserialize data from IPC format
        train_df = pl.read_ipc(io.BytesIO(fold_input.train_data_bytes))
        test_df = pl.read_ipc(io.BytesIO(fold_input.test_data_bytes))

        # Track memory after loading
        current_memory = get_memory_usage_mb()
        peak_memory = max(peak_memory, current_memory)

        if log_memory:
            logger.debug(
                "worker_data_loaded",
                fold=fold_input.fold_index,
                memory_mb=current_memory,
                train_rows=len(train_df),
                test_rows=len(test_df),
            )

        # Run evaluation
        result = evaluate_fn(train_df, test_df, fold_input.fold_index)

        # Track final memory
        current_memory = get_memory_usage_mb()
        peak_memory = max(peak_memory, current_memory)

        # Cleanup
        del train_df, test_df
        if enable_gc:
            gc.collect()

        processing_time = time.perf_counter() - start_time

        return ParallelFoldOutput(
            fold_index=fold_input.fold_index,
            result=result,
            error=None,
            memory_usage_mb=peak_memory,
            processing_time_s=processing_time,
        )

    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        logger.error(
            "worker_fold_failed",
            fold=fold_input.fold_index,
            error=str(e),
        )

        return ParallelFoldOutput(
            fold_index=fold_input.fold_index,
            result=None,
            error=error_msg,
            memory_usage_mb=peak_memory,
            processing_time_s=processing_time,
        )


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

    def run_parallel(
        self,
        data: pl.DataFrame,
        evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], FoldResult],
        parallel_config: ParallelConfig | None = None,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization with parallel processing.

        Uses ProcessPoolExecutor to distribute fold processing across multiple
        CPU cores, bypassing Python's GIL for true parallelism.

        Args:
            data: Full dataset for walk-forward analysis
            evaluate_fn: Function that takes (train_df, test_df, fold_idx)
                and returns a FoldResult. Must be picklable.
            parallel_config: Parallel execution configuration. If None,
                uses default ParallelConfig with CPU count - 1 workers.
            show_progress: Whether to display Rich progress bar

        Returns:
            WalkForwardResult with all fold results and aggregated metrics

        Note:
            - The evaluate_fn must be importable at the module level
              (not a lambda or local function) for multiprocessing to work.
            - Data is serialized to Polars IPC format for inter-process transfer.
            - Graceful shutdown on Ctrl+C: in-progress folds complete, pending
              folds are cancelled.
        """
        import io
        import time
        import numpy as np
        import polars as pl

        global _shutdown_requested
        _shutdown_requested = False

        parallel_config = parallel_config or ParallelConfig()
        n_splits = self.get_num_splits(len(data))

        if n_splits == 0:
            logger.warning(
                "insufficient_data_for_parallel_walk_forward",
                data_rows=len(data),
                required_rows=self.train_size + self.test_size,
            )
            return WalkForwardResult()

        # If only 1 worker, fall back to sequential execution
        if parallel_config.max_workers <= 1:
            logger.info(
                "parallel_disabled_single_worker",
                falling_back_to="sequential",
            )
            return self.run_with_progress(data, evaluate_fn, show_progress)

        logger.info(
            "parallel_walk_forward_started",
            total_folds=n_splits,
            max_workers=parallel_config.max_workers,
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
        )

        # Prepare fold inputs by serializing data upfront
        fold_inputs: list[ParallelFoldInput] = []

        for fold_idx, (train, test) in enumerate(self.split(data)):
            # Serialize to IPC format (efficient for Polars)
            train_buffer = io.BytesIO()
            test_buffer = io.BytesIO()
            train.write_ipc(train_buffer)
            test.write_ipc(test_buffer)

            fold_inputs.append(ParallelFoldInput(
                fold_index=fold_idx,
                train_data_bytes=train_buffer.getvalue(),
                test_data_bytes=test_buffer.getvalue(),
            ))

        logger.debug(
            "parallel_data_serialized",
            num_folds=len(fold_inputs),
            total_data_size_mb=sum(
                len(fi.train_data_bytes) + len(fi.test_data_bytes)
                for fi in fold_inputs
            ) / (1024 * 1024),
        )

        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.signal(signal.SIGINT, _signal_handler)
        original_sigterm_handler = signal.signal(signal.SIGTERM, _signal_handler)

        # Results storage
        fold_results: list[FoldResult] = [None] * n_splits  # type: ignore
        errors: list[tuple[int, str]] = []
        peak_memory_mb = 0.0
        total_processing_time = 0.0
        completed_count = 0
        start_time = time.perf_counter()

        try:
            # Use 'spawn' context to avoid issues with fork and threading
            mp_context = mp.get_context('spawn')

            with ProcessPoolExecutor(
                max_workers=parallel_config.max_workers,
                mp_context=mp_context,
            ) as executor:
                # Submit all folds
                future_to_fold: dict[Future[ParallelFoldOutput], int] = {}

                for fold_input in fold_inputs:
                    if _shutdown_requested:
                        break

                    future = executor.submit(
                        _process_fold_worker,
                        fold_input,
                        evaluate_fn,
                        parallel_config.enable_gc,
                        parallel_config.log_worker_memory,
                    )
                    future_to_fold[future] = fold_input.fold_index

                logger.debug(
                    "parallel_folds_submitted",
                    submitted=len(future_to_fold),
                )

                # Set up progress display
                if show_progress:
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TextColumn("[cyan]Completed: {task.fields[completed]}/{task.fields[total]}[/cyan]"),
                        TextColumn("[green]OK: {task.fields[ok]}[/green]"),
                        TextColumn("[red]Err: {task.fields[err]}[/red]"),
                        TextColumn("[yellow]Mem: {task.fields[mem]:.0f}MB[/yellow]"),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    )

                    with progress:
                        task = progress.add_task(
                            "Parallel Walk-Forward",
                            total=n_splits,
                            completed=0,
                            ok=0,
                            err=0,
                            mem=0.0,
                        )

                        # Process results as they complete
                        for future in as_completed(future_to_fold):
                            if _shutdown_requested:
                                # Cancel remaining futures
                                for f in future_to_fold:
                                    if not f.done():
                                        f.cancel()
                                logger.warning(
                                    "parallel_shutdown_cancelling_futures",
                                    completed=completed_count,
                                    total=n_splits,
                                )
                                break

                            fold_idx = future_to_fold[future]

                            try:
                                timeout = (
                                    parallel_config.timeout_per_fold
                                    if parallel_config.timeout_per_fold > 0
                                    else None
                                )
                                output = future.result(timeout=timeout)

                                peak_memory_mb = max(peak_memory_mb, output.memory_usage_mb)
                                total_processing_time += output.processing_time_s

                                if output.result is not None:
                                    fold_results[fold_idx] = output.result
                                    completed_count += 1

                                    progress.update(
                                        task,
                                        advance=1,
                                        completed=completed_count,
                                        ok=completed_count - len(errors),
                                        err=len(errors),
                                        mem=peak_memory_mb,
                                    )

                                    logger.debug(
                                        "parallel_fold_completed",
                                        fold=fold_idx,
                                        sharpe=output.result.sharpe_ratio,
                                        time_s=output.processing_time_s,
                                        memory_mb=output.memory_usage_mb,
                                    )
                                else:
                                    errors.append((fold_idx, output.error or "Unknown error"))
                                    completed_count += 1

                                    progress.update(
                                        task,
                                        advance=1,
                                        completed=completed_count,
                                        ok=completed_count - len(errors),
                                        err=len(errors),
                                        mem=peak_memory_mb,
                                    )

                                    logger.warning(
                                        "parallel_fold_error",
                                        fold=fold_idx,
                                        error=output.error[:200] if output.error else "Unknown",
                                    )

                            except TimeoutError:
                                errors.append((fold_idx, "Timeout exceeded"))
                                completed_count += 1
                                logger.warning(
                                    "parallel_fold_timeout",
                                    fold=fold_idx,
                                    timeout_s=parallel_config.timeout_per_fold,
                                )
                            except Exception as e:
                                errors.append((fold_idx, str(e)))
                                completed_count += 1
                                logger.error(
                                    "parallel_fold_exception",
                                    fold=fold_idx,
                                    error=str(e),
                                )
                else:
                    # No progress display - just process results
                    for future in as_completed(future_to_fold):
                        if _shutdown_requested:
                            for f in future_to_fold:
                                if not f.done():
                                    f.cancel()
                            break

                        fold_idx = future_to_fold[future]

                        try:
                            timeout = (
                                parallel_config.timeout_per_fold
                                if parallel_config.timeout_per_fold > 0
                                else None
                            )
                            output = future.result(timeout=timeout)

                            peak_memory_mb = max(peak_memory_mb, output.memory_usage_mb)
                            total_processing_time += output.processing_time_s

                            if output.result is not None:
                                fold_results[fold_idx] = output.result
                                completed_count += 1
                            else:
                                errors.append((fold_idx, output.error or "Unknown error"))
                                completed_count += 1

                        except TimeoutError:
                            errors.append((fold_idx, "Timeout exceeded"))
                            completed_count += 1
                        except Exception as e:
                            errors.append((fold_idx, str(e)))
                            completed_count += 1

        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)

        # Filter out None results (failed folds)
        successful_results = [fr for fr in fold_results if fr is not None]

        # Log errors if any
        if errors:
            logger.warning(
                "parallel_folds_with_errors",
                error_count=len(errors),
                error_folds=[e[0] for e in errors],
            )

        # Calculate aggregated metrics
        if successful_results:
            sharpes = [fr.sharpe_ratio for fr in successful_results]
            avg_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            avg_max_dd = float(np.mean([fr.max_drawdown for fr in successful_results]))
            avg_return = float(np.mean([fr.total_return for fr in successful_results]))
        else:
            avg_sharpe = std_sharpe = avg_max_dd = avg_return = 0.0

        result = WalkForwardResult(
            fold_results=successful_results,
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_max_dd,
            avg_return=avg_return,
            std_sharpe=std_sharpe,
            total_folds=len(successful_results),
        )

        wall_time = time.perf_counter() - start_time
        speedup = total_processing_time / wall_time if wall_time > 0 else 1.0

        logger.info(
            "parallel_walk_forward_completed",
            total_folds=result.total_folds,
            failed_folds=len(errors),
            avg_sharpe=result.avg_sharpe,
            std_sharpe=result.std_sharpe,
            peak_memory_mb=peak_memory_mb,
            wall_time_s=wall_time,
            total_cpu_time_s=total_processing_time,
            speedup=speedup,
            workers=parallel_config.max_workers,
        )

        return result

    def run_parallel_lazy(
        self,
        data_path: str | Path,
        evaluate_fn: Callable[[pl.DataFrame, pl.DataFrame, int], FoldResult],
        parallel_config: ParallelConfig | None = None,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run parallel walk-forward optimization with lazy data loading.

        Combines lazy data loading with parallel processing for optimal
        memory usage and performance on large datasets.

        Args:
            data_path: Path to Parquet file or directory
            evaluate_fn: Function that takes (train_df, test_df, fold_idx)
                and returns a FoldResult. Must be picklable.
            parallel_config: Parallel execution configuration.
            show_progress: Whether to display Rich progress bar

        Returns:
            WalkForwardResult with all fold results and aggregated metrics

        Note:
            Data is loaded lazily per fold, then serialized for worker transfer.
            This uses more I/O but less main process memory compared to run_parallel.
        """
        import io
        import time
        import numpy as np
        import polars as pl

        global _shutdown_requested
        _shutdown_requested = False

        parallel_config = parallel_config or ParallelConfig()

        # Get row count for split calculation
        lazy_df = pl.scan_parquet(data_path)
        n_rows = lazy_df.select(pl.len()).collect().item()
        n_splits = self.get_num_splits(n_rows)

        if n_splits == 0:
            logger.warning(
                "insufficient_data_for_parallel_lazy_walk_forward",
                data_rows=n_rows,
                required_rows=self.train_size + self.test_size,
            )
            return WalkForwardResult()

        if parallel_config.max_workers <= 1:
            logger.info(
                "parallel_disabled_single_worker",
                falling_back_to="lazy_sequential",
            )
            return self.run_lazy(
                data_path, evaluate_fn,
                parallel_config.memory_config,
                show_progress,
            )

        logger.info(
            "parallel_lazy_walk_forward_started",
            total_folds=n_splits,
            max_workers=parallel_config.max_workers,
            data_path=str(data_path),
        )

        # Set up signal handlers
        original_sigint_handler = signal.signal(signal.SIGINT, _signal_handler)
        original_sigterm_handler = signal.signal(signal.SIGTERM, _signal_handler)

        fold_results: list[FoldResult] = [None] * n_splits  # type: ignore
        errors: list[tuple[int, str]] = []
        peak_memory_mb = 0.0
        total_processing_time = 0.0
        completed_count = 0
        start_time = time.perf_counter()

        try:
            mp_context = mp.get_context('spawn')

            with ProcessPoolExecutor(
                max_workers=parallel_config.max_workers,
                mp_context=mp_context,
            ) as executor:
                future_to_fold: dict[Future[ParallelFoldOutput], int] = {}

                # Submit folds with lazy loading
                for fold_idx, (train, test) in enumerate(self.split_lazy(data_path)):
                    if _shutdown_requested:
                        break

                    # Serialize data
                    train_buffer = io.BytesIO()
                    test_buffer = io.BytesIO()
                    train.write_ipc(train_buffer)
                    test.write_ipc(test_buffer)

                    fold_input = ParallelFoldInput(
                        fold_index=fold_idx,
                        train_data_bytes=train_buffer.getvalue(),
                        test_data_bytes=test_buffer.getvalue(),
                    )

                    # Free memory in main process
                    del train, test
                    gc.collect()

                    future = executor.submit(
                        _process_fold_worker,
                        fold_input,
                        evaluate_fn,
                        parallel_config.enable_gc,
                        parallel_config.log_worker_memory,
                    )
                    future_to_fold[future] = fold_idx

                # Process results (similar to run_parallel)
                if show_progress:
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TextColumn("[cyan]Done: {task.fields[completed]}/{task.fields[total]}[/cyan]"),
                        TextColumn("[yellow]Mem: {task.fields[mem]:.0f}MB[/yellow]"),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    )

                    with progress:
                        task = progress.add_task(
                            "Parallel Lazy Walk-Forward",
                            total=n_splits,
                            completed=0,
                            mem=0.0,
                        )

                        for future in as_completed(future_to_fold):
                            if _shutdown_requested:
                                for f in future_to_fold:
                                    if not f.done():
                                        f.cancel()
                                break

                            fold_idx = future_to_fold[future]

                            try:
                                timeout = (
                                    parallel_config.timeout_per_fold
                                    if parallel_config.timeout_per_fold > 0
                                    else None
                                )
                                output = future.result(timeout=timeout)

                                peak_memory_mb = max(peak_memory_mb, output.memory_usage_mb)
                                total_processing_time += output.processing_time_s

                                if output.result is not None:
                                    fold_results[fold_idx] = output.result
                                    completed_count += 1
                                else:
                                    errors.append((fold_idx, output.error or "Unknown"))
                                    completed_count += 1

                                progress.update(
                                    task,
                                    advance=1,
                                    completed=completed_count,
                                    mem=peak_memory_mb,
                                )

                            except Exception as e:
                                errors.append((fold_idx, str(e)))
                                completed_count += 1
                else:
                    for future in as_completed(future_to_fold):
                        if _shutdown_requested:
                            for f in future_to_fold:
                                if not f.done():
                                    f.cancel()
                            break

                        fold_idx = future_to_fold[future]

                        try:
                            timeout = (
                                parallel_config.timeout_per_fold
                                if parallel_config.timeout_per_fold > 0
                                else None
                            )
                            output = future.result(timeout=timeout)

                            peak_memory_mb = max(peak_memory_mb, output.memory_usage_mb)
                            total_processing_time += output.processing_time_s

                            if output.result is not None:
                                fold_results[fold_idx] = output.result
                                completed_count += 1
                            else:
                                errors.append((fold_idx, output.error or "Unknown"))
                                completed_count += 1

                        except Exception as e:
                            errors.append((fold_idx, str(e)))
                            completed_count += 1

        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)

        # Filter and aggregate results
        successful_results = [fr for fr in fold_results if fr is not None]

        if errors:
            logger.warning(
                "parallel_lazy_folds_with_errors",
                error_count=len(errors),
                error_folds=[e[0] for e in errors],
            )

        if successful_results:
            sharpes = [fr.sharpe_ratio for fr in successful_results]
            avg_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            avg_max_dd = float(np.mean([fr.max_drawdown for fr in successful_results]))
            avg_return = float(np.mean([fr.total_return for fr in successful_results]))
        else:
            avg_sharpe = std_sharpe = avg_max_dd = avg_return = 0.0

        result = WalkForwardResult(
            fold_results=successful_results,
            avg_sharpe=avg_sharpe,
            avg_max_drawdown=avg_max_dd,
            avg_return=avg_return,
            std_sharpe=std_sharpe,
            total_folds=len(successful_results),
        )

        wall_time = time.perf_counter() - start_time
        speedup = total_processing_time / wall_time if wall_time > 0 else 1.0

        logger.info(
            "parallel_lazy_walk_forward_completed",
            total_folds=result.total_folds,
            failed_folds=len(errors),
            avg_sharpe=result.avg_sharpe,
            peak_memory_mb=peak_memory_mb,
            wall_time_s=wall_time,
            speedup=speedup,
        )

        return result


__all__ = [
    "WalkForwardOptimizer",
    "FoldResult",
    "WalkForwardResult",
    "MemoryConfig",
    "ParallelConfig",
    "ParallelFoldInput",
    "ParallelFoldOutput",
    "FoldEvaluator",
]
