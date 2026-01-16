"""
Parallel Parameter Grid Search Optimizer.

This module provides parallel parameter optimization for backtesting strategies:
- ProcessPoolExecutor for CPU-bound backtest runs
- Parameter grid generation with itertools.product
- Progress tracking with Rich progress bars
- Result caching for resumable searches
- Early stopping when threshold is met
- Memory limits per worker process
- Sorted results by configurable metric

Example usage:
    >>> from backtester_py import BacktestEngine, BacktestConfig
    >>> from backtester_py.parallel_optimizer import ParallelGridSearch
    >>>
    >>> optimizer = ParallelGridSearch(
    ...     engine=BacktestEngine(config),
    ...     max_workers=4,
    ... )
    >>>
    >>> results = optimizer.run_grid_search(
    ...     param_grid={
    ...         "lookback": [10, 20, 30, 50],
    ...         "threshold": [0.01, 0.02, 0.03],
    ...         "stop_loss": [0.02, 0.03, 0.05],
    ...     },
    ...     metric="sharpe_ratio",
    ...     min_trades=10,
    ... )
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import resource
import signal
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
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

    from backtester_py.engine import BacktestConfig, BacktestEngine, BacktestResult, Signal
    from backtester_py.metrics import BacktestMetrics

logger = get_logger("backtester.parallel_optimizer")


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB
    except (ImportError, AttributeError):
        return 0.0


def set_memory_limit_mb(limit_mb: float) -> None:
    """Set memory limit for the current process."""
    if limit_mb <= 0:
        return
    try:
        limit_bytes = int(limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error) as e:
        logger.warning("failed_to_set_memory_limit", limit_mb=limit_mb, error=str(e))


@dataclass
class GridSearchResult:
    """
    Result from a single parameter combination test.

    Attributes:
        params: Parameter dictionary used for this run
        sharpe_ratio: Sharpe ratio achieved
        sortino_ratio: Sortino ratio achieved
        max_drawdown: Maximum drawdown
        total_return: Total return
        profit_factor: Profit factor
        win_rate: Win rate
        num_trades: Number of trades executed
        total_pnl: Total P&L
        run_time_seconds: Time taken to run backtest
        metrics: Full metrics dictionary
        error: Error message if run failed
    """

    params: dict[str, Any]
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    total_pnl: float = 0.0
    run_time_seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if result is valid (no error and meets basic criteria)."""
        return self.error is None and self.num_trades > 0

    def get_metric(self, metric_name: str) -> float:
        """Get a specific metric value."""
        if hasattr(self, metric_name):
            return getattr(self, metric_name)
        return self.metrics.get(metric_name, 0.0)


@dataclass
class GridSearchSummary:
    """
    Summary of grid search optimization.

    Attributes:
        results: All results sorted by primary metric
        best_result: Best performing parameter combination
        total_combinations: Total parameter combinations tested
        successful_runs: Number of successful backtests
        failed_runs: Number of failed backtests
        total_time_seconds: Total optimization time
        early_stopped: Whether early stopping was triggered
        cache_hits: Number of results loaded from cache
    """

    results: list[GridSearchResult] = field(default_factory=list)
    best_result: GridSearchResult | None = None
    total_combinations: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_time_seconds: float = 0.0
    early_stopped: bool = False
    cache_hits: int = 0

    def top_n(self, n: int = 10) -> list[GridSearchResult]:
        """Get top N results."""
        return self.results[:n]

    def summary_table(self, console: Console | None = None, top_n: int = 10) -> Table:
        """Generate a Rich table summarizing grid search results."""
        table = Table(
            title=f"Grid Search Results (Top {min(top_n, len(self.results))})",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Sharpe", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Parameters", style="cyan")

        for idx, result in enumerate(self.results[:top_n]):
            ret_style = "green" if result.total_return > 0 else "red"
            dd_style = "red" if result.max_drawdown > 0.1 else "green"

            # Format parameters compactly
            params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
            if len(params_str) > 40:
                params_str = params_str[:37] + "..."

            table.add_row(
                str(idx + 1),
                f"{result.sharpe_ratio:.2f}",
                f"[{ret_style}]{result.total_return * 100:.1f}%[/{ret_style}]",
                f"[{dd_style}]{result.max_drawdown * 100:.1f}%[/{dd_style}]",
                f"{result.win_rate * 100:.1f}%",
                str(result.num_trades),
                params_str,
            )

        # Add summary row
        table.add_section()
        table.add_row(
            "STATS",
            f"Tested: {self.total_combinations}",
            f"OK: {self.successful_runs}",
            f"Fail: {self.failed_runs}",
            f"Cache: {self.cache_hits}",
            f"{self.total_time_seconds:.1f}s",
            "Early Stop" if self.early_stopped else "",
            style="bold",
        )

        return table


def _generate_param_hash(params: dict[str, Any]) -> str:
    """Generate a unique hash for parameter combination."""
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def _worker_run_backtest(
    params: dict[str, Any],
    market_data_bytes: bytes,
    signals_bytes: bytes,
    config_bytes: bytes,
    memory_limit_mb: float,
    worker_id: int,
) -> GridSearchResult:
    """
    Worker function to run a single backtest with given parameters.

    This function runs in a separate process and handles:
    - Memory limit enforcement
    - Pickle deserialization of inputs
    - Backtest execution with modified config
    - Metrics calculation

    Args:
        params: Parameter dictionary for this run
        market_data_bytes: Pickled market data DataFrame
        signals_bytes: Pickled signals list
        config_bytes: Pickled BacktestConfig
        memory_limit_mb: Memory limit for this worker
        worker_id: Worker identifier for logging

    Returns:
        GridSearchResult with metrics or error
    """
    import pickle

    start_time = time.time()

    try:
        # Set memory limit
        if memory_limit_mb > 0:
            set_memory_limit_mb(memory_limit_mb)

        # Deserialize inputs
        import polars as pl

        from backtester_py.engine import BacktestConfig, BacktestEngine
        from backtester_py.metrics import MetricsCalculator

        market_data: pl.DataFrame = pickle.loads(market_data_bytes)
        signals: list = pickle.loads(signals_bytes)
        base_config: BacktestConfig = pickle.loads(config_bytes)

        # Apply parameters to config
        config = _apply_params_to_config(base_config, params)

        # Create engine and run backtest
        engine = BacktestEngine(config, enable_logging=False)
        result = engine.run(market_data, signals)

        # Calculate metrics
        calculator = MetricsCalculator()
        trades = [{"pnl": f.notional} for f in result.fills]
        metrics = calculator.calculate(result.equity_curve, trades)

        run_time = time.time() - start_time

        return GridSearchResult(
            params=params,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            total_return=metrics.total_return,
            profit_factor=metrics.profit_factor,
            win_rate=metrics.win_rate,
            num_trades=metrics.num_trades,
            total_pnl=result.total_pnl,
            run_time_seconds=run_time,
            metrics={
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "cagr": metrics.cagr,
                "calmar_ratio": metrics.calmar_ratio,
            },
        )

    except MemoryError:
        return GridSearchResult(
            params=params,
            run_time_seconds=time.time() - start_time,
            error="Memory limit exceeded",
        )
    except Exception as e:
        return GridSearchResult(
            params=params,
            run_time_seconds=time.time() - start_time,
            error=str(e),
        )
    finally:
        gc.collect()


def _apply_params_to_config(config: BacktestConfig, params: dict[str, Any]) -> BacktestConfig:
    """
    Apply parameter dictionary to backtest config.

    Supports common parameters:
    - stop_loss_pct, take_profit_pct: Risk parameters
    - default_order_notional: Position sizing

    Args:
        config: Base BacktestConfig
        params: Parameters to apply

    Returns:
        Modified BacktestConfig (copy)
    """
    from dataclasses import replace

    # Create a copy with modified parameters
    updates = {}

    if "stop_loss" in params or "stop_loss_pct" in params:
        updates["stop_loss_pct"] = params.get("stop_loss") or params.get("stop_loss_pct")

    if "take_profit" in params or "take_profit_pct" in params:
        updates["take_profit_pct"] = params.get("take_profit") or params.get("take_profit_pct")

    if "order_notional" in params or "default_order_notional" in params:
        updates["default_order_notional"] = (
            params.get("order_notional") or params.get("default_order_notional")
        )

    if updates:
        return replace(config, **updates)

    return config


class ResultCache:
    """
    Cache for grid search results to enable resumable searches.

    Stores results on disk using pickle, indexed by parameter hash.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        cache_id: str | None = None,
    ) -> None:
        """
        Initialize result cache.

        Args:
            cache_dir: Directory to store cache files.
                      If None, uses system temp directory.
            cache_id: Unique identifier for this search.
                     If None, generates from timestamp.
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "backtester_grid_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_id = cache_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cache_file = self.cache_dir / f"grid_search_{self.cache_id}.pkl"
        self._cache: dict[str, GridSearchResult] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(
                    "cache_loaded",
                    cache_file=str(self.cache_file),
                    entries=len(self._cache),
                )
            except Exception as e:
                logger.warning("cache_load_failed", error=str(e))
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.warning("cache_save_failed", error=str(e))

    def get(self, params: dict[str, Any]) -> GridSearchResult | None:
        """Get cached result for parameters."""
        param_hash = _generate_param_hash(params)
        return self._cache.get(param_hash)

    def put(self, params: dict[str, Any], result: GridSearchResult) -> None:
        """Store result in cache."""
        param_hash = _generate_param_hash(params)
        self._cache[param_hash] = result
        # Save periodically (every 10 entries)
        if len(self._cache) % 10 == 0:
            self._save_cache()

    def save(self) -> None:
        """Force save cache to disk."""
        self._save_cache()

    def clear(self) -> None:
        """Clear cache."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def __len__(self) -> int:
        return len(self._cache)


class ParallelGridSearch:
    """
    Parallel parameter grid search optimizer for backtesting.

    Uses ProcessPoolExecutor to run backtests in parallel across
    parameter combinations. Supports early stopping, result caching,
    and memory limits per worker.

    Attributes:
        engine: BacktestEngine instance (used as template)
        max_workers: Maximum parallel workers
        memory_limit_mb: Memory limit per worker process
        cache: Result cache for resumable searches
    """

    def __init__(
        self,
        engine: BacktestEngine,
        max_workers: int = 4,
        memory_limit_mb: float = 0,
        cache_dir: Path | str | None = None,
        cache_id: str | None = None,
        console: Console | None = None,
    ) -> None:
        """
        Initialize ParallelGridSearch.

        Args:
            engine: BacktestEngine with base configuration
            max_workers: Maximum number of parallel worker processes.
                        Defaults to 4. Use -1 for CPU count.
            memory_limit_mb: Memory limit per worker in MB.
                            0 = unlimited.
            cache_dir: Directory for result caching.
                      None = system temp directory.
            cache_id: Unique ID for this search session.
                     None = auto-generated from timestamp.
            console: Rich console for progress display.

        Raises:
            ValueError: If max_workers is invalid
        """
        if max_workers == -1:
            max_workers = os.cpu_count() or 4
        if max_workers < 1:
            raise ValueError("max_workers must be positive or -1")

        self.engine = engine
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.console = console or Console()

        # Initialize cache
        self.cache = ResultCache(cache_dir=cache_dir, cache_id=cache_id)

        # Store serialized inputs for workers
        self._market_data_bytes: bytes | None = None
        self._signals_bytes: bytes | None = None
        self._config_bytes: bytes | None = None

    def _prepare_inputs(
        self,
        market_data: pl.DataFrame,
        signals: list[Signal],
    ) -> None:
        """Serialize inputs for worker processes."""
        self._market_data_bytes = pickle.dumps(market_data)
        self._signals_bytes = pickle.dumps(signals)
        self._config_bytes = pickle.dumps(self.engine.config)

    def _generate_combinations(
        self,
        param_grid: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate all parameter combinations from grid.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values

        Returns:
            List of parameter dictionaries
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def run_grid_search(
        self,
        param_grid: dict[str, list[Any]],
        market_data: pl.DataFrame,
        signals: list[Signal],
        metric: str = "sharpe_ratio",
        min_trades: int = 10,
        early_stop_threshold: float | None = None,
        ascending: bool = False,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> GridSearchSummary:
        """
        Run parallel grid search over parameter combinations.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
                       Example: {"lookback": [10, 20], "threshold": [0.01, 0.02]}
            market_data: Market data DataFrame for backtesting
            signals: List of trading signals
            metric: Metric to sort results by. Options:
                   "sharpe_ratio", "sortino_ratio", "total_return",
                   "max_drawdown", "profit_factor", "win_rate"
            min_trades: Minimum trades required for valid result
            early_stop_threshold: Stop search if metric exceeds this value.
                                 None = no early stopping.
            ascending: Sort results ascending (True) or descending (False).
                      Default False (best = highest).
            show_progress: Display Rich progress bar
            use_cache: Use result caching for resumable searches

        Returns:
            GridSearchSummary with sorted results and statistics

        Example:
            >>> results = optimizer.run_grid_search(
            ...     param_grid={
            ...         "lookback": [10, 20, 30, 50],
            ...         "threshold": [0.01, 0.02, 0.03],
            ...         "stop_loss": [0.02, 0.03, 0.05],
            ...     },
            ...     market_data=market_data,
            ...     signals=signals,
            ...     metric="sharpe_ratio",
            ...     min_trades=10,
            ...     early_stop_threshold=2.0,
            ... )
            >>> print(results.best_result.params)
        """
        start_time = time.time()

        # Generate all combinations
        combinations = self._generate_combinations(param_grid)
        total_combinations = len(combinations)

        if total_combinations == 0:
            logger.warning("empty_param_grid")
            return GridSearchSummary()

        logger.info(
            "grid_search_started",
            total_combinations=total_combinations,
            max_workers=self.max_workers,
            metric=metric,
            early_stop_threshold=early_stop_threshold,
        )

        # Prepare inputs for workers
        self._prepare_inputs(market_data, signals)

        # Track results
        results: list[GridSearchResult] = []
        cache_hits = 0
        early_stopped = False

        # Separate cached and uncached combinations
        uncached_combinations: list[dict[str, Any]] = []
        if use_cache:
            for params in combinations:
                cached = self.cache.get(params)
                if cached is not None:
                    results.append(cached)
                    cache_hits += 1
                else:
                    uncached_combinations.append(params)
        else:
            uncached_combinations = combinations

        logger.info(
            "cache_status",
            cache_hits=cache_hits,
            remaining=len(uncached_combinations),
        )

        # Run uncached combinations in parallel
        if uncached_combinations:
            if show_progress:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(complete_style="green"),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TextColumn("[cyan]Best: {task.fields[best]:.2f}[/cyan]"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                )

                with progress:
                    task = progress.add_task(
                        "Grid Search",
                        total=len(uncached_combinations),
                        best=0.0,
                    )

                    results, early_stopped = self._run_parallel(
                        uncached_combinations,
                        results,
                        metric,
                        min_trades,
                        early_stop_threshold,
                        use_cache,
                        progress,
                        task,
                    )
            else:
                results, early_stopped = self._run_parallel(
                    uncached_combinations,
                    results,
                    metric,
                    min_trades,
                    early_stop_threshold,
                    use_cache,
                    None,
                    None,
                )

        # Save cache
        if use_cache:
            self.cache.save()

        # Filter and sort results
        valid_results = [r for r in results if r.is_valid and r.num_trades >= min_trades]
        failed_runs = len(results) - len(valid_results)

        # Sort by metric
        reverse = not ascending
        valid_results.sort(key=lambda r: r.get_metric(metric), reverse=reverse)

        total_time = time.time() - start_time

        summary = GridSearchSummary(
            results=valid_results,
            best_result=valid_results[0] if valid_results else None,
            total_combinations=total_combinations,
            successful_runs=len(valid_results),
            failed_runs=failed_runs,
            total_time_seconds=total_time,
            early_stopped=early_stopped,
            cache_hits=cache_hits,
        )

        logger.info(
            "grid_search_completed",
            total_combinations=total_combinations,
            successful=summary.successful_runs,
            failed=summary.failed_runs,
            cache_hits=cache_hits,
            total_time_seconds=total_time,
            early_stopped=early_stopped,
            best_metric=summary.best_result.get_metric(metric) if summary.best_result else None,
        )

        return summary

    def _run_parallel(
        self,
        combinations: list[dict[str, Any]],
        results: list[GridSearchResult],
        metric: str,
        min_trades: int,
        early_stop_threshold: float | None,
        use_cache: bool,
        progress: Progress | None,
        task: Any | None,
    ) -> tuple[list[GridSearchResult], bool]:
        """Run parallel backtest execution."""
        early_stopped = False
        best_metric = float("-inf")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_params = {}
            for idx, params in enumerate(combinations):
                future = executor.submit(
                    _worker_run_backtest,
                    params,
                    self._market_data_bytes,
                    self._signals_bytes,
                    self._config_bytes,
                    self.memory_limit_mb,
                    idx,
                )
                future_to_params[future] = params

            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]

                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)

                    # Update cache
                    if use_cache and result.is_valid:
                        self.cache.put(params, result)

                    # Update best metric
                    if result.is_valid and result.num_trades >= min_trades:
                        metric_value = result.get_metric(metric)
                        if metric_value > best_metric:
                            best_metric = metric_value

                    # Update progress
                    if progress is not None and task is not None:
                        progress.update(
                            task,
                            advance=1,
                            best=best_metric if best_metric > float("-inf") else 0.0,
                        )

                    # Check early stopping
                    if (
                        early_stop_threshold is not None
                        and best_metric >= early_stop_threshold
                    ):
                        logger.info(
                            "early_stopping_triggered",
                            threshold=early_stop_threshold,
                            achieved=best_metric,
                        )
                        early_stopped = True
                        # Cancel remaining futures
                        for f in future_to_params:
                            f.cancel()
                        break

                except TimeoutError:
                    results.append(
                        GridSearchResult(params=params, error="Timeout exceeded")
                    )
                except Exception as e:
                    results.append(
                        GridSearchResult(params=params, error=str(e))
                    )

        return results, early_stopped

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache.clear()
        logger.info("cache_cleared")


def run_grid_search_cli(
    engine: BacktestEngine,
    market_data: pl.DataFrame,
    signals: list[Signal],
    param_grid: dict[str, list[Any]],
    max_workers: int = 4,
    metric: str = "sharpe_ratio",
    min_trades: int = 10,
    early_stop: float | None = None,
    memory_limit_mb: float = 0,
    cache_dir: str | None = None,
    no_cache: bool = False,
) -> GridSearchSummary:
    """
    CLI-friendly wrapper for grid search.

    This function provides a simple interface for running grid search
    from command-line tools or scripts.

    Args:
        engine: Configured BacktestEngine
        market_data: Market data DataFrame
        signals: Trading signals list
        param_grid: Parameter grid dictionary
        max_workers: Number of parallel workers
        metric: Sorting metric
        min_trades: Minimum trades filter
        early_stop: Early stopping threshold
        memory_limit_mb: Memory limit per worker
        cache_dir: Cache directory path
        no_cache: Disable caching

    Returns:
        GridSearchSummary with results
    """
    optimizer = ParallelGridSearch(
        engine=engine,
        max_workers=max_workers,
        memory_limit_mb=memory_limit_mb,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    return optimizer.run_grid_search(
        param_grid=param_grid,
        market_data=market_data,
        signals=signals,
        metric=metric,
        min_trades=min_trades,
        early_stop_threshold=early_stop,
        use_cache=not no_cache,
    )


__all__ = [
    "ParallelGridSearch",
    "GridSearchResult",
    "GridSearchSummary",
    "ResultCache",
    "run_grid_search_cli",
]
