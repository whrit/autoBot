"""
autoBot Optimizer - Rich Console Interface.

Provides a comprehensive CLI for running strategy optimization with:
- Live leaderboard during optimization
- GPU status display
- Progress bars for parameter sweeps
- Final results tables
- Export functionality

Usage:
    python -m optimizer_py.main --strategy trend --metric sharpe --grid

Example output:
    +-------------------------------------------------------------+
    |                   autoBot Optimizer                         |
    |              Strategy Family: TrendFollowing                |
    +-------------------------------------------------------------+

    Device: NVIDIA RTX 4090 | CUDA 12.1 | Memory: 24GB

    Parameter Sweep Progress
    Grid Search [145/200] ========== 72% | Best: 1.45 | 2:15 elapsed

    Live Leaderboard
    +------+--------+----------+-----------+--------+
    | Rank | Score  | lookback | threshold | vol_k  |
    +------+--------+----------+-----------+--------+
    | 1    | 1.4521 | 20       | 0.5       | 1.2    |
    | 2    | 1.3892 | 30       | 0.5       | 1.0    |
    | ...  | ...    | ...      | ...       | ...    |
    +------+--------+----------+-----------+--------+
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
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
from rich.text import Text

from optimizer_py.gpu_support import GPUConfig, GPUManager, TrainingDevice
from optimizer_py.logging_config import (
    OPTIMIZER_THEME,
    configure_logging,
    get_logger,
    get_tracker,
)
from optimizer_py.sweep import ParameterSweep, SweepConfig, SweepResult

logger = get_logger("optimizer.main")

# Create console with optimizer theme
console = Console(theme=OPTIMIZER_THEME)


@dataclass
class OptimizerRunConfig:
    """Configuration for an optimization run."""

    strategy_family: str
    parameter_grid: dict[str, list[Any]]
    metric: str = "sharpe"
    n_splits: int = 5
    search_type: str = "grid"
    n_iter: int = 100  # For random search
    data_path: Path | None = None
    output_dir: Path = Path("results")
    log_level: str = "INFO"


class OptimizerUI:
    """
    Rich console UI for the optimizer.

    Provides live-updating displays during optimization
    and formatted result tables upon completion.
    """

    def __init__(self, config: OptimizerRunConfig) -> None:
        """
        Initialize the UI.

        Args:
            config: Optimizer run configuration
        """
        self.config = config
        self.console = console
        self._start_time: datetime | None = None
        self._gpu_manager: GPUManager | None = None

    def show_header(self) -> None:
        """Display the application header."""
        header = Panel(
            Text.assemble(
                ("autoBot Optimizer\n", "bold header"),
                (f"Strategy Family: {self.config.strategy_family}", "cyan"),
            ),
            title="[bold white]Optimization Console[/bold white]",
            border_style="magenta",
        )
        self.console.print(header)
        self.console.print()

    def show_gpu_status(self) -> None:
        """Display GPU status information."""
        gpu_config = GPUConfig.from_env()
        self._gpu_manager = GPUManager(gpu_config)

        if self._gpu_manager.is_gpu_available():
            # Collect metadata for display
            metadata = self._gpu_manager.collect_metadata(0.0)

            gpu_info = Text.assemble(
                ("Device: ", "dim"),
                (f"{metadata.gpu_name or 'GPU'}", "gpu"),
                (" | ", "dim"),
                ("CUDA: ", "dim"),
                (f"{metadata.cuda_version or 'N/A'}", "cyan"),
                (" | ", "dim"),
                ("Memory: ", "dim"),
                (f"{metadata.gpu_memory_mb or 0}MB", "cyan"),
            )
        else:
            gpu_info = Text.assemble(
                ("Device: ", "dim"),
                ("CPU", "cpu"),
                (" (GPU not available)", "dim"),
            )

        panel = Panel(
            gpu_info,
            title="[bold]Compute Device[/bold]",
            border_style="yellow" if self._gpu_manager.is_gpu_available() else "dim",
        )
        self.console.print(panel)
        self.console.print()

    def create_leaderboard(
        self,
        results: list[dict[str, Any]],
        param_names: list[str],
        top_n: int = 10,
    ) -> Table:
        """Create a live leaderboard table."""
        table = Table(
            title="Live Leaderboard",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )

        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Score", justify="right", style="metric")

        for name in param_names:
            table.add_column(name, justify="right")

        sorted_results = sorted(
            results,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:top_n]

        for idx, result in enumerate(sorted_results):
            rank = str(idx + 1)
            score = f"{result.get('score', 0):.4f}"

            row = [rank, score]
            params = result.get("params", {})
            for name in param_names:
                value = params.get(name, "-")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))

            style = "best" if idx == 0 else None
            table.add_row(*row, style=style)

        return table

    def create_summary_panel(self, result: SweepResult) -> Panel:
        """Create a summary panel for final results."""
        stats = result.summary_stats()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row(
            "Best Score",
            Text(f"{result.best_score:.4f}", style="best"),
        )
        table.add_row("Combinations Tested", f"{int(stats.get('count', 0)):,}")
        table.add_row("Mean Score", f"{stats.get('mean', 0):.4f}")
        table.add_row("Std Dev", f"{stats.get('std', 0):.4f}")
        table.add_row("Median Score", f"{stats.get('median', 0):.4f}")
        table.add_row("Worst Score", f"{stats.get('worst', 0):.4f}")

        if result.cv_scores:
            table.add_section()
            table.add_row("CV Mean", f"{np.mean(result.cv_scores):.4f}")
            table.add_row("CV Std", f"{np.std(result.cv_scores):.4f}")

        return Panel(
            table,
            title=f"[bold white]Optimization Results - {self.config.strategy_family}[/bold white]",
            border_style="green",
        )

    def create_best_params_panel(self, result: SweepResult) -> Panel:
        """Create a panel showing best parameters."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Best Value", justify="right", style="best")

        for name, value in result.best_params.items():
            if isinstance(value, float):
                table.add_row(name, f"{value:.6f}")
            else:
                table.add_row(name, str(value))

        return Panel(
            table,
            title="[bold white]Best Parameters[/bold white]",
            border_style="magenta",
        )

    def show_results(self, result: SweepResult) -> None:
        """Display final optimization results."""
        self.console.print()

        # Summary panel
        self.console.print(self.create_summary_panel(result))
        self.console.print()

        # Best parameters
        self.console.print(self.create_best_params_panel(result))
        self.console.print()

        # Leaderboard
        self.console.print(result.leaderboard_table(top_n=15))
        self.console.print()

        # GPU training metadata if available
        if self._gpu_manager:
            elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            metadata = self._gpu_manager.collect_metadata(elapsed)

            device_text = (
                f"[gpu]{metadata.gpu_name}[/gpu]"
                if metadata.training_device == TrainingDevice.CUDA
                else "[cpu]CPU[/cpu]"
            )
            self.console.print(
                f"[dim]Optimization completed on {device_text} in {elapsed:.1f}s[/dim]"
            )

    def save_results(self, result: SweepResult, output_path: Path) -> None:
        """Save optimization results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        import json

        results_dict = {
            "strategy_family": self.config.strategy_family,
            "metric": self.config.metric,
            "search_type": self.config.search_type,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "cv_scores": result.cv_scores,
            "summary_stats": result.summary_stats(),
            "timestamp": datetime.now().isoformat(),
            "top_10": [
                {"params": r["params"], "score": r["score"]}
                for r in sorted(
                    result.all_results,
                    key=lambda x: x["score"],
                    reverse=True,
                )[:10]
            ],
        }

        output_path.write_text(json.dumps(results_dict, indent=2, default=str))
        self.console.print(f"[dim]Results saved to: {output_path}[/dim]")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="autoBot Optimizer - Strategy parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="trend",
        help="Strategy family name (default: trend)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="sharpe",
        choices=["sharpe", "sortino", "total_return"],
        help="Optimization metric (default: sharpe)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use grid search (exhaustive)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random search",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of iterations for random search (default: 100)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        help="Path to market data file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo data and parameters",
    )

    return parser.parse_args()


def generate_demo_data() -> pl.DataFrame:
    """Generate demo market data for testing."""
    np.random.seed(42)
    n_bars = 500

    timestamps = pl.datetime_range(
        datetime(2023, 1, 1),
        datetime(2023, 6, 30),
        interval="1h",
        eager=True,
    )[:n_bars]

    # Random walk with trend
    returns = np.random.normal(0.0002, 0.015, n_bars)
    close_prices = 100 * np.exp(np.cumsum(returns))

    # Add OHLCV data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100.0
    volumes = np.random.lognormal(10, 1, n_bars)

    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["DEMO"] * n_bars,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


def get_demo_parameter_grid() -> dict[str, list[Any]]:
    """Get demo parameter grid for testing."""
    return {
        "lookback": [10, 20, 30, 50],
        "threshold": [0.3, 0.5, 0.7, 1.0],
        "vol_multiplier": [0.8, 1.0, 1.2],
    }


# Demo strategy class for testing
class DemoStrategy:
    """Simple demo strategy for optimization testing."""

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.5,
        vol_multiplier: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self.vol_multiplier = vol_multiplier

    def generate_signals(self, data: pl.DataFrame) -> list[Any]:
        """Generate mock signals based on parameters."""
        from dataclasses import dataclass

        @dataclass
        class MockSignal:
            timestamp: datetime
            direction: float
            strength: float

        signals = []
        close = data["close"].to_numpy()
        timestamps = data["timestamp"].to_list()

        for i in range(self.lookback, len(close)):
            # Simple momentum signal
            momentum = close[i] / close[i - self.lookback] - 1
            if abs(momentum) > self.threshold * 0.01:
                direction = 1.0 if momentum > 0 else -1.0
                strength = min(abs(momentum) * self.vol_multiplier * 10, 1.0)
                signals.append(MockSignal(
                    timestamp=timestamps[i],
                    direction=direction,
                    strength=strength,
                ))

        return signals


def main() -> int:
    """Main entry point for the optimizer CLI."""
    args = parse_args()

    # Configure logging
    configure_logging(log_level=args.log_level)

    # Determine search type
    search_type = "random"
    if args.grid:
        search_type = "grid"

    # Create run config
    config = OptimizerRunConfig(
        strategy_family=args.strategy,
        parameter_grid={},  # Will be set below
        metric=args.metric,
        n_splits=args.splits,
        search_type=search_type,
        n_iter=args.n_iter,
        data_path=args.data,
        output_dir=args.output,
        log_level=args.log_level,
    )

    # Initialize UI
    ui = OptimizerUI(config)
    ui.show_header()
    ui.show_gpu_status()
    ui._start_time = datetime.now()

    try:
        if args.demo:
            console.print("[dim]Running with demo data and parameters...[/dim]\n")
            decision_frame = generate_demo_data()
            config.parameter_grid = get_demo_parameter_grid()
            strategy_class = DemoStrategy
        elif args.data:
            console.print(f"[dim]Loading data from {args.data}...[/dim]\n")
            if args.data.suffix == ".parquet":
                decision_frame = pl.read_parquet(args.data)
            else:
                decision_frame = pl.read_csv(args.data)
            # Would need real parameter grid - placeholder
            console.print("[yellow]Parameter grid not implemented - use --demo[/yellow]")
            return 1
        else:
            console.print("[yellow]No data specified. Use --data or --demo[/yellow]")
            return 1

        # Create sweep config
        sweep_config = SweepConfig(
            strategy_family=config.strategy_family,
            parameter_grid=config.parameter_grid,
            n_splits=config.n_splits,
            metric=config.metric,
        )

        # Create parameter sweep
        sweep = ParameterSweep(
            strategy_class=strategy_class,  # type: ignore
            config=sweep_config,
            console=console,
        )

        # Run optimization
        if search_type == "grid":
            result = sweep.grid_search(decision_frame, show_progress=True)
        else:
            result = sweep.random_search(
                decision_frame,
                n_iter=config.n_iter,
                show_progress=True,
            )

        # Show results
        ui.show_results(result)

        # Save results
        output_path = (
            config.output_dir
            / f"optimization_{config.strategy_family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        ui.save_results(result, output_path)

        logger.info(
            "optimization_complete",
            strategy_family=config.strategy_family,
            best_score=result.best_score,
            best_params=str(result.best_params),
        )

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("optimization_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
