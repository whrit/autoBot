"""
autoBot Backtester - Rich Console Interface.

Provides a comprehensive CLI for running backtests with:
- Live updating metrics during backtest
- Walk-forward progress bars
- Final results tables
- Equity curve visualization
- HTML report generation

Usage:
    python -m backtester_py.main --strategy TrendFollowing_v1 --data market_data.parquet

Example output:
    +-------------------------------------------------------------+
    |                   autoBot Backtester                        |
    |              Strategy: TrendFollowing_v1                    |
    +-------------------------------------------------------------+

    Walk-Forward Backtest
    Fold 1/5 ================ 100% | Sharpe: 1.45 | MDD: -8.2%
    Fold 2/5 ================ 100% | Sharpe: 1.32 | MDD: -9.1%
    ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from backtester_py.logging_config import TRADING_THEME, configure_logging, get_logger
from backtester_py.models import (
    CostModel,
    CostModelType,
    FixedCostModel,
    RiskChecker,
    VolumeImpactCostModel,
    ZeroCostModel,
    create_cost_model,
    create_risk_checker,
)
from backtester_py.visualization import (
    generate_equity_chart,
    generate_html_report,
    generate_trade_summary_table,
)

if TYPE_CHECKING:
    from backtester_py.engine import BacktestResult

logger = get_logger("backtester.main")


def load_from_lake(
    lake_path: Path,
    symbols: list[str],
    data_type: str = "bars_provider",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Load data from Hive-partitioned lake directory.

    Supports reading from lake/raw/{data_type}/dt=YYYY-MM-DD/symbol=XXX/data.parquet

    Args:
        lake_path: Path to lake directory (e.g., "./lake")
        symbols: List of symbols to load (e.g., ["SPY", "QQQ"])
        data_type: Type of data to load ("bars_provider", "trades", "quotes")
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Combined DataFrame with data from all symbols
    """
    import glob

    frames: list[pl.DataFrame] = []

    for symbol in symbols:
        # Pattern for Hive-partitioned data
        pattern = str(lake_path / "raw" / data_type / "*" / f"symbol={symbol}" / "*.parquet")
        matching_files = glob.glob(pattern)

        if not matching_files:
            logger.warning(f"No data found for {symbol} in {lake_path}/raw/{data_type}/")
            continue

        logger.info(f"Loading {len(matching_files)} partition(s) for {symbol}")

        # Scan with Hive partitioning
        lf = pl.scan_parquet(matching_files, hive_partitioning=True)

        # Apply date filters if provided (ts_event is timezone-aware UTC)
        if start_date:
            start_dt = datetime.fromisoformat(f"{start_date}T00:00:00+00:00")
            lf = lf.filter(pl.col("ts_event") >= start_dt)
        if end_date:
            end_dt = datetime.fromisoformat(f"{end_date}T23:59:59+00:00")
            lf = lf.filter(pl.col("ts_event") <= end_dt)

        # Collect the data
        df = lf.collect(engine="streaming")
        frames.append(df)

    if not frames:
        logger.error(f"No data found for any symbols in {lake_path}")
        return pl.DataFrame()

    # Combine all symbols and sort by timestamp
    combined = pl.concat(frames)

    # Sort by timestamp (ts_event is the standard column name from ingestor)
    if "ts_event" in combined.columns:
        combined = combined.sort("ts_event")
    elif "timestamp" in combined.columns:
        combined = combined.sort("timestamp")

    return combined


def transform_lake_data_for_backtest(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform lake data format to backtester expected format.

    Lake data has: ts_event, symbol, open, high, low, close, volume
    Backtester expects: timestamp, symbol, bid_price, ask_price, vol

    Args:
        df: Raw lake data

    Returns:
        Transformed DataFrame for backtester
    """
    if df.is_empty():
        return df

    # Check if already in backtester format
    if "bid_price" in df.columns and "ask_price" in df.columns:
        return df

    # Transform from OHLCV to bid/ask format
    # Use close price with a small spread for simulation
    spread_pct = 0.0001  # 1 basis point spread

    result = df.select([
        # Rename ts_event to timestamp
        (pl.col("ts_event") if "ts_event" in df.columns else pl.col("timestamp")).alias("timestamp"),
        pl.col("symbol"),
        # Use close price with spread
        (pl.col("close") * (1 - spread_pct / 2)).alias("bid_price"),
        (pl.col("close") * (1 + spread_pct / 2)).alias("ask_price"),
        # Calculate volatility from high-low range if available
        (
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
            if "high" in df.columns and "low" in df.columns
            else pl.lit(0.001)
        ).alias("vol"),
    ])

    return result

# Create console with trading theme
console = Console(theme=TRADING_THEME)


@dataclass
class BacktestRunConfig:
    """Configuration for a backtest run."""

    strategy_name: str
    data_path: Path | None = None
    initial_capital: float = 100_000.0
    walk_forward_folds: int = 5
    output_dir: Path = Path("reports")
    generate_html: bool = True
    show_trades: bool = True
    log_level: str = "INFO"
    parallel: bool = False
    max_workers: int | None = None  # None = auto (CPU count - 1)


class BacktesterUI:
    """
    Rich console UI for the backtester.

    Provides live-updating displays during backtest execution
    and formatted result tables upon completion.
    """

    def __init__(self, config: BacktestRunConfig) -> None:
        """
        Initialize the UI.

        Args:
            config: Backtest run configuration
        """
        self.config = config
        self.console = console
        self._start_time: datetime | None = None
        self._current_metrics: dict[str, float] = {}

    def show_header(self) -> None:
        """Display the application header."""
        header = Panel(
            Text.assemble(
                ("autoBot Backtester\n", "bold header"),
                (f"Strategy: {self.config.strategy_name}", "cyan"),
            ),
            title="[bold white]Backtest Console[/bold white]",
            border_style="cyan",
        )
        self.console.print(header)
        self.console.print()

    def create_metrics_table(self, metrics: dict[str, float]) -> Table:
        """Create a live metrics table."""
        table = Table(
            title="Live Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )

        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Format metrics with appropriate styling
        total_return = metrics.get("total_return", 0.0)
        return_style = "profit" if total_return >= 0 else "loss"

        table.add_row(
            "Total Return",
            Text(f"{total_return * 100:+.2f}%", style=return_style),
        )
        table.add_row(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0.0):.2f}",
        )

        mdd = metrics.get("max_drawdown", 0.0)
        mdd_style = "loss" if mdd > 0.1 else "dim"
        table.add_row(
            "Max Drawdown",
            Text(f"{mdd * 100:.1f}%", style=mdd_style),
        )

        table.add_row(
            "Win Rate",
            f"{metrics.get('win_rate', 0.0) * 100:.1f}%",
        )
        table.add_row(
            "Trades",
            f"{int(metrics.get('num_trades', 0)):,}",
        )

        return table

    def create_final_results_panel(
        self,
        result: BacktestResult,
        metrics: dict[str, float],
    ) -> Panel:
        """Create the final results panel."""
        # Create layout with multiple tables
        layout = Layout()

        # Main metrics table
        metrics_table = Table(
            title="Performance Summary",
            show_header=True,
            header_style="bold magenta",
            title_style="bold white",
        )

        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")

        # Add all metrics
        total_return = metrics.get("total_return", 0.0)
        return_style = "green bold" if total_return >= 0 else "red bold"

        metrics_table.add_row(
            "Total Return",
            Text(f"{total_return * 100:+.2f}%", style=return_style),
        )
        metrics_table.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
        metrics_table.add_row("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.2f}")
        metrics_table.add_row(
            "Max Drawdown",
            Text(f"{metrics.get('max_drawdown', 0.0) * 100:.1f}%", style="red"),
        )
        metrics_table.add_row(
            "Max DD Duration",
            f"{int(metrics.get('max_drawdown_duration', 0))} bars",
        )
        metrics_table.add_row("Profit Factor", f"{metrics.get('profit_factor', 0.0):.2f}")
        metrics_table.add_row("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.1f}%")
        metrics_table.add_row("CAGR", f"{metrics.get('cagr', 0.0) * 100:.2f}%")
        metrics_table.add_row("Calmar Ratio", f"{metrics.get('calmar_ratio', 0.0):.2f}")
        metrics_table.add_row("Total Trades", f"{result.total_trades:,}")

        if result.risk_violations:
            metrics_table.add_section()
            metrics_table.add_row(
                "Risk Violations",
                Text(str(len(result.risk_violations)), style="red bold"),
            )

        return Panel(
            metrics_table,
            title=f"[bold white]Backtest Results - {self.config.strategy_name}[/bold white]",
            border_style="green" if total_return >= 0 else "red",
        )

    def run_with_live_display(
        self,
        run_backtest_fn: Any,
        market_data: pl.DataFrame,
        signals: list[Any],
    ) -> tuple[BacktestResult, dict[str, float]]:
        """
        Run backtest with live-updating display.

        Args:
            run_backtest_fn: Function to execute backtest
            market_data: Market data DataFrame
            signals: Trading signals

        Returns:
            Tuple of (BacktestResult, metrics_dict)
        """
        self._start_time = datetime.now()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        metrics_table = self.create_metrics_table({})

        def progress_callback(current: int, total: int, metrics: dict[str, float]) -> None:
            """Update progress and metrics during backtest."""
            self._current_metrics = metrics
            progress.update(
                task,
                completed=current,
                description=f"Processing signals [{current}/{total}]",
            )

        with Live(
            Panel(
                Group(progress, metrics_table),
                title="[bold]Running Backtest[/bold]",
                border_style="cyan",
            ),
            console=self.console,
            refresh_per_second=4,
        ) as live:
            task = progress.add_task("Initializing...", total=len(signals))

            result = run_backtest_fn(
                market_data,
                signals,
                progress_callback=progress_callback,
            )

            # Calculate final metrics
            from backtester_py.metrics import MetricsCalculator

            calculator = MetricsCalculator()
            trades = [{"pnl": f.notional} for f in result.fills]
            final_metrics = calculator.calculate(result.equity_curve, trades)

            metrics_dict = {
                "total_return": final_metrics.total_return,
                "sharpe_ratio": final_metrics.sharpe_ratio,
                "sortino_ratio": final_metrics.sortino_ratio,
                "max_drawdown": final_metrics.max_drawdown,
                "max_drawdown_duration": final_metrics.max_drawdown_duration,
                "profit_factor": final_metrics.profit_factor,
                "win_rate": final_metrics.win_rate,
                "cagr": final_metrics.cagr,
                "calmar_ratio": final_metrics.calmar_ratio,
            }

            progress.update(task, completed=len(signals), description="Complete!")

        return result, metrics_dict

    def show_results(
        self,
        result: BacktestResult,
        metrics: dict[str, float],
    ) -> None:
        """Display final backtest results."""
        self.console.print()

        # Show results panel
        results_panel = self.create_final_results_panel(result, metrics)
        self.console.print(results_panel)
        self.console.print()

        # Show equity curve
        if not result.equity_curve.is_empty():
            generate_equity_chart(result.equity_curve, self.console)
            self.console.print()

        # Show recent trades
        if self.config.show_trades and result.fills:
            generate_trade_summary_table(result.fills, self.console, max_rows=15)
            self.console.print()

        # Generate HTML report
        if self.config.generate_html:
            output_path = (
                self.config.output_dir
                / f"backtest_{self.config.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            generate_html_report(
                result,
                metrics,
                self.config.strategy_name,
                output_path,
            )
            self.console.print(
                f"[dim]HTML report saved to: {output_path}[/dim]"
            )

    def show_walk_forward_results(
        self,
        wf_result: Any,
    ) -> None:
        """Display walk-forward optimization results."""
        self.console.print()
        self.console.print(wf_result.summary_table())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="autoBot Backtester - Strategy backtesting with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="DefaultStrategy",
        help="Strategy name for the backtest",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        help="Path to market data file (Parquet or CSV)",
    )
    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=100_000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--folds",
        "-f",
        type=int,
        default=5,
        help="Number of walk-forward folds (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Skip trade summary display",
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
        help="Run with demo data",
    )

    # Lake data loading arguments
    parser.add_argument(
        "--lake",
        type=Path,
        help="Path to data lake directory (e.g., ./lake)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to load (e.g., SPY,QQQ)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="bars_provider",
        help="Type of data to load from lake (bars_provider, trades, quotes)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY-MM-DD)",
    )

    # Parallel walk-forward arguments
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Enable parallel processing for walk-forward optimization",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout per fold in seconds (default: 0 = no timeout)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward optimization instead of single backtest",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="Training window size in bars for walk-forward (default: 500)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Test window size in bars for walk-forward (default: 100)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Step size in bars for walk-forward (default: 50)",
    )

    # Grid search arguments
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run parallel parameter grid search optimization",
    )
    parser.add_argument(
        "--grid-workers",
        type=int,
        default=4,
        help="Number of parallel workers for grid search (default: 4, -1 for CPU count)",
    )
    parser.add_argument(
        "--grid-metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor", "win_rate"],
        help="Metric to optimize in grid search (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--grid-min-trades",
        type=int,
        default=10,
        help="Minimum trades required for valid grid search result (default: 10)",
    )
    parser.add_argument(
        "--grid-early-stop",
        type=float,
        default=None,
        help="Early stop threshold for grid search metric",
    )
    parser.add_argument(
        "--grid-memory-limit",
        type=float,
        default=0,
        help="Memory limit per worker in MB (0 = unlimited)",
    )
    parser.add_argument(
        "--grid-cache-dir",
        type=Path,
        default=None,
        help="Directory for grid search result caching",
    )
    parser.add_argument(
        "--grid-no-cache",
        action="store_true",
        help="Disable grid search result caching",
    )
    parser.add_argument(
        "--grid-params",
        type=str,
        default=None,
        help="JSON string of parameter grid, e.g., '{\"lookback\": [10, 20], \"threshold\": [0.01, 0.02]}'",
    )

    # Cost model arguments
    parser.add_argument(
        "--cost-model",
        type=str,
        choices=["fixed", "volume", "zero"],
        default="fixed",
        help="Cost model type (default: fixed)",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Fee rate as decimal (0.001 = 0.1%%, default: 0.001)",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0005,
        help="Slippage rate as decimal (0.0005 = 0.05%%, default: 0.0005)",
    )
    parser.add_argument(
        "--spread-coef",
        type=float,
        default=1.0,
        help="Spread coefficient for volume cost model (default: 1.0)",
    )
    parser.add_argument(
        "--size-coef",
        type=float,
        default=1.0,
        help="Size coefficient for volume cost model (default: 1.0)",
    )
    parser.add_argument(
        "--vol-coef",
        type=float,
        default=1.0,
        help="Volatility coefficient for volume cost model (default: 1.0)",
    )

    # Risk checker arguments
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.10,
        help="Maximum position as %% of capital (0.10 = 10%%, default: 0.10)",
    )
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.20,
        help="Maximum drawdown as %% of peak equity (0.20 = 20%%, default: 0.20)",
    )
    parser.add_argument(
        "--max-gross-pct",
        type=float,
        default=1.0,
        help="Maximum gross exposure as %% of capital (1.0 = 100%%, default: 1.0)",
    )

    return parser.parse_args()


def generate_demo_data() -> tuple[pl.DataFrame, list[Any]]:
    """Generate demo market data and signals for testing."""
    import numpy as np

    from backtester_py.engine import Signal, SignalType

    # Generate synthetic price data
    np.random.seed(42)
    n_bars = 1000

    timestamps = pl.datetime_range(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31),
        interval="1h",
        eager=True,
    )[:n_bars]

    # Random walk with trend
    returns = np.random.normal(0.0001, 0.01, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))

    spread = 0.001
    market_data = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["DEMO"] * n_bars,
        "bid_price": prices * (1 - spread / 2),
        "ask_price": prices * (1 + spread / 2),
        "vol": np.abs(np.random.normal(0.001, 0.0005, n_bars)),
    })

    # Generate signals (simple momentum)
    signals = []
    for i in range(10, n_bars, 20):
        momentum = prices[i] / prices[i - 10] - 1
        signal_type = SignalType.LONG if momentum > 0 else SignalType.SHORT

        signals.append(Signal(
            timestamp=timestamps[i],
            symbol="DEMO",
            signal_type=signal_type,
            strength=min(abs(momentum) * 100, 1.0),
            target_notional=10_000.0,
        ))

    return market_data, signals


def generate_demo_signals_from_data(
    market_data: pl.DataFrame,
) -> tuple[pl.DataFrame, list[Any]]:
    """
    Generate demo momentum signals from actual market data.

    Uses simple price momentum to generate trading signals.

    Args:
        market_data: Market data with timestamp, symbol, bid_price, ask_price

    Returns:
        Tuple of (market_data, signals)
    """
    import numpy as np

    from backtester_py.engine import Signal, SignalType

    signals = []

    # Get unique symbols
    symbols = market_data["symbol"].unique().to_list()

    for symbol in symbols:
        symbol_data = market_data.filter(pl.col("symbol") == symbol)

        if len(symbol_data) < 20:
            continue

        timestamps = symbol_data["timestamp"].to_list()
        # Use mid price
        prices = (
            (symbol_data["bid_price"] + symbol_data["ask_price"]) / 2
        ).to_numpy()

        # Generate signals every 20 bars using momentum
        for i in range(20, len(prices), 20):
            momentum = prices[i] / prices[i - 10] - 1
            signal_type = SignalType.LONG if momentum > 0 else SignalType.SHORT

            signals.append(Signal(
                timestamp=timestamps[i],
                symbol=symbol,
                signal_type=signal_type,
                strength=min(abs(momentum) * 100, 1.0),
                target_notional=10_000.0,
            ))

    logger.info(f"Generated {len(signals)} signals from market data")
    return market_data, signals


class BacktestFoldEvaluator:
    """
    Picklable evaluator for walk-forward backtesting.

    This class stores the backtest configuration and signals,
    and can be pickled for multiprocessing.
    """

    def __init__(
        self,
        backtest_config: Any,
        signals: list[Any],
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            backtest_config: BacktestConfig instance
            signals: List of trading signals
        """
        self.backtest_config = backtest_config
        self.signals = signals

    def __call__(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        fold_idx: int,
    ) -> Any:
        """
        Evaluate a single fold.

        Args:
            train_df: Training data
            test_df: Test data
            fold_idx: Fold index

        Returns:
            FoldResult with metrics
        """
        from backtester_py.engine import BacktestEngine
        from backtester_py.metrics import MetricsCalculator
        from backtester_py.walk_forward import FoldResult

        # Create a fresh engine for this fold
        fold_engine = BacktestEngine(self.backtest_config, enable_logging=False)

        # Filter signals for this time period
        test_start = test_df["timestamp"].min()
        test_end = test_df["timestamp"].max()

        test_signals = [
            s for s in self.signals
            if test_start <= s.timestamp <= test_end
        ]

        # Run backtest on test data
        if test_signals:
            result = fold_engine.run(test_df, test_signals)

            # Calculate metrics
            calculator = MetricsCalculator()
            trades = [{"pnl": f.notional} for f in result.fills]

            if not result.equity_curve.is_empty():
                metrics = calculator.calculate(result.equity_curve, trades)
                return FoldResult(
                    fold_index=fold_idx,
                    train_size=len(train_df),
                    test_size=len(test_df),
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    total_return=metrics.total_return,
                    win_rate=metrics.win_rate,
                    num_trades=metrics.num_trades,
                )

        # Return empty result if no signals
        return FoldResult(
            fold_index=fold_idx,
            train_size=len(train_df),
            test_size=len(test_df),
        )


def run_walk_forward_mode(
    args: argparse.Namespace,
    engine: Any,
    market_data: pl.DataFrame,
    signals: list[Any],
    config: BacktestRunConfig,
    ui: BacktesterUI,
    backtest_config: Any,
) -> int:
    """
    Run walk-forward optimization mode.

    Supports both sequential and parallel execution.

    Args:
        args: Parsed command-line arguments
        engine: BacktestEngine instance
        market_data: Market data DataFrame
        signals: Trading signals
        config: Backtest run configuration
        ui: BacktesterUI instance
        backtest_config: BacktestConfig instance

    Returns:
        Exit code (0 for success)
    """
    import os
    from backtester_py.walk_forward import (
        WalkForwardOptimizer,
        ParallelConfig,
    )

    console.print(
        f"[bold cyan]Walk-Forward Optimization[/bold cyan] "
        f"({'parallel' if config.parallel else 'sequential'} mode)\n"
    )
    console.print(f"  Train size: {args.train_size} bars")
    console.print(f"  Test size: {args.test_size} bars")
    console.print(f"  Step size: {args.step_size} bars")

    if config.parallel:
        workers = config.max_workers or max(1, (os.cpu_count() or 2) - 1)
        console.print(f"  Workers: {workers}")
        if args.timeout > 0:
            console.print(f"  Timeout: {args.timeout}s per fold")
    console.print()

    # Create walk-forward optimizer
    optimizer = WalkForwardOptimizer(
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        console=console,
    )

    # Create picklable evaluator (required for parallel multiprocessing)
    evaluator = BacktestFoldEvaluator(backtest_config, signals)

    # Run optimization
    if config.parallel:
        parallel_config = ParallelConfig(
            max_workers=config.max_workers or max(1, (os.cpu_count() or 2) - 1),
            timeout_per_fold=args.timeout,
            log_worker_memory=args.log_level == "DEBUG",
        )

        wf_result = optimizer.run_parallel(
            market_data,
            evaluator,
            parallel_config=parallel_config,
            show_progress=True,
        )
    else:
        wf_result = optimizer.run_with_progress(
            market_data,
            evaluator,
            show_progress=True,
        )

    # Display results
    ui.show_walk_forward_results(wf_result)

    # Log summary
    logger.info(
        "walk_forward_complete",
        strategy=config.strategy_name,
        total_folds=wf_result.total_folds,
        avg_sharpe=wf_result.avg_sharpe,
        std_sharpe=wf_result.std_sharpe,
        avg_return=wf_result.avg_return,
        avg_max_drawdown=wf_result.avg_max_drawdown,
        parallel=config.parallel,
    )

    return 0


def run_grid_search_mode(
    args: argparse.Namespace,
    engine: Any,
    market_data: pl.DataFrame,
    signals: list[Any],
    config: BacktestRunConfig,
) -> int:
    """
    Run grid search optimization mode.

    Args:
        args: Parsed command-line arguments
        engine: BacktestEngine instance
        market_data: Market data DataFrame
        signals: Trading signals
        config: Backtest run configuration

    Returns:
        Exit code (0 for success)
    """
    import json

    from backtester_py.parallel_optimizer import ParallelGridSearch

    console.print("\n[bold cyan]Grid Search Optimization Mode[/bold cyan]\n")

    # Parse parameter grid
    if args.grid_params:
        try:
            param_grid = json.loads(args.grid_params)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing --grid-params JSON: {e}[/red]")
            return 1
    else:
        # Default demo parameter grid
        console.print("[dim]Using default demo parameter grid...[/dim]")
        param_grid = {
            "stop_loss": [0.01, 0.02, 0.03, 0.05],
            "take_profit": [0.02, 0.04, 0.06, 0.10],
        }

    # Display grid info
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    console.print("[cyan]Parameter Grid:[/cyan]")
    for param, values in param_grid.items():
        console.print(f"  {param}: {values}")
    console.print(f"\n[cyan]Total combinations:[/cyan] {total_combinations}")
    console.print(f"[cyan]Workers:[/cyan] {args.grid_workers}")
    console.print(f"[cyan]Metric:[/cyan] {args.grid_metric}")
    console.print(f"[cyan]Min trades:[/cyan] {args.grid_min_trades}")
    if args.grid_early_stop:
        console.print(f"[cyan]Early stop threshold:[/cyan] {args.grid_early_stop}")
    if args.grid_memory_limit > 0:
        console.print(f"[cyan]Memory limit:[/cyan] {args.grid_memory_limit} MB")
    console.print()

    # Create optimizer
    optimizer = ParallelGridSearch(
        engine=engine,
        max_workers=args.grid_workers,
        memory_limit_mb=args.grid_memory_limit,
        cache_dir=args.grid_cache_dir,
        console=console,
    )

    # Run grid search
    summary = optimizer.run_grid_search(
        param_grid=param_grid,
        market_data=market_data,
        signals=signals,
        metric=args.grid_metric,
        min_trades=args.grid_min_trades,
        early_stop_threshold=args.grid_early_stop,
        use_cache=not args.grid_no_cache,
    )

    # Display results
    console.print()
    console.print(summary.summary_table(console, top_n=15))

    if summary.best_result:
        console.print("\n[bold green]Best Parameters:[/bold green]")
        for param, value in summary.best_result.params.items():
            console.print(f"  {param}: {value}")
        console.print(
            f"\n[bold]Best {args.grid_metric}:[/bold] "
            f"{summary.best_result.get_metric(args.grid_metric):.4f}"
        )

    # Save results to file if output dir specified
    if config.output_dir:
        output_path = (
            config.output_dir
            / f"grid_search_{config.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)

        results_json = {
            "strategy": config.strategy_name,
            "metric": args.grid_metric,
            "total_combinations": summary.total_combinations,
            "successful_runs": summary.successful_runs,
            "failed_runs": summary.failed_runs,
            "total_time_seconds": summary.total_time_seconds,
            "early_stopped": summary.early_stopped,
            "best_params": summary.best_result.params if summary.best_result else None,
            "best_metrics": summary.best_result.metrics if summary.best_result else None,
            "top_results": [
                {
                    "params": r.params,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "num_trades": r.num_trades,
                }
                for r in summary.top_n(20)
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)

        console.print(f"\n[dim]Results saved to: {output_path}[/dim]")

    logger.info(
        "grid_search_complete",
        strategy=config.strategy_name,
        total_combinations=summary.total_combinations,
        successful=summary.successful_runs,
        best_metric=(
            summary.best_result.get_metric(args.grid_metric) if summary.best_result else None
        ),
    )

    return 0


import warnings


class MockCostModel:
    """
    DEPRECATED: Simple cost model for demo/testing purposes.

    Use backtester_py.models.FixedCostModel or backtester_py.models.VolumeImpactCostModel instead.

    This class is kept for backward compatibility with multiprocessing pickling,
    but will be removed in a future version.

    Picklable (module-level) for multiprocessing compatibility.
    """

    fixed_cost_bps = 1.0

    def __init__(self) -> None:
        warnings.warn(
            "MockCostModel is deprecated. Use backtester_py.models.FixedCostModel or "
            "backtester_py.models.create_cost_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def calculate_fill_price(
        self,
        side: str,
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """Calculate fill price with slippage based on volatility."""
        slippage = short_term_vol * 0.1
        if side == "buy":
            return ask_price * (1 + slippage)
        return bid_price * (1 - slippage)


class MockRiskChecker:
    """
    DEPRECATED: Pass-through risk checker for demo/testing purposes.

    Use backtester_py.models.RiskChecker or backtester_py.models.create_risk_checker() instead.

    This class is kept for backward compatibility with multiprocessing pickling,
    but will be removed in a future version.

    Picklable (module-level) for multiprocessing compatibility.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_gross_exposure: float = 1_000_000.0,
        max_drawdown_pct: float = 0.20,
    ) -> None:
        warnings.warn(
            "MockRiskChecker is deprecated. Use backtester_py.models.RiskChecker or "
            "backtester_py.models.create_risk_checker() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_position_pct = max_position_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_drawdown_pct = max_drawdown_pct

    def check_order(
        self,
        symbol: str,
        side: str,
        notional: float,
        risk_state: Any,
    ) -> list[Any]:
        """Check order against risk limits (always passes in mock)."""
        return []


def main() -> int:
    """Main entry point for the backtester CLI."""
    args = parse_args()

    # Configure logging
    configure_logging(log_level=args.log_level)

    # Create run config
    config = BacktestRunConfig(
        strategy_name=args.strategy,
        data_path=args.data,
        initial_capital=args.capital,
        walk_forward_folds=args.folds,
        output_dir=args.output,
        generate_html=not args.no_html,
        show_trades=not args.no_trades,
        log_level=args.log_level,
        parallel=args.parallel,
        max_workers=args.workers,
    )

    # Initialize UI
    ui = BacktesterUI(config)
    ui.show_header()

    try:
        signals = None  # Will be set if demo mode or generated from strategy

        if args.demo:
            console.print("[dim]Running with demo data...[/dim]\n")
            market_data, signals = generate_demo_data()
        elif args.lake and args.symbols:
            # Load from Hive-partitioned lake directory
            symbols = [s.strip() for s in args.symbols.split(",")]
            console.print(f"[dim]Loading from lake: {args.lake}[/dim]")
            console.print(f"[dim]Symbols: {', '.join(symbols)}[/dim]")
            if args.start_date:
                console.print(f"[dim]Start date: {args.start_date}[/dim]")
            if args.end_date:
                console.print(f"[dim]End date: {args.end_date}[/dim]")
            console.print()

            raw_data = load_from_lake(
                lake_path=args.lake,
                symbols=symbols,
                data_type=args.data_type,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            if raw_data.is_empty():
                console.print("[red]No data found in lake for specified symbols[/red]")
                return 1

            console.print(f"[dim]Loaded {len(raw_data):,} rows[/dim]\n")

            # Transform lake data to backtester format
            market_data = transform_lake_data_for_backtest(raw_data)

            # Generate demo signals since lake data doesn't include signals
            console.print("[dim]Generating momentum signals from data...[/dim]\n")
            _, signals = generate_demo_signals_from_data(market_data)

        elif args.data:
            console.print(f"[dim]Loading data from {args.data}...[/dim]\n")
            if args.data.suffix == ".parquet":
                market_data = pl.read_parquet(args.data)
            else:
                market_data = pl.read_csv(args.data)
            # Would need signals from strategy - placeholder
            console.print("[yellow]Signal generation not implemented - use --demo[/yellow]")
            return 1
        else:
            console.print("[yellow]No data specified. Use --data, --lake with --symbols, or --demo[/yellow]")
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  --demo                          Run with synthetic demo data[/dim]")
            console.print("[dim]  --lake lake --symbols SPY,QQQ   Load from Hive-partitioned lake[/dim]")
            console.print("[dim]  --data market.parquet           Load from single file[/dim]")
            return 1

        # Create cost and risk models from CLI arguments
        from backtester_py.engine import BacktestConfig, BacktestEngine

        # Create cost model based on CLI args
        cost_model = create_cost_model(
            model_type=args.cost_model,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            spread_coef=args.spread_coef,
            size_coef=args.size_coef,
            vol_coef=args.vol_coef,
        )

        # Log cost model configuration
        console.print(f"[dim]Cost model: {args.cost_model}[/dim]")
        if args.cost_model != "zero":
            console.print(f"[dim]  Fee rate: {args.fee_rate:.4%}[/dim]")
            if args.cost_model == "fixed":
                console.print(f"[dim]  Slippage rate: {args.slippage_rate:.4%}[/dim]")
            elif args.cost_model == "volume":
                console.print(f"[dim]  Spread coef: {args.spread_coef}, Size coef: {args.size_coef}, Vol coef: {args.vol_coef}[/dim]")

        # Create risk checker from CLI args
        risk_checker = create_risk_checker(
            max_position_pct=args.max_position_pct,
            max_gross_exposure=config.initial_capital * args.max_gross_pct,
            max_drawdown_pct=args.max_drawdown_pct,
            initial_capital=config.initial_capital,
        )

        # Log risk configuration
        console.print(f"[dim]Risk limits: Position {args.max_position_pct:.0%}, Gross {args.max_gross_pct:.0%}, Drawdown {args.max_drawdown_pct:.0%}[/dim]\n")

        backtest_config = BacktestConfig(
            initial_capital=config.initial_capital,
            cost_model=cost_model,  # type: ignore[arg-type]
            risk_checker=risk_checker,  # type: ignore[arg-type]
        )

        engine = BacktestEngine(backtest_config)

        # Check if grid search mode
        if args.grid_search:
            return run_grid_search_mode(args, engine, market_data, signals, config)

        # Check if walk-forward mode
        if args.walk_forward:
            return run_walk_forward_mode(
                args, engine, market_data, signals, config, ui, backtest_config
            )

        # Run backtest with live display
        result, metrics = ui.run_with_live_display(
            engine.run,
            market_data,
            signals,
        )

        # Show results
        ui.show_results(result, metrics)

        logger.info(
            "backtest_complete",
            strategy=config.strategy_name,
            total_return=metrics.get("total_return", 0.0),
            sharpe=metrics.get("sharpe_ratio", 0.0),
        )

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Backtest interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("backtest_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
