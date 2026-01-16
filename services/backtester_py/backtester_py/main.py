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
from backtester_py.visualization import (
    generate_equity_chart,
    generate_html_report,
    generate_trade_summary_table,
)

if TYPE_CHECKING:
    from backtester_py.engine import BacktestResult

logger = get_logger("backtester.main")

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
    )

    # Initialize UI
    ui = BacktesterUI(config)
    ui.show_header()

    try:
        if args.demo:
            console.print("[dim]Running with demo data...[/dim]\n")
            market_data, signals = generate_demo_data()
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
            console.print("[yellow]No data specified. Use --data or --demo[/yellow]")
            return 1

        # Create minimal cost/risk models for demo
        # In real usage, these would be injected from configuration
        from backtester_py.engine import BacktestConfig, BacktestEngine

        # Create mock models for demo (would be real models in production)
        class MockCostModel:
            fixed_cost_bps = 1.0

            def calculate_fill_price(
                self,
                side: str,
                ask_price: float,
                bid_price: float,
                order_notional: float,
                book_notional: float,
                short_term_vol: float,
            ) -> float:
                slippage = short_term_vol * 0.1
                if side == "buy":
                    return ask_price * (1 + slippage)
                return bid_price * (1 - slippage)

        class MockRiskChecker:
            def check_order(
                self,
                symbol: str,
                side: str,
                notional: float,
                risk_state: Any,
            ) -> list[Any]:
                return []

        backtest_config = BacktestConfig(
            initial_capital=config.initial_capital,
            cost_model=MockCostModel(),  # type: ignore
            risk_checker=MockRiskChecker(),  # type: ignore
        )

        engine = BacktestEngine(backtest_config)

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
