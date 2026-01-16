"""
Feature Builder CLI Entry Point.

Provides a rich console interface for building features across multiple symbols.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.decision_frame import DecisionFrameBuilder
from feature_builder_py.micro_bars import MicrostructureBarBuilder

# Configure structlog for rich console output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

console = Console()


@dataclass
class BuildStats:
    """Statistics for a feature build operation."""

    symbol: str
    bar_type: str
    input_rows: int = 0
    output_rows: int = 0
    processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class BuildSummary:
    """Summary of all build operations."""

    symbols: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    total_bars: int = 0
    total_frames: int = 0
    total_time: float = 0.0
    stats: list[BuildStats] = field(default_factory=list)


class FeatureBuilderCLI:
    """CLI interface for building features with rich progress display."""

    def __init__(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        data_dir: Path | str = Path("data"),
        output_dir: Path | str = Path("output"),
        verbose: bool = False,
    ) -> None:
        """
        Initialize the CLI builder.

        Args:
            symbols: List of stock symbols to process.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            data_dir: Directory containing input data.
            output_dir: Directory for output files.
            verbose: Enable verbose logging.
        """
        self.symbols = symbols
        self.start_date = start_date or "2025-10-18"
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Initialize builders
        self.bar_builders = {
            "1m": StandardBarBuilder(granularity="1m"),
            "5m": StandardBarBuilder(granularity="5m"),
            "15m": StandardBarBuilder(granularity="15m"),
        }
        self.micro_builder = MicrostructureBarBuilder(granularity="30s")
        self.decision_builder = DecisionFrameBuilder()

        # Statistics
        self.summary = BuildSummary(
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self._log = logger.bind(component="FeatureBuilderCLI")

    def _display_header(self) -> None:
        """Display the application header."""
        header = Panel(
            Text("autoBot Feature Builder", justify="center", style="bold cyan"),
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(header)
        console.print()

        # Display configuration
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Label", style="dim")
        config_table.add_column("Value", style="bold")
        config_table.add_row("Building features for:", ", ".join(self.symbols))
        config_table.add_row("Date range:", f"{self.start_date} to {self.end_date}")
        console.print(config_table)
        console.print()

    def _load_trades(self, symbol: str) -> pl.DataFrame:
        """Load trade data for a symbol."""
        trade_path = self.data_dir / f"{symbol}_trades.parquet"
        if trade_path.exists():
            return pl.read_parquet(trade_path)
        # Return sample data for demonstration
        return self._generate_sample_trades(symbol)

    def _load_quotes(self, symbol: str) -> pl.DataFrame:
        """Load quote data for a symbol."""
        quote_path = self.data_dir / f"{symbol}_quotes.parquet"
        if quote_path.exists():
            return pl.read_parquet(quote_path)
        # Return sample data for demonstration
        return self._generate_sample_quotes(symbol)

    def _generate_sample_trades(self, symbol: str, n_rows: int = 10000) -> pl.DataFrame:
        """Generate sample trade data for demonstration."""
        import random
        from datetime import timedelta

        base_price = 450.0 if symbol == "SPY" else 380.0
        base_time = datetime.fromisoformat(f"{self.start_date}T09:30:00")

        timestamps = []
        prices = []
        sizes = []

        current_price = base_price
        current_time = base_time

        for _ in range(n_rows):
            current_time += timedelta(seconds=random.uniform(0.1, 2.0))
            current_price += random.gauss(0, 0.05)
            timestamps.append(current_time)
            prices.append(current_price)
            sizes.append(random.randint(1, 1000))

        return pl.DataFrame({
            "ts_event": timestamps,
            "ts_recv": timestamps,
            "price": prices,
            "size": sizes,
            "exchange": ["XNAS"] * n_rows,
            "conditions": ["@"] * n_rows,
        }).with_columns([
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("ts_recv").dt.replace_time_zone("UTC"),
        ])

    def _generate_sample_quotes(self, symbol: str, n_rows: int = 20000) -> pl.DataFrame:
        """Generate sample quote data for demonstration."""
        import random
        from datetime import timedelta

        base_price = 450.0 if symbol == "SPY" else 380.0
        base_time = datetime.fromisoformat(f"{self.start_date}T09:30:00")

        timestamps = []
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []

        current_mid = base_price
        current_time = base_time

        for _ in range(n_rows):
            current_time += timedelta(seconds=random.uniform(0.05, 0.5))
            current_mid += random.gauss(0, 0.02)
            spread = random.uniform(0.01, 0.05)

            timestamps.append(current_time)
            bid_prices.append(current_mid - spread / 2)
            ask_prices.append(current_mid + spread / 2)
            bid_sizes.append(random.randint(100, 5000))
            ask_sizes.append(random.randint(100, 5000))

        return pl.DataFrame({
            "ts_event": timestamps,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        }).with_columns([
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        ])

    def _generate_decision_times(
        self, bars_1m: pl.DataFrame, interval_minutes: int = 5
    ) -> pl.DataFrame:
        """Generate decision times from 1m bars."""
        if bars_1m.is_empty():
            return pl.DataFrame({"decision_ts": pl.Series([], dtype=pl.Datetime("us", "UTC"))})

        # Take every Nth bar as a decision point
        decision_times = (
            bars_1m
            .select("bar_end")
            .rename({"bar_end": "decision_ts"})
            .gather_every(interval_minutes)
        )
        return decision_times

    def build_all(self) -> BuildSummary:
        """Build all features for all symbols."""
        self._display_header()
        start_time = time.perf_counter()

        # Create progress displays
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )

        with Live(progress, console=console, refresh_per_second=10):
            # Build standard bars
            console.print("[bold]Standard Bars (1m, 5m, 15m)[/bold]")
            self._build_standard_bars_all(progress)

            # Build microstructure bars
            console.print()
            console.print("[bold]Microstructure Bars (30s)[/bold]")
            self._build_micro_bars_all(progress)

            # Build decision frames
            console.print()
            console.print("[bold]Decision Frame[/bold]")
            self._build_decision_frames_all(progress)

        # Calculate totals
        self.summary.total_time = time.perf_counter() - start_time

        # Display summary
        self._display_summary()

        return self.summary

    def _build_standard_bars_all(self, progress: Progress) -> None:
        """Build standard bars for all symbols."""
        for symbol in self.symbols:
            task_id = progress.add_task(f"{symbol}", total=3)  # 3 timeframes

            trades = self._load_trades(symbol)
            symbol_bars = 0

            for timeframe, builder in self.bar_builders.items():
                start = time.perf_counter()
                bars = builder.build(trades, symbol)
                elapsed = time.perf_counter() - start

                output_rows = len(bars)
                symbol_bars += output_rows
                self.summary.total_bars += output_rows

                self.summary.stats.append(BuildStats(
                    symbol=symbol,
                    bar_type=f"standard_{timeframe}",
                    input_rows=len(trades),
                    output_rows=output_rows,
                    processing_time=elapsed,
                ))

                progress.advance(task_id)

            progress.update(
                task_id,
                description=f"{symbol} [green]{symbol_bars:,} bars[/green]",
                completed=3,
            )

    def _build_micro_bars_all(self, progress: Progress) -> None:
        """Build microstructure bars for all symbols."""
        for symbol in self.symbols:
            task_id = progress.add_task(f"{symbol}", total=1)

            trades = self._load_trades(symbol)
            quotes = self._load_quotes(symbol)

            start = time.perf_counter()
            bars = self.micro_builder.build(trades, quotes, symbol)
            elapsed = time.perf_counter() - start

            output_rows = len(bars)
            self.summary.total_bars += output_rows

            self.summary.stats.append(BuildStats(
                symbol=symbol,
                bar_type="micro_30s",
                input_rows=len(trades) + len(quotes),
                output_rows=output_rows,
                processing_time=elapsed,
            ))

            progress.update(
                task_id,
                description=f"{symbol} [green]{output_rows:,} bars[/green]",
                completed=1,
            )

    def _build_decision_frames_all(self, progress: Progress) -> None:
        """Build decision frames for all symbols."""
        for symbol in self.symbols:
            task_id = progress.add_task(f"{symbol}", total=1)

            trades = self._load_trades(symbol)
            quotes = self._load_quotes(symbol)

            # Build all required bar types
            bars_1m = self.bar_builders["1m"].build(trades, symbol)
            bars_5m = self.bar_builders["5m"].build(trades, symbol)
            bars_15m = self.bar_builders["15m"].build(trades, symbol)
            micro_30s = self.micro_builder.build(trades, quotes, symbol)

            # Generate decision times
            decision_times = self._generate_decision_times(bars_1m)

            start = time.perf_counter()
            frames = self.decision_builder.build(
                decision_times=decision_times,
                micro_bars_30s=micro_30s,
                bars_1m=bars_1m,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
                symbol=symbol,
            )
            elapsed = time.perf_counter() - start

            output_rows = len(frames)
            self.summary.total_frames += output_rows

            self.summary.stats.append(BuildStats(
                symbol=symbol,
                bar_type="decision_frame",
                input_rows=len(decision_times),
                output_rows=output_rows,
                processing_time=elapsed,
            ))

            progress.update(
                task_id,
                description=f"{symbol} [green]{output_rows:,} frames[/green]",
                completed=1,
            )

    def _display_summary(self) -> None:
        """Display build summary."""
        console.print()

        # Success message
        console.print("[bold green]Feature build complete[/bold green]")

        # Statistics table
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="bold")
        stats_table.add_row("Total bars:", f"{self.summary.total_bars:,}")
        stats_table.add_row("Total frames:", f"{self.summary.total_frames:,}")
        stats_table.add_row("Processing time:", f"{self.summary.total_time:.1f}s")

        console.print(stats_table)

        # Detailed stats if verbose
        if self.verbose:
            console.print()
            console.print("[bold]Detailed Statistics[/bold]")

            detail_table = Table()
            detail_table.add_column("Symbol")
            detail_table.add_column("Type")
            detail_table.add_column("Input Rows", justify="right")
            detail_table.add_column("Output Rows", justify="right")
            detail_table.add_column("Time (s)", justify="right")
            detail_table.add_column("Throughput", justify="right")

            for stat in self.summary.stats:
                throughput = (
                    stat.input_rows / stat.processing_time
                    if stat.processing_time > 0
                    else 0
                )
                detail_table.add_row(
                    stat.symbol,
                    stat.bar_type,
                    f"{stat.input_rows:,}",
                    f"{stat.output_rows:,}",
                    f"{stat.processing_time:.3f}",
                    f"{throughput:,.0f}/s",
                )

            console.print(detail_table)


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build trading features for autoBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  feature-builder SPY QQQ
  feature-builder SPY --start-date 2025-10-01 --end-date 2025-12-31
  feature-builder SPY QQQ IWM --verbose
        """,
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to process",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Input data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parsed = parser.parse_args(args)

    try:
        cli = FeatureBuilderCLI(
            symbols=parsed.symbols,
            start_date=parsed.start_date,
            end_date=parsed.end_date,
            data_dir=parsed.data_dir,
            output_dir=parsed.output_dir,
            verbose=parsed.verbose,
        )
        cli.build_all()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
