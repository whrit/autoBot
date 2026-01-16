"""
Ingestor Service Main Entry Point.

Supports two modes:
- backfill: Download historical data for configured symbols and date range.
- stream: Real-time streaming of trades and quotes via WebSocket.

Includes Rich console output with banners, configuration tables,
progress bars, and summary statistics.
"""

import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestor_py.backfill import BackfillOrchestrator, BackfillStats
from ingestor_py.logging_config import (
    configure_logging,
    format_bytes,
    format_duration,
    format_number,
    get_logger,
)
from ingestor_py.streaming import RealtimeStreamer

# Initialize console and logging
console = Console()
configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


def get_env_or_raise(key: str) -> str:
    """Get environment variable or raise error if not set."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def print_banner(mode: str) -> None:
    """Print startup banner with mode indicator.

    Args:
        mode: Operating mode ('backfill' or 'stream').
    """
    mode_display = mode.capitalize()
    banner = Panel(
        f"[bold white]autoBot Ingestor[/bold white]\n[dim]Mode: {mode_display}[/dim]",
        border_style="blue",
        padding=(1, 4),
    )
    console.print(banner)
    console.print()


def print_config_table(
    lake_path: Path,
    symbols: list[str],
    mode: str,
    days: int | None = None,
    data_types: list[str] | None = None,
    feed: str = "iex",
) -> None:
    """Print configuration summary table.

    Args:
        lake_path: Path to data lake.
        symbols: List of symbols.
        mode: Operating mode.
        days: Number of days (for backfill).
        data_types: Data types to ingest.
        feed: Data feed name.
    """
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Config", style="dim")
    table.add_column("Value")

    table.add_row("Lake Path", str(lake_path))
    table.add_row("Symbols", ", ".join(symbols))
    table.add_row("Feed", feed.upper())

    if mode == "backfill":
        table.add_row("Days", str(days))
        if data_types:
            table.add_row("Data Types", ", ".join(data_types))

    console.print(table)
    console.print()


def print_backfill_summary(stats: BackfillStats) -> None:
    """Print backfill completion summary.

    Args:
        stats: BackfillStats object with results.
    """
    console.print()
    console.print("[bold green]Backfill complete[/bold green]")

    # Summary table
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")

    if stats.total_trades > 0:
        summary_table.add_row("Trades", format_number(stats.total_trades))
    if stats.total_quotes > 0:
        summary_table.add_row("Quotes", format_number(stats.total_quotes))
    if stats.total_bars > 0:
        summary_table.add_row("Bars", format_number(stats.total_bars))

    summary_table.add_row("Total Records", format_number(stats.total_records))
    summary_table.add_row("Data Size", format_bytes(stats.total_bytes))
    summary_table.add_row("Duration", format_duration(stats.elapsed))
    summary_table.add_row("Throughput", f"{stats.avg_throughput:.0f} records/sec")

    console.print(summary_table)
    console.print()

    # Per-symbol breakdown
    if len(stats.symbols) > 1:
        console.print("[bold]Per-Symbol Breakdown[/bold]")
        symbol_table = Table(show_header=True, header_style="bold", box=None)
        symbol_table.add_column("Symbol")
        symbol_table.add_column("Trades", justify="right")
        symbol_table.add_column("Quotes", justify="right")
        symbol_table.add_column("Bars", justify="right")
        symbol_table.add_column("Size", justify="right")
        symbol_table.add_column("Time", justify="right")

        for symbol, s in stats.symbols.items():
            symbol_table.add_row(
                symbol,
                format_number(s.trades_count),
                format_number(s.quotes_count),
                format_number(s.bars_count),
                format_bytes(s.bytes_written),
                format_duration(s.elapsed),
            )

        console.print(symbol_table)
        console.print()


def run_backfill(
    api_key: str,
    api_secret: str,
    lake_path: Path,
    symbols: list[str],
    days: int = 30,
    data_types: list[str] | None = None,
    feed: str = "iex",
) -> None:
    """Run historical data backfill with progress display.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        lake_path: Path to data lake.
        symbols: List of symbols to backfill.
        days: Number of days to backfill.
        data_types: List of data types to backfill.
        feed: Data feed ('iex' or 'sip').
    """
    if data_types is None:
        data_types = ["trades", "quotes", "bars"]

    logger.info(
        "backfill_starting",
        symbols=symbols,
        days=days,
        data_types=data_types,
        feed=feed,
    )

    orchestrator = BackfillOrchestrator(
        api_key=api_key,
        api_secret=api_secret,
        lake_path=lake_path,
        feed=feed,
    )

    try:
        stats = orchestrator.backfill_date_range_with_progress(
            symbols=symbols,
            days=days,
            data_types=data_types,
        )
        print_backfill_summary(stats)

    except Exception as e:
        logger.error("backfill_failed", error=str(e), exc_info=True)
        console.print(f"\n[bold red]Backfill failed:[/bold red] {e}")
        sys.exit(1)


async def run_stream(
    api_key: str,
    api_secret: str,
    lake_path: Path,
    symbols: list[str],
    feed: str = "iex",
) -> None:
    """Run real-time streaming.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        lake_path: Path to data lake.
        symbols: List of symbols to stream.
        feed: Data feed ('iex' or 'sip').
    """
    logger.info(
        "stream_starting",
        symbols=symbols,
        feed=feed,
    )

    streamer = RealtimeStreamer(
        api_key=api_key,
        api_secret=api_secret,
        lake_path=lake_path,
        symbols=symbols,
        feed=feed,
    )

    try:
        await streamer.start()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        logger.error("stream_failed", error=str(e), exc_info=True)
        console.print(f"\n[bold red]Stream failed:[/bold red] {e}")
    finally:
        await streamer.stop()


def main() -> None:
    """Main entry point for ingestor service."""
    # Get configuration from environment
    try:
        api_key = get_env_or_raise("ALPACA_API_KEY")
        api_secret = get_env_or_raise("ALPACA_API_SECRET")
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        logger.error("configuration_error", error=str(e))
        sys.exit(1)

    lake_path = Path(os.environ.get("LAKE_PATH", "./lake"))
    symbols = os.environ.get("SYMBOLS", "SPY,QQQ").split(",")
    mode = os.environ.get("INGESTOR_MODE", "stream")
    days = int(os.environ.get("BACKFILL_DAYS", "30"))
    feed = os.environ.get("ALPACA_FEED", "iex")
    data_types_env = os.environ.get("DATA_TYPES", "trades,quotes,bars")
    data_types = [dt.strip() for dt in data_types_env.split(",")]

    # Ensure lake directory exists
    lake_path.mkdir(parents=True, exist_ok=True)

    # Print startup banner and config
    print_banner(mode)
    print_config_table(
        lake_path=lake_path,
        symbols=symbols,
        mode=mode,
        days=days if mode == "backfill" else None,
        data_types=data_types if mode == "backfill" else None,
        feed=feed,
    )

    logger.info(
        "ingestor_starting",
        mode=mode,
        lake_path=str(lake_path),
        symbols=symbols,
        feed=feed,
    )

    if mode == "backfill":
        run_backfill(
            api_key=api_key,
            api_secret=api_secret,
            lake_path=lake_path,
            symbols=symbols,
            days=days,
            data_types=data_types,
            feed=feed,
        )
    elif mode == "stream":
        asyncio.run(
            run_stream(
                api_key=api_key,
                api_secret=api_secret,
                lake_path=lake_path,
                symbols=symbols,
                feed=feed,
            )
        )
    else:
        console.print(f"[bold red]Unknown mode:[/bold red] {mode}")
        console.print("Use 'backfill' or 'stream'.")
        logger.error("unknown_mode", mode=mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
