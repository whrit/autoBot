#!/usr/bin/env python
"""
autoBot Pipeline Orchestrator.

Simple script to run the end-to-end pipeline:
  ingest → features → backtest → optimize

Usage:
    # Run full pipeline
    python scripts/run_pipeline.py

    # Run specific stages
    python scripts/run_pipeline.py ingest
    python scripts/run_pipeline.py features
    python scripts/run_pipeline.py backtest
    python scripts/run_pipeline.py optimize

    # Run multiple stages
    python scripts/run_pipeline.py features backtest

    # With options
    python scripts/run_pipeline.py --symbols SPY,QQQ --days 30
    python scripts/run_pipeline.py backtest --walk-forward
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
LAKE_PATH = Path("lake")
DEFAULT_SYMBOLS = ["SPY", "QQQ"]
DEFAULT_LOOKBACK_DAYS = 30


def run_command(cmd: list[str], description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command and arguments
        description: Human-readable description

    Returns:
        True if command succeeded
    """
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n[ERROR] Command not found: {e}")
        return False


def run_ingest(
    symbols: list[str],
    days: int,
    provider: str = "databento",
) -> bool:
    """
    Run data ingestion.

    Args:
        symbols: List of symbols to ingest
        days: Number of days to backfill
        provider: Data provider (databento, alpaca)

    Returns:
        True if successful
    """
    cmd = [
        "uv", "run", "-m", "ingestor_py.main",
        "backfill",
        "--days", str(days),
        "--symbols", ",".join(symbols),
    ]

    return run_command(cmd, f"Ingesting {days} days of data for {', '.join(symbols)}")


def run_features(
    symbols: list[str],
    data_dir: Path = LAKE_PATH,
    output_dir: Path | None = None,
    large_dataset: bool = True,
) -> bool:
    """
    Run feature building.

    Args:
        symbols: List of symbols to process
        data_dir: Input data directory (lake)
        output_dir: Output directory for features
        large_dataset: Use large dataset mode

    Returns:
        True if successful
    """
    output = output_dir or (data_dir / "features")

    cmd = [
        "uv", "run", "-m", "feature_builder_py.main",
        *symbols,
        "--data-dir", str(data_dir),
        "--output-dir", str(output),
    ]

    if large_dataset:
        cmd.append("--large-dataset")

    return run_command(cmd, f"Building features for {', '.join(symbols)}")


def run_backtest(
    symbols: list[str],
    lake_path: Path = LAKE_PATH,
    walk_forward: bool = False,
    parallel: bool = True,
    workers: int | None = None,
    train_size: int = 500,
    test_size: int = 100,
    strategy: str = "Momentum",
    start_date: str | None = None,
    end_date: str | None = None,
    cost_model: str = "fixed",
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    max_position_pct: float = 0.10,
    max_drawdown_pct: float = 0.20,
) -> bool:
    """
    Run backtesting.

    Args:
        symbols: List of symbols to backtest
        lake_path: Path to data lake
        walk_forward: Enable walk-forward optimization
        parallel: Enable parallel processing
        workers: Number of parallel workers (None = auto)
        train_size: Training window size (walk-forward)
        test_size: Test window size (walk-forward)
        strategy: Strategy name
        start_date: Optional start date filter
        end_date: Optional end date filter
        cost_model: Cost model type (fixed, volume, zero)
        fee_rate: Fee rate as decimal (0.001 = 0.1%)
        slippage_rate: Slippage rate for fixed model
        max_position_pct: Maximum position as % of capital
        max_drawdown_pct: Maximum drawdown limit

    Returns:
        True if successful
    """
    cmd = [
        "uv", "run", "-m", "backtester_py.main",
        "--lake", str(lake_path),
        "--symbols", ",".join(symbols),
        "--strategy", strategy,
        "--cost-model", cost_model,
        "--fee-rate", str(fee_rate),
        "--slippage-rate", str(slippage_rate),
        "--max-position-pct", str(max_position_pct),
        "--max-drawdown-pct", str(max_drawdown_pct),
    ]

    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])

    if walk_forward:
        cmd.extend([
            "--walk-forward",
            "--train-size", str(train_size),
            "--test-size", str(test_size),
        ])
        # Enable parallel by default for walk-forward
        if parallel:
            cmd.append("--parallel")
            if workers:
                cmd.extend(["--workers", str(workers)])

    return run_command(cmd, f"Backtesting {', '.join(symbols)}")


def run_optimize(
    symbols: list[str],
    lake_path: Path = LAKE_PATH,
    metric: str = "sharpe_ratio",
    workers: int = -1,  # -1 = auto (CPU count)
) -> bool:
    """
    Run parameter optimization.

    Args:
        symbols: List of symbols
        lake_path: Path to data lake
        metric: Metric to optimize
        workers: Number of parallel workers (-1 = auto)

    Returns:
        True if successful
    """
    import os
    actual_workers = workers if workers > 0 else (os.cpu_count() or 4)

    cmd = [
        "uv", "run", "-m", "backtester_py.main",
        "--lake", str(lake_path),
        "--symbols", ",".join(symbols),
        "--grid-search",
        "--grid-metric", metric,
        "--grid-workers", str(actual_workers),
    ]

    return run_command(cmd, f"Optimizing parameters for {', '.join(symbols)} ({actual_workers} workers)")


def run_full_pipeline(
    symbols: list[str],
    days: int = DEFAULT_LOOKBACK_DAYS,
    skip_ingest: bool = False,
    walk_forward: bool = False,
    cost_model: str = "fixed",
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    max_position_pct: float = 0.10,
    max_drawdown_pct: float = 0.20,
) -> bool:
    """
    Run the full pipeline.

    Args:
        symbols: List of symbols
        days: Lookback days for ingestion
        skip_ingest: Skip ingestion stage
        walk_forward: Enable walk-forward in backtest

    Returns:
        True if all stages succeeded
    """
    stages_run = 0
    stages_passed = 0

    # Stage 1: Ingest (optional)
    if not skip_ingest:
        stages_run += 1
        if run_ingest(symbols, days):
            stages_passed += 1
        else:
            print("\n[WARNING] Ingestion failed, continuing with existing data...")

    # Stage 2: Features
    stages_run += 1
    if run_features(symbols):
        stages_passed += 1
    else:
        print("\n[ERROR] Feature building failed")
        return False

    # Stage 3: Backtest
    stages_run += 1
    if run_backtest(
        symbols,
        walk_forward=walk_forward,
        cost_model=cost_model,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        max_position_pct=max_position_pct,
        max_drawdown_pct=max_drawdown_pct,
    ):
        stages_passed += 1
    else:
        print("\n[ERROR] Backtesting failed")
        return False

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete: {stages_passed}/{stages_run} stages passed")
    print(f"{'='*60}\n")

    return stages_passed == stages_run


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="autoBot Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with default symbols
    python scripts/run_pipeline.py

    # Run specific stages
    python scripts/run_pipeline.py features backtest

    # With custom symbols
    python scripts/run_pipeline.py --symbols SPY,QQQ,AAPL

    # Run backtest with walk-forward
    python scripts/run_pipeline.py backtest --walk-forward

    # Skip ingestion (use existing data)
    python scripts/run_pipeline.py --skip-ingest
        """,
    )

    parser.add_argument(
        "stages",
        nargs="*",
        default=[],
        help="Stages to run: ingest, features, backtest, optimize (default: all)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Lookback days for ingestion (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--lake",
        type=Path,
        default=LAKE_PATH,
        help=f"Data lake path (default: {LAKE_PATH})",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion stage",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable walk-forward optimization in backtest",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = auto, uses CPU count)",
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
        help="Fee rate as decimal, e.g., 0.001 = 0.1%% (default: 0.001)",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0005,
        help="Slippage rate for fixed model (default: 0.0005)",
    )
    # Risk management arguments
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.10,
        help="Max position as %% of capital (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=0.20,
        help="Max drawdown limit (default: 0.20 = 20%%)",
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"\nautoBot Pipeline")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Lake: {args.lake}")

    # Determine which stages to run
    stages = args.stages if args.stages else ["all"]

    success = True

    # Determine worker count
    import os
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    if "all" in stages:
        success = run_full_pipeline(
            symbols=symbols,
            days=args.days,
            skip_ingest=args.skip_ingest,
            walk_forward=args.walk_forward,
            cost_model=args.cost_model,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            max_position_pct=args.max_position_pct,
            max_drawdown_pct=args.max_drawdown_pct,
        )
    else:
        for stage in stages:
            if stage == "ingest":
                success = run_ingest(symbols, args.days) and success
            elif stage == "features":
                success = run_features(symbols, args.lake) and success
            elif stage == "backtest":
                success = run_backtest(
                    symbols,
                    args.lake,
                    walk_forward=args.walk_forward,
                    workers=workers,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    cost_model=args.cost_model,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    max_position_pct=args.max_position_pct,
                    max_drawdown_pct=args.max_drawdown_pct,
                ) and success
            elif stage == "optimize":
                success = run_optimize(symbols, args.lake, workers=workers) and success
            else:
                print(f"[WARNING] Unknown stage: {stage}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
