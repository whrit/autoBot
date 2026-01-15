"""
Ingestor Service Main Entry Point.

Supports two modes:
- backfill: Download historical data for configured symbols and date range.
- stream: Real-time streaming of trades and quotes via WebSocket.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from ingestor_py.backfill import BackfillOrchestrator
from ingestor_py.streaming import RealtimeStreamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env_or_raise(key: str) -> str:
    """Get environment variable or raise error if not set."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def run_backfill(
    api_key: str,
    api_secret: str,
    lake_path: Path,
    symbols: list[str],
    days: int = 30,
) -> None:
    """Run historical data backfill.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        lake_path: Path to data lake.
        symbols: List of symbols to backfill.
        days: Number of days to backfill.
    """
    logger.info(f"Starting backfill for {symbols}, last {days} days")

    orchestrator = BackfillOrchestrator(
        api_key=api_key,
        api_secret=api_secret,
        lake_path=lake_path,
    )

    result = orchestrator.backfill_date_range(
        symbols=symbols,
        days=days,
    )

    logger.info(f"Backfill complete: {result}")


async def run_stream(
    api_key: str,
    api_secret: str,
    lake_path: Path,
    symbols: list[str],
) -> None:
    """Run real-time streaming.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        lake_path: Path to data lake.
        symbols: List of symbols to stream.
    """
    logger.info(f"Starting real-time stream for {symbols}")

    streamer = RealtimeStreamer(
        api_key=api_key,
        api_secret=api_secret,
        lake_path=lake_path,
        symbols=symbols,
    )

    try:
        await streamer.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await streamer.stop()


def main() -> None:
    """Main entry point for ingestor service."""
    # Get configuration from environment
    try:
        api_key = get_env_or_raise("ALPACA_API_KEY")
        api_secret = get_env_or_raise("ALPACA_API_SECRET")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    lake_path = Path(os.environ.get("LAKE_PATH", "./lake"))
    symbols = os.environ.get("SYMBOLS", "SPY,QQQ").split(",")
    mode = os.environ.get("INGESTOR_MODE", "stream")
    days = int(os.environ.get("BACKFILL_DAYS", "30"))

    # Ensure lake directory exists
    lake_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingestor starting in {mode} mode")
    logger.info(f"Lake path: {lake_path}")
    logger.info(f"Symbols: {symbols}")

    if mode == "backfill":
        run_backfill(
            api_key=api_key,
            api_secret=api_secret,
            lake_path=lake_path,
            symbols=symbols,
            days=days,
        )
    elif mode == "stream":
        asyncio.run(
            run_stream(
                api_key=api_key,
                api_secret=api_secret,
                lake_path=lake_path,
                symbols=symbols,
            )
        )
    else:
        logger.error(f"Unknown mode: {mode}. Use 'backfill' or 'stream'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
