"""
Ingestor Service - Data ingestion from Alpaca Markets.

This service handles:
- Historical data fetching via REST API
- Real-time data streaming via WebSocket
- Parquet file writing with partitioning
- Historical backfill orchestration
- Symbol universe management

Supports both sequential and parallel modes for backfill:
- Sequential: Traditional day-by-day processing (lower API usage)
- Parallel: Concurrent symbol fetching using asyncio (faster for many symbols)
"""

from ingestor_py.backfill import (
    AsyncBackfillOrchestrator,
    BackfillOrchestrator,
    BackfillStats,
    SymbolStats,
)
from ingestor_py.client import (
    AlpacaDataClient,
    AsyncAlpacaDataClient,
    RateLimiter,
    SymbolFetchResult,
)
from ingestor_py.streaming import RealtimeStreamer
from ingestor_py.universe import (
    Symbol,
    SymbolExistsError,
    SymbolNotFoundError,
    UniverseManager,
)
from ingestor_py.writer import ParquetWriter

__all__ = [
    # Sync client
    "AlpacaDataClient",
    # Async client
    "AsyncAlpacaDataClient",
    "RateLimiter",
    "SymbolFetchResult",
    # Writer
    "ParquetWriter",
    # Backfill orchestrators
    "BackfillOrchestrator",
    "AsyncBackfillOrchestrator",
    "BackfillStats",
    "SymbolStats",
    # Streaming
    "RealtimeStreamer",
    # Universe management
    "UniverseManager",
    "Symbol",
    "SymbolExistsError",
    "SymbolNotFoundError",
]
