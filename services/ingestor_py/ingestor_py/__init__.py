"""
Ingestor Service - Data ingestion from Alpaca Markets.

This service handles:
- Historical data fetching via REST API
- Real-time data streaming via WebSocket
- Parquet file writing with partitioning
- Historical backfill orchestration
- Symbol universe management
"""

from ingestor_py.backfill import BackfillOrchestrator
from ingestor_py.client import AlpacaDataClient
from ingestor_py.streaming import RealtimeStreamer
from ingestor_py.universe import (
    Symbol,
    SymbolExistsError,
    SymbolNotFoundError,
    UniverseManager,
)
from ingestor_py.writer import ParquetWriter

__all__ = [
    "AlpacaDataClient",
    "ParquetWriter",
    "BackfillOrchestrator",
    "RealtimeStreamer",
    "UniverseManager",
    "Symbol",
    "SymbolExistsError",
    "SymbolNotFoundError",
]
