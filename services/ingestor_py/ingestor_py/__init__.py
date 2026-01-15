"""
Ingestor Service - Data ingestion from Alpaca Markets.

This service handles:
- Historical data fetching via REST API
- Real-time data streaming via WebSocket
- Parquet file writing with partitioning
- Historical backfill orchestration
"""

from ingestor_py.backfill import BackfillOrchestrator
from ingestor_py.client import AlpacaDataClient
from ingestor_py.streaming import RealtimeStreamer
from ingestor_py.writer import ParquetWriter

__all__ = [
    "AlpacaDataClient",
    "ParquetWriter",
    "BackfillOrchestrator",
    "RealtimeStreamer",
]
