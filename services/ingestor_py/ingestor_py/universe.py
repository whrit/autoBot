"""
Universe Manager - Symbol universe configuration and management.

Handles which symbols to track, their active/inactive status,
metadata (exchange, sector, etc.), and historical changes tracking.
Persists to Parquet files in the lake/universe/ directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class SymbolExistsError(Exception):
    """Raised when attempting to add a symbol that already exists."""

    pass


class SymbolNotFoundError(Exception):
    """Raised when attempting to operate on a symbol that does not exist."""

    pass


# Schema definition for universe Parquet file
UNIVERSE_SCHEMA = pa.schema(
    [
        ("ticker", pa.string()),
        ("exchange", pa.string()),
        ("sector", pa.string()),
        ("industry", pa.string()),
        ("is_active", pa.bool_()),
        ("added_at", pa.timestamp("us", tz="UTC")),
        ("deactivated_at", pa.timestamp("us", tz="UTC")),
        ("metadata_json", pa.string()),  # JSON-encoded metadata dict
    ]
)


@dataclass
class Symbol:
    """Represents a symbol in the universe with metadata.

    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        exchange: Exchange where the symbol trades (e.g., 'NASDAQ').
        sector: Business sector (e.g., 'Technology').
        industry: Specific industry (e.g., 'Consumer Electronics').
        is_active: Whether the symbol is actively tracked.
        added_at: When the symbol was added to the universe.
        deactivated_at: When the symbol was deactivated (if applicable).
        metadata: Additional metadata as a dictionary.
    """

    ticker: str
    exchange: str
    sector: str | None = None
    industry: str | None = None
    is_active: bool = True
    added_at: datetime | None = None
    deactivated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert Symbol to dictionary.

        Returns:
            Dictionary representation of the symbol.
        """
        return {
            "ticker": self.ticker,
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
            "is_active": self.is_active,
            "added_at": self.added_at,
            "deactivated_at": self.deactivated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Symbol:
        """Create Symbol from dictionary.

        Args:
            data: Dictionary with symbol data.

        Returns:
            Symbol instance.
        """
        return cls(
            ticker=data["ticker"],
            exchange=data["exchange"],
            sector=data.get("sector"),
            industry=data.get("industry"),
            is_active=data.get("is_active", True),
            added_at=data.get("added_at"),
            deactivated_at=data.get("deactivated_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HistoryEntry:
    """Represents a historical change event.

    Attributes:
        ticker: Stock ticker symbol.
        event: Type of event (added, deactivated, activated, metadata_updated).
        timestamp: When the event occurred.
        details: Additional details about the event.
    """

    ticker: str
    event: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


class UniverseManager:
    """Manage the symbol universe with CRUD operations and persistence.

    Handles adding, removing, and updating symbols, tracking their
    active/inactive status, and persisting to Parquet files.
    """

    def __init__(self, lake_path: Path | str) -> None:
        """Initialize the UniverseManager.

        Args:
            lake_path: Path to the data lake directory.
        """
        self.lake_path = Path(lake_path)
        self._symbols: dict[str, Symbol] = {}
        self._history: list[HistoryEntry] = []

    def _record_history(
        self,
        ticker: str,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a historical event.

        Args:
            ticker: Stock ticker symbol.
            event: Type of event.
            details: Additional details.
        """
        entry = HistoryEntry(
            ticker=ticker,
            event=event,
            timestamp=datetime.now(UTC),
            details=details or {},
        )
        self._history.append(entry)

    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the universe.

        Args:
            symbol: Symbol to add.

        Raises:
            SymbolExistsError: If symbol already exists.
        """
        if symbol.ticker in self._symbols:
            raise SymbolExistsError(f"Symbol {symbol.ticker} already exists")

        # Set added_at if not provided
        if symbol.added_at is None:
            symbol.added_at = datetime.now(UTC)

        self._symbols[symbol.ticker] = symbol
        self._record_history(symbol.ticker, "added", {"exchange": symbol.exchange})

    def add_symbol_from_dict(self, data: dict[str, Any]) -> None:
        """Add a symbol from dictionary data.

        Args:
            data: Dictionary with symbol data.

        Raises:
            SymbolExistsError: If symbol already exists.
        """
        symbol = Symbol.from_dict(data)
        self.add_symbol(symbol)

    def remove_symbol(self, ticker: str) -> None:
        """Remove a symbol from the universe.

        Args:
            ticker: Ticker of symbol to remove.

        Raises:
            SymbolNotFoundError: If symbol does not exist.
        """
        if ticker not in self._symbols:
            raise SymbolNotFoundError(f"Symbol {ticker} not found")

        del self._symbols[ticker]
        self._record_history(ticker, "removed")

    def get_symbol(self, ticker: str) -> Symbol | None:
        """Get a symbol by ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Symbol if found, None otherwise.
        """
        return self._symbols.get(ticker)

    def list_symbols(self) -> list[Symbol]:
        """List all symbols in the universe.

        Returns:
            List of all symbols.
        """
        return list(self._symbols.values())

    def list_active_symbols(self) -> list[Symbol]:
        """List only active symbols.

        Returns:
            List of active symbols.
        """
        return [s for s in self._symbols.values() if s.is_active]

    def get_active_tickers(self) -> list[str]:
        """Get list of active ticker strings.

        Returns:
            List of active ticker symbols.
        """
        return [s.ticker for s in self._symbols.values() if s.is_active]

    def deactivate_symbol(self, ticker: str) -> None:
        """Deactivate a symbol.

        Args:
            ticker: Ticker of symbol to deactivate.

        Raises:
            SymbolNotFoundError: If symbol does not exist.
        """
        symbol = self._symbols.get(ticker)
        if symbol is None:
            raise SymbolNotFoundError(f"Symbol {ticker} not found")

        if symbol.is_active:
            symbol.is_active = False
            symbol.deactivated_at = datetime.now(UTC)
            self._record_history(ticker, "deactivated")

    def activate_symbol(self, ticker: str) -> None:
        """Activate a symbol.

        Args:
            ticker: Ticker of symbol to activate.

        Raises:
            SymbolNotFoundError: If symbol does not exist.
        """
        symbol = self._symbols.get(ticker)
        if symbol is None:
            raise SymbolNotFoundError(f"Symbol {ticker} not found")

        if not symbol.is_active:
            symbol.is_active = True
            symbol.deactivated_at = None
            self._record_history(ticker, "activated")

    def get_symbol_metadata(self, ticker: str) -> dict[str, Any]:
        """Get metadata for a symbol.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary with symbol metadata.

        Raises:
            SymbolNotFoundError: If symbol does not exist.
        """
        symbol = self._symbols.get(ticker)
        if symbol is None:
            raise SymbolNotFoundError(f"Symbol {ticker} not found")

        return {
            "ticker": symbol.ticker,
            "exchange": symbol.exchange,
            "sector": symbol.sector,
            "industry": symbol.industry,
            "is_active": symbol.is_active,
            "added_at": symbol.added_at,
            "deactivated_at": symbol.deactivated_at,
            "metadata": symbol.metadata,
        }

    def update_symbol_metadata(
        self,
        ticker: str,
        updates: dict[str, Any],
    ) -> None:
        """Update metadata for a symbol.

        Args:
            ticker: Stock ticker symbol.
            updates: Dictionary with fields to update.

        Raises:
            SymbolNotFoundError: If symbol does not exist.
        """
        symbol = self._symbols.get(ticker)
        if symbol is None:
            raise SymbolNotFoundError(f"Symbol {ticker} not found")

        # Update known fields
        if "sector" in updates:
            symbol.sector = updates["sector"]
        if "industry" in updates:
            symbol.industry = updates["industry"]
        if "exchange" in updates:
            symbol.exchange = updates["exchange"]
        if "metadata" in updates:
            symbol.metadata.update(updates["metadata"])

        self._record_history(ticker, "metadata_updated", {"updates": updates})

    def filter_by_sector(self, sector: str) -> list[Symbol]:
        """Filter symbols by sector.

        Args:
            sector: Sector to filter by.

        Returns:
            List of symbols in the sector.
        """
        return [s for s in self._symbols.values() if s.sector == sector]

    def filter_by_exchange(self, exchange: str) -> list[Symbol]:
        """Filter symbols by exchange.

        Args:
            exchange: Exchange to filter by.

        Returns:
            List of symbols on the exchange.
        """
        return [s for s in self._symbols.values() if s.exchange == exchange]

    def get_symbol_history(self, ticker: str) -> list[dict[str, Any]]:
        """Get historical changes for a symbol.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of history entries for the symbol.
        """
        return [
            {"event": h.event, "timestamp": h.timestamp, "details": h.details}
            for h in self._history
            if h.ticker == ticker
        ]

    def get_all_history(self) -> list[dict[str, Any]]:
        """Get all historical changes.

        Returns:
            List of all history entries.
        """
        return [
            {
                "ticker": h.ticker,
                "event": h.event,
                "timestamp": h.timestamp,
                "details": h.details,
            }
            for h in self._history
        ]

    def add_symbols_bulk(self, symbols: list[Symbol]) -> None:
        """Add multiple symbols at once.

        Args:
            symbols: List of symbols to add.

        Raises:
            SymbolExistsError: If any symbol already exists.
        """
        for symbol in symbols:
            self.add_symbol(symbol)

    def add_symbols_from_list(
        self,
        tickers: list[str],
        exchange: str,
        sector: str | None = None,
        is_active: bool = True,
    ) -> None:
        """Add symbols from a list of tickers.

        Args:
            tickers: List of ticker symbols.
            exchange: Exchange for all symbols.
            sector: Optional sector for all symbols.
            is_active: Whether symbols are active (default True).
        """
        for ticker in tickers:
            symbol = Symbol(
                ticker=ticker,
                exchange=exchange,
                sector=sector,
                is_active=is_active,
            )
            self.add_symbol(symbol)

    def deactivate_symbols_bulk(self, tickers: list[str]) -> None:
        """Deactivate multiple symbols at once.

        Args:
            tickers: List of tickers to deactivate.
        """
        for ticker in tickers:
            if ticker in self._symbols:
                self.deactivate_symbol(ticker)

    def save(self) -> None:
        """Save the universe to Parquet file."""
        universe_path = self.lake_path / "universe"
        universe_path.mkdir(parents=True, exist_ok=True)

        # Prepare data for PyArrow
        data: dict[str, list[Any]] = {
            "ticker": [],
            "exchange": [],
            "sector": [],
            "industry": [],
            "is_active": [],
            "added_at": [],
            "deactivated_at": [],
            "metadata_json": [],
        }

        for symbol in self._symbols.values():
            data["ticker"].append(symbol.ticker)
            data["exchange"].append(symbol.exchange)
            data["sector"].append(symbol.sector)
            data["industry"].append(symbol.industry)
            data["is_active"].append(symbol.is_active)
            data["added_at"].append(symbol.added_at)
            data["deactivated_at"].append(symbol.deactivated_at)
            data["metadata_json"].append(json.dumps(symbol.metadata))

        # Create table and write
        table = pa.table(data, schema=UNIVERSE_SCHEMA)
        pq.write_table(table, universe_path / "symbols.parquet")

    def load(self) -> None:
        """Load the universe from Parquet file."""
        universe_path = self.lake_path / "universe"
        parquet_file = universe_path / "symbols.parquet"

        if not parquet_file.exists():
            # No existing universe, start fresh
            return

        # Read parquet file
        table = pq.read_table(parquet_file)

        # Convert to symbols
        self._symbols = {}
        for i in range(len(table)):
            ticker = table.column("ticker")[i].as_py()
            exchange = table.column("exchange")[i].as_py()
            sector = table.column("sector")[i].as_py()
            industry = table.column("industry")[i].as_py()
            is_active = table.column("is_active")[i].as_py()
            added_at = table.column("added_at")[i].as_py()
            deactivated_at = table.column("deactivated_at")[i].as_py()
            metadata_json = table.column("metadata_json")[i].as_py()

            # Parse metadata JSON
            metadata = json.loads(metadata_json) if metadata_json else {}

            # Ensure timestamps are timezone-aware
            if added_at is not None and added_at.tzinfo is None:
                added_at = added_at.replace(tzinfo=UTC)
            if deactivated_at is not None and deactivated_at.tzinfo is None:
                deactivated_at = deactivated_at.replace(tzinfo=UTC)

            symbol = Symbol(
                ticker=ticker,
                exchange=exchange,
                sector=sector,
                industry=industry,
                is_active=is_active,
                added_at=added_at,
                deactivated_at=deactivated_at,
                metadata=metadata,
            )
            self._symbols[ticker] = symbol
