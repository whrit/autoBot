"""
TDD Tests for Universe Module.

London School (Mockist) approach - testing object interactions and collaborations.
Covers symbol universe management: CRUD operations, status tracking, metadata, persistence.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest


class TestSymbolDataclass:
    """Test Symbol dataclass for symbol metadata."""

    def test_symbol_creation_with_required_fields(self) -> None:
        """Test Symbol can be created with required fields."""
        from ingestor_py.universe import Symbol

        symbol = Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
        )

        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert symbol.is_active is True  # Default value

    def test_symbol_creation_with_all_fields(self) -> None:
        """Test Symbol can be created with all optional fields."""
        from ingestor_py.universe import Symbol

        added_at = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        symbol = Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            is_active=True,
            added_at=added_at,
            deactivated_at=None,
            metadata={"market_cap": "large"},
        )

        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert symbol.sector == "Technology"
        assert symbol.industry == "Consumer Electronics"
        assert symbol.is_active is True
        assert symbol.added_at == added_at
        assert symbol.deactivated_at is None
        assert symbol.metadata == {"market_cap": "large"}

    def test_symbol_to_dict(self) -> None:
        """Test Symbol can be converted to dictionary."""
        from ingestor_py.universe import Symbol

        symbol = Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
        )

        data = symbol.to_dict()

        assert isinstance(data, dict)
        assert data["ticker"] == "AAPL"
        assert data["exchange"] == "NASDAQ"
        assert data["sector"] == "Technology"

    def test_symbol_from_dict(self) -> None:
        """Test Symbol can be created from dictionary."""
        from ingestor_py.universe import Symbol

        data = {
            "ticker": "AAPL",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "is_active": True,
        }

        symbol = Symbol.from_dict(data)

        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert symbol.sector == "Technology"
        assert symbol.is_active is True


class TestUniverseManager:
    """Test UniverseManager for CRUD operations on symbol universe."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_manager_initialization(self, manager: Any, temp_lake: Path) -> None:
        """Test manager initializes correctly."""
        assert manager is not None
        assert manager.lake_path == temp_lake

    def test_add_symbol_single(self, manager: Any) -> None:
        """Test adding a single symbol to universe."""
        from ingestor_py.universe import Symbol

        symbol = Symbol(ticker="AAPL", exchange="NASDAQ")
        manager.add_symbol(symbol)

        assert manager.get_symbol("AAPL") is not None
        assert manager.get_symbol("AAPL").ticker == "AAPL"

    def test_add_symbol_duplicate_raises_error(self, manager: Any) -> None:
        """Test adding duplicate symbol raises error."""
        from ingestor_py.universe import Symbol, SymbolExistsError

        symbol = Symbol(ticker="AAPL", exchange="NASDAQ")
        manager.add_symbol(symbol)

        with pytest.raises(SymbolExistsError):
            manager.add_symbol(symbol)

    def test_add_symbol_from_dict(self, manager: Any) -> None:
        """Test adding symbol from dictionary."""
        manager.add_symbol_from_dict({
            "ticker": "MSFT",
            "exchange": "NASDAQ",
            "sector": "Technology",
        })

        symbol = manager.get_symbol("MSFT")
        assert symbol is not None
        assert symbol.sector == "Technology"

    def test_remove_symbol(self, manager: Any) -> None:
        """Test removing a symbol from universe."""
        from ingestor_py.universe import Symbol

        symbol = Symbol(ticker="AAPL", exchange="NASDAQ")
        manager.add_symbol(symbol)
        manager.remove_symbol("AAPL")

        assert manager.get_symbol("AAPL") is None

    def test_remove_nonexistent_symbol_raises_error(self, manager: Any) -> None:
        """Test removing non-existent symbol raises error."""
        from ingestor_py.universe import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            manager.remove_symbol("NONEXISTENT")

    def test_list_all_symbols(self, manager: Any) -> None:
        """Test listing all symbols."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="GOOGL", exchange="NASDAQ"))

        symbols = manager.list_symbols()

        assert len(symbols) == 3
        tickers = [s.ticker for s in symbols]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_list_active_symbols_only(self, manager: Any) -> None:
        """Test listing only active symbols."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="INTC", exchange="NASDAQ", is_active=False))

        active_symbols = manager.list_active_symbols()

        assert len(active_symbols) == 2
        tickers = [s.ticker for s in active_symbols]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "INTC" not in tickers

    def test_get_active_tickers(self, manager: Any) -> None:
        """Test getting list of active ticker strings."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="INTC", exchange="NASDAQ", is_active=False))

        tickers = manager.get_active_tickers()

        assert len(tickers) == 2
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "INTC" not in tickers


class TestSymbolStatusManagement:
    """Test symbol status changes (active/inactive)."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_deactivate_symbol(self, manager: Any) -> None:
        """Test deactivating a symbol."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.deactivate_symbol("AAPL")

        symbol = manager.get_symbol("AAPL")
        assert symbol.is_active is False
        assert symbol.deactivated_at is not None

    def test_activate_symbol(self, manager: Any) -> None:
        """Test activating an inactive symbol."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=False))
        manager.activate_symbol("AAPL")

        symbol = manager.get_symbol("AAPL")
        assert symbol.is_active is True

    def test_deactivate_already_inactive_no_error(self, manager: Any) -> None:
        """Test deactivating already inactive symbol is idempotent."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=False))
        manager.deactivate_symbol("AAPL")  # Should not raise

        assert manager.get_symbol("AAPL").is_active is False

    def test_activate_already_active_no_error(self, manager: Any) -> None:
        """Test activating already active symbol is idempotent."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.activate_symbol("AAPL")  # Should not raise

        assert manager.get_symbol("AAPL").is_active is True


class TestSymbolMetadata:
    """Test symbol metadata operations."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_get_symbol_metadata(self, manager: Any) -> None:
        """Test getting symbol metadata."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            metadata={"market_cap": "large", "index": "SP500"},
        ))

        metadata = manager.get_symbol_metadata("AAPL")

        assert metadata["sector"] == "Technology"
        assert metadata["industry"] == "Consumer Electronics"
        assert metadata["metadata"]["market_cap"] == "large"

    def test_update_symbol_metadata(self, manager: Any) -> None:
        """Test updating symbol metadata."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
        ))

        manager.update_symbol_metadata("AAPL", {
            "sector": "Tech",  # Update existing
            "industry": "Hardware",  # Add new
        })

        symbol = manager.get_symbol("AAPL")
        assert symbol.sector == "Tech"
        assert symbol.industry == "Hardware"

    def test_filter_by_sector(self, manager: Any) -> None:
        """Test filtering symbols by sector."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", sector="Technology"))
        manager.add_symbol(Symbol(ticker="JPM", exchange="NYSE", sector="Finance"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ", sector="Technology"))

        tech_symbols = manager.filter_by_sector("Technology")

        assert len(tech_symbols) == 2
        tickers = [s.ticker for s in tech_symbols]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "JPM" not in tickers

    def test_filter_by_exchange(self, manager: Any) -> None:
        """Test filtering symbols by exchange."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="JPM", exchange="NYSE"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ"))

        nasdaq_symbols = manager.filter_by_exchange("NASDAQ")

        assert len(nasdaq_symbols) == 2
        tickers = [s.ticker for s in nasdaq_symbols]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestUniversePersistence:
    """Test persisting universe to Parquet files."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_save_creates_parquet_file(self, manager: Any, temp_lake: Path) -> None:
        """Test saving universe creates Parquet file."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ"))

        manager.save()

        # Check universe directory exists
        universe_path = temp_lake / "universe"
        assert universe_path.exists()

        # Check Parquet file exists
        parquet_files = list(universe_path.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_save_and_load_roundtrip(self, temp_lake: Path) -> None:
        """Test saving and loading preserves data."""
        from ingestor_py.universe import Symbol, UniverseManager

        # Create and save
        manager1 = UniverseManager(lake_path=temp_lake)
        manager1.add_symbol(Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
            is_active=True,
        ))
        manager1.add_symbol(Symbol(
            ticker="JPM",
            exchange="NYSE",
            sector="Finance",
            is_active=False,
        ))
        manager1.save()

        # Load in new manager
        manager2 = UniverseManager(lake_path=temp_lake)
        manager2.load()

        # Verify data
        symbols = manager2.list_symbols()
        assert len(symbols) == 2

        aapl = manager2.get_symbol("AAPL")
        assert aapl.exchange == "NASDAQ"
        assert aapl.sector == "Technology"
        assert aapl.is_active is True

        jpm = manager2.get_symbol("JPM")
        assert jpm.exchange == "NYSE"
        assert jpm.sector == "Finance"
        assert jpm.is_active is False

    def test_load_empty_universe(self, manager: Any, temp_lake: Path) -> None:
        """Test loading when no universe file exists."""
        manager.load()  # Should not raise
        assert len(manager.list_symbols()) == 0

    def test_parquet_schema_correct(self, manager: Any, temp_lake: Path) -> None:
        """Test Parquet file has correct schema."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(
            ticker="AAPL",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            is_active=True,
            metadata={"market_cap": "large"},
        ))
        manager.save()

        # Read parquet and check schema
        parquet_files = list((temp_lake / "universe").glob("*.parquet"))
        table = pq.read_table(parquet_files[0])

        expected_columns = ["ticker", "exchange", "sector", "industry", "is_active"]
        for col in expected_columns:
            assert col in table.column_names


class TestSymbolHistoryTracking:
    """Test historical symbol changes tracking."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_track_symbol_addition(self, manager: Any) -> None:
        """Test tracking when symbol is added."""
        from ingestor_py.universe import Symbol

        before = datetime.now(UTC)
        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        after = datetime.now(UTC)

        history = manager.get_symbol_history("AAPL")

        assert len(history) >= 1
        assert history[0]["event"] == "added"
        assert before <= history[0]["timestamp"] <= after

    def test_track_symbol_deactivation(self, manager: Any) -> None:
        """Test tracking when symbol is deactivated."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.deactivate_symbol("AAPL")

        history = manager.get_symbol_history("AAPL")

        events = [h["event"] for h in history]
        assert "deactivated" in events

    def test_track_symbol_activation(self, manager: Any) -> None:
        """Test tracking when symbol is activated."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=False))
        manager.activate_symbol("AAPL")

        history = manager.get_symbol_history("AAPL")

        events = [h["event"] for h in history]
        assert "activated" in events

    def test_track_metadata_update(self, manager: Any) -> None:
        """Test tracking when metadata is updated."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.update_symbol_metadata("AAPL", {"sector": "Technology"})

        history = manager.get_symbol_history("AAPL")

        events = [h["event"] for h in history]
        assert "metadata_updated" in events

    def test_get_all_history(self, manager: Any) -> None:
        """Test getting all historical changes."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ"))
        manager.deactivate_symbol("AAPL")

        all_history = manager.get_all_history()

        assert len(all_history) >= 3  # 2 adds + 1 deactivation


class TestBulkOperations:
    """Test bulk operations for efficiency."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def manager(self, temp_lake: Path) -> Any:
        """Create UniverseManager instance."""
        from ingestor_py.universe import UniverseManager

        return UniverseManager(lake_path=temp_lake)

    def test_add_symbols_bulk(self, manager: Any) -> None:
        """Test adding multiple symbols at once."""
        from ingestor_py.universe import Symbol

        symbols = [
            Symbol(ticker="AAPL", exchange="NASDAQ"),
            Symbol(ticker="MSFT", exchange="NASDAQ"),
            Symbol(ticker="GOOGL", exchange="NASDAQ"),
        ]

        manager.add_symbols_bulk(symbols)

        assert len(manager.list_symbols()) == 3

    def test_add_symbols_from_list(self, manager: Any) -> None:
        """Test adding symbols from list of tickers."""
        manager.add_symbols_from_list(
            tickers=["AAPL", "MSFT", "GOOGL"],
            exchange="NASDAQ",
        )

        assert len(manager.list_symbols()) == 3
        assert all(s.exchange == "NASDAQ" for s in manager.list_symbols())

    def test_deactivate_symbols_bulk(self, manager: Any) -> None:
        """Test deactivating multiple symbols at once."""
        from ingestor_py.universe import Symbol

        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ"))
        manager.add_symbol(Symbol(ticker="GOOGL", exchange="NASDAQ"))

        manager.deactivate_symbols_bulk(["AAPL", "MSFT"])

        assert manager.get_symbol("AAPL").is_active is False
        assert manager.get_symbol("MSFT").is_active is False
        assert manager.get_symbol("GOOGL").is_active is True


class TestIntegrationWithBackfill:
    """Test integration between UniverseManager and BackfillOrchestrator."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    def test_universe_provides_symbols_for_backfill(self, temp_lake: Path) -> None:
        """Test UniverseManager can provide symbols for backfill."""
        from ingestor_py.universe import Symbol, UniverseManager

        manager = UniverseManager(lake_path=temp_lake)
        manager.add_symbol(Symbol(ticker="AAPL", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="MSFT", exchange="NASDAQ", is_active=True))
        manager.add_symbol(Symbol(ticker="INTC", exchange="NASDAQ", is_active=False))

        # Get active tickers for backfill
        active_tickers = manager.get_active_tickers()

        assert len(active_tickers) == 2
        assert "AAPL" in active_tickers
        assert "MSFT" in active_tickers
        # INTC is inactive, should not be included
        assert "INTC" not in active_tickers
