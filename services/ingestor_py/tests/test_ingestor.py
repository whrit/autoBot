"""
TDD Tests for Ingestor Service.

London School (Mockist) approach - testing object interactions and collaborations.
Covers T1.06-T1.10: Alpaca REST client, Parquet writer, backfill, WebSocket streaming.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pyarrow.parquet as pq
import pytest


class TestAlpacaDataClient:
    """Test Alpaca REST client for historical data (T1.06)."""

    @pytest.fixture
    def mock_historical_client(self) -> MagicMock:
        """Create a mock StockHistoricalDataClient."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_historical_client: MagicMock) -> Any:
        """Create AlpacaDataClient with mocked SDK."""
        with patch(
            "ingestor_py.client.StockHistoricalDataClient",
            return_value=mock_historical_client,
        ):
            from ingestor_py.client import AlpacaDataClient

            return AlpacaDataClient(
                api_key="test_key",
                api_secret="test_secret",
                feed="iex",
            )

    def test_client_initialization(self, client: Any) -> None:
        """Test client initializes with correct credentials."""
        assert client is not None
        assert client.feed == "iex"

    def test_client_initialization_with_sip_feed(self) -> None:
        """Test client can use SIP feed."""
        with patch("ingestor_py.client.StockHistoricalDataClient"):
            from ingestor_py.client import AlpacaDataClient

            client = AlpacaDataClient(
                api_key="test_key",
                api_secret="test_secret",
                feed="sip",
            )
            assert client.feed == "sip"

    def test_get_trades_returns_iterator(
        self, client: Any, mock_historical_client: MagicMock
    ) -> None:
        """Test fetching historical trades returns iterator of dicts."""
        # Mock the response from Alpaca
        mock_trade = MagicMock()
        mock_trade.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        mock_trade.price = 450.50
        mock_trade.size = 100.0
        mock_trade.exchange = "V"
        mock_trade.conditions = ["@", "F"]

        mock_response = {"SPY": [mock_trade]}
        mock_historical_client.get_stock_trades.return_value = mock_response

        start = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

        trades = list(client.get_trades("SPY", start, end))

        assert len(trades) == 1
        assert trades[0]["symbol"] == "SPY"
        assert trades[0]["price"] == 450.50
        assert trades[0]["size"] == 100.0
        assert "ts_event" in trades[0]
        assert "ts_recv" in trades[0]

    def test_get_quotes_returns_iterator(
        self, client: Any, mock_historical_client: MagicMock
    ) -> None:
        """Test fetching historical quotes returns iterator of dicts."""
        mock_quote = MagicMock()
        mock_quote.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        mock_quote.bid_price = 450.45
        mock_quote.bid_size = 500.0
        mock_quote.ask_price = 450.55
        mock_quote.ask_size = 300.0
        mock_quote.bid_exchange = "V"
        mock_quote.ask_exchange = "V"
        mock_quote.conditions = ["R"]

        mock_response = {"SPY": [mock_quote]}
        mock_historical_client.get_stock_quotes.return_value = mock_response

        start = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

        quotes = list(client.get_quotes("SPY", start, end))

        assert len(quotes) == 1
        assert quotes[0]["symbol"] == "SPY"
        assert quotes[0]["bid_price"] == 450.45
        assert quotes[0]["ask_price"] == 450.55
        assert "ts_event" in quotes[0]
        assert "ts_recv" in quotes[0]

    def test_get_bars_returns_iterator(
        self, client: Any, mock_historical_client: MagicMock
    ) -> None:
        """Test fetching historical bars returns iterator of dicts."""
        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        mock_bar.open = 450.00
        mock_bar.high = 451.00
        mock_bar.low = 449.50
        mock_bar.close = 450.75
        mock_bar.volume = 100000.0
        mock_bar.trade_count = 500
        mock_bar.vwap = 450.40

        mock_response = {"SPY": [mock_bar]}
        mock_historical_client.get_stock_bars.return_value = mock_response

        start = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

        bars = list(client.get_bars("SPY", start, end))

        assert len(bars) == 1
        assert bars[0]["symbol"] == "SPY"
        assert bars[0]["open"] == 450.00
        assert bars[0]["close"] == 450.75
        assert "ts_event" in bars[0]
        assert "ts_recv" in bars[0]

    def test_timestamps_both_present_in_trades(
        self, client: Any, mock_historical_client: MagicMock
    ) -> None:
        """Verify both ts_event and ts_recv are captured for trades."""
        mock_trade = MagicMock()
        mock_trade.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        mock_trade.price = 450.50
        mock_trade.size = 100.0
        mock_trade.exchange = "V"
        mock_trade.conditions = ["@"]

        mock_historical_client.get_stock_trades.return_value = {"SPY": [mock_trade]}

        start = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

        trades = list(client.get_trades("SPY", start, end))

        # ts_event = exchange timestamp from API
        assert trades[0]["ts_event"] == mock_trade.timestamp
        # ts_recv = receipt timestamp (should be set during processing)
        assert trades[0]["ts_recv"] is not None
        assert isinstance(trades[0]["ts_recv"], datetime)

    def test_get_trades_handles_empty_response(
        self, client: Any, mock_historical_client: MagicMock
    ) -> None:
        """Test handling of empty response from Alpaca."""
        mock_historical_client.get_stock_trades.return_value = {}

        start = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)

        trades = list(client.get_trades("SPY", start, end))

        assert len(trades) == 0


class TestParquetWriter:
    """Test Parquet writer with partitioning (T1.07)."""

    @pytest.fixture
    def temp_lake(self, tmp_path: Path) -> Path:
        """Create temporary lake directory."""
        lake = tmp_path / "lake"
        lake.mkdir()
        return lake

    @pytest.fixture
    def writer(self, temp_lake: Path) -> Any:
        """Create ParquetWriter instance."""
        from ingestor_py.writer import ParquetWriter

        return ParquetWriter(base_path=temp_lake)

    def test_write_trades_creates_partitioned_files(
        self, writer: Any, temp_lake: Path
    ) -> None:
        """Test writing trades to partitioned Parquet."""
        trades = [
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, 100, tzinfo=UTC),
                "price": 450.50,
                "size": 100.0,
                "exchange": "V",
                "conditions": "@ F",
            }
        ]

        writer.write_trades(trades)

        # Check partitioned path exists: lake/raw/trades/dt=2024-01-15/symbol=SPY/
        expected_path = temp_lake / "raw" / "trades" / "dt=2024-01-15" / "symbol=SPY"
        assert expected_path.exists(), f"Expected path {expected_path} does not exist"

        # Read back and verify content
        table = pq.read_table(expected_path)
        assert len(table) == 1
        assert "ts_event" in table.column_names
        assert "ts_recv" in table.column_names
        assert "price" in table.column_names
        assert "size" in table.column_names

    def test_write_quotes_creates_partitioned_files(
        self, writer: Any, temp_lake: Path
    ) -> None:
        """Test writing quotes to partitioned Parquet."""
        quotes = [
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, 50, tzinfo=UTC),
                "bid_price": 450.45,
                "bid_size": 500.0,
                "ask_price": 450.55,
                "ask_size": 300.0,
                "bid_exchange": "V",
                "ask_exchange": "V",
            }
        ]

        writer.write_quotes(quotes)

        expected_path = temp_lake / "raw" / "quotes" / "dt=2024-01-15" / "symbol=SPY"
        assert expected_path.exists()

        table = pq.read_table(expected_path)
        assert len(table) == 1
        assert "bid_price" in table.column_names
        assert "ask_price" in table.column_names

    def test_write_bars_creates_partitioned_files(
        self, writer: Any, temp_lake: Path
    ) -> None:
        """Test writing bars to partitioned Parquet."""
        bars = [
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "open": 450.00,
                "high": 451.00,
                "low": 449.50,
                "close": 450.75,
                "volume": 100000.0,
                "trade_count": 500,
                "vwap": 450.40,
            }
        ]

        writer.write_bars(bars)

        expected_path = temp_lake / "raw" / "bars_provider" / "dt=2024-01-15" / "symbol=SPY"
        assert expected_path.exists()

        table = pq.read_table(expected_path)
        assert len(table) == 1
        assert "open" in table.column_names
        assert "close" in table.column_names

    def test_partitioning_scheme_multiple_dates(
        self, writer: Any, temp_lake: Path
    ) -> None:
        """Verify dt=YYYY-MM-DD/symbol=XXX partitioning with multiple dates."""
        trades = [
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "price": 450.50,
                "size": 100.0,
                "exchange": "V",
                "conditions": "@",
            },
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 16, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 16, 10, 0, 0, tzinfo=UTC),
                "price": 451.50,
                "size": 200.0,
                "exchange": "V",
                "conditions": "@",
            },
        ]

        writer.write_trades(trades)

        # Check both date partitions exist
        path_day1 = temp_lake / "raw" / "trades" / "dt=2024-01-15" / "symbol=SPY"
        path_day2 = temp_lake / "raw" / "trades" / "dt=2024-01-16" / "symbol=SPY"
        assert path_day1.exists()
        assert path_day2.exists()

    def test_partitioning_scheme_multiple_symbols(
        self, writer: Any, temp_lake: Path
    ) -> None:
        """Verify partitioning with multiple symbols."""
        trades = [
            {
                "symbol": "SPY",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "price": 450.50,
                "size": 100.0,
                "exchange": "V",
                "conditions": "@",
            },
            {
                "symbol": "QQQ",
                "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                "price": 380.25,
                "size": 150.0,
                "exchange": "V",
                "conditions": "@",
            },
        ]

        writer.write_trades(trades)

        # Check both symbol partitions exist
        path_spy = temp_lake / "raw" / "trades" / "dt=2024-01-15" / "symbol=SPY"
        path_qqq = temp_lake / "raw" / "trades" / "dt=2024-01-15" / "symbol=QQQ"
        assert path_spy.exists()
        assert path_qqq.exists()

    def test_write_empty_list_no_error(self, writer: Any, temp_lake: Path) -> None:
        """Test writing empty list does not raise error."""
        writer.write_trades([])
        # Should not raise, just do nothing


class TestBackfillOrchestrator:
    """Test historical data backfill (T1.08)."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock AlpacaDataClient."""
        mock = MagicMock()
        mock.get_trades.return_value = iter(
            [
                {
                    "symbol": "SPY",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "price": 450.50,
                    "size": 100.0,
                    "exchange": "V",
                    "conditions": "@",
                }
            ]
        )
        mock.get_quotes.return_value = iter(
            [
                {
                    "symbol": "SPY",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "bid_price": 450.45,
                    "bid_size": 500.0,
                    "ask_price": 450.55,
                    "ask_size": 300.0,
                    "bid_exchange": "V",
                    "ask_exchange": "V",
                }
            ]
        )
        mock.get_bars.return_value = iter(
            [
                {
                    "symbol": "SPY",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "open": 450.0,
                    "high": 451.0,
                    "low": 449.5,
                    "close": 450.75,
                    "volume": 100000.0,
                    "trade_count": 500,
                    "vwap": 450.4,
                }
            ]
        )
        return mock

    @pytest.fixture
    def orchestrator(self, tmp_path: Path, mock_client: MagicMock) -> Any:
        """Create BackfillOrchestrator with mocked client."""
        with patch("ingestor_py.backfill.AlpacaDataClient", return_value=mock_client):
            from ingestor_py.backfill import BackfillOrchestrator

            return BackfillOrchestrator(
                api_key="test_key",
                api_secret="test_secret",
                lake_path=tmp_path / "lake",
                feed="iex",
            )

    def test_backfill_date_range(
        self, orchestrator: Any, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test backfilling a date range."""
        start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)

        result = orchestrator.backfill(
            symbols=["SPY"],
            start=start,
            end=end,
            data_types=["trades", "quotes", "bars"],
        )

        assert "trades" in result
        assert "quotes" in result
        assert "bars" in result
        assert result["trades"]["count"] > 0

    def test_backfill_multiple_symbols(
        self, orchestrator: Any, mock_client: MagicMock
    ) -> None:
        """Test backfilling multiple symbols."""
        start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)

        # Update mock to return data for QQQ as well
        mock_client.get_trades.return_value = iter(
            [
                {
                    "symbol": "SPY",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "price": 450.50,
                    "size": 100.0,
                    "exchange": "V",
                    "conditions": "@",
                },
                {
                    "symbol": "QQQ",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "price": 380.25,
                    "size": 150.0,
                    "exchange": "V",
                    "conditions": "@",
                },
            ]
        )

        result = orchestrator.backfill(
            symbols=["SPY", "QQQ"],
            start=start,
            end=end,
            data_types=["trades"],
        )

        assert result["trades"]["count"] >= 1

    def test_backfill_returns_summary(self, orchestrator: Any) -> None:
        """Test backfill returns summary dict."""
        start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)

        result = orchestrator.backfill(
            symbols=["SPY"],
            start=start,
            end=end,
            data_types=["trades"],
        )

        assert isinstance(result, dict)
        assert "trades" in result
        assert "count" in result["trades"]
        assert "start" in result["trades"]
        assert "end" in result["trades"]


class TestRealtimeStreamer:
    """Test WebSocket streaming (T1.09)."""

    @pytest.fixture
    def mock_stream(self) -> MagicMock:
        """Create a mock StockDataStream."""
        mock = MagicMock()
        # run() is blocking, called via run_in_executor
        mock.run = MagicMock()
        # close() is async
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def streamer(self, tmp_path: Path, mock_stream: MagicMock) -> Any:
        """Create RealtimeStreamer with mocked WebSocket."""
        with patch(
            "ingestor_py.streaming.StockDataStream", return_value=mock_stream
        ):
            from ingestor_py.streaming import RealtimeStreamer

            return RealtimeStreamer(
                api_key="test_key",
                api_secret="test_secret",
                lake_path=tmp_path / "lake",
                symbols=["SPY", "QQQ"],
                feed="iex",
            )

    def test_streamer_initialization(self, streamer: Any) -> None:
        """Test streamer initializes correctly."""
        assert streamer is not None
        assert streamer.symbols == ["SPY", "QQQ"]

    def test_subscribe_symbols(self, streamer: Any, mock_stream: MagicMock) -> None:
        """Test subscribing to symbols."""
        streamer.subscribe()

        mock_stream.subscribe_trades.assert_called()
        mock_stream.subscribe_quotes.assert_called()

    @pytest.mark.asyncio
    async def test_start_begins_streaming(
        self, streamer: Any, mock_stream: MagicMock
    ) -> None:
        """Test start method begins streaming."""
        import asyncio

        # Make mock run() raise StopIteration to break out of the loop cleanly
        # This simulates the stream being stopped
        def mock_run_side_effect() -> None:
            # Set running to False to exit the reconnect loop
            streamer._running = False

        mock_stream.run.side_effect = mock_run_side_effect

        # Start should complete without hanging
        await asyncio.wait_for(streamer.start(), timeout=2.0)

        # Verify run was called
        mock_stream.run.assert_called()

    @pytest.mark.asyncio
    async def test_stop_graceful_shutdown(
        self, streamer: Any, mock_stream: MagicMock
    ) -> None:
        """Test stop performs graceful shutdown."""
        await streamer.stop()
        mock_stream.close.assert_called()

    @pytest.mark.asyncio
    async def test_ts_recv_captured_on_trade(self, tmp_path: Path) -> None:
        """Verify ts_recv is captured on message receipt for trades."""
        mock_stream = MagicMock()
        mock_stream.run = MagicMock()
        mock_stream.close = AsyncMock()

        with patch(
            "ingestor_py.streaming.StockDataStream", return_value=mock_stream
        ):
            from ingestor_py.streaming import RealtimeStreamer

            streamer = RealtimeStreamer(
                api_key="test_key",
                api_secret="test_secret",
                lake_path=tmp_path / "lake",
                symbols=["SPY"],
                feed="iex",
            )

            # Create a mock trade
            mock_trade = MagicMock()
            mock_trade.symbol = "SPY"
            mock_trade.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
            mock_trade.price = 450.50
            mock_trade.size = 100.0
            mock_trade.exchange = "V"
            mock_trade.conditions = ["@"]

            # Process the trade (async handler)
            before_process = datetime.now(UTC)
            await streamer._on_trade(mock_trade)
            after_process = datetime.now(UTC)

            # Check that ts_recv was captured
            assert streamer.last_trade is not None
            assert "ts_recv" in streamer.last_trade
            assert streamer.last_trade["ts_recv"] >= before_process
            assert streamer.last_trade["ts_recv"] <= after_process

    @pytest.mark.asyncio
    async def test_ts_recv_captured_on_quote(self, tmp_path: Path) -> None:
        """Verify ts_recv is captured on message receipt for quotes."""
        mock_stream = MagicMock()
        mock_stream.run = MagicMock()
        mock_stream.close = AsyncMock()

        with patch(
            "ingestor_py.streaming.StockDataStream", return_value=mock_stream
        ):
            from ingestor_py.streaming import RealtimeStreamer

            streamer = RealtimeStreamer(
                api_key="test_key",
                api_secret="test_secret",
                lake_path=tmp_path / "lake",
                symbols=["SPY"],
                feed="iex",
            )

            # Create a mock quote
            mock_quote = MagicMock()
            mock_quote.symbol = "SPY"
            mock_quote.timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
            mock_quote.bid_price = 450.45
            mock_quote.bid_size = 500.0
            mock_quote.ask_price = 450.55
            mock_quote.ask_size = 300.0
            mock_quote.bid_exchange = "V"
            mock_quote.ask_exchange = "V"
            mock_quote.conditions = ["R"]

            # Process the quote (async handler)
            before_process = datetime.now(UTC)
            await streamer._on_quote(mock_quote)
            after_process = datetime.now(UTC)

            # Check that ts_recv was captured
            assert streamer.last_quote is not None
            assert "ts_recv" in streamer.last_quote
            assert streamer.last_quote["ts_recv"] >= before_process
            assert streamer.last_quote["ts_recv"] <= after_process

    def test_reconnection_tracking(self, streamer: Any) -> None:
        """Test reconnection attempt tracking."""
        # The streamer should track reconnection attempts
        assert hasattr(streamer, "reconnect_attempts")
        assert streamer.reconnect_attempts == 0


class TestIntegration:
    """Integration tests for the full ingestor pipeline."""

    @pytest.fixture
    def mock_alpaca_client(self) -> MagicMock:
        """Create a fully mocked Alpaca client."""
        mock = MagicMock()
        mock.get_trades.return_value = iter(
            [
                {
                    "symbol": "SPY",
                    "ts_event": datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                    "ts_recv": datetime(2024, 1, 15, 10, 0, 0, 100, tzinfo=UTC),
                    "price": 450.50,
                    "size": 100.0,
                    "exchange": "V",
                    "conditions": "@",
                }
            ]
        )
        return mock

    def test_end_to_end_trades_pipeline(
        self, tmp_path: Path, mock_alpaca_client: MagicMock
    ) -> None:
        """Test full pipeline: client -> writer -> parquet files."""
        from ingestor_py.writer import ParquetWriter

        lake_path = tmp_path / "lake"
        lake_path.mkdir()

        # Get trades from mock client
        trades = list(mock_alpaca_client.get_trades("SPY", None, None))

        # Write to parquet
        writer = ParquetWriter(base_path=lake_path)
        writer.write_trades(trades)

        # Verify files exist
        expected_path = lake_path / "raw" / "trades" / "dt=2024-01-15" / "symbol=SPY"
        assert expected_path.exists()

        # Read and verify content
        table = pq.read_table(expected_path)
        assert len(table) == 1
        assert table.column("price")[0].as_py() == 450.50

    def test_timestamp_consistency(
        self, tmp_path: Path, mock_alpaca_client: MagicMock
    ) -> None:
        """Verify timestamps are consistent through pipeline."""
        from ingestor_py.writer import ParquetWriter

        lake_path = tmp_path / "lake"
        lake_path.mkdir()

        trades = list(mock_alpaca_client.get_trades("SPY", None, None))
        original_ts_event = trades[0]["ts_event"]
        original_ts_recv = trades[0]["ts_recv"]

        writer = ParquetWriter(base_path=lake_path)
        writer.write_trades(trades)

        # Read back
        expected_path = lake_path / "raw" / "trades" / "dt=2024-01-15" / "symbol=SPY"
        table = pq.read_table(expected_path)

        # Convert pyarrow timestamp to datetime for comparison
        read_ts_event = table.column("ts_event")[0].as_py()
        read_ts_recv = table.column("ts_recv")[0].as_py()

        # Timestamps should be preserved (may need timezone handling)
        assert read_ts_event.replace(tzinfo=UTC) == original_ts_event
        assert read_ts_recv.replace(tzinfo=UTC) == original_ts_recv
