"""
Tests for Fill Confirmation Stream (T6.04).

Tests WebSocket-based fill confirmations via Alpaca TradingStream.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runner_py.fill_stream import FillConfirmationStream, FillStreamConfig
from runner_py.paper import PaperFill


@pytest.fixture
def fill_stream_config() -> FillStreamConfig:
    """Create test fill stream config."""
    return FillStreamConfig(
        api_key="test-api-key",
        api_secret="test-api-secret",
        paper=True,
    )


@pytest.fixture
def sample_fill() -> PaperFill:
    """Create a sample fill."""
    return PaperFill(
        order_id="order-123",
        symbol="AAPL",
        side="buy",
        qty=10.0,
        filled_price=150.50,
        filled_at=datetime.now(UTC),
        status="filled",
    )


@pytest.fixture
def sample_trade_update() -> dict:
    """Create a sample trade update event from WebSocket."""
    return {
        "event": "fill",
        "order": {
            "id": "order-456",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10.0",
            "filled_qty": "10.0",
            "filled_avg_price": "150.75",
            "status": "filled",
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


class TestFillStreamConfig:
    """Tests for FillStreamConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = FillStreamConfig(
            api_key="key",
            api_secret="secret",
        )
        assert config.paper is True

    def test_paper_mode_enforced(self) -> None:
        """Test that paper mode is default True."""
        config = FillStreamConfig(
            api_key="key",
            api_secret="secret",
            paper=True,
        )
        assert config.paper is True


class TestFillConfirmationStreamInit:
    """Tests for FillConfirmationStream initialization."""

    @patch("runner_py.fill_stream.TradingStream")
    def test_init_creates_stream(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test stream creation with correct params."""
        FillConfirmationStream(fill_stream_config)

        mock_stream_class.assert_called_once_with(
            api_key=fill_stream_config.api_key,
            secret_key=fill_stream_config.api_secret,
            paper=True,
        )

    @patch("runner_py.fill_stream.TradingStream")
    def test_init_with_paper_true(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test stream always uses paper=True."""
        _ = FillConfirmationStream(fill_stream_config)

        call_kwargs = mock_stream_class.call_args[1]
        assert call_kwargs["paper"] is True

    @patch("runner_py.fill_stream.TradingStream")
    def test_init_empty_state(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test stream initializes with empty state."""
        stream = FillConfirmationStream(fill_stream_config)

        assert len(stream._callbacks) == 0
        assert len(stream._fill_history) == 0


class TestFillConfirmationStreamCallbacks:
    """Tests for callback registration and handling."""

    @patch("runner_py.fill_stream.TradingStream")
    def test_register_callback(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test registering a callback."""
        stream = FillConfirmationStream(fill_stream_config)

        callback = MagicMock()
        stream.register_callback(callback)

        assert callback in stream._callbacks
        assert len(stream._callbacks) == 1

    @patch("runner_py.fill_stream.TradingStream")
    def test_register_multiple_callbacks(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test registering multiple callbacks."""
        stream = FillConfirmationStream(fill_stream_config)

        callback1 = MagicMock()
        callback2 = MagicMock()
        stream.register_callback(callback1)
        stream.register_callback(callback2)

        assert len(stream._callbacks) == 2

    @patch("runner_py.fill_stream.TradingStream")
    def test_unregister_callback(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test unregistering a callback."""
        stream = FillConfirmationStream(fill_stream_config)

        callback = MagicMock()
        stream.register_callback(callback)
        stream.unregister_callback(callback)

        assert callback not in stream._callbacks
        assert len(stream._callbacks) == 0


class TestFillConfirmationStreamHandleUpdate:
    """Tests for handling trade update events."""

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_handle_trade_update_fill(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test handling a fill event."""
        stream = FillConfirmationStream(fill_stream_config)

        callback = MagicMock()
        stream.register_callback(callback)

        # Create mock trade update data
        mock_data = MagicMock()
        mock_data.event = "fill"
        mock_data.order = MagicMock()
        mock_data.order.id = "order-789"
        mock_data.order.symbol = "AAPL"
        mock_data.order.side = "buy"
        mock_data.order.filled_qty = "10.0"
        mock_data.order.filled_avg_price = "151.25"
        mock_data.order.status = "filled"
        mock_data.timestamp = datetime.now(UTC)

        await stream._handle_trade_update(mock_data)

        # Verify callback was called
        callback.assert_called_once()
        fill = callback.call_args[0][0]
        assert fill.order_id == "order-789"
        assert fill.symbol == "AAPL"
        assert fill.side == "buy"
        assert fill.qty == 10.0
        assert fill.filled_price == 151.25

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_handle_trade_update_partial_fill(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test handling a partial fill event."""
        stream = FillConfirmationStream(fill_stream_config)

        callback = MagicMock()
        stream.register_callback(callback)

        mock_data = MagicMock()
        mock_data.event = "partial_fill"
        mock_data.order = MagicMock()
        mock_data.order.id = "order-partial"
        mock_data.order.symbol = "MSFT"
        mock_data.order.side = "sell"
        mock_data.order.filled_qty = "5.0"
        mock_data.order.filled_avg_price = "300.50"
        mock_data.order.status = "partially_filled"
        mock_data.timestamp = datetime.now(UTC)

        await stream._handle_trade_update(mock_data)

        # Should still notify callbacks for partial fills
        callback.assert_called_once()
        fill = callback.call_args[0][0]
        assert fill.status == "partially_filled"

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_handle_trade_update_stores_history(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test that fills are stored in history."""
        stream = FillConfirmationStream(fill_stream_config)

        mock_data = MagicMock()
        mock_data.event = "fill"
        mock_data.order = MagicMock()
        mock_data.order.id = "history-order"
        mock_data.order.symbol = "SPY"
        mock_data.order.side = "buy"
        mock_data.order.filled_qty = "100.0"
        mock_data.order.filled_avg_price = "450.00"
        mock_data.order.status = "filled"
        mock_data.timestamp = datetime.now(UTC)

        await stream._handle_trade_update(mock_data)

        assert len(stream._fill_history) == 1
        assert stream._fill_history[0].order_id == "history-order"

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_handle_trade_update_non_fill_event(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test handling non-fill events (new, canceled, etc.)."""
        stream = FillConfirmationStream(fill_stream_config)

        callback = MagicMock()
        stream.register_callback(callback)

        mock_data = MagicMock()
        mock_data.event = "new"  # Order accepted but not filled
        mock_data.order = MagicMock()
        mock_data.order.id = "new-order"
        mock_data.order.status = "accepted"
        mock_data.timestamp = datetime.now(UTC)

        await stream._handle_trade_update(mock_data)

        # Should not call callback for non-fill events
        callback.assert_not_called()


class TestFillConfirmationStreamLifecycle:
    """Tests for stream lifecycle management."""

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_start_subscribes_to_updates(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test that start subscribes to trade updates."""
        mock_stream = MagicMock()
        # Mock .run() to complete immediately (sync method called via run_in_executor)
        mock_stream.run = MagicMock(side_effect=Exception("Test stop"))
        mock_stream_class.return_value = mock_stream

        stream = FillConfirmationStream(fill_stream_config)

        with pytest.raises(Exception, match="Test stop"):
            await stream.start()

        mock_stream.subscribe_trade_updates.assert_called_once()

    @patch("runner_py.fill_stream.TradingStream")
    @pytest.mark.asyncio
    async def test_stop_closes_stream(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test that stop closes the stream."""
        mock_stream = MagicMock()
        mock_stream.close = AsyncMock()
        mock_stream_class.return_value = mock_stream

        stream = FillConfirmationStream(fill_stream_config)
        await stream.stop()

        mock_stream.close.assert_called_once()


class TestFillConfirmationStreamHistory:
    """Tests for fill history operations."""

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_recent_fills(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test getting recent fills."""
        stream = FillConfirmationStream(fill_stream_config)

        # Add some fills to history
        now = datetime.now(UTC)
        for i in range(10):
            fill = PaperFill(
                order_id=f"order-{i}",
                symbol="AAPL",
                side="buy",
                qty=float(i + 1),
                filled_price=150.0 + i,
                filled_at=now,
                status="filled",
            )
            stream._fill_history.append(fill)

        # Get last 5
        recent = stream.get_recent_fills(limit=5)

        assert len(recent) == 5
        # Should be most recent first
        assert recent[0].order_id == "order-9"
        assert recent[4].order_id == "order-5"

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_recent_fills_empty(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test getting recent fills when empty."""
        stream = FillConfirmationStream(fill_stream_config)

        recent = stream.get_recent_fills(limit=10)

        assert len(recent) == 0

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_fill_by_order_id_found(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test finding a fill by order ID."""
        stream = FillConfirmationStream(fill_stream_config)

        fill = PaperFill(
            order_id="target-order",
            symbol="MSFT",
            side="sell",
            qty=20.0,
            filled_price=299.99,
            filled_at=datetime.now(UTC),
            status="filled",
        )
        stream._fill_history.append(fill)

        found = stream.get_fill_by_order_id("target-order")

        assert found is not None
        assert found.order_id == "target-order"
        assert found.symbol == "MSFT"

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_fill_by_order_id_not_found(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test searching for non-existent order."""
        stream = FillConfirmationStream(fill_stream_config)

        found = stream.get_fill_by_order_id("nonexistent-order")

        assert found is None

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_fills_by_symbol(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test filtering fills by symbol."""
        stream = FillConfirmationStream(fill_stream_config)

        now = datetime.now(UTC)
        stream._fill_history.append(
            PaperFill("o1", "AAPL", "buy", 10.0, 150.0, now, "filled")
        )
        stream._fill_history.append(
            PaperFill("o2", "MSFT", "buy", 5.0, 300.0, now, "filled")
        )
        stream._fill_history.append(
            PaperFill("o3", "AAPL", "sell", 10.0, 151.0, now, "filled")
        )

        aapl_fills = stream.get_fills_by_symbol("AAPL")

        assert len(aapl_fills) == 2
        assert all(f.symbol == "AAPL" for f in aapl_fills)

    @patch("runner_py.fill_stream.TradingStream")
    def test_clear_history(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test clearing fill history."""
        stream = FillConfirmationStream(fill_stream_config)

        now = datetime.now(UTC)
        stream._fill_history.append(
            PaperFill("o1", "AAPL", "buy", 10.0, 150.0, now, "filled")
        )

        assert len(stream._fill_history) == 1

        stream.clear_history()

        assert len(stream._fill_history) == 0


class TestFillConfirmationStreamStats:
    """Tests for stream statistics."""

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_stats_empty(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test stats with no fills."""
        stream = FillConfirmationStream(fill_stream_config)

        stats = stream.get_stats()

        assert stats["total_fills"] == 0
        assert stats["total_notional"] == 0.0

    @patch("runner_py.fill_stream.TradingStream")
    def test_get_stats_with_fills(
        self, mock_stream_class: MagicMock, fill_stream_config: FillStreamConfig
    ) -> None:
        """Test stats with multiple fills."""
        stream = FillConfirmationStream(fill_stream_config)

        now = datetime.now(UTC)
        stream._fill_history.append(
            PaperFill("o1", "AAPL", "buy", 10.0, 150.0, now, "filled")
        )
        stream._fill_history.append(
            PaperFill("o2", "MSFT", "sell", 5.0, 300.0, now, "filled")
        )

        stats = stream.get_stats()

        assert stats["total_fills"] == 2
        assert stats["total_notional"] == pytest.approx(10.0 * 150.0 + 5.0 * 300.0)
        assert stats["symbols_traded"] == {"AAPL", "MSFT"}
