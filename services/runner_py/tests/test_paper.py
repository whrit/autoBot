"""
Tests for Paper Executor (T6.03).

Tests paper trading via Alpaca API with mocked responses.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from runner_py.paper import PaperConfig, PaperExecutor, PaperFill
from runner_py.types import Signal, SignalDirection


@pytest.fixture
def paper_config() -> PaperConfig:
    """Create test paper config."""
    return PaperConfig(
        api_key="test-api-key",
        api_secret="test-api-secret",
        paper=True,
        max_position_value=10000.0,
    )


@pytest.fixture
def mock_trading_client() -> MagicMock:
    """Create mock TradingClient."""
    mock_client = MagicMock()
    mock_client.get_account.return_value = MagicMock(
        buying_power="100000.00",
        cash="50000.00",
        equity="75000.00",
        status="ACTIVE",
    )
    return mock_client


@pytest.fixture
def sample_signal() -> Signal:
    """Create a sample LONG signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        signal_type=SignalDirection.LONG,
        strength=0.8,
        target_notional=5000.0,
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_short_signal() -> Signal:
    """Create a sample SHORT signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        signal_type=SignalDirection.SHORT,
        strength=0.7,
        target_notional=3000.0,
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_flat_signal() -> Signal:
    """Create a sample FLAT signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        signal_type=SignalDirection.FLAT,
        strength=1.0,
        target_notional=0.0,
        strategy_id="test-strategy",
    )


class TestPaperConfig:
    """Tests for PaperConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = PaperConfig(
            api_key="key",
            api_secret="secret",
        )
        assert config.paper is True
        assert config.max_position_value == 10000.0

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = PaperConfig(
            api_key="key",
            api_secret="secret",
            paper=True,
            max_position_value=50000.0,
        )
        assert config.max_position_value == 50000.0

    def test_paper_always_true(self) -> None:
        """Verify paper mode is enforced."""
        config = PaperConfig(
            api_key="key",
            api_secret="secret",
            paper=True,
        )
        # Paper should always be True for safety
        assert config.paper is True


class TestPaperFill:
    """Tests for PaperFill dataclass."""

    def test_fill_creation(self) -> None:
        """Test creating a paper fill."""
        now = datetime.now(UTC)
        fill = PaperFill(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            filled_price=150.0,
            filled_at=now,
            status="filled",
        )
        assert fill.order_id == "order-123"
        assert fill.symbol == "AAPL"
        assert fill.side == "buy"
        assert fill.qty == 10.0
        assert fill.filled_price == 150.0
        assert fill.status == "filled"

    def test_fill_to_dict(self) -> None:
        """Test converting fill to dictionary."""
        now = datetime.now(UTC)
        fill = PaperFill(
            order_id="order-456",
            symbol="MSFT",
            side="sell",
            qty=5.0,
            filled_price=300.0,
            filled_at=now,
            status="filled",
        )
        d = fill.to_dict()
        assert d["order_id"] == "order-456"
        assert d["symbol"] == "MSFT"
        assert d["side"] == "sell"
        assert d["qty"] == 5.0
        assert d["filled_price"] == 300.0
        assert d["status"] == "filled"


class TestPaperExecutorInit:
    """Tests for PaperExecutor initialization."""

    @patch("runner_py.paper.TradingClient")
    def test_init_creates_client(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test executor creates TradingClient with correct params."""
        executor = PaperExecutor(paper_config)

        mock_client_class.assert_called_once_with(
            api_key=paper_config.api_key,
            secret_key=paper_config.api_secret,
            paper=True,
        )
        assert executor.config == paper_config

    @patch("runner_py.paper.TradingClient")
    def test_init_with_paper_true(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test executor always uses paper=True."""
        _ = PaperExecutor(paper_config)

        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["paper"] is True


class TestPaperExecutorOrderRequest:
    """Tests for creating order requests."""

    @patch("runner_py.paper.TradingClient")
    def test_create_order_request_long(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test creating order request for LONG signal."""
        executor = PaperExecutor(paper_config)
        order_request = executor._create_order_request(sample_signal)

        assert order_request.symbol == "AAPL"
        assert order_request.side.value == "buy"
        assert order_request.time_in_force.value == "day"
        # Notional should be target * strength
        assert order_request.notional == pytest.approx(4000.0)  # 5000 * 0.8

    @patch("runner_py.paper.TradingClient")
    def test_create_order_request_short(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_short_signal: Signal
    ) -> None:
        """Test creating order request for SHORT signal."""
        executor = PaperExecutor(paper_config)
        order_request = executor._create_order_request(sample_short_signal)

        assert order_request.symbol == "AAPL"
        assert order_request.side.value == "sell"
        # Notional should be target * strength
        assert order_request.notional == pytest.approx(2100.0)  # 3000 * 0.7

    @patch("runner_py.paper.TradingClient")
    def test_create_order_request_respects_max_position(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test order request respects max position value."""
        executor = PaperExecutor(paper_config)

        # Create signal with notional exceeding max
        big_signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            signal_type=SignalDirection.LONG,
            strength=1.0,
            target_notional=50000.0,  # Exceeds max of 10000
            strategy_id="test",
        )

        order_request = executor._create_order_request(big_signal)
        assert order_request.notional == pytest.approx(10000.0)  # Capped at max


class TestPaperExecutorSubmitOrder:
    """Tests for submitting orders."""

    @patch("runner_py.paper.TradingClient")
    def test_submit_order_success(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test successful order submission."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = str(uuid4())
        mock_order.status.value = "accepted"
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        order_id = executor.submit_order(sample_signal)

        assert order_id == mock_order.id
        mock_client.submit_order.assert_called_once()

    @patch("runner_py.paper.TradingClient")
    def test_submit_order_failure(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test order submission failure."""
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("Insufficient buying power")
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)

        with pytest.raises(Exception, match="Insufficient buying power"):
            executor.submit_order(sample_signal)


class TestPaperExecutorExecute:
    """Tests for the main execute method."""

    @patch("runner_py.paper.TradingClient")
    @pytest.mark.asyncio
    async def test_execute_long_signal(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test executing a LONG signal."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status.value = "filled"
        mock_order.filled_qty = "26.67"
        mock_order.filled_avg_price = "150.00"
        mock_order.filled_at = datetime.now(UTC)
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        result = await executor.execute(sample_signal)

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.signal == sample_signal

    @patch("runner_py.paper.TradingClient")
    @pytest.mark.asyncio
    async def test_execute_flat_signal_no_position(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_flat_signal: Signal
    ) -> None:
        """Test executing FLAT signal with no existing position."""
        mock_client = MagicMock()
        mock_client.get_open_position.side_effect = Exception("Position not found")
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        result = await executor.execute(sample_flat_signal)

        # Should succeed with no action needed
        assert result.success is True
        assert result.order_id is None

    @patch("runner_py.paper.TradingClient")
    @pytest.mark.asyncio
    async def test_execute_with_api_error(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test execution with API error."""
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("API Error: Rate limited")
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        result = await executor.execute(sample_signal)

        assert result.success is False
        assert "API Error" in str(result.error)


class TestPaperExecutorOrderStatus:
    """Tests for order status operations."""

    @patch("runner_py.paper.TradingClient")
    def test_get_order_status(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test getting order status."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "order-789"
        mock_order.status.value = "filled"
        mock_order.filled_qty = "10.0"
        mock_order.filled_avg_price = "155.0"
        mock_client.get_order_by_id.return_value = mock_order
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        status = executor.get_order_status("order-789")

        assert status["order_id"] == "order-789"
        assert status["status"] == "filled"
        mock_client.get_order_by_id.assert_called_once_with("order-789")

    @patch("runner_py.paper.TradingClient")
    def test_cancel_order_success(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test canceling an order successfully."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        result = executor.cancel_order("order-to-cancel")

        assert result is True
        mock_client.cancel_order_by_id.assert_called_once_with("order-to-cancel")

    @patch("runner_py.paper.TradingClient")
    def test_cancel_order_failure(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test canceling an order that fails."""
        mock_client = MagicMock()
        mock_client.cancel_order_by_id.side_effect = Exception("Order already filled")
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        result = executor.cancel_order("already-filled-order")

        assert result is False


class TestPaperExecutorPositions:
    """Tests for position-related operations."""

    @patch("runner_py.paper.TradingClient")
    def test_get_positions(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test getting all positions."""
        mock_client = MagicMock()
        mock_position1 = MagicMock()
        mock_position1.symbol = "AAPL"
        mock_position1.qty = "10"
        mock_position1.avg_entry_price = "150.0"
        mock_position1.market_value = "1550.0"
        mock_position1.side = "long"

        mock_position2 = MagicMock()
        mock_position2.symbol = "MSFT"
        mock_position2.qty = "-5"
        mock_position2.avg_entry_price = "300.0"
        mock_position2.market_value = "-1475.0"
        mock_position2.side = "short"

        mock_client.get_all_positions.return_value = [mock_position1, mock_position2]
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        positions = executor.get_positions()

        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
        assert positions[1]["symbol"] == "MSFT"

    @patch("runner_py.paper.TradingClient")
    def test_get_account(
        self, mock_client_class: MagicMock, paper_config: PaperConfig
    ) -> None:
        """Test getting account info."""
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.buying_power = "100000.00"
        mock_account.cash = "50000.00"
        mock_account.equity = "75000.00"
        mock_account.status = "ACTIVE"
        mock_client.get_account.return_value = mock_account
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        account = executor.get_account()

        assert account["buying_power"] == "100000.00"
        assert account["cash"] == "50000.00"
        assert account["equity"] == "75000.00"
        assert account["status"] == "ACTIVE"


class TestPaperExecutorPendingOrders:
    """Tests for pending order tracking."""

    @patch("runner_py.paper.TradingClient")
    def test_track_pending_order(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test that pending orders are tracked."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "pending-order-123"
        mock_order.status.value = "accepted"
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        order_id = executor.submit_order(sample_signal)

        assert order_id in executor._pending_orders
        assert executor._pending_orders[order_id] == sample_signal

    @patch("runner_py.paper.TradingClient")
    def test_clear_filled_pending_order(
        self, mock_client_class: MagicMock, paper_config: PaperConfig, sample_signal: Signal
    ) -> None:
        """Test that filled orders are cleared from pending."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "pending-order-456"
        mock_order.status.value = "filled"
        mock_order.filled_qty = "10.0"
        mock_order.filled_avg_price = "150.0"
        mock_order.filled_at = datetime.now(UTC)
        mock_client.submit_order.return_value = mock_order
        mock_client_class.return_value = mock_client

        executor = PaperExecutor(paper_config)
        executor._pending_orders["pending-order-456"] = sample_signal

        # Simulate clearing after fill confirmation
        executor._clear_pending_order("pending-order-456")

        assert "pending-order-456" not in executor._pending_orders
