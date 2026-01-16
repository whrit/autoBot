"""
Paper Executor - Execute signals via Alpaca Paper Trading API (T6.03).

Provides real paper trading execution through Alpaca's paper trading
environment without risking real capital.

CRITICAL: Always uses paper=True to ensure no live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest

from runner_py.types import ExecutionResult, Signal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class PaperConfig:
    """
    Configuration for paper trading executor.

    Attributes:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: Must be True for paper trading (enforced)
        max_position_value: Maximum notional value per position
    """

    api_key: str
    api_secret: str
    paper: bool = True
    max_position_value: float = 10000.0


@dataclass
class PaperFill:
    """
    Represents a paper trading fill from Alpaca.

    Attributes:
        order_id: Alpaca order ID
        symbol: Trading symbol
        side: Buy or sell
        qty: Quantity filled
        filled_price: Average fill price
        filled_at: Timestamp when filled
        status: Order status (filled, partially_filled, etc.)
    """

    order_id: str
    symbol: str
    side: str
    qty: float
    filled_price: float
    filled_at: datetime
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert fill to dictionary for logging."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "filled_price": self.filled_price,
            "filled_at": self.filled_at.isoformat(),
            "status": self.status,
        }


class PaperExecutor:
    """
    Execute signals via Alpaca paper trading API.

    Provides methods to:
    - Submit market orders based on signals
    - Track pending orders
    - Query order status and positions
    - Cancel orders

    CRITICAL: Always creates TradingClient with paper=True.
    """

    def __init__(self, config: PaperConfig) -> None:
        """
        Initialize the paper executor.

        Args:
            config: Paper trading configuration
        """
        self.config = config

        # CRITICAL: Always use paper=True regardless of config
        self.client = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=True,  # NEVER change this to False
        )

        self._pending_orders: dict[str, Signal] = {}

        logger.info("PaperExecutor initialized with paper trading mode")

    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Execute a signal via Alpaca paper API.

        Args:
            signal: Trading signal to execute

        Returns:
            ExecutionResult with order details or error
        """
        submitted_at = datetime.now(UTC)

        try:
            # Handle FLAT signal (close position)
            if signal.signal_type == SignalDirection.FLAT:
                return await self._execute_flat(signal, submitted_at)

            # Submit order for LONG or SHORT
            order_id = self.submit_order(signal)

            # Get order details
            order_response = self.client.get_order_by_id(order_id)
            order = cast(Order, order_response)

            return ExecutionResult(
                success=True,
                order_id=order_id,
                signal=signal,
                submitted_at=submitted_at,
                filled_at=order.filled_at if order.filled_at else None,
                filled_qty=float(order.filled_qty) if order.filled_qty else None,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )

        except Exception as e:
            logger.error(f"Paper execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                signal=signal,
                error=str(e),
                submitted_at=submitted_at,
            )

    async def _execute_flat(
        self, signal: Signal, submitted_at: datetime
    ) -> ExecutionResult:
        """
        Execute a FLAT signal by closing any existing position.

        Args:
            signal: FLAT signal
            submitted_at: When execution started

        Returns:
            ExecutionResult indicating success/failure
        """
        try:
            # Check if we have a position to close
            try:
                self.client.get_open_position(signal.symbol)
            except Exception:
                # No position to close
                logger.info(f"No position to close for {signal.symbol}")
                return ExecutionResult(
                    success=True,
                    signal=signal,
                    submitted_at=submitted_at,
                )

            # Close the position
            self.client.close_position(signal.symbol)

            return ExecutionResult(
                success=True,
                signal=signal,
                submitted_at=submitted_at,
            )

        except Exception as e:
            logger.error(f"Failed to close position for {signal.symbol}: {e}")
            return ExecutionResult(
                success=False,
                signal=signal,
                error=str(e),
                submitted_at=submitted_at,
            )

    def _create_order_request(self, signal: Signal) -> MarketOrderRequest:
        """
        Create Alpaca order request from signal.

        Args:
            signal: Trading signal

        Returns:
            MarketOrderRequest for Alpaca API
        """
        # Calculate notional with strength adjustment
        notional = signal.target_notional * signal.strength

        # Cap at max position value
        notional = min(notional, self.config.max_position_value)

        # Determine side based on signal type
        if signal.signal_type == SignalDirection.LONG:
            side = OrderSide.BUY
        elif signal.signal_type == SignalDirection.SHORT:
            side = OrderSide.SELL
        else:
            # FLAT should not reach here
            raise ValueError(f"Invalid signal type for order: {signal.signal_type}")

        return MarketOrderRequest(
            symbol=signal.symbol,
            notional=notional,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

    def submit_order(self, signal: Signal) -> str:
        """
        Submit order and return order ID.

        Args:
            signal: Trading signal to execute

        Returns:
            Alpaca order ID

        Raises:
            Exception: If order submission fails
        """
        order_request = self._create_order_request(signal)

        logger.info(
            f"Submitting paper order: {signal.symbol} {signal.signal_type.value} "
            f"notional=${order_request.notional}"
        )

        order_response = self.client.submit_order(order_request)
        order = cast(Order, order_response)

        # Track pending order - convert UUID to str
        order_id_str = str(order.id)
        self._pending_orders[order_id_str] = signal

        logger.info(f"Paper order submitted: {order_id_str} status={order.status.value}")

        return order_id_str

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get current order status.

        Args:
            order_id: Alpaca order ID

        Returns:
            Dictionary with order status details
        """
        order_response = self.client.get_order_by_id(order_id)
        order = cast(Order, order_response)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if order.side else None,
            "status": order.status.value if order.status else None,
            "filled_qty": str(order.filled_qty) if order.filled_qty else None,
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Alpaca order ID to cancel

        Returns:
            True if canceled successfully, False otherwise
        """
        try:
            self.client.cancel_order_by_id(order_id)
            self._clear_pending_order(order_id)
            logger.info(f"Paper order canceled: {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current paper positions.

        Returns:
            List of position dictionaries
        """
        positions_response = self.client.get_all_positions()

        result = []
        for p_response in positions_response:
            p = cast(Position, p_response)
            result.append({
                "symbol": p.symbol,
                "qty": str(p.qty),
                "avg_entry_price": str(p.avg_entry_price),
                "market_value": str(p.market_value),
                "side": str(p.side) if p.side else None,
            })
        return result

    def get_account(self) -> dict[str, Any]:
        """
        Get paper account info.

        Returns:
            Account details dictionary
        """
        account_response = self.client.get_account()
        account = cast(TradeAccount, account_response)

        return {
            "buying_power": str(account.buying_power),
            "cash": str(account.cash),
            "equity": str(account.equity),
            "status": str(account.status) if account.status else None,
        }

    def _clear_pending_order(self, order_id: str) -> None:
        """
        Clear an order from pending tracking.

        Args:
            order_id: Order ID to clear
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]

    def get_pending_orders(self) -> dict[str, Signal]:
        """
        Get all pending orders.

        Returns:
            Dictionary mapping order IDs to signals
        """
        return dict(self._pending_orders)
