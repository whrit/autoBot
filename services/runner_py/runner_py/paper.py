"""
Paper Executor - Execute signals via Alpaca Paper Trading API (T6.03).

Provides real paper trading execution through Alpaca's paper trading
environment without risking real capital.

CRITICAL: Always uses paper=True to ensure no live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest

from runner_py.types import ExecutionResult, Signal, SignalDirection

logger = structlog.get_logger(__name__)


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


@dataclass
class PnLTracker:
    """Track real-time P&L for paper trading."""

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    total_notional: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "total_trades": self.total_trades,
            "total_notional": self.total_notional,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": win_rate,
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
        self._pnl_tracker = PnLTracker()
        self._order_history: list[dict[str, Any]] = []
        self._fill_history: list[PaperFill] = []
        self._session_start = datetime.now(UTC)
        self._logger = logger.bind(component="paper_executor")

        self._logger.info(
            "paper_executor_initialized",
            max_position_value=config.max_position_value,
            paper_mode=True,
        )

    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Execute a signal via Alpaca paper API.

        Args:
            signal: Trading signal to execute

        Returns:
            ExecutionResult with order details or error
        """
        submitted_at = datetime.now(UTC)

        # Log the incoming signal
        self._logger.info(
            "order_submission_started",
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            strength=signal.strength,
            target_notional=signal.target_notional,
            effective_notional=signal.target_notional * signal.strength,
            strategy_id=signal.strategy_id,
        )

        try:
            # Handle FLAT signal (close position)
            if signal.signal_type == SignalDirection.FLAT:
                return await self._execute_flat(signal, submitted_at)

            # Submit order for LONG or SHORT
            order_id = self.submit_order(signal)

            # Get order details
            order_response = self.client.get_order_by_id(order_id)
            order = cast(Order, order_response)

            # Log successful order submission
            self._logger.info(
                "order_submitted",
                order_id=order_id,
                symbol=signal.symbol,
                side="buy" if signal.signal_type == SignalDirection.LONG else "sell",
                status=order.status.value if order.status else None,
                submitted_at=submitted_at.isoformat(),
            )

            # Track order in history
            self._order_history.append({
                "order_id": order_id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "submitted_at": submitted_at.isoformat(),
                "status": order.status.value if order.status else None,
            })

            result = ExecutionResult(
                success=True,
                order_id=order_id,
                signal=signal,
                submitted_at=submitted_at,
                filled_at=order.filled_at if order.filled_at else None,
                filled_qty=float(order.filled_qty) if order.filled_qty else None,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )

            # Log fill if order was filled immediately
            if order.filled_at and order.filled_avg_price:
                self._log_fill(order, signal)

            return result

        except Exception as e:
            self._logger.error(
                "order_submission_failed",
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                error=str(e),
                error_type=type(e).__name__,
                submitted_at=submitted_at.isoformat(),
            )
            return ExecutionResult(
                success=False,
                signal=signal,
                error=str(e),
                submitted_at=submitted_at,
            )

    def _log_fill(self, order: Order, signal: Signal) -> None:
        """Log a fill and update P&L tracker."""
        filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
        filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
        notional = filled_qty * filled_price

        fill = PaperFill(
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value if order.side else "unknown",
            qty=filled_qty,
            filled_price=filled_price,
            filled_at=order.filled_at or datetime.now(UTC),
            status=order.status.value if order.status else "unknown",
        )
        self._fill_history.append(fill)

        # Update P&L tracker
        self._pnl_tracker.total_trades += 1
        self._pnl_tracker.total_notional += notional

        self._logger.info(
            "fill_received",
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value if order.side else None,
            qty=filled_qty,
            filled_price=filled_price,
            notional=notional,
            filled_at=order.filled_at.isoformat() if order.filled_at else None,
            status=order.status.value if order.status else None,
            total_trades=self._pnl_tracker.total_trades,
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
                position = self.client.get_open_position(signal.symbol)
                position = cast(Position, position)
                qty = float(position.qty) if position.qty else 0.0
                avg_price = float(position.avg_entry_price) if position.avg_entry_price else 0.0

                self._logger.info(
                    "closing_position",
                    symbol=signal.symbol,
                    qty=qty,
                    avg_entry_price=avg_price,
                    market_value=float(position.market_value) if position.market_value else 0.0,
                )
            except Exception:
                # No position to close
                self._logger.info(
                    "flat_signal_no_position",
                    symbol=signal.symbol,
                    message="No position to close",
                )
                return ExecutionResult(
                    success=True,
                    signal=signal,
                    submitted_at=submitted_at,
                )

            # Close the position
            self.client.close_position(signal.symbol)

            self._logger.info(
                "position_closed",
                symbol=signal.symbol,
                closed_qty=qty,
                avg_entry_price=avg_price,
            )

            return ExecutionResult(
                success=True,
                signal=signal,
                submitted_at=submitted_at,
            )

        except Exception as e:
            self._logger.error(
                "position_close_failed",
                symbol=signal.symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
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
        original_notional = notional
        notional = min(notional, self.config.max_position_value)

        if notional < original_notional:
            self._logger.warning(
                "notional_capped",
                symbol=signal.symbol,
                original_notional=original_notional,
                capped_notional=notional,
                max_position_value=self.config.max_position_value,
            )

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

        self._logger.info(
            "submitting_order",
            symbol=signal.symbol,
            side="buy" if signal.signal_type == SignalDirection.LONG else "sell",
            notional=order_request.notional,
            time_in_force="day",
            strategy_id=signal.strategy_id,
        )

        order_response = self.client.submit_order(order_request)
        order = cast(Order, order_response)

        # Track pending order - convert UUID to str
        order_id_str = str(order.id)
        self._pending_orders[order_id_str] = signal

        self._logger.info(
            "order_accepted",
            order_id=order_id_str,
            symbol=signal.symbol,
            status=order.status.value if order.status else None,
            client_order_id=str(order.client_order_id) if order.client_order_id else None,
        )

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

        status_info = {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if order.side else None,
            "status": order.status.value if order.status else None,
            "filled_qty": str(order.filled_qty) if order.filled_qty else None,
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }

        self._logger.debug(
            "order_status_retrieved",
            **status_info,
        )

        return status_info

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

            self._logger.info(
                "order_canceled",
                order_id=order_id,
            )
            return True
        except Exception as e:
            self._logger.warning(
                "order_cancel_failed",
                order_id=order_id,
                error=str(e),
                error_type=type(e).__name__,
            )
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
            pos_dict = {
                "symbol": p.symbol,
                "qty": str(p.qty),
                "avg_entry_price": str(p.avg_entry_price),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl) if p.unrealized_pl else "0",
                "unrealized_plpc": str(p.unrealized_plpc) if p.unrealized_plpc else "0",
                "side": str(p.side) if p.side else None,
            }
            result.append(pos_dict)

        self._logger.debug(
            "positions_retrieved",
            position_count=len(result),
            positions=result,
        )

        return result

    def get_positions_with_pnl(self) -> list[dict[str, Any]]:
        """
        Get current positions with P&L details.

        Returns:
            List of position dictionaries with P&L
        """
        positions = self.get_positions()

        total_unrealized_pnl = 0.0
        for pos in positions:
            unrealized = float(pos.get("unrealized_pl", 0))
            total_unrealized_pnl += unrealized

        self._pnl_tracker.unrealized_pnl = total_unrealized_pnl

        self._logger.info(
            "positions_pnl_summary",
            position_count=len(positions),
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=self._pnl_tracker.realized_pnl,
            total_pnl=self._pnl_tracker.realized_pnl + total_unrealized_pnl,
        )

        return positions

    def get_account(self) -> dict[str, Any]:
        """
        Get paper account info.

        Returns:
            Account details dictionary
        """
        account_response = self.client.get_account()
        account = cast(TradeAccount, account_response)

        account_info = {
            "buying_power": str(account.buying_power),
            "cash": str(account.cash),
            "equity": str(account.equity),
            "status": str(account.status) if account.status else None,
            "portfolio_value": str(account.portfolio_value) if account.portfolio_value else None,
        }

        self._logger.debug(
            "account_info_retrieved",
            **account_info,
        )

        return account_info

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

    def get_pnl_summary(self) -> dict[str, Any]:
        """
        Get real-time P&L summary.

        Returns:
            Dictionary with P&L metrics
        """
        # Refresh unrealized P&L from positions
        self.get_positions_with_pnl()
        return self._pnl_tracker.to_dict()

    def get_fill_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent fill history.

        Args:
            limit: Maximum number of fills to return

        Returns:
            List of fill dictionaries
        """
        fills = self._fill_history[-limit:]
        return [f.to_dict() for f in reversed(fills)]

    def log_session_summary(self) -> None:
        """Log a comprehensive session summary."""
        session_duration = (datetime.now(UTC) - self._session_start).total_seconds()
        pnl_summary = self.get_pnl_summary()
        positions = self.get_positions()

        self._logger.info(
            "session_summary",
            session_duration_seconds=session_duration,
            pnl_summary=pnl_summary,
            position_count=len(positions),
            order_count=len(self._order_history),
            fill_count=len(self._fill_history),
            pending_order_count=len(self._pending_orders),
        )
