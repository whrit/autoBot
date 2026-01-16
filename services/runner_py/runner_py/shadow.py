"""
Shadow Executor for simulating signal execution.

This module provides shadow execution mode that simulates trading signals
without placing real orders, useful for validation and analysis.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from runner_py.types import (
    ExecutionMode,
    MarketData,
    Position,
    ShadowExecutionResult,
    ShadowFill,
    Signal,
    SignalDirection,
)

logger = structlog.get_logger(__name__)


@dataclass
class SignalSummary:
    """Summary of signals by type for reporting."""

    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    total_notional: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "hold_count": self.hold_count,
            "total_signals": self.buy_count + self.sell_count + self.hold_count,
            "total_notional": self.total_notional,
        }


@dataclass
class ShadowConfig:
    """
    Configuration for shadow execution.

    Attributes:
        strategy_id: Identifier for the strategy being executed
        symbols: List of allowed symbols for execution
        max_position_size: Maximum position size in notional value
        slippage_model_bps: Slippage to apply in basis points
        check_risk_limits: Whether to check risk limits before execution
    """

    strategy_id: str
    symbols: list[str]
    max_position_size: float = 10000.0
    slippage_model_bps: float = 2.0
    check_risk_limits: bool = True


class ShadowExecutor:
    """
    Execute signals in shadow mode (no real orders).

    This executor simulates what would happen if signals were executed
    against real market data, including slippage modeling and position
    tracking.
    """

    def __init__(self, config: ShadowConfig) -> None:
        """
        Initialize shadow executor.

        Args:
            config: Shadow execution configuration
        """
        self.config = config
        self._positions: dict[str, Position] = {}
        self._trade_history: list[ShadowFill] = []
        self._market_prices: dict[str, float] = {}
        self._signal_summary = SignalSummary()
        self._session_start = datetime.now(UTC)
        self._logger = logger.bind(
            component="shadow_executor",
            strategy_id=config.strategy_id,
        )

        self._logger.info(
            "shadow_executor_initialized",
            symbols=config.symbols,
            max_position_size=config.max_position_size,
            slippage_model_bps=config.slippage_model_bps,
            check_risk_limits=config.check_risk_limits,
        )

    def execute(
        self, signal: Signal, market_data: MarketData
    ) -> ShadowExecutionResult:
        """
        Execute a signal in shadow mode.

        Args:
            signal: The signal to execute
            market_data: Current market data (bid, ask, mid prices)

        Returns:
            ShadowExecutionResult with simulated fill
        """
        # Log incoming signal with full structured context
        self._logger.info(
            "signal_received",
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            strength=signal.strength,
            target_notional=signal.target_notional,
            timestamp=signal.timestamp.isoformat(),
            market_bid=market_data.bid,
            market_ask=market_data.ask,
            market_mid=market_data.mid,
            spread_bps=market_data.spread_bps,
        )

        # Update signal summary
        self._update_signal_summary(signal)

        # Check risk limits if enabled
        if self.config.check_risk_limits:
            passed, reason = self.check_risk_limits(signal)
            if not passed:
                self._logger.warning(
                    "risk_limit_failed",
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    reason=reason,
                    strength=signal.strength,
                    target_notional=signal.target_notional,
                )
                return ShadowExecutionResult(
                    mode=ExecutionMode.SHADOW,
                    signal=signal,
                    fill=None,
                    success=False,
                    message=f"Risk limit check failed: {reason}",
                )

        # Handle FLAT signal (close position)
        if signal.signal_type == SignalDirection.FLAT:
            current_pos = self.get_position(signal.symbol)
            if current_pos == 0:
                self._logger.info(
                    "flat_signal_no_position",
                    symbol=signal.symbol,
                    message="No position to close",
                )
                return ShadowExecutionResult(
                    mode=ExecutionMode.SHADOW,
                    signal=signal,
                    fill=None,
                    success=True,
                    message="No position to close",
                )
            # Close existing position
            fill = self._create_close_fill(signal, market_data, current_pos)
        else:
            # Simulate the fill
            fill = self.simulate_fill(signal, market_data)

        # Update position if fill would have executed
        if fill.would_have_executed:
            self.update_position(fill)
            self._trade_history.append(fill)

            # Log the simulated trade (would-be trade)
            self._logger.info(
                "would_be_trade",
                symbol=signal.symbol,
                direction=signal.signal_type.value,
                simulated_price=fill.simulated_price,
                slippage_bps=fill.simulated_slippage_bps,
                effective_notional=signal.target_notional * signal.strength,
                timestamp=fill.timestamp.isoformat(),
                position_after=self.get_position(signal.symbol),
                total_pnl=self.get_total_pnl(),
            )

        self._logger.info(
            "shadow_execution_complete",
            symbol=signal.symbol,
            direction=signal.signal_type.value,
            price=fill.simulated_price,
            would_execute=fill.would_have_executed,
            success=fill.would_have_executed,
            trade_count=len(self._trade_history),
        )

        return ShadowExecutionResult(
            mode=ExecutionMode.SHADOW,
            signal=signal,
            fill=fill,
            success=fill.would_have_executed,
            message="Shadow execution successful"
            if fill.would_have_executed
            else "Execution failed",
        )

    def _update_signal_summary(self, signal: Signal) -> None:
        """Update the signal summary counters."""
        if signal.signal_type == SignalDirection.LONG:
            self._signal_summary.buy_count += 1
        elif signal.signal_type == SignalDirection.SHORT:
            self._signal_summary.sell_count += 1
        elif signal.signal_type == SignalDirection.FLAT:
            self._signal_summary.hold_count += 1

        self._signal_summary.total_notional += signal.target_notional * signal.strength

    def simulate_fill(self, signal: Signal, market_data: MarketData) -> ShadowFill:
        """
        Simulate what the fill would have been.

        Args:
            signal: The signal to simulate
            market_data: Current market data

        Returns:
            ShadowFill with simulated price and slippage
        """
        # Determine base price based on direction
        if signal.signal_type == SignalDirection.LONG:
            # Buy at ask + slippage
            base_price = market_data.ask
            slippage_direction = 1
        elif signal.signal_type == SignalDirection.SHORT:
            # Sell at bid - slippage
            base_price = market_data.bid
            slippage_direction = -1
        else:
            # FLAT - use mid price
            base_price = market_data.mid
            slippage_direction = 0

        # Apply slippage
        slippage_amount = base_price * (self.config.slippage_model_bps / 10000)
        simulated_price = base_price + (slippage_amount * slippage_direction)

        self._logger.debug(
            "fill_simulation",
            symbol=signal.symbol,
            direction=signal.signal_type.value,
            base_price=base_price,
            slippage_bps=self.config.slippage_model_bps,
            slippage_amount=slippage_amount,
            simulated_price=simulated_price,
        )

        return ShadowFill(
            signal=signal,
            simulated_price=simulated_price,
            simulated_slippage_bps=self.config.slippage_model_bps,
            timestamp=datetime.now(UTC),
            would_have_executed=True,
        )

    def _create_close_fill(
        self, signal: Signal, market_data: MarketData, position: float
    ) -> ShadowFill:
        """Create a fill for closing a position."""
        # Closing a long position = sell at bid
        # Closing a short position = buy at ask
        if position > 0:
            base_price = market_data.bid
            slippage_direction = -1
        else:
            base_price = market_data.ask
            slippage_direction = 1

        slippage_amount = base_price * (self.config.slippage_model_bps / 10000)
        simulated_price = base_price + (slippage_amount * slippage_direction)

        self._logger.info(
            "closing_position",
            symbol=signal.symbol,
            position_size=position,
            close_price=simulated_price,
            slippage_bps=self.config.slippage_model_bps,
        )

        return ShadowFill(
            signal=signal,
            simulated_price=simulated_price,
            simulated_slippage_bps=self.config.slippage_model_bps,
            timestamp=datetime.now(UTC),
            would_have_executed=True,
        )

    def check_risk_limits(self, signal: Signal) -> tuple[bool, str | None]:
        """
        Check if signal passes risk limits.

        Args:
            signal: The signal to check

        Returns:
            Tuple of (passed, reason) where reason is None if passed
        """
        if not self.config.check_risk_limits:
            return True, None

        # Check symbol is allowed
        if signal.symbol not in self.config.symbols:
            return False, f"Symbol {signal.symbol} not in allowed symbols"

        # Check position size
        effective_notional = signal.target_notional * signal.strength
        if effective_notional > self.config.max_position_size:
            max_size = self.config.max_position_size
            return False, f"Position size {effective_notional} exceeds maximum {max_size}"

        return True, None

    def get_position(self, symbol: str) -> float:
        """
        Get current shadow position for symbol.

        Args:
            symbol: The ticker symbol

        Returns:
            Position quantity (positive for long, negative for short)
        """
        if symbol not in self._positions:
            return 0.0
        return self._positions[symbol].quantity

    def update_position(self, fill: ShadowFill) -> None:
        """
        Update shadow position after simulated fill.

        Args:
            fill: The simulated fill to apply
        """
        symbol = fill.signal.symbol
        signal = fill.signal

        # Calculate quantity based on notional and price
        effective_notional = signal.target_notional * signal.strength
        quantity = effective_notional / fill.simulated_price

        # Direction adjustment
        if signal.signal_type == SignalDirection.SHORT:
            quantity = -quantity
        elif signal.signal_type == SignalDirection.FLAT:
            # Close position
            quantity = -self.get_position(symbol)

        old_qty = self.get_position(symbol)

        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill.simulated_price,
            )
        else:
            pos = self._positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + quantity

            # Calculate new average price
            if new_qty == 0:
                # Position closed - calculate realized P&L
                if old_qty > 0:
                    # Was long, now closing
                    realized_pnl = (fill.simulated_price - pos.avg_price) * abs(quantity)
                else:
                    # Was short, now closing
                    realized_pnl = (pos.avg_price - fill.simulated_price) * abs(quantity)
                pos.realized_pnl += realized_pnl
                pos.quantity = 0
                pos.avg_price = 0

                self._logger.info(
                    "position_closed",
                    symbol=symbol,
                    realized_pnl=realized_pnl,
                    total_realized_pnl=pos.realized_pnl,
                )
            elif (old_qty > 0 and quantity > 0) or (old_qty < 0 and quantity < 0):
                # Adding to position - weighted average
                total_cost = (abs(old_qty) * pos.avg_price) + (abs(quantity) * fill.simulated_price)
                pos.avg_price = total_cost / abs(new_qty)
                pos.quantity = new_qty
            else:
                # Reducing position
                if abs(quantity) >= abs(old_qty):
                    # Reversing position
                    remaining = quantity + old_qty
                    if old_qty > 0:
                        realized_pnl = (fill.simulated_price - pos.avg_price) * old_qty
                    else:
                        realized_pnl = (pos.avg_price - fill.simulated_price) * abs(old_qty)
                    pos.realized_pnl += realized_pnl
                    pos.quantity = remaining
                    pos.avg_price = fill.simulated_price if remaining != 0 else 0
                else:
                    # Partial close
                    if old_qty > 0:
                        realized_pnl = (fill.simulated_price - pos.avg_price) * abs(quantity)
                    else:
                        realized_pnl = (pos.avg_price - fill.simulated_price) * abs(quantity)
                    pos.realized_pnl += realized_pnl
                    pos.quantity = new_qty

        # Update market price for unrealized P&L
        self._market_prices[symbol] = fill.simulated_price

        # Log position update
        self._logger.info(
            "position_updated",
            symbol=symbol,
            old_quantity=old_qty,
            new_quantity=self.get_position(symbol),
            avg_price=self._positions[symbol].avg_price if symbol in self._positions else 0,
            fill_price=fill.simulated_price,
        )

    def mark_to_market(self, market_data: MarketData) -> None:
        """
        Update unrealized P&L based on current market data.

        Args:
            market_data: Current market data
        """
        symbol = market_data.symbol
        if symbol in self._positions:
            mark_price = market_data.mid
            old_unrealized = self._positions[symbol].unrealized_pnl
            self._positions[symbol].update_unrealized_pnl(mark_price)
            self._market_prices[symbol] = mark_price

            self._logger.debug(
                "mark_to_market",
                symbol=symbol,
                mark_price=mark_price,
                old_unrealized_pnl=old_unrealized,
                new_unrealized_pnl=self._positions[symbol].unrealized_pnl,
            )

    def get_pnl(self) -> dict[str, float]:
        """
        Get current shadow P&L by symbol.

        Returns:
            Dictionary mapping symbol to total P&L (realized + unrealized)
        """
        pnl: dict[str, float] = {}
        for symbol, pos in self._positions.items():
            pnl[symbol] = pos.realized_pnl + pos.unrealized_pnl
        return pnl

    def get_total_pnl(self) -> float:
        """
        Get total P&L across all positions.

        Returns:
            Total P&L (realized + unrealized)
        """
        return sum(self.get_pnl().values())

    def get_trade_history(self) -> list[ShadowFill]:
        """
        Get history of all simulated trades.

        Returns:
            List of ShadowFill objects
        """
        return list(self._trade_history)

    def get_signal_summary(self) -> dict[str, Any]:
        """
        Get summary of signals processed by type.

        Returns:
            Dictionary with signal counts by type
        """
        return self._signal_summary.to_dict()

    def log_session_summary(self) -> None:
        """Log a comprehensive session summary."""
        session_duration = (datetime.now(UTC) - self._session_start).total_seconds()

        self._logger.info(
            "session_summary",
            strategy_id=self.config.strategy_id,
            session_duration_seconds=session_duration,
            total_trades=len(self._trade_history),
            total_pnl=self.get_total_pnl(),
            signal_summary=self._signal_summary.to_dict(),
            positions=self.get_position_summary(),
            pnl_by_symbol=self.get_pnl(),
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get executor metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "total_trades": len(self._trade_history),
            "total_pnl": self.get_total_pnl(),
            "positions": {s: p.to_dict() for s, p in self._positions.items()},
            "strategy_id": self.config.strategy_id,
            "signal_summary": self._signal_summary.to_dict(),
            "session_start": self._session_start.isoformat(),
        }

    def get_position_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get summary of all positions.

        Returns:
            Dictionary mapping symbol to position details
        """
        return {symbol: pos.to_dict() for symbol, pos in self._positions.items()}
