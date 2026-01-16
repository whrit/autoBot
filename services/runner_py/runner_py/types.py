"""
Common types for the runner service.

Defines Signal, ExecutionResult, and related types used across
shadow execution, paper execution, and fill streaming.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    """Execution mode types."""

    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class SignalDirection(str, Enum):
    """Signal direction for compatibility with optimizer signals."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# Alias for backward compatibility
SignalType = SignalDirection


@dataclass(frozen=True, slots=True)
class Signal:
    """
    Trading signal for execution.

    Attributes:
        timestamp: Signal generation time
        symbol: Trading symbol (e.g., "AAPL", "SPY")
        signal_type: Long, short, or flat
        strength: Signal confidence (0.0 to 1.0)
        target_notional: Target position size in dollars
        strategy_id: Source strategy identifier
        metadata: Additional signal context
    """

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float = 1.0
    target_notional: float = 10000.0
    strategy_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "target_notional": self.target_notional,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """
    Result of a paper execution attempt.

    Attributes:
        success: Whether execution succeeded
        order_id: Alpaca order ID if submitted
        signal: Original signal
        error: Error message if failed
        submitted_at: Timestamp when order was submitted
        filled_at: Timestamp when order was filled (if known)
        filled_qty: Quantity filled
        filled_price: Average fill price
    """

    success: bool
    order_id: str | None = None
    signal: Signal | None = None
    error: str | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_qty: float | None = None
    filled_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for logging."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "signal": self.signal.to_dict() if self.signal else None,
            "error": self.error,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
        }


@dataclass
class ShadowFill:
    """
    Simulated fill for shadow execution.

    Attributes:
        signal: The original signal being simulated
        simulated_price: Simulated fill price based on quote data
        simulated_slippage_bps: Slippage applied in basis points
        timestamp: When the simulation was performed
        would_have_executed: Whether the order would have been filled
        rejection_reason: Reason for rejection if not executed
    """

    signal: Signal
    simulated_price: float
    simulated_slippage_bps: float
    timestamp: datetime
    would_have_executed: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "signal": self.signal.to_dict(),
            "simulated_price": self.simulated_price,
            "simulated_slippage_bps": self.simulated_slippage_bps,
            "timestamp": self.timestamp.isoformat(),
            "would_have_executed": self.would_have_executed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ShadowExecutionResult:
    """
    Result of executing a signal in shadow mode.

    Attributes:
        mode: Execution mode (always SHADOW for this type)
        signal: The original signal
        fill: The simulated fill information (None if not executed)
        success: Whether execution was successful
        message: Human-readable result message
    """

    mode: ExecutionMode
    signal: Signal
    fill: ShadowFill | None
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for logging."""
        return {
            "mode": self.mode.value,
            "signal": self.signal.to_dict(),
            "fill": self.fill.to_dict() if self.fill else None,
            "success": self.success,
            "message": self.message,
        }


@dataclass
class MarketData:
    """
    Current market data for a symbol.

    Attributes:
        symbol: The ticker symbol
        timestamp: When the data was captured
        bid: Best bid price
        ask: Best ask price
        last: Last trade price
        bid_size: Size at bid
        ask_size: Size at ask
    """

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points relative to mid price."""
        if self.mid == 0:
            return 0.0
        return (self.spread / self.mid) * 10000

    def to_dict(self) -> dict[str, Any]:
        """Convert market data to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
        }


@dataclass
class Position:
    """
    A position held in shadow or paper mode.

    Attributes:
        symbol: The ticker symbol
        quantity: Number of shares (negative for short)
        avg_price: Average entry price
        realized_pnl: Realized P&L from closed portions
        unrealized_pnl: Unrealized P&L at current mark
    """

    symbol: str
    quantity: float
    avg_price: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Get market value based on average price."""
        return self.quantity * self.avg_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0

    def update_unrealized_pnl(self, mark_price: float) -> None:
        """Update unrealized P&L based on mark price."""
        if self.quantity != 0:
            self.unrealized_pnl = (mark_price - self.avg_price) * self.quantity

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "market_value": self.market_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "is_long": self.is_long,
            "is_short": self.is_short,
            "is_flat": self.is_flat,
        }
