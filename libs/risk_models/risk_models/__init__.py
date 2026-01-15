"""
Risk Models Library - Position limits, exposure management, and risk checks.

This module provides:
- RiskLimits: Configuration for risk thresholds
- RiskState: Current risk state tracking
- RiskChecker: Pre-trade and state risk validation
- RiskViolation: Represents a risk limit breach
- RiskViolationType: Enumeration of violation types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class RiskViolationType(str, Enum):
    """Types of risk limit violations."""

    MAX_POSITION = "max_position"
    MAX_GROSS_EXPOSURE = "max_gross_exposure"
    MAX_NET_EXPOSURE = "max_net_exposure"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class RiskViolation:
    """Represents a risk limit violation.

    Attributes:
        type: The type of violation that occurred
        limit: The configured limit that was breached
        current: The current/proposed value that violates the limit
        message: Human-readable description of the violation
    """

    type: RiskViolationType
    limit: float
    current: float
    message: str


@dataclass
class RiskLimits:
    """Risk limit configuration.

    All limits must be positive values. These thresholds define the maximum
    acceptable risk exposure for the trading system.

    Attributes:
        max_position_notional: Maximum notional value for a single position
        max_gross_exposure: Maximum total absolute exposure across all positions
        max_net_exposure: Maximum net long/short exposure
        max_daily_loss: Maximum loss allowed in a single trading day
        max_drawdown_pct: Maximum drawdown as percentage of peak equity (0 to 1)

    Raises:
        ValueError: If any limit is non-positive or drawdown_pct is out of range
    """

    max_position_notional: float
    max_gross_exposure: float
    max_net_exposure: float
    max_daily_loss: float
    max_drawdown_pct: float

    def __post_init__(self) -> None:
        """Validate all limits are positive."""
        if self.max_position_notional <= 0:
            raise ValueError("max_position_notional must be positive")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.max_net_exposure <= 0:
            raise ValueError("max_net_exposure must be positive")
        if self.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError("max_drawdown_pct must be between 0 and 1")


@dataclass
class RiskState:
    """Current risk state tracking.

    Maintains the current positions, P&L, and equity levels for risk monitoring.

    Attributes:
        positions: Dictionary mapping symbols to notional values (+ = long, - = short)
        daily_pnl: Profit/loss for the current trading day
        peak_equity: Highest equity level reached (for drawdown calculation)
        current_equity: Current account equity value
    """

    positions: dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0

    @property
    def current_gross_exposure(self) -> float:
        """Total absolute position value across all positions.

        Returns:
            Sum of absolute values of all positions
        """
        return sum(abs(v) for v in self.positions.values())

    @property
    def current_net_exposure(self) -> float:
        """Net long/short exposure.

        Returns:
            Absolute value of sum of all positions
        """
        return abs(sum(self.positions.values()))

    def update_position(self, symbol: str, notional: float) -> None:
        """Update position for a symbol.

        Args:
            symbol: Trading symbol
            notional: Notional value to add (positive = buy, negative = sell)
        """
        current = self.positions.get(symbol, 0.0)
        new_value = current + notional

        if new_value == 0.0:
            # Remove position when flat
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_value

    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current notional position (0 if no position)
        """
        return self.positions.get(symbol, 0.0)

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown as percentage of peak equity.

        Returns:
            Drawdown percentage (0 to 1), or 0 if peak is zero
        """
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity


@dataclass
class RiskChecker:
    """Pre-trade and state risk checker.

    Validates orders and state against configured risk limits.

    Attributes:
        limits: The risk limits configuration to check against
    """

    limits: RiskLimits

    def check_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        notional: float,
        state: RiskState,
    ) -> list[RiskViolation]:
        """Check if an order would violate any risk limits.

        Performs pre-trade risk validation by calculating the hypothetical
        new state after the order and checking against limits.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            notional: Order size in notional terms
            state: Current risk state

        Returns:
            List of violations (empty if order is acceptable)
        """
        violations: list[RiskViolation] = []

        # Calculate hypothetical new position
        current_position = state.get_position(symbol)
        if side == "buy":
            new_position = current_position + notional
        else:
            new_position = current_position - notional

        # Check position limit
        if abs(new_position) > self.limits.max_position_notional:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_POSITION,
                    limit=self.limits.max_position_notional,
                    current=abs(new_position),
                    message=f"Position {symbol} would exceed limit: "
                    f"{abs(new_position):.0f} > {self.limits.max_position_notional:.0f}",
                )
            )

        # Calculate hypothetical gross exposure
        hypothetical_positions = state.positions.copy()
        if new_position == 0.0:
            hypothetical_positions.pop(symbol, None)
        else:
            hypothetical_positions[symbol] = new_position

        new_gross = sum(abs(v) for v in hypothetical_positions.values())
        if new_gross > self.limits.max_gross_exposure:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_GROSS_EXPOSURE,
                    limit=self.limits.max_gross_exposure,
                    current=new_gross,
                    message=f"Gross exposure would exceed limit: "
                    f"{new_gross:.0f} > {self.limits.max_gross_exposure:.0f}",
                )
            )

        # Calculate hypothetical net exposure
        new_net = abs(sum(hypothetical_positions.values()))
        if new_net > self.limits.max_net_exposure:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_NET_EXPOSURE,
                    limit=self.limits.max_net_exposure,
                    current=new_net,
                    message=f"Net exposure would exceed limit: "
                    f"{new_net:.0f} > {self.limits.max_net_exposure:.0f}",
                )
            )

        return violations

    def check_state(self, state: RiskState) -> list[RiskViolation]:
        """Check current state for risk violations.

        Validates the current risk state against all configured limits.

        Args:
            state: Current risk state

        Returns:
            List of violations
        """
        violations: list[RiskViolation] = []

        # Check daily loss
        if state.daily_pnl < -self.limits.max_daily_loss:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_DAILY_LOSS,
                    limit=self.limits.max_daily_loss,
                    current=abs(state.daily_pnl),
                    message=f"Daily loss exceeds limit: "
                    f"{abs(state.daily_pnl):.0f} > {self.limits.max_daily_loss:.0f}",
                )
            )

        # Check drawdown
        if state.current_drawdown_pct > self.limits.max_drawdown_pct:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_DRAWDOWN,
                    limit=self.limits.max_drawdown_pct,
                    current=state.current_drawdown_pct,
                    message=f"Drawdown exceeds limit: "
                    f"{state.current_drawdown_pct:.1%} > {self.limits.max_drawdown_pct:.1%}",
                )
            )

        # Check gross exposure
        if state.current_gross_exposure > self.limits.max_gross_exposure:
            violations.append(
                RiskViolation(
                    type=RiskViolationType.MAX_GROSS_EXPOSURE,
                    limit=self.limits.max_gross_exposure,
                    current=state.current_gross_exposure,
                    message=f"Gross exposure exceeds limit: "
                    f"{state.current_gross_exposure:.0f} > {self.limits.max_gross_exposure:.0f}",
                )
            )

        return violations

    def is_kill_switch_triggered(self, state: RiskState) -> bool:
        """Check if kill switch should be triggered.

        The kill switch is activated on critical violations that require
        immediate position liquidation: daily loss and drawdown.

        Args:
            state: Current risk state

        Returns:
            True if kill switch should be triggered
        """
        violations = self.check_state(state)
        # Kill switch on daily loss or drawdown violations only
        critical_types = {
            RiskViolationType.MAX_DAILY_LOSS,
            RiskViolationType.MAX_DRAWDOWN,
        }
        return any(v.type in critical_types for v in violations)


# Public API exports
__all__ = [
    "RiskViolationType",
    "RiskViolation",
    "RiskLimits",
    "RiskState",
    "RiskChecker",
]
