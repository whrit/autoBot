"""
Risk Management Module (T3.04).

Implements position limits and stops:
- Max position size per symbol
- Max gross exposure
- Stop-loss logic (price-based and time-based)
- Integration with RiskChecker from risk_models

Example usage:
    >>> from backtester_py.risk import RiskManager, StopConfig
    >>> manager = RiskManager(risk_limits, StopConfig(stop_loss_pct=0.02))
    >>> allowed, adjusted, violation = manager.check_order("AAPL", "buy", 10000, state)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from risk_models import RiskChecker, RiskLimits, RiskState, RiskViolation

if TYPE_CHECKING:
    pass


@dataclass
class StopConfig:
    """
    Configuration for stop-loss and take-profit logic.

    Attributes:
        stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%).
        take_profit_pct: Take profit percentage (e.g., 0.05 for 5%).
        max_hold_time_seconds: Maximum hold time before forced exit.
        trailing_stop_pct: Trailing stop percentage (None for fixed stop).
    """

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_hold_time_seconds: float | None = None
    trailing_stop_pct: float | None = None


@dataclass
class PositionEntry:
    """
    Tracks an open position entry.

    Attributes:
        symbol: Trading symbol.
        side: Position side ("buy" for long, "sell" for short).
        entry_price: Price at which position was entered.
        entry_time: Time of position entry.
        notional: Position size in notional terms.
        highest_price: Highest price since entry (for trailing stops).
        lowest_price: Lowest price since entry (for trailing stops).
        exit_reason: Reason for exit (None while open).
    """

    symbol: str
    side: Literal["buy", "sell"]
    entry_price: float
    entry_time: datetime
    notional: float
    highest_price: float | None = None
    lowest_price: float | None = None
    exit_reason: str | None = None


class RiskManager:
    """
    Manages position limits and stop-loss logic.

    Integrates with the risk_models library for pre-trade validation
    and implements stop-loss triggers for position management.

    Attributes:
        risk_limits: Configuration for risk thresholds.
        stop_config: Configuration for stop-loss logic.
        risk_checker: RiskChecker instance from risk_models.
        positions: Dictionary of open positions by symbol.
    """

    def __init__(
        self,
        risk_limits: RiskLimits,
        stop_config: StopConfig | None = None,
    ) -> None:
        """
        Initialize the risk manager.

        Args:
            risk_limits: RiskLimits configuration from risk_models.
            stop_config: Optional stop-loss configuration.
        """
        self.risk_limits = risk_limits
        self.stop_config = stop_config or StopConfig()
        self.risk_checker = RiskChecker(limits=risk_limits)
        self._positions: dict[str, PositionEntry] = {}
        self._exit_reasons: dict[str, str] = {}

    def check_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        notional: float,
        current_state: RiskState,
    ) -> tuple[bool, float, list[RiskViolation]]:
        """
        Check if an order is allowed and potentially adjust size.

        Args:
            symbol: Trading symbol.
            side: Order side ("buy" or "sell").
            notional: Requested order size in notional terms.
            current_state: Current risk state.

        Returns:
            Tuple of (allowed, adjusted_notional, violations).
            If not allowed, adjusted_notional is the maximum allowed size.
        """
        violations = self.risk_checker.check_order(
            symbol=symbol,
            side=side,
            notional=notional,
            state=current_state,
        )

        if not violations:
            return True, notional, []

        # Try to find a valid size
        adjusted_notional = self._calculate_max_allowed_size(
            symbol=symbol,
            side=side,
            requested_notional=notional,
            current_state=current_state,
        )

        if adjusted_notional > 0:
            # Re-check with adjusted size
            new_violations = self.risk_checker.check_order(
                symbol=symbol,
                side=side,
                notional=adjusted_notional,
                state=current_state,
            )
            if not new_violations:
                return True, adjusted_notional, violations

        return False, adjusted_notional, violations

    def _calculate_max_allowed_size(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        requested_notional: float,
        current_state: RiskState,
    ) -> float:
        """Calculate the maximum allowed order size given current state."""
        max_position = self.risk_limits.max_position_notional
        max_gross = self.risk_limits.max_gross_exposure
        max_net = self.risk_limits.max_net_exposure

        # Current position in this symbol
        current_position = current_state.get_position(symbol)
        current_gross = current_state.current_gross_exposure
        current_net = sum(current_state.positions.values())

        # Calculate limits
        limits = []

        # Position limit
        if side == "buy":
            pos_limit = max_position - current_position
        else:
            pos_limit = max_position + current_position
        limits.append(max(0, pos_limit))

        # Gross exposure limit
        gross_available = max_gross - current_gross
        limits.append(max(0, gross_available))

        # Net exposure limit
        net_available = max_net - current_net if side == "buy" else max_net + current_net
        limits.append(max(0, net_available))

        # Return minimum of all limits, capped at requested
        return float(min(min(limits), requested_notional))

    def record_entry(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        entry_price: float,
        entry_time: datetime,
        notional: float,
    ) -> None:
        """
        Record a position entry for stop tracking.

        Args:
            symbol: Trading symbol.
            side: Position side ("buy" or "sell").
            entry_price: Entry price.
            entry_time: Entry timestamp.
            notional: Position notional.
        """
        self._positions[symbol] = PositionEntry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            notional=notional,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        # Clear any previous exit reason
        self._exit_reasons.pop(symbol, None)

    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
    ) -> None:
        """
        Record a position exit.

        Args:
            symbol: Trading symbol.
            exit_price: Exit price.
            exit_time: Exit timestamp.
        """
        self._positions.pop(symbol, None)

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position in a symbol."""
        return symbol in self._positions

    def get_position_side(self, symbol: str) -> Literal["buy", "sell"] | None:
        """Get the side of an open position."""
        position = self._positions.get(symbol)
        return position.side if position else None

    def get_position(self, symbol: str) -> PositionEntry | None:
        """Get full position entry details."""
        return self._positions.get(symbol)

    def check_stop(
        self,
        symbol: str,
        current_price: float,
        current_time: datetime,
    ) -> bool:
        """
        Check if any stop condition is triggered.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            current_time: Current timestamp.

        Returns:
            True if a stop is triggered and position should be exited.
        """
        position = self._positions.get(symbol)
        if not position:
            return False

        # Update price tracking
        if position.highest_price is None or current_price > position.highest_price:
            position.highest_price = current_price
        if position.lowest_price is None or current_price < position.lowest_price:
            position.lowest_price = current_price

        # Check stop loss
        if self._check_stop_loss(position, current_price):
            self._exit_reasons[symbol] = "stop_loss"
            return True

        # Check take profit
        if self._check_take_profit(position, current_price):
            self._exit_reasons[symbol] = "take_profit"
            return True

        # Check time stop
        if self._check_time_stop(position, current_time):
            self._exit_reasons[symbol] = "time_stop"
            return True

        # Check trailing stop
        if self._check_trailing_stop(position, current_price):
            self._exit_reasons[symbol] = "trailing_stop"
            return True

        return False

    def _check_stop_loss(
        self, position: PositionEntry, current_price: float
    ) -> bool:
        """Check if stop loss is triggered."""
        if self.stop_config.stop_loss_pct is None:
            return False

        entry_price = position.entry_price
        stop_pct = self.stop_config.stop_loss_pct

        if position.side == "buy":
            # Long: stop if price drops below entry - stop%
            stop_price = entry_price * (1 - stop_pct)
            return current_price <= stop_price
        else:
            # Short: stop if price rises above entry + stop%
            stop_price = entry_price * (1 + stop_pct)
            return current_price >= stop_price

    def _check_take_profit(
        self, position: PositionEntry, current_price: float
    ) -> bool:
        """Check if take profit is triggered."""
        if self.stop_config.take_profit_pct is None:
            return False

        entry_price = position.entry_price
        tp_pct = self.stop_config.take_profit_pct

        if position.side == "buy":
            # Long: take profit if price rises above entry + tp%
            tp_price = entry_price * (1 + tp_pct)
            return current_price >= tp_price
        else:
            # Short: take profit if price drops below entry - tp%
            tp_price = entry_price * (1 - tp_pct)
            return current_price <= tp_price

    def _check_time_stop(
        self, position: PositionEntry, current_time: datetime
    ) -> bool:
        """Check if time-based stop is triggered."""
        if self.stop_config.max_hold_time_seconds is None:
            return False

        hold_seconds = (current_time - position.entry_time).total_seconds()
        return hold_seconds >= self.stop_config.max_hold_time_seconds

    def _check_trailing_stop(
        self, position: PositionEntry, current_price: float
    ) -> bool:
        """Check if trailing stop is triggered."""
        if self.stop_config.trailing_stop_pct is None:
            return False

        trail_pct = self.stop_config.trailing_stop_pct

        if position.side == "buy":
            # Long: trail from highest price
            if position.highest_price is not None:
                trail_price = position.highest_price * (1 - trail_pct)
                return current_price <= trail_price
        else:
            # Short: trail from lowest price
            if position.lowest_price is not None:
                trail_price = position.lowest_price * (1 + trail_pct)
                return current_price >= trail_price

        return False

    def get_exit_reason(self, symbol: str) -> str | None:
        """Get the exit reason for a symbol."""
        return self._exit_reasons.get(symbol)

    def calculate_unrealized_pnl(
        self,
        symbol: str,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized P&L for a position.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.

        Returns:
            Unrealized P&L in notional terms.
        """
        position = self._positions.get(symbol)
        if not position:
            return 0.0

        entry_price = position.entry_price
        notional = position.notional

        # Calculate return
        price_return = (current_price - entry_price) / entry_price

        if position.side == "buy":
            return price_return * notional
        else:
            # Short: profit when price goes down
            return -price_return * notional

    def check_kill_switch(self, state: RiskState) -> bool:
        """
        Check if kill switch should be triggered.

        Args:
            state: Current risk state.

        Returns:
            True if trading should halt immediately.
        """
        return bool(self.risk_checker.is_kill_switch_triggered(state))


__all__ = ["RiskManager", "StopConfig", "PositionEntry"]
