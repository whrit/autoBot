"""
Risk Checker Module.

Pre-trade risk validation with position sizing for the backtester.

Classes:
    RiskSeverity: Enumeration of violation severity levels.
    RiskViolation: A risk rule violation.
    RiskChecker: Pre-trade risk validation.
    PositionSizer: Calculate optimal position size based on risk parameters.
    RiskManager: Combines RiskChecker and PositionSizer for comprehensive risk management.

Example usage:
    >>> from backtester_py.models.risk_checker import RiskChecker, PositionSizer, RiskManager
    >>> checker = RiskChecker(max_position_pct=0.10, max_drawdown_pct=0.20)
    >>> sizer = PositionSizer(risk_per_trade_pct=0.02)
    >>> manager = RiskManager(checker, sizer)
    >>> size, violations = manager.get_sized_order("AAPL", "buy", Decimal("100"), ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal


class RiskSeverity(Enum):
    """Severity levels for risk violations.

    Attributes:
        WARNING: Advisory violation that allows the trade but recommends caution.
        BLOCK: Critical violation that should prevent the trade.
    """

    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True)
class RiskViolation:
    """A risk rule violation.

    Immutable dataclass representing a violation of a risk rule.

    Attributes:
        rule: The name of the rule that was violated.
        message: Human-readable description of the violation.
        severity: The severity level of the violation.
    """

    rule: str
    message: str
    severity: RiskSeverity


class RiskChecker:
    """Pre-trade risk validation.

    Validates orders against configurable risk limits including:
    - Maximum position size as percentage of portfolio
    - Maximum order value
    - Maximum daily trades
    - Maximum drawdown percentage

    Attributes:
        max_position_pct: Maximum position size as fraction of portfolio (0.10 = 10%).
        max_order_value: Maximum absolute order value in notional terms.
        max_daily_trades: Maximum number of trades allowed per day.
        max_drawdown_pct: Maximum drawdown before blocking trades (0.20 = 20%).
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_order_value: Decimal = Decimal("100000"),
        max_daily_trades: int = 100,
        max_drawdown_pct: float = 0.20,
    ) -> None:
        """Initialize the risk checker.

        Args:
            max_position_pct: Maximum position size as fraction of portfolio.
                Defaults to 0.10 (10%).
            max_order_value: Maximum absolute order value.
                Defaults to $100,000.
            max_daily_trades: Maximum number of trades per day.
                Defaults to 100.
            max_drawdown_pct: Maximum drawdown before blocking.
                Defaults to 0.20 (20%).
        """
        self.max_position_pct = max_position_pct
        self.max_order_value = max_order_value
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_pct = max_drawdown_pct
        self._daily_trade_count: int = 0

    def check_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        notional: Decimal,
        portfolio_value: Decimal,
        current_drawdown: float,
    ) -> list[RiskViolation]:
        """Validate order against risk rules.

        Checks the order against all configured risk limits and returns
        a list of any violations found.

        Args:
            symbol: Trading symbol (e.g., "AAPL").
            side: Order side ("buy" or "sell").
            notional: Order size in notional (dollar) terms.
            portfolio_value: Current total portfolio value.
            current_drawdown: Current drawdown as a fraction (0.10 = 10% drawdown).

        Returns:
            List of RiskViolation objects. Empty list if order passes all checks.
        """
        violations: list[RiskViolation] = []

        # Check for zero or negative portfolio value
        if portfolio_value <= Decimal("0"):
            if portfolio_value == Decimal("0"):
                violations.append(
                    RiskViolation(
                        rule="zero_portfolio",
                        message="Cannot trade with zero portfolio value",
                        severity=RiskSeverity.BLOCK,
                    )
                )
            else:
                violations.append(
                    RiskViolation(
                        rule="invalid_portfolio",
                        message=f"Invalid negative portfolio value: {portfolio_value}",
                        severity=RiskSeverity.BLOCK,
                    )
                )
            return violations

        # Zero notional is a no-op, always passes
        if notional == Decimal("0"):
            return violations

        # Check position size percentage
        position_pct = float(notional / portfolio_value)
        if position_pct > self.max_position_pct:
            violations.append(
                RiskViolation(
                    rule="max_position_pct",
                    message=(
                        f"Position size {position_pct:.2%} exceeds limit of "
                        f"{self.max_position_pct:.2%} for {symbol}"
                    ),
                    severity=RiskSeverity.BLOCK,
                )
            )

        # Check max order value
        if notional > self.max_order_value:
            violations.append(
                RiskViolation(
                    rule="max_order_value",
                    message=(
                        f"Order value ${notional:,.2f} exceeds limit of "
                        f"${self.max_order_value:,.2f}"
                    ),
                    severity=RiskSeverity.BLOCK,
                )
            )

        # Check daily trade limit
        if self._daily_trade_count >= self.max_daily_trades:
            violations.append(
                RiskViolation(
                    rule="max_daily_trades",
                    message=(
                        f"Daily trade count {self._daily_trade_count} has reached "
                        f"limit of {self.max_daily_trades}"
                    ),
                    severity=RiskSeverity.BLOCK,
                )
            )

        # Check drawdown limits
        if current_drawdown > self.max_drawdown_pct:
            violations.append(
                RiskViolation(
                    rule="max_drawdown",
                    message=(
                        f"Current drawdown {current_drawdown:.2%} exceeds limit of "
                        f"{self.max_drawdown_pct:.2%}"
                    ),
                    severity=RiskSeverity.BLOCK,
                )
            )
        elif current_drawdown >= self.max_drawdown_pct * 0.80:
            # Warning when approaching limit (80%+ of max)
            violations.append(
                RiskViolation(
                    rule="drawdown_warning",
                    message=(
                        f"Current drawdown {current_drawdown:.2%} is approaching "
                        f"limit of {self.max_drawdown_pct:.2%}"
                    ),
                    severity=RiskSeverity.WARNING,
                )
            )

        return violations

    def record_trade(self) -> None:
        """Record a completed trade.

        Increments the daily trade counter. Call this after each trade execution.
        """
        self._daily_trade_count += 1

    def reset_daily(self) -> None:
        """Reset daily counters.

        Should be called at the start of each trading day to reset:
        - Daily trade count
        """
        self._daily_trade_count = 0


class PositionSizer:
    """Calculate optimal position size based on risk parameters.

    Implements fixed-fractional position sizing based on:
    - Account risk per trade (e.g., 2% of account)
    - Stop loss distance
    - Maximum position size cap

    Attributes:
        risk_per_trade_pct: Fraction of account to risk per trade (0.02 = 2%).
        max_position_pct: Maximum position size as fraction of account (0.10 = 10%).
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 0.02,
        max_position_pct: float = 0.10,
    ) -> None:
        """Initialize the position sizer.

        Args:
            risk_per_trade_pct: Fraction of account to risk per trade.
                Defaults to 0.02 (2%).
            max_position_pct: Maximum position size as fraction of account.
                Defaults to 0.10 (10%).
        """
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct

    def calculate_size(
        self,
        account_balance: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
    ) -> Decimal:
        """Calculate position size based on stop loss distance.

        Uses the formula:
            risk_amount = account_balance * risk_per_trade_pct
            stop_distance = abs(entry_price - stop_loss_price)
            shares = risk_amount / stop_distance
            notional = shares * entry_price

        The result is capped at max_position_pct of account.

        Args:
            account_balance: Current account equity value.
            entry_price: Intended entry price.
            stop_loss_price: Stop loss price level.

        Returns:
            Position size in notional (dollar) terms.
            Returns Decimal("0") for invalid inputs.
        """
        # Handle invalid inputs
        if account_balance <= Decimal("0"):
            return Decimal("0")

        if entry_price <= Decimal("0") or stop_loss_price <= Decimal("0"):
            return Decimal("0")

        # Calculate stop distance (absolute value for both long and short)
        stop_distance = abs(entry_price - stop_loss_price)

        # Same price means undefined risk - return 0
        if stop_distance == Decimal("0"):
            return Decimal("0")

        # Calculate risk amount (dollars we're willing to lose)
        risk_amount = account_balance * Decimal(str(self.risk_per_trade_pct))

        # Calculate number of shares based on risk
        # shares = risk_amount / stop_distance_per_share
        shares = risk_amount / stop_distance

        # Calculate notional value
        notional = shares * entry_price

        # Cap at maximum position size
        max_notional = account_balance * Decimal(str(self.max_position_pct))
        if notional > max_notional:
            notional = max_notional

        # Round to 2 decimal places for currency
        return notional.quantize(Decimal("0.01"))


class RiskManager:
    """Combines RiskChecker and PositionSizer for comprehensive risk management.

    Provides a unified interface for:
    1. Calculating optimal position sizes
    2. Validating orders against risk rules
    3. Managing daily trade limits

    Attributes:
        checker: The RiskChecker instance for validation.
        sizer: The PositionSizer instance for position sizing.
    """

    def __init__(self, checker: RiskChecker, sizer: PositionSizer) -> None:
        """Initialize the risk manager.

        Args:
            checker: RiskChecker instance for order validation.
            sizer: PositionSizer instance for position sizing.
        """
        self.checker = checker
        self.sizer = sizer

    def get_sized_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        entry_price: Decimal,
        stop_loss_price: Decimal,
        account_balance: Decimal,
        current_drawdown: float,
    ) -> tuple[Decimal, list[RiskViolation]]:
        """Calculate position size and validate against risk rules.

        First calculates the optimal position size using the sizer,
        then validates the resulting order against risk rules.

        Args:
            symbol: Trading symbol (e.g., "AAPL").
            side: Order side ("buy" or "sell").
            entry_price: Intended entry price.
            stop_loss_price: Stop loss price level.
            account_balance: Current account equity value.
            current_drawdown: Current drawdown as a fraction.

        Returns:
            Tuple of (position_size, violations).
            position_size: Calculated notional value (may be 0 if blocked).
            violations: List of any risk violations found.
        """
        # Calculate position size
        size = self.sizer.calculate_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )

        # If size is zero, return with no violations (it's a no-op)
        if size == Decimal("0"):
            return Decimal("0"), []

        # Validate against risk rules
        violations = self.checker.check_order(
            symbol=symbol,
            side=side,
            notional=size,
            portfolio_value=account_balance,
            current_drawdown=current_drawdown,
        )

        return size, violations

    def record_trade(self) -> None:
        """Record a completed trade.

        Delegates to the checker's record_trade method.
        """
        self.checker.record_trade()

    def reset_daily(self) -> None:
        """Reset daily counters.

        Delegates to the checker's reset_daily method.
        """
        self.checker.reset_daily()


def create_risk_checker(
    max_position_pct: float = 0.10,
    max_order_value: float = 100000.0,
    max_daily_trades: int = 100,
    max_drawdown_pct: float = 0.20,
) -> RiskChecker:
    """Factory function to create a RiskChecker.

    Args:
        max_position_pct: Maximum position size as fraction of portfolio.
        max_order_value: Maximum absolute order value.
        max_daily_trades: Maximum number of trades per day.
        max_drawdown_pct: Maximum drawdown before blocking.

    Returns:
        Configured RiskChecker instance.
    """
    return RiskChecker(
        max_position_pct=max_position_pct,
        max_order_value=Decimal(str(max_order_value)),
        max_daily_trades=max_daily_trades,
        max_drawdown_pct=max_drawdown_pct,
    )


# Backward compatibility alias
BacktesterRiskChecker = RiskChecker


__all__ = [
    "RiskSeverity",
    "RiskViolation",
    "RiskChecker",
    "PositionSizer",
    "RiskManager",
    "create_risk_checker",
    "BacktesterRiskChecker",
]
