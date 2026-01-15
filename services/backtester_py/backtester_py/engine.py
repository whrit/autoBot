"""
Backtester Core Engine (T3.01-T3.04).

Provides the core simulation loop with quote-based fills, slippage modeling,
and position/risk management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from cost_models import TransactionCostModel
    from risk_models import RiskChecker, RiskState


class SignalType(str, Enum):
    """Trading signal types."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """
    Trading signal at a decision point.

    Attributes:
        timestamp: Decision timestamp
        symbol: Trading symbol
        signal_type: Long, short, or flat
        strength: Signal strength/confidence (0 to 1)
        target_notional: Target position size in notional terms (optional)
    """

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float = 1.0
    target_notional: float | None = None


@dataclass
class Fill:
    """
    Represents an executed fill.

    Attributes:
        timestamp: Fill timestamp
        symbol: Trading symbol
        side: Buy or sell
        quantity: Number of shares
        price: Fill price (after slippage)
        notional: Total notional value
        slippage_bps: Slippage applied in basis points
        commission: Commission/fees paid
    """

    timestamp: datetime
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    notional: float
    slippage_bps: float
    commission: float


@dataclass
class Position:
    """
    Current position state.

    Attributes:
        symbol: Trading symbol
        quantity: Number of shares (positive = long, negative = short)
        avg_entry_price: Average entry price
        notional: Current notional value
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L from closed positions
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    notional: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """True if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """True if no position."""
        return self.quantity == 0

    def update_mtm(self, current_price: float) -> None:
        """Update mark-to-market values."""
        if self.quantity != 0:
            self.notional = abs(self.quantity * current_price)
            self.unrealized_pnl = self.quantity * (current_price - self.avg_entry_price)


@dataclass
class BacktestConfig:
    """
    Backtest configuration.

    Attributes:
        initial_capital: Starting capital in dollars
        cost_model: Transaction cost model for slippage/fees
        risk_checker: Risk checker for position limits
        default_order_notional: Default order size if not specified in signal
        stop_loss_pct: Stop-loss percentage (0.02 = 2%)
        take_profit_pct: Take-profit percentage (optional)
        book_notional_default: Default book notional for slippage calc
        short_term_vol_default: Default short-term volatility
    """

    initial_capital: float
    cost_model: TransactionCostModel
    risk_checker: RiskChecker
    default_order_notional: float = 10_000.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float | None = None
    book_notional_default: float = 500_000.0
    short_term_vol_default: float = 0.001


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        fills: List of all executed fills
        positions: Final positions
        equity_curve: Time series of equity values
        daily_returns: Daily returns series
        total_pnl: Total P&L
        total_trades: Number of trades executed
        risk_violations: List of risk violations encountered
    """

    fills: list[Fill] = field(default_factory=list)
    positions: dict[str, Position] = field(default_factory=dict)
    equity_curve: pl.DataFrame = field(default_factory=pl.DataFrame)
    daily_returns: pl.DataFrame = field(default_factory=pl.DataFrame)
    total_pnl: float = 0.0
    total_trades: int = 0
    risk_violations: list[str] = field(default_factory=list)


class BacktestEngine:
    """
    Core backtesting engine.

    Simulates trading with realistic execution assumptions:
    - Buys at ask price + slippage
    - Sells at bid price - slippage
    - Respects position limits via RiskChecker
    - Applies stop-loss/take-profit logic
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self._positions: dict[str, Position] = {}
        self._cash: float = config.initial_capital
        self._fills: list[Fill] = []
        self._equity_history: list[dict[str, datetime | float]] = []
        self._risk_violations: list[str] = []

    def run(
        self,
        market_data: pl.DataFrame,
        signals: list[Signal],
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            market_data: DataFrame with columns:
                - timestamp: Datetime
                - symbol: str
                - bid_price: float
                - ask_price: float
                - bid_size: float (optional)
                - ask_size: float (optional)
                - vol: float (optional, short-term volatility)
            signals: List of trading signals to execute

        Returns:
            BacktestResult with fills, positions, and metrics
        """
        # Reset state
        self._positions = {}
        self._cash = self.config.initial_capital
        self._fills = []
        self._equity_history = []
        self._risk_violations = []

        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        # Process each signal
        for signal in sorted_signals:
            self._process_signal(signal, market_data)

        # Calculate final results
        return self._build_result()

    def _process_signal(
        self,
        signal: Signal,
        market_data: pl.DataFrame,
    ) -> None:
        """Process a single trading signal."""
        # Get market data at signal time
        quote = self._get_quote_at_time(market_data, signal.symbol, signal.timestamp)
        if quote is None:
            return

        bid_price = quote["bid_price"]
        ask_price = quote["ask_price"]
        bid_size = quote.get("bid_size", self.config.book_notional_default / bid_price)
        ask_size = quote.get("ask_size", self.config.book_notional_default / ask_price)
        vol = quote.get("vol", self.config.short_term_vol_default)

        # Get current position
        position = self._positions.get(signal.symbol, Position(symbol=signal.symbol))

        # Check stop-loss first
        if not position.is_flat:
            self._check_stop_loss(position, bid_price, ask_price, signal.timestamp)

        # Determine target position based on signal
        target_notional = signal.target_notional or self.config.default_order_notional

        if signal.signal_type == SignalType.LONG:
            self._execute_long(
                signal, position, target_notional, bid_price, ask_price, bid_size, ask_size, vol
            )
        elif signal.signal_type == SignalType.SHORT:
            self._execute_short(
                signal, position, target_notional, bid_price, ask_price, bid_size, ask_size, vol
            )
        elif signal.signal_type == SignalType.FLAT:
            self._execute_flat(signal, position, bid_price, ask_price, bid_size, ask_size, vol)

        # Update position in dictionary
        self._positions[signal.symbol] = position

        # Record equity
        self._record_equity(signal.timestamp, bid_price, ask_price)

    def _get_quote_at_time(
        self,
        market_data: pl.DataFrame,
        symbol: str,
        timestamp: datetime,
    ) -> dict[str, float] | None:
        """Get the most recent quote at or before the given timestamp."""
        filtered = market_data.filter(
            (pl.col("symbol") == symbol) & (pl.col("timestamp") <= timestamp)
        ).sort("timestamp", descending=True)

        if filtered.is_empty():
            return None

        row = filtered.row(0, named=True)
        return {
            "bid_price": row["bid_price"],
            "ask_price": row["ask_price"],
            "bid_size": row.get("bid_size", self.config.book_notional_default / row["bid_price"]),
            "ask_size": row.get("ask_size", self.config.book_notional_default / row["ask_price"]),
            "vol": row.get("vol", self.config.short_term_vol_default),
        }

    def _check_stop_loss(
        self,
        position: Position,
        bid_price: float,
        ask_price: float,
        timestamp: datetime,
    ) -> None:
        """Check and execute stop-loss if triggered."""
        if position.is_flat:
            return

        current_price = (bid_price + ask_price) / 2
        position.update_mtm(current_price)

        # Calculate P&L percentage
        if position.avg_entry_price > 0:
            pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
            if position.is_short:
                pnl_pct = -pnl_pct

            # Check stop-loss
            if pnl_pct < -self.config.stop_loss_pct:
                # Execute stop-loss
                self._close_position(
                    position,
                    bid_price,
                    ask_price,
                    timestamp,
                    reason="stop_loss",
                )

    def _execute_long(
        self,
        signal: Signal,
        position: Position,
        target_notional: float,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        vol: float,
    ) -> None:
        """Execute a long signal."""
        # If short, close first
        if position.is_short:
            self._close_position(position, bid_price, ask_price, signal.timestamp, reason="flip")

        # Open/add to long position
        if position.quantity < 0 or position.is_flat:
            # Calculate order notional
            order_notional = target_notional * signal.strength

            # Check risk limits

            risk_state = self._build_risk_state()
            violations = self.config.risk_checker.check_order(
                signal.symbol, "buy", order_notional, risk_state
            )

            if violations:
                for v in violations:
                    self._risk_violations.append(v.message)
                return

            # Calculate fill price with slippage
            book_notional = ask_size * ask_price
            fill_price = self.config.cost_model.calculate_fill_price(
                side="buy",
                ask_price=ask_price,
                bid_price=bid_price,
                order_notional=order_notional,
                book_notional=book_notional,
                short_term_vol=vol,
            )

            # Execute fill
            quantity = order_notional / fill_price
            commission = self.config.cost_model.fixed_cost_bps * order_notional / 10000

            fill = Fill(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                side="buy",
                quantity=quantity,
                price=fill_price,
                notional=order_notional,
                slippage_bps=self._calculate_slippage_bps(fill_price, ask_price),
                commission=commission,
            )
            self._fills.append(fill)

            # Update position
            self._update_position_on_buy(position, quantity, fill_price)

            # Update cash
            self._cash -= order_notional + commission

    def _execute_short(
        self,
        signal: Signal,
        position: Position,
        target_notional: float,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        vol: float,
    ) -> None:
        """Execute a short signal."""
        # If long, close first
        if position.is_long:
            self._close_position(position, bid_price, ask_price, signal.timestamp, reason="flip")

        # Open/add to short position
        if position.quantity > 0 or position.is_flat:
            # Calculate order notional
            order_notional = target_notional * signal.strength

            # Check risk limits

            risk_state = self._build_risk_state()
            violations = self.config.risk_checker.check_order(
                signal.symbol, "sell", order_notional, risk_state
            )

            if violations:
                for v in violations:
                    self._risk_violations.append(v.message)
                return

            # Calculate fill price with slippage
            book_notional = bid_size * bid_price
            fill_price = self.config.cost_model.calculate_fill_price(
                side="sell",
                ask_price=ask_price,
                bid_price=bid_price,
                order_notional=order_notional,
                book_notional=book_notional,
                short_term_vol=vol,
            )

            # Execute fill
            quantity = order_notional / fill_price
            commission = self.config.cost_model.fixed_cost_bps * order_notional / 10000

            fill = Fill(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                side="sell",
                quantity=quantity,
                price=fill_price,
                notional=order_notional,
                slippage_bps=self._calculate_slippage_bps(bid_price, fill_price),
                commission=commission,
            )
            self._fills.append(fill)

            # Update position (negative for short)
            self._update_position_on_sell(position, quantity, fill_price)

            # Update cash (receive proceeds for short)
            self._cash += order_notional - commission

    def _execute_flat(
        self,
        signal: Signal,
        position: Position,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        vol: float,
    ) -> None:
        """Execute a flat signal (close position)."""
        if not position.is_flat:
            self._close_position(position, bid_price, ask_price, signal.timestamp, reason="signal")

    def _close_position(
        self,
        position: Position,
        bid_price: float,
        ask_price: float,
        timestamp: datetime,
        reason: str = "",
    ) -> None:
        """Close an existing position."""
        if position.is_flat:
            return

        order_notional = abs(position.quantity * position.avg_entry_price)

        if position.is_long:
            # Sell to close long
            book_notional = self.config.book_notional_default
            fill_price = self.config.cost_model.calculate_fill_price(
                side="sell",
                ask_price=ask_price,
                bid_price=bid_price,
                order_notional=order_notional,
                book_notional=book_notional,
                short_term_vol=self.config.short_term_vol_default,
            )

            commission = self.config.cost_model.fixed_cost_bps * order_notional / 10000
            realized_pnl = position.quantity * (fill_price - position.avg_entry_price) - commission

            fill = Fill(
                timestamp=timestamp,
                symbol=position.symbol,
                side="sell",
                quantity=position.quantity,
                price=fill_price,
                notional=order_notional,
                slippage_bps=self._calculate_slippage_bps(bid_price, fill_price),
                commission=commission,
            )
            self._fills.append(fill)

            # Update cash
            self._cash += position.quantity * fill_price - commission

            # Update position
            position.realized_pnl += realized_pnl
            position.quantity = 0
            position.avg_entry_price = 0
            position.notional = 0
            position.unrealized_pnl = 0

        else:
            # Buy to close short
            book_notional = self.config.book_notional_default
            fill_price = self.config.cost_model.calculate_fill_price(
                side="buy",
                ask_price=ask_price,
                bid_price=bid_price,
                order_notional=order_notional,
                book_notional=book_notional,
                short_term_vol=self.config.short_term_vol_default,
            )

            commission = self.config.cost_model.fixed_cost_bps * order_notional / 10000
            # Short P&L: entry - exit (inverted)
            realized_pnl = (
                abs(position.quantity) * (position.avg_entry_price - fill_price) - commission
            )

            fill = Fill(
                timestamp=timestamp,
                symbol=position.symbol,
                side="buy",
                quantity=abs(position.quantity),
                price=fill_price,
                notional=order_notional,
                slippage_bps=self._calculate_slippage_bps(fill_price, ask_price),
                commission=commission,
            )
            self._fills.append(fill)

            # Update cash
            self._cash -= abs(position.quantity) * fill_price + commission

            # Update position
            position.realized_pnl += realized_pnl
            position.quantity = 0
            position.avg_entry_price = 0
            position.notional = 0
            position.unrealized_pnl = 0

    def _update_position_on_buy(
        self, position: Position, quantity: float, price: float
    ) -> None:
        """Update position after a buy."""
        if position.quantity >= 0:
            # Adding to long or new long
            total_cost = position.quantity * position.avg_entry_price + quantity * price
            position.quantity += quantity
            if position.quantity > 0:
                position.avg_entry_price = total_cost / position.quantity
        else:
            # Covering short
            position.quantity += quantity
            if position.quantity > 0:
                position.avg_entry_price = price

    def _update_position_on_sell(
        self, position: Position, quantity: float, price: float
    ) -> None:
        """Update position after a sell."""
        if position.quantity <= 0:
            # Adding to short or new short
            total_cost = abs(position.quantity) * position.avg_entry_price + quantity * price
            position.quantity -= quantity
            if position.quantity < 0:
                position.avg_entry_price = total_cost / abs(position.quantity)
        else:
            # Reducing long
            position.quantity -= quantity
            if position.quantity < 0:
                position.avg_entry_price = price

    def _calculate_slippage_bps(self, fill_price: float, reference_price: float) -> float:
        """Calculate slippage in basis points."""
        if reference_price == 0:
            return 0.0
        return abs(fill_price - reference_price) / reference_price * 10000

    def _build_risk_state(self) -> RiskState:
        """Build current risk state from positions."""
        from risk_models import RiskState

        positions_dict: dict[str, float] = {}
        for symbol, pos in self._positions.items():
            if pos.quantity != 0:
                # Calculate notional from quantity and entry price
                # This is more reliable than pos.notional which may not be updated
                notional = abs(pos.quantity * pos.avg_entry_price)
                positions_dict[symbol] = notional if pos.is_long else -notional

        total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self._positions.values())
        equity = self.config.initial_capital + total_pnl

        return RiskState(
            positions=positions_dict,
            daily_pnl=total_pnl,
            peak_equity=max(self.config.initial_capital, equity),
            current_equity=equity,
        )

    def _record_equity(
        self, timestamp: datetime, bid_price: float, ask_price: float
    ) -> None:
        """Record current equity value."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_realized = sum(p.realized_pnl for p in self._positions.values())
        equity = self._cash + total_unrealized + total_realized

        # Adjust for any open position values
        for pos in self._positions.values():
            if not pos.is_flat:
                equity += pos.notional

        self._equity_history.append({"timestamp": timestamp, "equity": equity})

    def _build_result(self) -> BacktestResult:
        """Build the final backtest result."""
        # Build equity curve DataFrame
        if self._equity_history:
            equity_df = pl.DataFrame(self._equity_history)
        else:
            equity_df = pl.DataFrame({"timestamp": [], "equity": []})

        # Calculate daily returns
        if len(equity_df) > 1:
            daily_returns = equity_df.with_columns(
                [
                    (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("return"),
                ]
            ).drop_nulls()
        else:
            daily_returns = pl.DataFrame({"timestamp": [], "equity": [], "return": []})

        # Calculate total P&L
        total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self._positions.values())

        return BacktestResult(
            fills=self._fills,
            positions=self._positions,
            equity_curve=equity_df,
            daily_returns=daily_returns,
            total_pnl=total_pnl,
            total_trades=len(self._fills),
            risk_violations=self._risk_violations,
        )
