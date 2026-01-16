"""Turnover Analysis.

Monitor portfolio turnover for cost management.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TurnoverConfig:
    """Configuration for turnover analysis.

    Attributes:
        daily_turnover_warning: Daily turnover above this triggers warning.
        daily_turnover_critical: Daily turnover above this triggers critical alert.
        weekly_turnover_warning: Weekly turnover above this triggers warning.
        cost_per_turnover_bps: Estimated cost per unit of turnover in basis points.
        window_days: Number of days to track trades.
    """

    daily_turnover_warning: float = 0.20
    daily_turnover_critical: float = 0.50
    weekly_turnover_warning: float = 1.0
    cost_per_turnover_bps: float = 5.0
    window_days: int = 30


@dataclass
class TradeRecord:
    """Record of a trade.

    Attributes:
        trade_id: Unique trade identifier.
        symbol: Trading symbol.
        side: Trade side (buy/sell).
        quantity: Trade quantity.
        price: Trade price.
        timestamp: Trade timestamp.
    """

    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime

    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade."""
        return self.quantity * self.price


@dataclass
class TurnoverStats:
    """Turnover statistics.

    Attributes:
        total_trades: Total number of trades tracked.
        total_volume: Total traded volume in notional value.
        daily_turnover: Today's turnover as decimal of portfolio.
        weekly_turnover: This week's turnover as decimal of portfolio.
        total_turnover: Total turnover in window as decimal of portfolio.
        estimated_cost: Estimated cost from turnover.
        alert_level: Current alert level.
    """

    total_trades: int
    total_volume: float
    daily_turnover: float
    weekly_turnover: float
    total_turnover: float
    estimated_cost: float
    alert_level: str


@dataclass
class TurnoverAnalyzer:
    """Analyze portfolio turnover for cost management.

    Monitors trading activity to detect excessive turnover
    and estimate associated costs.
    """

    config: TurnoverConfig
    portfolio_value: float
    _trade_history: list[TradeRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._trade_history = []

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
    ) -> TradeRecord:
        """Record a trade.

        Args:
            trade_id: Unique trade identifier.
            symbol: Trading symbol.
            side: Trade side (buy/sell).
            quantity: Trade quantity.
            price: Trade price.
            timestamp: Trade timestamp (defaults to now).

        Returns:
            TradeRecord with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )

        self._trade_history.append(record)

        logger.debug(
            "trade_recorded",
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            notional_value=record.notional_value,
        )

        return record

    def get_stats(self, symbol: str | None = None) -> TurnoverStats:
        """Get current turnover statistics.

        Args:
            symbol: Optional symbol to filter stats.

        Returns:
            TurnoverStats with current metrics.
        """
        now = datetime.now(UTC)
        window_start = now - timedelta(days=self.config.window_days)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)

        # Filter by symbol and window
        if symbol:
            trades = [
                t for t in self._trade_history
                if t.symbol == symbol and t.timestamp >= window_start
            ]
        else:
            trades = [t for t in self._trade_history if t.timestamp >= window_start]

        if not trades:
            return TurnoverStats(
                total_trades=0,
                total_volume=0.0,
                daily_turnover=0.0,
                weekly_turnover=0.0,
                total_turnover=0.0,
                estimated_cost=0.0,
                alert_level="none",
            )

        # Calculate volumes
        total_volume = sum(t.notional_value for t in trades)
        daily_volume = sum(t.notional_value for t in trades if t.timestamp >= today_start)
        weekly_volume = sum(t.notional_value for t in trades if t.timestamp >= week_start)

        # Calculate turnover as percentage of portfolio
        if self.portfolio_value > 0:
            daily_turnover = daily_volume / self.portfolio_value
            weekly_turnover = weekly_volume / self.portfolio_value
            total_turnover = total_volume / self.portfolio_value
        else:
            daily_turnover = 0.0
            weekly_turnover = 0.0
            total_turnover = 0.0

        # Estimate cost (turnover * cost_per_turnover_bps * portfolio_value / 10000)
        cost_bps = self.config.cost_per_turnover_bps
        estimated_cost = daily_turnover * cost_bps * self.portfolio_value / 10000

        # Determine alert level
        alert_level = self._determine_alert_level(daily_turnover)

        return TurnoverStats(
            total_trades=len(trades),
            total_volume=total_volume,
            daily_turnover=daily_turnover,
            weekly_turnover=weekly_turnover,
            total_turnover=total_turnover,
            estimated_cost=estimated_cost,
            alert_level=alert_level,
        )

    def should_alert(self) -> bool:
        """Check if turnover alert should be triggered.

        Returns:
            True if turnover exceeds warning threshold.
        """
        stats = self.get_stats()
        return stats.alert_level in ("warning", "critical")

    def update_portfolio_value(self, portfolio_value: float) -> None:
        """Update portfolio value for turnover calculations.

        Args:
            portfolio_value: New portfolio value.
        """
        self.portfolio_value = portfolio_value
        logger.info("portfolio_value_updated", portfolio_value=portfolio_value)

    def get_daily_breakdown(self, days: int = 7) -> list[dict[str, object]]:
        """Get daily turnover breakdown.

        Args:
            days: Number of days to include.

        Returns:
            List of daily turnover data.
        """
        now = datetime.now(UTC)
        breakdown = []

        for i in range(days):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_trades = [
                t for t in self._trade_history
                if day_start <= t.timestamp < day_end
            ]

            day_volume = sum(t.notional_value for t in day_trades)
            day_turnover = day_volume / self.portfolio_value if self.portfolio_value > 0 else 0.0

            breakdown.append({
                "date": day_start.date().isoformat(),
                "trades": len(day_trades),
                "volume": day_volume,
                "turnover": day_turnover,
            })

        return breakdown

    def reset(self) -> None:
        """Reset turnover statistics."""
        self._trade_history.clear()
        logger.info("turnover_analyzer_reset")

    def _determine_alert_level(self, daily_turnover: float) -> str:
        """Determine alert level based on daily turnover.

        Args:
            daily_turnover: Current daily turnover as decimal.

        Returns:
            Alert level string.
        """
        if daily_turnover >= self.config.daily_turnover_critical:
            return "critical"
        elif daily_turnover >= self.config.daily_turnover_warning:
            return "warning"
        return "none"
