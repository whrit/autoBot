"""Turnover Analysis.

Monitor portfolio turnover for cost management.
Provides trade logging with notional values, daily turnover summaries,
and threshold breach alerts for dashboard display.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class TurnoverAlertSeverity(str, Enum):
    """Alert severity levels for turnover monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TurnoverAlert:
    """Alert generated from turnover monitoring.

    Attributes:
        timestamp: When the alert was generated.
        severity: Alert severity level.
        message: Human-readable alert message.
        daily_turnover: Daily turnover as percentage at time of alert.
        estimated_cost: Estimated cost from turnover.
        details: Additional context details.
    """

    timestamp: datetime
    severity: TurnoverAlertSeverity
    message: str
    daily_turnover: float
    estimated_cost: float
    details: dict[str, object] | None = None


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
        target_daily_turnover: Target daily turnover threshold.
    """

    total_trades: int
    total_volume: float
    daily_turnover: float
    weekly_turnover: float
    total_turnover: float
    estimated_cost: float
    alert_level: str
    target_daily_turnover: float = 0.20

    @property
    def daily_turnover_pct(self) -> str:
        """Return daily turnover as formatted percentage string."""
        return f"{self.daily_turnover * 100:.1f}%"

    @property
    def target_pct(self) -> str:
        """Return target daily turnover as formatted percentage string."""
        return f"<{self.target_daily_turnover * 100:.0f}%"

    @property
    def status_emoji(self) -> str:
        """Return status emoji based on alert level."""
        return {
            "none": "[green]OK[/green]",
            "warning": "[yellow]WARN[/yellow]",
            "critical": "[red]CRIT[/red]",
        }.get(self.alert_level, "[green]OK[/green]")


@dataclass
class TurnoverAnalyzer:
    """Analyze portfolio turnover for cost management.

    Monitors trading activity to detect excessive turnover
    and estimate associated costs. Provides structured logging
    and alert history for dashboard display.
    """

    config: TurnoverConfig
    portfolio_value: float
    _trade_history: list[TradeRecord] = field(default_factory=list, init=False)
    _alert_history: deque[TurnoverAlert] = field(default_factory=deque, init=False)
    _stats_callbacks: list[Callable[[TurnoverStats], None]] = field(
        default_factory=list, init=False
    )
    _last_alert_level: str = field(default="none", init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._trade_history = []
        self._alert_history = deque(maxlen=100)  # Keep last 100 alerts
        self._stats_callbacks = []
        self._last_alert_level = "none"

    def register_stats_callback(
        self, callback: Callable[[TurnoverStats], None]
    ) -> None:
        """Register a callback to receive stats updates.

        Args:
            callback: Function to call with updated stats.
        """
        self._stats_callbacks.append(callback)

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
    ) -> TradeRecord:
        """Record a trade with structured logging.

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

        # Structured logging for trade with notional value
        turnover_pct = (
            record.notional_value / self.portfolio_value * 100
            if self.portfolio_value > 0
            else 0.0
        )

        logger.info(
            "trade_recorded",
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            notional_value=round(record.notional_value, 2),
            turnover_pct=round(turnover_pct, 4),
            portfolio_value=self.portfolio_value,
            timestamp=timestamp.isoformat(),
        )

        # Emit stats update and check for alerts
        self._emit_stats_and_check_alerts()

        return record

    def _emit_stats_and_check_alerts(self) -> None:
        """Emit stats to callbacks and check for alert level changes."""
        stats = self.get_stats()

        # Notify callbacks
        for callback in self._stats_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error("stats_callback_error", error=str(e))

        # Check for alert level change
        if stats.alert_level != self._last_alert_level:
            self._generate_alert(stats)
            self._last_alert_level = stats.alert_level

    def _generate_alert(self, stats: TurnoverStats) -> None:
        """Generate and log an alert based on current stats.

        Args:
            stats: Current turnover statistics.
        """
        now = datetime.now(UTC)

        if stats.alert_level == "critical":
            severity = TurnoverAlertSeverity.CRITICAL
            message = f"Daily turnover critical: {stats.daily_turnover_pct} (threshold: {self.config.daily_turnover_critical * 100:.0f}%)"
        elif stats.alert_level == "warning":
            severity = TurnoverAlertSeverity.WARNING
            message = f"Daily turnover approaching limit: {stats.daily_turnover_pct} (threshold: {self.config.daily_turnover_warning * 100:.0f}%)"
        else:
            severity = TurnoverAlertSeverity.INFO
            message = f"Daily turnover normalized: {stats.daily_turnover_pct}"

        alert = TurnoverAlert(
            timestamp=now,
            severity=severity,
            message=message,
            daily_turnover=stats.daily_turnover,
            estimated_cost=stats.estimated_cost,
            details={
                "total_trades": stats.total_trades,
                "total_volume": stats.total_volume,
                "weekly_turnover": stats.weekly_turnover,
            },
        )

        self._alert_history.append(alert)

        # Log with appropriate level
        log_func = {
            TurnoverAlertSeverity.CRITICAL: logger.error,
            TurnoverAlertSeverity.WARNING: logger.warning,
            TurnoverAlertSeverity.INFO: logger.info,
        }[severity]

        log_func(
            "turnover_alert",
            severity=severity.value,
            message=message,
            daily_turnover=stats.daily_turnover,
            daily_turnover_pct=stats.daily_turnover_pct,
            estimated_cost=round(stats.estimated_cost, 2),
            total_trades=stats.total_trades,
        )

    def get_alert_history(self, limit: int = 10) -> list[TurnoverAlert]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent TurnoverAlert objects.
        """
        return list(self._alert_history)[-limit:]

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
                target_daily_turnover=self.config.daily_turnover_warning,
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
            target_daily_turnover=self.config.daily_turnover_warning,
        )

    def emit_daily_summary(self) -> TurnoverStats:
        """Emit a daily turnover summary with structured logging.

        Returns:
            Current TurnoverStats.
        """
        stats = self.get_stats()
        breakdown = self.get_daily_breakdown(days=7)

        logger.info(
            "turnover_daily_summary",
            total_trades=stats.total_trades,
            total_volume=round(stats.total_volume, 2),
            daily_turnover=round(stats.daily_turnover, 4),
            daily_turnover_pct=stats.daily_turnover_pct,
            weekly_turnover=round(stats.weekly_turnover, 4),
            estimated_cost=round(stats.estimated_cost, 2),
            alert_level=stats.alert_level,
            portfolio_value=self.portfolio_value,
            breakdown=breakdown,
        )

        return stats

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
