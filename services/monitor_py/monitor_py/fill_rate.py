"""Fill Rate Monitoring.

Track order fill rates to detect execution quality issues.
Provides structured logging for fill events, alerts with severity levels,
and statistics emission for real-time dashboard display.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels for fill rate monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FillRateConfig:
    """Configuration for fill rate tracking.

    Attributes:
        expected_fill_rate: Expected fill rate as decimal (0.95 = 95%).
        warning_threshold: Fill rate below this triggers warning.
        critical_threshold: Fill rate below this triggers critical alert.
        window_size: Number of orders to track in rolling window.
        latency_warning_ms: Fill latency above this triggers warning.
    """

    expected_fill_rate: float = 0.95
    warning_threshold: float = 0.90
    critical_threshold: float = 0.80
    window_size: int = 100
    latency_warning_ms: float = 500.0


@dataclass
class FillEvent:
    """Record of a fill event.

    Attributes:
        order_id: Unique order identifier.
        symbol: Trading symbol.
        requested_qty: Requested quantity.
        filled_qty: Actual filled quantity.
        fill_time_ms: Time to fill in milliseconds.
        timestamp: Event timestamp.
    """

    order_id: str
    symbol: str
    requested_qty: float
    filled_qty: float
    fill_time_ms: float
    timestamp: datetime

    @property
    def fill_ratio(self) -> float:
        """Calculate fill ratio (0.0 to 1.0)."""
        if self.requested_qty <= 0:
            return 0.0
        return self.filled_qty / self.requested_qty

    @property
    def is_full_fill(self) -> bool:
        """Check if order was fully filled."""
        return self.fill_ratio >= 1.0

    @property
    def is_partial_fill(self) -> bool:
        """Check if order was partially filled."""
        return 0.0 < self.fill_ratio < 1.0


@dataclass
class FillAlert:
    """Alert generated from fill rate monitoring.

    Attributes:
        timestamp: When the alert was generated.
        severity: Alert severity level.
        message: Human-readable alert message.
        fill_rate: Current fill rate at time of alert.
        symbol: Optional symbol filter (None = all symbols).
        details: Additional context details.
    """

    timestamp: datetime
    severity: AlertSeverity
    message: str
    fill_rate: float
    symbol: str | None = None
    details: dict[str, object] | None = None


@dataclass
class FillRateStats:
    """Fill rate statistics.

    Attributes:
        total_orders: Total number of orders tracked.
        full_fills: Number of fully filled orders.
        partial_fills: Number of partially filled orders.
        unfilled: Number of unfilled orders.
        fill_rate: Average fill rate as decimal.
        avg_latency_ms: Average fill latency in milliseconds.
        max_latency_ms: Maximum fill latency in milliseconds.
        alert_level: Current alert level.
        latency_warning: Whether latency warning is triggered.
        target_fill_rate: Target fill rate for comparison.
    """

    total_orders: int
    full_fills: int
    partial_fills: int
    unfilled: int
    fill_rate: float
    avg_latency_ms: float
    max_latency_ms: float
    alert_level: str
    latency_warning: bool
    target_fill_rate: float = 0.95

    @property
    def fill_rate_pct(self) -> str:
        """Return fill rate as formatted percentage string."""
        return f"{self.fill_rate * 100:.1f}%"

    @property
    def target_pct(self) -> str:
        """Return target fill rate as formatted percentage string."""
        return f"{self.target_fill_rate * 100:.0f}%"

    @property
    def status_emoji(self) -> str:
        """Return status emoji based on alert level."""
        return {
            "none": "[green]OK[/green]",
            "warning": "[yellow]WARN[/yellow]",
            "critical": "[red]CRIT[/red]",
        }.get(self.alert_level, "[green]OK[/green]")


@dataclass
class FillRateTracker:
    """Track order fill rates and detect execution quality issues.

    Monitors fill rates, partial fills, and fill latency to detect
    execution quality degradation. Provides structured logging and
    alert history for dashboard display.
    """

    config: FillRateConfig
    _fill_history: deque[FillEvent] = field(default_factory=deque, init=False)
    _alert_history: deque[FillAlert] = field(default_factory=deque, init=False)
    _stats_callbacks: list[Callable[[FillRateStats], None]] = field(
        default_factory=list, init=False
    )
    _last_alert_level: str = field(default="none", init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._fill_history = deque(maxlen=self.config.window_size)
        self._alert_history = deque(maxlen=100)  # Keep last 100 alerts
        self._stats_callbacks = []
        self._last_alert_level = "none"

    def register_stats_callback(
        self, callback: Callable[[FillRateStats], None]
    ) -> None:
        """Register a callback to receive stats updates.

        Args:
            callback: Function to call with updated stats.
        """
        self._stats_callbacks.append(callback)

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        requested_qty: float,
        filled_qty: float,
        fill_time_ms: float,
        timestamp: datetime | None = None,
    ) -> FillEvent:
        """Record a fill event with structured logging.

        Args:
            order_id: Unique order identifier.
            symbol: Trading symbol.
            requested_qty: Requested quantity.
            filled_qty: Actual filled quantity.
            fill_time_ms: Time to fill in milliseconds.
            timestamp: Event timestamp (defaults to now).

        Returns:
            FillEvent with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        event = FillEvent(
            order_id=order_id,
            symbol=symbol,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            fill_time_ms=fill_time_ms,
            timestamp=timestamp,
        )

        self._fill_history.append(event)

        # Structured logging for fill event
        log_data = {
            "event_type": "fill_recorded",
            "order_id": order_id,
            "symbol": symbol,
            "requested_qty": requested_qty,
            "filled_qty": filled_qty,
            "fill_ratio": round(event.fill_ratio, 4),
            "fill_time_ms": round(fill_time_ms, 2),
            "is_full_fill": event.is_full_fill,
            "is_partial_fill": event.is_partial_fill,
            "timestamp": timestamp.isoformat(),
        }

        if event.is_full_fill:
            logger.info("fill_complete", **log_data)
        elif event.is_partial_fill:
            logger.warning("fill_partial", **log_data)
        else:
            logger.error("fill_failed", **log_data)

        # Check for latency warning
        if fill_time_ms > self.config.latency_warning_ms:
            logger.warning(
                "fill_latency_warning",
                order_id=order_id,
                symbol=symbol,
                fill_time_ms=fill_time_ms,
                threshold_ms=self.config.latency_warning_ms,
            )

        # Emit stats update and check for alerts
        self._emit_stats_and_check_alerts(symbol)

        return event

    def _emit_stats_and_check_alerts(self, symbol: str | None = None) -> None:
        """Emit stats to callbacks and check for alert level changes.

        Args:
            symbol: Optional symbol to filter stats.
        """
        stats = self.get_stats(symbol)

        # Notify callbacks
        for callback in self._stats_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error("stats_callback_error", error=str(e))

        # Check for alert level change
        if stats.alert_level != self._last_alert_level:
            self._generate_alert(stats, symbol)
            self._last_alert_level = stats.alert_level

    def _generate_alert(self, stats: FillRateStats, symbol: str | None = None) -> None:
        """Generate and log an alert based on current stats.

        Args:
            stats: Current fill rate statistics.
            symbol: Optional symbol filter.
        """
        now = datetime.now(UTC)

        if stats.alert_level == "critical":
            severity = AlertSeverity.CRITICAL
            message = f"Fill rate critical: {stats.fill_rate_pct} (threshold: {self.config.critical_threshold * 100:.0f}%)"
        elif stats.alert_level == "warning":
            severity = AlertSeverity.WARNING
            message = f"Fill rate warning: {stats.fill_rate_pct} (threshold: {self.config.warning_threshold * 100:.0f}%)"
        else:
            severity = AlertSeverity.INFO
            message = f"Fill rate recovered: {stats.fill_rate_pct}"

        alert = FillAlert(
            timestamp=now,
            severity=severity,
            message=message,
            fill_rate=stats.fill_rate,
            symbol=symbol,
            details={
                "total_orders": stats.total_orders,
                "full_fills": stats.full_fills,
                "partial_fills": stats.partial_fills,
                "unfilled": stats.unfilled,
                "avg_latency_ms": stats.avg_latency_ms,
            },
        )

        self._alert_history.append(alert)

        # Log with appropriate level
        log_func = {
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.INFO: logger.info,
        }[severity]

        log_func(
            "fill_rate_alert",
            severity=severity.value,
            message=message,
            fill_rate=stats.fill_rate,
            symbol=symbol,
            total_orders=stats.total_orders,
        )

    def get_alert_history(self, limit: int = 10) -> list[FillAlert]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent FillAlert objects.
        """
        return list(self._alert_history)[-limit:]

    def get_stats(self, symbol: str | None = None) -> FillRateStats:
        """Get current fill rate statistics.

        Args:
            symbol: Optional symbol to filter stats.

        Returns:
            FillRateStats with current metrics.
        """
        # Filter by symbol if specified
        if symbol:
            events = [e for e in self._fill_history if e.symbol == symbol]
        else:
            events = list(self._fill_history)

        if not events:
            return FillRateStats(
                total_orders=0,
                full_fills=0,
                partial_fills=0,
                unfilled=0,
                fill_rate=0.0,
                avg_latency_ms=0.0,
                max_latency_ms=0.0,
                alert_level="none",
                latency_warning=False,
                target_fill_rate=self.config.expected_fill_rate,
            )

        # Calculate fill statistics
        full_fills = sum(1 for e in events if e.is_full_fill)
        partial_fills = sum(1 for e in events if e.is_partial_fill)
        unfilled = sum(1 for e in events if e.fill_ratio == 0.0)

        # Calculate average fill rate (weighted by fill ratio)
        fill_rate = sum(e.fill_ratio for e in events) / len(events)

        # Calculate latency metrics (only for filled orders)
        filled_events = [e for e in events if e.fill_time_ms > 0]
        if filled_events:
            avg_latency = sum(e.fill_time_ms for e in filled_events) / len(filled_events)
            max_latency = max(e.fill_time_ms for e in filled_events)
        else:
            avg_latency = 0.0
            max_latency = 0.0

        # Determine alert level
        alert_level = self._determine_alert_level(fill_rate)

        # Check latency warning
        latency_warning = avg_latency > self.config.latency_warning_ms

        return FillRateStats(
            total_orders=len(events),
            full_fills=full_fills,
            partial_fills=partial_fills,
            unfilled=unfilled,
            fill_rate=fill_rate,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            alert_level=alert_level,
            latency_warning=latency_warning,
            target_fill_rate=self.config.expected_fill_rate,
        )

    def emit_statistics(self, symbol: str | None = None) -> FillRateStats:
        """Emit current statistics with structured logging.

        Args:
            symbol: Optional symbol to filter stats.

        Returns:
            Current FillRateStats.
        """
        stats = self.get_stats(symbol)

        logger.info(
            "fill_rate_statistics",
            total_orders=stats.total_orders,
            full_fills=stats.full_fills,
            partial_fills=stats.partial_fills,
            unfilled=stats.unfilled,
            fill_rate=round(stats.fill_rate, 4),
            fill_rate_pct=stats.fill_rate_pct,
            target_pct=stats.target_pct,
            avg_latency_ms=round(stats.avg_latency_ms, 2),
            max_latency_ms=round(stats.max_latency_ms, 2),
            alert_level=stats.alert_level,
            latency_warning=stats.latency_warning,
            symbol=symbol,
        )

        return stats

    def should_alert(self) -> bool:
        """Check if fill rate alert should be triggered.

        Returns:
            True if fill rate is below warning threshold.
        """
        stats = self.get_stats()
        return stats.alert_level in ("warning", "critical")

    def reset(self) -> None:
        """Reset fill rate statistics."""
        self._fill_history.clear()
        logger.info("fill_rate_tracker_reset")

    def _determine_alert_level(self, fill_rate: float) -> str:
        """Determine alert level based on fill rate.

        Args:
            fill_rate: Current fill rate as decimal.

        Returns:
            Alert level string.
        """
        if fill_rate < self.config.critical_threshold:
            return "critical"
        elif fill_rate < self.config.warning_threshold:
            return "warning"
        return "none"
