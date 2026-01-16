"""Fill Rate Monitoring.

Track order fill rates to detect execution quality issues.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger(__name__)


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


@dataclass
class FillRateTracker:
    """Track order fill rates and detect execution quality issues.

    Monitors fill rates, partial fills, and fill latency to detect
    execution quality degradation.
    """

    config: FillRateConfig
    _fill_history: deque[FillEvent] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._fill_history = deque(maxlen=self.config.window_size)

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        requested_qty: float,
        filled_qty: float,
        fill_time_ms: float,
        timestamp: datetime | None = None,
    ) -> FillEvent:
        """Record a fill event.

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

        logger.debug(
            "fill_recorded",
            order_id=order_id,
            symbol=symbol,
            fill_ratio=event.fill_ratio,
            fill_time_ms=fill_time_ms,
        )

        return event

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
        )

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
