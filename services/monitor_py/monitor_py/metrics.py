"""Prometheus Metrics Export (T5.10).

Export monitoring metrics in Prometheus format for scraping.
"""

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from monitor_py.drawdown import DrawdownStatus
from monitor_py.feed_health import FeedHealthStatus
from monitor_py.slippage import SlippageStats

logger = structlog.get_logger(__name__)

# Feed metrics
FEED_LAG = Gauge(
    "feed_lag_seconds",
    "Current feed lag in seconds",
)
FEED_MESSAGES_TOTAL = Counter(
    "feed_messages_total",
    "Total feed messages received (counter for prometheus)",
)
FEED_MESSAGES_PER_SECOND = Gauge(
    "feed_messages_per_second",
    "Feed messages per second",
)
FEED_DISCONNECTS = Counter(
    "feed_disconnects_total",
    "Total feed disconnections",
)
FEED_STATUS = Gauge(
    "feed_status",
    "Feed status (0=disconnected, 1=degraded, 2=healthy)",
)

# Slippage metrics
SLIPPAGE_BPS = Histogram(
    "slippage_bps",
    "Fill slippage in basis points",
    buckets=[0.5, 1, 2, 5, 10, 20, 50],
)
SLIPPAGE_AVG = Gauge(
    "slippage_avg_bps",
    "Average slippage in basis points",
)
SLIPPAGE_MAX = Gauge(
    "slippage_max_bps",
    "Maximum slippage in basis points",
)
SLIPPAGE_RATIO = Gauge(
    "slippage_ratio",
    "Ratio of actual to expected slippage",
)

# Drawdown metrics
DRAWDOWN_CURRENT = Gauge(
    "drawdown_current",
    "Current drawdown percentage",
)
DRAWDOWN_MAX = Gauge(
    "drawdown_max",
    "Maximum drawdown percentage",
)
EQUITY_CURRENT = Gauge(
    "equity_current",
    "Current portfolio equity",
)
EQUITY_PEAK = Gauge(
    "equity_peak",
    "Peak portfolio equity",
)
DAYS_IN_DRAWDOWN = Gauge(
    "days_in_drawdown",
    "Number of days in current drawdown",
)

# Alert metrics
ALERTS_TRIGGERED = Counter(
    "alerts_triggered_total",
    "Total alerts triggered",
    ["alert_type"],
)
ROLLBACKS_TRIGGERED = Counter(
    "rollbacks_triggered_total",
    "Total rollbacks triggered",
)


class MetricsExporter:
    """Export metrics in Prometheus format.

    Collects metrics from various monitoring components and
    exposes them in Prometheus-compatible format.
    """

    def __init__(self) -> None:
        """Initialize the metrics exporter."""
        self._last_disconnect_count = 0
        logger.info("metrics_exporter_initialized")

    def update_feed_metrics(self, status: FeedHealthStatus) -> None:
        """Update feed health metrics.

        Args:
            status: Current feed health status.
        """
        lag = 0 if status.current_lag_seconds == float("inf") else status.current_lag_seconds
        FEED_LAG.set(lag)
        FEED_MESSAGES_PER_SECOND.set(status.messages_per_second)

        # Map status to numeric value
        status_map = {"disconnected": 0, "degraded": 1, "healthy": 2}
        FEED_STATUS.set(status_map.get(status.status.value, 0))

        # Handle disconnect count (counter can only increase)
        new_disconnects = status.disconnect_count - self._last_disconnect_count
        if new_disconnects > 0:
            FEED_DISCONNECTS.inc(new_disconnects)
            self._last_disconnect_count = status.disconnect_count

        logger.debug(
            "feed_metrics_updated",
            lag=status.current_lag_seconds,
            mps=status.messages_per_second,
            status=status.status.value,
        )

    def update_slippage_metrics(self, stats: SlippageStats) -> None:
        """Update slippage metrics.

        Args:
            stats: Current slippage statistics.
        """
        SLIPPAGE_AVG.set(stats.avg_slippage_bps)
        SLIPPAGE_MAX.set(stats.max_slippage_bps)
        SLIPPAGE_RATIO.set(stats.ratio_to_expected)

        # Record to histogram if we have data
        if stats.avg_slippage_bps > 0:
            SLIPPAGE_BPS.observe(stats.avg_slippage_bps)

        logger.debug(
            "slippage_metrics_updated",
            avg=stats.avg_slippage_bps,
            max=stats.max_slippage_bps,
            ratio=stats.ratio_to_expected,
        )

    def update_drawdown_metrics(self, status: DrawdownStatus) -> None:
        """Update drawdown metrics.

        Args:
            status: Current drawdown status.
        """
        DRAWDOWN_CURRENT.set(status.current_drawdown)
        DRAWDOWN_MAX.set(status.max_drawdown)
        EQUITY_CURRENT.set(status.current_equity)
        EQUITY_PEAK.set(status.peak_equity)
        DAYS_IN_DRAWDOWN.set(status.days_in_drawdown)

        logger.debug(
            "drawdown_metrics_updated",
            current=status.current_drawdown,
            max=status.max_drawdown,
            equity=status.current_equity,
        )

    def record_alert(self, alert_type: str) -> None:
        """Record an alert being triggered.

        Args:
            alert_type: Type of alert triggered.
        """
        ALERTS_TRIGGERED.labels(alert_type=alert_type).inc()
        logger.info("alert_recorded", alert_type=alert_type)

    def record_rollback(self) -> None:
        """Record a rollback being triggered."""
        ROLLBACKS_TRIGGERED.inc()
        logger.warning("rollback_recorded")

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics as bytes.
        """
        return generate_latest()
