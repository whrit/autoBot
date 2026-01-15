"""Feed Health Monitoring (T5.07).

Monitor WebSocket feed connectivity and lag to detect issues early.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class FeedStatus(str, Enum):
    """Feed health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


@dataclass
class FeedHealthConfig:
    """Configuration for feed health monitoring.

    Attributes:
        max_lag_seconds: Maximum acceptable lag before feed is degraded.
        stale_threshold_seconds: Time after which feed is considered stale/disconnected.
        disconnect_threshold_seconds: Time threshold for considering feed disconnected.
    """

    max_lag_seconds: float = 5.0
    stale_threshold_seconds: float = 30.0
    disconnect_threshold_seconds: float = 60.0


@dataclass
class FeedHealthStatus:
    """Current feed health status.

    Attributes:
        status: Current status (healthy, degraded, disconnected).
        last_message_time: Timestamp of last received message.
        current_lag_seconds: Current lag in seconds.
        messages_per_second: Message rate.
        disconnect_count: Number of disconnections.
        message: Human-readable status message.
    """

    status: FeedStatus
    last_message_time: datetime | None
    current_lag_seconds: float
    messages_per_second: float
    disconnect_count: int
    message: str


@dataclass
class FeedHealthMonitor:
    """Monitor WebSocket feed health.

    Tracks message timing, lag, and disconnections to provide
    real-time health assessment of market data feeds.
    """

    config: FeedHealthConfig
    _last_message_time: datetime | None = field(default=None, init=False)
    _disconnect_count: int = field(default=0, init=False)
    _message_times: deque[datetime] = field(
        default_factory=lambda: deque(maxlen=1000), init=False
    )
    _rate_window_seconds: float = field(default=1.0, init=False)

    def record_message(self, timestamp: datetime) -> None:
        """Record receipt of a message.

        Args:
            timestamp: The timestamp when the message was received.
        """
        self._last_message_time = timestamp
        self._message_times.append(timestamp)
        logger.debug(
            "message_recorded",
            timestamp=timestamp.isoformat(),
            queue_size=len(self._message_times),
        )

    def record_disconnect(self) -> None:
        """Record a disconnection event."""
        self._disconnect_count += 1
        logger.warning(
            "feed_disconnect_recorded",
            disconnect_count=self._disconnect_count,
        )

    def get_status(self) -> FeedHealthStatus:
        """Get current feed health status.

        Returns:
            FeedHealthStatus with current health information.
        """
        now = datetime.now(UTC)

        # No messages ever received
        if self._last_message_time is None:
            return FeedHealthStatus(
                status=FeedStatus.DISCONNECTED,
                last_message_time=None,
                current_lag_seconds=float("inf"),
                messages_per_second=0.0,
                disconnect_count=self._disconnect_count,
                message="No messages received yet",
            )

        # Calculate lag
        lag_seconds = (now - self._last_message_time).total_seconds()

        # Calculate message rate
        messages_per_second = self._calculate_message_rate(now)

        # Determine status
        if lag_seconds > self.config.stale_threshold_seconds:
            status = FeedStatus.DISCONNECTED
            message = f"Feed stale - no messages for {lag_seconds:.1f}s"
        elif lag_seconds > self.config.max_lag_seconds:
            status = FeedStatus.DEGRADED
            message = f"Feed degraded - lag is {lag_seconds:.1f}s"
        else:
            status = FeedStatus.HEALTHY
            message = f"Feed healthy - lag is {lag_seconds:.3f}s"

        return FeedHealthStatus(
            status=status,
            last_message_time=self._last_message_time,
            current_lag_seconds=lag_seconds,
            messages_per_second=messages_per_second,
            disconnect_count=self._disconnect_count,
            message=message,
        )

    def is_healthy(self) -> bool:
        """Check if feed is currently healthy.

        Returns:
            True if feed status is HEALTHY.
        """
        return self.get_status().status == FeedStatus.HEALTHY

    def _calculate_message_rate(self, now: datetime) -> float:
        """Calculate messages per second over the rate window.

        Args:
            now: Current timestamp.

        Returns:
            Messages per second.
        """
        if not self._message_times:
            return 0.0

        # Count messages in the rate window
        window_start = now - timedelta(seconds=self._rate_window_seconds)
        count = sum(1 for t in self._message_times if t >= window_start)

        return count / self._rate_window_seconds
