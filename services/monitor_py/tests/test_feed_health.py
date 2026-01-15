"""Tests for Feed Health Monitoring (T5.07)."""

from datetime import UTC, datetime, timedelta

from monitor_py.feed_health import (
    FeedHealthConfig,
    FeedHealthMonitor,
    FeedHealthStatus,
    FeedStatus,
)


class TestFeedHealthConfig:
    """Test FeedHealthConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeedHealthConfig()
        assert config.max_lag_seconds == 5.0
        assert config.stale_threshold_seconds == 30.0
        assert config.disconnect_threshold_seconds == 60.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeedHealthConfig(
            max_lag_seconds=10.0,
            stale_threshold_seconds=60.0,
            disconnect_threshold_seconds=120.0,
        )
        assert config.max_lag_seconds == 10.0
        assert config.stale_threshold_seconds == 60.0
        assert config.disconnect_threshold_seconds == 120.0


class TestFeedHealthMonitor:
    """Test FeedHealthMonitor functionality."""

    def test_initial_status_unknown(self) -> None:
        """Test that initial status is disconnected (no messages yet)."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        status = monitor.get_status()
        assert status.status == FeedStatus.DISCONNECTED
        assert status.last_message_time is None
        assert status.disconnect_count == 0

    def test_healthy_feed_after_message(self) -> None:
        """Test feed becomes healthy after receiving message."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        now = datetime.now(UTC)
        monitor.record_message(now)
        status = monitor.get_status()
        assert status.status == FeedStatus.HEALTHY
        assert status.last_message_time == now
        assert status.current_lag_seconds < 1.0

    def test_degraded_feed_on_lag(self) -> None:
        """Test feed becomes degraded when lag exceeds threshold."""
        config = FeedHealthConfig(max_lag_seconds=5.0)
        monitor = FeedHealthMonitor(config)
        # Record a message from 10 seconds ago
        old_time = datetime.now(UTC) - timedelta(seconds=10)
        monitor.record_message(old_time)
        status = monitor.get_status()
        assert status.status == FeedStatus.DEGRADED
        assert status.current_lag_seconds >= 10.0

    def test_disconnected_feed_on_stale(self) -> None:
        """Test feed becomes disconnected when messages are stale."""
        config = FeedHealthConfig(stale_threshold_seconds=30.0)
        monitor = FeedHealthMonitor(config)
        # Record a message from 60 seconds ago
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        monitor.record_message(old_time)
        status = monitor.get_status()
        assert status.status == FeedStatus.DISCONNECTED

    def test_record_disconnect(self) -> None:
        """Test disconnect counting."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        monitor.record_disconnect()
        monitor.record_disconnect()
        status = monitor.get_status()
        assert status.disconnect_count == 2

    def test_message_rate_calculation(self) -> None:
        """Test messages per second calculation."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        now = datetime.now(UTC)
        # Record 10 messages over 1 second
        for i in range(10):
            monitor.record_message(now - timedelta(milliseconds=100 * i))
        status = monitor.get_status()
        # Should have approximately 10 messages
        assert status.messages_per_second >= 5.0

    def test_is_healthy_true(self) -> None:
        """Test is_healthy returns True for healthy feed."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        now = datetime.now(UTC)
        monitor.record_message(now)
        assert monitor.is_healthy() is True

    def test_is_healthy_false_when_degraded(self) -> None:
        """Test is_healthy returns False when degraded."""
        config = FeedHealthConfig(max_lag_seconds=5.0)
        monitor = FeedHealthMonitor(config)
        old_time = datetime.now(UTC) - timedelta(seconds=10)
        monitor.record_message(old_time)
        assert monitor.is_healthy() is False

    def test_is_healthy_false_when_disconnected(self) -> None:
        """Test is_healthy returns False when disconnected."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        assert monitor.is_healthy() is False

    def test_status_message(self) -> None:
        """Test status message is descriptive."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        status = monitor.get_status()
        assert len(status.message) > 0
        assert "No messages received" in status.message or "disconnect" in status.message.lower()


class TestFeedHealthStatus:
    """Test FeedHealthStatus dataclass."""

    def test_status_creation(self) -> None:
        """Test FeedHealthStatus can be created."""
        status = FeedHealthStatus(
            status=FeedStatus.HEALTHY,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=0.5,
            messages_per_second=100.0,
            disconnect_count=0,
            message="Feed is healthy",
        )
        assert status.status == FeedStatus.HEALTHY
        assert status.messages_per_second == 100.0
