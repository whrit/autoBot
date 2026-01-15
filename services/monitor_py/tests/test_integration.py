"""Integration tests for Registry + Monitor (T5.12)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from monitor_py.drawdown import DrawdownConfig, DrawdownMonitor
from monitor_py.feed_health import FeedHealthConfig, FeedHealthMonitor
from monitor_py.metrics import MetricsExporter
from monitor_py.rollback import RollbackConfig, RollbackManager
from monitor_py.slippage import SlippageConfig, SlippageTracker


class TestFullMonitoringPipeline:
    """Test the full monitoring pipeline integration."""

    @pytest.fixture
    def monitoring_stack(self) -> dict:
        """Create a complete monitoring stack."""
        return {
            "feed_monitor": FeedHealthMonitor(FeedHealthConfig()),
            "slippage_tracker": SlippageTracker(SlippageConfig()),
            "drawdown_monitor": DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0),
            "metrics_exporter": MetricsExporter(),
            "rollback_manager": RollbackManager(RollbackConfig()),
        }

    def test_healthy_operation_flow(self, monitoring_stack: dict) -> None:
        """Test monitoring flow under healthy operation."""
        feed_monitor = monitoring_stack["feed_monitor"]
        slippage_tracker = monitoring_stack["slippage_tracker"]
        drawdown_monitor = monitoring_stack["drawdown_monitor"]
        metrics_exporter = monitoring_stack["metrics_exporter"]
        rollback_manager = monitoring_stack["rollback_manager"]

        # Simulate healthy market data flow
        now = datetime.now(UTC)
        for _ in range(10):
            feed_monitor.record_message(now)

        # Simulate normal fills
        for _ in range(10):
            slippage_tracker.record_fill(100.0, 100.02, "buy")  # 2 bps slippage

        # Simulate stable equity
        drawdown_monitor.update(100500.0)

        # Get all statuses
        feed_status = feed_monitor.get_status()
        slippage_stats = slippage_tracker.get_stats()
        drawdown_status = drawdown_monitor.get_status()

        # Update metrics
        metrics_exporter.update_feed_metrics(feed_status)
        metrics_exporter.update_slippage_metrics(slippage_stats)
        metrics_exporter.update_drawdown_metrics(drawdown_status)

        # Check no rollback triggered
        action = rollback_manager.check_rollback_conditions(
            drawdown_status=drawdown_status,
            slippage_stats=slippage_stats,
            feed_status=feed_status,
            strategy_id=1,
        )

        assert action is None
        assert feed_monitor.is_healthy()
        assert not slippage_tracker.should_alert()
        assert not drawdown_monitor.should_halt_trading()

    def test_degraded_feed_triggers_alert(self, monitoring_stack: dict) -> None:
        """Test degraded feed triggers appropriate alerts."""
        feed_monitor = monitoring_stack["feed_monitor"]
        metrics_exporter = monitoring_stack["metrics_exporter"]

        # Simulate stale messages (10 seconds ago)
        old_time = datetime.now(UTC) - timedelta(seconds=10)
        feed_monitor.record_message(old_time)

        feed_status = feed_monitor.get_status()
        metrics_exporter.update_feed_metrics(feed_status)

        # Feed should be degraded
        assert not feed_monitor.is_healthy()
        assert feed_status.current_lag_seconds >= 10.0

    def test_high_slippage_triggers_alert(self, monitoring_stack: dict) -> None:
        """Test high slippage triggers alert."""
        slippage_tracker = monitoring_stack["slippage_tracker"]
        metrics_exporter = monitoring_stack["metrics_exporter"]

        # Simulate high slippage fills (10 bps = 5x expected 2 bps)
        for _ in range(10):
            slippage_tracker.record_fill(100.0, 100.10, "buy")  # 10 bps

        slippage_stats = slippage_tracker.get_stats()
        metrics_exporter.update_slippage_metrics(slippage_stats)

        # Should trigger alert
        assert slippage_tracker.should_alert()
        assert slippage_stats.alert_triggered

    def test_drawdown_triggers_halt(self, monitoring_stack: dict) -> None:
        """Test excessive drawdown triggers trading halt."""
        drawdown_monitor = monitoring_stack["drawdown_monitor"]
        metrics_exporter = monitoring_stack["metrics_exporter"]

        # Simulate major drawdown (16%)
        drawdown_monitor.update(84000.0)

        drawdown_status = drawdown_monitor.get_status()
        metrics_exporter.update_drawdown_metrics(drawdown_status)

        # Should trigger halt
        assert drawdown_monitor.should_halt_trading()
        assert drawdown_status.alert_level == "kill_switch"

    def test_alert_to_rollback_flow(self, monitoring_stack: dict) -> None:
        """Test the flow from alert detection to rollback execution."""
        feed_monitor = monitoring_stack["feed_monitor"]
        slippage_tracker = monitoring_stack["slippage_tracker"]
        drawdown_monitor = monitoring_stack["drawdown_monitor"]
        metrics_exporter = monitoring_stack["metrics_exporter"]
        rollback_manager = monitoring_stack["rollback_manager"]

        # Setup: healthy feed, normal slippage
        now = datetime.now(UTC)
        feed_monitor.record_message(now)
        for _ in range(5):
            slippage_tracker.record_fill(100.0, 100.02, "buy")

        # Trigger: excessive drawdown
        drawdown_monitor.update(83000.0)  # 17% drawdown

        # Get statuses
        feed_status = feed_monitor.get_status()
        slippage_stats = slippage_tracker.get_stats()
        drawdown_status = drawdown_monitor.get_status()

        # Update metrics
        metrics_exporter.update_feed_metrics(feed_status)
        metrics_exporter.update_slippage_metrics(slippage_stats)
        metrics_exporter.update_drawdown_metrics(drawdown_status)

        # Check rollback conditions
        action = rollback_manager.check_rollback_conditions(
            drawdown_status=drawdown_status,
            slippage_stats=slippage_stats,
            feed_status=feed_status,
            strategy_id=1,
        )

        # Should trigger rollback
        assert action is not None
        assert action.triggered
        assert "drawdown" in action.reason.lower()

        # Record alert
        metrics_exporter.record_alert("drawdown_kill_switch")

        # Execute rollback (mocked)
        with patch.object(rollback_manager, "_call_registry_api", return_value=True):
            result = rollback_manager.execute_rollback(action)

        assert result is True
        metrics_exporter.record_rollback()

    def test_metrics_export_complete(self, monitoring_stack: dict) -> None:
        """Test complete metrics are exported in Prometheus format."""
        feed_monitor = monitoring_stack["feed_monitor"]
        slippage_tracker = monitoring_stack["slippage_tracker"]
        drawdown_monitor = monitoring_stack["drawdown_monitor"]
        metrics_exporter = monitoring_stack["metrics_exporter"]

        # Generate some activity
        now = datetime.now(UTC)
        feed_monitor.record_message(now)
        slippage_tracker.record_fill(100.0, 100.02, "buy")
        drawdown_monitor.update(99000.0)

        # Update all metrics
        metrics_exporter.update_feed_metrics(feed_monitor.get_status())
        metrics_exporter.update_slippage_metrics(slippage_tracker.get_stats())
        metrics_exporter.update_drawdown_metrics(drawdown_monitor.get_status())
        metrics_exporter.record_alert("test_alert")

        # Get metrics
        metrics = metrics_exporter.get_metrics().decode("utf-8")

        # Should contain all metric types
        assert "feed_lag" in metrics
        assert "slippage" in metrics
        assert "drawdown" in metrics
        assert "alerts_triggered" in metrics

    def test_recovery_scenario(self, monitoring_stack: dict) -> None:
        """Test monitoring during recovery from adverse conditions."""
        drawdown_monitor = monitoring_stack["drawdown_monitor"]

        # Phase 1: Bad conditions
        drawdown_monitor.update(85000.0)  # 15% drawdown
        assert drawdown_monitor.should_halt_trading()

        # Phase 2: Recovery
        drawdown_monitor.update(95000.0)  # Back to 5% drawdown
        drawdown_status = drawdown_monitor.get_status()

        # Should not halt trading after recovery
        assert not drawdown_monitor.should_halt_trading()
        assert drawdown_status.alert_level == "warning"  # Just warning now


class TestRegistryIntegration:
    """Test integration with registry API (mocked)."""

    def test_rollback_calls_registry(self) -> None:
        """Test rollback manager calls registry API."""
        manager = RollbackManager(RollbackConfig())

        # Create rollback action
        from monitor_py.rollback import RollbackAction

        action = RollbackAction(
            triggered=True,
            reason="Test",
            strategy_id=1,
            target_state="shadow",
            timestamp=datetime.now(UTC),
        )

        # Mock the internal API call
        with patch.object(manager, "_call_registry_api") as mock_call:
            mock_call.return_value = True
            result = manager.execute_rollback(action)

            assert result is True
            mock_call.assert_called_once()

    def test_rollback_handles_registry_failure(self) -> None:
        """Test rollback handles registry API failure gracefully."""
        manager = RollbackManager(RollbackConfig())

        from monitor_py.rollback import RollbackAction

        action = RollbackAction(
            triggered=True,
            reason="Test",
            strategy_id=1,
            target_state="shadow",
            timestamp=datetime.now(UTC),
        )

        # Mock the internal API call to fail
        with patch.object(manager, "_call_registry_api") as mock_call:
            mock_call.return_value = False
            result = manager.execute_rollback(action)

            assert result is False


class TestConcurrentMonitoring:
    """Test concurrent monitoring scenarios."""

    def test_rapid_equity_updates(self) -> None:
        """Test rapid equity updates are handled correctly."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        # Simulate rapid updates
        for i in range(100):
            equity = 100000.0 - (i * 100)  # Gradual decline
            monitor.update(equity)

        status = monitor.get_status()
        assert status.current_equity == 90100.0
        assert status.max_drawdown >= status.current_drawdown

    def test_rapid_message_recording(self) -> None:
        """Test rapid message recording."""
        monitor = FeedHealthMonitor(FeedHealthConfig())
        now = datetime.now(UTC)

        # Simulate burst of messages
        for i in range(1000):
            monitor.record_message(now - timedelta(milliseconds=i))

        status = monitor.get_status()
        assert status.messages_per_second > 0

    def test_window_overflow(self) -> None:
        """Test slippage tracker handles window overflow."""
        config = SlippageConfig(window_size=10)
        tracker = SlippageTracker(config)

        # Record many fills
        for i in range(100):
            tracker.record_fill(100.0, 100.0 + (i * 0.001), "buy")

        stats = tracker.get_stats()
        assert stats.sample_count == 10  # Window size
