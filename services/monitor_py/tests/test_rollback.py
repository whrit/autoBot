"""Tests for Rollback Trigger Logic (T5.11)."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from monitor_py.drawdown import DrawdownStatus
from monitor_py.feed_health import FeedHealthStatus, FeedStatus
from monitor_py.rollback import RollbackAction, RollbackConfig, RollbackManager
from monitor_py.slippage import SlippageStats


class TestRollbackConfig:
    """Test RollbackConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RollbackConfig()
        assert config.max_drawdown_trigger == 0.15
        assert config.max_slippage_ratio_trigger == 3.0
        assert config.feed_disconnect_trigger == 5
        assert config.cooldown_minutes == 30

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RollbackConfig(
            max_drawdown_trigger=0.20,
            max_slippage_ratio_trigger=4.0,
            feed_disconnect_trigger=10,
            cooldown_minutes=60,
        )
        assert config.max_drawdown_trigger == 0.20
        assert config.max_slippage_ratio_trigger == 4.0
        assert config.feed_disconnect_trigger == 10
        assert config.cooldown_minutes == 60


class TestRollbackManager:
    """Test RollbackManager functionality."""

    @pytest.fixture
    def manager(self) -> RollbackManager:
        """Create a RollbackManager for testing."""
        return RollbackManager(RollbackConfig())

    @pytest.fixture
    def healthy_status(self) -> tuple[DrawdownStatus, SlippageStats, FeedHealthStatus]:
        """Create healthy status objects."""
        drawdown = DrawdownStatus(
            current_drawdown=0.05,
            max_drawdown=0.07,
            peak_equity=100000.0,
            current_equity=95000.0,
            alert_level="warning",
            days_in_drawdown=2,
        )
        slippage = SlippageStats(
            avg_slippage_bps=2.0,
            max_slippage_bps=3.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=1.0,
            sample_count=50,
            alert_triggered=False,
        )
        feed = FeedHealthStatus(
            status=FeedStatus.HEALTHY,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=0.5,
            messages_per_second=100.0,
            disconnect_count=0,
            message="Feed is healthy",
        )
        return drawdown, slippage, feed

    def test_no_rollback_when_healthy(
        self, manager: RollbackManager, healthy_status: tuple
    ) -> None:
        """Test no rollback triggered when all metrics are healthy."""
        drawdown, slippage, feed = healthy_status

        action = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=1,
        )

        assert action is None

    def test_drawdown_rollback(self, manager: RollbackManager, healthy_status: tuple) -> None:
        """Test rollback triggered by excessive drawdown."""
        _, slippage, feed = healthy_status
        drawdown = DrawdownStatus(
            current_drawdown=0.16,  # Above 15% threshold
            max_drawdown=0.16,
            peak_equity=100000.0,
            current_equity=84000.0,
            alert_level="kill_switch",
            days_in_drawdown=5,
        )

        action = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=1,
        )

        assert action is not None
        assert action.triggered is True
        assert "drawdown" in action.reason.lower()
        assert action.strategy_id == 1
        assert action.target_state in ["shadow", "candidate"]

    def test_slippage_rollback(self, manager: RollbackManager, healthy_status: tuple) -> None:
        """Test rollback triggered by excessive slippage."""
        drawdown, _, feed = healthy_status
        slippage = SlippageStats(
            avg_slippage_bps=8.0,  # 4x expected (above 3x threshold)
            max_slippage_bps=12.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=4.0,
            sample_count=50,
            alert_triggered=True,
        )

        action = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=2,
        )

        assert action is not None
        assert action.triggered is True
        assert "slippage" in action.reason.lower()
        assert action.strategy_id == 2

    def test_feed_disconnect_rollback(
        self, manager: RollbackManager, healthy_status: tuple
    ) -> None:
        """Test rollback triggered by too many feed disconnects."""
        drawdown, slippage, _ = healthy_status
        feed = FeedHealthStatus(
            status=FeedStatus.DISCONNECTED,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=60.0,
            messages_per_second=0.0,
            disconnect_count=6,  # Above 5 threshold
            message="Feed disconnected",
        )

        action = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=3,
        )

        assert action is not None
        assert action.triggered is True
        assert "disconnect" in action.reason.lower() or "feed" in action.reason.lower()

    def test_cooldown_period(self, manager: RollbackManager, healthy_status: tuple) -> None:
        """Test cooldown prevents rapid rollbacks."""
        drawdown = DrawdownStatus(
            current_drawdown=0.16,
            max_drawdown=0.16,
            peak_equity=100000.0,
            current_equity=84000.0,
            alert_level="kill_switch",
            days_in_drawdown=5,
        )
        _, slippage, feed = healthy_status

        # First rollback should trigger
        action1 = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=1,
        )
        assert action1 is not None
        # Execute the rollback to start cooldown (mock the API call)
        with patch.object(manager, "_call_registry_api", return_value=True):
            manager.execute_rollback(action1)

        # Second rollback should be blocked by cooldown
        action2 = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=1,
        )
        # Should return None due to cooldown
        assert action2 is None

    def test_in_cooldown(self, manager: RollbackManager) -> None:
        """Test in_cooldown method."""
        assert manager.in_cooldown() is False

        # Trigger a rollback
        drawdown = DrawdownStatus(
            current_drawdown=0.16,
            max_drawdown=0.16,
            peak_equity=100000.0,
            current_equity=84000.0,
            alert_level="kill_switch",
            days_in_drawdown=5,
        )
        slippage = SlippageStats(
            avg_slippage_bps=2.0,
            max_slippage_bps=3.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=1.0,
            sample_count=50,
            alert_triggered=False,
        )
        feed = FeedHealthStatus(
            status=FeedStatus.HEALTHY,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=0.5,
            messages_per_second=100.0,
            disconnect_count=0,
            message="Feed is healthy",
        )

        action = manager.check_rollback_conditions(drawdown, slippage, feed, 1)
        if action:
            # Mock the API call so execute_rollback succeeds and sets cooldown
            with patch.object(manager, "_call_registry_api", return_value=True):
                manager.execute_rollback(action)

        assert manager.in_cooldown() is True

    def test_execute_rollback_success(self, manager: RollbackManager) -> None:
        """Test execute_rollback returns True on success."""
        action = RollbackAction(
            triggered=True,
            reason="Test rollback",
            strategy_id=1,
            target_state="shadow",
            timestamp=datetime.now(UTC),
        )

        # Mock the registry API call
        with patch.object(manager, "_call_registry_api", return_value=True):
            result = manager.execute_rollback(action)

        assert result is True

    def test_execute_rollback_failure(self, manager: RollbackManager) -> None:
        """Test execute_rollback returns False on failure."""
        action = RollbackAction(
            triggered=True,
            reason="Test rollback",
            strategy_id=1,
            target_state="shadow",
            timestamp=datetime.now(UTC),
        )

        # Mock the registry API call to fail
        with patch.object(manager, "_call_registry_api", return_value=False):
            result = manager.execute_rollback(action)

        assert result is False

    def test_multiple_trigger_conditions(
        self, manager: RollbackManager, healthy_status: tuple
    ) -> None:
        """Test rollback reports first trigger condition found."""
        # Multiple bad conditions
        drawdown = DrawdownStatus(
            current_drawdown=0.16,
            max_drawdown=0.16,
            peak_equity=100000.0,
            current_equity=84000.0,
            alert_level="kill_switch",
            days_in_drawdown=5,
        )
        slippage = SlippageStats(
            avg_slippage_bps=8.0,
            max_slippage_bps=12.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=4.0,
            sample_count=50,
            alert_triggered=True,
        )
        _, _, feed = healthy_status

        action = manager.check_rollback_conditions(
            drawdown_status=drawdown,
            slippage_stats=slippage,
            feed_status=feed,
            strategy_id=1,
        )

        # Should trigger on one of the conditions
        assert action is not None
        assert action.triggered is True


class TestRollbackAction:
    """Test RollbackAction dataclass."""

    def test_action_creation(self) -> None:
        """Test RollbackAction can be created."""
        action = RollbackAction(
            triggered=True,
            reason="Drawdown exceeded threshold",
            strategy_id=1,
            target_state="shadow",
            timestamp=datetime.now(UTC),
        )
        assert action.triggered is True
        assert action.strategy_id == 1
        assert action.target_state == "shadow"

    def test_action_with_none_values(self) -> None:
        """Test RollbackAction with None values."""
        action = RollbackAction(
            triggered=False,
            reason="No rollback needed",
            strategy_id=None,
            target_state=None,
            timestamp=datetime.now(UTC),
        )
        assert action.triggered is False
        assert action.strategy_id is None
