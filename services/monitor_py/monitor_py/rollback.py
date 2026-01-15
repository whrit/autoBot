"""Rollback Trigger Logic (T5.11).

Automatic rollback based on monitoring conditions.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import structlog

from monitor_py.drawdown import DrawdownStatus
from monitor_py.feed_health import FeedHealthStatus
from monitor_py.slippage import SlippageStats

logger = structlog.get_logger(__name__)


@dataclass
class RollbackConfig:
    """Configuration for rollback triggers.

    Attributes:
        max_drawdown_trigger: Drawdown threshold to trigger rollback.
        max_slippage_ratio_trigger: Slippage ratio threshold to trigger rollback.
        feed_disconnect_trigger: Number of disconnects to trigger rollback.
        cooldown_minutes: Cooldown period between rollbacks.
        registry_url: URL of the registry API.
    """

    max_drawdown_trigger: float = 0.15
    max_slippage_ratio_trigger: float = 3.0
    feed_disconnect_trigger: int = 5
    cooldown_minutes: int = 30
    registry_url: str = "http://localhost:8000"


@dataclass
class RollbackAction:
    """Rollback action details.

    Attributes:
        triggered: Whether rollback was triggered.
        reason: Reason for rollback.
        strategy_id: Strategy to roll back.
        target_state: Target state after rollback ("shadow" or "candidate").
        timestamp: When the rollback was triggered.
    """

    triggered: bool
    reason: str
    strategy_id: int | None
    target_state: str | None
    timestamp: datetime


@dataclass
class RollbackManager:
    """Manage automatic strategy rollbacks.

    Monitors conditions and triggers rollbacks when thresholds
    are exceeded, with cooldown to prevent rapid rollbacks.
    """

    config: RollbackConfig
    _last_rollback_time: datetime | None = field(default=None, init=False)
    _registry_client: httpx.Client | None = field(default=None, init=False)

    def check_rollback_conditions(
        self,
        drawdown_status: DrawdownStatus,
        slippage_stats: SlippageStats,
        feed_status: FeedHealthStatus,
        strategy_id: int,
    ) -> RollbackAction | None:
        """Check if rollback conditions are met.

        Args:
            drawdown_status: Current drawdown status.
            slippage_stats: Current slippage statistics.
            feed_status: Current feed health status.
            strategy_id: Strategy ID to potentially roll back.

        Returns:
            RollbackAction if triggered, None otherwise.
        """
        # Check cooldown
        if self.in_cooldown():
            logger.debug(
                "rollback_check_skipped_cooldown",
                strategy_id=strategy_id,
            )
            return None

        now = datetime.now(UTC)

        # Check drawdown threshold
        if drawdown_status.current_drawdown >= self.config.max_drawdown_trigger:
            reason = (
                f"Drawdown {drawdown_status.current_drawdown:.1%} "
                f"exceeds threshold {self.config.max_drawdown_trigger:.1%}"
            )
            logger.warning(
                "rollback_triggered_drawdown",
                drawdown=drawdown_status.current_drawdown,
                threshold=self.config.max_drawdown_trigger,
                strategy_id=strategy_id,
            )
            return RollbackAction(
                triggered=True,
                reason=reason,
                strategy_id=strategy_id,
                target_state="shadow",
                timestamp=now,
            )

        # Check slippage threshold
        if slippage_stats.ratio_to_expected >= self.config.max_slippage_ratio_trigger:
            reason = (
                f"Slippage ratio {slippage_stats.ratio_to_expected:.1f}x "
                f"exceeds threshold {self.config.max_slippage_ratio_trigger:.1f}x"
            )
            logger.warning(
                "rollback_triggered_slippage",
                ratio=slippage_stats.ratio_to_expected,
                threshold=self.config.max_slippage_ratio_trigger,
                strategy_id=strategy_id,
            )
            return RollbackAction(
                triggered=True,
                reason=reason,
                strategy_id=strategy_id,
                target_state="shadow",
                timestamp=now,
            )

        # Check feed disconnects
        if feed_status.disconnect_count >= self.config.feed_disconnect_trigger:
            reason = (
                f"Feed disconnects {feed_status.disconnect_count} "
                f"exceeds threshold {self.config.feed_disconnect_trigger}"
            )
            logger.warning(
                "rollback_triggered_feed_disconnects",
                disconnects=feed_status.disconnect_count,
                threshold=self.config.feed_disconnect_trigger,
                strategy_id=strategy_id,
            )
            return RollbackAction(
                triggered=True,
                reason=reason,
                strategy_id=strategy_id,
                target_state="shadow",
                timestamp=now,
            )

        return None

    def execute_rollback(self, action: RollbackAction) -> bool:
        """Execute rollback (calls registry API).

        Args:
            action: The rollback action to execute.

        Returns:
            True if rollback was successful.
        """
        if not action.triggered:
            return False

        logger.warning(
            "executing_rollback",
            strategy_id=action.strategy_id,
            reason=action.reason,
            target_state=action.target_state,
        )

        # Call registry API
        success = self._call_registry_api(action)

        if success:
            self._last_rollback_time = datetime.now(UTC)
            logger.info(
                "rollback_executed",
                strategy_id=action.strategy_id,
                target_state=action.target_state,
            )
        else:
            logger.error(
                "rollback_failed",
                strategy_id=action.strategy_id,
                reason="Registry API call failed",
            )

        return success

    def in_cooldown(self) -> bool:
        """Check if manager is in cooldown period.

        Returns:
            True if cooldown is active.
        """
        if self._last_rollback_time is None:
            return False

        cooldown_end = self._last_rollback_time + timedelta(
            minutes=self.config.cooldown_minutes
        )
        return datetime.now(UTC) < cooldown_end

    def _call_registry_api(self, action: RollbackAction) -> bool:
        """Call registry API to execute rollback.

        Args:
            action: The rollback action.

        Returns:
            True if API call was successful.
        """
        if action.strategy_id is None:
            return False

        try:
            # Create client if needed
            if self._registry_client is None:
                self._registry_client = httpx.Client(
                    base_url=self.config.registry_url,
                    timeout=30.0,
                )

            # Call the registry API to demote the strategy
            response = self._registry_client.post(
                f"/api/v1/strategies/{action.strategy_id}/demote",
                json={
                    "target_state": action.target_state,
                    "reason": action.reason,
                    "timestamp": action.timestamp.isoformat(),
                },
            )
            return response.status_code == 200

        except httpx.HTTPError as e:
            logger.error(
                "registry_api_error",
                error=str(e),
                strategy_id=action.strategy_id,
            )
            return False
        except Exception as e:
            logger.error(
                "unexpected_error_calling_registry",
                error=str(e),
                strategy_id=action.strategy_id,
            )
            return False
