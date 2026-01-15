"""Drawdown Monitoring (T5.09).

Monitor portfolio drawdown and generate alerts when thresholds are exceeded.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DrawdownConfig:
    """Configuration for drawdown monitoring.

    Attributes:
        warning_threshold: Drawdown percentage to trigger warning (e.g., 0.05 = 5%).
        critical_threshold: Drawdown percentage to trigger critical alert.
        kill_switch_threshold: Drawdown percentage to trigger trading halt.
    """

    warning_threshold: float = 0.05
    critical_threshold: float = 0.10
    kill_switch_threshold: float = 0.15


@dataclass
class DrawdownStatus:
    """Drawdown status information.

    Attributes:
        current_drawdown: Current drawdown as decimal (0.05 = 5%).
        max_drawdown: Maximum drawdown seen as decimal.
        peak_equity: Peak equity value reached.
        current_equity: Current equity value.
        alert_level: Alert level ("none", "warning", "critical", "kill_switch").
        days_in_drawdown: Number of days in current drawdown.
    """

    current_drawdown: float
    max_drawdown: float
    peak_equity: float
    current_equity: float
    alert_level: str
    days_in_drawdown: int


@dataclass
class DrawdownMonitor:
    """Monitor portfolio drawdown.

    Tracks equity changes and calculates drawdown metrics,
    triggering alerts when thresholds are exceeded.
    """

    config: DrawdownConfig
    initial_equity: float = 100000.0
    _peak_equity: float = field(default=0.0, init=False)
    _current_equity: float = field(default=0.0, init=False)
    _max_drawdown: float = field(default=0.0, init=False)
    _drawdown_start_time: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._peak_equity = self.initial_equity
        self._current_equity = self.initial_equity

    def update(self, equity: float) -> DrawdownStatus:
        """Update equity and recalculate drawdown.

        Args:
            equity: Current portfolio equity value.

        Returns:
            Updated DrawdownStatus.
        """
        self._current_equity = equity

        # Update peak if new high
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._drawdown_start_time = None  # No longer in drawdown
            logger.info(
                "new_peak_equity",
                peak_equity=self._peak_equity,
            )
        elif self._drawdown_start_time is None and equity < self._peak_equity:
            # Entering drawdown
            self._drawdown_start_time = datetime.now(UTC)

        # Calculate current drawdown
        current_drawdown = self._calculate_drawdown()

        # Update max drawdown
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown
            logger.warning(
                "new_max_drawdown",
                max_drawdown=self._max_drawdown,
                current_equity=equity,
                peak_equity=self._peak_equity,
            )

        return self.get_status()

    def get_status(self) -> DrawdownStatus:
        """Get current drawdown status.

        Returns:
            DrawdownStatus with current metrics.
        """
        current_drawdown = self._calculate_drawdown()
        alert_level = self._determine_alert_level(current_drawdown)
        days_in_drawdown = self._calculate_days_in_drawdown()

        return DrawdownStatus(
            current_drawdown=current_drawdown,
            max_drawdown=self._max_drawdown,
            peak_equity=self._peak_equity,
            current_equity=self._current_equity,
            alert_level=alert_level,
            days_in_drawdown=days_in_drawdown,
        )

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted due to drawdown.

        Returns:
            True if kill switch threshold is breached.
        """
        current_drawdown = self._calculate_drawdown()
        return current_drawdown >= self.config.kill_switch_threshold

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak.

        Returns:
            Drawdown as decimal (0.05 = 5%).
        """
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def _determine_alert_level(self, drawdown: float) -> str:
        """Determine alert level based on drawdown.

        Args:
            drawdown: Current drawdown as decimal.

        Returns:
            Alert level string.
        """
        if drawdown >= self.config.kill_switch_threshold:
            return "kill_switch"
        elif drawdown >= self.config.critical_threshold:
            return "critical"
        elif drawdown >= self.config.warning_threshold:
            return "warning"
        return "none"

    def _calculate_days_in_drawdown(self) -> int:
        """Calculate days in current drawdown.

        Returns:
            Number of days in drawdown.
        """
        if self._drawdown_start_time is None:
            return 0
        delta = datetime.now(UTC) - self._drawdown_start_time
        return max(0, delta.days)
