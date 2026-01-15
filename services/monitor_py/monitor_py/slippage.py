"""Slippage Tracking (T5.08).

Track actual vs expected slippage to detect execution quality issues.
"""

from collections import deque
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SlippageConfig:
    """Configuration for slippage tracking.

    Attributes:
        expected_slippage_bps: Expected slippage in basis points.
        alert_threshold_multiplier: Multiplier of expected slippage to trigger alert.
        window_size: Number of fills to track in rolling window.
    """

    expected_slippage_bps: float = 2.0
    alert_threshold_multiplier: float = 2.0
    window_size: int = 100


@dataclass
class SlippageStats:
    """Slippage statistics.

    Attributes:
        avg_slippage_bps: Average slippage in basis points.
        max_slippage_bps: Maximum slippage in basis points.
        expected_slippage_bps: Expected slippage for comparison.
        ratio_to_expected: Ratio of actual to expected slippage.
        sample_count: Number of fills in the sample.
        alert_triggered: Whether slippage alert is triggered.
    """

    avg_slippage_bps: float
    max_slippage_bps: float
    expected_slippage_bps: float
    ratio_to_expected: float
    sample_count: int
    alert_triggered: bool


@dataclass
class SlippageTracker:
    """Track fill slippage vs expectations.

    Monitors execution quality by comparing actual fill prices
    to expected prices and calculating slippage metrics.
    """

    config: SlippageConfig
    _slippage_history: deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._slippage_history = deque(maxlen=self.config.window_size)

    def record_fill(self, expected_price: float, actual_price: float, side: str) -> float:
        """Record a fill and return slippage in bps.

        For buys: positive slippage means worse execution (paid more).
        For sells: positive slippage means worse execution (received less).

        Args:
            expected_price: The expected/target price.
            actual_price: The actual fill price.
            side: "buy" or "sell".

        Returns:
            Slippage in basis points (positive = adverse).
        """
        if expected_price <= 0:
            logger.warning("invalid_expected_price", price=expected_price)
            return 0.0

        # Calculate slippage in bps
        # For buy: (actual - expected) / expected * 10000
        # For sell: (expected - actual) / expected * 10000
        if side.lower() == "buy":
            slippage_bps = (actual_price - expected_price) / expected_price * 10000
        else:  # sell
            slippage_bps = (expected_price - actual_price) / expected_price * 10000

        self._slippage_history.append(slippage_bps)

        logger.debug(
            "fill_recorded",
            expected_price=expected_price,
            actual_price=actual_price,
            side=side,
            slippage_bps=slippage_bps,
        )

        return slippage_bps

    def get_stats(self) -> SlippageStats:
        """Get current slippage statistics.

        Returns:
            SlippageStats with current metrics.
        """
        if not self._slippage_history:
            return SlippageStats(
                avg_slippage_bps=0.0,
                max_slippage_bps=0.0,
                expected_slippage_bps=self.config.expected_slippage_bps,
                ratio_to_expected=0.0,
                sample_count=0,
                alert_triggered=False,
            )

        avg_slippage = sum(self._slippage_history) / len(self._slippage_history)
        max_slippage = max(self._slippage_history)

        # Calculate ratio to expected (avoid division by zero)
        if self.config.expected_slippage_bps > 0:
            ratio = avg_slippage / self.config.expected_slippage_bps
        else:
            ratio = 0.0

        # Check if alert should be triggered
        alert_threshold = (
            self.config.expected_slippage_bps * self.config.alert_threshold_multiplier
        )
        alert_triggered = avg_slippage > alert_threshold

        return SlippageStats(
            avg_slippage_bps=avg_slippage,
            max_slippage_bps=max_slippage,
            expected_slippage_bps=self.config.expected_slippage_bps,
            ratio_to_expected=ratio,
            sample_count=len(self._slippage_history),
            alert_triggered=alert_triggered,
        )

    def should_alert(self) -> bool:
        """Check if slippage alert should be triggered.

        Returns:
            True if average slippage exceeds threshold.
        """
        return self.get_stats().alert_triggered
