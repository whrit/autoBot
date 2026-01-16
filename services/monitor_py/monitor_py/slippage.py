"""Slippage Tracking (T5.08).

Track actual vs expected slippage to detect execution quality issues.
Provides per-symbol slippage tracking, excessive slippage alerts,
and structured logging for dashboard display.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class SlippageAlertSeverity(str, Enum):
    """Alert severity levels for slippage tracking."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SlippageEvent:
    """Record of a slippage event.

    Attributes:
        symbol: Trading symbol.
        side: Trade side (buy/sell).
        expected_price: Expected/target price.
        actual_price: Actual fill price.
        slippage_bps: Calculated slippage in basis points.
        timestamp: Event timestamp.
    """

    symbol: str
    side: str
    expected_price: float
    actual_price: float
    slippage_bps: float
    timestamp: datetime


@dataclass
class SlippageAlert:
    """Alert generated from slippage tracking.

    Attributes:
        timestamp: When the alert was generated.
        severity: Alert severity level.
        message: Human-readable alert message.
        avg_slippage_bps: Average slippage at time of alert.
        symbol: Optional symbol filter (None = all symbols).
        details: Additional context details.
    """

    timestamp: datetime
    severity: SlippageAlertSeverity
    message: str
    avg_slippage_bps: float
    symbol: str | None = None
    details: dict[str, object] | None = None


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
        alert_level: Current alert level string.
    """

    avg_slippage_bps: float
    max_slippage_bps: float
    expected_slippage_bps: float
    ratio_to_expected: float
    sample_count: int
    alert_triggered: bool
    alert_level: str = "none"

    @property
    def slippage_display(self) -> str:
        """Return average slippage as formatted string."""
        return f"{self.avg_slippage_bps:.1f} bps"

    @property
    def target_display(self) -> str:
        """Return target slippage as formatted string."""
        return f"<{self.expected_slippage_bps:.0f} bps"

    @property
    def status_emoji(self) -> str:
        """Return status emoji based on alert level."""
        return {
            "none": "[green]OK[/green]",
            "warning": "[yellow]WARN[/yellow]",
            "critical": "[red]CRIT[/red]",
        }.get(self.alert_level, "[green]OK[/green]")


@dataclass
class SlippageTracker:
    """Track fill slippage vs expectations.

    Monitors execution quality by comparing actual fill prices
    to expected prices and calculating slippage metrics.
    Provides per-symbol tracking and alert history for dashboard display.
    """

    config: SlippageConfig
    _slippage_history: deque[float] = field(default_factory=deque, init=False)
    _event_history: deque[SlippageEvent] = field(default_factory=deque, init=False)
    _symbol_slippage: dict[str, deque[float]] = field(default_factory=dict, init=False)
    _alert_history: deque[SlippageAlert] = field(default_factory=deque, init=False)
    _stats_callbacks: list[Callable[[SlippageStats], None]] = field(
        default_factory=list, init=False
    )
    _last_alert_triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._slippage_history = deque(maxlen=self.config.window_size)
        self._event_history = deque(maxlen=self.config.window_size)
        self._symbol_slippage = {}
        self._alert_history = deque(maxlen=100)  # Keep last 100 alerts
        self._stats_callbacks = []
        self._last_alert_triggered = False

    def register_stats_callback(
        self, callback: Callable[[SlippageStats], None]
    ) -> None:
        """Register a callback to receive stats updates.

        Args:
            callback: Function to call with updated stats.
        """
        self._stats_callbacks.append(callback)

    def record_fill(
        self,
        expected_price: float,
        actual_price: float,
        side: str,
        symbol: str = "UNKNOWN",
        timestamp: datetime | None = None,
    ) -> float:
        """Record a fill and return slippage in bps with structured logging.

        For buys: positive slippage means worse execution (paid more).
        For sells: positive slippage means worse execution (received less).

        Args:
            expected_price: The expected/target price.
            actual_price: The actual fill price.
            side: "buy" or "sell".
            symbol: Trading symbol for per-symbol tracking.
            timestamp: Event timestamp (defaults to now).

        Returns:
            Slippage in basis points (positive = adverse).
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if expected_price <= 0:
            logger.warning("invalid_expected_price", price=expected_price, symbol=symbol)
            return 0.0

        # Calculate slippage in bps
        # For buy: (actual - expected) / expected * 10000
        # For sell: (expected - actual) / expected * 10000
        if side.lower() == "buy":
            slippage_bps = (actual_price - expected_price) / expected_price * 10000
        else:  # sell
            slippage_bps = (expected_price - actual_price) / expected_price * 10000

        # Record overall slippage
        self._slippage_history.append(slippage_bps)

        # Record event for history
        event = SlippageEvent(
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_bps=slippage_bps,
            timestamp=timestamp,
        )
        self._event_history.append(event)

        # Per-symbol tracking
        if symbol not in self._symbol_slippage:
            self._symbol_slippage[symbol] = deque(maxlen=self.config.window_size)
        self._symbol_slippage[symbol].append(slippage_bps)

        # Structured logging
        alert_threshold = (
            self.config.expected_slippage_bps * self.config.alert_threshold_multiplier
        )
        is_excessive = slippage_bps > alert_threshold

        log_data = {
            "symbol": symbol,
            "side": side,
            "expected_price": round(expected_price, 6),
            "actual_price": round(actual_price, 6),
            "slippage_bps": round(slippage_bps, 2),
            "expected_slippage_bps": self.config.expected_slippage_bps,
            "is_excessive": is_excessive,
            "timestamp": timestamp.isoformat(),
        }

        if is_excessive:
            logger.warning("slippage_excessive", **log_data)
        elif slippage_bps > self.config.expected_slippage_bps:
            logger.info("slippage_above_expected", **log_data)
        else:
            logger.debug("slippage_recorded", **log_data)

        # Emit stats and check for alerts
        self._emit_stats_and_check_alerts(symbol)

        return slippage_bps

    def _emit_stats_and_check_alerts(self, symbol: str | None = None) -> None:
        """Emit stats to callbacks and check for alert level changes.

        Args:
            symbol: Optional symbol for context.
        """
        stats = self.get_stats()

        # Notify callbacks
        for callback in self._stats_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error("stats_callback_error", error=str(e))

        # Check for alert state change
        if stats.alert_triggered != self._last_alert_triggered:
            self._generate_alert(stats, symbol)
            self._last_alert_triggered = stats.alert_triggered

    def _generate_alert(self, stats: SlippageStats, symbol: str | None = None) -> None:
        """Generate and log an alert based on current stats.

        Args:
            stats: Current slippage statistics.
            symbol: Optional symbol for context.
        """
        now = datetime.now(UTC)

        if stats.alert_triggered:
            severity = SlippageAlertSeverity.WARNING
            message = f"Excessive slippage: {stats.slippage_display} (threshold: {self.config.expected_slippage_bps * self.config.alert_threshold_multiplier:.0f} bps)"
        else:
            severity = SlippageAlertSeverity.INFO
            message = f"Slippage normalized: {stats.slippage_display}"

        alert = SlippageAlert(
            timestamp=now,
            severity=severity,
            message=message,
            avg_slippage_bps=stats.avg_slippage_bps,
            symbol=symbol,
            details={
                "max_slippage_bps": stats.max_slippage_bps,
                "expected_slippage_bps": stats.expected_slippage_bps,
                "ratio_to_expected": stats.ratio_to_expected,
                "sample_count": stats.sample_count,
            },
        )

        self._alert_history.append(alert)

        # Log with appropriate level
        log_func = {
            SlippageAlertSeverity.CRITICAL: logger.error,
            SlippageAlertSeverity.WARNING: logger.warning,
            SlippageAlertSeverity.INFO: logger.info,
        }[severity]

        log_func(
            "slippage_alert",
            severity=severity.value,
            message=message,
            avg_slippage_bps=round(stats.avg_slippage_bps, 2),
            max_slippage_bps=round(stats.max_slippage_bps, 2),
            sample_count=stats.sample_count,
            symbol=symbol,
        )

    def get_alert_history(self, limit: int = 10) -> list[SlippageAlert]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent SlippageAlert objects.
        """
        return list(self._alert_history)[-limit:]

    def get_symbol_stats(self, symbol: str) -> SlippageStats:
        """Get slippage statistics for a specific symbol.

        Args:
            symbol: Trading symbol to get stats for.

        Returns:
            SlippageStats for the symbol.
        """
        if symbol not in self._symbol_slippage or not self._symbol_slippage[symbol]:
            return SlippageStats(
                avg_slippage_bps=0.0,
                max_slippage_bps=0.0,
                expected_slippage_bps=self.config.expected_slippage_bps,
                ratio_to_expected=0.0,
                sample_count=0,
                alert_triggered=False,
                alert_level="none",
            )

        symbol_history = self._symbol_slippage[symbol]
        avg_slippage = sum(symbol_history) / len(symbol_history)
        max_slippage = max(symbol_history)

        # Calculate ratio to expected
        if self.config.expected_slippage_bps > 0:
            ratio = avg_slippage / self.config.expected_slippage_bps
        else:
            ratio = 0.0

        # Check if alert should be triggered
        alert_threshold = (
            self.config.expected_slippage_bps * self.config.alert_threshold_multiplier
        )
        alert_triggered = avg_slippage > alert_threshold

        alert_level = "warning" if alert_triggered else "none"

        return SlippageStats(
            avg_slippage_bps=avg_slippage,
            max_slippage_bps=max_slippage,
            expected_slippage_bps=self.config.expected_slippage_bps,
            ratio_to_expected=ratio,
            sample_count=len(symbol_history),
            alert_triggered=alert_triggered,
            alert_level=alert_level,
        )

    def get_all_symbol_stats(self) -> dict[str, SlippageStats]:
        """Get slippage statistics for all tracked symbols.

        Returns:
            Dictionary mapping symbol to SlippageStats.
        """
        return {symbol: self.get_symbol_stats(symbol) for symbol in self._symbol_slippage}

    def emit_symbol_report(self) -> dict[str, SlippageStats]:
        """Emit a per-symbol slippage report with structured logging.

        Returns:
            Dictionary mapping symbol to SlippageStats.
        """
        all_stats = self.get_all_symbol_stats()
        overall_stats = self.get_stats()

        logger.info(
            "slippage_symbol_report",
            total_symbols=len(all_stats),
            overall_avg_slippage_bps=round(overall_stats.avg_slippage_bps, 2),
            overall_max_slippage_bps=round(overall_stats.max_slippage_bps, 2),
            sample_count=overall_stats.sample_count,
            alert_triggered=overall_stats.alert_triggered,
        )

        # Log each symbol's stats
        for symbol, stats in all_stats.items():
            log_func = logger.warning if stats.alert_triggered else logger.info
            log_func(
                "slippage_symbol_stats",
                symbol=symbol,
                avg_slippage_bps=round(stats.avg_slippage_bps, 2),
                max_slippage_bps=round(stats.max_slippage_bps, 2),
                ratio_to_expected=round(stats.ratio_to_expected, 2),
                sample_count=stats.sample_count,
                alert_triggered=stats.alert_triggered,
            )

        return all_stats

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
                alert_level="none",
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

        alert_level = "warning" if alert_triggered else "none"

        return SlippageStats(
            avg_slippage_bps=avg_slippage,
            max_slippage_bps=max_slippage,
            expected_slippage_bps=self.config.expected_slippage_bps,
            ratio_to_expected=ratio,
            sample_count=len(self._slippage_history),
            alert_triggered=alert_triggered,
            alert_level=alert_level,
        )

    def should_alert(self) -> bool:
        """Check if slippage alert should be triggered.

        Returns:
            True if average slippage exceeds threshold.
        """
        return self.get_stats().alert_triggered
