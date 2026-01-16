"""Shadow vs Backtest Behavior Comparison.

Validate shadow trading matches backtest behavior.
Provides signal alignment logging, drift detection alerts,
and divergence reports for dashboard display.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class BehaviorAlertSeverity(str, Enum):
    """Alert severity levels for behavior comparison."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BehaviorAlert:
    """Alert generated from behavior comparison.

    Attributes:
        timestamp: When the alert was generated.
        severity: Alert severity level.
        message: Human-readable alert message.
        drift_score: Drift score at time of alert.
        signal_alignment_rate: Signal alignment rate at time of alert.
        details: Additional context details.
    """

    timestamp: datetime
    severity: BehaviorAlertSeverity
    message: str
    drift_score: float
    signal_alignment_rate: float
    details: dict[str, object] | None = None


@dataclass
class BehaviorConfig:
    """Configuration for behavior comparison.

    Attributes:
        signal_alignment_threshold: Required signal alignment rate (0.95 = 95%).
        timing_tolerance_ms: Acceptable timing drift in milliseconds.
        drift_warning_threshold: Drift above this triggers warning.
        drift_critical_threshold: Drift above this triggers critical alert.
        window_size: Number of signals to track in rolling window.
    """

    signal_alignment_threshold: float = 0.95
    timing_tolerance_ms: float = 100.0
    drift_warning_threshold: float = 0.05
    drift_critical_threshold: float = 0.10
    window_size: int = 1000


@dataclass
class SignalRecord:
    """Record of a trading signal.

    Attributes:
        signal_id: Unique signal identifier.
        symbol: Trading symbol.
        signal_type: Signal type (buy/sell/hold).
        strength: Signal strength (0.0 to 1.0).
        timestamp: Signal timestamp.
        source: Signal source (backtest/shadow).
    """

    signal_id: str
    symbol: str
    signal_type: str
    strength: float
    timestamp: datetime
    source: str


@dataclass
class ExecutionRecord:
    """Record of a trade execution.

    Attributes:
        execution_id: Unique execution identifier.
        signal_id: Related signal identifier.
        symbol: Trading symbol.
        side: Trade side (buy/sell).
        quantity: Trade quantity.
        price: Execution price.
        timestamp: Execution timestamp.
        source: Execution source (backtest/shadow).
    """

    execution_id: str
    signal_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    source: str

    @property
    def notional_value(self) -> float:
        """Calculate notional value of execution."""
        return self.quantity * self.price


@dataclass
class BehaviorStats:
    """Behavior comparison statistics.

    Attributes:
        total_signals: Total signals compared.
        aligned_signals: Number of aligned signals.
        signal_alignment_rate: Signal alignment rate as decimal.
        total_executions: Total executions compared.
        aligned_executions: Number of aligned executions.
        execution_alignment_rate: Execution alignment rate as decimal.
        avg_timing_drift_ms: Average timing drift in milliseconds.
        max_timing_drift_ms: Maximum timing drift in milliseconds.
        timing_within_tolerance: Whether timing is within tolerance.
        drift_score: Overall drift score (0.0 = perfect alignment).
        alert_level: Current alert level.
        target_alignment_rate: Target signal alignment rate.
    """

    total_signals: int
    aligned_signals: int
    signal_alignment_rate: float
    total_executions: int
    aligned_executions: int
    execution_alignment_rate: float
    avg_timing_drift_ms: float
    max_timing_drift_ms: float
    timing_within_tolerance: bool
    drift_score: float
    alert_level: str
    target_alignment_rate: float = 0.95

    @property
    def alignment_rate_pct(self) -> str:
        """Return signal alignment rate as formatted percentage string."""
        return f"{self.signal_alignment_rate * 100:.1f}%"

    @property
    def target_pct(self) -> str:
        """Return target alignment rate as formatted percentage string."""
        return f"{self.target_alignment_rate * 100:.0f}%"

    @property
    def status_emoji(self) -> str:
        """Return status emoji based on alert level."""
        return {
            "none": "[green]OK[/green]",
            "warning": "[yellow]WARN[/yellow]",
            "critical": "[red]CRIT[/red]",
        }.get(self.alert_level, "[green]OK[/green]")


@dataclass
class BehaviorComparator:
    """Compare shadow and backtest behavior for validation.

    Monitors signal alignment, execution timing, and drift
    to ensure shadow trading matches backtest behavior.
    Provides structured logging and alert history for dashboard display.
    """

    config: BehaviorConfig
    _backtest_signals: dict[str, SignalRecord] = field(default_factory=dict, init=False)
    _shadow_signals: dict[str, SignalRecord] = field(default_factory=dict, init=False)
    _backtest_executions: dict[str, ExecutionRecord] = field(default_factory=dict, init=False)
    _shadow_executions: dict[str, ExecutionRecord] = field(default_factory=dict, init=False)
    _signal_order: deque[str] = field(default_factory=deque, init=False)
    _execution_order: deque[str] = field(default_factory=deque, init=False)
    _alert_history: deque[BehaviorAlert] = field(default_factory=deque, init=False)
    _stats_callbacks: list[Callable[[BehaviorStats], None]] = field(
        default_factory=list, init=False
    )
    _last_alert_level: str = field(default="none", init=False)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        self._backtest_signals = {}
        self._shadow_signals = {}
        self._backtest_executions = {}
        self._shadow_executions = {}
        self._signal_order = deque(maxlen=self.config.window_size)
        self._execution_order = deque(maxlen=self.config.window_size)
        self._alert_history = deque(maxlen=100)  # Keep last 100 alerts
        self._stats_callbacks = []
        self._last_alert_level = "none"

    def register_stats_callback(
        self, callback: Callable[[BehaviorStats], None]
    ) -> None:
        """Register a callback to receive stats updates.

        Args:
            callback: Function to call with updated stats.
        """
        self._stats_callbacks.append(callback)

    def record_backtest_signal(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        strength: float,
        timestamp: datetime | None = None,
    ) -> SignalRecord:
        """Record a backtest signal.

        Args:
            signal_id: Unique signal identifier.
            symbol: Trading symbol.
            signal_type: Signal type (buy/sell/hold).
            strength: Signal strength.
            timestamp: Signal timestamp (defaults to now).

        Returns:
            SignalRecord with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        record = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=timestamp,
            source="backtest",
        )

        self._backtest_signals[signal_id] = record
        if signal_id not in self._signal_order:
            self._signal_order.append(signal_id)

        logger.debug("backtest_signal_recorded", signal_id=signal_id, signal_type=signal_type)
        return record

    def record_shadow_signal(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        strength: float,
        timestamp: datetime | None = None,
    ) -> SignalRecord:
        """Record a shadow signal with alignment logging.

        Args:
            signal_id: Unique signal identifier.
            symbol: Trading symbol.
            signal_type: Signal type (buy/sell/hold).
            strength: Signal strength.
            timestamp: Signal timestamp (defaults to now).

        Returns:
            SignalRecord with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        record = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=timestamp,
            source="shadow",
        )

        self._shadow_signals[signal_id] = record
        if signal_id not in self._signal_order:
            self._signal_order.append(signal_id)

        # Log shadow signal with alignment check
        backtest_signal = self._backtest_signals.get(signal_id)
        if backtest_signal:
            is_aligned = backtest_signal.signal_type == signal_type
            strength_diff = abs(backtest_signal.strength - strength)

            if is_aligned:
                logger.info(
                    "shadow_signal_aligned",
                    signal_id=signal_id,
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    strength_diff=round(strength_diff, 4),
                    timestamp=timestamp.isoformat(),
                )
            else:
                logger.warning(
                    "shadow_signal_diverged",
                    signal_id=signal_id,
                    symbol=symbol,
                    shadow_type=signal_type,
                    backtest_type=backtest_signal.signal_type,
                    shadow_strength=strength,
                    backtest_strength=backtest_signal.strength,
                    timestamp=timestamp.isoformat(),
                )
        else:
            logger.debug(
                "shadow_signal_recorded",
                signal_id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                timestamp=timestamp.isoformat(),
            )

        # Check for drift and emit stats
        self._emit_stats_and_check_alerts()

        return record

    def _emit_stats_and_check_alerts(self) -> None:
        """Emit stats to callbacks and check for alert level changes."""
        stats = self.get_stats()

        # Notify callbacks
        for callback in self._stats_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error("stats_callback_error", error=str(e))

        # Check for alert level change
        if stats.alert_level != self._last_alert_level:
            self._generate_alert(stats)
            self._last_alert_level = stats.alert_level

    def _generate_alert(self, stats: BehaviorStats) -> None:
        """Generate and log an alert based on current stats.

        Args:
            stats: Current behavior statistics.
        """
        now = datetime.now(UTC)

        if stats.alert_level == "critical":
            severity = BehaviorAlertSeverity.CRITICAL
            message = f"Behavior drift critical: {stats.drift_score:.1%} (alignment: {stats.alignment_rate_pct})"
        elif stats.alert_level == "warning":
            severity = BehaviorAlertSeverity.WARNING
            message = f"Behavior drift warning: {stats.drift_score:.1%} (alignment: {stats.alignment_rate_pct})"
        else:
            severity = BehaviorAlertSeverity.INFO
            message = f"Behavior alignment recovered: {stats.alignment_rate_pct}"

        alert = BehaviorAlert(
            timestamp=now,
            severity=severity,
            message=message,
            drift_score=stats.drift_score,
            signal_alignment_rate=stats.signal_alignment_rate,
            details={
                "total_signals": stats.total_signals,
                "aligned_signals": stats.aligned_signals,
                "avg_timing_drift_ms": stats.avg_timing_drift_ms,
                "timing_within_tolerance": stats.timing_within_tolerance,
            },
        )

        self._alert_history.append(alert)

        # Log with appropriate level
        log_func = {
            BehaviorAlertSeverity.CRITICAL: logger.error,
            BehaviorAlertSeverity.WARNING: logger.warning,
            BehaviorAlertSeverity.INFO: logger.info,
        }[severity]

        log_func(
            "behavior_drift_alert",
            severity=severity.value,
            message=message,
            drift_score=round(stats.drift_score, 4),
            signal_alignment_rate=round(stats.signal_alignment_rate, 4),
            alignment_rate_pct=stats.alignment_rate_pct,
            total_signals=stats.total_signals,
        )

    def get_alert_history(self, limit: int = 10) -> list[BehaviorAlert]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent BehaviorAlert objects.
        """
        return list(self._alert_history)[-limit:]

    def record_backtest_execution(
        self,
        execution_id: str,
        signal_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
    ) -> ExecutionRecord:
        """Record a backtest execution.

        Args:
            execution_id: Unique execution identifier.
            signal_id: Related signal identifier.
            symbol: Trading symbol.
            side: Trade side (buy/sell).
            quantity: Trade quantity.
            price: Execution price.
            timestamp: Execution timestamp (defaults to now).

        Returns:
            ExecutionRecord with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        record = ExecutionRecord(
            execution_id=execution_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            source="backtest",
        )

        self._backtest_executions[execution_id] = record
        if execution_id not in self._execution_order:
            self._execution_order.append(execution_id)

        logger.debug("backtest_execution_recorded", execution_id=execution_id, side=side)
        return record

    def record_shadow_execution(
        self,
        execution_id: str,
        signal_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
    ) -> ExecutionRecord:
        """Record a shadow execution.

        Args:
            execution_id: Unique execution identifier.
            signal_id: Related signal identifier.
            symbol: Trading symbol.
            side: Trade side (buy/sell).
            quantity: Trade quantity.
            price: Execution price.
            timestamp: Execution timestamp (defaults to now).

        Returns:
            ExecutionRecord with recorded data.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        record = ExecutionRecord(
            execution_id=execution_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            source="shadow",
        )

        self._shadow_executions[execution_id] = record
        if execution_id not in self._execution_order:
            self._execution_order.append(execution_id)

        logger.debug("shadow_execution_recorded", execution_id=execution_id, side=side)
        return record

    def get_stats(self, symbol: str | None = None) -> BehaviorStats:
        """Get current behavior comparison statistics.

        Args:
            symbol: Optional symbol to filter stats.

        Returns:
            BehaviorStats with current metrics.
        """
        # Get signals in window (limited by window_size)
        signal_ids = list(self._signal_order)

        # Filter by symbol if specified
        if symbol:
            signal_ids = [
                sid for sid in signal_ids
                if (sid in self._backtest_signals and self._backtest_signals[sid].symbol == symbol)
                or (sid in self._shadow_signals and self._shadow_signals[sid].symbol == symbol)
            ]

        # Count aligned signals
        aligned_signals = 0
        for signal_id in signal_ids:
            backtest = self._backtest_signals.get(signal_id)
            shadow = self._shadow_signals.get(signal_id)
            if backtest and shadow and backtest.signal_type == shadow.signal_type:
                aligned_signals += 1

        total_signals = len(signal_ids)
        signal_alignment_rate = aligned_signals / total_signals if total_signals > 0 else 0.0

        # Calculate execution alignment and timing drift
        execution_ids = list(self._execution_order)
        if symbol:
            execution_ids = [
                eid for eid in execution_ids
                if (
                    eid in self._backtest_executions
                    and self._backtest_executions[eid].symbol == symbol
                ) or (
                    eid in self._shadow_executions
                    and self._shadow_executions[eid].symbol == symbol
                )
            ]

        aligned_executions = 0
        timing_drifts: list[float] = []

        for exec_id in execution_ids:
            backtest_exec = self._backtest_executions.get(exec_id)
            shadow_exec = self._shadow_executions.get(exec_id)
            if backtest_exec and shadow_exec:
                if backtest_exec.side == shadow_exec.side:
                    aligned_executions += 1
                # Calculate timing drift
                drift_ms = abs(
                    (shadow_exec.timestamp - backtest_exec.timestamp).total_seconds() * 1000
                )
                timing_drifts.append(drift_ms)

        total_executions = len(execution_ids)
        exec_rate = aligned_executions / total_executions if total_executions > 0 else 0.0
        execution_alignment_rate = exec_rate

        # Calculate timing metrics
        avg_timing_drift_ms = sum(timing_drifts) / len(timing_drifts) if timing_drifts else 0.0
        max_timing_drift_ms = max(timing_drifts) if timing_drifts else 0.0
        timing_within_tolerance = avg_timing_drift_ms <= self.config.timing_tolerance_ms

        # Calculate drift score (1 - alignment rate)
        drift_score = 1.0 - signal_alignment_rate

        # Determine alert level
        alert_level = self._determine_alert_level(drift_score)

        # Return early if no data at all
        if not signal_ids and not execution_ids:
            return BehaviorStats(
                total_signals=0,
                aligned_signals=0,
                signal_alignment_rate=0.0,
                total_executions=0,
                aligned_executions=0,
                execution_alignment_rate=0.0,
                avg_timing_drift_ms=0.0,
                max_timing_drift_ms=0.0,
                timing_within_tolerance=True,
                drift_score=0.0,
                alert_level="none",
                target_alignment_rate=self.config.signal_alignment_threshold,
            )

        return BehaviorStats(
            total_signals=total_signals,
            aligned_signals=aligned_signals,
            signal_alignment_rate=signal_alignment_rate,
            total_executions=total_executions,
            aligned_executions=aligned_executions,
            execution_alignment_rate=execution_alignment_rate,
            avg_timing_drift_ms=avg_timing_drift_ms,
            max_timing_drift_ms=max_timing_drift_ms,
            timing_within_tolerance=timing_within_tolerance,
            drift_score=drift_score,
            alert_level=alert_level,
            target_alignment_rate=self.config.signal_alignment_threshold,
        )

    def should_alert(self) -> bool:
        """Check if behavior drift alert should be triggered.

        Returns:
            True if drift exceeds warning threshold.
        """
        stats = self.get_stats()
        return stats.alert_level in ("warning", "critical")

    def get_divergence_report(self, symbol: str | None = None) -> list[dict[str, object]]:
        """Get detailed report of divergences between backtest and shadow.

        Args:
            symbol: Optional symbol to filter report.

        Returns:
            List of divergence records.
        """
        divergences = []
        signal_ids = list(self._signal_order)

        if symbol:
            signal_ids = [
                sid for sid in signal_ids
                if (sid in self._backtest_signals and self._backtest_signals[sid].symbol == symbol)
                or (sid in self._shadow_signals and self._shadow_signals[sid].symbol == symbol)
            ]

        for signal_id in signal_ids:
            backtest = self._backtest_signals.get(signal_id)
            shadow = self._shadow_signals.get(signal_id)

            if backtest and shadow and backtest.signal_type != shadow.signal_type:
                divergences.append({
                    "signal_id": signal_id,
                    "symbol": backtest.symbol,
                    "backtest_type": backtest.signal_type,
                    "shadow_type": shadow.signal_type,
                    "backtest_strength": backtest.strength,
                    "shadow_strength": shadow.strength,
                    "timestamp": backtest.timestamp.isoformat(),
                })
            elif backtest and not shadow:
                divergences.append({
                    "signal_id": signal_id,
                    "symbol": backtest.symbol,
                    "backtest_type": backtest.signal_type,
                    "shadow_type": None,
                    "reason": "missing_shadow_signal",
                    "timestamp": backtest.timestamp.isoformat(),
                })

        return divergences

    def emit_divergence_report(self, symbol: str | None = None) -> list[dict[str, object]]:
        """Emit a divergence report with structured logging.

        Args:
            symbol: Optional symbol to filter report.

        Returns:
            List of divergence records.
        """
        divergences = self.get_divergence_report(symbol)
        stats = self.get_stats(symbol)

        logger.info(
            "behavior_divergence_report",
            total_divergences=len(divergences),
            total_signals=stats.total_signals,
            aligned_signals=stats.aligned_signals,
            signal_alignment_rate=round(stats.signal_alignment_rate, 4),
            alignment_rate_pct=stats.alignment_rate_pct,
            drift_score=round(stats.drift_score, 4),
            alert_level=stats.alert_level,
            symbol=symbol,
        )

        # Log each divergence
        for div in divergences:
            logger.warning(
                "behavior_divergence",
                signal_id=div.get("signal_id"),
                symbol=div.get("symbol"),
                backtest_type=div.get("backtest_type"),
                shadow_type=div.get("shadow_type"),
                reason=div.get("reason", "signal_mismatch"),
                timestamp=div.get("timestamp"),
            )

        return divergences

    def reset(self) -> None:
        """Reset behavior comparison statistics."""
        self._backtest_signals.clear()
        self._shadow_signals.clear()
        self._backtest_executions.clear()
        self._shadow_executions.clear()
        self._signal_order.clear()
        self._execution_order.clear()
        logger.info("behavior_comparator_reset")

    def _determine_alert_level(self, drift_score: float) -> str:
        """Determine alert level based on drift score.

        Args:
            drift_score: Current drift score (0.0 = perfect).

        Returns:
            Alert level string.
        """
        if drift_score >= self.config.drift_critical_threshold:
            return "critical"
        elif drift_score >= self.config.drift_warning_threshold:
            return "warning"
        return "none"
