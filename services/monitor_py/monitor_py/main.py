"""Main entry point for monitor service with Rich dashboard.

Provides a real-time terminal dashboard displaying system health,
alerts, and performance metrics with auto-refresh capability.
"""

import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from monitor_py import (
    BehaviorComparator,
    BehaviorConfig,
    DrawdownConfig,
    DrawdownMonitor,
    FeedHealthConfig,
    FeedHealthMonitor,
    FillRateConfig,
    FillRateTracker,
    MetricsExporter,
    RollbackConfig,
    RollbackManager,
    SlippageConfig,
    SlippageTracker,
    TurnoverAnalyzer,
    TurnoverConfig,
)

logger = structlog.get_logger(__name__)

# Configure structured logging for dashboard mode
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


@dataclass
class AlertRecord:
    """Record of an alert for display.

    Attributes:
        timestamp: When the alert occurred.
        level: Alert level (info, warning, critical).
        message: Alert message.
        source: Source component.
    """

    timestamp: datetime
    level: str
    message: str
    source: str

    @property
    def level_icon(self) -> str:
        """Return icon for alert level."""
        return {
            "info": "[cyan]i[/cyan]",
            "warning": "[yellow]![/yellow]",
            "critical": "[red]X[/red]",
        }.get(self.level, "[white]?[/white]")

    @property
    def time_str(self) -> str:
        """Return formatted time string."""
        return self.timestamp.strftime("%H:%M:%S")


@dataclass
class StrategyPerformance:
    """Performance data for a trading strategy.

    Attributes:
        name: Strategy name.
        pnl: Profit and loss.
        trades: Number of trades.
        win_rate: Win rate as percentage.
    """

    name: str
    pnl: float
    trades: int
    win_rate: float

    @property
    def pnl_display(self) -> str:
        """Return formatted P&L display."""
        if self.pnl >= 0:
            return f"[green]+${self.pnl:,.2f}[/green]"
        return f"[red]-${abs(self.pnl):,.2f}[/red]"

    @property
    def win_rate_display(self) -> str:
        """Return formatted win rate display."""
        return f"{self.win_rate:.1f}%"


@dataclass
class MonitorDashboard:
    """Real-time monitoring dashboard.

    Displays system health metrics, recent alerts, and strategy
    performance in a terminal UI with auto-refresh.
    """

    refresh_interval: float = 5.0
    initial_equity: float = 100000.0
    _console: Console = field(default_factory=Console, init=False)
    _alerts: deque[AlertRecord] = field(default_factory=deque, init=False)
    _strategies: list[StrategyPerformance] = field(default_factory=list, init=False)
    _running: bool = field(default=True, init=False)

    # Monitoring components
    _fill_tracker: FillRateTracker | None = field(default=None, init=False)
    _slippage_tracker: SlippageTracker | None = field(default=None, init=False)
    _drawdown_monitor: DrawdownMonitor | None = field(default=None, init=False)
    _turnover_analyzer: TurnoverAnalyzer | None = field(default=None, init=False)
    _behavior_comparator: BehaviorComparator | None = field(default=None, init=False)
    _feed_monitor: FeedHealthMonitor | None = field(default=None, init=False)
    _metrics_exporter: MetricsExporter | None = field(default=None, init=False)
    _rollback_manager: RollbackManager | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize dashboard components."""
        self._alerts = deque(maxlen=50)
        self._strategies = []
        self._running = True

        # Initialize monitoring components
        self._fill_tracker = FillRateTracker(FillRateConfig())
        self._slippage_tracker = SlippageTracker(SlippageConfig())
        self._drawdown_monitor = DrawdownMonitor(
            DrawdownConfig(), initial_equity=self.initial_equity
        )
        self._turnover_analyzer = TurnoverAnalyzer(
            TurnoverConfig(), portfolio_value=self.initial_equity
        )
        self._behavior_comparator = BehaviorComparator(BehaviorConfig())
        self._feed_monitor = FeedHealthMonitor(FeedHealthConfig())
        self._metrics_exporter = MetricsExporter()
        self._rollback_manager = RollbackManager(RollbackConfig())

        # Add demo strategies
        self._strategies = [
            StrategyPerformance("TrendFollowing_v1", 234.50, 18, 55.6),
            StrategyPerformance("MeanReversion_v2", 127.80, 24, 58.3),
            StrategyPerformance("Volatility_v1", -45.20, 8, 37.5),
        ]

        # Add initial info alert
        self._add_alert("info", "Monitor dashboard started", "system")

        logger.info("monitor_dashboard_initialized", initial_equity=self.initial_equity)

    def _add_alert(self, level: str, message: str, source: str) -> None:
        """Add an alert to the history.

        Args:
            level: Alert level.
            message: Alert message.
            source: Source component.
        """
        alert = AlertRecord(
            timestamp=datetime.now(UTC),
            level=level,
            message=message,
            source=source,
        )
        self._alerts.append(alert)

    def _make_header(self) -> Panel:
        """Create the dashboard header panel.

        Returns:
            Panel with header content.
        """
        now = datetime.now(UTC).strftime("%H:%M:%S")
        title = Text()
        title.append("autoBot Monitor", style="bold white")
        title.append(f"\nLast Update: {now}", style="dim")

        return Panel(
            title,
            title="",
            border_style="blue",
            padding=(0, 2),
        )

    def _make_health_table(self) -> Table:
        """Create the system health metrics table.

        Returns:
            Table with health metrics.
        """
        table = Table(title="System Health", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan", width=18)
        table.add_column("Status", width=10)
        table.add_column("Value", width=30)

        # Fill Rate
        if self._fill_tracker:
            fill_stats = self._fill_tracker.get_stats()
            fill_status = fill_stats.status_emoji
            fill_value = f"{fill_stats.fill_rate_pct} (target: {fill_stats.target_pct})"
            table.add_row("Fill Rate", fill_status, fill_value)
        else:
            table.add_row("Fill Rate", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        # Slippage
        if self._slippage_tracker:
            slip_stats = self._slippage_tracker.get_stats()
            slip_status = slip_stats.status_emoji
            slip_value = f"{slip_stats.slippage_display} (target: {slip_stats.target_display})"
            table.add_row("Slippage", slip_status, slip_value)
        else:
            table.add_row("Slippage", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        # Drawdown
        if self._drawdown_monitor:
            dd_status = self._drawdown_monitor.get_status()
            dd_level = dd_status.alert_level
            dd_status_str = {
                "none": "[green]OK[/green]",
                "warning": "[yellow]WARN[/yellow]",
                "critical": "[red]CRIT[/red]",
                "kill_switch": "[red bold]HALT[/red bold]",
            }.get(dd_level, "[green]OK[/green]")
            dd_value = f"{dd_status.current_drawdown * -100:.1f}% (limit: -10%)"
            table.add_row("Drawdown", dd_status_str, dd_value)
        else:
            table.add_row("Drawdown", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        # Daily Turnover
        if self._turnover_analyzer:
            turn_stats = self._turnover_analyzer.get_stats()
            turn_status = turn_stats.status_emoji
            turn_value = f"{turn_stats.daily_turnover_pct} (target: {turn_stats.target_pct})"
            table.add_row("Daily Turnover", turn_status, turn_value)
        else:
            table.add_row("Daily Turnover", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        # Signal Alignment
        if self._behavior_comparator:
            beh_stats = self._behavior_comparator.get_stats()
            beh_status = beh_stats.status_emoji
            beh_value = f"{beh_stats.alignment_rate_pct} (target: {beh_stats.target_pct})"
            table.add_row("Signal Alignment", beh_status, beh_value)
        else:
            table.add_row("Signal Alignment", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        # Feed Latency
        if self._feed_monitor:
            feed_status = self._feed_monitor.get_status()
            feed_level = {
                "healthy": "[green]OK[/green]",
                "degraded": "[yellow]WARN[/yellow]",
                "disconnected": "[red]CRIT[/red]",
            }.get(feed_status.status.value, "[green]OK[/green]")
            if feed_status.current_lag_seconds == float("inf"):
                feed_value = "No data (target: <500ms)"
            else:
                feed_value = f"{feed_status.current_lag_seconds * 1000:.0f}ms (target: <500ms)"
            table.add_row("Feed Latency", feed_level, feed_value)
        else:
            table.add_row("Feed Latency", "[dim]N/A[/dim]", "[dim]No data[/dim]")

        return table

    def _make_alerts_panel(self) -> Panel:
        """Create the recent alerts panel.

        Returns:
            Panel with recent alerts.
        """
        lines = []

        # Get last 5 alerts
        recent_alerts = list(self._alerts)[-5:]

        if not recent_alerts:
            lines.append("[dim]No alerts[/dim]")
        else:
            for alert in reversed(recent_alerts):
                line = f"[{alert.time_str}] {alert.level_icon} {alert.level.upper()}: {alert.message}"
                lines.append(line)

        content = "\n".join(lines)

        return Panel(
            content,
            title="Recent Alerts",
            border_style="yellow",
        )

    def _make_performance_table(self) -> Table:
        """Create the strategy performance table.

        Returns:
            Table with strategy performance.
        """
        table = Table(title="Performance Today", show_header=True, header_style="bold")
        table.add_column("Strategy", style="cyan", width=21)
        table.add_column("P&L", width=11)
        table.add_column("Trades", width=11)
        table.add_column("Win Rate", width=11)

        for strategy in self._strategies:
            table.add_row(
                strategy.name,
                strategy.pnl_display,
                str(strategy.trades),
                strategy.win_rate_display,
            )

        return table

    def _make_layout(self) -> Layout:
        """Create the dashboard layout.

        Returns:
            Complete dashboard layout.
        """
        layout = Layout()

        # Main structure
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
        )

        # Body split
        layout["body"].split_column(
            Layout(name="health", size=12),
            Layout(name="alerts", size=9),
            Layout(name="performance", size=8),
        )

        # Set content
        layout["header"].update(self._make_header())
        layout["health"].update(Panel(self._make_health_table(), border_style="green"))
        layout["alerts"].update(self._make_alerts_panel())
        layout["performance"].update(
            Panel(self._make_performance_table(), border_style="blue")
        )

        return layout

    def update_fill_rate(
        self,
        order_id: str,
        symbol: str,
        requested_qty: float,
        filled_qty: float,
        fill_time_ms: float,
    ) -> None:
        """Update fill rate tracker with new fill.

        Args:
            order_id: Unique order identifier.
            symbol: Trading symbol.
            requested_qty: Requested quantity.
            filled_qty: Actual filled quantity.
            fill_time_ms: Time to fill in milliseconds.
        """
        if self._fill_tracker:
            event = self._fill_tracker.record_fill(
                order_id=order_id,
                symbol=symbol,
                requested_qty=requested_qty,
                filled_qty=filled_qty,
                fill_time_ms=fill_time_ms,
            )
            if not event.is_full_fill:
                self._add_alert(
                    "warning",
                    f"Partial fill: {symbol} {event.fill_ratio:.0%}",
                    "fill_rate",
                )

    def update_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
    ) -> None:
        """Update slippage tracker with new fill.

        Args:
            symbol: Trading symbol.
            side: Trade side (buy/sell).
            expected_price: Expected price.
            actual_price: Actual fill price.
        """
        if self._slippage_tracker:
            slippage_bps = self._slippage_tracker.record_fill(
                expected_price=expected_price,
                actual_price=actual_price,
                side=side,
                symbol=symbol,
            )
            if slippage_bps > 4.0:  # 2x expected
                self._add_alert(
                    "warning",
                    f"Excessive slippage: {symbol} {slippage_bps:.1f}bps",
                    "slippage",
                )

    def update_equity(self, equity: float) -> None:
        """Update equity for drawdown monitoring.

        Args:
            equity: Current portfolio equity.
        """
        if self._drawdown_monitor:
            status = self._drawdown_monitor.update(equity)
            if status.alert_level in ("warning", "critical", "kill_switch"):
                self._add_alert(
                    "warning" if status.alert_level == "warning" else "critical",
                    f"Drawdown: {status.current_drawdown * -100:.1f}%",
                    "drawdown",
                )

    def update_turnover(
        self,
        portfolio_value: float,
        trade_volume: float,
    ) -> None:
        """Update turnover analyzer.

        Args:
            portfolio_value: Current portfolio value.
            trade_volume: Today's trade volume.
        """
        if self._turnover_analyzer:
            self._turnover_analyzer.update_portfolio_value(portfolio_value)
            stats = self._turnover_analyzer.get_stats()
            if stats.alert_level in ("warning", "critical"):
                self._add_alert(
                    stats.alert_level,
                    f"Daily turnover: {stats.daily_turnover_pct}",
                    "turnover",
                )

    def update_strategy_performance(
        self,
        name: str,
        pnl: float,
        trades: int,
        win_rate: float,
    ) -> None:
        """Update strategy performance data.

        Args:
            name: Strategy name.
            pnl: Profit and loss.
            trades: Number of trades.
            win_rate: Win rate percentage.
        """
        # Find and update or add strategy
        for i, strategy in enumerate(self._strategies):
            if strategy.name == name:
                self._strategies[i] = StrategyPerformance(name, pnl, trades, win_rate)
                return

        self._strategies.append(StrategyPerformance(name, pnl, trades, win_rate))

    def add_alert(self, level: str, message: str, source: str = "system") -> None:
        """Add an alert to the dashboard.

        Args:
            level: Alert level (info, warning, critical).
            message: Alert message.
            source: Source component.
        """
        self._add_alert(level, message, source)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        logger.info("monitor_dashboard_stopped")

    def run(self) -> None:
        """Run the dashboard with live updates."""
        logger.info("monitor_dashboard_running", refresh_interval=self.refresh_interval)

        # Set up signal handler for graceful shutdown
        def signal_handler(signum: int, frame: Any) -> None:
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        with Live(
            self._make_layout(),
            console=self._console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while self._running:
                live.update(self._make_layout())
                time.sleep(self.refresh_interval)

        self._console.print("[bold green]Dashboard stopped.[/bold green]")


def create_monitoring_stack(
    initial_equity: float = 100000.0,
) -> dict[str, Any]:
    """Create a complete monitoring stack.

    Args:
        initial_equity: Initial portfolio equity.

    Returns:
        Dictionary with all monitoring components.
    """
    return {
        "feed_monitor": FeedHealthMonitor(FeedHealthConfig()),
        "slippage_tracker": SlippageTracker(SlippageConfig()),
        "drawdown_monitor": DrawdownMonitor(DrawdownConfig(), initial_equity=initial_equity),
        "fill_tracker": FillRateTracker(FillRateConfig()),
        "turnover_analyzer": TurnoverAnalyzer(TurnoverConfig(), portfolio_value=initial_equity),
        "behavior_comparator": BehaviorComparator(BehaviorConfig()),
        "metrics_exporter": MetricsExporter(),
        "rollback_manager": RollbackManager(RollbackConfig()),
    }


def main() -> None:
    """Main entry point."""
    # Check if we should run dashboard or basic mode
    if "--dashboard" in sys.argv or "-d" in sys.argv:
        logger.info("Starting monitor dashboard...")
        dashboard = MonitorDashboard()

        # Add some demo alerts
        dashboard.add_alert("warning", "Daily turnover approaching limit (185%)", "turnover")
        dashboard.add_alert("info", "Strategy TrendFollowing_v1 promoted to PAPER", "promotion")
        dashboard.add_alert("info", "All systems nominal", "system")

        dashboard.run()
    else:
        # Basic mode without dashboard
        logger.info("monitor_py service started")
        stack = create_monitoring_stack()
        logger.info(
            "monitoring_stack_created",
            components=list(stack.keys()),
        )


if __name__ == "__main__":
    main()
