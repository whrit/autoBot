"""
autoBot Runner - Rich Console Dashboard.

Provides a live dashboard for monitoring shadow and paper trading execution
with real-time position display, signal history, and P&L summaries.
"""

from __future__ import annotations

import asyncio
import signal
import sys
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

from runner_py.fill_stream import FillConfirmationStream, FillStreamConfig
from runner_py.orchestrator import (
    AutonomousOrchestrator,
    OrchestratorConfig,
    StrategyState,
)
from runner_py.paper import PaperExecutor, PaperConfig, PaperFill
from runner_py.shadow import ShadowConfig, ShadowExecutor
from runner_py.types import ExecutionMode, MarketData, Signal, SignalDirection

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class DashboardState:
    """State for the dashboard display."""

    mode: ExecutionMode = ExecutionMode.PAPER
    strategies: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    recent_activity: list[dict[str, Any]] = field(default_factory=list)
    total_pnl_today: float = 0.0
    signal_count: int = 0
    fill_count: int = 0
    last_update: datetime | None = None

    def add_activity(self, activity: dict[str, Any]) -> None:
        """Add activity to the log, keeping last 20 entries."""
        activity["timestamp"] = datetime.now(UTC)
        self.recent_activity.insert(0, activity)
        self.recent_activity = self.recent_activity[:20]


class RunnerDashboard:
    """Rich console dashboard for the autoBot runner."""

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        alpaca_api_key: str | None = None,
        alpaca_api_secret: str | None = None,
    ) -> None:
        """Initialize the dashboard."""
        self.mode = mode
        self.state = DashboardState(mode=mode)
        self._running = False
        self._logger = logger.bind(component="dashboard")

        # Initialize executors based on mode
        self._shadow_executors: dict[str, ShadowExecutor] = {}
        self._paper_executor: PaperExecutor | None = None
        self._fill_stream: FillConfirmationStream | None = None
        self._orchestrator: AutonomousOrchestrator | None = None

        if alpaca_api_key and alpaca_api_secret:
            self._paper_executor = PaperExecutor(
                PaperConfig(
                    api_key=alpaca_api_key,
                    api_secret=alpaca_api_secret,
                )
            )
            self._fill_stream = FillConfirmationStream(
                FillStreamConfig(
                    api_key=alpaca_api_key,
                    api_secret=alpaca_api_secret,
                )
            )
            # Register fill callback
            self._fill_stream.register_callback(self._on_fill_received)

        self._logger.info(
            "dashboard_initialized",
            mode=mode.value,
            has_paper_executor=self._paper_executor is not None,
            has_fill_stream=self._fill_stream is not None,
        )

    def _on_fill_received(self, fill: PaperFill) -> None:
        """Handle fill notification from the stream."""
        self.state.fill_count += 1
        self.state.add_activity({
            "type": "fill",
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": fill.qty,
            "price": fill.filled_price,
            "order_id": fill.order_id,
        })

        self._logger.info(
            "fill_activity_logged",
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            price=fill.filled_price,
        )

    def add_strategy(
        self,
        name: str,
        state: StrategyState,
        symbols: list[str],
    ) -> None:
        """Add a strategy to track."""
        strategy = {
            "name": name,
            "state": state.value.upper(),
            "pnl_today": 0.0,
            "signal_count": 0,
            "symbols": symbols,
        }
        self.state.strategies.append(strategy)

        # Create shadow executor for this strategy
        self._shadow_executors[name] = ShadowExecutor(
            ShadowConfig(
                strategy_id=name,
                symbols=symbols,
            )
        )

        self._logger.info(
            "strategy_added",
            name=name,
            state=state.value,
            symbols=symbols,
        )

    def record_signal(
        self,
        strategy_name: str,
        signal: Signal,
        market_data: MarketData,
    ) -> None:
        """Record a signal and execute it."""
        self.state.signal_count += 1

        # Update strategy signal count
        for s in self.state.strategies:
            if s["name"] == strategy_name:
                s["signal_count"] += 1
                break

        # Determine signal type for activity
        if signal.signal_type == SignalDirection.LONG:
            signal_icon = "BUY"
            activity_type = "buy"
        elif signal.signal_type == SignalDirection.SHORT:
            signal_icon = "SELL"
            activity_type = "sell"
        else:
            signal_icon = "HOLD"
            activity_type = "hold"

        # Execute in shadow mode
        if strategy_name in self._shadow_executors:
            executor = self._shadow_executors[strategy_name]
            result = executor.execute(signal, market_data)

            # Update P&L
            pnl = executor.get_total_pnl()
            for s in self.state.strategies:
                if s["name"] == strategy_name:
                    s["pnl_today"] = pnl
                    break

            self.state.total_pnl_today = sum(
                s["pnl_today"] for s in self.state.strategies
            )

            # Add activity
            self.state.add_activity({
                "type": activity_type,
                "strategy": strategy_name,
                "symbol": signal.symbol,
                "price": market_data.mid,
                "signal_icon": signal_icon,
                "strength": signal.strength,
            })

        self._logger.info(
            "signal_recorded",
            strategy=strategy_name,
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            strength=signal.strength,
        )

    def record_fill(
        self,
        strategy_name: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        slippage_bps: float | None = None,
    ) -> None:
        """Record a fill confirmation."""
        self.state.fill_count += 1

        slip_text = f" (slip: {slippage_bps:.1f}bps)" if slippage_bps else ""

        self.state.add_activity({
            "type": "fill_confirm",
            "strategy": strategy_name,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "slippage_bps": slippage_bps,
            "slip_text": slip_text,
        })

        self._logger.info(
            "fill_recorded",
            strategy=strategy_name,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            slippage_bps=slippage_bps,
        )

    def update_positions(self) -> None:
        """Update positions from all executors."""
        positions = []

        # Collect from shadow executors
        for name, executor in self._shadow_executors.items():
            for symbol, pos_dict in executor.get_position_summary().items():
                if pos_dict["quantity"] != 0:
                    positions.append({
                        "symbol": symbol,
                        "qty": pos_dict["quantity"],
                        "avg_price": pos_dict["avg_price"],
                        "current": pos_dict["avg_price"],  # Would need market data
                        "pnl": pos_dict["realized_pnl"] + pos_dict["unrealized_pnl"],
                        "strategy": name,
                    })

        # Collect from paper executor if available
        if self._paper_executor:
            try:
                paper_positions = self._paper_executor.get_positions()
                for pos in paper_positions:
                    positions.append({
                        "symbol": pos["symbol"],
                        "qty": float(pos["qty"]),
                        "avg_price": float(pos["avg_entry_price"]),
                        "current": float(pos.get("current_price", pos["avg_entry_price"])),
                        "pnl": float(pos.get("unrealized_pl", 0)),
                        "strategy": "paper",
                    })
            except Exception as e:
                self._logger.warning(
                    "position_update_failed",
                    error=str(e),
                )

        self.state.positions = positions
        self.state.last_update = datetime.now(UTC)

    def _create_header(self) -> Panel:
        """Create the header panel."""
        mode_color = "green" if self.mode == ExecutionMode.PAPER else "yellow"
        mode_text = f"[bold {mode_color}]{self.mode.value.title()} Trading[/]"

        title = Text()
        title.append("autoBot Runner", style="bold white")
        title.append("\n")
        title.append(f"Mode: {mode_text}")

        return Panel(
            title,
            title="",
            border_style="blue",
        )

    def _create_strategies_table(self) -> Panel:
        """Create the active strategies table."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Strategy", style="white")
        table.add_column("State", justify="center")
        table.add_column("P&L Today", justify="right")
        table.add_column("Signals", justify="right")

        for strategy in self.state.strategies:
            state = strategy["state"]
            if state == "PAPER":
                state_style = "green"
            elif state == "SHADOW":
                state_style = "yellow"
            else:
                state_style = "dim"

            pnl = strategy["pnl_today"]
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

            table.add_row(
                strategy["name"],
                f"[{state_style}]{state}[/]",
                f"[{pnl_style}]{pnl_text}[/]",
                str(strategy["signal_count"]),
            )

        return Panel(table, title="Active Strategies", border_style="cyan")

    def _create_activity_log(self) -> Panel:
        """Create the recent activity log."""
        lines = []

        for activity in self.state.recent_activity[:10]:
            timestamp = activity.get("timestamp", datetime.now(UTC))
            time_str = timestamp.strftime("%H:%M:%S")

            if activity["type"] == "buy":
                icon = "[green]BUY[/]"
                text = f"{activity['symbol']} @ ${activity['price']:.2f} ({activity['strategy']})"
            elif activity["type"] == "sell":
                icon = "[red]SELL[/]"
                text = f"{activity['symbol']} @ ${activity['price']:.2f} ({activity['strategy']})"
            elif activity["type"] == "hold":
                icon = "[yellow]HOLD[/]"
                text = f"{activity['symbol']} (strength: {activity['strength']:.2f})"
            elif activity["type"] == "fill":
                icon = "[cyan]FILL[/]"
                text = f"{activity['qty']} {activity['symbol']} @ ${activity['price']:.2f}"
            elif activity["type"] == "fill_confirm":
                icon = "[green]OK[/]"
                text = f"Fill confirmed: {activity['qty']} {activity['symbol']} @ ${activity['price']:.2f}{activity.get('slip_text', '')}"
            else:
                icon = "[dim]INFO[/]"
                text = str(activity)

            lines.append(f"[dim][{time_str}][/] {icon} {text}")

        content = "\n".join(lines) if lines else "[dim]No recent activity[/]"

        return Panel(content, title="Recent Activity", border_style="yellow")

    def _create_positions_table(self) -> Panel:
        """Create the positions table."""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Symbol", style="white")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")

        for pos in self.state.positions:
            qty = pos["qty"]
            qty_style = "green" if qty > 0 else "red" if qty < 0 else "dim"

            pnl = pos["pnl"]
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

            table.add_row(
                pos["symbol"],
                f"[{qty_style}]{qty}[/]",
                f"${pos['avg_price']:.2f}",
                f"${pos['current']:.2f}",
                f"[{pnl_style}]{pnl_text}[/]",
            )

        if not self.state.positions:
            table.add_row("[dim]No positions[/]", "", "", "", "")

        return Panel(table, title="Positions", border_style="magenta")

    def _create_summary(self) -> Panel:
        """Create the summary panel."""
        pnl = self.state.total_pnl_today
        pnl_style = "green" if pnl >= 0 else "red"
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        last_update = self.state.last_update
        update_text = last_update.strftime("%H:%M:%S") if last_update else "Never"

        text = Text()
        text.append(f"Total P&L: [{pnl_style}]{pnl_text}[/]\n", style="bold")
        text.append(f"Signals: {self.state.signal_count}  ")
        text.append(f"Fills: {self.state.fill_count}\n")
        text.append(f"Last Update: {update_text}", style="dim")

        return Panel(text, title="Summary", border_style="green")

    def generate_layout(self) -> Layout:
        """Generate the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=6),
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="strategies"),
            Layout(name="activity"),
        )

        layout["header"].update(self._create_header())
        layout["strategies"].update(self._create_strategies_table())
        layout["activity"].update(self._create_activity_log())
        layout["right"].update(self._create_positions_table())
        layout["footer"].update(self._create_summary())

        return layout

    async def run(self, refresh_rate: float = 1.0) -> None:
        """Run the dashboard with live updates."""
        self._running = True
        self._logger.info(
            "dashboard_started",
            refresh_rate=refresh_rate,
        )

        # Start fill stream if available
        fill_stream_task = None
        if self._fill_stream:
            fill_stream_task = asyncio.create_task(self._fill_stream.start())

        try:
            with Live(
                self.generate_layout(),
                console=console,
                refresh_per_second=1 / refresh_rate,
                screen=True,
            ) as live:
                while self._running:
                    self.update_positions()
                    live.update(self.generate_layout())
                    await asyncio.sleep(refresh_rate)

        except KeyboardInterrupt:
            self._logger.info("dashboard_interrupted")
        finally:
            self._running = False
            if self._fill_stream:
                await self._fill_stream.stop()
            if fill_stream_task:
                fill_stream_task.cancel()

            self._logger.info("dashboard_stopped")

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


def create_demo_dashboard() -> RunnerDashboard:
    """Create a demo dashboard with sample data."""
    dashboard = RunnerDashboard(mode=ExecutionMode.PAPER)

    # Add sample strategies
    dashboard.add_strategy(
        name="TrendFollowing_v1",
        state=StrategyState.PAPER,
        symbols=["SPY", "QQQ"],
    )
    dashboard.add_strategy(
        name="MeanReversion_v2",
        state=StrategyState.SHADOW,
        symbols=["AAPL", "MSFT"],
    )

    return dashboard


async def main() -> None:
    """Main entry point."""
    # Configure structlog for console output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    dashboard = create_demo_dashboard()

    # Handle signals
    def signal_handler(sig: int, frame: Any) -> None:
        dashboard.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Add some demo activity
    now = datetime.now(UTC)
    demo_signal = Signal(
        timestamp=now,
        symbol="SPY",
        signal_type=SignalDirection.LONG,
        strength=0.85,
        target_notional=10000.0,
        strategy_id="TrendFollowing_v1",
    )
    demo_market = MarketData(
        symbol="SPY",
        timestamp=now,
        bid=478.24,
        ask=478.26,
        last=478.25,
    )

    dashboard.record_signal("TrendFollowing_v1", demo_signal, demo_market)
    dashboard.record_fill(
        "TrendFollowing_v1",
        symbol="SPY",
        side="buy",
        qty=10,
        price=478.26,
        slippage_bps=0.2,
    )

    # Run dashboard
    await dashboard.run()


if __name__ == "__main__":
    asyncio.run(main())
