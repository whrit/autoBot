"""Autonomous Trading Loop Orchestrator (T6.05).

This module coordinates the full trading pipeline:
- Pre-market: Refresh features, load active strategies
- Trading: Generate signals, execute (shadow or paper)
- Post-market: Evaluate performance, log results

The orchestrator manages the daily/weekly trading loop and
handles transitions between market phases.

Additionally, it provides scheduled task management for:
- Feature refresh (every N minutes)
- Signal generation (every minute)
- Performance evaluation (daily)
- Promotion checks (weekly)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class OrchestrationMode(str, Enum):
    """Orchestration mode for the trading loop."""

    DAILY = "daily"
    WEEKLY = "weekly"
    CONTINUOUS = "continuous"


class OrchestrationPhase(str, Enum):
    """Current phase of the trading day."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    TRADING = "trading"
    POST_MARKET = "post_market"


class TaskType(str, Enum):
    """Types of scheduled tasks in the autonomous loop."""

    FEATURE_REFRESH = "feature_refresh"
    SIGNAL_GENERATION = "signal_generation"
    SHADOW_EXECUTION = "shadow_execution"
    PAPER_EXECUTION = "paper_execution"
    PERFORMANCE_EVAL = "performance_eval"
    PROMOTION_CHECK = "promotion_check"
    DATA_SYNC = "data_sync"


class ScheduleFrequency(str, Enum):
    """Frequency of scheduled task execution."""

    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class StrategyState(str, Enum):
    """State of a strategy in the promotion workflow."""

    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
    DISABLED = "disabled"


@dataclass
class ScheduledTask:
    """A scheduled task in the autonomous loop."""

    task_type: TaskType
    frequency: ScheduleFrequency
    time_of_day: time | None = None  # For daily/weekly tasks
    day_of_week: int | None = None  # 0=Monday, 6=Sunday
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    handler: Callable[[], Any] | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    avg_duration_ms: float = 0.0


@dataclass
class PromotionEvent:
    """Record of a strategy promotion/demotion event."""

    strategy_id: str
    from_state: StrategyState
    to_state: StrategyState
    timestamp: datetime
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration loop."""

    mode: OrchestrationMode = OrchestrationMode.DAILY
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    pre_market_minutes: int = 30
    post_market_minutes: int = 15
    feature_refresh_interval_seconds: int = 60
    signal_check_interval_seconds: int = 5
    registry_api_url: str = "http://localhost:8080"


@dataclass
class OrchestratorConfig:
    """Configuration for the AutonomousOrchestrator."""

    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    feature_refresh_interval_minutes: int = 5
    signal_generation_interval_minutes: int = 1
    daily_eval_time: time = field(default_factory=lambda: time(16, 30))
    weekly_optimization_day: int = 6  # Sunday
    weekly_optimization_time: time = field(default_factory=lambda: time(18, 0))
    promotion_check_days: int = 7


@dataclass
class OrchestrationState:
    """Current state of the orchestration loop."""

    is_running: bool
    current_phase: OrchestrationPhase
    active_strategies: list[str]
    last_feature_refresh: datetime | None = None
    last_signal_check: datetime | None = None
    errors: list[str] = field(default_factory=list)


class AutonomousOrchestrator:
    """Orchestrate the autonomous trading loop with scheduled tasks."""

    def __init__(self, config: OrchestratorConfig) -> None:
        """Initialize the autonomous orchestrator."""
        self.config = config
        self._tasks: dict[TaskType, ScheduledTask] = {}
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._promotion_history: list[PromotionEvent] = []
        self._strategy_states: dict[str, StrategyState] = {}
        self._logger = logger.bind(component="autonomous_orchestrator")

        self._logger.info(
            "orchestrator_created",
            market_open=config.market_open.isoformat(),
            market_close=config.market_close.isoformat(),
            daily_eval_time=config.daily_eval_time.isoformat(),
            weekly_optimization_day=config.weekly_optimization_day,
        )

    def register_task(
        self,
        task_type: TaskType,
        handler: Callable[[], Any],
        frequency: ScheduleFrequency,
        time_of_day: time | None = None,
        day_of_week: int | None = None,
    ) -> None:
        """Register a task with the orchestrator."""
        task = ScheduledTask(
            task_type=task_type,
            frequency=frequency,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            handler=handler,
        )
        self._tasks[task_type] = task

        self._logger.info(
            "task_registered",
            task_type=task_type.value,
            frequency=frequency.value,
            time_of_day=time_of_day.isoformat() if time_of_day else None,
            day_of_week=day_of_week,
        )

    def schedule_default_tasks(self) -> None:
        """Set up the default autonomous loop schedule."""
        self.register_task(
            task_type=TaskType.FEATURE_REFRESH,
            handler=self._default_feature_refresh,
            frequency=ScheduleFrequency.MINUTELY,
        )
        self.register_task(
            task_type=TaskType.SIGNAL_GENERATION,
            handler=self._default_signal_generation,
            frequency=ScheduleFrequency.MINUTELY,
        )
        self.register_task(
            task_type=TaskType.PERFORMANCE_EVAL,
            handler=self._default_performance_eval,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=self.config.daily_eval_time,
        )
        self.register_task(
            task_type=TaskType.PROMOTION_CHECK,
            handler=self._default_promotion_check,
            frequency=ScheduleFrequency.WEEKLY,
            time_of_day=self.config.weekly_optimization_time,
            day_of_week=self.config.weekly_optimization_day,
        )

        self._logger.info(
            "default_tasks_scheduled",
            task_count=len(self._tasks),
            tasks=[t.value for t in self._tasks.keys()],
        )

    async def start(self) -> None:
        """Start the autonomous loop."""
        self._logger.info(
            "orchestrator_starting",
            task_count=len(self._tasks),
        )

        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        try:
            await self._loop_task
        except asyncio.CancelledError:
            self._logger.info("orchestrator_cancelled")
        finally:
            self._running = False
            self._logger.info(
                "orchestrator_stopped",
                total_task_runs=sum(t.run_count for t in self._tasks.values()),
                total_errors=sum(t.error_count for t in self._tasks.values()),
            )

    async def stop(self) -> None:
        """Stop the autonomous loop gracefully."""
        self._logger.info(
            "orchestrator_stopping",
            was_running=self._running,
        )

        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def _run_loop(self) -> None:
        """Main loop that checks and executes scheduled tasks."""
        self._logger.debug("main_loop_started")

        while self._running:
            await self._check_and_run_tasks()
            await asyncio.sleep(1)

    async def _check_and_run_tasks(self) -> None:
        """Check if any tasks are due and run them."""
        now = datetime.now()
        for task in self._tasks.values():
            if task.enabled and self._is_task_due(task, now):
                await self._execute_task(task)

    def _is_task_due(self, task: ScheduledTask, now: datetime) -> bool:
        """Check if a task is due to run."""
        if not task.enabled:
            return False
        if task.last_run is None:
            if task.frequency == ScheduleFrequency.DAILY and task.time_of_day:
                return now.time() >= task.time_of_day
            elif task.frequency == ScheduleFrequency.WEEKLY:
                if task.day_of_week is not None and now.weekday() != task.day_of_week:
                    return False
                if task.time_of_day and now.time() < task.time_of_day:
                    return False
            return True
        elapsed = now - task.last_run
        if task.frequency == ScheduleFrequency.MINUTELY:
            return elapsed >= timedelta(minutes=1)
        elif task.frequency == ScheduleFrequency.HOURLY:
            return elapsed >= timedelta(hours=1)
        elif task.frequency == ScheduleFrequency.DAILY:
            if task.time_of_day:
                last_run_date = task.last_run.date()
                current_date = now.date()
                return current_date > last_run_date and now.time() >= task.time_of_day
            return elapsed >= timedelta(days=1)
        elif task.frequency == ScheduleFrequency.WEEKLY:
            if task.day_of_week is not None and now.weekday() != task.day_of_week:
                return False
            if task.time_of_day and now.time() < task.time_of_day:
                return False
            return elapsed >= timedelta(days=7)
        return False

    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        if task.handler is None:
            self._logger.warning(
                "task_no_handler",
                task_type=task.task_type.value,
            )
            return

        start_time = datetime.now()
        self._logger.info(
            "task_execution_started",
            task_type=task.task_type.value,
            run_count=task.run_count + 1,
            last_run=task.last_run.isoformat() if task.last_run else None,
        )

        try:
            if inspect.iscoroutinefunction(task.handler):
                await task.handler()
            else:
                task.handler()

            task.run_count += 1
            task.last_run = datetime.now()

            duration_ms = (task.last_run - start_time).total_seconds() * 1000
            # Update rolling average
            task.avg_duration_ms = (
                (task.avg_duration_ms * (task.run_count - 1) + duration_ms) / task.run_count
            )

            self._logger.info(
                "task_execution_completed",
                task_type=task.task_type.value,
                duration_ms=round(duration_ms, 2),
                avg_duration_ms=round(task.avg_duration_ms, 2),
                run_count=task.run_count,
            )
        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)

            self._logger.error(
                "task_execution_failed",
                task_type=task.task_type.value,
                error=str(e),
                error_type=type(e).__name__,
                error_count=task.error_count,
                run_count=task.run_count,
            )

    def record_strategy_state_change(
        self,
        strategy_id: str,
        from_state: StrategyState,
        to_state: StrategyState,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record a strategy state change (promotion/demotion)."""
        event = PromotionEvent(
            strategy_id=strategy_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            reason=reason,
            metrics=metrics or {},
        )
        self._promotion_history.append(event)
        self._strategy_states[strategy_id] = to_state

        self._logger.info(
            "strategy_state_changed",
            strategy_id=strategy_id,
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
            metrics=metrics,
            promotion_history_count=len(self._promotion_history),
        )

    def get_promotion_history(self, strategy_id: str | None = None) -> list[PromotionEvent]:
        """Get promotion history, optionally filtered by strategy."""
        if strategy_id:
            return [e for e in self._promotion_history if e.strategy_id == strategy_id]
        return list(self._promotion_history)

    def is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return self.config.market_open <= current_time <= self.config.market_close

    def get_task_status(self) -> dict[str, Any]:
        """Get status of all scheduled tasks."""
        tasks_info = []
        for task in self._tasks.values():
            tasks_info.append({
                "type": task.task_type.value,
                "frequency": task.frequency.value,
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "time_of_day": task.time_of_day.isoformat() if task.time_of_day else None,
                "day_of_week": task.day_of_week,
                "run_count": task.run_count,
                "error_count": task.error_count,
                "last_error": task.last_error,
                "avg_duration_ms": round(task.avg_duration_ms, 2),
            })

        return {
            "running": self._running,
            "tasks": tasks_info,
            "market_hours": self.is_market_hours(),
            "strategy_states": {k: v.value for k, v in self._strategy_states.items()},
            "promotion_count": len(self._promotion_history),
        }

    def log_status_summary(self) -> None:
        """Log a comprehensive status summary."""
        status = self.get_task_status()
        self._logger.info(
            "orchestrator_status_summary",
            running=status["running"],
            market_hours=status["market_hours"],
            task_count=len(status["tasks"]),
            strategy_count=len(status["strategy_states"]),
            promotion_count=status["promotion_count"],
            tasks=status["tasks"],
        )

    async def _default_feature_refresh(self) -> None:
        """Default feature refresh handler."""
        self._logger.debug(
            "feature_refresh_executed",
            handler="default",
        )

    async def _default_signal_generation(self) -> None:
        """Default signal generation handler."""
        self._logger.debug(
            "signal_generation_executed",
            handler="default",
        )

    async def _default_performance_eval(self) -> None:
        """Default performance evaluation handler."""
        self._logger.info(
            "performance_eval_executed",
            handler="default",
            strategy_count=len(self._strategy_states),
        )

    async def _default_promotion_check(self) -> None:
        """Default promotion check handler."""
        self._logger.info(
            "promotion_check_executed",
            handler="default",
            strategy_count=len(self._strategy_states),
            promotion_history_count=len(self._promotion_history),
        )


class Orchestrator:
    """Coordinate the autonomous trading loop."""

    def __init__(self, config: OrchestrationConfig) -> None:
        """Initialize the orchestrator."""
        self.config = config
        self._state = OrchestrationState(
            is_running=False, current_phase=OrchestrationPhase.CLOSED, active_strategies=[])
        self._stop_event = asyncio.Event()
        self._http_client: httpx.AsyncClient | None = None
        self._phase_start_time: datetime | None = None
        self._logger = logger.bind(component="orchestrator")

        self._logger.info(
            "orchestrator_initialized",
            mode=config.mode.value,
            market_open=config.market_open.isoformat(),
            market_close=config.market_close.isoformat(),
            registry_api_url=config.registry_api_url,
        )

    async def start(self) -> None:
        """Start the orchestration loop."""
        self._logger.info(
            "orchestration_starting",
            mode=self.config.mode.value,
        )

        self._state.is_running = True
        self._stop_event.clear()
        self._http_client = httpx.AsyncClient(base_url=self.config.registry_api_url, timeout=30.0)

        try:
            if self.config.mode == OrchestrationMode.DAILY:
                await self._run_daily_loop()
            elif self.config.mode == OrchestrationMode.WEEKLY:
                await self._run_weekly_loop()
            elif self.config.mode == OrchestrationMode.CONTINUOUS:
                await self._run_continuous_loop()
        except asyncio.CancelledError:
            self._logger.info("orchestration_cancelled")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the orchestration loop gracefully."""
        self._logger.info(
            "orchestration_stopping",
            current_phase=self._state.current_phase.value,
            active_strategies=len(self._state.active_strategies),
        )

        self._stop_event.set()
        self._state.is_running = False

    def get_state(self) -> OrchestrationState:
        """Get the current orchestration state."""
        return self._state

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return self.config.market_open <= current_time <= self.config.market_close

    async def _run_daily_loop(self) -> None:
        """Run the daily trading loop."""
        self._logger.info("daily_loop_started")

        await self._pre_market_phase()
        await self._trading_phase()
        await self._post_market_phase()

        self._logger.info(
            "daily_loop_completed",
            total_errors=len(self._state.errors),
        )

    async def _run_weekly_loop(self) -> None:
        """Run the weekly trading loop."""
        self._logger.info("weekly_loop_started")

        for day in range(5):
            if self._stop_event.is_set():
                self._logger.info(
                    "weekly_loop_stopped_early",
                    days_completed=day,
                )
                break

            self._logger.info(
                "weekly_loop_day_started",
                day_number=day + 1,
            )

            await self._run_daily_loop()
            await asyncio.sleep(1)

        self._logger.info("weekly_loop_completed")

    async def _run_continuous_loop(self) -> None:
        """Run continuous trading loop."""
        loop_count = 0
        self._logger.info("continuous_loop_started")

        while not self._stop_event.is_set():
            loop_count += 1
            self._logger.debug(
                "continuous_loop_iteration",
                loop_count=loop_count,
            )

            await self._run_daily_loop()
            await asyncio.sleep(1)

        self._logger.info(
            "continuous_loop_completed",
            total_loops=loop_count,
        )

    async def _pre_market_phase(self) -> None:
        """Execute pre-market preparation."""
        self._phase_start_time = datetime.now()
        self._state.current_phase = OrchestrationPhase.PRE_MARKET

        self._logger.info(
            "phase_entered",
            phase="pre_market",
            started_at=self._phase_start_time.isoformat(),
        )

        strategies = await self._load_active_strategies()
        self._state.active_strategies = [s["name"] for s in strategies]

        self._logger.info(
            "strategies_loaded",
            phase="pre_market",
            strategy_count=len(strategies),
            strategies=self._state.active_strategies,
        )

        await self._refresh_features()

        if self.config.pre_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.pre_market_minutes * 0.1)

        phase_duration = (datetime.now() - self._phase_start_time).total_seconds()
        self._logger.info(
            "phase_completed",
            phase="pre_market",
            duration_seconds=round(phase_duration, 2),
            strategies_ready=len(self._state.active_strategies),
        )

    async def _trading_phase(self) -> None:
        """Execute main trading phase."""
        self._phase_start_time = datetime.now()
        self._state.current_phase = OrchestrationPhase.TRADING
        signal_count = 0

        self._logger.info(
            "phase_entered",
            phase="trading",
            started_at=self._phase_start_time.isoformat(),
            active_strategies=len(self._state.active_strategies),
        )

        while not self._stop_event.is_set() and self.is_market_open():
            try:
                await self._generate_and_execute_signals()
                signal_count += 1
                self._state.last_signal_check = datetime.now()

                if self._should_refresh_features():
                    await self._refresh_features()

                self._logger.debug(
                    "trading_iteration",
                    signal_check_count=signal_count,
                    last_signal_check=self._state.last_signal_check.isoformat(),
                )
            except Exception as e:
                self._logger.error(
                    "trading_phase_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    signal_check_count=signal_count,
                )
                self._state.errors.append(f"Trading error: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.config.signal_check_interval_seconds)
                break
            except TimeoutError:
                pass

        phase_duration = (datetime.now() - self._phase_start_time).total_seconds()
        self._logger.info(
            "phase_completed",
            phase="trading",
            duration_seconds=round(phase_duration, 2),
            signal_checks=signal_count,
            error_count=len(self._state.errors),
        )

    async def _post_market_phase(self) -> None:
        """Execute post-market evaluation."""
        self._phase_start_time = datetime.now()
        self._state.current_phase = OrchestrationPhase.POST_MARKET

        self._logger.info(
            "phase_entered",
            phase="post_market",
            started_at=self._phase_start_time.isoformat(),
        )

        await self._evaluate_daily_performance()

        if self.config.post_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.post_market_minutes * 0.1)

        phase_duration = (datetime.now() - self._phase_start_time).total_seconds()
        self._logger.info(
            "phase_completed",
            phase="post_market",
            duration_seconds=round(phase_duration, 2),
        )

    async def _load_active_strategies(self) -> list[dict[str, Any]]:
        """Load active strategies from the registry API."""
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.registry_api_url, timeout=30.0)

            response = await self._http_client.get("/strategies", params={"limit": 100})
            response.raise_for_status()
            data = response.json()
            all_strategies: list[dict[str, Any]] = data.get("items", [])
            active_strategies = [s for s in all_strategies if s.get("state") in ("shadow", "paper")]

            self._logger.info(
                "strategies_fetched",
                total_count=len(all_strategies),
                active_count=len(active_strategies),
                shadow_count=len([s for s in active_strategies if s.get("state") == "shadow"]),
                paper_count=len([s for s in active_strategies if s.get("state") == "paper"]),
            )

            return active_strategies
        except Exception as e:
            self._logger.error(
                "strategy_load_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._state.errors.append(f"Strategy load error: {e}")
            return []

    async def _refresh_features(self) -> None:
        """Refresh feature data for active strategies."""
        try:
            refresh_start = datetime.now()

            self._logger.debug(
                "feature_refresh_started",
                active_strategies=len(self._state.active_strategies),
            )

            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.registry_api_url, timeout=30.0)

            self._state.last_feature_refresh = datetime.now()
            duration_ms = (self._state.last_feature_refresh - refresh_start).total_seconds() * 1000

            self._logger.info(
                "feature_refresh_completed",
                duration_ms=round(duration_ms, 2),
                timestamp=self._state.last_feature_refresh.isoformat(),
            )
        except Exception as e:
            self._logger.error(
                "feature_refresh_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._state.errors.append(f"Feature refresh error: {e}")

    async def _generate_and_execute_signals(self) -> None:
        """Generate and execute signals for all active strategies."""
        self._logger.debug(
            "signal_generation_started",
            strategy_count=len(self._state.active_strategies),
        )

    async def _evaluate_daily_performance(self) -> None:
        """Evaluate end-of-day performance for all strategies."""
        self._logger.info(
            "performance_evaluation_started",
            strategy_count=len(self._state.active_strategies),
        )

        # Placeholder for actual evaluation logic
        self._logger.info(
            "performance_evaluation_completed",
            strategy_count=len(self._state.active_strategies),
        )

    def _should_refresh_features(self) -> bool:
        """Check if features should be refreshed."""
        if self._state.last_feature_refresh is None:
            return True
        elapsed = (datetime.now() - self._state.last_feature_refresh).total_seconds()
        return elapsed >= self.config.feature_refresh_interval_seconds

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self._logger.info(
            "cleanup_started",
            phase=self._state.current_phase.value,
            error_count=len(self._state.errors),
        )

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        self._state.is_running = False
        self._state.current_phase = OrchestrationPhase.CLOSED

        self._logger.info(
            "cleanup_completed",
            total_errors=len(self._state.errors),
        )

    def log_status_summary(self) -> None:
        """Log a comprehensive status summary."""
        self._logger.info(
            "orchestrator_status_summary",
            is_running=self._state.is_running,
            current_phase=self._state.current_phase.value,
            active_strategies=len(self._state.active_strategies),
            last_feature_refresh=self._state.last_feature_refresh.isoformat() if self._state.last_feature_refresh else None,
            last_signal_check=self._state.last_signal_check.isoformat() if self._state.last_signal_check else None,
            error_count=len(self._state.errors),
            market_open=self.is_market_open(),
        )


__all__ = [
    "Orchestrator",
    "OrchestrationConfig",
    "OrchestrationMode",
    "OrchestrationPhase",
    "OrchestrationState",
    "TaskType",
    "ScheduleFrequency",
    "ScheduledTask",
    "AutonomousOrchestrator",
    "OrchestratorConfig",
    "StrategyState",
    "PromotionEvent",
]
