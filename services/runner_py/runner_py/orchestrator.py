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
        self._logger = logger.bind(component="autonomous_orchestrator")

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
        self._logger.debug("Registered task", task_type=task_type.value, frequency=frequency.value)

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
        self._logger.info("Scheduled default tasks", task_count=len(self._tasks))

    async def start(self) -> None:
        """Start the autonomous loop."""
        self._logger.info("Starting autonomous orchestrator")
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        try:
            await self._loop_task
        except asyncio.CancelledError:
            self._logger.info("Autonomous orchestrator cancelled")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the autonomous loop gracefully."""
        self._logger.info("Stopping autonomous orchestrator")
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def _run_loop(self) -> None:
        """Main loop that checks and executes scheduled tasks."""
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
            self._logger.warning("Task has no handler", task_type=task.task_type.value)
            return
        try:
            self._logger.debug("Executing task", task_type=task.task_type.value)
            if inspect.iscoroutinefunction(task.handler):
                await task.handler()
            else:
                task.handler()
            task.last_run = datetime.now()
            self._logger.debug("Task completed", task_type=task.task_type.value)
        except Exception as e:
            self._logger.error(
                "Task execution failed", task_type=task.task_type.value, error=str(e)
            )

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
            })
        return {
            "running": self._running, "tasks": tasks_info, "market_hours": self.is_market_hours()
        }

    async def _default_feature_refresh(self) -> None:
        """Default feature refresh handler."""
        self._logger.debug("Default feature refresh - no-op")

    async def _default_signal_generation(self) -> None:
        """Default signal generation handler."""
        self._logger.debug("Default signal generation - no-op")

    async def _default_performance_eval(self) -> None:
        """Default performance evaluation handler."""
        self._logger.debug("Default performance eval - no-op")

    async def _default_promotion_check(self) -> None:
        """Default promotion check handler."""
        self._logger.debug("Default promotion check - no-op")


class Orchestrator:
    """Coordinate the autonomous trading loop."""

    def __init__(self, config: OrchestrationConfig) -> None:
        """Initialize the orchestrator."""
        self.config = config
        self._state = OrchestrationState(
            is_running=False, current_phase=OrchestrationPhase.CLOSED, active_strategies=[])
        self._stop_event = asyncio.Event()
        self._http_client: httpx.AsyncClient | None = None
        self._logger = logger.bind(component="orchestrator")

    async def start(self) -> None:
        """Start the orchestration loop."""
        self._logger.info("Starting orchestrator", mode=self.config.mode.value)
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
            self._logger.info("Orchestrator cancelled")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the orchestration loop gracefully."""
        self._logger.info("Stopping orchestrator")
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
        self._logger.info("Running daily loop")
        await self._pre_market_phase()
        await self._trading_phase()
        await self._post_market_phase()

    async def _run_weekly_loop(self) -> None:
        """Run the weekly trading loop."""
        for _ in range(5):
            if self._stop_event.is_set():
                break
            await self._run_daily_loop()
            await asyncio.sleep(1)

    async def _run_continuous_loop(self) -> None:
        """Run continuous trading loop."""
        while not self._stop_event.is_set():
            await self._run_daily_loop()
            await asyncio.sleep(1)

    async def _pre_market_phase(self) -> None:
        """Execute pre-market preparation."""
        self._logger.info("Entering pre-market phase")
        self._state.current_phase = OrchestrationPhase.PRE_MARKET
        strategies = await self._load_active_strategies()
        self._state.active_strategies = [s["name"] for s in strategies]
        self._logger.info("Loaded strategies", count=len(strategies))
        await self._refresh_features()
        if self.config.pre_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.pre_market_minutes * 0.1)

    async def _trading_phase(self) -> None:
        """Execute main trading phase."""
        self._logger.info("Entering trading phase")
        self._state.current_phase = OrchestrationPhase.TRADING
        while not self._stop_event.is_set() and self.is_market_open():
            try:
                await self._generate_and_execute_signals()
                self._state.last_signal_check = datetime.now()
                if self._should_refresh_features():
                    await self._refresh_features()
            except Exception as e:
                self._logger.error("Error in trading loop", error=str(e))
                self._state.errors.append(f"Trading error: {e}")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.config.signal_check_interval_seconds)
                break
            except TimeoutError:
                pass

    async def _post_market_phase(self) -> None:
        """Execute post-market evaluation."""
        self._logger.info("Entering post-market phase")
        self._state.current_phase = OrchestrationPhase.POST_MARKET
        await self._evaluate_daily_performance()
        if self.config.post_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.post_market_minutes * 0.1)

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
            self._logger.debug("Loaded strategies from registry",
                             total=len(all_strategies), active=len(active_strategies))
            return active_strategies
        except Exception as e:
            self._logger.error("Failed to load strategies", error=str(e))
            self._state.errors.append(f"Strategy load error: {e}")
            return []

    async def _refresh_features(self) -> None:
        """Refresh feature data for active strategies."""
        try:
            self._logger.debug("Refreshing features")
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.registry_api_url, timeout=30.0)
            self._state.last_feature_refresh = datetime.now()
            self._logger.debug("Features refreshed", timestamp=self._state.last_feature_refresh)
        except Exception as e:
            self._logger.error("Failed to refresh features", error=str(e))
            self._state.errors.append(f"Feature refresh error: {e}")

    async def _generate_and_execute_signals(self) -> None:
        """Generate and execute signals for all active strategies."""
        pass

    async def _evaluate_daily_performance(self) -> None:
        """Evaluate end-of-day performance for all strategies."""
        self._logger.info("Evaluating daily performance")

    def _should_refresh_features(self) -> bool:
        """Check if features should be refreshed."""
        if self._state.last_feature_refresh is None:
            return True
        elapsed = (datetime.now() - self._state.last_feature_refresh).total_seconds()
        return elapsed >= self.config.feature_refresh_interval_seconds

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        self._state.is_running = False
        self._logger.info("Orchestrator cleanup complete")


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
]
