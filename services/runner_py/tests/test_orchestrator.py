"""Tests for the Orchestrator module (T6.05).

Tests cover:
- Configuration initialization
- Start/stop lifecycle
- State management
- Market hours detection
- Phase transitions (pre-market, trading, post-market)
- Strategy loading from registry
- Daily orchestration loop
- Scheduled task management (TaskType, ScheduleFrequency, ScheduledTask)
- AutonomousOrchestrator functionality
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runner_py.orchestrator import (
    AutonomousOrchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    OrchestrationPhase,
    OrchestrationState,
    Orchestrator,
    OrchestratorConfig,
    ScheduledTask,
    ScheduleFrequency,
    TaskType,
)

if TYPE_CHECKING:
    pass


class TestOrchestrationConfig:
    """Tests for OrchestrationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OrchestrationConfig()

        assert config.mode == OrchestrationMode.DAILY
        assert config.market_open == time(9, 30)
        assert config.market_close == time(16, 0)
        assert config.pre_market_minutes == 30
        assert config.post_market_minutes == 15
        assert config.feature_refresh_interval_seconds == 60
        assert config.signal_check_interval_seconds == 5
        assert config.registry_api_url == "http://localhost:8080"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.WEEKLY,
            market_open=time(10, 0),
            market_close=time(15, 30),
            pre_market_minutes=45,
            post_market_minutes=30,
            feature_refresh_interval_seconds=120,
            signal_check_interval_seconds=10,
            registry_api_url="http://registry:8000",
        )

        assert config.mode == OrchestrationMode.WEEKLY
        assert config.market_open == time(10, 0)
        assert config.market_close == time(15, 30)
        assert config.pre_market_minutes == 45
        assert config.post_market_minutes == 30
        assert config.feature_refresh_interval_seconds == 120
        assert config.signal_check_interval_seconds == 10
        assert config.registry_api_url == "http://registry:8000"

    def test_continuous_mode(self) -> None:
        """Test continuous orchestration mode."""
        config = OrchestrationConfig(mode=OrchestrationMode.CONTINUOUS)
        assert config.mode == OrchestrationMode.CONTINUOUS


class TestOrchestrationMode:
    """Tests for OrchestrationMode enum."""

    def test_mode_values(self) -> None:
        """Test all mode values."""
        assert OrchestrationMode.DAILY.value == "daily"
        assert OrchestrationMode.WEEKLY.value == "weekly"
        assert OrchestrationMode.CONTINUOUS.value == "continuous"


class TestOrchestrationState:
    """Tests for OrchestrationState dataclass."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = OrchestrationState(
            is_running=False,
            current_phase=OrchestrationPhase.CLOSED,
            active_strategies=[],
        )

        assert state.is_running is False
        assert state.current_phase == OrchestrationPhase.CLOSED
        assert state.active_strategies == []
        assert state.last_feature_refresh is None
        assert state.last_signal_check is None
        assert state.errors == []


class TestOrchestrator:
    """Tests for Orchestrator class."""

    def test_orchestrator_init(self) -> None:
        """Test orchestrator initialization."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        assert orchestrator.config == config
        state = orchestrator.get_state()
        assert state.is_running is False
        assert state.current_phase == OrchestrationPhase.CLOSED

    def test_orchestrator_init_custom_config(self) -> None:
        """Test orchestrator initialization with custom config."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.CONTINUOUS,
            registry_api_url="http://custom:9000",
        )
        orchestrator = Orchestrator(config)

        assert orchestrator.config.mode == OrchestrationMode.CONTINUOUS
        assert orchestrator.config.registry_api_url == "http://custom:9000"

    def test_get_state_initial(self) -> None:
        """Test initial state retrieval."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        state = orchestrator.get_state()
        assert isinstance(state, OrchestrationState)
        assert state.is_running is False
        assert state.current_phase == OrchestrationPhase.CLOSED
        assert state.active_strategies == []

    @pytest.mark.parametrize(
        "current_time,expected",
        [
            # During trading hours (10:00 EST)
            (datetime(2024, 1, 15, 10, 0, 0), True),
            # Before market open (8:00 EST)
            (datetime(2024, 1, 15, 8, 0, 0), False),
            # After market close (17:00 EST)
            (datetime(2024, 1, 15, 17, 0, 0), False),
            # At market open (9:30 EST)
            (datetime(2024, 1, 15, 9, 30, 0), True),
            # At market close (16:00 EST)
            (datetime(2024, 1, 15, 16, 0, 0), True),
            # Just after close (16:01 EST)
            (datetime(2024, 1, 15, 16, 1, 0), False),
        ],
    )
    def test_is_market_open(self, current_time: datetime, expected: bool) -> None:
        """Test market open detection at various times."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        with patch("runner_py.orchestrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_time
            result = orchestrator.is_market_open()
            assert result == expected

    def test_is_market_open_weekend(self) -> None:
        """Test market is closed on weekends."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Saturday during normal trading hours
        saturday = datetime(2024, 1, 13, 12, 0, 0)  # Saturday
        with patch("runner_py.orchestrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = saturday
            assert orchestrator.is_market_open() is False

    async def test_start_stop(self) -> None:
        """Test orchestrator start and stop."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Mock the main loop to exit quickly
        orchestrator._stop_event = asyncio.Event()

        # Start in background
        start_task = asyncio.create_task(orchestrator.start())

        # Give it a moment to start
        await asyncio.sleep(0.05)

        assert orchestrator.get_state().is_running is True

        # Stop the orchestrator
        await orchestrator.stop()

        # Wait for clean shutdown
        await asyncio.wait_for(start_task, timeout=1.0)

        assert orchestrator.get_state().is_running is False

    async def test_stop_without_start(self) -> None:
        """Test stopping orchestrator that was never started."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Should not raise any errors
        await orchestrator.stop()
        assert orchestrator.get_state().is_running is False

    async def test_load_active_strategies(self) -> None:
        """Test loading strategies from registry API."""
        config = OrchestrationConfig(registry_api_url="http://localhost:8080")
        orchestrator = Orchestrator(config)

        mock_strategies = [
            {"id": 1, "name": "strategy_1", "state": "shadow"},
            {"id": 2, "name": "strategy_2", "state": "paper"},
        ]

        # Directly assign the mock client (not using patch.object)
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": mock_strategies, "total": 2}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        orchestrator._http_client = mock_client

        strategies = await orchestrator._load_active_strategies()

        assert len(strategies) == 2
        assert strategies[0]["name"] == "strategy_1"
        assert strategies[1]["name"] == "strategy_2"

    async def test_load_active_strategies_filters_by_state(self) -> None:
        """Test that only shadow/paper strategies are loaded."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Mock response includes candidate and retired strategies
        mock_strategies = [
            {"id": 1, "name": "shadow_strat", "state": "shadow"},
            {"id": 2, "name": "paper_strat", "state": "paper"},
            {"id": 3, "name": "candidate_strat", "state": "candidate"},
            {"id": 4, "name": "retired_strat", "state": "retired"},
        ]

        # Directly assign the mock client
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": mock_strategies, "total": 4}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        orchestrator._http_client = mock_client

        strategies = await orchestrator._load_active_strategies()

        # Only shadow and paper should be included
        assert len(strategies) == 2
        states = {s["state"] for s in strategies}
        assert "shadow" in states
        assert "paper" in states
        assert "candidate" not in states
        assert "retired" not in states

    async def test_pre_market_phase(self) -> None:
        """Test pre-market phase execution."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Mock dependencies
        mock_strategies = [{"id": 1, "name": "test_strategy", "state": "shadow"}]

        with patch.object(
            orchestrator, "_load_active_strategies", return_value=mock_strategies
        ), patch.object(
            orchestrator, "_refresh_features", return_value=None
        ):
            await orchestrator._pre_market_phase()

            state = orchestrator.get_state()
            assert state.current_phase == OrchestrationPhase.PRE_MARKET

    async def test_trading_phase(self) -> None:
        """Test trading phase execution."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Set up state with active strategies
        orchestrator._state.active_strategies = ["strategy_1"]

        with patch.object(
            orchestrator, "_generate_and_execute_signals", return_value=None
        ):
            await orchestrator._trading_phase()

            state = orchestrator.get_state()
            assert state.current_phase == OrchestrationPhase.TRADING

    async def test_post_market_phase(self) -> None:
        """Test post-market phase execution."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        with patch.object(
            orchestrator, "_evaluate_daily_performance", return_value=None
        ):
            await orchestrator._post_market_phase()

            state = orchestrator.get_state()
            assert state.current_phase == OrchestrationPhase.POST_MARKET

    async def test_daily_loop_phase_transitions(self) -> None:
        """Test that daily loop transitions through phases correctly."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        # Track phase transitions
        phases_visited: list[OrchestrationPhase] = []

        original_pre_market = orchestrator._pre_market_phase
        _ = orchestrator._trading_phase  # Stored for reference only
        original_post_market = orchestrator._post_market_phase

        async def mock_pre_market() -> None:
            phases_visited.append(OrchestrationPhase.PRE_MARKET)
            await original_pre_market()

        async def mock_trading() -> None:
            phases_visited.append(OrchestrationPhase.TRADING)
            orchestrator._state.current_phase = OrchestrationPhase.TRADING

        async def mock_post_market() -> None:
            phases_visited.append(OrchestrationPhase.POST_MARKET)
            await original_post_market()

        with patch.object(orchestrator, "_pre_market_phase", mock_pre_market), \
             patch.object(orchestrator, "_trading_phase", mock_trading), \
             patch.object(orchestrator, "_post_market_phase", mock_post_market), \
             patch.object(orchestrator, "_load_active_strategies", return_value=[]), \
             patch.object(orchestrator, "_refresh_features", return_value=None), \
             patch.object(orchestrator, "_evaluate_daily_performance", return_value=None), \
             patch.object(orchestrator, "is_market_open", return_value=False):

            await orchestrator._run_daily_loop()

            # All phases should be visited in order
            assert OrchestrationPhase.PRE_MARKET in phases_visited
            assert OrchestrationPhase.TRADING in phases_visited
            assert OrchestrationPhase.POST_MARKET in phases_visited

    async def test_refresh_features(self) -> None:
        """Test feature refresh mechanism."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        with patch.object(orchestrator, "_http_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            await orchestrator._refresh_features()

            state = orchestrator.get_state()
            assert state.last_feature_refresh is not None

    async def test_error_handling_in_load_strategies(self) -> None:
        """Test error handling when loading strategies fails."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        with patch.object(orchestrator, "_http_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))

            strategies = await orchestrator._load_active_strategies()

            # Should return empty list on error and log error
            assert strategies == []
            state = orchestrator.get_state()
            assert len(state.errors) > 0

    async def test_graceful_shutdown_during_trading(self) -> None:
        """Test graceful shutdown during trading phase."""
        config = OrchestrationConfig()
        orchestrator = Orchestrator(config)

        orchestrator._state.is_running = True
        orchestrator._state.current_phase = OrchestrationPhase.TRADING

        await orchestrator.stop()

        state = orchestrator.get_state()
        assert state.is_running is False


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with mocked registry API."""

    async def test_full_daily_cycle_mock(self) -> None:
        """Test a full daily cycle with mocked dependencies."""
        config = OrchestrationConfig(
            pre_market_minutes=0,  # Skip pre-market wait
            post_market_minutes=0,  # Skip post-market wait
        )
        orchestrator = Orchestrator(config)

        mock_strategies = [
            {"id": 1, "name": "test_shadow", "state": "shadow"},
        ]

        with patch.object(
            orchestrator, "_load_active_strategies", return_value=mock_strategies
        ), patch.object(
            orchestrator, "_refresh_features", return_value=None
        ), patch.object(
            orchestrator, "_generate_and_execute_signals", return_value=None
        ), patch.object(
            orchestrator, "_evaluate_daily_performance", return_value=None
        ), patch.object(
            orchestrator, "is_market_open", return_value=False
        ):
            await orchestrator._run_daily_loop()

            state = orchestrator.get_state()
            assert state.current_phase == OrchestrationPhase.POST_MARKET


# =============================================================================
# T6.05 Scheduled Task Tests
# =============================================================================


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self) -> None:
        """Test all task type values."""
        assert TaskType.FEATURE_REFRESH.value == "feature_refresh"
        assert TaskType.SIGNAL_GENERATION.value == "signal_generation"
        assert TaskType.SHADOW_EXECUTION.value == "shadow_execution"
        assert TaskType.PAPER_EXECUTION.value == "paper_execution"
        assert TaskType.PERFORMANCE_EVAL.value == "performance_eval"
        assert TaskType.PROMOTION_CHECK.value == "promotion_check"
        assert TaskType.DATA_SYNC.value == "data_sync"

    def test_task_type_is_string_enum(self) -> None:
        """Test that TaskType is a string enum."""
        assert isinstance(TaskType.FEATURE_REFRESH, str)
        assert TaskType.FEATURE_REFRESH == "feature_refresh"


class TestScheduleFrequency:
    """Tests for ScheduleFrequency enum."""

    def test_frequency_values(self) -> None:
        """Test all frequency values."""
        assert ScheduleFrequency.MINUTELY.value == "minutely"
        assert ScheduleFrequency.HOURLY.value == "hourly"
        assert ScheduleFrequency.DAILY.value == "daily"
        assert ScheduleFrequency.WEEKLY.value == "weekly"

    def test_frequency_is_string_enum(self) -> None:
        """Test that ScheduleFrequency is a string enum."""
        assert isinstance(ScheduleFrequency.DAILY, str)
        assert ScheduleFrequency.DAILY == "daily"


class TestScheduledTask:
    """Tests for ScheduledTask dataclass."""

    def test_scheduled_task_creation(self) -> None:
        """Test creating a scheduled task."""
        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
        )

        assert task.task_type == TaskType.FEATURE_REFRESH
        assert task.frequency == ScheduleFrequency.MINUTELY
        assert task.time_of_day is None
        assert task.day_of_week is None
        assert task.enabled is True
        assert task.last_run is None
        assert task.next_run is None
        assert task.handler is None

    def test_scheduled_task_with_time(self) -> None:
        """Test creating a scheduled task with time_of_day."""
        eval_time = time(16, 30)
        task = ScheduledTask(
            task_type=TaskType.PERFORMANCE_EVAL,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=eval_time,
        )

        assert task.time_of_day == eval_time
        assert task.frequency == ScheduleFrequency.DAILY

    def test_scheduled_task_weekly_with_day(self) -> None:
        """Test creating a weekly scheduled task with day_of_week."""
        task = ScheduledTask(
            task_type=TaskType.PROMOTION_CHECK,
            frequency=ScheduleFrequency.WEEKLY,
            time_of_day=time(18, 0),
            day_of_week=6,  # Sunday
        )

        assert task.day_of_week == 6
        assert task.frequency == ScheduleFrequency.WEEKLY

    def test_scheduled_task_with_handler(self) -> None:
        """Test creating a scheduled task with a handler."""
        def my_handler() -> None:
            pass

        task = ScheduledTask(
            task_type=TaskType.DATA_SYNC,
            frequency=ScheduleFrequency.HOURLY,
            handler=my_handler,
        )

        assert task.handler == my_handler

    def test_scheduled_task_disabled(self) -> None:
        """Test creating a disabled scheduled task."""
        task = ScheduledTask(
            task_type=TaskType.SHADOW_EXECUTION,
            frequency=ScheduleFrequency.MINUTELY,
            enabled=False,
        )

        assert task.enabled is False


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default OrchestratorConfig values."""
        config = OrchestratorConfig()

        assert config.market_open == time(9, 30)
        assert config.market_close == time(16, 0)
        assert config.feature_refresh_interval_minutes == 5
        assert config.signal_generation_interval_minutes == 1
        assert config.daily_eval_time == time(16, 30)
        assert config.weekly_optimization_day == 6  # Sunday
        assert config.weekly_optimization_time == time(18, 0)
        assert config.promotion_check_days == 7

    def test_custom_config(self) -> None:
        """Test custom OrchestratorConfig values."""
        config = OrchestratorConfig(
            market_open=time(10, 0),
            market_close=time(15, 0),
            feature_refresh_interval_minutes=10,
            signal_generation_interval_minutes=5,
            daily_eval_time=time(15, 30),
            weekly_optimization_day=0,  # Monday
            weekly_optimization_time=time(20, 0),
            promotion_check_days=14,
        )

        assert config.market_open == time(10, 0)
        assert config.market_close == time(15, 0)
        assert config.feature_refresh_interval_minutes == 10
        assert config.signal_generation_interval_minutes == 5
        assert config.daily_eval_time == time(15, 30)
        assert config.weekly_optimization_day == 0
        assert config.weekly_optimization_time == time(20, 0)
        assert config.promotion_check_days == 14


class TestAutonomousOrchestrator:
    """Tests for AutonomousOrchestrator class."""

    def test_init(self) -> None:
        """Test AutonomousOrchestrator initialization."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator._running is False
        assert len(orchestrator._tasks) == 0

    def test_register_task(self) -> None:
        """Test registering a task with the orchestrator."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        async def my_handler() -> None:
            pass

        orchestrator.register_task(
            task_type=TaskType.FEATURE_REFRESH,
            handler=my_handler,
            frequency=ScheduleFrequency.MINUTELY,
        )

        assert TaskType.FEATURE_REFRESH in orchestrator._tasks
        task = orchestrator._tasks[TaskType.FEATURE_REFRESH]
        assert task.handler == my_handler
        assert task.frequency == ScheduleFrequency.MINUTELY

    def test_register_task_with_time(self) -> None:
        """Test registering a daily task with time_of_day."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        async def eval_handler() -> None:
            pass

        orchestrator.register_task(
            task_type=TaskType.PERFORMANCE_EVAL,
            handler=eval_handler,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=time(16, 30),
        )

        task = orchestrator._tasks[TaskType.PERFORMANCE_EVAL]
        assert task.time_of_day == time(16, 30)

    def test_register_weekly_task(self) -> None:
        """Test registering a weekly task with day_of_week."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        async def promotion_handler() -> None:
            pass

        orchestrator.register_task(
            task_type=TaskType.PROMOTION_CHECK,
            handler=promotion_handler,
            frequency=ScheduleFrequency.WEEKLY,
            time_of_day=time(18, 0),
            day_of_week=6,  # Sunday
        )

        task = orchestrator._tasks[TaskType.PROMOTION_CHECK]
        assert task.day_of_week == 6
        assert task.time_of_day == time(18, 0)

    def test_schedule_default_tasks(self) -> None:
        """Test scheduling default autonomous tasks."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        orchestrator.schedule_default_tasks()

        # Should have 4 default tasks
        assert len(orchestrator._tasks) == 4
        assert TaskType.FEATURE_REFRESH in orchestrator._tasks
        assert TaskType.SIGNAL_GENERATION in orchestrator._tasks
        assert TaskType.PERFORMANCE_EVAL in orchestrator._tasks
        assert TaskType.PROMOTION_CHECK in orchestrator._tasks

    def test_is_market_hours_during_trading(self) -> None:
        """Test market hours detection during trading hours."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        # Weekday during market hours (10:00)
        with patch("runner_py.orchestrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 0, 0)  # Tuesday
            assert orchestrator.is_market_hours() is True

    def test_is_market_hours_before_open(self) -> None:
        """Test market hours detection before market open."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        # Weekday before market open (8:00)
        with patch("runner_py.orchestrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 8, 0, 0)  # Tuesday
            assert orchestrator.is_market_hours() is False

    def test_is_market_hours_weekend(self) -> None:
        """Test market hours detection on weekend."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        # Saturday during normal trading hours
        with patch("runner_py.orchestrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 13, 12, 0, 0)  # Saturday
            assert orchestrator.is_market_hours() is False

    def test_get_task_status(self) -> None:
        """Test getting status of scheduled tasks."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)
        orchestrator.schedule_default_tasks()

        status = orchestrator.get_task_status()

        assert "running" in status
        assert "tasks" in status
        assert "market_hours" in status
        assert status["running"] is False
        assert len(status["tasks"]) == 4

    def test_is_task_due_minutely_first_run(self) -> None:
        """Test minutely task is due on first run."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
        )

        now = datetime.now()
        assert orchestrator._is_task_due(task, now) is True

    def test_is_task_due_minutely_after_run(self) -> None:
        """Test minutely task is due after 1 minute."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        now = datetime.now()
        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
            last_run=now - timedelta(minutes=2),
        )

        assert orchestrator._is_task_due(task, now) is True

    def test_is_task_due_minutely_too_soon(self) -> None:
        """Test minutely task is not due before 1 minute."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        now = datetime.now()
        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
            last_run=now - timedelta(seconds=30),
        )

        assert orchestrator._is_task_due(task, now) is False

    def test_is_task_due_hourly(self) -> None:
        """Test hourly task due detection."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        now = datetime.now()
        task = ScheduledTask(
            task_type=TaskType.DATA_SYNC,
            frequency=ScheduleFrequency.HOURLY,
            last_run=now - timedelta(hours=2),
        )

        assert orchestrator._is_task_due(task, now) is True

    def test_is_task_due_daily_with_time(self) -> None:
        """Test daily task with specific time_of_day."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        eval_time = time(16, 30)
        yesterday = datetime(2024, 1, 14, 16, 30, 0)
        today_after_time = datetime(2024, 1, 15, 17, 0, 0)

        task = ScheduledTask(
            task_type=TaskType.PERFORMANCE_EVAL,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=eval_time,
            last_run=yesterday,
        )

        assert orchestrator._is_task_due(task, today_after_time) is True

    def test_is_task_due_disabled_task(self) -> None:
        """Test disabled task is never due."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
            enabled=False,
        )

        now = datetime.now()
        assert orchestrator._is_task_due(task, now) is False

    async def test_execute_task_sync_handler(self) -> None:
        """Test executing a task with sync handler."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        executed = []

        def sync_handler() -> None:
            executed.append(True)

        task = ScheduledTask(
            task_type=TaskType.DATA_SYNC,
            frequency=ScheduleFrequency.HOURLY,
            handler=sync_handler,
        )

        await orchestrator._execute_task(task)

        assert len(executed) == 1
        assert task.last_run is not None

    async def test_execute_task_async_handler(self) -> None:
        """Test executing a task with async handler."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        executed = []

        async def async_handler() -> None:
            executed.append(True)

        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
            handler=async_handler,
        )

        await orchestrator._execute_task(task)

        assert len(executed) == 1
        assert task.last_run is not None

    async def test_execute_task_no_handler(self) -> None:
        """Test executing a task with no handler does nothing."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        task = ScheduledTask(
            task_type=TaskType.DATA_SYNC,
            frequency=ScheduleFrequency.HOURLY,
            handler=None,
        )

        # Should not raise
        await orchestrator._execute_task(task)
        assert task.last_run is None

    async def test_execute_task_handler_error(self) -> None:
        """Test that handler errors are caught and logged."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        async def failing_handler() -> None:
            raise ValueError("Handler failed")

        task = ScheduledTask(
            task_type=TaskType.FEATURE_REFRESH,
            frequency=ScheduleFrequency.MINUTELY,
            handler=failing_handler,
        )

        # Should not raise
        await orchestrator._execute_task(task)
        # last_run should NOT be updated on failure
        assert task.last_run is None

    async def test_start_stop_lifecycle(self) -> None:
        """Test starting and stopping the orchestrator."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        # Start in background
        start_task = asyncio.create_task(orchestrator.start())

        # Give it a moment to start
        await asyncio.sleep(0.05)
        assert orchestrator._running is True

        # Stop the orchestrator
        await orchestrator.stop()

        # Wait for clean shutdown
        await asyncio.wait_for(start_task, timeout=1.0)

        assert orchestrator._running is False

    async def test_stop_without_start(self) -> None:
        """Test stopping orchestrator that was never started."""
        config = OrchestratorConfig()
        orchestrator = AutonomousOrchestrator(config)

        # Should not raise
        await orchestrator.stop()
        assert orchestrator._running is False
