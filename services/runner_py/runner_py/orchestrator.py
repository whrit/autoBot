"""Autonomous Trading Loop Orchestrator (T6.05).

This module coordinates the full trading pipeline:
- Pre-market: Refresh features, load active strategies
- Trading: Generate signals, execute (shadow or paper)
- Post-market: Evaluate performance, log results

The orchestrator manages the daily/weekly trading loop and
handles transitions between market phases.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time
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
class OrchestrationState:
    """Current state of the orchestration loop."""

    is_running: bool
    current_phase: OrchestrationPhase
    active_strategies: list[str]
    last_feature_refresh: datetime | None = None
    last_signal_check: datetime | None = None
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """Coordinate the autonomous trading loop.

    This class manages the daily trading loop, transitioning through
    pre-market, trading, and post-market phases. It loads strategies
    from the registry, refreshes features, generates signals, and
    coordinates execution.
    """

    def __init__(self, config: OrchestrationConfig) -> None:
        """Initialize the orchestrator.

        Args:
            config: Orchestration configuration.
        """
        self.config = config
        self._state = OrchestrationState(
            is_running=False,
            current_phase=OrchestrationPhase.CLOSED,
            active_strategies=[],
        )
        self._stop_event = asyncio.Event()
        self._http_client: httpx.AsyncClient | None = None
        self._logger = logger.bind(component="orchestrator")

    async def start(self) -> None:
        """Start the orchestration loop.

        This method begins the main orchestration loop, which will
        continue until stop() is called.
        """
        self._logger.info("Starting orchestrator", mode=self.config.mode.value)
        self._state.is_running = True
        self._stop_event.clear()

        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(
            base_url=self.config.registry_api_url,
            timeout=30.0,
        )

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
        """Stop the orchestration loop gracefully.

        This method signals the orchestration loop to stop and waits
        for a clean shutdown.
        """
        self._logger.info("Stopping orchestrator")
        self._stop_event.set()
        self._state.is_running = False

    def get_state(self) -> OrchestrationState:
        """Get the current orchestration state.

        Returns:
            Current orchestration state.
        """
        return self._state

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if the market is open, False otherwise.
        """
        now = datetime.now()

        # Check if weekend (0=Monday, 5=Saturday, 6=Sunday)
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return self.config.market_open <= current_time <= self.config.market_close

    async def _run_daily_loop(self) -> None:
        """Run the daily trading loop.

        This method executes a single daily loop through:
        1. Pre-market: Refresh features, load strategies
        2. Trading: Generate signals, execute trades
        3. Post-market: Evaluate performance, log results
        """
        self._logger.info("Running daily loop")

        # Pre-market phase
        await self._pre_market_phase()

        # Trading phase - run until market close or stop signal
        await self._trading_phase()

        # Post-market phase
        await self._post_market_phase()

    async def _run_weekly_loop(self) -> None:
        """Run the weekly trading loop."""
        for _ in range(5):  # Monday-Friday
            if self._stop_event.is_set():
                break
            await self._run_daily_loop()
            # Wait for next trading day (simplified)
            await asyncio.sleep(1)

    async def _run_continuous_loop(self) -> None:
        """Run continuous trading loop."""
        while not self._stop_event.is_set():
            await self._run_daily_loop()
            # Brief pause between cycles
            await asyncio.sleep(1)

    async def _pre_market_phase(self) -> None:
        """Execute pre-market preparation.

        This phase:
        1. Loads active strategies from the registry
        2. Refreshes feature data
        3. Validates strategy configurations
        """
        self._logger.info("Entering pre-market phase")
        self._state.current_phase = OrchestrationPhase.PRE_MARKET

        # Load active strategies (shadow and paper only)
        strategies = await self._load_active_strategies()
        self._state.active_strategies = [s["name"] for s in strategies]

        self._logger.info(
            "Loaded strategies",
            count=len(strategies),
            strategies=self._state.active_strategies,
        )

        # Refresh features
        await self._refresh_features()

        # Wait for pre-market period (if configured)
        if self.config.pre_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.pre_market_minutes * 0.1)  # Scaled down for testing

    async def _trading_phase(self) -> None:
        """Execute main trading phase.

        This phase runs the main signal generation and execution loop
        until the market closes or stop is signaled.
        """
        self._logger.info("Entering trading phase")
        self._state.current_phase = OrchestrationPhase.TRADING

        # Main trading loop
        while not self._stop_event.is_set() and self.is_market_open():
            try:
                await self._generate_and_execute_signals()
                self._state.last_signal_check = datetime.now()

                # Check if we should refresh features
                if self._should_refresh_features():
                    await self._refresh_features()

            except Exception as e:
                self._logger.error("Error in trading loop", error=str(e))
                self._state.errors.append(f"Trading error: {e}")

            # Wait for next signal check
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.signal_check_interval_seconds,
                )
                break  # Stop event was set
            except TimeoutError:
                pass  # Continue trading loop

    async def _post_market_phase(self) -> None:
        """Execute post-market evaluation.

        This phase:
        1. Evaluates daily performance for each strategy
        2. Logs results to the registry
        3. Triggers any end-of-day processes
        """
        self._logger.info("Entering post-market phase")
        self._state.current_phase = OrchestrationPhase.POST_MARKET

        # Evaluate daily performance
        await self._evaluate_daily_performance()

        # Wait for post-market period (if configured)
        if self.config.post_market_minutes > 0 and not self._stop_event.is_set():
            await asyncio.sleep(self.config.post_market_minutes * 0.1)  # Scaled down

    async def _load_active_strategies(self) -> list[dict[str, Any]]:
        """Load active strategies from the registry API.

        Returns:
            List of active strategy dictionaries (shadow and paper states).
        """
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.registry_api_url,
                    timeout=30.0,
                )

            response = await self._http_client.get("/strategies", params={"limit": 100})
            response.raise_for_status()
            data = response.json()

            # Filter to only shadow and paper strategies
            all_strategies: list[dict[str, Any]] = data.get("items", [])
            active_strategies = [
                s for s in all_strategies if s.get("state") in ("shadow", "paper")
            ]

            self._logger.debug(
                "Loaded strategies from registry",
                total=len(all_strategies),
                active=len(active_strategies),
            )

            return active_strategies

        except Exception as e:
            self._logger.error("Failed to load strategies", error=str(e))
            self._state.errors.append(f"Strategy load error: {e}")
            return []

    async def _refresh_features(self) -> None:
        """Refresh feature data for active strategies.

        This method triggers feature computation for all symbols
        used by active strategies.
        """
        try:
            self._logger.debug("Refreshing features")

            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    base_url=self.config.registry_api_url,
                    timeout=30.0,
                )

            # In a full implementation, this would call the feature builder service
            # For now, we just record the refresh time
            self._state.last_feature_refresh = datetime.now()

            self._logger.debug("Features refreshed", timestamp=self._state.last_feature_refresh)

        except Exception as e:
            self._logger.error("Failed to refresh features", error=str(e))
            self._state.errors.append(f"Feature refresh error: {e}")

    async def _generate_and_execute_signals(self) -> None:
        """Generate and execute signals for all active strategies.

        This method iterates through active strategies, generates
        signals based on current market data, and executes them
        according to the strategy's state (shadow or paper).
        """
        # Placeholder for signal generation logic
        # In full implementation, this would:
        # 1. Load current feature data
        # 2. Run each strategy's signal generation
        # 3. Execute signals via shadow or paper executor
        pass

    async def _evaluate_daily_performance(self) -> None:
        """Evaluate end-of-day performance for all strategies.

        This method calculates daily metrics and logs them to
        the registry for later analysis.
        """
        self._logger.info("Evaluating daily performance")

        # Placeholder for performance evaluation
        # In full implementation, this would:
        # 1. Calculate daily returns, Sharpe, drawdown, etc.
        # 2. Compare shadow execution to backtest expectations
        # 3. Log results to registry

    def _should_refresh_features(self) -> bool:
        """Check if features should be refreshed.

        Returns:
            True if features should be refreshed based on interval.
        """
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
]
