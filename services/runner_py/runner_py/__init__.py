"""
Runner Python Service - Shadow and Paper Execution for Trading Signals.

This module provides:
- Shadow execution mode for simulating signal execution (T6.01-T6.02)
- Paper execution via Alpaca Paper Trading API (T6.03)
- Fill confirmation via Alpaca WebSocket (T6.04)
- Signal logging for analysis and backtesting comparison
"""

# Acceptance Testing (T6.12)
from runner_py.acceptance import AcceptanceCriteria, AcceptanceResult, AcceptanceTester

# Performance Benchmarking (T6.10)
from runner_py.benchmark import BenchmarkConfig, BenchmarkResult, PerformanceBenchmark

# Fill streaming (T6.04)
from runner_py.fill_stream import (
    ConnectionStats,
    FillConfirmationStream,
    FillStreamConfig,
    LatencyStats,
)

# Orchestrator (T6.05)
from runner_py.orchestrator import (
    AutonomousOrchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    OrchestrationPhase,
    OrchestrationState,
    Orchestrator,
    OrchestratorConfig,
    PromotionEvent,
    ScheduledTask,
    ScheduleFrequency,
    StrategyState,
    TaskType,
)

# Dashboard (main.py)
from runner_py.main import DashboardState, RunnerDashboard

# Paper execution (T6.03)
from runner_py.paper import PaperConfig, PaperExecutor, PaperFill, PnLTracker

# Promotion Manager (T6.06)
from runner_py.promotion import (
    AutoPromoter,
    PromotionCriteria,
    PromotionDecision,
    PromotionEvaluation,
    PromotionManager,
    PromotionState,
)

# Shadow Execution (T6.01) and Signal Logging (T6.02)
from runner_py.shadow import ShadowConfig, ShadowExecutor, SignalSummary
from runner_py.signal_log import SignalLogConfig, SignalLogger

# Slippage Analysis (T6.08)
from runner_py.slippage_analysis import (
    SlippageAnalysisConfig,
    SlippageAnalyzer,
    SlippageReport,
)

# Additional types for shadow execution
from runner_py.types import (
    ExecutionMode,
    ExecutionResult,
    MarketData,
    Position,
    ShadowExecutionResult,
    ShadowFill,
    Signal,
    SignalDirection,
)

# Validation (T6.07)
from runner_py.validation import BehaviorValidator, ValidationConfig, ValidationResult

__all__ = [
    # Types
    "ExecutionMode",
    "ExecutionResult",
    "MarketData",
    "Position",
    "ShadowExecutionResult",
    "ShadowFill",
    "Signal",
    "SignalDirection",
    # Shadow Execution (T6.01)
    "ShadowConfig",
    "ShadowExecutor",
    "SignalSummary",
    # Signal Logging (T6.02)
    "SignalLogConfig",
    "SignalLogger",
    # Paper Execution (T6.03)
    "PaperConfig",
    "PaperExecutor",
    "PaperFill",
    "PnLTracker",
    # Fill Streaming (T6.04)
    "ConnectionStats",
    "FillConfirmationStream",
    "FillStreamConfig",
    "LatencyStats",
    # Orchestrator (T6.05)
    "AutonomousOrchestrator",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestrationConfig",
    "OrchestrationMode",
    "OrchestrationPhase",
    "OrchestrationState",
    "PromotionEvent",
    "ScheduledTask",
    "ScheduleFrequency",
    "StrategyState",
    "TaskType",
    # Dashboard
    "DashboardState",
    "RunnerDashboard",
    # Promotion Manager (T6.06)
    "AutoPromoter",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionEvaluation",
    "PromotionManager",
    "PromotionState",
    # Validation (T6.07)
    "BehaviorValidator",
    "ValidationConfig",
    "ValidationResult",
    # Slippage Analysis (T6.08)
    "SlippageAnalysisConfig",
    "SlippageAnalyzer",
    "SlippageReport",
    # Performance Benchmarking (T6.10)
    "BenchmarkConfig",
    "BenchmarkResult",
    "PerformanceBenchmark",
    # Acceptance Testing (T6.12)
    "AcceptanceCriteria",
    "AcceptanceResult",
    "AcceptanceTester",
]

__version__ = "0.1.0"
