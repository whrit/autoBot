"""
Backtester Service for Autonomous Equities Trading Engine.

This service provides realistic taker-style backtest evaluation with:
- Quote-based fill simulation (buy @ ask + slippage, sell @ bid - slippage)
- Slippage model integration from cost_models
- Position limits and stop-loss logic from risk_models
- Walk-forward evaluation framework
- Performance metrics computation (Sharpe, Sortino, MDD, etc.)

Key Features:
- Uses realistic execution assumptions (no look-ahead bias)
- Integrates with cost_models for slippage estimation
- Integrates with risk_models for position limits
- Deterministic and reproducible results

Example usage:
    >>> from backtester_py import BacktestEngine, BacktestConfig
    >>> from cost_models import SlippageModel, TransactionCostModel
    >>> from risk_models import RiskLimits, RiskChecker
    >>>
    >>> slippage = SlippageModel()
    >>> cost_model = TransactionCostModel(slippage, fixed_cost_bps=0.35)
    >>> risk_limits = RiskLimits(...)
    >>> risk_checker = RiskChecker(risk_limits)
    >>>
    >>> config = BacktestConfig(
    ...     initial_capital=100_000,
    ...     cost_model=cost_model,
    ...     risk_checker=risk_checker,
    ... )
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(decision_frames, signals)
"""

from backtester_py.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Fill,
    Position,
    Signal,
)
from backtester_py.evaluation import (
    EvaluationMetrics,
    WalkForwardConfig,
    WalkForwardEvaluator,
    calculate_metrics,
)
from backtester_py.metrics import BacktestMetrics, MetricsCalculator
from backtester_py.purged_cv import PurgedCrossValidator
from backtester_py.regimes import RegimeAnalyzer, RegimeType
from backtester_py.walk_forward import WalkForwardOptimizer

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Signal",
    "Position",
    "Fill",
    # Evaluation (existing)
    "EvaluationMetrics",
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "calculate_metrics",
    # Metrics (T3.06 - additional)
    "BacktestMetrics",
    "MetricsCalculator",
    # Walk-forward (T3.05 - additional)
    "WalkForwardOptimizer",
    # Regimes (T3.07)
    "RegimeAnalyzer",
    "RegimeType",
    # Purged CV (T3.08)
    "PurgedCrossValidator",
]

__version__ = "0.1.0"
