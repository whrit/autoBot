"""
Optimizer service for strategy generation and evaluation.

This module provides:
- Strategy family framework (T4.01)
- Trend-following strategies (T4.02)
- Mean-reversion strategies (T4.03)
- Parameter sweep optimization (T4.04)
- XGBoost ML strategy implementation (T4.05)
- GPU-accelerated ML training support for XGBoost with graceful CPU fallback (T4.06, T4.07)
- ONNX model export (T4.08)
- Candidate ranking within and across strategy families (T4.09)
- Cost sensitivity testing for strategy robustness (T4.10)
"""

from optimizer_py.cost_sensitivity import (
    CostScenario,
    CostSensitivityConfig,
    CostSensitivityTester,
    SensitivityResult,
)
from optimizer_py.gpu_support import (
    GPUConfig,
    GPUManager,
    GPUMetadata,
    TrainingDevice,
    get_cuda_info,
)
from optimizer_py.mean_reversion import MeanReversionStrategy
from optimizer_py.ml_strategy import (
    MLStrategyConfig,
    XGBoostStrategy,
)
from optimizer_py.onnx_export import (
    ONNXExportConfig,
    ONNXExporter,
    load_onnx_model,
    run_onnx_inference,
)
from optimizer_py.ranking import (
    CandidateRanker,
    CandidateScore,
    RankingConfig,
    RankingMetric,
)
from optimizer_py.strategies import (
    StrategyFamily as OptimizerStrategyFamily,
)
from optimizer_py.strategies import (
    StrategySignal as OptimizerStrategySignal,
)
from optimizer_py.strategy_family import StrategyFamily, StrategySignal
from optimizer_py.sweep import (
    ParameterSweep,
    SweepConfig,
    SweepResult,
    generate_parameter_combinations,
)
from optimizer_py.trend import TrendStrategy

__all__ = [
    # Strategy Framework (T4.01)
    "StrategyFamily",
    "StrategySignal",
    "OptimizerStrategyFamily",
    "OptimizerStrategySignal",
    # Trend Strategy (T4.02)
    "TrendStrategy",
    # Mean Reversion Strategy (T4.03)
    "MeanReversionStrategy",
    # Parameter Sweep (T4.04)
    "ParameterSweep",
    "SweepConfig",
    "SweepResult",
    "generate_parameter_combinations",
    # ML Strategy (T4.05)
    "MLStrategyConfig",
    "XGBoostStrategy",
    # GPU Support (T4.06, T4.07)
    "GPUConfig",
    "GPUManager",
    "GPUMetadata",
    "TrainingDevice",
    "get_cuda_info",
    # ONNX Export (T4.08)
    "ONNXExporter",
    "ONNXExportConfig",
    "load_onnx_model",
    "run_onnx_inference",
    # Candidate Ranking (T4.09)
    "CandidateRanker",
    "CandidateScore",
    "RankingConfig",
    "RankingMetric",
    # Cost Sensitivity (T4.10)
    "CostScenario",
    "CostSensitivityConfig",
    "CostSensitivityTester",
    "SensitivityResult",
]

__version__ = "0.1.0"
