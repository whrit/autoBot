"""
Models package for backtester_py.

This package provides cost models and risk management utilities for the backtester.

Cost Models:
    - CostModel: Abstract base class for cost models
    - FixedCostModel: Simple fixed-cost model
    - VolumeImpactCostModel: Volume-dependent cost model with market impact
    - ZeroCostModel: No transaction costs (for benchmarking)
    - create_cost_model: Factory function to create cost models

Risk Management:
    - RiskChecker: Validates orders against risk limits
    - PositionSizer: Calculates position sizes based on risk parameters
    - RiskManager: Combined risk checking and position sizing
    - RiskViolation: Represents a risk check violation
    - RiskSeverity: Severity levels for risk violations

Example usage:
    >>> from backtester_py.models import CostModel, FixedCostModel, RiskChecker, PositionSizer
    >>> cost_model = FixedCostModel(fee_rate=0.001, slippage_rate=0.0005)
    >>> risk_checker = RiskChecker(max_position_pct=0.10)
"""

from backtester_py.models.cost_model import (
    CostModel,
    CostModelType,
    FillResult,
    FixedCostModel,
    TieredCostModel,
    VolumeImpactCostModel,
    ZeroCostModel,
    create_cost_model,
)
from backtester_py.models.risk_checker import (
    BacktesterRiskChecker,
    PositionSizer,
    RiskChecker,
    RiskManager,
    RiskSeverity,
    RiskViolation,
    create_risk_checker,
)

__all__ = [
    # Cost Models
    "CostModel",
    "CostModelType",
    "FillResult",
    "FixedCostModel",
    "TieredCostModel",
    "VolumeImpactCostModel",
    "ZeroCostModel",
    "create_cost_model",
    # Risk Management
    "RiskSeverity",
    "RiskViolation",
    "RiskChecker",
    "PositionSizer",
    "RiskManager",
    "create_risk_checker",
    "BacktesterRiskChecker",
]
