"""
Complexity Penalties Module for Strategy Optimization.

Provides complexity penalty calculations to penalize overly complex strategies
during optimization. This helps prevent overfitting and encourages simpler,
more robust strategies.

Key Features:
- Parameter count penalty
- Lookback period penalty
- ML model complexity penalty (tree depth, number of estimators)
- Combined complexity score calculation
- Configurable penalty weights and thresholds

Example usage:
    >>> from optimizer_py.complexity import ComplexityPenaltyConfig, compute_complexity_penalty
    >>> config = ComplexityPenaltyConfig(max_parameters=10, max_lookback=200)
    >>> penalty = compute_complexity_penalty(strategy_params, config)
    >>> adjusted_score = raw_score - penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComplexityPenaltyConfig:
    """
    Configuration for complexity penalty calculations.

    Attributes:
        max_parameters: Maximum number of parameters before penalty applies (default: 10)
        max_lookback: Maximum lookback period before penalty applies (default: 200)
        max_estimators: Maximum ML estimators before penalty applies (default: 500)
        max_depth: Maximum ML tree depth before penalty applies (default: 12)
        parameter_penalty_weight: Weight for parameter count penalty (default: 0.1)
        lookback_penalty_weight: Weight for lookback period penalty (default: 0.1)
        ml_penalty_weight: Weight for ML complexity penalty (default: 0.1)
    """

    max_parameters: int = 10
    max_lookback: int = 200
    max_estimators: int = 500
    max_depth: int = 12
    parameter_penalty_weight: float = 0.1
    lookback_penalty_weight: float = 0.1
    ml_penalty_weight: float = 0.1


def _get_numeric_value(params: dict[str, Any], key: str) -> float | None:
    """
    Safely extract a numeric value from params dict.

    Args:
        params: Parameters dictionary
        key: Key to look up

    Returns:
        Float value if found and numeric, None otherwise
    """
    value = params.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_parameter_count_penalty(
    params: dict[str, Any],
    config: ComplexityPenaltyConfig,
) -> float:
    """
    Compute penalty based on number of strategy parameters.

    More parameters increase the risk of overfitting. This penalty
    grows logarithmically with excess parameters to avoid over-penalizing
    moderate complexity.

    Args:
        params: Strategy parameters dictionary
        config: Complexity penalty configuration

    Returns:
        Penalty value (0.0 if below threshold)
    """
    if config.parameter_penalty_weight == 0.0:
        return 0.0

    num_params = len(params)

    if num_params <= config.max_parameters:
        return 0.0

    # Calculate excess ratio (how much over the threshold)
    excess = num_params - config.max_parameters
    excess_ratio = excess / config.max_parameters

    # Apply penalty weight with diminishing returns (log scale)
    import math
    penalty = config.parameter_penalty_weight * math.log1p(excess_ratio)

    return penalty


def compute_lookback_penalty(
    params: dict[str, Any],
    config: ComplexityPenaltyConfig,
) -> float:
    """
    Compute penalty based on lookback period.

    Long lookback periods can lead to data snooping and reduced
    adaptability. This penalty applies when lookback exceeds threshold.

    Recognizes common lookback parameter names:
    - lookback
    - window
    - period
    - lookback_period

    Args:
        params: Strategy parameters dictionary
        config: Complexity penalty configuration

    Returns:
        Penalty value (0.0 if below threshold or no lookback found)
    """
    if config.lookback_penalty_weight == 0.0:
        return 0.0

    # Look for common lookback parameter names
    lookback_keys = ["lookback", "window", "period", "lookback_period"]

    max_lookback_value = 0.0
    for key in lookback_keys:
        value = _get_numeric_value(params, key)
        if value is not None and value > max_lookback_value:
            max_lookback_value = value

    if max_lookback_value <= 0 or max_lookback_value <= config.max_lookback:
        return 0.0

    # Calculate excess ratio
    excess = max_lookback_value - config.max_lookback
    excess_ratio = excess / config.max_lookback

    # Apply penalty weight with diminishing returns
    import math
    penalty = config.lookback_penalty_weight * math.log1p(excess_ratio)

    return penalty


def compute_ml_complexity_penalty(
    params: dict[str, Any],
    config: ComplexityPenaltyConfig,
) -> float:
    """
    Compute penalty based on ML model complexity.

    Complex ML models with many estimators or deep trees are more
    prone to overfitting. This penalty applies when ML parameters
    exceed thresholds.

    Recognizes ML complexity parameters:
    - n_estimators: Number of trees/estimators
    - max_depth: Maximum tree depth
    - num_leaves: Number of leaves (for LGBM)

    Args:
        params: Strategy parameters dictionary
        config: Complexity penalty configuration

    Returns:
        Penalty value (0.0 if below threshold or no ML params found)
    """
    if config.ml_penalty_weight == 0.0:
        return 0.0

    import math

    total_penalty = 0.0

    # Check n_estimators
    n_estimators = _get_numeric_value(params, "n_estimators")
    if n_estimators is not None and n_estimators > config.max_estimators:
        excess = n_estimators - config.max_estimators
        excess_ratio = excess / config.max_estimators
        total_penalty += config.ml_penalty_weight * math.log1p(excess_ratio)

    # Check max_depth
    max_depth = _get_numeric_value(params, "max_depth")
    if max_depth is not None and max_depth > config.max_depth:
        excess = max_depth - config.max_depth
        excess_ratio = excess / config.max_depth
        total_penalty += config.ml_penalty_weight * math.log1p(excess_ratio)

    # Check num_leaves (for LightGBM-style models)
    num_leaves = _get_numeric_value(params, "num_leaves")
    if num_leaves is not None:
        # Approximate depth from num_leaves (depth ~ log2(num_leaves))
        effective_depth = math.log2(max(num_leaves, 2))
        if effective_depth > config.max_depth:
            excess = effective_depth - config.max_depth
            excess_ratio = excess / config.max_depth
            total_penalty += config.ml_penalty_weight * math.log1p(excess_ratio)

    return total_penalty


def compute_complexity_penalty(
    params: dict[str, Any],
    config: ComplexityPenaltyConfig | None = None,
) -> float:
    """
    Compute total complexity penalty for a strategy.

    Combines parameter count, lookback period, and ML complexity penalties.
    The total penalty is bounded to prevent extreme values.

    Args:
        params: Strategy parameters dictionary
        config: Complexity penalty configuration (uses defaults if not provided)

    Returns:
        Total complexity penalty value

    Example:
        >>> config = ComplexityPenaltyConfig(max_parameters=5, max_lookback=100)
        >>> params = {"lookback": 200, "threshold": 0.02, "n_estimators": 500}
        >>> penalty = compute_complexity_penalty(params, config)
        >>> print(f"Complexity penalty: {penalty:.4f}")
    """
    if config is None:
        config = ComplexityPenaltyConfig()

    # Compute individual penalties
    param_penalty = compute_parameter_count_penalty(params, config)
    lookback_penalty = compute_lookback_penalty(params, config)
    ml_penalty = compute_ml_complexity_penalty(params, config)

    # Sum penalties
    total_penalty = param_penalty + lookback_penalty + ml_penalty

    # Bound the total penalty to prevent extreme values
    # Max penalty of ~5.0 is reasonable (would significantly impact scores)
    max_penalty = 5.0
    bounded_penalty = min(total_penalty, max_penalty)

    return bounded_penalty


# Public API exports
__all__ = [
    "ComplexityPenaltyConfig",
    "compute_complexity_penalty",
    "compute_parameter_count_penalty",
    "compute_lookback_penalty",
    "compute_ml_complexity_penalty",
]
