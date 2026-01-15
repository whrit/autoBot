"""Gate evaluation module for strategy quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GateType(str, Enum):
    """Types of quality gates for strategy evaluation."""

    MIN_SHARPE = "min_sharpe"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_TRADES = "min_trades"
    MIN_WIN_RATE = "min_win_rate"
    MIN_PROFIT_FACTOR = "min_profit_factor"
    COST_SENSITIVITY = "cost_sensitivity"
    MIN_SORTINO = "min_sortino"


@dataclass
class GateConfig:
    """Configuration for a quality gate."""

    gate_type: GateType
    threshold: float
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_type": self.gate_type.value,
            "threshold": self.threshold,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateConfig:
        """Create from dictionary."""
        return cls(
            gate_type=GateType(data["gate_type"]),
            threshold=data["threshold"],
            required=data.get("required", True),
        )


@dataclass
class GateResult:
    """Result of evaluating a single gate."""

    gate_type: GateType
    passed: bool
    actual_value: float
    threshold: float
    message: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_type": self.gate_type.value,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
            "required": self.required,
        }


class GateEvaluator:
    """Evaluate strategies against quality gates.

    Default gates ensure minimum quality standards:
    - Sharpe ratio >= 0.5 (required)
    - Max drawdown <= 15% (required)
    - Minimum 30 trades (required)
    - Win rate >= 45% (optional)
    """

    DEFAULT_GATES: list[GateConfig] = [
        GateConfig(GateType.MIN_SHARPE, 0.5, required=True),
        GateConfig(GateType.MAX_DRAWDOWN, 0.15, required=True),
        GateConfig(GateType.MIN_TRADES, 30, required=True),
        GateConfig(GateType.MIN_WIN_RATE, 0.45, required=False),
    ]

    # Mapping of gate types to metric keys
    METRIC_KEYS: dict[GateType, str] = {
        GateType.MIN_SHARPE: "sharpe",
        GateType.MAX_DRAWDOWN: "max_drawdown",
        GateType.MIN_TRADES: "num_trades",
        GateType.MIN_WIN_RATE: "win_rate",
        GateType.MIN_PROFIT_FACTOR: "profit_factor",
        GateType.COST_SENSITIVITY: "cost_sensitivity",
        GateType.MIN_SORTINO: "sortino",
    }

    def __init__(self, gates: list[GateConfig] | None = None) -> None:
        """Initialize gate evaluator.

        Args:
            gates: List of gate configurations. Uses DEFAULT_GATES if not provided.
        """
        self._gates = gates if gates is not None else list(self.DEFAULT_GATES)

    @property
    def gates(self) -> list[GateConfig]:
        """Get current gate configurations."""
        return self._gates

    @gates.setter
    def gates(self, value: list[GateConfig]) -> None:
        """Set gate configurations."""
        self._gates = value

    def _get_metric_value(self, metrics: dict[str, Any], gate_type: GateType) -> float | None:
        """Get metric value for a gate type.

        Args:
            metrics: Dictionary of strategy metrics.
            gate_type: Type of gate to get metric for.

        Returns:
            Metric value or None if not found.
        """
        metric_key = self.METRIC_KEYS.get(gate_type)
        if metric_key is None:
            return None

        value = metrics.get(metric_key)
        if value is None:
            return None

        return float(value)

    def _evaluate_gate(self, gate: GateConfig, metrics: dict[str, Any]) -> GateResult:
        """Evaluate a single gate.

        Args:
            gate: Gate configuration.
            metrics: Dictionary of strategy metrics.

        Returns:
            Gate evaluation result.
        """
        actual_value = self._get_metric_value(metrics, gate.gate_type)

        if actual_value is None:
            return GateResult(
                gate_type=gate.gate_type,
                passed=False,
                actual_value=0.0,
                threshold=gate.threshold,
                message=f"Metric not found for {gate.gate_type.value}",
                required=gate.required,
            )

        # Evaluate based on gate type
        if gate.gate_type == GateType.MAX_DRAWDOWN:
            # For max drawdown, actual must be <= threshold
            passed = actual_value <= gate.threshold
            comparison = "<="
        elif gate.gate_type == GateType.COST_SENSITIVITY:
            # Cost sensitivity should be low (actual <= threshold)
            passed = actual_value <= gate.threshold
            comparison = "<="
        else:
            # For all other gates, actual must be >= threshold
            passed = actual_value >= gate.threshold
            comparison = ">="

        if passed:
            message = (
                f"{gate.gate_type.value} passed: "
                f"{actual_value:.4f} {comparison} {gate.threshold:.4f}"
            )
        else:
            message = (
                f"{gate.gate_type.value} failed: "
                f"{actual_value:.4f} not {comparison} {gate.threshold:.4f}"
            )

        return GateResult(
            gate_type=gate.gate_type,
            passed=passed,
            actual_value=actual_value,
            threshold=gate.threshold,
            message=message,
            required=gate.required,
        )

    def evaluate(self, metrics: dict[str, Any]) -> list[GateResult]:
        """Evaluate all gates against provided metrics.

        Args:
            metrics: Dictionary of strategy metrics. Expected keys:
                - sharpe: Sharpe ratio
                - sortino: Sortino ratio
                - max_drawdown: Maximum drawdown (0-1)
                - profit_factor: Profit factor
                - win_rate: Win rate (0-1)
                - num_trades: Number of trades

        Returns:
            List of gate evaluation results.
        """
        return [self._evaluate_gate(gate, metrics) for gate in self._gates]

    def passes_all_required(self, results: list[GateResult]) -> bool:
        """Check if all required gates passed.

        Args:
            results: List of gate evaluation results.

        Returns:
            True if all required gates passed.
        """
        return all(result.passed for result in results if result.required)

    def passes_all(self, results: list[GateResult]) -> bool:
        """Check if all gates (required and optional) passed.

        Args:
            results: List of gate evaluation results.

        Returns:
            True if all gates passed.
        """
        return all(result.passed for result in results)

    def get_summary(self, results: list[GateResult]) -> dict[str, Any]:
        """Get a summary of gate evaluation results.

        Args:
            results: List of gate evaluation results.

        Returns:
            Summary dictionary with:
                - passed: Whether all required gates passed
                - passed_count: Number of gates that passed
                - total_count: Total number of gates
                - required_passed: Number of required gates that passed
                - required_total: Total number of required gates
                - results: List of individual gate results
        """
        total_count = len(results)
        passed_count = sum(1 for r in results if r.passed)

        required_results = [r for r in results if r.required]
        required_total = len(required_results)
        required_passed = sum(1 for r in required_results if r.passed)

        optional_results = [r for r in results if not r.required]
        optional_total = len(optional_results)
        optional_passed = sum(1 for r in optional_results if r.passed)

        return {
            "passed": self.passes_all_required(results),
            "passed_count": passed_count,
            "total_count": total_count,
            "required_passed": required_passed,
            "required_total": required_total,
            "optional_passed": optional_passed,
            "optional_total": optional_total,
            "results": [r.to_dict() for r in results],
        }

    def add_gate(self, gate: GateConfig) -> None:
        """Add a gate configuration.

        Args:
            gate: Gate configuration to add.
        """
        # Check for duplicate gate types
        existing_types = {g.gate_type for g in self._gates}
        if gate.gate_type in existing_types:
            # Replace existing gate of same type
            self._gates = [g for g in self._gates if g.gate_type != gate.gate_type]

        self._gates.append(gate)

    def remove_gate(self, gate_type: GateType) -> bool:
        """Remove a gate configuration.

        Args:
            gate_type: Type of gate to remove.

        Returns:
            True if gate was removed, False if not found.
        """
        original_len = len(self._gates)
        self._gates = [g for g in self._gates if g.gate_type != gate_type]
        return len(self._gates) < original_len

    def reset_to_defaults(self) -> None:
        """Reset gates to default configuration."""
        self._gates = list(self.DEFAULT_GATES)
