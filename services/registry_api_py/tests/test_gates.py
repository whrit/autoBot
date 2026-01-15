"""Tests for GateEvaluator class and gate API endpoints."""

from __future__ import annotations

import pytest

from registry_api_py.gates import (
    GateConfig,
    GateEvaluator,
    GateResult,
    GateType,
)


class TestGateEvaluator:
    """Tests for GateEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> GateEvaluator:
        """Create a GateEvaluator with default gates."""
        return GateEvaluator()

    @pytest.fixture
    def passing_metrics(self) -> dict[str, float | int]:
        """Metrics that pass all default gates."""
        return {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,  # 10% - below 15% threshold
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

    @pytest.fixture
    def failing_sharpe_metrics(self) -> dict[str, float | int]:
        """Metrics that fail the Sharpe gate."""
        return {
            "sharpe": 0.3,  # Below 0.5 threshold
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

    @pytest.fixture
    def failing_drawdown_metrics(self) -> dict[str, float | int]:
        """Metrics that fail the max drawdown gate."""
        return {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.25,  # Above 15% threshold
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

    @pytest.fixture
    def failing_trades_metrics(self) -> dict[str, float | int]:
        """Metrics that fail the minimum trades gate."""
        return {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 20,  # Below 30 threshold
        }

    def test_passes_all_gates(
        self, evaluator: GateEvaluator, passing_metrics: dict[str, float | int]
    ) -> None:
        """Test that all gates pass with good metrics."""
        results = evaluator.evaluate(passing_metrics)

        assert evaluator.passes_all_required(results) is True
        assert all(r.passed for r in results if r.required)

    def test_fails_sharpe_gate(
        self, evaluator: GateEvaluator, failing_sharpe_metrics: dict[str, float | int]
    ) -> None:
        """Test that Sharpe gate fails with low Sharpe ratio."""
        results = evaluator.evaluate(failing_sharpe_metrics)

        assert evaluator.passes_all_required(results) is False

        sharpe_result = next(r for r in results if r.gate_type == GateType.MIN_SHARPE)
        assert sharpe_result.passed is False
        assert sharpe_result.actual_value == 0.3
        assert sharpe_result.threshold == 0.5

    def test_fails_drawdown_gate(
        self, evaluator: GateEvaluator, failing_drawdown_metrics: dict[str, float | int]
    ) -> None:
        """Test that max drawdown gate fails with high drawdown."""
        results = evaluator.evaluate(failing_drawdown_metrics)

        assert evaluator.passes_all_required(results) is False

        dd_result = next(r for r in results if r.gate_type == GateType.MAX_DRAWDOWN)
        assert dd_result.passed is False
        assert dd_result.actual_value == 0.25
        assert dd_result.threshold == 0.15

    def test_fails_trades_gate(
        self, evaluator: GateEvaluator, failing_trades_metrics: dict[str, float | int]
    ) -> None:
        """Test that minimum trades gate fails with too few trades."""
        results = evaluator.evaluate(failing_trades_metrics)

        assert evaluator.passes_all_required(results) is False

        trades_result = next(r for r in results if r.gate_type == GateType.MIN_TRADES)
        assert trades_result.passed is False
        assert trades_result.actual_value == 20
        assert trades_result.threshold == 30

    def test_optional_gate_failure_allowed(self, evaluator: GateEvaluator) -> None:
        """Test that failing optional gates doesn't fail overall."""
        # Win rate gate is optional by default
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.40,  # Below 0.45 threshold (optional gate)
            "num_trades": 100,
        }

        results = evaluator.evaluate(metrics)

        # Should pass all required gates
        assert evaluator.passes_all_required(results) is True

        # But win rate gate should fail
        win_rate_result = next(r for r in results if r.gate_type == GateType.MIN_WIN_RATE)
        assert win_rate_result.passed is False
        assert win_rate_result.required is False

        # passes_all should return False since optional gate failed
        assert evaluator.passes_all(results) is False

    def test_custom_gate_config(self) -> None:
        """Test using custom gate configuration."""
        custom_gates = [
            GateConfig(GateType.MIN_SHARPE, 2.0, required=True),
            GateConfig(GateType.MAX_DRAWDOWN, 0.10, required=True),
            GateConfig(GateType.MIN_PROFIT_FACTOR, 1.5, required=False),
        ]

        evaluator = GateEvaluator(gates=custom_gates)

        metrics = {
            "sharpe": 1.8,  # Below custom 2.0 threshold
            "max_drawdown": 0.08,  # Below 10% threshold
            "profit_factor": 1.6,
        }

        results = evaluator.evaluate(metrics)

        # Sharpe should fail (1.8 < 2.0)
        sharpe_result = next(r for r in results if r.gate_type == GateType.MIN_SHARPE)
        assert sharpe_result.passed is False

        # Drawdown should pass (0.08 <= 0.10)
        dd_result = next(r for r in results if r.gate_type == GateType.MAX_DRAWDOWN)
        assert dd_result.passed is True

        # Profit factor should pass (1.6 >= 1.5)
        pf_result = next(r for r in results if r.gate_type == GateType.MIN_PROFIT_FACTOR)
        assert pf_result.passed is True

    def test_get_summary(
        self, evaluator: GateEvaluator, passing_metrics: dict[str, float | int]
    ) -> None:
        """Test getting evaluation summary."""
        results = evaluator.evaluate(passing_metrics)
        summary = evaluator.get_summary(results)

        assert summary["passed"] is True
        assert summary["total_count"] == 4  # Default has 4 gates
        assert summary["passed_count"] == 4
        assert summary["required_total"] == 3
        assert summary["required_passed"] == 3
        assert summary["optional_total"] == 1
        assert summary["optional_passed"] == 1
        assert "results" in summary
        assert len(summary["results"]) == 4

    def test_get_summary_with_failures(
        self, evaluator: GateEvaluator, failing_sharpe_metrics: dict[str, float | int]
    ) -> None:
        """Test summary with failed gates."""
        results = evaluator.evaluate(failing_sharpe_metrics)
        summary = evaluator.get_summary(results)

        assert summary["passed"] is False
        assert summary["passed_count"] < summary["total_count"]

    def test_add_gate(self, evaluator: GateEvaluator) -> None:
        """Test adding a new gate."""
        new_gate = GateConfig(GateType.MIN_PROFIT_FACTOR, 1.5, required=True)
        evaluator.add_gate(new_gate)

        assert len(evaluator.gates) == 5
        assert any(g.gate_type == GateType.MIN_PROFIT_FACTOR for g in evaluator.gates)

    def test_add_gate_replaces_existing(self, evaluator: GateEvaluator) -> None:
        """Test that adding gate with same type replaces existing."""
        original_count = len(evaluator.gates)

        # Add gate with same type but different threshold
        new_gate = GateConfig(GateType.MIN_SHARPE, 2.0, required=True)
        evaluator.add_gate(new_gate)

        # Count should stay the same
        assert len(evaluator.gates) == original_count

        # Threshold should be updated
        sharpe_gate = next(g for g in evaluator.gates if g.gate_type == GateType.MIN_SHARPE)
        assert sharpe_gate.threshold == 2.0

    def test_remove_gate(self, evaluator: GateEvaluator) -> None:
        """Test removing a gate."""
        original_count = len(evaluator.gates)

        result = evaluator.remove_gate(GateType.MIN_WIN_RATE)

        assert result is True
        assert len(evaluator.gates) == original_count - 1
        assert not any(g.gate_type == GateType.MIN_WIN_RATE for g in evaluator.gates)

    def test_remove_nonexistent_gate(self, evaluator: GateEvaluator) -> None:
        """Test removing a gate that doesn't exist."""
        result = evaluator.remove_gate(GateType.MIN_PROFIT_FACTOR)
        assert result is False

    def test_reset_to_defaults(self) -> None:
        """Test resetting gates to defaults."""
        custom_gates = [GateConfig(GateType.MIN_SHARPE, 5.0, required=True)]
        evaluator = GateEvaluator(gates=custom_gates)

        assert len(evaluator.gates) == 1

        evaluator.reset_to_defaults()

        assert len(evaluator.gates) == 4
        sharpe_gate = next(g for g in evaluator.gates if g.gate_type == GateType.MIN_SHARPE)
        assert sharpe_gate.threshold == 0.5

    def test_missing_metric(self, evaluator: GateEvaluator) -> None:
        """Test evaluation with missing metric."""
        metrics = {
            "sharpe": 1.5,
            # Missing max_drawdown, num_trades, win_rate
        }

        results = evaluator.evaluate(metrics)

        # Gates for missing metrics should fail
        failed_results = [r for r in results if not r.passed]
        assert len(failed_results) >= 3  # At least 3 missing metrics

        # Check that missing metric results have appropriate message
        for result in failed_results:
            if result.gate_type != GateType.MIN_SHARPE:
                assert "not found" in result.message.lower()

    def test_gate_result_to_dict(self) -> None:
        """Test GateResult.to_dict() method."""
        result = GateResult(
            gate_type=GateType.MIN_SHARPE,
            passed=True,
            actual_value=1.5,
            threshold=0.5,
            message="min_sharpe passed: 1.5000 >= 0.5000",
            required=True,
        )

        data = result.to_dict()

        assert data["gate_type"] == "min_sharpe"
        assert data["passed"] is True
        assert data["actual_value"] == 1.5
        assert data["threshold"] == 0.5
        assert data["required"] is True

    def test_gate_config_to_dict(self) -> None:
        """Test GateConfig.to_dict() method."""
        config = GateConfig(GateType.MIN_SHARPE, 0.5, required=True)

        data = config.to_dict()

        assert data["gate_type"] == "min_sharpe"
        assert data["threshold"] == 0.5
        assert data["required"] is True

    def test_gate_config_from_dict(self) -> None:
        """Test GateConfig.from_dict() method."""
        data = {"gate_type": "min_sharpe", "threshold": 0.5, "required": True}

        config = GateConfig.from_dict(data)

        assert config.gate_type == GateType.MIN_SHARPE
        assert config.threshold == 0.5
        assert config.required is True

    def test_cost_sensitivity_gate(self) -> None:
        """Test cost sensitivity gate evaluation."""
        gates = [GateConfig(GateType.COST_SENSITIVITY, 0.5, required=True)]
        evaluator = GateEvaluator(gates=gates)

        # Cost sensitivity should be low (actual <= threshold)
        passing_metrics = {"cost_sensitivity": 0.3}
        failing_metrics = {"cost_sensitivity": 0.8}

        passing_results = evaluator.evaluate(passing_metrics)
        failing_results = evaluator.evaluate(failing_metrics)

        assert passing_results[0].passed is True
        assert failing_results[0].passed is False

    def test_sortino_gate(self) -> None:
        """Test Sortino ratio gate evaluation."""
        gates = [GateConfig(GateType.MIN_SORTINO, 1.0, required=True)]
        evaluator = GateEvaluator(gates=gates)

        passing_metrics = {"sortino": 1.5}
        failing_metrics = {"sortino": 0.5}

        passing_results = evaluator.evaluate(passing_metrics)
        failing_results = evaluator.evaluate(failing_metrics)

        assert passing_results[0].passed is True
        assert failing_results[0].passed is False

    def test_gates_property_setter(self) -> None:
        """Test setting gates via property."""
        evaluator = GateEvaluator()
        new_gates = [GateConfig(GateType.MIN_SHARPE, 3.0, required=True)]

        evaluator.gates = new_gates

        assert len(evaluator.gates) == 1
        assert evaluator.gates[0].threshold == 3.0


class TestGateConfigAndResult:
    """Tests for GateConfig and GateResult dataclasses."""

    def test_gate_config_default_required(self) -> None:
        """Test that GateConfig defaults to required=True."""
        config = GateConfig(GateType.MIN_SHARPE, 0.5)
        assert config.required is True

    def test_gate_result_message_contains_values(self) -> None:
        """Test that result messages contain actual and threshold values."""
        evaluator = GateEvaluator()
        metrics = {"sharpe": 1.5, "max_drawdown": 0.10, "num_trades": 50, "win_rate": 0.55}

        results = evaluator.evaluate(metrics)

        for result in results:
            assert str(result.actual_value) in result.message or f"{result.actual_value:.4f}" in result.message
            assert str(result.threshold) in result.message or f"{result.threshold:.4f}" in result.message

    def test_all_gate_types_have_metric_keys(self) -> None:
        """Test that all GateTypes have corresponding metric keys."""
        for gate_type in GateType:
            assert gate_type in GateEvaluator.METRIC_KEYS, f"Missing metric key for {gate_type}"
