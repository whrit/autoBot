"""
Tests for complexity penalties module.

Tests cover:
- Parameter count penalty
- Lookback period penalty
- ML model complexity penalty (tree depth, number of estimators)
- Combined complexity score
- Penalty application to ranking
"""

from __future__ import annotations

import pytest

from optimizer_py.complexity import (
    ComplexityPenaltyConfig,
    compute_complexity_penalty,
    compute_lookback_penalty,
    compute_ml_complexity_penalty,
    compute_parameter_count_penalty,
)
from optimizer_py.ranking import (
    CandidateRanker,
    CandidateScore,
    RankingConfig,
)


class TestComplexityPenaltyConfig:
    """Tests for ComplexityPenaltyConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ComplexityPenaltyConfig()

        assert config.max_parameters == 10
        assert config.max_lookback == 200
        assert config.max_estimators == 500
        assert config.max_depth == 12
        assert config.parameter_penalty_weight == 0.1
        assert config.lookback_penalty_weight == 0.1
        assert config.ml_penalty_weight == 0.1

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ComplexityPenaltyConfig(
            max_parameters=5,
            max_lookback=100,
            max_estimators=200,
            max_depth=8,
            parameter_penalty_weight=0.2,
            lookback_penalty_weight=0.15,
            ml_penalty_weight=0.25,
        )

        assert config.max_parameters == 5
        assert config.max_lookback == 100
        assert config.max_estimators == 200
        assert config.max_depth == 8
        assert config.parameter_penalty_weight == 0.2
        assert config.lookback_penalty_weight == 0.15
        assert config.ml_penalty_weight == 0.25


class TestParameterCountPenalty:
    """Tests for parameter count penalty calculation."""

    def test_no_penalty_for_few_parameters(self) -> None:
        """Test no penalty when parameter count is below threshold."""
        config = ComplexityPenaltyConfig(max_parameters=10)
        params = {"lookback": 20, "threshold": 0.02}

        penalty = compute_parameter_count_penalty(params, config)

        assert penalty == 0.0

    def test_penalty_for_many_parameters(self) -> None:
        """Test penalty increases with parameter count."""
        config = ComplexityPenaltyConfig(max_parameters=5, parameter_penalty_weight=0.1)
        params = {
            "param1": 1, "param2": 2, "param3": 3, "param4": 4,
            "param5": 5, "param6": 6, "param7": 7, "param8": 8,
        }

        penalty = compute_parameter_count_penalty(params, config)

        # 8 params > 5 max, so should have penalty
        assert penalty > 0.0

    def test_penalty_scales_with_excess_parameters(self) -> None:
        """Test penalty scales proportionally with excess parameters."""
        config = ComplexityPenaltyConfig(max_parameters=5, parameter_penalty_weight=0.1)

        params_6 = {f"p{i}": i for i in range(6)}
        params_10 = {f"p{i}": i for i in range(10)}

        penalty_6 = compute_parameter_count_penalty(params_6, config)
        penalty_10 = compute_parameter_count_penalty(params_10, config)

        assert penalty_10 > penalty_6

    def test_empty_params_no_penalty(self) -> None:
        """Test empty parameters dict has no penalty."""
        config = ComplexityPenaltyConfig()
        params: dict = {}

        penalty = compute_parameter_count_penalty(params, config)

        assert penalty == 0.0


class TestLookbackPenalty:
    """Tests for lookback period penalty calculation."""

    def test_no_penalty_for_short_lookback(self) -> None:
        """Test no penalty when lookback is below threshold."""
        config = ComplexityPenaltyConfig(max_lookback=200)
        params = {"lookback": 50}

        penalty = compute_lookback_penalty(params, config)

        assert penalty == 0.0

    def test_penalty_for_long_lookback(self) -> None:
        """Test penalty for excessive lookback period."""
        config = ComplexityPenaltyConfig(max_lookback=100, lookback_penalty_weight=0.1)
        params = {"lookback": 250}

        penalty = compute_lookback_penalty(params, config)

        assert penalty > 0.0

    def test_penalty_scales_with_lookback(self) -> None:
        """Test penalty scales with lookback period excess."""
        config = ComplexityPenaltyConfig(max_lookback=100, lookback_penalty_weight=0.1)

        params_150 = {"lookback": 150}
        params_300 = {"lookback": 300}

        penalty_150 = compute_lookback_penalty(params_150, config)
        penalty_300 = compute_lookback_penalty(params_300, config)

        assert penalty_300 > penalty_150

    def test_no_lookback_param_no_penalty(self) -> None:
        """Test no penalty when lookback parameter is missing."""
        config = ComplexityPenaltyConfig()
        params = {"threshold": 0.02}

        penalty = compute_lookback_penalty(params, config)

        assert penalty == 0.0

    def test_alternative_lookback_names(self) -> None:
        """Test penalty applies to alternative lookback parameter names."""
        config = ComplexityPenaltyConfig(max_lookback=100, lookback_penalty_weight=0.1)

        # Test with 'window' parameter
        params_window = {"window": 250}
        penalty_window = compute_lookback_penalty(params_window, config)
        assert penalty_window > 0.0

        # Test with 'period' parameter
        params_period = {"period": 250}
        penalty_period = compute_lookback_penalty(params_period, config)
        assert penalty_period > 0.0


class TestMLComplexityPenalty:
    """Tests for ML model complexity penalty calculation."""

    def test_no_penalty_for_simple_ml_model(self) -> None:
        """Test no penalty for simple ML model configuration."""
        config = ComplexityPenaltyConfig(
            max_estimators=500,
            max_depth=12,
        )
        params = {
            "n_estimators": 100,
            "max_depth": 6,
        }

        penalty = compute_ml_complexity_penalty(params, config)

        assert penalty == 0.0

    def test_penalty_for_many_estimators(self) -> None:
        """Test penalty for excessive number of estimators."""
        config = ComplexityPenaltyConfig(
            max_estimators=200,
            ml_penalty_weight=0.1,
        )
        params = {"n_estimators": 500}

        penalty = compute_ml_complexity_penalty(params, config)

        assert penalty > 0.0

    def test_penalty_for_deep_trees(self) -> None:
        """Test penalty for excessive tree depth."""
        config = ComplexityPenaltyConfig(
            max_depth=8,
            ml_penalty_weight=0.1,
        )
        params = {"max_depth": 15}

        penalty = compute_ml_complexity_penalty(params, config)

        assert penalty > 0.0

    def test_combined_ml_complexity_penalty(self) -> None:
        """Test combined penalty for multiple ML complexity factors."""
        config = ComplexityPenaltyConfig(
            max_estimators=200,
            max_depth=8,
            ml_penalty_weight=0.1,
        )
        params = {
            "n_estimators": 500,
            "max_depth": 15,
        }

        penalty = compute_ml_complexity_penalty(params, config)

        # Both factors should contribute to penalty
        penalty_estimators_only = compute_ml_complexity_penalty(
            {"n_estimators": 500}, config
        )
        penalty_depth_only = compute_ml_complexity_penalty(
            {"max_depth": 15}, config
        )

        assert penalty >= penalty_estimators_only
        assert penalty >= penalty_depth_only

    def test_non_ml_params_no_penalty(self) -> None:
        """Test no ML penalty for non-ML parameters."""
        config = ComplexityPenaltyConfig()
        params = {"lookback": 100, "threshold": 0.02}

        penalty = compute_ml_complexity_penalty(params, config)

        assert penalty == 0.0


class TestCombinedComplexityPenalty:
    """Tests for combined complexity score calculation."""

    def test_compute_total_complexity_penalty(self) -> None:
        """Test total complexity penalty combines all factors."""
        config = ComplexityPenaltyConfig(
            max_parameters=5,
            max_lookback=100,
            max_estimators=200,
            max_depth=8,
        )
        params = {
            "p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5, "p6": 6, "p7": 7,
            "lookback": 250,
            "n_estimators": 500,
            "max_depth": 15,
        }

        total_penalty = compute_complexity_penalty(params, config)

        # All factors should contribute
        assert total_penalty > 0.0

    def test_zero_penalty_for_simple_strategy(self) -> None:
        """Test zero total penalty for simple strategy."""
        config = ComplexityPenaltyConfig()
        params = {
            "lookback": 20,
            "threshold": 0.02,
        }

        total_penalty = compute_complexity_penalty(params, config)

        assert total_penalty == 0.0

    def test_penalty_bounded(self) -> None:
        """Test that penalty is bounded to reasonable range."""
        config = ComplexityPenaltyConfig()
        # Extremely complex params
        params = {f"p{i}": i for i in range(100)}
        params["lookback"] = 10000
        params["n_estimators"] = 10000
        params["max_depth"] = 100

        total_penalty = compute_complexity_penalty(params, config)

        # Penalty should be bounded (not infinite)
        assert total_penalty < 10.0  # Reasonable upper bound


class TestPenaltyApplicationToRanking:
    """Tests for integrating complexity penalties with ranking."""

    @pytest.fixture
    def simple_candidate(self) -> CandidateScore:
        """Create a simple strategy candidate."""
        return CandidateScore(
            strategy_id="simple_trend",
            family="trend",
            params={"lookback": 20, "threshold": 0.02},
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.10,
            profit_factor=1.8,
            win_rate=0.55,
            num_trades=100,
        )

    @pytest.fixture
    def complex_candidate(self) -> CandidateScore:
        """Create a complex strategy candidate with many parameters."""
        params = {f"param{i}": i for i in range(15)}
        params["lookback"] = 500
        params["n_estimators"] = 1000
        params["max_depth"] = 20
        return CandidateScore(
            strategy_id="complex_ml",
            family="ml",
            params=params,
            sharpe=1.6,  # Slightly higher raw performance
            sortino=2.1,
            max_drawdown=0.09,
            profit_factor=1.9,
            win_rate=0.56,
            num_trades=100,
        )

    def test_simple_strategy_preferred_over_complex(
        self, simple_candidate: CandidateScore, complex_candidate: CandidateScore
    ) -> None:
        """Test that simple strategy is preferred when complexity penalty applied."""
        config = ComplexityPenaltyConfig(
            max_parameters=10,
            max_lookback=200,
            max_estimators=500,
            max_depth=12,
            parameter_penalty_weight=0.1,
            lookback_penalty_weight=0.1,
            ml_penalty_weight=0.1,
        )

        simple_penalty = compute_complexity_penalty(simple_candidate.params, config)
        complex_penalty = compute_complexity_penalty(complex_candidate.params, config)

        # Complex strategy should have higher penalty
        assert complex_penalty > simple_penalty

        # Adjusted score for complex should be lower
        simple_adjusted = simple_candidate.sharpe - simple_penalty
        complex_adjusted = complex_candidate.sharpe - complex_penalty

        # Even though complex has higher raw sharpe, adjusted should be lower
        assert simple_adjusted > complex_adjusted

    def test_ranking_with_complexity_adjustment(
        self, simple_candidate: CandidateScore, complex_candidate: CandidateScore
    ) -> None:
        """Test ranking candidates with complexity-adjusted scores."""
        config = ComplexityPenaltyConfig()
        candidates = [complex_candidate, simple_candidate]

        # Apply complexity penalties to adjust scores
        adjusted_candidates = []
        for candidate in candidates:
            penalty = compute_complexity_penalty(candidate.params, config)
            adjusted = CandidateScore(
                strategy_id=candidate.strategy_id,
                family=candidate.family,
                params=candidate.params,
                sharpe=candidate.sharpe - penalty,
                sortino=candidate.sortino,
                max_drawdown=candidate.max_drawdown,
                profit_factor=candidate.profit_factor,
                win_rate=candidate.win_rate,
                num_trades=candidate.num_trades,
            )
            adjusted_candidates.append(adjusted)

        # Rank using standard ranker
        ranker = CandidateRanker(RankingConfig())
        ranked = ranker.rank_across_families(adjusted_candidates)

        # Simple strategy should rank first
        assert ranked[0].strategy_id == "simple_trend"

    def test_moderate_complexity_acceptable(self) -> None:
        """Test that moderate complexity doesn't over-penalize."""
        config = ComplexityPenaltyConfig()

        # Moderately complex params
        params = {
            "lookback": 150,
            "threshold": 0.03,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "n_estimators": 300,
            "max_depth": 8,
        }

        penalty = compute_complexity_penalty(params, config)

        # Penalty should be modest
        assert penalty < 0.5


class TestEdgeCases:
    """Edge case tests for complexity penalties."""

    def test_none_params_handling(self) -> None:
        """Test handling of None parameter values."""
        config = ComplexityPenaltyConfig()
        params = {"lookback": None, "threshold": 0.02}

        # Should not raise, just ignore None values
        penalty = compute_complexity_penalty(params, config)
        assert isinstance(penalty, float)

    def test_negative_values_handling(self) -> None:
        """Test handling of negative parameter values."""
        config = ComplexityPenaltyConfig()
        params = {"lookback": -50}  # Invalid but shouldn't crash

        penalty = compute_lookback_penalty(params, config)
        assert penalty == 0.0  # No penalty for values below threshold

    def test_string_param_values(self) -> None:
        """Test handling of string parameter values."""
        config = ComplexityPenaltyConfig()
        params = {
            "model_type": "xgboost",
            "lookback": 50,
        }

        # Should handle mixed types gracefully
        penalty = compute_complexity_penalty(params, config)
        assert isinstance(penalty, float)

    def test_empty_config_defaults(self) -> None:
        """Test that empty config uses sensible defaults."""
        config = ComplexityPenaltyConfig()
        params = {"lookback": 50}

        penalty = compute_complexity_penalty(params, config)
        assert penalty >= 0.0

    def test_zero_weight_disables_penalty(self) -> None:
        """Test that zero weight disables specific penalty."""
        config = ComplexityPenaltyConfig(
            parameter_penalty_weight=0.0,
            lookback_penalty_weight=0.0,
            ml_penalty_weight=0.0,
        )
        params = {
            "p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5,
            "p6": 6, "p7": 7, "p8": 8, "p9": 9, "p10": 10,
            "p11": 11, "p12": 12, "p13": 13, "p14": 14, "p15": 15,
            "lookback": 1000,
            "n_estimators": 5000,
            "max_depth": 50,
        }

        penalty = compute_complexity_penalty(params, config)

        assert penalty == 0.0
