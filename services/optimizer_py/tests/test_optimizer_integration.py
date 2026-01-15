"""
Integration tests for optimizer_py service (T4.11).

Tests cover:
- Full optimization pipeline from candidates to ranking
- Integration between ranking and cost sensitivity
- Best candidate selection across workflows
- End-to-end strategy evaluation flow
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from optimizer_py.cost_sensitivity import (
    CostSensitivityConfig,
    CostSensitivityTester,
)
from optimizer_py.ranking import (
    CandidateRanker,
    CandidateScore,
    RankingConfig,
)

# --- Mock Components for Integration Testing ---


@dataclass
class MockBacktestResult:
    """Mock backtest result for integration testing."""

    total_pnl: float
    total_trades: int
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    fills: list[dict] = field(default_factory=list)


class MockBacktesterForIntegration:
    """
    Mock backtester that generates realistic results for integration tests.

    Results vary based on strategy ID and cost parameters.
    """

    def __init__(self, base_results: dict[str, dict[str, float]] | None = None) -> None:
        """
        Initialize with base results per strategy.

        Args:
            base_results: Dict mapping strategy_id to base metrics
        """
        self.base_results = base_results or {}
        self.run_count = 0

    def run(
        self,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
        spread_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
        strategy_id: str = "default",
    ) -> MockBacktestResult:
        """Run mock backtest."""
        self.run_count += 1

        # Get base metrics for strategy or use defaults
        base = self.base_results.get(
            strategy_id,
            {
                "pnl": 5000.0,
                "trades": 50,
                "sharpe": 1.5,
                "sortino": 2.0,
                "max_drawdown": 0.10,
                "profit_factor": 1.8,
                "win_rate": 0.55,
            },
        )

        # Apply cost degradation
        cost_factor = (spread_multiplier + slippage_multiplier) / 2
        pnl_mult = 1.0 - (cost_factor - 1.0) * 0.3
        sharpe_mult = 1.0 - (cost_factor - 1.0) * 0.25

        return MockBacktestResult(
            total_pnl=base["pnl"] * pnl_mult,
            total_trades=int(base["trades"]),
            sharpe=base["sharpe"] * sharpe_mult,
            sortino=base["sortino"] * sharpe_mult,
            max_drawdown=base["max_drawdown"] * (1 + (cost_factor - 1.0) * 0.1),
            profit_factor=base["profit_factor"] * pnl_mult,
            win_rate=base["win_rate"] * (1 - (cost_factor - 1.0) * 0.05),
            fills=[{"pnl": base["pnl"] * pnl_mult / base["trades"]}]
            * int(base["trades"]),
        )


@dataclass
class MockStrategy:
    """Mock strategy for integration testing."""

    strategy_id: str
    family: str
    params: dict = field(default_factory=dict)


class StrategyEvaluator:
    """
    Evaluator that combines backtesting with ranking and cost sensitivity.

    Represents a typical workflow in the optimizer service.
    """

    def __init__(
        self,
        backtester: MockBacktesterForIntegration,
        ranker: CandidateRanker,
        cost_tester: CostSensitivityTester,
    ) -> None:
        self.backtester = backtester
        self.ranker = ranker
        self.cost_tester = cost_tester

    def evaluate_strategy(
        self,
        strategy: MockStrategy,
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
    ) -> CandidateScore:
        """
        Evaluate a strategy and return a CandidateScore.

        Args:
            strategy: Strategy to evaluate
            decision_frame: Feature data
            quotes: Quote data

        Returns:
            CandidateScore with metrics from backtest
        """
        result = self.backtester.run(
            decision_frame,
            quotes,
            strategy_id=strategy.strategy_id,
        )

        return CandidateScore(
            strategy_id=strategy.strategy_id,
            family=strategy.family,
            params=strategy.params,
            sharpe=result.sharpe,
            sortino=result.sortino,
            max_drawdown=result.max_drawdown,
            profit_factor=result.profit_factor,
            win_rate=result.win_rate,
            num_trades=result.total_trades,
            composite_score=0.0,
        )

    def evaluate_and_rank(
        self,
        strategies: list[MockStrategy],
        decision_frame: pl.DataFrame,
        quotes: pl.DataFrame,
    ) -> list[CandidateScore]:
        """
        Evaluate multiple strategies and rank them.

        Args:
            strategies: List of strategies to evaluate
            decision_frame: Feature data
            quotes: Quote data

        Returns:
            Ranked list of CandidateScore
        """
        candidates = [
            self.evaluate_strategy(s, decision_frame, quotes) for s in strategies
        ]
        return self.ranker.rank_across_families(candidates)


# --- Integration Test Classes ---


class TestFullOptimizationPipeline:
    """Tests for the full optimization pipeline."""

    def test_full_optimization_pipeline(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test complete pipeline from strategies to ranked candidates."""
        # Setup components
        base_results = {
            "trend_ma_20": {
                "pnl": 6000.0,
                "trades": 80,
                "sharpe": 1.8,
                "sortino": 2.2,
                "max_drawdown": 0.08,
                "profit_factor": 2.0,
                "win_rate": 0.58,
            },
            "trend_ma_50": {
                "pnl": 4500.0,
                "trades": 60,
                "sharpe": 1.4,
                "sortino": 1.8,
                "max_drawdown": 0.12,
                "profit_factor": 1.6,
                "win_rate": 0.52,
            },
            "meanrev_bb": {
                "pnl": 5500.0,
                "trades": 90,
                "sharpe": 1.6,
                "sortino": 2.0,
                "max_drawdown": 0.10,
                "profit_factor": 1.8,
                "win_rate": 0.55,
            },
        }

        backtester = MockBacktesterForIntegration(base_results)
        ranker = CandidateRanker()
        cost_config = CostSensitivityConfig(
            spread_multipliers=[1.0, 1.5],
            slippage_multipliers=[1.0],
        )
        cost_tester = CostSensitivityTester(backtester, cost_config)

        evaluator = StrategyEvaluator(backtester, ranker, cost_tester)

        # Create strategies
        strategies = [
            MockStrategy("trend_ma_20", "trend", {"lookback": 20}),
            MockStrategy("trend_ma_50", "trend", {"lookback": 50}),
            MockStrategy("meanrev_bb", "mean_reversion", {"bands": 2.0}),
        ]

        # Run pipeline
        ranked = evaluator.evaluate_and_rank(
            strategies, sample_decision_frame, mock_quotes
        )

        # Verify results
        assert len(ranked) == 3
        # Best strategy should be first
        assert ranked[0].strategy_id == "trend_ma_20"
        # All should have computed composite scores
        assert all(c.composite_score != 0.0 for c in ranked)

    def test_pipeline_with_filtering(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test pipeline with candidate filtering."""
        base_results = {
            "good_strategy": {
                "pnl": 5000.0,
                "trades": 100,
                "sharpe": 1.5,
                "sortino": 2.0,
                "max_drawdown": 0.10,
                "profit_factor": 1.8,
                "win_rate": 0.55,
            },
            "low_trades": {
                "pnl": 3000.0,
                "trades": 15,  # Below threshold
                "sharpe": 2.0,
                "sortino": 2.5,
                "max_drawdown": 0.05,
                "profit_factor": 2.0,
                "win_rate": 0.65,
            },
            "high_drawdown": {
                "pnl": 4000.0,
                "trades": 80,
                "sharpe": 1.3,
                "sortino": 1.7,
                "max_drawdown": 0.30,  # Above threshold
                "profit_factor": 1.5,
                "win_rate": 0.50,
            },
        }

        backtester = MockBacktesterForIntegration(base_results)
        config = RankingConfig(min_trades=30, max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)
        cost_tester = CostSensitivityTester(backtester)

        evaluator = StrategyEvaluator(backtester, ranker, cost_tester)

        strategies = [
            MockStrategy("good_strategy", "test"),
            MockStrategy("low_trades", "test"),
            MockStrategy("high_drawdown", "test"),
        ]

        candidates = [
            evaluator.evaluate_strategy(s, sample_decision_frame, mock_quotes)
            for s in strategies
        ]

        # Filter candidates
        filtered = ranker.filter_candidates(candidates)

        # Only good_strategy should pass
        assert len(filtered) == 1
        assert filtered[0].strategy_id == "good_strategy"


class TestRankingToSensitivityFlow:
    """Tests for integration between ranking and cost sensitivity."""

    def test_strategy_to_ranking_flow(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test flow from strategy candidates to ranking."""
        ranker = CandidateRanker()

        # Convert dict candidates to CandidateScore objects
        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]

        # Rank candidates
        ranked = ranker.rank_across_families(candidates)

        # Verify ranking order
        assert len(ranked) == len(candidates)
        for i in range(len(ranked) - 1):
            assert ranked[i].composite_score >= ranked[i + 1].composite_score

    def test_top_candidate_cost_sensitivity(
        self,
        sample_candidate_scores: list[dict],
        sample_decision_frame: pl.DataFrame,
        mock_quotes: pl.DataFrame,
    ) -> None:
        """Test cost sensitivity analysis on top ranked candidate."""
        ranker = CandidateRanker()
        backtester = MockBacktesterForIntegration()
        config = CostSensitivityConfig(
            spread_multipliers=[1.0, 2.0, 3.0],
            slippage_multipliers=[1.0],
        )
        cost_tester = CostSensitivityTester(backtester, config)

        # Rank candidates
        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        ranked = ranker.rank_across_families(candidates)

        # Get top candidate
        top_candidate = ranked[0]

        # Run cost sensitivity on top candidate
        strategy = MockStrategy(
            top_candidate.strategy_id,
            top_candidate.family,
            top_candidate.params,
        )
        sensitivity_results = cost_tester.run_sensitivity_test(
            strategy, sample_decision_frame, mock_quotes
        )

        # Verify sensitivity results
        assert len(sensitivity_results) == 3
        assert all(r.strategy_id == top_candidate.strategy_id for r in sensitivity_results)

        # Calculate robustness
        robustness = cost_tester.robustness_score(sensitivity_results)
        assert 0 <= robustness <= 1


class TestBestCandidateSelection:
    """Tests for selecting best candidates."""

    def test_best_candidate_selection(
        self, multiple_families_candidates: list[dict]
    ) -> None:
        """Test selection of best candidate across all families."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in multiple_families_candidates
        ]

        # Get top candidate
        top = ranker.get_top_candidates(candidates, n=1)

        assert len(top) == 1
        # Top should have highest composite score
        all_ranked = ranker.rank_across_families(candidates)
        assert top[0].strategy_id == all_ranked[0].strategy_id

    def test_best_per_family_selection(
        self, multiple_families_candidates: list[dict]
    ) -> None:
        """Test selection of best candidate per family."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in multiple_families_candidates
        ]

        # Get top per family
        top_per_family = ranker.get_top_per_family(candidates, n=1)

        # Should have one per family
        families = {c["family"] for c in multiple_families_candidates}
        assert len(top_per_family) == len(families)

        # Each should be the best in its family
        for top_candidate in top_per_family:
            family_candidates = [
                c for c in candidates if c.family == top_candidate.family
            ]
            ranked_family = ranker.rank_within_family(family_candidates)
            assert top_candidate.strategy_id == ranked_family[0].strategy_id

    def test_selection_with_cost_robustness(
        self,
        multiple_families_candidates: list[dict],
        sample_decision_frame: pl.DataFrame,
        mock_quotes: pl.DataFrame,
    ) -> None:
        """Test selection considering cost robustness."""
        ranker = CandidateRanker()
        backtester = MockBacktesterForIntegration()
        cost_config = CostSensitivityConfig(
            spread_multipliers=[1.0, 1.5, 2.0],
            slippage_multipliers=[1.0],
        )
        cost_tester = CostSensitivityTester(backtester, cost_config)

        candidates = [
            CandidateScore.from_dict(c) for c in multiple_families_candidates
        ]

        # Get top candidates
        top_5 = ranker.get_top_candidates(candidates, n=5)

        # Evaluate cost robustness for each
        robustness_scores: dict[str, float] = {}
        for candidate in top_5:
            strategy = MockStrategy(
                candidate.strategy_id, candidate.family, candidate.params
            )
            results = cost_tester.run_sensitivity_test(
                strategy, sample_decision_frame, mock_quotes
            )
            robustness_scores[candidate.strategy_id] = cost_tester.robustness_score(
                results
            )

        # All should have valid robustness scores
        assert len(robustness_scores) == 5
        assert all(0 <= s <= 1 for s in robustness_scores.values())


class TestCompositeScoreIntegration:
    """Tests for composite score in integration scenarios."""

    def test_composite_score_calculation(self) -> None:
        """Test composite score calculation matches expected."""
        config = RankingConfig(
            sharpe_weight=0.4,
            sortino_weight=0.3,
            profit_factor_weight=0.2,
            drawdown_penalty_weight=0.1,
        )
        ranker = CandidateRanker(config)

        # Create candidate with known metrics
        candidate = CandidateScore(
            strategy_id="test",
            family="test",
            params={},
            sharpe=2.0,
            sortino=2.5,
            max_drawdown=0.10,
            profit_factor=1.5,
            win_rate=0.55,
            num_trades=100,
            composite_score=0.0,
        )

        # Rank to compute score
        ranked = ranker.rank_within_family([candidate])

        # Calculate expected score
        expected = 0.4 * 2.0 + 0.3 * 2.5 + 0.2 * 1.5 - 0.1 * 0.10
        assert abs(ranked[0].composite_score - expected) < 1e-6

    def test_ranking_config_weights(self) -> None:
        """Test that different weight configurations produce different rankings."""
        # Sharpe-dominant config
        sharpe_config = RankingConfig(
            sharpe_weight=0.8,
            sortino_weight=0.1,
            profit_factor_weight=0.05,
            drawdown_penalty_weight=0.05,
        )

        # Sortino-dominant config
        sortino_config = RankingConfig(
            sharpe_weight=0.1,
            sortino_weight=0.8,
            profit_factor_weight=0.05,
            drawdown_penalty_weight=0.05,
        )

        sharpe_ranker = CandidateRanker(sharpe_config)
        sortino_ranker = CandidateRanker(sortino_config)

        # Create candidates with different strengths
        high_sharpe = CandidateScore(
            strategy_id="high_sharpe",
            family="test",
            params={},
            sharpe=2.5,
            sortino=1.0,
            max_drawdown=0.15,
            profit_factor=1.5,
            win_rate=0.50,
            num_trades=100,
            composite_score=0.0,
        )

        high_sortino = CandidateScore(
            strategy_id="high_sortino",
            family="test",
            params={},
            sharpe=1.0,
            sortino=3.0,
            max_drawdown=0.10,
            profit_factor=1.8,
            win_rate=0.55,
            num_trades=100,
            composite_score=0.0,
        )

        candidates = [high_sharpe, high_sortino]

        # Rank with different configs
        sharpe_ranked = sharpe_ranker.rank_within_family(candidates.copy())
        sortino_ranked = sortino_ranker.rank_within_family(candidates.copy())

        # Different configs should produce different orderings
        assert sharpe_ranked[0].strategy_id == "high_sharpe"
        assert sortino_ranked[0].strategy_id == "high_sortino"


class TestFilterAndRankIntegration:
    """Tests for filter and rank integration."""

    def test_filter_removes_low_trade_count(
        self, sample_candidate_scores: list[dict], low_trade_candidate: dict
    ) -> None:
        """Test that filtering correctly removes low trade candidates before ranking."""
        config = RankingConfig(min_trades=30, max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        candidates.append(CandidateScore.from_dict(low_trade_candidate))

        # Filter and rank in one step
        result = ranker.rank_and_filter(candidates)

        # Low trade candidate should not be in results
        assert not any(c.strategy_id == "low_trades" for c in result)

    def test_filter_removes_high_drawdown(
        self, sample_candidate_scores: list[dict], high_drawdown_candidate: dict
    ) -> None:
        """Test that filtering correctly removes high drawdown candidates."""
        config = RankingConfig(min_trades=30, max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        candidates.append(CandidateScore.from_dict(high_drawdown_candidate))

        # Filter and rank
        result = ranker.rank_and_filter(candidates)

        # High drawdown candidate should not be in results
        assert not any(c.strategy_id == "high_dd" for c in result)

    def test_filter_preserves_order(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that filtering preserves ranking order."""
        config = RankingConfig(min_trades=50, max_drawdown_threshold=0.15)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]

        # Filter and rank
        result = ranker.rank_and_filter(candidates)

        # Result should be in descending score order
        for i in range(len(result) - 1):
            assert result[i].composite_score >= result[i + 1].composite_score


class TestEndToEndStrategyEvaluation:
    """Tests for end-to-end strategy evaluation flow."""

    def test_full_evaluation_flow(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test complete evaluation from strategies through ranking and cost analysis."""
        # Setup - define a set of strategies with known characteristics
        base_results = {
            "robust_trend": {
                "pnl": 8000.0,
                "trades": 100,
                "sharpe": 2.0,
                "sortino": 2.5,
                "max_drawdown": 0.08,
                "profit_factor": 2.2,
                "win_rate": 0.60,
            },
            "sensitive_meanrev": {
                "pnl": 10000.0,  # Higher base P&L
                "trades": 150,
                "sharpe": 2.2,
                "sortino": 2.8,
                "max_drawdown": 0.05,
                "profit_factor": 2.5,
                "win_rate": 0.65,
            },
            "mediocre_vol": {
                "pnl": 3000.0,
                "trades": 50,
                "sharpe": 1.0,
                "sortino": 1.2,
                "max_drawdown": 0.18,
                "profit_factor": 1.3,
                "win_rate": 0.48,
            },
        }

        backtester = MockBacktesterForIntegration(base_results)
        ranker = CandidateRanker()
        cost_config = CostSensitivityConfig(
            spread_multipliers=[1.0, 1.5, 2.0, 2.5, 3.0],
            slippage_multipliers=[1.0, 1.5],
        )
        cost_tester = CostSensitivityTester(backtester, cost_config)

        evaluator = StrategyEvaluator(backtester, ranker, cost_tester)

        strategies = [
            MockStrategy("robust_trend", "trend", {"lookback": 20}),
            MockStrategy("sensitive_meanrev", "mean_reversion", {"bands": 2.0}),
            MockStrategy("mediocre_vol", "volatility", {"atr": 1.5}),
        ]

        # Step 1: Evaluate all strategies
        candidates = [
            evaluator.evaluate_strategy(s, sample_decision_frame, mock_quotes)
            for s in strategies
        ]

        # Step 2: Rank candidates
        ranked = ranker.rank_across_families(candidates)

        # Step 3: Analyze cost sensitivity for top candidates
        robustness_results: dict[str, dict] = {}
        for candidate in ranked[:2]:  # Top 2
            strategy = MockStrategy(
                candidate.strategy_id, candidate.family, candidate.params
            )
            sensitivity_results = cost_tester.run_sensitivity_test(
                strategy, sample_decision_frame, mock_quotes
            )
            summary = cost_tester.get_sensitivity_summary(sensitivity_results)
            robustness_results[candidate.strategy_id] = summary

        # Verify complete flow
        assert len(ranked) == 3
        assert len(robustness_results) == 2

        # Top ranked strategies should have robustness analysis
        for strategy_id in robustness_results:
            assert "robustness_score" in robustness_results[strategy_id]
            assert 0 <= robustness_results[strategy_id]["robustness_score"] <= 1

    def test_evaluation_determinism(
        self, sample_decision_frame: pl.DataFrame, mock_quotes: pl.DataFrame
    ) -> None:
        """Test that evaluation produces deterministic results."""
        base_results = {
            f"strategy_{i}": {
                "pnl": 5000.0 + i * 100,
                "trades": 80,
                "sharpe": 1.5 + i * 0.1,
                "sortino": 2.0 + i * 0.1,
                "max_drawdown": 0.10,
                "profit_factor": 1.8,
                "win_rate": 0.55,
            }
            for i in range(5)
        }

        backtester = MockBacktesterForIntegration(base_results)
        ranker = CandidateRanker()
        cost_tester = CostSensitivityTester(backtester)
        evaluator = StrategyEvaluator(backtester, ranker, cost_tester)

        strategies = [
            MockStrategy(f"strategy_{i}", "test") for i in range(5)
        ]

        # Run evaluation twice
        ranked1 = evaluator.evaluate_and_rank(
            strategies, sample_decision_frame, mock_quotes
        )
        ranked2 = evaluator.evaluate_and_rank(
            strategies, sample_decision_frame, mock_quotes
        )

        # Results should be identical
        assert [c.strategy_id for c in ranked1] == [c.strategy_id for c in ranked2]
        assert [c.composite_score for c in ranked1] == [c.composite_score for c in ranked2]
