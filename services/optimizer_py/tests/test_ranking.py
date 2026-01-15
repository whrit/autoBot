"""
Tests for candidate ranking module (T4.09).

Tests cover:
- Composite score calculation with weighted metrics
- Ranking within a single strategy family
- Ranking across multiple families
- Filtering candidates by trade count and drawdown thresholds
- Configuration of ranking weights
- Deterministic ranking (no random tie-breaking)
"""

from __future__ import annotations

from optimizer_py.ranking import (
    CandidateRanker,
    CandidateScore,
    RankingConfig,
    RankingMetric,
)


class TestCandidateScore:
    """Tests for CandidateScore dataclass."""

    def test_create_candidate_score(self) -> None:
        """Test creating a CandidateScore instance."""
        score = CandidateScore(
            strategy_id="test_strategy",
            family="trend",
            params={"lookback": 20},
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.10,
            profit_factor=1.8,
            win_rate=0.55,
            num_trades=100,
            composite_score=0.0,
        )

        assert score.strategy_id == "test_strategy"
        assert score.family == "trend"
        assert score.sharpe == 1.5
        assert score.sortino == 2.0
        assert score.max_drawdown == 0.10
        assert score.profit_factor == 1.8
        assert score.win_rate == 0.55
        assert score.num_trades == 100

    def test_candidate_score_from_dict(self) -> None:
        """Test creating CandidateScore from a dictionary."""
        data = {
            "strategy_id": "trend_ma_20",
            "family": "trend",
            "params": {"lookback": 20},
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

        score = CandidateScore.from_dict(data)

        assert score.strategy_id == "trend_ma_20"
        assert score.sharpe == 1.5
        assert score.composite_score == 0.0  # Default before computation


class TestRankingConfig:
    """Tests for RankingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RankingConfig()

        assert config.metric == RankingMetric.COMPOSITE
        assert config.min_trades == 30
        assert config.max_drawdown_threshold == 0.20
        assert config.sharpe_weight == 0.4
        assert config.sortino_weight == 0.3
        assert config.profit_factor_weight == 0.2
        assert config.drawdown_penalty_weight == 0.1

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        config = RankingConfig()
        total = (
            config.sharpe_weight
            + config.sortino_weight
            + config.profit_factor_weight
            + config.drawdown_penalty_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RankingConfig(
            metric=RankingMetric.SHARPE,
            min_trades=50,
            max_drawdown_threshold=0.15,
            sharpe_weight=0.5,
            sortino_weight=0.3,
            profit_factor_weight=0.15,
            drawdown_penalty_weight=0.05,
        )

        assert config.metric == RankingMetric.SHARPE
        assert config.min_trades == 50
        assert config.max_drawdown_threshold == 0.15


class TestCandidateRanker:
    """Tests for CandidateRanker class."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default configuration."""
        ranker = CandidateRanker()
        assert ranker.config.metric == RankingMetric.COMPOSITE

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = RankingConfig(min_trades=50, max_drawdown_threshold=0.15)
        ranker = CandidateRanker(config)

        assert ranker.config.min_trades == 50
        assert ranker.config.max_drawdown_threshold == 0.15


class TestCompositeScoreCalculation:
    """Tests for composite score calculation."""

    def test_compute_composite_score_basic(self) -> None:
        """Test basic composite score calculation."""
        config = RankingConfig(
            sharpe_weight=0.4,
            sortino_weight=0.3,
            profit_factor_weight=0.2,
            drawdown_penalty_weight=0.1,
        )
        ranker = CandidateRanker(config)

        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "profit_factor": 1.8,
            "max_drawdown": 0.10,
        }

        score = ranker.compute_composite_score(metrics)

        # Expected: 0.4*1.5 + 0.3*2.0 + 0.2*1.8 - 0.1*0.10
        # = 0.6 + 0.6 + 0.36 - 0.01 = 1.55
        expected = 0.4 * 1.5 + 0.3 * 2.0 + 0.2 * 1.8 - 0.1 * 0.10
        assert abs(score - expected) < 1e-6

    def test_composite_score_with_zero_metrics(self) -> None:
        """Test composite score when some metrics are zero."""
        ranker = CandidateRanker()
        metrics = {
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

        score = ranker.compute_composite_score(metrics)
        assert score == 0.0

    def test_composite_score_with_negative_sharpe(self) -> None:
        """Test composite score with negative Sharpe ratio."""
        ranker = CandidateRanker()
        metrics = {
            "sharpe": -0.5,
            "sortino": -0.3,
            "profit_factor": 0.8,
            "max_drawdown": 0.25,
        }

        score = ranker.compute_composite_score(metrics)
        # Score should be lower due to negative ratios
        assert score < 0

    def test_composite_score_high_drawdown_penalty(self) -> None:
        """Test that high drawdown significantly penalizes score."""
        ranker = CandidateRanker()

        low_dd_metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "profit_factor": 1.8,
            "max_drawdown": 0.05,
        }

        high_dd_metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "profit_factor": 1.8,
            "max_drawdown": 0.30,
        }

        low_dd_score = ranker.compute_composite_score(low_dd_metrics)
        high_dd_score = ranker.compute_composite_score(high_dd_metrics)

        assert low_dd_score > high_dd_score


class TestRankWithinFamily:
    """Tests for ranking candidates within a single family."""

    def test_rank_within_family_orders_by_score(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that ranking orders candidates by composite score descending."""
        ranker = CandidateRanker()

        # Filter to trend family only
        trend_candidates = [
            CandidateScore.from_dict(c)
            for c in sample_candidate_scores
            if c["family"] == "trend"
        ]

        ranked = ranker.rank_within_family(trend_candidates)

        # Check descending order
        for i in range(len(ranked) - 1):
            assert ranked[i].composite_score >= ranked[i + 1].composite_score

    def test_rank_within_family_updates_composite_score(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that composite scores are computed during ranking."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c)
            for c in sample_candidate_scores
            if c["family"] == "trend"
        ]

        ranked = ranker.rank_within_family(candidates)

        # All candidates should have non-default composite scores
        for candidate in ranked:
            # At least one non-zero metric means non-zero score
            if candidate.sharpe > 0 or candidate.sortino > 0:
                assert candidate.composite_score != 0.0

    def test_rank_within_family_empty_list(self) -> None:
        """Test ranking with empty candidate list."""
        ranker = CandidateRanker()
        ranked = ranker.rank_within_family([])
        assert ranked == []

    def test_rank_within_family_single_candidate(self) -> None:
        """Test ranking with single candidate."""
        ranker = CandidateRanker()

        candidate = CandidateScore(
            strategy_id="single",
            family="test",
            params={},
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.10,
            profit_factor=1.8,
            win_rate=0.55,
            num_trades=100,
            composite_score=0.0,
        )

        ranked = ranker.rank_within_family([candidate])

        assert len(ranked) == 1
        assert ranked[0].strategy_id == "single"

    def test_rank_within_family_deterministic(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that ranking is deterministic (no random tie-breaking)."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c)
            for c in sample_candidate_scores
            if c["family"] == "trend"
        ]

        ranked1 = ranker.rank_within_family(candidates)
        ranked2 = ranker.rank_within_family(candidates)

        # Same order each time
        assert [c.strategy_id for c in ranked1] == [c.strategy_id for c in ranked2]


class TestRankAcrossFamilies:
    """Tests for ranking candidates across all families."""

    def test_rank_across_families(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test ranking candidates across multiple families."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Should have all candidates
        assert len(ranked) == len(candidates)

        # Should be in descending score order
        for i in range(len(ranked) - 1):
            assert ranked[i].composite_score >= ranked[i + 1].composite_score

    def test_rank_across_families_preserves_family_info(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that family information is preserved after ranking."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Check that all original families are present
        original_families = {c["family"] for c in sample_candidate_scores}
        ranked_families = {c.family for c in ranked}

        assert original_families == ranked_families

    def test_rank_across_families_mixed_quality(
        self, multiple_families_candidates: list[dict]
    ) -> None:
        """Test ranking with candidates of varying quality across families."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in multiple_families_candidates
        ]

        ranked = ranker.rank_across_families(candidates)

        # Top candidate should have highest composite score
        top_candidate = ranked[0]
        assert all(top_candidate.composite_score >= c.composite_score for c in ranked)


class TestFilterCandidates:
    """Tests for filtering candidates by thresholds."""

    def test_filter_removes_low_trade_count(
        self, sample_candidate_scores: list[dict], low_trade_candidate: dict
    ) -> None:
        """Test that candidates with low trade count are filtered out."""
        config = RankingConfig(min_trades=30)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        candidates.append(CandidateScore.from_dict(low_trade_candidate))

        filtered = ranker.filter_candidates(candidates)

        # Low trade candidate should be removed
        assert all(c.num_trades >= 30 for c in filtered)
        assert not any(c.strategy_id == "low_trades" for c in filtered)

    def test_filter_removes_high_drawdown(
        self, sample_candidate_scores: list[dict], high_drawdown_candidate: dict
    ) -> None:
        """Test that candidates with high drawdown are filtered out."""
        config = RankingConfig(max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        candidates.append(CandidateScore.from_dict(high_drawdown_candidate))

        filtered = ranker.filter_candidates(candidates)

        # High drawdown candidate should be removed
        assert all(c.max_drawdown <= 0.20 for c in filtered)
        assert not any(c.strategy_id == "high_dd" for c in filtered)

    def test_filter_applies_both_thresholds(
        self,
        sample_candidate_scores: list[dict],
        low_trade_candidate: dict,
        high_drawdown_candidate: dict,
    ) -> None:
        """Test that both min_trades and max_drawdown filters are applied."""
        config = RankingConfig(min_trades=30, max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]
        candidates.append(CandidateScore.from_dict(low_trade_candidate))
        candidates.append(CandidateScore.from_dict(high_drawdown_candidate))

        filtered = ranker.filter_candidates(candidates)

        # Both problematic candidates should be removed
        assert not any(c.strategy_id in ["low_trades", "high_dd"] for c in filtered)

    def test_filter_with_strict_thresholds(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test filtering with very strict thresholds."""
        config = RankingConfig(min_trades=100, max_drawdown_threshold=0.10)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]

        filtered = ranker.filter_candidates(candidates)

        # Only candidates meeting both strict criteria should remain
        for c in filtered:
            assert c.num_trades >= 100
            assert c.max_drawdown <= 0.10

    def test_filter_preserves_passing_candidates(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test that candidates meeting thresholds are preserved."""
        config = RankingConfig(min_trades=50, max_drawdown_threshold=0.20)
        ranker = CandidateRanker(config)

        candidates = [CandidateScore.from_dict(c) for c in sample_candidate_scores]

        filtered = ranker.filter_candidates(candidates)

        # Count expected passing candidates
        expected_passing = [
            c for c in sample_candidate_scores
            if c["num_trades"] >= 50 and c["max_drawdown"] <= 0.20
        ]

        assert len(filtered) == len(expected_passing)


class TestRankingByMetric:
    """Tests for ranking by different metrics."""

    def test_rank_by_sharpe_only(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test ranking by Sharpe ratio only."""
        config = RankingConfig(metric=RankingMetric.SHARPE)
        ranker = CandidateRanker(config)

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Should be ordered by Sharpe descending
        for i in range(len(ranked) - 1):
            assert ranked[i].sharpe >= ranked[i + 1].sharpe

    def test_rank_by_sortino_only(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test ranking by Sortino ratio only."""
        config = RankingConfig(metric=RankingMetric.SORTINO)
        ranker = CandidateRanker(config)

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Should be ordered by Sortino descending
        for i in range(len(ranked) - 1):
            assert ranked[i].sortino >= ranked[i + 1].sortino

    def test_rank_by_calmar(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test ranking by Calmar ratio (Sharpe/max_drawdown)."""
        config = RankingConfig(metric=RankingMetric.CALMAR)
        ranker = CandidateRanker(config)

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Calculate expected Calmar ratios
        def calmar(c: CandidateScore) -> float:
            return c.sharpe / c.max_drawdown if c.max_drawdown > 0 else 0.0

        # Should be ordered by Calmar descending
        for i in range(len(ranked) - 1):
            assert calmar(ranked[i]) >= calmar(ranked[i + 1])

    def test_rank_by_profit_factor(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test ranking by profit factor only."""
        config = RankingConfig(metric=RankingMetric.PROFIT_FACTOR)
        ranker = CandidateRanker(config)

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        ranked = ranker.rank_across_families(candidates)

        # Should be ordered by profit factor descending
        for i in range(len(ranked) - 1):
            assert ranked[i].profit_factor >= ranked[i + 1].profit_factor


class TestGetTopCandidates:
    """Tests for getting top N candidates."""

    def test_get_top_n_candidates(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test getting top N candidates across all families."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        top_3 = ranker.get_top_candidates(candidates, n=3)

        assert len(top_3) == 3
        # Should be the 3 highest scoring candidates
        all_ranked = ranker.rank_across_families(candidates)
        assert [c.strategy_id for c in top_3] == [c.strategy_id for c in all_ranked[:3]]

    def test_get_top_n_per_family(
        self, multiple_families_candidates: list[dict]
    ) -> None:
        """Test getting top N candidates per family."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in multiple_families_candidates
        ]

        top_per_family = ranker.get_top_per_family(candidates, n=1)

        # Should have one candidate per family
        families = {c.family for c in top_per_family}
        expected_families = {c["family"] for c in multiple_families_candidates}
        assert families == expected_families

    def test_get_top_n_exceeds_available(
        self, sample_candidate_scores: list[dict]
    ) -> None:
        """Test getting more candidates than available."""
        ranker = CandidateRanker()

        candidates = [
            CandidateScore.from_dict(c) for c in sample_candidate_scores
        ]

        # Request more than available
        top_100 = ranker.get_top_candidates(candidates, n=100)

        # Should return all candidates
        assert len(top_100) == len(candidates)


class TestRankingConfigWeights:
    """Tests for different weight configurations."""

    def test_sharpe_dominant_weights(self) -> None:
        """Test ranking with Sharpe-dominant weight configuration."""
        config = RankingConfig(
            sharpe_weight=0.7,
            sortino_weight=0.2,
            profit_factor_weight=0.05,
            drawdown_penalty_weight=0.05,
        )
        ranker = CandidateRanker(config)

        # Create two candidates: one with high Sharpe, one with high Sortino
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

        ranked = ranker.rank_within_family([high_sharpe, high_sortino])

        # High Sharpe should win with Sharpe-dominant weights
        assert ranked[0].strategy_id == "high_sharpe"

    def test_balanced_weights(self) -> None:
        """Test ranking with balanced weight configuration."""
        config = RankingConfig(
            sharpe_weight=0.25,
            sortino_weight=0.25,
            profit_factor_weight=0.25,
            drawdown_penalty_weight=0.25,
        )
        ranker = CandidateRanker(config)

        # With balanced weights, the candidate with best overall metrics should win
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "profit_factor": 1.8,
            "max_drawdown": 0.10,
        }

        score = ranker.compute_composite_score(metrics)
        expected = 0.25 * 1.5 + 0.25 * 2.0 + 0.25 * 1.8 - 0.25 * 0.10
        assert abs(score - expected) < 1e-6
