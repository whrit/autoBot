"""
Candidate Ranking Module for Strategy Optimization (T4.09).

Provides ranking and filtering of strategy candidates based on performance metrics.
Supports ranking by individual metrics or weighted composite scores.

Key Features:
- Composite score calculation with configurable weights
- Ranking within strategy families
- Ranking across all families
- Filtering by minimum trade count and maximum drawdown
- Deterministic ranking (no random tie-breaking)

Example usage:
    >>> from optimizer_py.ranking import CandidateRanker, RankingConfig
    >>> config = RankingConfig(min_trades=50, max_drawdown_threshold=0.15)
    >>> ranker = CandidateRanker(config)
    >>> ranked = ranker.rank_across_families(candidates)
    >>> top_3 = ranker.get_top_candidates(candidates, n=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RankingMetric(str, Enum):
    """Metrics for ranking strategy candidates."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    PROFIT_FACTOR = "profit_factor"
    COMPOSITE = "composite"


@dataclass
class CandidateScore:
    """
    Performance scores for a strategy candidate.

    Attributes:
        strategy_id: Unique identifier for the strategy
        family: Strategy family name (e.g., 'trend', 'mean_reversion')
        params: Strategy parameters dictionary
        sharpe: Annualized Sharpe ratio
        sortino: Annualized Sortino ratio
        max_drawdown: Maximum drawdown as decimal (0.15 = 15%)
        profit_factor: Gross profit / gross loss
        win_rate: Percentage of winning trades (0.55 = 55%)
        num_trades: Total number of trades in evaluation period
        composite_score: Weighted composite score (computed by ranker)
    """

    strategy_id: str
    family: str
    params: dict[str, Any]
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    num_trades: int
    composite_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateScore:
        """
        Create CandidateScore from a dictionary.

        Args:
            data: Dictionary with candidate data

        Returns:
            CandidateScore instance
        """
        return cls(
            strategy_id=data["strategy_id"],
            family=data["family"],
            params=data.get("params", {}),
            sharpe=float(data["sharpe"]),
            sortino=float(data["sortino"]),
            max_drawdown=float(data["max_drawdown"]),
            profit_factor=float(data["profit_factor"]),
            win_rate=float(data["win_rate"]),
            num_trades=int(data["num_trades"]),
            composite_score=float(data.get("composite_score", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert CandidateScore to dictionary.

        Returns:
            Dictionary representation of the candidate
        """
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "params": self.params,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "composite_score": self.composite_score,
        }


@dataclass
class RankingConfig:
    """
    Configuration for candidate ranking.

    Attributes:
        metric: Primary metric for ranking (default: composite)
        min_trades: Minimum number of trades required (default: 30)
        max_drawdown_threshold: Maximum acceptable drawdown (default: 0.20)
        sharpe_weight: Weight for Sharpe ratio in composite score (default: 0.4)
        sortino_weight: Weight for Sortino ratio (default: 0.3)
        profit_factor_weight: Weight for profit factor (default: 0.2)
        drawdown_penalty_weight: Penalty weight for drawdown (default: 0.1)
    """

    metric: RankingMetric = RankingMetric.COMPOSITE
    min_trades: int = 30
    max_drawdown_threshold: float = 0.20
    sharpe_weight: float = 0.4
    sortino_weight: float = 0.3
    profit_factor_weight: float = 0.2
    drawdown_penalty_weight: float = 0.1


class CandidateRanker:
    """
    Rank strategy candidates within and across families.

    Provides methods for:
    - Computing composite scores from multiple metrics
    - Ranking candidates within a single family
    - Ranking candidates across all families
    - Filtering candidates by minimum criteria
    - Selecting top N candidates

    The ranking is always deterministic - candidates with equal scores
    are ordered by strategy_id to ensure reproducibility.
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        """
        Initialize the CandidateRanker.

        Args:
            config: Ranking configuration (uses defaults if not provided)
        """
        self.config = config or RankingConfig()

    def compute_composite_score(self, metrics: dict[str, float]) -> float:
        """
        Compute weighted composite score from performance metrics.

        The composite score combines multiple metrics with configurable weights:
        - Sharpe ratio (positive contribution)
        - Sortino ratio (positive contribution)
        - Profit factor (positive contribution)
        - Max drawdown (negative contribution as penalty)

        Args:
            metrics: Dictionary with keys: sharpe, sortino, profit_factor, max_drawdown

        Returns:
            Weighted composite score
        """
        sharpe = metrics.get("sharpe", 0.0)
        sortino = metrics.get("sortino", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)
        max_drawdown = metrics.get("max_drawdown", 0.0)

        score = (
            self.config.sharpe_weight * sharpe
            + self.config.sortino_weight * sortino
            + self.config.profit_factor_weight * profit_factor
            - self.config.drawdown_penalty_weight * max_drawdown
        )

        return score

    def _get_ranking_key(self, candidate: CandidateScore) -> float:
        """
        Get the ranking key based on the configured metric.

        Args:
            candidate: Candidate to get ranking key for

        Returns:
            Ranking key value (higher is better)
        """
        if self.config.metric == RankingMetric.SHARPE:
            return candidate.sharpe
        elif self.config.metric == RankingMetric.SORTINO:
            return candidate.sortino
        elif self.config.metric == RankingMetric.CALMAR:
            # Calmar = Sharpe / max_drawdown
            if candidate.max_drawdown > 0:
                return candidate.sharpe / candidate.max_drawdown
            return 0.0
        elif self.config.metric == RankingMetric.PROFIT_FACTOR:
            return candidate.profit_factor
        else:
            # COMPOSITE - use computed composite score
            return candidate.composite_score

    def _update_composite_scores(
        self, candidates: list[CandidateScore]
    ) -> list[CandidateScore]:
        """
        Update composite scores for all candidates.

        Args:
            candidates: List of candidates to update

        Returns:
            List of candidates with updated composite scores
        """
        for candidate in candidates:
            metrics = {
                "sharpe": candidate.sharpe,
                "sortino": candidate.sortino,
                "profit_factor": candidate.profit_factor,
                "max_drawdown": candidate.max_drawdown,
            }
            candidate.composite_score = self.compute_composite_score(metrics)

        return candidates

    def rank_within_family(
        self, candidates: list[CandidateScore]
    ) -> list[CandidateScore]:
        """
        Rank candidates within a single strategy family.

        Computes composite scores and sorts candidates in descending order.
        Uses strategy_id as secondary sort key for determinism.

        Args:
            candidates: List of candidates from the same family

        Returns:
            Sorted list of candidates (highest score first)
        """
        if not candidates:
            return []

        # Update composite scores
        self._update_composite_scores(candidates)

        # Sort by ranking key descending, then strategy_id for determinism
        ranked = sorted(
            candidates,
            key=lambda c: (-self._get_ranking_key(c), c.strategy_id),
        )

        return ranked

    def rank_across_families(
        self, candidates: list[CandidateScore]
    ) -> list[CandidateScore]:
        """
        Rank candidates across all strategy families.

        Computes composite scores and sorts all candidates regardless
        of family. Uses strategy_id as secondary sort key for determinism.

        Args:
            candidates: List of candidates from any family

        Returns:
            Sorted list of candidates (highest score first)
        """
        if not candidates:
            return []

        # Update composite scores
        self._update_composite_scores(candidates)

        # Sort by ranking key descending, then strategy_id for determinism
        ranked = sorted(
            candidates,
            key=lambda c: (-self._get_ranking_key(c), c.strategy_id),
        )

        return ranked

    def filter_candidates(
        self, candidates: list[CandidateScore]
    ) -> list[CandidateScore]:
        """
        Filter candidates that don't meet minimum criteria.

        Removes candidates that:
        - Have fewer trades than min_trades
        - Have higher drawdown than max_drawdown_threshold

        Args:
            candidates: List of candidates to filter

        Returns:
            List of candidates meeting all criteria
        """
        filtered = [
            c
            for c in candidates
            if c.num_trades >= self.config.min_trades
            and c.max_drawdown <= self.config.max_drawdown_threshold
        ]

        return filtered

    def get_top_candidates(
        self, candidates: list[CandidateScore], n: int = 10
    ) -> list[CandidateScore]:
        """
        Get top N candidates across all families.

        Args:
            candidates: List of candidates to rank
            n: Number of top candidates to return

        Returns:
            Top N candidates sorted by score
        """
        ranked = self.rank_across_families(candidates)
        return ranked[:n]

    def get_top_per_family(
        self, candidates: list[CandidateScore], n: int = 1
    ) -> list[CandidateScore]:
        """
        Get top N candidates from each family.

        Args:
            candidates: List of candidates to rank
            n: Number of top candidates per family

        Returns:
            Top N candidates from each family
        """
        # Group by family
        families: dict[str, list[CandidateScore]] = {}
        for candidate in candidates:
            if candidate.family not in families:
                families[candidate.family] = []
            families[candidate.family].append(candidate)

        # Get top N from each family
        result: list[CandidateScore] = []
        for _family_name, family_candidates in families.items():
            ranked = self.rank_within_family(family_candidates)
            result.extend(ranked[:n])

        return result

    def rank_and_filter(
        self, candidates: list[CandidateScore]
    ) -> list[CandidateScore]:
        """
        Filter and rank candidates in a single operation.

        First filters out candidates that don't meet criteria,
        then ranks the remaining candidates.

        Args:
            candidates: List of candidates to process

        Returns:
            Filtered and ranked list of candidates
        """
        filtered = self.filter_candidates(candidates)
        return self.rank_across_families(filtered)


# Public API exports
__all__ = [
    "RankingMetric",
    "CandidateScore",
    "RankingConfig",
    "CandidateRanker",
]
