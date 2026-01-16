"""Automatic Shadow to Paper Promotion Manager (T6.06).

This module handles automatic promotion of strategies from shadow
to paper trading mode based on performance criteria:
- Minimum shadow trading days
- Minimum Sharpe ratio
- Maximum drawdown
- Minimum trade count
- No active alerts

The AutoPromoter evaluates shadow strategies and promotes eligible
ones to paper trading automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class PromotionCriteria:
    """Criteria for automatic shadow to paper promotion."""

    min_shadow_days: int = 5
    min_shadow_sharpe: float = 0.4
    max_shadow_drawdown: float = 0.10
    min_shadow_trades: int = 20
    no_alerts_required: bool = True


@dataclass
class PromotionEvaluation:
    """Result of evaluating a strategy for promotion."""

    strategy_id: str
    current_state: str
    target_state: str
    criteria_met: dict[str, bool]
    eligible: bool
    message: str


class AutoPromoter:
    """Manage automatic shadow to paper promotion.

    This class evaluates shadow strategies against promotion criteria
    and automatically promotes eligible strategies to paper trading.
    """

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        registry_api_url: str = "http://localhost:8080",
    ) -> None:
        """Initialize the auto promoter.

        Args:
            criteria: Promotion criteria. Uses defaults if not provided.
            registry_api_url: URL of the registry API.
        """
        self.criteria = criteria or PromotionCriteria()
        self.registry_api_url = registry_api_url
        self._http_client: httpx.AsyncClient | None = None
        self._logger = logger.bind(component="auto_promoter")

    async def evaluate_promotion(self, strategy_id: str) -> PromotionEvaluation:
        """Evaluate if a strategy is eligible for promotion.

        Args:
            strategy_id: ID of the strategy to evaluate.

        Returns:
            PromotionEvaluation with eligibility details.
        """
        self._logger.info("Evaluating strategy for promotion", strategy_id=strategy_id)

        # Check if strategy is in shadow state
        current_state = await self._get_strategy_state(strategy_id)
        if current_state != "shadow":
            return PromotionEvaluation(
                strategy_id=strategy_id,
                current_state=current_state,
                target_state="paper",
                criteria_met={},
                eligible=False,
                message=f"Strategy is not in shadow state (current: {current_state})",
            )

        # Get shadow performance metrics
        performance = await self._get_shadow_performance(strategy_id)

        # Get alerts for strategy
        alerts = await self._get_strategy_alerts(strategy_id)

        # Evaluate each criterion
        criteria_met: dict[str, bool] = {
            "min_shadow_days": self._check_shadow_days(performance.get("shadow_days", 0)),
            "min_shadow_sharpe": self._check_sharpe(performance.get("sharpe", 0.0)),
            "max_shadow_drawdown": self._check_drawdown(performance.get("max_drawdown", 1.0)),
            "min_shadow_trades": self._check_trades(performance.get("num_trades", 0)),
            "no_alerts": self._check_alerts(alerts),
        }

        # Determine overall eligibility
        eligible = all(criteria_met.values())

        # Generate message
        if eligible:
            message = "Strategy meets all promotion criteria"
        else:
            failed = [k for k, v in criteria_met.items() if not v]
            message = f"Strategy does not meet promotion criteria: {', '.join(failed)}"

        self._logger.info(
            "Evaluation complete",
            strategy_id=strategy_id,
            eligible=eligible,
            criteria_met=criteria_met,
        )

        return PromotionEvaluation(
            strategy_id=strategy_id,
            current_state="shadow",
            target_state="paper",
            criteria_met=criteria_met,
            eligible=eligible,
            message=message,
        )

    async def promote_strategy(self, strategy_id: str) -> bool:
        """Promote a strategy from shadow to paper.

        Args:
            strategy_id: ID of the strategy to promote.

        Returns:
            True if promotion was successful, False otherwise.
        """
        self._logger.info("Attempting to promote strategy", strategy_id=strategy_id)

        # First evaluate eligibility
        evaluation = await self.evaluate_promotion(strategy_id)

        if not evaluation.eligible:
            self._logger.warning(
                "Strategy not eligible for promotion",
                strategy_id=strategy_id,
                message=evaluation.message,
            )
            return False

        # Call registry API to promote
        success = await self._call_registry_promote(strategy_id, "paper")

        if success:
            self._logger.info(
                "Strategy promoted successfully",
                strategy_id=strategy_id,
                from_state="shadow",
                to_state="paper",
            )
        else:
            self._logger.error(
                "Failed to promote strategy",
                strategy_id=strategy_id,
            )

        return success

    async def check_all_candidates(self) -> list[PromotionEvaluation]:
        """Check all shadow strategies for promotion eligibility.

        Returns:
            List of PromotionEvaluation for each shadow strategy.
        """
        self._logger.info("Checking all shadow strategies for promotion")

        strategies = await self._get_shadow_strategies()
        evaluations: list[PromotionEvaluation] = []

        for strategy in strategies:
            strategy_id = str(strategy.get("id", ""))
            if strategy_id:
                evaluation = await self.evaluate_promotion(strategy_id)
                evaluations.append(evaluation)

        eligible_count = sum(1 for e in evaluations if e.eligible)
        self._logger.info(
            "Candidate check complete",
            total=len(evaluations),
            eligible=eligible_count,
        )

        return evaluations

    async def _get_strategy_state(self, strategy_id: str) -> str:
        """Get the current state of a strategy.

        Args:
            strategy_id: Strategy ID.

        Returns:
            Current state string.
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/strategies/{strategy_id}")
            response.raise_for_status()
            data = response.json()
            return str(data.get("state", "unknown"))
        except Exception as e:
            self._logger.error("Failed to get strategy state", error=str(e))
            return "unknown"

    async def _get_shadow_performance(self, strategy_id: str) -> dict[str, Any]:
        """Get shadow execution performance metrics.

        Args:
            strategy_id: Strategy ID.

        Returns:
            Dictionary of performance metrics.
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/strategies/{strategy_id}/shadow-performance")
            response.raise_for_status()
            return dict(response.json())
        except Exception as e:
            self._logger.error(
                "Failed to get shadow performance",
                strategy_id=strategy_id,
                error=str(e),
            )
            return {}

    async def _get_strategy_alerts(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get alerts for a strategy.

        Args:
            strategy_id: Strategy ID.

        Returns:
            List of alert dictionaries.
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/strategies/{strategy_id}/alerts")
            response.raise_for_status()
            return list(response.json())
        except Exception as e:
            self._logger.error(
                "Failed to get strategy alerts",
                strategy_id=strategy_id,
                error=str(e),
            )
            return []

    async def _get_shadow_strategies(self) -> list[dict[str, Any]]:
        """Get all strategies in shadow state.

        Returns:
            List of shadow strategy dictionaries.
        """
        try:
            client = await self._get_client()
            response = await client.get("/strategies", params={"state": "shadow"})
            response.raise_for_status()
            data = response.json()
            return list(data.get("items", []))
        except Exception as e:
            self._logger.error("Failed to get shadow strategies", error=str(e))
            return []

    async def _call_registry_promote(self, strategy_id: str, target_state: str) -> bool:
        """Call registry API to promote strategy.

        Args:
            strategy_id: Strategy ID.
            target_state: Target promotion state.

        Returns:
            True if successful, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"/strategies/{strategy_id}/promote",
                json={"target_state": target_state},
            )
            response.raise_for_status()
            result = response.json()
            return bool(result.get("success", False))
        except Exception as e:
            self._logger.error(
                "Failed to call registry promote",
                strategy_id=strategy_id,
                error=str(e),
            )
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            AsyncClient instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.registry_api_url,
                timeout=30.0,
            )
        return self._http_client

    def _check_shadow_days(self, days: int) -> bool:
        """Check if shadow days criterion is met.

        Args:
            days: Number of shadow trading days.

        Returns:
            True if criterion is met.
        """
        return days >= self.criteria.min_shadow_days

    def _check_sharpe(self, sharpe: float) -> bool:
        """Check if Sharpe ratio criterion is met.

        Args:
            sharpe: Sharpe ratio value.

        Returns:
            True if criterion is met.
        """
        return sharpe >= self.criteria.min_shadow_sharpe

    def _check_drawdown(self, drawdown: float) -> bool:
        """Check if drawdown criterion is met.

        Args:
            drawdown: Maximum drawdown value.

        Returns:
            True if criterion is met (drawdown is below threshold).
        """
        return drawdown <= self.criteria.max_shadow_drawdown

    def _check_trades(self, trades: int) -> bool:
        """Check if trade count criterion is met.

        Args:
            trades: Number of trades.

        Returns:
            True if criterion is met.
        """
        return trades >= self.criteria.min_shadow_trades

    def _check_alerts(self, alerts: list[dict[str, Any]]) -> bool:
        """Check if alerts criterion is met.

        Args:
            alerts: List of alerts.

        Returns:
            True if criterion is met (no alerts or alerts not required).
        """
        if not self.criteria.no_alerts_required:
            return True
        return len(alerts) == 0

    async def close(self) -> None:
        """Close HTTP client and clean up resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


__all__ = [
    "AutoPromoter",
    "PromotionCriteria",
    "PromotionEvaluation",
]
