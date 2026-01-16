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

The PromotionManager provides a higher-level interface for managing
the full promotion lifecycle including state tracking and history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class PromotionState(str, Enum):
    """Strategy promotion lifecycle states."""

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"

    @classmethod
    def valid_transitions(cls) -> dict[PromotionState, list[PromotionState]]:
        """Return valid state transitions."""
        return {
            cls.CANDIDATE: [cls.SHADOW, cls.RETIRED],
            cls.SHADOW: [cls.PAPER, cls.RETIRED],
            cls.PAPER: [cls.LIVE, cls.SHADOW, cls.RETIRED],
            cls.LIVE: [cls.PAPER, cls.RETIRED],
            cls.RETIRED: [],
        }

    def can_transition_to(self, target: PromotionState) -> bool:
        """Check if transition to target state is valid."""
        return target in self.valid_transitions().get(self, [])


@dataclass
class PromotionDecision:
    """Result of a promotion decision with full context."""

    strategy_id: str
    from_state: PromotionState
    to_state: PromotionState
    approved: bool
    criteria_results: dict[str, bool]
    performance_metrics: dict[str, float]
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    auto_promoted: bool = False
    reviewer_id: str | None = None


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


class PromotionManager:
    """High-level promotion lifecycle manager.

    Manages the full promotion lifecycle including:
    - State transition validation
    - Promotion history tracking
    - Batch promotion operations
    - Criteria configuration per transition type
    """

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        registry_api_url: str = "http://localhost:8080",
    ) -> None:
        """Initialize the promotion manager.

        Args:
            criteria: Default promotion criteria.
            registry_api_url: URL of the registry API.
        """
        self.default_criteria = criteria or PromotionCriteria()
        self.registry_api_url = registry_api_url
        self._promoter = AutoPromoter(
            criteria=self.default_criteria,
            registry_api_url=registry_api_url,
        )
        self._decision_history: list[PromotionDecision] = []
        self._logger = logger.bind(component="promotion_manager")

    def get_state_from_string(self, state_str: str) -> PromotionState:
        """Convert string state to PromotionState enum.

        Args:
            state_str: State string (e.g., "shadow", "paper").

        Returns:
            Corresponding PromotionState enum value.

        Raises:
            ValueError: If state string is invalid.
        """
        try:
            return PromotionState(state_str.lower())
        except ValueError as e:
            raise ValueError(f"Invalid promotion state: {state_str}") from e

    def validate_transition(
        self,
        from_state: PromotionState,
        to_state: PromotionState,
    ) -> bool:
        """Validate if a state transition is allowed.

        Args:
            from_state: Current state.
            to_state: Target state.

        Returns:
            True if transition is valid, False otherwise.
        """
        return from_state.can_transition_to(to_state)

    async def evaluate_promotion(
        self,
        strategy_id: str,
        target_state: PromotionState | None = None,
    ) -> PromotionDecision:
        """Evaluate a strategy for promotion.

        Args:
            strategy_id: ID of strategy to evaluate.
            target_state: Optional target state (defaults to next in lifecycle).

        Returns:
            PromotionDecision with full evaluation results.
        """
        self._logger.info(
            "Evaluating promotion",
            strategy_id=strategy_id,
            target_state=target_state.value if target_state else None,
        )

        # Get current evaluation from AutoPromoter
        evaluation = await self._promoter.evaluate_promotion(strategy_id)

        # Convert to PromotionState
        try:
            from_state = self.get_state_from_string(evaluation.current_state)
        except ValueError:
            from_state = PromotionState.CANDIDATE

        # Determine target state
        if target_state is None:
            target_state = self._get_next_state(from_state)

        # Get performance metrics for decision
        performance = await self._promoter._get_shadow_performance(strategy_id)

        # Create decision
        decision = PromotionDecision(
            strategy_id=strategy_id,
            from_state=from_state,
            to_state=target_state,
            approved=evaluation.eligible and self.validate_transition(from_state, target_state),
            criteria_results=evaluation.criteria_met,
            performance_metrics={
                "sharpe": performance.get("sharpe", 0.0),
                "max_drawdown": performance.get("max_drawdown", 0.0),
                "num_trades": float(performance.get("num_trades", 0)),
                "shadow_days": float(performance.get("shadow_days", 0)),
            },
            reason=evaluation.message,
        )

        self._decision_history.append(decision)
        return decision

    async def promote(
        self,
        strategy_id: str,
        target_state: PromotionState | None = None,
        auto: bool = True,
        reviewer_id: str | None = None,
    ) -> PromotionDecision:
        """Execute a promotion if criteria are met.

        Args:
            strategy_id: ID of strategy to promote.
            target_state: Target state (optional).
            auto: Whether this is an automatic promotion.
            reviewer_id: Optional reviewer ID for manual promotions.

        Returns:
            PromotionDecision with result.
        """
        decision = await self.evaluate_promotion(strategy_id, target_state)
        decision.auto_promoted = auto
        decision.reviewer_id = reviewer_id

        if not decision.approved:
            self._logger.warning(
                "Promotion not approved",
                strategy_id=strategy_id,
                reason=decision.reason,
            )
            return decision

        # Execute promotion via AutoPromoter
        success = await self._promoter._call_registry_promote(
            strategy_id,
            decision.to_state.value,
        )

        if not success:
            decision.approved = False
            decision.reason = "Registry API promotion failed"
            self._logger.error(
                "Promotion execution failed",
                strategy_id=strategy_id,
            )
        else:
            self._logger.info(
                "Promotion executed",
                strategy_id=strategy_id,
                from_state=decision.from_state.value,
                to_state=decision.to_state.value,
            )

        return decision

    async def batch_evaluate(
        self,
        from_state: PromotionState = PromotionState.SHADOW,
    ) -> list[PromotionDecision]:
        """Evaluate all strategies in a given state for promotion.

        Args:
            from_state: State to filter strategies by.

        Returns:
            List of PromotionDecision for each strategy.
        """
        self._logger.info("Batch evaluating strategies", from_state=from_state.value)

        evaluations = await self._promoter.check_all_candidates()
        decisions: list[PromotionDecision] = []

        for eval_result in evaluations:
            try:
                current_state = self.get_state_from_string(eval_result.current_state)
            except ValueError:
                continue

            if current_state != from_state:
                continue

            # Get performance for this strategy
            performance = await self._promoter._get_shadow_performance(eval_result.strategy_id)

            target_state = self._get_next_state(current_state)
            decision = PromotionDecision(
                strategy_id=eval_result.strategy_id,
                from_state=current_state,
                to_state=target_state,
                approved=eval_result.eligible,
                criteria_results=eval_result.criteria_met,
                performance_metrics={
                    "sharpe": performance.get("sharpe", 0.0),
                    "max_drawdown": performance.get("max_drawdown", 0.0),
                    "num_trades": float(performance.get("num_trades", 0)),
                    "shadow_days": float(performance.get("shadow_days", 0)),
                },
                reason=eval_result.message,
            )
            decisions.append(decision)
            self._decision_history.append(decision)

        self._logger.info(
            "Batch evaluation complete",
            total=len(decisions),
            approved=sum(1 for d in decisions if d.approved),
        )

        return decisions

    async def batch_promote(
        self,
        from_state: PromotionState = PromotionState.SHADOW,
        max_promotions: int | None = None,
    ) -> list[PromotionDecision]:
        """Promote all eligible strategies from a given state.

        Args:
            from_state: State to filter strategies by.
            max_promotions: Maximum number of promotions to execute.

        Returns:
            List of PromotionDecision for executed promotions.
        """
        decisions = await self.batch_evaluate(from_state)
        approved = [d for d in decisions if d.approved]

        if max_promotions is not None:
            approved = approved[:max_promotions]

        results: list[PromotionDecision] = []
        for decision in approved:
            result = await self.promote(
                decision.strategy_id,
                decision.to_state,
                auto=True,
            )
            results.append(result)

        return results

    def get_decision_history(
        self,
        strategy_id: str | None = None,
        limit: int | None = None,
    ) -> list[PromotionDecision]:
        """Get promotion decision history.

        Args:
            strategy_id: Filter by strategy ID (optional).
            limit: Maximum number of decisions to return.

        Returns:
            List of PromotionDecision objects.
        """
        history = self._decision_history
        if strategy_id is not None:
            history = [d for d in history if d.strategy_id == strategy_id]
        if limit is not None:
            history = history[-limit:]
        return history

    def clear_history(self) -> None:
        """Clear the decision history."""
        self._decision_history.clear()

    def _get_next_state(self, current: PromotionState) -> PromotionState:
        """Get the next state in the promotion lifecycle.

        Args:
            current: Current state.

        Returns:
            Next promotion state.
        """
        state_order = [
            PromotionState.CANDIDATE,
            PromotionState.SHADOW,
            PromotionState.PAPER,
            PromotionState.LIVE,
        ]
        try:
            idx = state_order.index(current)
            if idx < len(state_order) - 1:
                return state_order[idx + 1]
        except ValueError:
            pass
        return PromotionState.RETIRED

    async def close(self) -> None:
        """Close resources."""
        await self._promoter.close()


__all__ = [
    "AutoPromoter",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionEvaluation",
    "PromotionManager",
    "PromotionState",
]
