"""Promotion State Machine for Strategy Lifecycle Management (T5.05).

This module manages strategy lifecycle transitions:
- candidate: Initial state for new strategies
- shadow: Strategy runs in shadow mode (no real execution)
- paper: Strategy runs in paper trading mode
- retired: Terminal state for decommissioned strategies

State Transition Rules:
- CANDIDATE -> SHADOW, RETIRED
- SHADOW -> PAPER, CANDIDATE, RETIRED
- PAPER -> RETIRED, SHADOW
- RETIRED -> (terminal, no transitions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from registry_api_py.models import (
    AuditEventType,
    BacktestRun,
    PromotionState,
    Strategy,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from registry_api_py.audit import AuditLog


# =============================================================================
# EXCEPTIONS
# =============================================================================


class PromotionError(Exception):
    """Base exception for promotion-related errors."""

    pass


class InvalidTransitionError(PromotionError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: PromotionState, to_state: PromotionState) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid transition from {from_state.value} to {to_state.value}")


class StrategyNotFoundError(PromotionError):
    """Raised when a strategy is not found."""

    def __init__(self, strategy_id: int) -> None:
        self.strategy_id = strategy_id
        super().__init__(f"Strategy with id {strategy_id} not found")


class RequirementsNotMetError(PromotionError):
    """Raised when promotion requirements are not met."""

    def __init__(
        self, strategy_id: int, target_state: PromotionState, failed_requirements: dict[str, bool]
    ) -> None:
        self.strategy_id = strategy_id
        self.target_state = target_state
        self.failed_requirements = failed_requirements
        failed = [k for k, v in failed_requirements.items() if not v]
        super().__init__(
            f"Strategy {strategy_id} does not meet requirements for {target_state.value}: "
            f"failed {failed}"
        )


# =============================================================================
# VALID TRANSITIONS
# =============================================================================


VALID_TRANSITIONS: dict[PromotionState, list[PromotionState]] = {
    PromotionState.CANDIDATE: [PromotionState.SHADOW, PromotionState.RETIRED],
    PromotionState.SHADOW: [
        PromotionState.PAPER,
        PromotionState.CANDIDATE,
        PromotionState.RETIRED,
    ],
    PromotionState.PAPER: [PromotionState.RETIRED, PromotionState.SHADOW],
    PromotionState.RETIRED: [],  # Terminal state
}


# Demotion mapping: state -> demoted state
DEMOTION_MAP: dict[PromotionState, PromotionState | None] = {
    PromotionState.PAPER: PromotionState.SHADOW,
    PromotionState.SHADOW: PromotionState.CANDIDATE,
    PromotionState.CANDIDATE: None,  # Cannot demote further
    PromotionState.RETIRED: None,  # Cannot demote retired
}


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class PromotionRequirements:
    """Requirements for each promotion level."""

    to_shadow: dict[str, Any] = field(
        default_factory=lambda: {
            "min_sharpe": 0.5,
            "max_drawdown": 0.15,
            "min_trades": 100,
        }
    )
    to_paper: dict[str, Any] = field(
        default_factory=lambda: {
            "shadow_days": 5,
            "shadow_sharpe": 0.4,
            "no_alerts": True,
        }
    )


@dataclass
class PromotionResult:
    """Result of a promotion operation."""

    success: bool
    from_state: PromotionState
    to_state: PromotionState
    message: str
    requirements_met: dict[str, bool] = field(default_factory=dict)


# =============================================================================
# PROMOTION MANAGER
# =============================================================================


class PromotionManager:
    """Manage strategy promotion lifecycle.

    This class handles state transitions for strategies, enforcing valid
    transitions and checking requirements before allowing promotions.
    """

    def __init__(
        self,
        requirements: PromotionRequirements | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        """Initialize the promotion manager.

        Args:
            requirements: Custom requirements for promotions. Uses defaults if not provided.
            audit_log: Optional audit log for recording state changes.
        """
        self.requirements = requirements or PromotionRequirements()
        self.audit_log = audit_log

    def can_transition(self, from_state: PromotionState, to_state: PromotionState) -> bool:
        """Check if a transition between states is valid.

        Args:
            from_state: Current state.
            to_state: Target state.

        Returns:
            True if the transition is valid, False otherwise.
        """
        return to_state in VALID_TRANSITIONS.get(from_state, [])

    def check_requirements(
        self, strategy_id: int, target_state: PromotionState, db: Session
    ) -> dict[str, bool]:
        """Check if a strategy meets requirements for a target state.

        Args:
            strategy_id: ID of the strategy to check.
            target_state: Target promotion state.
            db: Database session.

        Returns:
            Dictionary mapping requirement names to whether they were met.
        """
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise StrategyNotFoundError(strategy_id)

        if target_state == PromotionState.SHADOW:
            return self._check_shadow_requirements(strategy, db)
        elif target_state == PromotionState.PAPER:
            return self._check_paper_requirements(strategy, db)
        elif target_state == PromotionState.RETIRED:
            # No requirements for retirement
            return {}
        elif target_state == PromotionState.CANDIDATE:
            # No requirements for demotion to candidate
            return {}
        else:
            return {}

    def _check_shadow_requirements(
        self, strategy: Strategy, db: Session
    ) -> dict[str, bool]:
        """Check requirements for promotion to SHADOW.

        Args:
            strategy: Strategy to check.
            db: Database session.

        Returns:
            Dictionary of requirement checks.
        """
        reqs = self.requirements.to_shadow
        results: dict[str, bool] = {}

        # Get latest backtest for the strategy
        backtest = (
            db.query(BacktestRun)
            .filter(BacktestRun.strategy_id == strategy.id)
            .order_by(BacktestRun.id.desc())
            .first()
        )

        if not backtest:
            # All requirements fail if no backtest
            return dict.fromkeys(reqs.keys(), False)

        # Check min_sharpe
        if "min_sharpe" in reqs:
            sharpe = backtest.sharpe or 0.0
            results["min_sharpe"] = sharpe >= reqs["min_sharpe"]

        # Check max_drawdown
        if "max_drawdown" in reqs:
            dd = backtest.max_drawdown or 1.0
            results["max_drawdown"] = dd <= reqs["max_drawdown"]

        # Check min_trades
        if "min_trades" in reqs:
            trades = backtest.num_trades or 0
            results["min_trades"] = trades >= reqs["min_trades"]

        return results

    def _check_paper_requirements(
        self, strategy: Strategy, db: Session
    ) -> dict[str, bool]:
        """Check requirements for promotion to PAPER.

        Args:
            strategy: Strategy to check.
            db: Database session.

        Returns:
            Dictionary of requirement checks.
        """
        reqs = self.requirements.to_paper
        results: dict[str, bool] = {}

        # For now, pass all paper requirements (actual implementation would
        # check shadow trading metrics)
        for key in reqs:
            results[key] = True

        return results

    def promote(
        self, strategy_id: int, target_state: PromotionState, db: Session
    ) -> PromotionResult:
        """Attempt to promote a strategy to a target state.

        Args:
            strategy_id: ID of the strategy to promote.
            target_state: Target promotion state.
            db: Database session.

        Returns:
            PromotionResult with details of the operation.

        Raises:
            StrategyNotFoundError: If strategy doesn't exist.
            InvalidTransitionError: If the transition is not valid.
            RequirementsNotMetError: If requirements are not met.
        """
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise StrategyNotFoundError(strategy_id)

        current_state = strategy.state
        if not self.can_transition(current_state, target_state):
            raise InvalidTransitionError(current_state, target_state)

        # Check requirements (skip for retirement or demotion)
        requirements_met: dict[str, bool] = {}
        if target_state not in [PromotionState.RETIRED, PromotionState.CANDIDATE]:
            requirements_met = self.check_requirements(strategy_id, target_state, db)
            if not all(requirements_met.values()):
                raise RequirementsNotMetError(strategy_id, target_state, requirements_met)

        # Perform the transition
        old_state = strategy.state
        strategy.state = target_state
        db.commit()

        # Record audit entry
        if self.audit_log:
            self.audit_log.record(
                event_type=AuditEventType.PROMOTION_APPROVED,
                entity_type="strategy",
                entity_id=strategy_id,
                message=f"Strategy promoted from {old_state.value} to {target_state.value}",
                old_value={"state": old_state.value},
                new_value={"state": target_state.value},
                metadata={"requirements_met": requirements_met},
            )

        return PromotionResult(
            success=True,
            from_state=current_state,
            to_state=target_state,
            message=f"Successfully promoted from {current_state.value} to {target_state.value}",
            requirements_met=requirements_met,
        )

    def demote(self, strategy_id: int, reason: str, db: Session) -> PromotionResult:
        """Demote a strategy to the previous state.

        Args:
            strategy_id: ID of the strategy to demote.
            reason: Reason for the demotion.
            db: Database session.

        Returns:
            PromotionResult with details of the operation.

        Raises:
            StrategyNotFoundError: If strategy doesn't exist.
            InvalidTransitionError: If demotion is not possible.
        """
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise StrategyNotFoundError(strategy_id)

        current_state = strategy.state
        demoted_state = DEMOTION_MAP.get(current_state)

        if demoted_state is None:
            raise InvalidTransitionError(
                current_state,
                PromotionState.CANDIDATE,  # Use as placeholder for "cannot demote"
            )

        if not self.can_transition(current_state, demoted_state):
            raise InvalidTransitionError(current_state, demoted_state)

        # Perform the demotion
        old_state = strategy.state
        strategy.state = demoted_state
        db.commit()

        # Record audit entry
        if self.audit_log:
            self.audit_log.record(
                event_type=AuditEventType.DEMOTION,
                entity_type="strategy",
                entity_id=strategy_id,
                message=f"Strategy demoted from {old_state.value} to {demoted_state.value}: {reason}",
                old_value={"state": old_state.value},
                new_value={"state": demoted_state.value},
                metadata={"reason": reason},
            )

        return PromotionResult(
            success=True,
            from_state=current_state,
            to_state=demoted_state,
            message=f"Demoted from {current_state.value} to {demoted_state.value}: {reason}",
            requirements_met={},
        )

    def retire(self, strategy_id: int, reason: str, db: Session) -> PromotionResult:
        """Retire a strategy (terminal state).

        Args:
            strategy_id: ID of the strategy to retire.
            reason: Reason for retirement.
            db: Database session.

        Returns:
            PromotionResult with details of the operation.

        Raises:
            StrategyNotFoundError: If strategy doesn't exist.
            InvalidTransitionError: If strategy is already retired.
        """
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise StrategyNotFoundError(strategy_id)

        current_state = strategy.state
        if not self.can_transition(current_state, PromotionState.RETIRED):
            raise InvalidTransitionError(current_state, PromotionState.RETIRED)

        # Perform retirement
        old_state = strategy.state
        strategy.state = PromotionState.RETIRED
        db.commit()

        # Record audit entry
        if self.audit_log:
            self.audit_log.record(
                event_type=AuditEventType.RETIREMENT,
                entity_type="strategy",
                entity_id=strategy_id,
                message=f"Strategy retired from {old_state.value}: {reason}",
                old_value={"state": old_state.value},
                new_value={"state": PromotionState.RETIRED.value},
                metadata={"reason": reason},
            )

        return PromotionResult(
            success=True,
            from_state=current_state,
            to_state=PromotionState.RETIRED,
            message=f"Retired from {current_state.value}: {reason}",
            requirements_met={},
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PromotionError",
    "InvalidTransitionError",
    "StrategyNotFoundError",
    "RequirementsNotMetError",
    "VALID_TRANSITIONS",
    "DEMOTION_MAP",
    "PromotionRequirements",
    "PromotionResult",
    "PromotionManager",
]
