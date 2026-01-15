"""Tests for the promotion state machine (T5.05).

This module tests the PromotionManager class and related functionality
for managing strategy lifecycle transitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.orm import Session

from registry_api_py.models import (
    BacktestRun,
    PromotionState,
    Strategy,
)
from registry_api_py.promotions import (
    VALID_TRANSITIONS,
    InvalidTransitionError,
    PromotionManager,
    PromotionRequirements,
    PromotionResult,
    RequirementsNotMetError,
    StrategyNotFoundError,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST VALID TRANSITIONS
# =============================================================================


class TestValidTransitions:
    """Tests for the transition validity rules."""

    def test_valid_transitions_from_candidate(self) -> None:
        """Test that CANDIDATE can transition to SHADOW or RETIRED."""
        assert PromotionState.SHADOW in VALID_TRANSITIONS[PromotionState.CANDIDATE]
        assert PromotionState.RETIRED in VALID_TRANSITIONS[PromotionState.CANDIDATE]
        assert len(VALID_TRANSITIONS[PromotionState.CANDIDATE]) == 2

    def test_valid_transitions_from_shadow(self) -> None:
        """Test that SHADOW can transition to PAPER, CANDIDATE, or RETIRED."""
        assert PromotionState.PAPER in VALID_TRANSITIONS[PromotionState.SHADOW]
        assert PromotionState.CANDIDATE in VALID_TRANSITIONS[PromotionState.SHADOW]
        assert PromotionState.RETIRED in VALID_TRANSITIONS[PromotionState.SHADOW]
        assert len(VALID_TRANSITIONS[PromotionState.SHADOW]) == 3

    def test_valid_transitions_from_paper(self) -> None:
        """Test that PAPER can transition to RETIRED or SHADOW."""
        assert PromotionState.RETIRED in VALID_TRANSITIONS[PromotionState.PAPER]
        assert PromotionState.SHADOW in VALID_TRANSITIONS[PromotionState.PAPER]
        assert len(VALID_TRANSITIONS[PromotionState.PAPER]) == 2

    def test_retired_is_terminal(self) -> None:
        """Test that RETIRED is a terminal state with no valid transitions."""
        assert VALID_TRANSITIONS[PromotionState.RETIRED] == []


# =============================================================================
# TEST PROMOTION MANAGER INITIALIZATION
# =============================================================================


class TestPromotionManagerInit:
    """Tests for PromotionManager initialization."""

    def test_init_with_default_requirements(self) -> None:
        """Test that manager initializes with default requirements."""
        manager = PromotionManager()
        assert manager.requirements is not None
        assert manager.requirements.to_shadow is not None
        assert manager.requirements.to_paper is not None

    def test_init_with_custom_requirements(self) -> None:
        """Test that manager initializes with custom requirements."""
        custom_reqs = PromotionRequirements(
            to_shadow={"min_sharpe": 0.8, "max_drawdown": 0.10},
            to_paper={"shadow_days": 10, "shadow_sharpe": 0.6},
        )
        manager = PromotionManager(requirements=custom_reqs)
        assert manager.requirements.to_shadow["min_sharpe"] == 0.8
        assert manager.requirements.to_paper["shadow_days"] == 10


# =============================================================================
# TEST CAN_TRANSITION METHOD
# =============================================================================


class TestCanTransition:
    """Tests for the can_transition method."""

    def test_candidate_to_shadow_valid(self) -> None:
        """Test that CANDIDATE -> SHADOW is valid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.CANDIDATE, PromotionState.SHADOW) is True

    def test_candidate_to_paper_invalid(self) -> None:
        """Test that CANDIDATE -> PAPER (skipping SHADOW) is invalid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.CANDIDATE, PromotionState.PAPER) is False

    def test_shadow_to_paper_valid(self) -> None:
        """Test that SHADOW -> PAPER is valid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.SHADOW, PromotionState.PAPER) is True

    def test_shadow_to_candidate_valid(self) -> None:
        """Test that SHADOW -> CANDIDATE (demotion) is valid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.SHADOW, PromotionState.CANDIDATE) is True

    def test_paper_to_shadow_valid(self) -> None:
        """Test that PAPER -> SHADOW (demotion) is valid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.PAPER, PromotionState.SHADOW) is True

    def test_paper_to_candidate_invalid(self) -> None:
        """Test that PAPER -> CANDIDATE (skipping SHADOW) is invalid."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.PAPER, PromotionState.CANDIDATE) is False

    def test_retired_to_anything_invalid(self) -> None:
        """Test that RETIRED cannot transition to any state."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.RETIRED, PromotionState.CANDIDATE) is False
        assert manager.can_transition(PromotionState.RETIRED, PromotionState.SHADOW) is False
        assert manager.can_transition(PromotionState.RETIRED, PromotionState.PAPER) is False

    def test_any_to_retired_valid(self) -> None:
        """Test that any state can transition to RETIRED."""
        manager = PromotionManager()
        assert manager.can_transition(PromotionState.CANDIDATE, PromotionState.RETIRED) is True
        assert manager.can_transition(PromotionState.SHADOW, PromotionState.RETIRED) is True
        assert manager.can_transition(PromotionState.PAPER, PromotionState.RETIRED) is True


# =============================================================================
# TEST PROMOTE METHOD
# =============================================================================


class TestPromote:
    """Tests for the promote method."""

    def test_promote_candidate_to_shadow_success(self, db_session: Session) -> None:
        """Test successful promotion from CANDIDATE to SHADOW."""
        # Create strategy with good backtest results
        strategy = Strategy(
            name="good_strategy",
            family="trend",
            version="1.0.0",
            parameters={"lookback": 20},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Add passing backtest
        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.8,
            max_drawdown=0.10,
            num_trades=150,
        )
        db_session.add(backtest)
        db_session.commit()

        manager = PromotionManager()
        result = manager.promote(strategy.id, PromotionState.SHADOW, db_session)

        assert result.success is True
        assert result.from_state == PromotionState.CANDIDATE
        assert result.to_state == PromotionState.SHADOW
        assert strategy.state == PromotionState.SHADOW

    def test_promote_shadow_to_paper_success(self, db_session: Session) -> None:
        """Test successful promotion from SHADOW to PAPER."""
        strategy = Strategy(
            name="shadow_to_paper",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager(
            requirements=PromotionRequirements(
                to_shadow={"min_sharpe": 0.5},
                to_paper={"shadow_sharpe": 0.4},  # Relaxed for test
            )
        )
        result = manager.promote(strategy.id, PromotionState.PAPER, db_session)

        assert result.success is True
        assert result.from_state == PromotionState.SHADOW
        assert result.to_state == PromotionState.PAPER
        assert strategy.state == PromotionState.PAPER

    def test_promote_invalid_transition_raises(self, db_session: Session) -> None:
        """Test that invalid transition raises InvalidTransitionError."""
        strategy = Strategy(
            name="invalid_transition",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        with pytest.raises(InvalidTransitionError):
            manager.promote(strategy.id, PromotionState.PAPER, db_session)

    def test_promote_nonexistent_strategy_raises(self, db_session: Session) -> None:
        """Test that promoting non-existent strategy raises StrategyNotFoundError."""
        manager = PromotionManager()
        with pytest.raises(StrategyNotFoundError):
            manager.promote(99999, PromotionState.SHADOW, db_session)

    def test_promote_requirements_not_met(self, db_session: Session) -> None:
        """Test that promotion fails when requirements not met."""
        strategy = Strategy(
            name="failing_strategy",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Add failing backtest (sharpe too low)
        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.1,  # Below threshold
            max_drawdown=0.30,  # Above threshold
            num_trades=50,  # Below threshold
        )
        db_session.add(backtest)
        db_session.commit()

        manager = PromotionManager()
        with pytest.raises(RequirementsNotMetError):
            manager.promote(strategy.id, PromotionState.SHADOW, db_session)


# =============================================================================
# TEST DEMOTE METHOD
# =============================================================================


class TestDemote:
    """Tests for the demote method."""

    def test_demote_paper_to_shadow(self, db_session: Session) -> None:
        """Test demotion from PAPER to SHADOW."""
        strategy = Strategy(
            name="demote_paper",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.PAPER,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        result = manager.demote(strategy.id, "Performance degradation", db_session)

        assert result.success is True
        assert result.from_state == PromotionState.PAPER
        assert result.to_state == PromotionState.SHADOW
        assert "Performance degradation" in result.message
        assert strategy.state == PromotionState.SHADOW

    def test_demote_shadow_to_candidate(self, db_session: Session) -> None:
        """Test demotion from SHADOW to CANDIDATE."""
        strategy = Strategy(
            name="demote_shadow",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        result = manager.demote(strategy.id, "Shadow testing failed", db_session)

        assert result.success is True
        assert result.from_state == PromotionState.SHADOW
        assert result.to_state == PromotionState.CANDIDATE
        assert strategy.state == PromotionState.CANDIDATE

    def test_demote_candidate_cannot_demote(self, db_session: Session) -> None:
        """Test that CANDIDATE cannot be demoted further."""
        strategy = Strategy(
            name="demote_candidate",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        with pytest.raises(InvalidTransitionError):
            manager.demote(strategy.id, "Already at lowest state", db_session)

    def test_demote_retired_cannot_demote(self, db_session: Session) -> None:
        """Test that RETIRED cannot be demoted."""
        strategy = Strategy(
            name="demote_retired",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.RETIRED,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        with pytest.raises(InvalidTransitionError):
            manager.demote(strategy.id, "Cannot demote retired", db_session)


# =============================================================================
# TEST RETIRE METHOD
# =============================================================================


class TestRetire:
    """Tests for the retire method."""

    def test_retire_from_candidate(self, db_session: Session) -> None:
        """Test retirement from CANDIDATE state."""
        strategy = Strategy(
            name="retire_candidate",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        result = manager.retire(strategy.id, "No longer viable", db_session)

        assert result.success is True
        assert result.from_state == PromotionState.CANDIDATE
        assert result.to_state == PromotionState.RETIRED
        assert strategy.state == PromotionState.RETIRED

    def test_retire_from_shadow(self, db_session: Session) -> None:
        """Test retirement from SHADOW state."""
        strategy = Strategy(
            name="retire_shadow",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        result = manager.retire(strategy.id, "Shadow testing concluded", db_session)

        assert result.success is True
        assert result.to_state == PromotionState.RETIRED
        assert strategy.state == PromotionState.RETIRED

    def test_retire_from_paper(self, db_session: Session) -> None:
        """Test retirement from PAPER state."""
        strategy = Strategy(
            name="retire_paper",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.PAPER,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        result = manager.retire(strategy.id, "Paper trading concluded", db_session)

        assert result.success is True
        assert result.to_state == PromotionState.RETIRED
        assert strategy.state == PromotionState.RETIRED

    def test_retire_already_retired(self, db_session: Session) -> None:
        """Test that retiring already retired strategy raises error."""
        strategy = Strategy(
            name="already_retired",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.RETIRED,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        with pytest.raises(InvalidTransitionError):
            manager.retire(strategy.id, "Cannot retire twice", db_session)


# =============================================================================
# TEST CHECK_REQUIREMENTS METHOD
# =============================================================================


class TestCheckRequirements:
    """Tests for the check_requirements method."""

    def test_check_requirements_to_shadow_all_pass(self, db_session: Session) -> None:
        """Test requirements check when all criteria pass."""
        strategy = Strategy(
            name="check_reqs_pass",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=1.0,  # Above min 0.5
            max_drawdown=0.10,  # Below max 0.15
            num_trades=200,  # Above min 100
        )
        db_session.add(backtest)
        db_session.commit()

        manager = PromotionManager()
        results = manager.check_requirements(strategy.id, PromotionState.SHADOW, db_session)

        assert results["min_sharpe"] is True
        assert results["max_drawdown"] is True
        assert results["min_trades"] is True

    def test_check_requirements_to_shadow_some_fail(self, db_session: Session) -> None:
        """Test requirements check when some criteria fail."""
        strategy = Strategy(
            name="check_reqs_fail",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.3,  # Below min 0.5
            max_drawdown=0.20,  # Above max 0.15
            num_trades=50,  # Below min 100
        )
        db_session.add(backtest)
        db_session.commit()

        manager = PromotionManager()
        results = manager.check_requirements(strategy.id, PromotionState.SHADOW, db_session)

        assert results["min_sharpe"] is False
        assert results["max_drawdown"] is False
        assert results["min_trades"] is False

    def test_check_requirements_no_backtest(self, db_session: Session) -> None:
        """Test requirements check when no backtest exists."""
        strategy = Strategy(
            name="no_backtest",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        manager = PromotionManager()
        results = manager.check_requirements(strategy.id, PromotionState.SHADOW, db_session)

        # All should fail when no backtest
        assert all(v is False for v in results.values())


# =============================================================================
# TEST PROMOTION RESULT DATACLASS
# =============================================================================


class TestPromotionResult:
    """Tests for the PromotionResult dataclass."""

    def test_promotion_result_success(self) -> None:
        """Test creating a successful promotion result."""
        result = PromotionResult(
            success=True,
            from_state=PromotionState.CANDIDATE,
            to_state=PromotionState.SHADOW,
            message="Promotion successful",
            requirements_met={"min_sharpe": True, "max_drawdown": True},
        )
        assert result.success is True
        assert result.from_state == PromotionState.CANDIDATE
        assert result.to_state == PromotionState.SHADOW

    def test_promotion_result_failure(self) -> None:
        """Test creating a failed promotion result."""
        result = PromotionResult(
            success=False,
            from_state=PromotionState.CANDIDATE,
            to_state=PromotionState.SHADOW,
            message="Requirements not met",
            requirements_met={"min_sharpe": False, "max_drawdown": True},
        )
        assert result.success is False
        assert "min_sharpe" in result.requirements_met


# =============================================================================
# TEST PROMOTION REQUIREMENTS DATACLASS
# =============================================================================


class TestPromotionRequirements:
    """Tests for the PromotionRequirements dataclass."""

    def test_promotion_requirements_init(self) -> None:
        """Test creating promotion requirements."""
        reqs = PromotionRequirements(
            to_shadow={"min_sharpe": 0.5, "max_drawdown": 0.15},
            to_paper={"shadow_days": 5, "shadow_sharpe": 0.4},
        )
        assert reqs.to_shadow["min_sharpe"] == 0.5
        assert reqs.to_paper["shadow_days"] == 5

    def test_default_requirements(self) -> None:
        """Test default promotion requirements."""
        manager = PromotionManager()
        assert "min_sharpe" in manager.requirements.to_shadow
        assert "max_drawdown" in manager.requirements.to_shadow
