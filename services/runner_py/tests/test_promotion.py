"""Tests for the AutoPromoter module (T6.06).

Tests cover:
- Promotion criteria configuration
- Promotion eligibility evaluation
- Shadow performance metric checking
- Alert checking
- Automatic promotion execution
- Candidate scanning
- PromotionState lifecycle
- PromotionDecision dataclass
- PromotionManager high-level operations
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runner_py.promotion import (
    AutoPromoter,
    PromotionCriteria,
    PromotionDecision,
    PromotionEvaluation,
    PromotionManager,
    PromotionState,
)


class TestPromotionCriteria:
    """Tests for PromotionCriteria dataclass."""

    def test_default_criteria(self) -> None:
        """Test default promotion criteria values."""
        criteria = PromotionCriteria()

        assert criteria.min_shadow_days == 5
        assert criteria.min_shadow_sharpe == 0.4
        assert criteria.max_shadow_drawdown == 0.10
        assert criteria.min_shadow_trades == 20
        assert criteria.no_alerts_required is True

    def test_custom_criteria(self) -> None:
        """Test custom promotion criteria values."""
        criteria = PromotionCriteria(
            min_shadow_days=10,
            min_shadow_sharpe=0.6,
            max_shadow_drawdown=0.05,
            min_shadow_trades=50,
            no_alerts_required=False,
        )

        assert criteria.min_shadow_days == 10
        assert criteria.min_shadow_sharpe == 0.6
        assert criteria.max_shadow_drawdown == 0.05
        assert criteria.min_shadow_trades == 50
        assert criteria.no_alerts_required is False

    def test_lenient_criteria(self) -> None:
        """Test more lenient criteria for testing environments."""
        criteria = PromotionCriteria(
            min_shadow_days=1,
            min_shadow_sharpe=0.0,
            max_shadow_drawdown=1.0,
            min_shadow_trades=1,
            no_alerts_required=False,
        )

        assert criteria.min_shadow_days == 1
        assert criteria.min_shadow_sharpe == 0.0


class TestPromotionEvaluation:
    """Tests for PromotionEvaluation dataclass."""

    def test_evaluation_creation(self) -> None:
        """Test promotion evaluation creation."""
        eval_result = PromotionEvaluation(
            strategy_id="strategy_123",
            current_state="shadow",
            target_state="paper",
            criteria_met={
                "min_shadow_days": True,
                "min_shadow_sharpe": True,
                "max_shadow_drawdown": True,
                "min_shadow_trades": True,
                "no_alerts": True,
            },
            eligible=True,
            message="Strategy meets all promotion criteria",
        )

        assert eval_result.strategy_id == "strategy_123"
        assert eval_result.current_state == "shadow"
        assert eval_result.target_state == "paper"
        assert eval_result.eligible is True
        assert all(eval_result.criteria_met.values())

    def test_evaluation_not_eligible(self) -> None:
        """Test promotion evaluation when not eligible."""
        eval_result = PromotionEvaluation(
            strategy_id="strategy_456",
            current_state="shadow",
            target_state="paper",
            criteria_met={
                "min_shadow_days": True,
                "min_shadow_sharpe": False,  # Failed
                "max_shadow_drawdown": True,
                "min_shadow_trades": False,  # Failed
                "no_alerts": True,
            },
            eligible=False,
            message="Strategy does not meet criteria: min_shadow_sharpe, min_shadow_trades",
        )

        assert eval_result.eligible is False
        assert not eval_result.criteria_met["min_shadow_sharpe"]
        assert not eval_result.criteria_met["min_shadow_trades"]


class TestAutoPromoter:
    """Tests for AutoPromoter class."""

    def test_promoter_init_default_criteria(self) -> None:
        """Test promoter initialization with default criteria."""
        promoter = AutoPromoter()

        assert promoter.criteria.min_shadow_days == 5
        assert promoter.criteria.min_shadow_sharpe == 0.4
        assert promoter.registry_api_url == "http://localhost:8080"

    def test_promoter_init_custom_criteria(self) -> None:
        """Test promoter initialization with custom criteria."""
        criteria = PromotionCriteria(
            min_shadow_days=7,
            min_shadow_sharpe=0.5,
        )
        promoter = AutoPromoter(
            criteria=criteria,
            registry_api_url="http://registry:9000",
        )

        assert promoter.criteria.min_shadow_days == 7
        assert promoter.criteria.min_shadow_sharpe == 0.5
        assert promoter.registry_api_url == "http://registry:9000"

    async def test_evaluate_promotion_eligible(self) -> None:
        """Test evaluation when strategy is eligible for promotion."""
        criteria = PromotionCriteria(
            min_shadow_days=5,
            min_shadow_sharpe=0.4,
            max_shadow_drawdown=0.10,
            min_shadow_trades=20,
            no_alerts_required=True,
        )
        promoter = AutoPromoter(criteria=criteria)

        # Mock excellent shadow performance
        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.8,
            "max_drawdown": 0.05,
            "num_trades": 50,
            "started_at": (datetime.now() - timedelta(days=10)).isoformat(),
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is True
            assert evaluation.criteria_met["min_shadow_days"] is True
            assert evaluation.criteria_met["min_shadow_sharpe"] is True
            assert evaluation.criteria_met["max_shadow_drawdown"] is True
            assert evaluation.criteria_met["min_shadow_trades"] is True
            assert evaluation.criteria_met["no_alerts"] is True

    async def test_evaluate_promotion_not_eligible_low_sharpe(self) -> None:
        """Test evaluation when Sharpe ratio is too low."""
        criteria = PromotionCriteria(min_shadow_sharpe=0.4)
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.2,  # Below threshold
            "max_drawdown": 0.05,
            "num_trades": 50,
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert evaluation.criteria_met["min_shadow_sharpe"] is False

    async def test_evaluate_promotion_not_eligible_high_drawdown(self) -> None:
        """Test evaluation when drawdown exceeds threshold."""
        criteria = PromotionCriteria(max_shadow_drawdown=0.10)
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.8,
            "max_drawdown": 0.15,  # Above threshold
            "num_trades": 50,
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert evaluation.criteria_met["max_shadow_drawdown"] is False

    async def test_evaluate_promotion_not_eligible_insufficient_trades(self) -> None:
        """Test evaluation when trade count is insufficient."""
        criteria = PromotionCriteria(min_shadow_trades=20)
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.8,
            "max_drawdown": 0.05,
            "num_trades": 10,  # Below threshold
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert evaluation.criteria_met["min_shadow_trades"] is False

    async def test_evaluate_promotion_not_eligible_has_alerts(self) -> None:
        """Test evaluation when strategy has alerts."""
        criteria = PromotionCriteria(no_alerts_required=True)
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.8,
            "max_drawdown": 0.05,
            "num_trades": 50,
        }

        mock_alerts = [
            {"id": 1, "type": "drawdown_exceeded", "timestamp": datetime.now().isoformat()}
        ]

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=mock_alerts
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert evaluation.criteria_met["no_alerts"] is False

    async def test_evaluate_promotion_insufficient_shadow_days(self) -> None:
        """Test evaluation when shadow period is too short."""
        criteria = PromotionCriteria(min_shadow_days=5)
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 3,  # Below threshold
            "sharpe": 0.8,
            "max_drawdown": 0.05,
            "num_trades": 50,
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert evaluation.criteria_met["min_shadow_days"] is False

    async def test_promote_strategy_success(self) -> None:
        """Test successful strategy promotion."""
        promoter = AutoPromoter()

        with patch.object(
            promoter, "evaluate_promotion"
        ) as mock_eval, patch.object(
            promoter, "_call_registry_promote", return_value=True
        ) as mock_promote:
            mock_eval.return_value = PromotionEvaluation(
                strategy_id="1",
                current_state="shadow",
                target_state="paper",
                criteria_met={"all": True},
                eligible=True,
                message="Eligible",
            )

            result = await promoter.promote_strategy("1")

            assert result is True
            mock_promote.assert_called_once_with("1", "paper")

    async def test_promote_strategy_not_eligible(self) -> None:
        """Test promotion attempt when not eligible."""
        promoter = AutoPromoter()

        with patch.object(promoter, "evaluate_promotion") as mock_eval:
            mock_eval.return_value = PromotionEvaluation(
                strategy_id="1",
                current_state="shadow",
                target_state="paper",
                criteria_met={"sharpe": False},
                eligible=False,
                message="Not eligible",
            )

            result = await promoter.promote_strategy("1")

            assert result is False

    async def test_check_all_candidates(self) -> None:
        """Test checking all shadow strategies for promotion."""
        promoter = AutoPromoter()

        mock_strategies = [
            {"id": 1, "name": "strategy_1", "state": "shadow"},
            {"id": 2, "name": "strategy_2", "state": "shadow"},
            {"id": 3, "name": "strategy_3", "state": "shadow"},
        ]

        with patch.object(
            promoter, "_get_shadow_strategies", return_value=mock_strategies
        ), patch.object(
            promoter, "evaluate_promotion"
        ) as mock_eval:
            # First two are eligible, third is not
            mock_eval.side_effect = [
                PromotionEvaluation(
                    strategy_id="1",
                    current_state="shadow",
                    target_state="paper",
                    criteria_met={"all": True},
                    eligible=True,
                    message="Eligible",
                ),
                PromotionEvaluation(
                    strategy_id="2",
                    current_state="shadow",
                    target_state="paper",
                    criteria_met={"all": True},
                    eligible=True,
                    message="Eligible",
                ),
                PromotionEvaluation(
                    strategy_id="3",
                    current_state="shadow",
                    target_state="paper",
                    criteria_met={"sharpe": False},
                    eligible=False,
                    message="Not eligible",
                ),
            ]

            evaluations = await promoter.check_all_candidates()

            assert len(evaluations) == 3
            assert evaluations[0].eligible is True
            assert evaluations[1].eligible is True
            assert evaluations[2].eligible is False

    async def test_get_shadow_performance(self) -> None:
        """Test fetching shadow performance metrics."""
        promoter = AutoPromoter()

        mock_response_data = {
            "strategy_id": "1",
            "sharpe": 0.75,
            "max_drawdown": 0.08,
            "num_trades": 45,
            "shadow_days": 7,
        }

        # Directly assign the mock client (not using patch.object)
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        promoter._http_client = mock_client

        performance = await promoter._get_shadow_performance("1")

        assert performance["sharpe"] == 0.75
        assert performance["max_drawdown"] == 0.08
        assert performance["num_trades"] == 45

    async def test_get_strategy_alerts(self) -> None:
        """Test fetching strategy alerts."""
        promoter = AutoPromoter()

        mock_alerts = [
            {"id": 1, "type": "drawdown", "severity": "warning"},
            {"id": 2, "type": "latency", "severity": "info"},
        ]

        # Directly assign the mock client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alerts
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        promoter._http_client = mock_client

        alerts = await promoter._get_strategy_alerts("1")

        assert len(alerts) == 2
        assert alerts[0]["type"] == "drawdown"

    async def test_call_registry_promote(self) -> None:
        """Test calling registry API to promote strategy."""
        promoter = AutoPromoter()

        # Directly assign the mock client
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        promoter._http_client = mock_client

        result = await promoter._call_registry_promote("1", "paper")

        assert result is True
        mock_client.post.assert_called_once()

    async def test_call_registry_promote_failure(self) -> None:
        """Test handling promotion API failure."""
        promoter = AutoPromoter()

        with patch.object(promoter, "_http_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=Exception("API Error"))

            result = await promoter._call_registry_promote("1", "paper")

            assert result is False

    def test_criteria_checking_logic(self) -> None:
        """Test the criteria checking logic in detail."""
        criteria = PromotionCriteria(
            min_shadow_days=5,
            min_shadow_sharpe=0.4,
            max_shadow_drawdown=0.10,
            min_shadow_trades=20,
        )
        promoter = AutoPromoter(criteria=criteria)

        # Test check methods directly
        assert promoter._check_shadow_days(5) is True
        assert promoter._check_shadow_days(4) is False
        assert promoter._check_shadow_days(10) is True

        assert promoter._check_sharpe(0.4) is True
        assert promoter._check_sharpe(0.39) is False
        assert promoter._check_sharpe(0.8) is True

        assert promoter._check_drawdown(0.10) is True
        assert promoter._check_drawdown(0.11) is False
        assert promoter._check_drawdown(0.05) is True

        assert promoter._check_trades(20) is True
        assert promoter._check_trades(19) is False
        assert promoter._check_trades(50) is True

    async def test_evaluate_non_shadow_strategy(self) -> None:
        """Test evaluation of strategy not in shadow state."""
        promoter = AutoPromoter()

        with patch.object(
            promoter, "_get_strategy_state", return_value="candidate"
        ):
            evaluation = await promoter.evaluate_promotion("1")

            assert evaluation.eligible is False
            assert "not in shadow state" in evaluation.message.lower()


class TestAutoPromoterIntegration:
    """Integration tests for AutoPromoter with mocked registry API."""

    async def test_full_promotion_workflow(self) -> None:
        """Test complete promotion workflow from evaluation to execution."""
        criteria = PromotionCriteria(
            min_shadow_days=5,
            min_shadow_sharpe=0.4,
            max_shadow_drawdown=0.10,
            min_shadow_trades=20,
            no_alerts_required=True,
        )
        promoter = AutoPromoter(criteria=criteria)

        mock_performance = {
            "strategy_id": "1",
            "state": "shadow",
            "shadow_days": 10,
            "sharpe": 0.75,
            "max_drawdown": 0.05,
            "num_trades": 50,
        }

        with patch.object(
            promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ), patch.object(
            promoter, "_call_registry_promote", return_value=True
        ) as mock_promote:
            # First evaluate
            evaluation = await promoter.evaluate_promotion("1")
            assert evaluation.eligible is True

            # Then promote
            result = await promoter.promote_strategy("1")
            assert result is True
            mock_promote.assert_called_once_with("1", "paper")

    async def test_batch_promotion_check(self) -> None:
        """Test batch checking and promotion of multiple strategies."""
        promoter = AutoPromoter()

        mock_strategies = [
            {"id": 1, "name": "eligible_1", "state": "shadow"},
            {"id": 2, "name": "eligible_2", "state": "shadow"},
        ]

        eligible_performance = {
            "shadow_days": 10,
            "sharpe": 0.75,
            "max_drawdown": 0.05,
            "num_trades": 50,
        }

        with patch.object(
            promoter, "_get_shadow_strategies", return_value=mock_strategies
        ), patch.object(
            promoter, "_get_shadow_performance", return_value=eligible_performance
        ), patch.object(
            promoter, "_get_strategy_alerts", return_value=[]
        ), patch.object(
            promoter, "_get_strategy_state", return_value="shadow"
        ):
            evaluations = await promoter.check_all_candidates()

            eligible_count = sum(1 for e in evaluations if e.eligible)
            assert eligible_count == 2


# =============================================================================
# T6.06 PromotionState Tests
# =============================================================================


class TestPromotionState:
    """Tests for PromotionState enum."""

    def test_state_values(self) -> None:
        """Test all promotion state values."""
        assert PromotionState.CANDIDATE.value == "candidate"
        assert PromotionState.SHADOW.value == "shadow"
        assert PromotionState.PAPER.value == "paper"
        assert PromotionState.LIVE.value == "live"
        assert PromotionState.RETIRED.value == "retired"

    def test_state_is_string_enum(self) -> None:
        """Test that PromotionState is a string enum."""
        assert isinstance(PromotionState.SHADOW, str)
        assert PromotionState.SHADOW == "shadow"

    def test_valid_transitions(self) -> None:
        """Test valid state transitions mapping."""
        transitions = PromotionState.valid_transitions()

        assert PromotionState.SHADOW in transitions[PromotionState.CANDIDATE]
        assert PromotionState.PAPER in transitions[PromotionState.SHADOW]
        assert PromotionState.LIVE in transitions[PromotionState.PAPER]
        assert transitions[PromotionState.RETIRED] == []

    def test_can_transition_to_valid(self) -> None:
        """Test valid transition checks."""
        assert PromotionState.CANDIDATE.can_transition_to(PromotionState.SHADOW) is True
        assert PromotionState.SHADOW.can_transition_to(PromotionState.PAPER) is True
        assert PromotionState.PAPER.can_transition_to(PromotionState.LIVE) is True

    def test_can_transition_to_invalid(self) -> None:
        """Test invalid transition checks."""
        assert PromotionState.CANDIDATE.can_transition_to(PromotionState.PAPER) is False
        assert PromotionState.SHADOW.can_transition_to(PromotionState.LIVE) is False
        assert PromotionState.RETIRED.can_transition_to(PromotionState.SHADOW) is False

    def test_can_transition_to_retired(self) -> None:
        """Test that all states can transition to retired."""
        assert PromotionState.CANDIDATE.can_transition_to(PromotionState.RETIRED) is True
        assert PromotionState.SHADOW.can_transition_to(PromotionState.RETIRED) is True
        assert PromotionState.PAPER.can_transition_to(PromotionState.RETIRED) is True
        assert PromotionState.LIVE.can_transition_to(PromotionState.RETIRED) is True

    def test_paper_can_demote_to_shadow(self) -> None:
        """Test that paper can be demoted back to shadow."""
        assert PromotionState.PAPER.can_transition_to(PromotionState.SHADOW) is True

    def test_live_can_demote_to_paper(self) -> None:
        """Test that live can be demoted back to paper."""
        assert PromotionState.LIVE.can_transition_to(PromotionState.PAPER) is True


class TestPromotionDecision:
    """Tests for PromotionDecision dataclass."""

    def test_decision_creation(self) -> None:
        """Test creating a promotion decision."""
        decision = PromotionDecision(
            strategy_id="strategy_123",
            from_state=PromotionState.SHADOW,
            to_state=PromotionState.PAPER,
            approved=True,
            criteria_results={"min_shadow_days": True, "min_sharpe": True},
            performance_metrics={"sharpe": 0.8, "max_drawdown": 0.05},
            reason="Strategy meets all criteria",
        )

        assert decision.strategy_id == "strategy_123"
        assert decision.from_state == PromotionState.SHADOW
        assert decision.to_state == PromotionState.PAPER
        assert decision.approved is True
        assert decision.criteria_results["min_shadow_days"] is True
        assert decision.performance_metrics["sharpe"] == 0.8
        assert decision.auto_promoted is False
        assert decision.reviewer_id is None

    def test_decision_with_auto_promoted(self) -> None:
        """Test decision with auto_promoted flag."""
        decision = PromotionDecision(
            strategy_id="strategy_456",
            from_state=PromotionState.SHADOW,
            to_state=PromotionState.PAPER,
            approved=True,
            criteria_results={},
            performance_metrics={},
            reason="Auto promoted",
            auto_promoted=True,
        )

        assert decision.auto_promoted is True

    def test_decision_with_reviewer(self) -> None:
        """Test decision with manual reviewer."""
        decision = PromotionDecision(
            strategy_id="strategy_789",
            from_state=PromotionState.PAPER,
            to_state=PromotionState.LIVE,
            approved=True,
            criteria_results={},
            performance_metrics={},
            reason="Manual approval",
            reviewer_id="admin_user",
        )

        assert decision.reviewer_id == "admin_user"
        assert decision.auto_promoted is False

    def test_decision_has_timestamp(self) -> None:
        """Test that decision has a timestamp."""
        decision = PromotionDecision(
            strategy_id="test",
            from_state=PromotionState.SHADOW,
            to_state=PromotionState.PAPER,
            approved=False,
            criteria_results={},
            performance_metrics={},
            reason="Test",
        )

        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)


class TestPromotionManager:
    """Tests for PromotionManager class."""

    def test_manager_init_default(self) -> None:
        """Test manager initialization with defaults."""
        manager = PromotionManager()

        assert manager.default_criteria.min_shadow_days == 5
        assert manager.registry_api_url == "http://localhost:8080"

    def test_manager_init_custom_criteria(self) -> None:
        """Test manager initialization with custom criteria."""
        criteria = PromotionCriteria(min_shadow_days=10, min_shadow_sharpe=0.6)
        manager = PromotionManager(
            criteria=criteria,
            registry_api_url="http://registry:9000",
        )

        assert manager.default_criteria.min_shadow_days == 10
        assert manager.default_criteria.min_shadow_sharpe == 0.6
        assert manager.registry_api_url == "http://registry:9000"

    def test_get_state_from_string_valid(self) -> None:
        """Test converting valid state strings."""
        manager = PromotionManager()

        assert manager.get_state_from_string("shadow") == PromotionState.SHADOW
        assert manager.get_state_from_string("PAPER") == PromotionState.PAPER
        assert manager.get_state_from_string("Candidate") == PromotionState.CANDIDATE

    def test_get_state_from_string_invalid(self) -> None:
        """Test converting invalid state string raises error."""
        manager = PromotionManager()

        with pytest.raises(ValueError, match="Invalid promotion state"):
            manager.get_state_from_string("invalid_state")

    def test_validate_transition_valid(self) -> None:
        """Test validating valid transitions."""
        manager = PromotionManager()

        assert manager.validate_transition(
            PromotionState.SHADOW, PromotionState.PAPER
        ) is True
        assert manager.validate_transition(
            PromotionState.CANDIDATE, PromotionState.SHADOW
        ) is True

    def test_validate_transition_invalid(self) -> None:
        """Test validating invalid transitions."""
        manager = PromotionManager()

        assert manager.validate_transition(
            PromotionState.CANDIDATE, PromotionState.LIVE
        ) is False
        assert manager.validate_transition(
            PromotionState.RETIRED, PromotionState.SHADOW
        ) is False

    def test_get_next_state(self) -> None:
        """Test getting next state in lifecycle."""
        manager = PromotionManager()

        assert manager._get_next_state(PromotionState.CANDIDATE) == PromotionState.SHADOW
        assert manager._get_next_state(PromotionState.SHADOW) == PromotionState.PAPER
        assert manager._get_next_state(PromotionState.PAPER) == PromotionState.LIVE
        assert manager._get_next_state(PromotionState.LIVE) == PromotionState.RETIRED
        assert manager._get_next_state(PromotionState.RETIRED) == PromotionState.RETIRED

    def test_decision_history_empty(self) -> None:
        """Test empty decision history."""
        manager = PromotionManager()

        history = manager.get_decision_history()
        assert len(history) == 0

    def test_clear_history(self) -> None:
        """Test clearing decision history."""
        manager = PromotionManager()
        # Manually add a decision to history
        manager._decision_history.append(
            PromotionDecision(
                strategy_id="test",
                from_state=PromotionState.SHADOW,
                to_state=PromotionState.PAPER,
                approved=True,
                criteria_results={},
                performance_metrics={},
                reason="Test",
            )
        )

        assert len(manager.get_decision_history()) == 1
        manager.clear_history()
        assert len(manager.get_decision_history()) == 0

    def test_get_decision_history_with_filter(self) -> None:
        """Test filtering decision history by strategy_id."""
        manager = PromotionManager()

        # Add decisions for different strategies
        manager._decision_history.append(
            PromotionDecision(
                strategy_id="strat_1",
                from_state=PromotionState.SHADOW,
                to_state=PromotionState.PAPER,
                approved=True,
                criteria_results={},
                performance_metrics={},
                reason="Test 1",
            )
        )
        manager._decision_history.append(
            PromotionDecision(
                strategy_id="strat_2",
                from_state=PromotionState.SHADOW,
                to_state=PromotionState.PAPER,
                approved=False,
                criteria_results={},
                performance_metrics={},
                reason="Test 2",
            )
        )

        all_history = manager.get_decision_history()
        assert len(all_history) == 2

        filtered = manager.get_decision_history(strategy_id="strat_1")
        assert len(filtered) == 1
        assert filtered[0].strategy_id == "strat_1"

    def test_get_decision_history_with_limit(self) -> None:
        """Test limiting decision history results."""
        manager = PromotionManager()

        # Add multiple decisions
        for i in range(5):
            manager._decision_history.append(
                PromotionDecision(
                    strategy_id=f"strat_{i}",
                    from_state=PromotionState.SHADOW,
                    to_state=PromotionState.PAPER,
                    approved=True,
                    criteria_results={},
                    performance_metrics={},
                    reason=f"Test {i}",
                )
            )

        limited = manager.get_decision_history(limit=3)
        assert len(limited) == 3
        # Should return the last 3 entries
        assert limited[0].strategy_id == "strat_2"
        assert limited[2].strategy_id == "strat_4"

    async def test_evaluate_promotion(self) -> None:
        """Test evaluating a strategy for promotion."""
        manager = PromotionManager()

        mock_evaluation = PromotionEvaluation(
            strategy_id="1",
            current_state="shadow",
            target_state="paper",
            criteria_met={"min_shadow_days": True, "min_sharpe": True},
            eligible=True,
            message="Eligible",
        )

        mock_performance = {
            "sharpe": 0.8,
            "max_drawdown": 0.05,
            "num_trades": 50,
            "shadow_days": 10,
        }

        with patch.object(
            manager._promoter, "evaluate_promotion", return_value=mock_evaluation
        ), patch.object(
            manager._promoter, "_get_shadow_performance", return_value=mock_performance
        ):
            decision = await manager.evaluate_promotion("1")

            assert decision.strategy_id == "1"
            assert decision.from_state == PromotionState.SHADOW
            assert decision.to_state == PromotionState.PAPER
            assert decision.approved is True
            assert decision.performance_metrics["sharpe"] == 0.8

    async def test_promote_success(self) -> None:
        """Test successful promotion execution."""
        manager = PromotionManager()

        mock_evaluation = PromotionEvaluation(
            strategy_id="1",
            current_state="shadow",
            target_state="paper",
            criteria_met={"all": True},
            eligible=True,
            message="Eligible",
        )

        mock_performance = {
            "sharpe": 0.8, "max_drawdown": 0.05, "num_trades": 50, "shadow_days": 10
        }

        with patch.object(
            manager._promoter, "evaluate_promotion", return_value=mock_evaluation
        ), patch.object(
            manager._promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            manager._promoter, "_call_registry_promote", return_value=True
        ) as mock_promote:
            decision = await manager.promote("1")

            assert decision.approved is True
            assert decision.auto_promoted is True
            mock_promote.assert_called_once_with("1", "paper")

    async def test_promote_not_approved(self) -> None:
        """Test promotion when criteria not met."""
        manager = PromotionManager()

        mock_evaluation = PromotionEvaluation(
            strategy_id="1",
            current_state="shadow",
            target_state="paper",
            criteria_met={"sharpe": False},
            eligible=False,
            message="Not eligible - low sharpe",
        )

        mock_performance = {"sharpe": 0.2, "max_drawdown": 0.15, "num_trades": 10, "shadow_days": 3}

        with patch.object(
            manager._promoter, "evaluate_promotion", return_value=mock_evaluation
        ), patch.object(
            manager._promoter, "_get_shadow_performance", return_value=mock_performance
        ):
            decision = await manager.promote("1")

            assert decision.approved is False

    async def test_promote_with_reviewer(self) -> None:
        """Test manual promotion with reviewer ID."""
        manager = PromotionManager()

        mock_evaluation = PromotionEvaluation(
            strategy_id="1",
            current_state="shadow",
            target_state="paper",
            criteria_met={"all": True},
            eligible=True,
            message="Eligible",
        )

        mock_performance = {
            "sharpe": 0.8, "max_drawdown": 0.05, "num_trades": 50, "shadow_days": 10
        }

        with patch.object(
            manager._promoter, "evaluate_promotion", return_value=mock_evaluation
        ), patch.object(
            manager._promoter, "_get_shadow_performance", return_value=mock_performance
        ), patch.object(
            manager._promoter, "_call_registry_promote", return_value=True
        ):
            decision = await manager.promote("1", auto=False, reviewer_id="admin")

            assert decision.auto_promoted is False
            assert decision.reviewer_id == "admin"

    async def test_close(self) -> None:
        """Test closing manager resources."""
        manager = PromotionManager()

        with patch.object(manager._promoter, "close") as mock_close:
            await manager.close()
            mock_close.assert_called_once()
