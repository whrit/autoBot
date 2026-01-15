"""Tests for promotion API endpoints.

This module tests the REST API endpoints for strategy promotions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from registry_api_py.models import (
    BacktestRun,
    PromotionState,
    Strategy,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST PROMOTE ENDPOINT
# =============================================================================


class TestPromoteEndpoint:
    """Tests for POST /strategies/{strategy_id}/promote."""

    def test_promote_candidate_to_shadow(self, client: TestClient, db_session: Session) -> None:
        """Test promoting a strategy from CANDIDATE to SHADOW."""
        # Create strategy with good backtest
        strategy = Strategy(
            name="promotable",
            family="trend",
            version="1.0.0",
            parameters={"lookback": 20},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.8,
            max_drawdown=0.10,
            num_trades=150,
        )
        db_session.add(backtest)
        db_session.commit()

        response = client.post(
            f"/strategies/{strategy.id}/promote",
            json={"target_state": "shadow"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["from_state"] == "candidate"
        assert data["to_state"] == "shadow"

    def test_promote_shadow_to_paper(self, client: TestClient, db_session: Session) -> None:
        """Test promoting a strategy from SHADOW to PAPER."""
        strategy = Strategy(
            name="shadow_strategy",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        response = client.post(
            f"/strategies/{strategy.id}/promote",
            json={"target_state": "paper"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["to_state"] == "paper"

    def test_promote_invalid_transition(self, client: TestClient, db_session: Session) -> None:
        """Test that invalid transitions return 400."""
        strategy = Strategy(
            name="candidate_skip",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Try to skip SHADOW and go directly to PAPER
        response = client.post(
            f"/strategies/{strategy.id}/promote",
            json={"target_state": "paper"},
        )

        assert response.status_code == 400
        assert "Invalid transition" in response.json()["detail"]

    def test_promote_nonexistent_strategy(self, client: TestClient) -> None:
        """Test promoting a non-existent strategy returns 404."""
        response = client.post(
            "/strategies/99999/promote",
            json={"target_state": "shadow"},
        )

        assert response.status_code == 404

    def test_promote_requirements_not_met(
        self, client: TestClient, db_session: Session
    ) -> None:
        """Test that promotion fails when requirements not met."""
        strategy = Strategy(
            name="poor_performer",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Add failing backtest
        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.1,  # Too low
            max_drawdown=0.30,  # Too high
            num_trades=10,  # Too few
        )
        db_session.add(backtest)
        db_session.commit()

        response = client.post(
            f"/strategies/{strategy.id}/promote",
            json={"target_state": "shadow"},
        )

        assert response.status_code == 400
        assert "Requirements not met" in response.json()["detail"]


# =============================================================================
# TEST DEMOTE ENDPOINT
# =============================================================================


class TestDemoteEndpoint:
    """Tests for POST /strategies/{strategy_id}/demote."""

    def test_demote_paper_to_shadow(self, client: TestClient, db_session: Session) -> None:
        """Test demoting from PAPER to SHADOW."""
        strategy = Strategy(
            name="paper_to_demote",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.PAPER,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        response = client.post(
            f"/strategies/{strategy.id}/demote",
            json={"reason": "Performance degradation"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["from_state"] == "paper"
        assert data["to_state"] == "shadow"
        assert "Performance degradation" in data["message"]

    def test_demote_shadow_to_candidate(self, client: TestClient, db_session: Session) -> None:
        """Test demoting from SHADOW to CANDIDATE."""
        strategy = Strategy(
            name="shadow_to_demote",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        response = client.post(
            f"/strategies/{strategy.id}/demote",
            json={"reason": "Shadow testing failed"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["to_state"] == "candidate"

    def test_demote_candidate_fails(self, client: TestClient, db_session: Session) -> None:
        """Test that demoting CANDIDATE fails."""
        strategy = Strategy(
            name="candidate_no_demote",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        response = client.post(
            f"/strategies/{strategy.id}/demote",
            json={"reason": "Cannot demote"},
        )

        assert response.status_code == 400

    def test_demote_nonexistent(self, client: TestClient) -> None:
        """Test demoting non-existent strategy returns 404."""
        response = client.post(
            "/strategies/99999/demote",
            json={"reason": "Test"},
        )

        assert response.status_code == 404


# =============================================================================
# TEST RETIRE ENDPOINT
# =============================================================================


class TestRetireEndpoint:
    """Tests for POST /strategies/{strategy_id}/retire."""

    def test_retire_from_candidate(self, client: TestClient, db_session: Session) -> None:
        """Test retiring from CANDIDATE state."""
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

        response = client.post(
            f"/strategies/{strategy.id}/retire",
            json={"reason": "No longer viable"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["to_state"] == "retired"

    def test_retire_from_shadow(self, client: TestClient, db_session: Session) -> None:
        """Test retiring from SHADOW state."""
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

        response = client.post(
            f"/strategies/{strategy.id}/retire",
            json={"reason": "Testing concluded"},
        )

        assert response.status_code == 200
        assert response.json()["to_state"] == "retired"

    def test_retire_from_paper(self, client: TestClient, db_session: Session) -> None:
        """Test retiring from PAPER state."""
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

        response = client.post(
            f"/strategies/{strategy.id}/retire",
            json={"reason": "Paper trading done"},
        )

        assert response.status_code == 200
        assert response.json()["to_state"] == "retired"

    def test_retire_already_retired(self, client: TestClient, db_session: Session) -> None:
        """Test retiring already retired strategy fails."""
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

        response = client.post(
            f"/strategies/{strategy.id}/retire",
            json={"reason": "Try again"},
        )

        assert response.status_code == 400


# =============================================================================
# TEST PROMOTION HISTORY ENDPOINT
# =============================================================================


class TestPromotionHistoryEndpoint:
    """Tests for GET /strategies/{strategy_id}/promotion-history."""

    def test_get_promotion_history(self, client: TestClient, db_session: Session) -> None:
        """Test getting promotion history for a strategy."""
        strategy = Strategy(
            name="history_test",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Promote the strategy to generate history
        backtest = BacktestRun(
            strategy_id=strategy.id,
            sharpe=0.8,
            max_drawdown=0.10,
            num_trades=150,
        )
        db_session.add(backtest)
        db_session.commit()

        client.post(f"/strategies/{strategy.id}/promote", json={"target_state": "shadow"})

        response = client.get(f"/strategies/{strategy.id}/promotion-history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_history_nonexistent_strategy(self, client: TestClient) -> None:
        """Test getting history for non-existent strategy."""
        response = client.get("/strategies/99999/promotion-history")

        # Should return 404 or empty list depending on implementation
        assert response.status_code in [200, 404]
