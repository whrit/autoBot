"""Tests for audit log API endpoints.

This module tests the REST API endpoints for audit log queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from registry_api_py.audit import AuditLog
from registry_api_py.models import (
    AuditEventType,
    BacktestRun,
    PromotionState,
    Strategy,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# TEST QUERY AUDIT ENDPOINT
# =============================================================================


class TestQueryAuditEndpoint:
    """Tests for GET /audit."""

    def test_query_all_events(self, client: TestClient, db_session: Session) -> None:
        """Test querying all audit events."""
        # Create some audit entries
        audit_log = AuditLog(db_session)
        audit_log.record(
            event_type=AuditEventType.STRATEGY_CREATED,
            entity_type="strategy",
            entity_id=1,
            message="Strategy created",
        )
        audit_log.record(
            event_type=AuditEventType.STRATEGY_UPDATED,
            entity_type="strategy",
            entity_id=1,
            message="Strategy updated",
        )

        response = client.get("/audit")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_query_by_event_type(self, client: TestClient, db_session: Session) -> None:
        """Test querying by event type."""
        audit_log = AuditLog(db_session)
        audit_log.record(
            event_type=AuditEventType.STRATEGY_CREATED,
            entity_type="strategy",
            entity_id=1,
            message="Created",
        )
        audit_log.record(
            event_type=AuditEventType.PROMOTION_APPROVED,
            entity_type="strategy",
            entity_id=1,
            message="Promoted",
        )

        response = client.get("/audit", params={"event_type": "strategy_created"})

        assert response.status_code == 200
        data = response.json()
        assert all(e["event_type"] == "strategy_created" for e in data)

    def test_query_by_entity_type(self, client: TestClient, db_session: Session) -> None:
        """Test querying by entity type."""
        audit_log = AuditLog(db_session)
        audit_log.record(
            event_type=AuditEventType.STRATEGY_CREATED,
            entity_type="strategy",
            entity_id=1,
            message="Strategy",
        )
        audit_log.record(
            event_type=AuditEventType.ARTIFACT_UPLOADED,
            entity_type="artifact",
            entity_id=1,
            message="Artifact",
        )

        response = client.get("/audit", params={"entity_type": "strategy"})

        assert response.status_code == 200
        data = response.json()
        assert all(e["entity_type"] == "strategy" for e in data)

    def test_query_with_limit(self, client: TestClient, db_session: Session) -> None:
        """Test querying with limit."""
        audit_log = AuditLog(db_session)
        for i in range(10):
            audit_log.record(
                event_type=AuditEventType.STRATEGY_CREATED,
                entity_type="strategy",
                entity_id=i,
                message=f"Strategy {i}",
            )

        response = client.get("/audit", params={"limit": 5})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_query_empty_results(self, client: TestClient, db_session: Session) -> None:
        """Test querying when no events match."""
        response = client.get("/audit", params={"event_type": "nonexistent_type"})

        # Should return 200 with empty list, or 422 for invalid param
        assert response.status_code in [200, 422]


# =============================================================================
# TEST STRATEGY AUDIT ENDPOINT
# =============================================================================


class TestStrategyAuditEndpoint:
    """Tests for GET /strategies/{strategy_id}/audit."""

    def test_get_strategy_audit(self, client: TestClient, db_session: Session) -> None:
        """Test getting audit history for a strategy."""
        # Create strategy and audit entries
        strategy = Strategy(
            name="audit_strategy",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        audit_log = AuditLog(db_session)
        audit_log.record(
            event_type=AuditEventType.STRATEGY_CREATED,
            entity_type="strategy",
            entity_id=strategy.id,
            message="Strategy created",
        )
        audit_log.record(
            event_type=AuditEventType.STRATEGY_UPDATED,
            entity_type="strategy",
            entity_id=strategy.id,
            message="Strategy updated",
        )

        response = client.get(f"/strategies/{strategy.id}/audit")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all(e["entity_id"] == strategy.id for e in data)

    def test_get_strategy_audit_empty(self, client: TestClient, db_session: Session) -> None:
        """Test getting audit for strategy with no history."""
        strategy = Strategy(
            name="no_history",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        response = client.get(f"/strategies/{strategy.id}/audit")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_strategy_audit_nonexistent(self, client: TestClient) -> None:
        """Test getting audit for non-existent strategy."""
        response = client.get("/strategies/99999/audit")

        # Should return 200 with empty list (no audit for non-existent)
        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# TEST AUDIT INTEGRATION WITH PROMOTIONS
# =============================================================================


class TestAuditPromotionIntegration:
    """Tests for audit logging during promotion operations."""

    def test_promotion_creates_audit_entry(
        self, client: TestClient, db_session: Session
    ) -> None:
        """Test that successful promotion creates an audit entry."""
        strategy = Strategy(
            name="promote_audit",
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
            sharpe=0.8,
            max_drawdown=0.10,
            num_trades=150,
        )
        db_session.add(backtest)
        db_session.commit()

        # Promote the strategy
        client.post(
            f"/strategies/{strategy.id}/promote",
            json={"target_state": "shadow"},
        )

        # Check audit log
        response = client.get(f"/strategies/{strategy.id}/audit")
        assert response.status_code == 200
        data = response.json()

        # Should have at least one entry for the promotion
        promotion_events = [
            e for e in data if e["event_type"] == "promotion_approved"
        ]
        assert len(promotion_events) >= 1

    def test_demotion_creates_audit_entry(
        self, client: TestClient, db_session: Session
    ) -> None:
        """Test that demotion creates an audit entry."""
        strategy = Strategy(
            name="demote_audit",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.SHADOW,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Demote the strategy
        client.post(
            f"/strategies/{strategy.id}/demote",
            json={"reason": "Testing demotion audit"},
        )

        # Check audit log
        response = client.get(f"/strategies/{strategy.id}/audit")
        assert response.status_code == 200
        data = response.json()

        # Should have a demotion entry
        demotion_events = [e for e in data if e["event_type"] == "demotion"]
        assert len(demotion_events) >= 1

    def test_retirement_creates_audit_entry(
        self, client: TestClient, db_session: Session
    ) -> None:
        """Test that retirement creates an audit entry."""
        strategy = Strategy(
            name="retire_audit",
            family="trend",
            version="1.0.0",
            parameters={},
            state=PromotionState.CANDIDATE,
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)

        # Retire the strategy
        client.post(
            f"/strategies/{strategy.id}/retire",
            json={"reason": "Testing retirement audit"},
        )

        # Check audit log
        response = client.get(f"/strategies/{strategy.id}/audit")
        assert response.status_code == 200
        data = response.json()

        # Should have a retirement entry
        retirement_events = [e for e in data if e["event_type"] == "retirement"]
        assert len(retirement_events) >= 1
