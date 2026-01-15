"""Promotion State Machine and Audit Log API Endpoints (T5.05, T5.06).

This module provides API endpoints for:
- Strategy promotion lifecycle management
- Audit log queries
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from registry_api_py.audit import AuditLog
from registry_api_py.database import get_db
from registry_api_py.models import AuditEventType, PromotionState
from registry_api_py.promotions import (
    InvalidTransitionError,
    PromotionManager,
    RequirementsNotMetError,
    StrategyNotFoundError,
)
from registry_api_py.schemas import (
    AuditEntryResponse,
    DemoteRequest,
    ErrorResponse,
    PromoteRequest,
    PromotionResultResponse,
    RetireRequest,
)

# Create router for promotion and audit endpoints
router = APIRouter()

# Type alias
DbDep = Annotated[Session, Depends(get_db)]


def _get_promotion_manager(db: Session) -> PromotionManager:
    """Create a promotion manager with audit logging enabled."""
    audit_log = AuditLog(db)
    return PromotionManager(audit_log=audit_log)


# =============================================================================
# Promotion State Machine Endpoints (T5.05)
# =============================================================================


@router.post(
    "/strategies/{strategy_id}/promote",
    response_model=PromotionResultResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    tags=["Promotion State Machine"],
)
def promote_strategy(
    strategy_id: int,
    request: PromoteRequest,
    db: DbDep,
) -> PromotionResultResponse:
    """Promote a strategy to a target state.

    Valid transitions:
    - candidate -> shadow (requires passing backtest gates)
    - shadow -> paper (requires successful shadow period)
    - Any state -> retired

    Args:
        strategy_id: ID of the strategy to promote.
        request: Promotion request with target state.

    Returns:
        Result of the promotion operation.
    """
    try:
        target_state = PromotionState(request.target_state)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target state: {request.target_state}. "
            f"Valid states: {[s.value for s in PromotionState]}",
        ) from None

    manager = _get_promotion_manager(db)

    try:
        result = manager.promote(strategy_id, target_state, db)
        return PromotionResultResponse(
            success=result.success,
            from_state=result.from_state.value,
            to_state=result.to_state.value,
            message=result.message,
            requirements_met=result.requirements_met,
        )
    except StrategyNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        ) from None
    except InvalidTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transition from {e.from_state.value} to {e.to_state.value}",
        ) from None
    except RequirementsNotMetError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requirements not met for {e.target_state.value}: "
            f"{[k for k, v in e.failed_requirements.items() if not v]}",
        ) from None


@router.post(
    "/strategies/{strategy_id}/demote",
    response_model=PromotionResultResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    tags=["Promotion State Machine"],
)
def demote_strategy(
    strategy_id: int,
    request: DemoteRequest,
    db: DbDep,
) -> PromotionResultResponse:
    """Demote a strategy to a lower state.

    Valid demotions:
    - paper -> shadow
    - shadow -> candidate

    Args:
        strategy_id: ID of the strategy to demote.
        request: Demotion request with reason.

    Returns:
        Result of the demotion operation.
    """
    manager = _get_promotion_manager(db)

    try:
        result = manager.demote(strategy_id, request.reason, db)
        return PromotionResultResponse(
            success=result.success,
            from_state=result.from_state.value,
            to_state=result.to_state.value,
            message=result.message,
            requirements_met=result.requirements_met,
        )
    except StrategyNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        ) from None
    except InvalidTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot demote from {e.from_state.value}",
        ) from None


@router.post(
    "/strategies/{strategy_id}/retire",
    response_model=PromotionResultResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    tags=["Promotion State Machine"],
)
def retire_strategy(
    strategy_id: int,
    request: RetireRequest,
    db: DbDep,
) -> PromotionResultResponse:
    """Retire a strategy (terminal state).

    Any active state can be retired.
    Retired is a terminal state with no further transitions.

    Args:
        strategy_id: ID of the strategy to retire.
        request: Retirement request with reason.

    Returns:
        Result of the retirement operation.
    """
    manager = _get_promotion_manager(db)

    try:
        result = manager.retire(strategy_id, request.reason, db)
        return PromotionResultResponse(
            success=result.success,
            from_state=result.from_state.value,
            to_state=result.to_state.value,
            message=result.message,
            requirements_met=result.requirements_met,
        )
    except StrategyNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        ) from None
    except InvalidTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retire from {e.from_state.value}",
        ) from None


@router.get(
    "/strategies/{strategy_id}/promotion-history",
    response_model=list[AuditEntryResponse],
    tags=["Promotion State Machine"],
)
def get_promotion_history(
    strategy_id: int,
    db: DbDep,
) -> list[AuditEntryResponse]:
    """Get promotion history for a strategy.

    Returns all promotion-related audit entries for the strategy.

    Args:
        strategy_id: ID of the strategy.

    Returns:
        List of audit entries for the strategy's promotion history.
    """
    audit_log = AuditLog(db)
    entries = audit_log.get_history("strategy", strategy_id)

    # Filter to promotion-related events
    promotion_types = {
        AuditEventType.PROMOTION_APPROVED,
        AuditEventType.PROMOTION_REJECTED,
        AuditEventType.PROMOTION_REQUESTED,
        AuditEventType.DEMOTION,
        AuditEventType.RETIREMENT,
    }

    filtered = [e for e in entries if e.event_type in promotion_types]

    return [
        AuditEntryResponse(
            id=e.id,
            event_type=e.event_type.value,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            user_id=e.user_id,
            old_value=e.old_value,
            new_value=e.new_value,
            message=e.message,
            timestamp=e.timestamp,
            metadata=e.metadata,
        )
        for e in filtered
    ]


# =============================================================================
# Audit Log Endpoints (T5.06)
# =============================================================================


@router.get(
    "/audit",
    response_model=list[AuditEntryResponse],
    tags=["Audit Log"],
)
def query_audit_log(
    db: DbDep,
    event_type: str | None = Query(None, description="Filter by event type"),
    entity_type: str | None = Query(None, description="Filter by entity type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
) -> list[AuditEntryResponse]:
    """Query the audit log.

    Returns audit entries matching the specified filters.

    Args:
        event_type: Optional filter by event type (e.g., 'strategy_created').
        entity_type: Optional filter by entity type (e.g., 'strategy').
        limit: Maximum number of entries to return.

    Returns:
        List of matching audit entries.
    """
    audit_log = AuditLog(db)

    # Convert event_type string to enum if provided
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AuditEventType(event_type)
        except ValueError:
            # Invalid event type returns empty list
            return []

    entries = audit_log.query(
        event_type=event_type_enum,
        entity_type=entity_type,
        limit=limit,
    )

    return [
        AuditEntryResponse(
            id=e.id,
            event_type=e.event_type.value,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            user_id=e.user_id,
            old_value=e.old_value,
            new_value=e.new_value,
            message=e.message,
            timestamp=e.timestamp,
            metadata=e.metadata,
        )
        for e in entries
    ]


@router.get(
    "/strategies/{strategy_id}/audit",
    response_model=list[AuditEntryResponse],
    tags=["Audit Log"],
)
def get_strategy_audit(
    strategy_id: int,
    db: DbDep,
) -> list[AuditEntryResponse]:
    """Get all audit entries for a strategy.

    Returns complete audit history for a strategy, including
    creation, updates, promotions, and all other events.

    Args:
        strategy_id: ID of the strategy.

    Returns:
        List of all audit entries for the strategy.
    """
    audit_log = AuditLog(db)
    entries = audit_log.get_history("strategy", strategy_id)

    return [
        AuditEntryResponse(
            id=e.id,
            event_type=e.event_type.value,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            user_id=e.user_id,
            old_value=e.old_value,
            new_value=e.new_value,
            message=e.message,
            timestamp=e.timestamp,
            metadata=e.metadata,
        )
        for e in entries
    ]
