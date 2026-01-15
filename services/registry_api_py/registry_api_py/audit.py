"""Audit Log Implementation (T5.06).

This module provides comprehensive audit logging for all state changes
and important events in the strategy registry system.

Features:
- Record all state changes with before/after values
- Query audit history by entity, event type, and time range
- Support for user attribution and system events
- Metadata storage for additional context
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import and_, desc

from registry_api_py.models import AuditEventType, AuditLogEntry

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class AuditEntry:
    """Represents an audit log entry.

    This dataclass provides a clean interface for audit entries,
    decoupling from the SQLAlchemy model.
    """

    id: int
    event_type: AuditEventType
    entity_type: str
    entity_id: int
    user_id: str | None
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    message: str
    timestamp: datetime
    metadata: dict[str, Any]


# =============================================================================
# AUDIT LOG CLASS
# =============================================================================


class AuditLog:
    """Record and query audit events.

    This class provides methods to record audit events and query
    the audit history for entities and event types.
    """

    def __init__(self, db: Session) -> None:
        """Initialize the audit log with a database session.

        Args:
            db: SQLAlchemy database session.
        """
        self.db = db

    def record(
        self,
        event_type: AuditEventType,
        entity_type: str,
        entity_id: int,
        message: str,
        old_value: dict[str, Any] | None = None,
        new_value: dict[str, Any] | None = None,
        user_id: str | None = "system",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            event_type: Type of the event.
            entity_type: Type of the entity (e.g., "strategy", "artifact").
            entity_id: ID of the entity.
            message: Human-readable description of the event.
            old_value: Previous state/value (for updates).
            new_value: New state/value (for creates/updates).
            user_id: ID of the user who triggered the event. Defaults to "system".
            metadata: Additional metadata about the event.

        Returns:
            AuditEntry representing the recorded event.
        """
        entry = AuditLogEntry(
            event_type=event_type.value,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            old_value=old_value,
            new_value=new_value,
            message=message,
            timestamp=datetime.now(UTC),
            metadata_=metadata or {},
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)

        return self._to_audit_entry(entry)

    def get_history(self, entity_type: str, entity_id: int) -> list[AuditEntry]:
        """Get the audit history for a specific entity.

        Args:
            entity_type: Type of the entity.
            entity_id: ID of the entity.

        Returns:
            List of AuditEntry objects, ordered by timestamp (most recent first).
        """
        entries = (
            self.db.query(AuditLogEntry)
            .filter(
                and_(
                    AuditLogEntry.entity_type == entity_type,
                    AuditLogEntry.entity_id == entity_id,
                )
            )
            .order_by(desc(AuditLogEntry.timestamp))
            .all()
        )

        return [self._to_audit_entry(e) for e in entries]

    def query(
        self,
        event_type: AuditEventType | None = None,
        entity_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries with filters.

        Args:
            event_type: Filter by event type.
            entity_type: Filter by entity type.
            start_time: Filter entries after this time.
            end_time: Filter entries before this time.
            limit: Maximum number of entries to return. Defaults to 100.

        Returns:
            List of AuditEntry objects matching the filters.
        """
        query = self.db.query(AuditLogEntry)

        if event_type is not None:
            query = query.filter(AuditLogEntry.event_type == event_type.value)

        if entity_type is not None:
            query = query.filter(AuditLogEntry.entity_type == entity_type)

        if start_time is not None:
            query = query.filter(AuditLogEntry.timestamp >= start_time)

        if end_time is not None:
            query = query.filter(AuditLogEntry.timestamp <= end_time)

        entries = query.order_by(desc(AuditLogEntry.timestamp)).limit(limit).all()

        return [self._to_audit_entry(e) for e in entries]

    def _to_audit_entry(self, db_entry: AuditLogEntry) -> AuditEntry:
        """Convert a database entry to an AuditEntry dataclass.

        Args:
            db_entry: SQLAlchemy model instance.

        Returns:
            AuditEntry dataclass instance.
        """
        return AuditEntry(
            id=cast(int, db_entry.id),
            event_type=AuditEventType(cast(str, db_entry.event_type)),
            entity_type=cast(str, db_entry.entity_type),
            entity_id=cast(int, db_entry.entity_id),
            user_id=cast("str | None", db_entry.user_id),
            old_value=cast("dict[str, Any] | None", db_entry.old_value),
            new_value=cast("dict[str, Any] | None", db_entry.new_value),
            message=cast(str, db_entry.message),
            timestamp=cast(datetime, db_entry.timestamp),
            metadata=cast("dict[str, Any]", db_entry.metadata_ or {}),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_audit_log(db: Session) -> AuditLog:
    """Create an AuditLog instance with the given database session.

    Args:
        db: SQLAlchemy database session.

    Returns:
        AuditLog instance.
    """
    return AuditLog(db)


def record_strategy_event(
    db: Session,
    event_type: AuditEventType,
    strategy_id: int,
    message: str,
    old_value: dict[str, Any] | None = None,
    new_value: dict[str, Any] | None = None,
    user_id: str = "system",
    metadata: dict[str, Any] | None = None,
) -> AuditEntry:
    """Convenience function to record a strategy-related event.

    Args:
        db: Database session.
        event_type: Type of the event.
        strategy_id: ID of the strategy.
        message: Event message.
        old_value: Previous value.
        new_value: New value.
        user_id: User who triggered the event.
        metadata: Additional metadata.

    Returns:
        Recorded AuditEntry.
    """
    audit_log = AuditLog(db)
    return audit_log.record(
        event_type=event_type,
        entity_type="strategy",
        entity_id=strategy_id,
        message=message,
        old_value=old_value,
        new_value=new_value,
        user_id=user_id,
        metadata=metadata,
    )


def record_artifact_event(
    db: Session,
    event_type: AuditEventType,
    artifact_id: int,
    message: str,
    old_value: dict[str, Any] | None = None,
    new_value: dict[str, Any] | None = None,
    user_id: str = "system",
    metadata: dict[str, Any] | None = None,
) -> AuditEntry:
    """Convenience function to record an artifact-related event.

    Args:
        db: Database session.
        event_type: Type of the event.
        artifact_id: ID of the artifact.
        message: Event message.
        old_value: Previous value.
        new_value: New value.
        user_id: User who triggered the event.
        metadata: Additional metadata.

    Returns:
        Recorded AuditEntry.
    """
    audit_log = AuditLog(db)
    return audit_log.record(
        event_type=event_type,
        entity_type="artifact",
        entity_id=artifact_id,
        message=message,
        old_value=old_value,
        new_value=new_value,
        user_id=user_id,
        metadata=metadata,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AuditEntry",
    "AuditLog",
    "create_audit_log",
    "record_strategy_event",
    "record_artifact_event",
]
