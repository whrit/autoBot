"""SQLAlchemy database models for Registry API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class PromotionState(str, Enum):
    """Promotion state for strategies in the trading pipeline."""

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PAPER = "paper"
    RETIRED = "retired"


class AuditEventType(str, Enum):
    """Audit event types for comprehensive logging."""

    STRATEGY_CREATED = "strategy_created"
    STRATEGY_UPDATED = "strategy_updated"
    STRATEGY_DELETED = "strategy_deleted"
    ARTIFACT_CREATED = "artifact_created"
    ARTIFACT_UPDATED = "artifact_updated"
    ARTIFACT_DELETED = "artifact_deleted"
    PROMOTION_REQUESTED = "promotion_requested"
    PROMOTION_APPROVED = "promotion_approved"
    PROMOTION_REJECTED = "promotion_rejected"
    DEMOTION = "demotion"
    RETIREMENT = "retirement"
    ARTIFACT_UPLOADED = "artifact_uploaded"
    GATE_EVALUATION = "gate_evaluation"
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"
    ALERT_TRIGGERED = "alert_triggered"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"


class Strategy(Base):
    """Strategy model representing a trading strategy configuration."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    family = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    state: PromotionState = Column(SQLEnum(PromotionState), default=PromotionState.CANDIDATE, nullable=False)  # type: ignore[assignment]
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(UTC))

    # Relationships
    artifacts = relationship("Artifact", back_populates="strategy", cascade="all, delete-orphan")
    backtest_runs = relationship(
        "BacktestRun", back_populates="strategy", cascade="all, delete-orphan"
    )


class Artifact(Base):
    """Artifact model for storing strategy-related files (ONNX, JSON, config)."""

    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    artifact_type = Column(String, nullable=False)  # "onnx", "json", "config"
    path = Column(String, nullable=False)
    checksum = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Relationships
    strategy = relationship("Strategy", back_populates="artifacts")


class DatasetSnapshot(Base):
    """Dataset snapshot model for versioned market data snapshots."""

    __tablename__ = "dataset_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    version = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    symbols = Column(JSON, nullable=False)  # list of symbols
    feature_schema_version = Column(String)
    row_count = Column(Integer)
    checksum = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Relationships
    backtest_runs = relationship("BacktestRun", back_populates="dataset_snapshot")


class BacktestRun(Base):
    """Backtest run model for storing backtest execution results."""

    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    dataset_snapshot_id = Column(
        Integer, ForeignKey("dataset_snapshots.id"), nullable=True, index=True
    )

    # Performance metrics
    sharpe = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    profit_factor = Column(Float)
    win_rate = Column(Float)
    num_trades = Column(Integer)

    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Additional metadata
    metadata_ = Column("metadata", JSON)

    # GPU Training Metadata (Phase 1 - GPU Determinism)
    training_device = Column(String, nullable=True)  # "cpu" | "cuda"
    gpu_backend = Column(String, nullable=True)  # "xgboost_gpu" | "pytorch_cuda" | "cpu"
    cuda_version = Column(String, nullable=True)  # e.g., "12.1"
    driver_version = Column(String, nullable=True)  # e.g., "535.104.05"
    seed = Column(Integer, nullable=True)  # Random seed for reproducibility
    determinism_flags = Column(String, nullable=True)  # e.g., "CUDA_LAUNCH_BLOCKING=1"
    training_time_sec = Column(Float, nullable=True)  # Training duration in seconds

    # Relationships
    strategy = relationship("Strategy", back_populates="backtest_runs")
    dataset_snapshot = relationship("DatasetSnapshot", back_populates="backtest_runs")


class Gate(Base):
    """Gate model for promotion gates in the strategy lifecycle."""

    __tablename__ = "gates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(String)
    gate_type = Column(String, nullable=False)  # "backtest", "shadow", "paper"
    criteria = Column(JSON, nullable=False)  # JSON object with gate criteria
    is_active = Column(Integer, default=1)  # Boolean as integer for SQLite compatibility
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(UTC))


class Promotion(Base):
    """Promotion model for tracking strategy promotions through gates."""

    __tablename__ = "promotions"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    gate_id = Column(Integer, ForeignKey("gates.id"), nullable=False, index=True)
    from_state: PromotionState = Column(SQLEnum(PromotionState), nullable=False)  # type: ignore[assignment]
    to_state: PromotionState = Column(SQLEnum(PromotionState), nullable=False)  # type: ignore[assignment]
    passed = Column(Integer, nullable=False)  # Boolean as integer for SQLite compatibility
    evaluation_results = Column(JSON)  # Detailed results from gate evaluation
    promoted_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    promoted_by = Column(String)  # User or system that triggered promotion


class AuditLogEntry(Base):
    """Audit log entry for tracking all state changes and important events."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=False, index=True)
    entity_id = Column(Integer, nullable=False, index=True)
    user_id = Column(String, nullable=True)  # "system" for automated actions
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)
    metadata_ = Column("metadata", JSON, nullable=True)
