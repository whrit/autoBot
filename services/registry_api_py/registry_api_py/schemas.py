"""Pydantic schemas for Registry API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from registry_api_py.models import PromotionState

# =============================================================================
# Strategy Schemas
# =============================================================================


class StrategyCreate(BaseModel):
    """Schema for creating a new strategy."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique strategy name")
    family: str = Field(..., min_length=1, max_length=100, description="Strategy family")
    version: str = Field(..., min_length=1, max_length=50, description="Strategy version")
    parameters: dict[str, Any] = Field(..., description="Strategy parameters as JSON")


class StrategyUpdate(BaseModel):
    """Schema for updating an existing strategy."""

    name: str | None = Field(None, min_length=1, max_length=255)
    family: str | None = Field(None, min_length=1, max_length=100)
    version: str | None = Field(None, min_length=1, max_length=50)
    parameters: dict[str, Any] | None = None
    state: PromotionState | None = None


class StrategyResponse(BaseModel):
    """Schema for strategy response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    family: str
    version: str
    parameters: dict[str, Any]
    state: PromotionState
    created_at: datetime
    updated_at: datetime | None = None


class StrategyListResponse(BaseModel):
    """Schema for paginated strategy list response."""

    items: list[StrategyResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Artifact Schemas
# =============================================================================


class ArtifactCreate(BaseModel):
    """Schema for creating a new artifact."""

    strategy_id: int = Field(..., description="Associated strategy ID")
    artifact_type: str = Field(..., min_length=1, max_length=50, description="Artifact type")
    path: str = Field(..., min_length=1, description="File path or S3 key")
    checksum: str = Field(..., min_length=1, description="File checksum (SHA256)")


class ArtifactResponse(BaseModel):
    """Schema for artifact response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    artifact_type: str
    path: str
    checksum: str
    created_at: datetime


class ArtifactListResponse(BaseModel):
    """Schema for paginated artifact list response."""

    items: list[ArtifactResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Dataset Snapshot Schemas
# =============================================================================


class DatasetSnapshotCreate(BaseModel):
    """Schema for creating a new dataset snapshot."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique snapshot name")
    version: str = Field(..., min_length=1, max_length=50, description="Snapshot version")
    start_date: datetime = Field(..., description="Start date of data range")
    end_date: datetime = Field(..., description="End date of data range")
    symbols: list[str] = Field(..., min_length=1, description="List of symbols in snapshot")
    feature_schema_version: str | None = Field(None, description="Feature schema version")
    row_count: int | None = Field(None, ge=0, description="Number of rows in dataset")
    checksum: str | None = Field(None, description="Dataset checksum")


class DatasetSnapshotUpdate(BaseModel):
    """Schema for updating a dataset snapshot."""

    name: str | None = Field(None, min_length=1, max_length=255)
    version: str | None = Field(None, min_length=1, max_length=50)
    feature_schema_version: str | None = None
    row_count: int | None = Field(None, ge=0)
    checksum: str | None = None


class DatasetSnapshotResponse(BaseModel):
    """Schema for dataset snapshot response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    feature_schema_version: str | None = None
    row_count: int | None = None
    checksum: str | None = None
    created_at: datetime


class DatasetSnapshotListResponse(BaseModel):
    """Schema for paginated dataset snapshot list response."""

    items: list[DatasetSnapshotResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Backtest Run Schemas
# =============================================================================


class BacktestRunCreate(BaseModel):
    """Schema for creating a new backtest run."""

    strategy_id: int = Field(..., description="Associated strategy ID")
    dataset_snapshot_id: int | None = Field(None, description="Associated dataset snapshot ID")
    sharpe: float | None = Field(None, description="Sharpe ratio")
    sortino: float | None = Field(None, description="Sortino ratio")
    max_drawdown: float | None = Field(None, ge=0, le=1, description="Maximum drawdown")
    profit_factor: float | None = Field(None, ge=0, description="Profit factor")
    win_rate: float | None = Field(None, ge=0, le=1, description="Win rate")
    num_trades: int | None = Field(None, ge=0, description="Number of trades")
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    # GPU Training Metadata (Phase 1 - GPU Determinism)
    training_device: str | None = Field(None, description="Training device: cpu or cuda")
    gpu_backend: str | None = Field(None, description="GPU backend: xgboost_gpu, pytorch_cuda, or cpu")
    cuda_version: str | None = Field(None, description="CUDA version (e.g., 12.1)")
    driver_version: str | None = Field(None, description="GPU driver version")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    determinism_flags: str | None = Field(None, description="Determinism flags for reproducibility")
    training_time_sec: float | None = Field(None, ge=0, description="Training duration in seconds")


class BacktestRunResponse(BaseModel):
    """Schema for backtest run response."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    strategy_id: int
    dataset_snapshot_id: int | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    num_trades: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] | None = Field(None, validation_alias="metadata_")
    # GPU Training Metadata (Phase 1 - GPU Determinism)
    training_device: str | None = None
    gpu_backend: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    seed: int | None = None
    determinism_flags: str | None = None
    training_time_sec: float | None = None


class BacktestRunListResponse(BaseModel):
    """Schema for paginated backtest run list response."""

    items: list[BacktestRunResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Gate Schemas
# =============================================================================


class GateCreate(BaseModel):
    """Schema for creating a new gate."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique gate name")
    description: str | None = Field(None, description="Gate description")
    gate_type: str = Field(..., min_length=1, max_length=50, description="Gate type")
    criteria: dict[str, Any] = Field(..., description="Gate criteria as JSON")
    is_active: bool = Field(True, description="Whether the gate is active")


class GateUpdate(BaseModel):
    """Schema for updating a gate."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    gate_type: str | None = Field(None, min_length=1, max_length=50)
    criteria: dict[str, Any] | None = None
    is_active: bool | None = None


class GateResponse(BaseModel):
    """Schema for gate response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    gate_type: str
    criteria: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime | None = None


class GateListResponse(BaseModel):
    """Schema for paginated gate list response."""

    items: list[GateResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Promotion Schemas
# =============================================================================


class PromotionCreate(BaseModel):
    """Schema for creating a new promotion record."""

    strategy_id: int = Field(..., description="Strategy being promoted")
    gate_id: int = Field(..., description="Gate used for evaluation")
    from_state: PromotionState = Field(..., description="State before promotion")
    to_state: PromotionState = Field(..., description="State after promotion")
    passed: bool = Field(..., description="Whether the gate was passed")
    evaluation_results: dict[str, Any] | None = Field(None, description="Evaluation details")
    promoted_by: str | None = Field(None, description="User or system that triggered promotion")


class PromotionResponse(BaseModel):
    """Schema for promotion response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    gate_id: int
    from_state: PromotionState
    to_state: PromotionState
    passed: bool
    evaluation_results: dict[str, Any] | None = None
    promoted_at: datetime
    promoted_by: str | None = None


class PromotionListResponse(BaseModel):
    """Schema for paginated promotion list response."""

    items: list[PromotionResponse]
    total: int
    skip: int
    limit: int


# =============================================================================
# Artifact Upload/Download Schemas
# =============================================================================


class ArtifactUploadResponse(BaseModel):
    """Response schema for artifact upload."""

    id: int = Field(..., description="Unique artifact identifier")
    strategy_id: int = Field(..., description="ID of the strategy")
    artifact_type: str = Field(..., description="Type of artifact")
    filename: str = Field(..., description="Stored filename")
    checksum: str = Field(..., description="SHA-256 checksum")
    size_bytes: int = Field(..., description="Size in bytes")
    message: str = Field(..., description="Status message")


class ArtifactDownloadResponse(BaseModel):
    """Response schema for artifact download metadata."""

    id: int
    strategy_id: int
    artifact_type: str
    filename: str
    checksum: str
    size_bytes: int


class ArtifactDeleteResponse(BaseModel):
    """Response schema for artifact deletion."""

    deleted: bool = Field(..., description="Whether the artifact was deleted")
    artifact_id: int = Field(..., description="ID of the deleted artifact")
    message: str = Field(..., description="Status message")


# =============================================================================
# Gate Evaluation Schemas
# =============================================================================


class GateConfigSchema(BaseModel):
    """Schema for gate configuration."""

    model_config = ConfigDict(from_attributes=True)

    gate_type: str = Field(..., description="Type of quality gate")
    threshold: float = Field(..., description="Threshold value for the gate")
    required: bool = Field(True, description="Whether passing this gate is required")


class GateResultSchema(BaseModel):
    """Schema for a single gate evaluation result."""

    model_config = ConfigDict(from_attributes=True)

    gate_type: str = Field(..., description="Type of quality gate")
    passed: bool = Field(..., description="Whether the gate passed")
    actual_value: float = Field(..., description="Actual metric value")
    threshold: float = Field(..., description="Threshold value")
    message: str = Field(..., description="Human-readable result message")
    required: bool = Field(..., description="Whether this gate is required")


class GateEvaluationRequest(BaseModel):
    """Request schema for gate evaluation."""

    sharpe: float = Field(..., description="Sharpe ratio")
    sortino: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum drawdown (0-1)")
    profit_factor: float = Field(..., ge=0, description="Profit factor")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    num_trades: int = Field(..., ge=0, description="Number of trades")
    cost_sensitivity: float | None = Field(
        None, description="Cost sensitivity metric (optional)"
    )


class GateEvaluationResponse(BaseModel):
    """Response schema for gate evaluation."""

    strategy_id: int = Field(..., description="ID of the evaluated strategy")
    passed: bool = Field(..., description="Whether all required gates passed")
    passed_count: int = Field(..., description="Number of gates that passed")
    total_count: int = Field(..., description="Total number of gates evaluated")
    required_passed: int = Field(..., description="Number of required gates that passed")
    required_total: int = Field(..., description="Total number of required gates")
    optional_passed: int = Field(..., description="Number of optional gates that passed")
    optional_total: int = Field(..., description="Total number of optional gates")
    results: list[GateResultSchema] = Field(
        default_factory=list, description="Individual gate results"
    )


class GateConfigListResponse(BaseModel):
    """Response schema for listing gate configurations."""

    gates: list[GateConfigSchema] = Field(default_factory=list)
    total: int = Field(..., description="Total number of configured gates")


class GateConfigUpdateRequest(BaseModel):
    """Request schema for updating gate configurations."""

    gates: list[GateConfigSchema] = Field(..., description="New gate configurations")


class GateConfigUpdateResponse(BaseModel):
    """Response schema for gate configuration update."""

    updated: bool = Field(..., description="Whether the configuration was updated")
    message: str = Field(..., description="Status message")
    gates: list[GateConfigSchema] = Field(default_factory=list)


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


# =============================================================================
# Promotion State Machine Schemas (T5.05)
# =============================================================================


class PromoteRequest(BaseModel):
    """Request schema for promoting a strategy."""

    target_state: str = Field(..., description="Target promotion state")


class DemoteRequest(BaseModel):
    """Request schema for demoting a strategy."""

    reason: str = Field(..., min_length=1, description="Reason for demotion")


class RetireRequest(BaseModel):
    """Request schema for retiring a strategy."""

    reason: str = Field(..., min_length=1, description="Reason for retirement")


class PromotionResultResponse(BaseModel):
    """Response schema for promotion operations."""

    success: bool = Field(..., description="Whether the operation succeeded")
    from_state: str = Field(..., description="Previous state")
    to_state: str = Field(..., description="New state")
    message: str = Field(..., description="Status message")
    requirements_met: dict[str, bool] = Field(
        default_factory=dict, description="Requirements check results"
    )


class PromotionHistoryEntry(BaseModel):
    """Schema for a single promotion history entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    from_state: str | None = None
    to_state: str | None = None
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionHistoryResponse(BaseModel):
    """Response schema for promotion history."""

    items: list[PromotionHistoryEntry]
    total: int


# =============================================================================
# Audit Log Schemas (T5.06)
# =============================================================================


class AuditEntryResponse(BaseModel):
    """Response schema for audit log entries."""

    id: int = Field(..., description="Audit entry ID")
    event_type: str = Field(..., description="Type of event")
    entity_type: str = Field(..., description="Type of entity affected")
    entity_id: int = Field(..., description="ID of entity affected")
    user_id: str | None = Field(None, description="User who triggered the event")
    old_value: dict[str, Any] | None = Field(None, description="Previous value")
    new_value: dict[str, Any] | None = Field(None, description="New value")
    message: str = Field(..., description="Event description")
    timestamp: datetime = Field(..., description="When the event occurred")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AuditQueryResponse(BaseModel):
    """Response schema for audit log queries."""

    items: list[AuditEntryResponse]
    total: int
