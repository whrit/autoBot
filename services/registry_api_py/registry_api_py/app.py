"""Registry API FastAPI application with full CRUD endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from registry_api_py.artifacts import (
    ArtifactNotFoundError,
    ArtifactStorage,
    ArtifactType,
)
from registry_api_py.database import get_db
from registry_api_py.gates import GateConfig, GateEvaluator, GateType
from registry_api_py.models import (
    Artifact,
    BacktestRun,
    DatasetSnapshot,
    Gate,
    Promotion,
    PromotionState,
    Strategy,
)
from registry_api_py.routes_promotion import router as promotion_router
from registry_api_py.schemas import (
    ArtifactCreate,
    ArtifactDeleteResponse,
    ArtifactListResponse,
    ArtifactResponse,
    ArtifactUploadResponse,
    BacktestRunCreate,
    BacktestRunListResponse,
    BacktestRunResponse,
    DatasetSnapshotCreate,
    DatasetSnapshotListResponse,
    DatasetSnapshotResponse,
    DatasetSnapshotUpdate,
    ErrorResponse,
    GateConfigListResponse,
    GateConfigSchema,
    GateConfigUpdateRequest,
    GateConfigUpdateResponse,
    GateCreate,
    GateEvaluationRequest,
    GateEvaluationResponse,
    GateListResponse,
    GateResponse,
    GateResultSchema,
    GateUpdate,
    PromotionCreate,
    PromotionListResponse,
    PromotionResponse,
    StrategyCreate,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdate,
)

# =============================================================================
# Configuration and Global Instances
# =============================================================================

ARTIFACT_STORAGE_PATH = Path(os.getenv("ARTIFACT_STORAGE_PATH", "./artifacts"))

_artifact_storage: ArtifactStorage | None = None
_gate_evaluator: GateEvaluator | None = None


def get_artifact_storage() -> ArtifactStorage:
    """Get or create the artifact storage instance."""
    global _artifact_storage
    if _artifact_storage is None:
        _artifact_storage = ArtifactStorage(ARTIFACT_STORAGE_PATH)
    return _artifact_storage


def get_gate_evaluator() -> GateEvaluator:
    """Get or create the gate evaluator instance."""
    global _gate_evaluator
    if _gate_evaluator is None:
        _gate_evaluator = GateEvaluator()
    return _gate_evaluator

app = FastAPI(
    title="Registry API",
    description="System memory and audit log for autonomous equities trading engine",
    version="0.1.0",
)


# Type aliases for dependencies
DbDep = Annotated[Session, Depends(get_db)]
StorageDep = Annotated[ArtifactStorage, Depends(get_artifact_storage)]
EvaluatorDep = Annotated[GateEvaluator, Depends(get_gate_evaluator)]


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# =============================================================================
# Strategy CRUD Endpoints
# =============================================================================


@app.post("/strategies", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
def create_strategy(strategy: StrategyCreate, db: DbDep) -> Strategy:
    """Create a new strategy."""
    db_strategy = Strategy(
        name=strategy.name,
        family=strategy.family,
        version=strategy.version,
        parameters=strategy.parameters,
        state=PromotionState.CANDIDATE,
    )
    try:
        db.add(db_strategy)
        db.commit()
        db.refresh(db_strategy)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Strategy with name '{strategy.name}' already exists",
        ) from None
    return db_strategy


@app.get("/strategies/{strategy_id}", response_model=StrategyResponse)
def get_strategy(strategy_id: int, db: DbDep) -> Strategy:
    """Get a strategy by ID."""
    db_strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if db_strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {strategy_id} not found",
        )
    return db_strategy


@app.get("/strategies", response_model=StrategyListResponse)
def list_strategies(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    family: str | None = None,
    state: PromotionState | None = None,
) -> StrategyListResponse:
    """List all strategies with optional filtering."""
    query = db.query(Strategy)

    if family:
        query = query.filter(Strategy.family == family)
    if state:
        query = query.filter(Strategy.state == state)

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return StrategyListResponse(
        items=[StrategyResponse.model_validate(s) for s in items],
        total=total,
        skip=skip,
        limit=limit,
    )


@app.put("/strategies/{strategy_id}", response_model=StrategyResponse)
def update_strategy(strategy_id: int, strategy: StrategyUpdate, db: DbDep) -> Strategy:
    """Update a strategy."""
    db_strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if db_strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {strategy_id} not found",
        )

    update_data = strategy.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_strategy, key, value)

    try:
        db.commit()
        db.refresh(db_strategy)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Strategy with name '{strategy.name}' already exists",
        ) from None
    return db_strategy


@app.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_strategy(strategy_id: int, db: DbDep) -> None:
    """Delete a strategy."""
    db_strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if db_strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {strategy_id} not found",
        )
    db.delete(db_strategy)
    db.commit()


# =============================================================================
# Dataset Snapshot CRUD Endpoints
# =============================================================================


@app.post(
    "/dataset-snapshots",
    response_model=DatasetSnapshotResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset_snapshot(snapshot: DatasetSnapshotCreate, db: DbDep) -> DatasetSnapshot:
    """Create a new dataset snapshot."""
    db_snapshot = DatasetSnapshot(
        name=snapshot.name,
        version=snapshot.version,
        start_date=snapshot.start_date,
        end_date=snapshot.end_date,
        symbols=snapshot.symbols,
        feature_schema_version=snapshot.feature_schema_version,
        row_count=snapshot.row_count,
        checksum=snapshot.checksum,
    )
    try:
        db.add(db_snapshot)
        db.commit()
        db.refresh(db_snapshot)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset snapshot with name '{snapshot.name}' already exists",
        ) from None
    return db_snapshot


@app.get("/dataset-snapshots/{snapshot_id}", response_model=DatasetSnapshotResponse)
def get_dataset_snapshot(snapshot_id: int, db: DbDep) -> DatasetSnapshot:
    """Get a dataset snapshot by ID."""
    db_snapshot = db.query(DatasetSnapshot).filter(DatasetSnapshot.id == snapshot_id).first()
    if db_snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset snapshot with id {snapshot_id} not found",
        )
    return db_snapshot


@app.get("/dataset-snapshots", response_model=DatasetSnapshotListResponse)
def list_dataset_snapshots(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    version: str | None = None,
) -> DatasetSnapshotListResponse:
    """List all dataset snapshots with optional filtering."""
    query = db.query(DatasetSnapshot)

    if version:
        query = query.filter(DatasetSnapshot.version == version)

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return DatasetSnapshotListResponse(
        items=[DatasetSnapshotResponse.model_validate(s) for s in items],
        total=total,
        skip=skip,
        limit=limit,
    )


@app.put("/dataset-snapshots/{snapshot_id}", response_model=DatasetSnapshotResponse)
def update_dataset_snapshot(
    snapshot_id: int, snapshot: DatasetSnapshotUpdate, db: DbDep
) -> DatasetSnapshot:
    """Update a dataset snapshot."""
    db_snapshot = db.query(DatasetSnapshot).filter(DatasetSnapshot.id == snapshot_id).first()
    if db_snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset snapshot with id {snapshot_id} not found",
        )

    update_data = snapshot.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_snapshot, key, value)

    try:
        db.commit()
        db.refresh(db_snapshot)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset snapshot with name '{snapshot.name}' already exists",
        ) from None
    return db_snapshot


@app.delete("/dataset-snapshots/{snapshot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_snapshot(snapshot_id: int, db: DbDep) -> None:
    """Delete a dataset snapshot."""
    db_snapshot = db.query(DatasetSnapshot).filter(DatasetSnapshot.id == snapshot_id).first()
    if db_snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset snapshot with id {snapshot_id} not found",
        )
    db.delete(db_snapshot)
    db.commit()


# =============================================================================
# Artifact CRUD Endpoints
# =============================================================================


@app.post("/artifacts", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
def create_artifact(artifact: ArtifactCreate, db: DbDep) -> Artifact:
    """Create a new artifact."""
    # Verify strategy exists
    strategy = db.query(Strategy).filter(Strategy.id == artifact.strategy_id).first()
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {artifact.strategy_id} not found",
        )

    db_artifact = Artifact(
        strategy_id=artifact.strategy_id,
        artifact_type=artifact.artifact_type,
        path=artifact.path,
        checksum=artifact.checksum,
    )
    db.add(db_artifact)
    db.commit()
    db.refresh(db_artifact)
    return db_artifact


@app.get("/artifacts/{artifact_id}", response_model=ArtifactResponse)
def get_artifact(artifact_id: int, db: DbDep) -> Artifact:
    """Get an artifact by ID."""
    db_artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if db_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact with id {artifact_id} not found",
        )
    return db_artifact


@app.get("/artifacts", response_model=ArtifactListResponse)
def list_artifacts(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    strategy_id: int | None = None,
    artifact_type: str | None = None,
) -> ArtifactListResponse:
    """List all artifacts with optional filtering."""
    query = db.query(Artifact)

    if strategy_id:
        query = query.filter(Artifact.strategy_id == strategy_id)
    if artifact_type:
        query = query.filter(Artifact.artifact_type == artifact_type)

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return ArtifactListResponse(
        items=[ArtifactResponse.model_validate(a) for a in items],
        total=total,
        skip=skip,
        limit=limit,
    )


@app.delete("/artifacts/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_artifact(artifact_id: int, db: DbDep) -> None:
    """Delete an artifact."""
    db_artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    if db_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact with id {artifact_id} not found",
        )
    db.delete(db_artifact)
    db.commit()


# =============================================================================
# Backtest Run CRUD Endpoints
# =============================================================================


@app.post(
    "/backtest-runs", response_model=BacktestRunResponse, status_code=status.HTTP_201_CREATED
)
def create_backtest_run(run: BacktestRunCreate, db: DbDep) -> BacktestRun:
    """Create a new backtest run."""
    # Verify strategy exists
    strategy = db.query(Strategy).filter(Strategy.id == run.strategy_id).first()
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {run.strategy_id} not found",
        )

    # Verify dataset snapshot exists if provided
    if run.dataset_snapshot_id:
        snapshot = (
            db.query(DatasetSnapshot)
            .filter(DatasetSnapshot.id == run.dataset_snapshot_id)
            .first()
        )
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset snapshot with id {run.dataset_snapshot_id} not found",
            )

    db_run = BacktestRun(
        strategy_id=run.strategy_id,
        dataset_snapshot_id=run.dataset_snapshot_id,
        sharpe=run.sharpe,
        sortino=run.sortino,
        max_drawdown=run.max_drawdown,
        profit_factor=run.profit_factor,
        win_rate=run.win_rate,
        num_trades=run.num_trades,
        started_at=run.started_at,
        completed_at=run.completed_at,
        metadata_=run.metadata,
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run


@app.get("/backtest-runs/{run_id}", response_model=BacktestRunResponse)
def get_backtest_run(run_id: int, db: DbDep) -> BacktestRun:
    """Get a backtest run by ID."""
    db_run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
    if db_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run with id {run_id} not found",
        )
    return db_run


@app.get("/backtest-runs", response_model=BacktestRunListResponse)
def list_backtest_runs(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    strategy_id: int | None = None,
    dataset_snapshot_id: int | None = None,
) -> BacktestRunListResponse:
    """List all backtest runs with optional filtering."""
    query = db.query(BacktestRun)

    if strategy_id:
        query = query.filter(BacktestRun.strategy_id == strategy_id)
    if dataset_snapshot_id:
        query = query.filter(BacktestRun.dataset_snapshot_id == dataset_snapshot_id)

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return BacktestRunListResponse(
        items=[BacktestRunResponse.model_validate(r) for r in items],
        total=total,
        skip=skip,
        limit=limit,
    )


# =============================================================================
# Gate CRUD Endpoints
# =============================================================================


@app.post("/gates", response_model=GateResponse, status_code=status.HTTP_201_CREATED)
def create_gate(gate: GateCreate, db: DbDep) -> Gate:
    """Create a new gate."""
    db_gate = Gate(
        name=gate.name,
        description=gate.description,
        gate_type=gate.gate_type,
        criteria=gate.criteria,
        is_active=1 if gate.is_active else 0,
    )
    try:
        db.add(db_gate)
        db.commit()
        db.refresh(db_gate)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gate with name '{gate.name}' already exists",
        ) from None
    return db_gate


@app.get("/gates/{gate_id}", response_model=GateResponse)
def get_gate(gate_id: int, db: DbDep) -> Gate:
    """Get a gate by ID."""
    db_gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if db_gate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gate with id {gate_id} not found",
        )
    return db_gate


@app.get("/gates", response_model=GateListResponse)
def list_gates(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    gate_type: str | None = None,
    is_active: bool | None = None,
) -> GateListResponse:
    """List all gates with optional filtering."""
    query = db.query(Gate)

    if gate_type:
        query = query.filter(Gate.gate_type == gate_type)
    if is_active is not None:
        query = query.filter(Gate.is_active == (1 if is_active else 0))

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return GateListResponse(
        items=[GateResponse.model_validate(g) for g in items],
        total=total,
        skip=skip,
        limit=limit,
    )


@app.put("/gates/{gate_id}", response_model=GateResponse)
def update_gate(gate_id: int, gate: GateUpdate, db: DbDep) -> Gate:
    """Update a gate."""
    db_gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if db_gate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gate with id {gate_id} not found",
        )

    update_data = gate.model_dump(exclude_unset=True)

    # Convert boolean to integer for SQLite
    if "is_active" in update_data:
        update_data["is_active"] = 1 if update_data["is_active"] else 0

    for key, value in update_data.items():
        setattr(db_gate, key, value)

    try:
        db.commit()
        db.refresh(db_gate)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gate with name '{gate.name}' already exists",
        ) from None
    return db_gate


@app.delete("/gates/{gate_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_gate(gate_id: int, db: DbDep) -> None:
    """Delete a gate."""
    db_gate = db.query(Gate).filter(Gate.id == gate_id).first()
    if db_gate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gate with id {gate_id} not found",
        )
    db.delete(db_gate)
    db.commit()


# =============================================================================
# Promotion CRUD Endpoints
# =============================================================================


@app.post("/promotions", response_model=PromotionResponse, status_code=status.HTTP_201_CREATED)
def create_promotion(promotion: PromotionCreate, db: DbDep) -> Promotion:
    """Create a new promotion record."""
    # Verify strategy exists
    strategy = db.query(Strategy).filter(Strategy.id == promotion.strategy_id).first()
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id {promotion.strategy_id} not found",
        )

    # Verify gate exists
    gate = db.query(Gate).filter(Gate.id == promotion.gate_id).first()
    if gate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gate with id {promotion.gate_id} not found",
        )

    db_promotion = Promotion(
        strategy_id=promotion.strategy_id,
        gate_id=promotion.gate_id,
        from_state=promotion.from_state,
        to_state=promotion.to_state,
        passed=1 if promotion.passed else 0,
        evaluation_results=promotion.evaluation_results,
        promoted_by=promotion.promoted_by,
    )
    db.add(db_promotion)
    db.commit()
    db.refresh(db_promotion)
    return db_promotion


@app.get("/promotions/{promotion_id}", response_model=PromotionResponse)
def get_promotion(promotion_id: int, db: DbDep) -> Promotion:
    """Get a promotion by ID."""
    db_promotion = db.query(Promotion).filter(Promotion.id == promotion_id).first()
    if db_promotion is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Promotion with id {promotion_id} not found",
        )
    return db_promotion


@app.get("/promotions", response_model=PromotionListResponse)
def list_promotions(
    db: DbDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    strategy_id: int | None = None,
    gate_id: int | None = None,
    passed: bool | None = None,
) -> PromotionListResponse:
    """List all promotions with optional filtering."""
    query = db.query(Promotion)

    if strategy_id:
        query = query.filter(Promotion.strategy_id == strategy_id)
    if gate_id:
        query = query.filter(Promotion.gate_id == gate_id)
    if passed is not None:
        query = query.filter(Promotion.passed == (1 if passed else 0))

    total = query.count()
    items = query.offset(skip).limit(limit).all()

    return PromotionListResponse(
        items=[PromotionResponse.model_validate(p) for p in items],
        total=total,
        skip=skip,
        limit=limit,
    )


# =============================================================================
# Artifact Upload/Download Endpoints (T5.03)
# =============================================================================


def _validate_artifact_type(artifact_type: str) -> ArtifactType:
    """Validate and convert artifact type string to enum."""
    try:
        return ArtifactType(artifact_type)
    except ValueError as e:
        valid_types = [t.value for t in ArtifactType]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid artifact type '{artifact_type}'. Valid types: {valid_types}",
        ) from e


@app.post(
    "/strategies/{strategy_id}/artifacts",
    response_model=ArtifactUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["Artifact Storage"],
)
async def upload_artifact(
    strategy_id: int,
    file: UploadFile,
    artifact_type: str,
    db: DbDep,
    storage: StorageDep,
) -> ArtifactUploadResponse:
    """Upload an artifact file for a strategy.

    Supports ONNX models, JSON configs, parameters, and metrics files.
    Computes SHA-256 checksum automatically.

    Args:
        strategy_id: ID of the strategy to attach artifact to.
        file: The artifact file to upload.
        artifact_type: Type of artifact (onnx, json, params, metrics).

    Returns:
        Metadata for the uploaded artifact.
    """
    # Verify strategy exists
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    # Validate artifact type
    validated_type = _validate_artifact_type(artifact_type)

    # Read file content
    content = await file.read()

    # Upload to storage
    metadata = storage.upload(
        strategy_id=strategy_id,
        artifact_type=validated_type,
        data=content,
        filename=file.filename,
    )

    return ArtifactUploadResponse(
        id=metadata.artifact_id,
        strategy_id=metadata.strategy_id,
        artifact_type=metadata.artifact_type.value,
        filename=metadata.filename,
        checksum=metadata.checksum,
        size_bytes=metadata.size_bytes,
        message="Artifact uploaded successfully",
    )


@app.get(
    "/strategies/{strategy_id}/artifacts",
    response_model=ArtifactListResponse,
    tags=["Artifact Storage"],
)
async def list_strategy_artifacts(
    strategy_id: int,
    storage: StorageDep,
) -> ArtifactListResponse:
    """List all artifacts for a strategy.

    Args:
        strategy_id: ID of the strategy.

    Returns:
        List of artifact metadata.
    """
    artifacts = storage.list_by_strategy(strategy_id)

    return ArtifactListResponse(
        items=[
            ArtifactResponse(
                id=a.artifact_id,
                strategy_id=a.strategy_id,
                artifact_type=a.artifact_type.value,
                path=f"artifacts/{a.strategy_id}/{a.filename}",
                checksum=a.checksum,
                created_at=a.created_at,
            )
            for a in artifacts
        ],
        total=len(artifacts),
        skip=0,
        limit=len(artifacts),
    )


@app.get(
    "/artifacts/{artifact_id}/download",
    responses={404: {"model": ErrorResponse}},
    tags=["Artifact Storage"],
)
async def download_artifact_file(
    artifact_id: int,
    storage: StorageDep,
) -> Response:
    """Download an artifact file by ID.

    Returns the raw file content with appropriate content type.

    Args:
        artifact_id: ID of the artifact to download.

    Returns:
        The artifact file content.
    """
    try:
        metadata = storage.get_metadata(artifact_id)
        content = storage.download(artifact_id)

        # Determine content type based on artifact type
        content_types = {
            ArtifactType.ONNX: "application/octet-stream",
            ArtifactType.JSON_CONFIG: "application/json",
            ArtifactType.PARAMETERS: "application/json",
            ArtifactType.METRICS: "application/json",
        }
        content_type = content_types.get(metadata.artifact_type, "application/octet-stream")

        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{metadata.filename}"',
                "X-Checksum-SHA256": metadata.checksum,
            },
        )

    except ArtifactNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found",
        ) from e


@app.delete(
    "/artifacts/{artifact_id}",
    response_model=ArtifactDeleteResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Artifact Storage"],
)
async def delete_artifact_file(
    artifact_id: int,
    storage: StorageDep,
) -> ArtifactDeleteResponse:
    """Delete an artifact file by ID.

    Args:
        artifact_id: ID of the artifact to delete.

    Returns:
        Deletion confirmation.
    """
    deleted = storage.delete(artifact_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found",
        )

    return ArtifactDeleteResponse(
        deleted=True,
        artifact_id=artifact_id,
        message="Artifact deleted successfully",
    )


# =============================================================================
# Gate Evaluation Endpoints (T5.04)
# =============================================================================


def _validate_gate_type(gate_type: str) -> GateType:
    """Validate and convert gate type string to enum."""
    try:
        return GateType(gate_type)
    except ValueError as e:
        valid_types = [t.value for t in GateType]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid gate type '{gate_type}'. Valid types: {valid_types}",
        ) from e


@app.post(
    "/strategies/{strategy_id}/evaluate-gates",
    response_model=GateEvaluationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Gate Evaluation"],
)
async def evaluate_strategy_gates(
    strategy_id: int,
    metrics: GateEvaluationRequest,
    db: DbDep,
    evaluator: EvaluatorDep,
) -> GateEvaluationResponse:
    """Evaluate a strategy against quality gates.

    Uses configurable quality gates to determine if a strategy
    meets minimum performance thresholds.

    Default gates:
    - Sharpe ratio >= 0.5 (required)
    - Max drawdown <= 15% (required)
    - Minimum 30 trades (required)
    - Win rate >= 45% (optional)

    Args:
        strategy_id: ID of the strategy to evaluate.
        metrics: Strategy performance metrics.

    Returns:
        Gate evaluation results with pass/fail status.
    """
    # Verify strategy exists
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found",
        )

    # Convert request to dict for evaluation
    metrics_dict = metrics.model_dump(exclude_none=True)

    # Evaluate gates
    results = evaluator.evaluate(metrics_dict)
    summary = evaluator.get_summary(results)

    return GateEvaluationResponse(
        strategy_id=strategy_id,
        passed=summary["passed"],
        passed_count=summary["passed_count"],
        total_count=summary["total_count"],
        required_passed=summary["required_passed"],
        required_total=summary["required_total"],
        optional_passed=summary["optional_passed"],
        optional_total=summary["optional_total"],
        results=[
            GateResultSchema(
                gate_type=r.gate_type.value,
                passed=r.passed,
                actual_value=r.actual_value,
                threshold=r.threshold,
                message=r.message,
                required=r.required,
            )
            for r in results
        ],
    )


@app.get(
    "/gates/config",
    response_model=GateConfigListResponse,
    tags=["Gate Evaluation"],
)
async def get_evaluation_gate_config(
    evaluator: EvaluatorDep,
) -> GateConfigListResponse:
    """Get current gate evaluation configuration.

    Returns the list of quality gates and their thresholds
    used for strategy evaluation.

    Returns:
        Current gate configuration.
    """
    gates = evaluator.gates

    return GateConfigListResponse(
        gates=[
            GateConfigSchema(
                gate_type=g.gate_type.value,
                threshold=g.threshold,
                required=g.required,
            )
            for g in gates
        ],
        total=len(gates),
    )


@app.put(
    "/gates/config",
    response_model=GateConfigUpdateResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["Gate Evaluation"],
)
async def update_evaluation_gate_config(
    request: GateConfigUpdateRequest,
    evaluator: EvaluatorDep,
) -> GateConfigUpdateResponse:
    """Update gate evaluation configuration.

    Replaces all existing gates with the provided configuration.
    Use this to customize quality thresholds for strategy evaluation.

    Args:
        request: New gate configuration.

    Returns:
        Updated gate configuration.
    """
    # Validate all gate types
    new_gates: list[GateConfig] = []
    for gate_schema in request.gates:
        gate_type = _validate_gate_type(gate_schema.gate_type)
        new_gates.append(
            GateConfig(
                gate_type=gate_type,
                threshold=gate_schema.threshold,
                required=gate_schema.required,
            )
        )

    # Update evaluator
    evaluator.gates = new_gates

    return GateConfigUpdateResponse(
        updated=True,
        message=f"Gate configuration updated with {len(new_gates)} gates",
        gates=[
            GateConfigSchema(
                gate_type=g.gate_type.value,
                threshold=g.threshold,
                required=g.required,
            )
            for g in new_gates
        ],
    )


# =============================================================================
# Include Promotion and Audit Routes (T5.05, T5.06)
# =============================================================================

app.include_router(promotion_router)


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
