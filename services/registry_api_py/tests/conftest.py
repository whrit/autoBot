"""Pytest configuration and fixtures for Registry API tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from registry_api_py.app import app, get_artifact_storage, get_gate_evaluator
from registry_api_py.artifacts import ArtifactStorage
from registry_api_py.database import get_db
from registry_api_py.gates import GateEvaluator
from registry_api_py.models import Base

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_engine() -> Generator[Engine, None, None]:
    """Create a test database engine with in-memory SQLite."""
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def artifact_storage() -> Generator[ArtifactStorage, None, None]:
    """Create a temporary artifact storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArtifactStorage(Path(tmpdir) / "test_artifacts")
        yield storage
        storage.cleanup()


@pytest.fixture(scope="function")
def gate_evaluator() -> GateEvaluator:
    """Create a gate evaluator with default gates."""
    return GateEvaluator()


@pytest.fixture(scope="function")
def client(
    db_session: Session,
    artifact_storage: ArtifactStorage,
    gate_evaluator: GateEvaluator,
) -> Generator[TestClient, None, None]:
    """Create a test client with overridden database dependency."""

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    def override_get_storage() -> ArtifactStorage:
        return artifact_storage

    def override_get_evaluator() -> GateEvaluator:
        return gate_evaluator

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_artifact_storage] = override_get_storage
    app.dependency_overrides[get_gate_evaluator] = override_get_evaluator

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_strategy_data() -> dict[str, object]:
    """Sample strategy data for testing."""
    return {
        "name": "test_momentum_v1",
        "family": "trend",
        "version": "1.0.0",
        "parameters": {
            "lookback_period": 20,
            "threshold": 0.02,
            "max_position": 1000,
        },
    }


@pytest.fixture
def sample_dataset_snapshot_data() -> dict[str, object]:
    """Sample dataset snapshot data for testing."""
    return {
        "name": "spy_qqq_jan_2024",
        "version": "1.0.0",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-01-31T23:59:59",
        "symbols": ["SPY", "QQQ"],
        "feature_schema_version": "v1.2.0",
        "row_count": 100000,
        "checksum": "sha256:abc123def456",
    }


@pytest.fixture
def sample_artifact_data() -> dict[str, object]:
    """Sample artifact data for testing."""
    return {
        "artifact_type": "onnx",
        "path": "s3://models/strategy_123/model.onnx",
        "checksum": "sha256:xyz789",
    }


@pytest.fixture
def sample_backtest_run_data() -> dict[str, object]:
    """Sample backtest run data for testing."""
    return {
        "sharpe": 1.5,
        "sortino": 2.0,
        "max_drawdown": 0.15,
        "profit_factor": 1.8,
        "win_rate": 0.55,
        "num_trades": 150,
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
        "metadata": {"regime": "bull", "slippage_model": "linear"},
    }


@pytest.fixture
def sample_gate_data() -> dict[str, object]:
    """Sample gate data for testing."""
    return {
        "name": "backtest_gate_v1",
        "description": "Minimum backtest performance gate",
        "gate_type": "backtest",
        "criteria": {
            "min_sharpe": 1.0,
            "min_win_rate": 0.5,
            "max_drawdown": 0.2,
        },
        "is_active": True,
    }
