"""Tests for artifact and gate API endpoints."""

from __future__ import annotations

import io
import json
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
from registry_api_py.models import Base, Strategy

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Test database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_engine() -> Generator[Engine, None, None]:
    """Create a test database engine."""
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
    """Create a temporary artifact storage."""
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
    """Create a test client with overridden dependencies."""

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

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
def sample_strategy(db_session: Session) -> Strategy:
    """Create a sample strategy in the database."""
    strategy = Strategy(
        name="test_strategy_v1",
        family="trend",
        version="1.0.0",
        parameters={"lookback": 20, "threshold": 0.02},
    )
    db_session.add(strategy)
    db_session.commit()
    db_session.refresh(strategy)
    return strategy


class TestArtifactEndpoints:
    """Tests for artifact upload/download endpoints."""

    def test_upload_via_api(self, client: TestClient, sample_strategy: Strategy) -> None:
        """Test uploading an artifact via API."""
        config_data = json.dumps({"param1": 1.0, "param2": 0.5})

        response = client.post(
            f"/strategies/{sample_strategy.id}/artifacts",
            files={"file": ("config.json", io.BytesIO(config_data.encode()), "application/json")},
            data={"artifact_type": "json"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["strategy_id"] == sample_strategy.id
        assert data["artifact_type"] == "json"
        assert data["size_bytes"] == len(config_data)
        assert "checksum" in data
        assert data["message"] == "Artifact uploaded successfully"

    def test_upload_onnx_artifact(self, client: TestClient, sample_strategy: Strategy) -> None:
        """Test uploading an ONNX model artifact."""
        onnx_data = b"ONNX_MOCK_MODEL_DATA"

        response = client.post(
            f"/strategies/{sample_strategy.id}/artifacts",
            files={"file": ("model.onnx", io.BytesIO(onnx_data), "application/octet-stream")},
            data={"artifact_type": "onnx"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["artifact_type"] == "onnx"
        assert data["size_bytes"] == len(onnx_data)

    def test_download_via_api(self, client: TestClient, sample_strategy: Strategy) -> None:
        """Test downloading an artifact via API."""
        # First upload
        config_data = json.dumps({"test": "data"})
        upload_response = client.post(
            f"/strategies/{sample_strategy.id}/artifacts",
            files={"file": ("config.json", io.BytesIO(config_data.encode()), "application/json")},
            data={"artifact_type": "json"},
        )
        artifact_id = upload_response.json()["id"]

        # Then download
        response = client.get(f"/artifacts/{artifact_id}/download")

        assert response.status_code == 200
        assert response.content == config_data.encode()

    def test_download_nonexistent_artifact(self, client: TestClient) -> None:
        """Test downloading a non-existent artifact returns 404."""
        response = client.get("/artifacts/9999/download")
        assert response.status_code == 404

    def test_list_artifacts_for_strategy(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test listing artifacts for a strategy."""
        # Upload multiple artifacts
        for i in range(3):
            config_data = json.dumps({"version": i})
            client.post(
                f"/strategies/{sample_strategy.id}/artifacts",
                files={
                    "file": (f"config_{i}.json", io.BytesIO(config_data.encode()), "application/json")
                },
                data={"artifact_type": "json"},
            )

        response = client.get(f"/strategies/{sample_strategy.id}/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_artifacts_empty(self, client: TestClient, sample_strategy: Strategy) -> None:
        """Test listing artifacts when none exist."""
        response = client.get(f"/strategies/{sample_strategy.id}/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_delete_artifact(self, client: TestClient, sample_strategy: Strategy) -> None:
        """Test deleting an artifact."""
        # First upload
        config_data = json.dumps({"test": "data"})
        upload_response = client.post(
            f"/strategies/{sample_strategy.id}/artifacts",
            files={"file": ("config.json", io.BytesIO(config_data.encode()), "application/json")},
            data={"artifact_type": "json"},
        )
        artifact_id = upload_response.json()["id"]

        # Then delete
        response = client.delete(f"/artifacts/{artifact_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["artifact_id"] == artifact_id

        # Verify it's gone
        download_response = client.get(f"/artifacts/{artifact_id}/download")
        assert download_response.status_code == 404

    def test_delete_nonexistent_artifact(self, client: TestClient) -> None:
        """Test deleting a non-existent artifact."""
        response = client.delete("/artifacts/9999")
        assert response.status_code == 404

    def test_upload_invalid_artifact_type(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test uploading with invalid artifact type returns 422."""
        response = client.post(
            f"/strategies/{sample_strategy.id}/artifacts",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
            data={"artifact_type": "invalid_type"},
        )

        assert response.status_code == 422

    def test_upload_to_nonexistent_strategy(self, client: TestClient) -> None:
        """Test uploading to non-existent strategy returns 404."""
        response = client.post(
            "/strategies/9999/artifacts",
            files={"file": ("test.json", io.BytesIO(b"{}"), "application/json")},
            data={"artifact_type": "json"},
        )

        assert response.status_code == 404


class TestGateEndpoints:
    """Tests for gate evaluation endpoints."""

    def test_evaluate_gates_api_passes(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test gate evaluation that passes all gates."""
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

        response = client.post(
            f"/strategies/{sample_strategy.id}/evaluate-gates",
            json=metrics,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is True
        assert data["strategy_id"] == sample_strategy.id
        assert data["total_count"] == 4
        assert data["passed_count"] == 4

    def test_evaluate_gates_api_fails(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test gate evaluation that fails."""
        metrics = {
            "sharpe": 0.2,  # Below threshold
            "sortino": 0.5,
            "max_drawdown": 0.30,  # Above threshold
            "profit_factor": 0.8,
            "win_rate": 0.35,
            "num_trades": 10,  # Below threshold
        }

        response = client.post(
            f"/strategies/{sample_strategy.id}/evaluate-gates",
            json=metrics,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is False
        assert data["passed_count"] < data["total_count"]

    def test_evaluate_gates_nonexistent_strategy(self, client: TestClient) -> None:
        """Test gate evaluation for non-existent strategy."""
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

        response = client.post("/strategies/9999/evaluate-gates", json=metrics)
        assert response.status_code == 404

    def test_get_gate_config(self, client: TestClient) -> None:
        """Test getting current gate configuration."""
        response = client.get("/gates/config")

        assert response.status_code == 200
        data = response.json()
        assert "gates" in data
        assert data["total"] == 4  # Default gates count
        assert any(g["gate_type"] == "min_sharpe" for g in data["gates"])

    def test_update_gate_config(self, client: TestClient) -> None:
        """Test updating gate configuration."""
        new_config = {
            "gates": [
                {"gate_type": "min_sharpe", "threshold": 2.0, "required": True},
                {"gate_type": "max_drawdown", "threshold": 0.20, "required": True},
            ]
        }

        response = client.put("/gates/config", json=new_config)

        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert len(data["gates"]) == 2

        # Verify by getting config again
        get_response = client.get("/gates/config")
        assert get_response.json()["total"] == 2

    def test_update_gate_config_invalid_type(self, client: TestClient) -> None:
        """Test updating with invalid gate type returns 422."""
        new_config = {
            "gates": [
                {"gate_type": "invalid_gate", "threshold": 1.0, "required": True},
            ]
        }

        response = client.put("/gates/config", json=new_config)
        assert response.status_code == 422

    def test_evaluate_gates_results_structure(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test that gate evaluation results have correct structure."""
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
        }

        response = client.post(
            f"/strategies/{sample_strategy.id}/evaluate-gates",
            json=metrics,
        )

        data = response.json()
        assert "results" in data

        for result in data["results"]:
            assert "gate_type" in result
            assert "passed" in result
            assert "actual_value" in result
            assert "threshold" in result
            assert "message" in result
            assert "required" in result

    def test_evaluate_gates_missing_optional_metric(
        self, client: TestClient, sample_strategy: Strategy
    ) -> None:
        """Test evaluation with missing optional metric still works."""
        metrics = {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": 0.10,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "num_trades": 100,
            # cost_sensitivity is optional
        }

        response = client.post(
            f"/strategies/{sample_strategy.id}/evaluate-gates",
            json=metrics,
        )

        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
