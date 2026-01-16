"""CRUD tests for Registry API - TDD approach."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestStrategyCRUD:
    """Test suite for Strategy CRUD operations."""

    def test_create_strategy(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test creating a new strategy."""
        response = client.post("/strategies", json=sample_strategy_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_strategy_data["name"]
        assert data["family"] == sample_strategy_data["family"]
        assert data["version"] == sample_strategy_data["version"]
        assert data["parameters"] == sample_strategy_data["parameters"]
        assert data["state"] == "candidate"
        assert "id" in data
        assert "created_at" in data

    def test_create_strategy_duplicate_name(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test that creating a strategy with duplicate name fails."""
        response = client.post("/strategies", json=sample_strategy_data)
        assert response.status_code == 201

        response = client.post("/strategies", json=sample_strategy_data)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_create_strategy_invalid_data(self, client: TestClient) -> None:
        """Test creating a strategy with invalid data."""
        invalid_data = {"name": ""}  # Missing required fields
        response = client.post("/strategies", json=invalid_data)
        assert response.status_code == 422

    def test_get_strategy(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test getting a strategy by ID."""
        # Create a strategy first
        create_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = create_response.json()["id"]

        # Get the strategy
        response = client.get(f"/strategies/{strategy_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == strategy_id
        assert data["name"] == sample_strategy_data["name"]

    def test_get_strategy_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent strategy."""
        response = client.get("/strategies/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_strategies(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test listing all strategies."""
        # Create multiple strategies
        for i in range(3):
            data = {**sample_strategy_data, "name": f"strategy_{i}"}
            client.post("/strategies", json=data)

        response = client.get("/strategies")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_strategies_pagination(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test pagination of strategy listing."""
        # Create multiple strategies
        for i in range(5):
            data = {**sample_strategy_data, "name": f"strategy_{i}"}
            client.post("/strategies", json=data)

        # Test skip and limit
        response = client.get("/strategies?skip=2&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["skip"] == 2
        assert data["limit"] == 2

    def test_filter_strategies_by_family(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test filtering strategies by family."""
        # Create strategies with different families
        client.post("/strategies", json={**sample_strategy_data, "name": "trend_1", "family": "trend"})
        client.post("/strategies", json={**sample_strategy_data, "name": "mr_1", "family": "mean_reversion"})
        client.post("/strategies", json={**sample_strategy_data, "name": "trend_2", "family": "trend"})

        response = client.get("/strategies?family=trend")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert item["family"] == "trend"

    def test_filter_strategies_by_state(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test filtering strategies by state."""
        # Create strategy and update state
        response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = response.json()["id"]
        client.put(f"/strategies/{strategy_id}", json={"state": "shadow"})

        # Create another strategy with default state
        client.post("/strategies", json={**sample_strategy_data, "name": "strategy_2"})

        response = client.get("/strategies?state=candidate")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["state"] == "candidate"

    def test_update_strategy(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test updating a strategy."""
        # Create a strategy
        create_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = create_response.json()["id"]

        # Update the strategy
        update_data = {"version": "2.0.0", "state": "shadow"}
        response = client.put(f"/strategies/{strategy_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0.0"
        assert data["state"] == "shadow"

    def test_update_strategy_not_found(self, client: TestClient) -> None:
        """Test updating a non-existent strategy."""
        response = client.put("/strategies/99999", json={"version": "2.0.0"})
        assert response.status_code == 404

    def test_update_strategy_partial(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test partial update of a strategy."""
        create_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = create_response.json()["id"]

        # Only update version
        response = client.put(f"/strategies/{strategy_id}", json={"version": "1.1.0"})
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.1.0"
        assert data["name"] == sample_strategy_data["name"]  # Unchanged

    def test_delete_strategy(
        self, client: TestClient, sample_strategy_data: dict[str, Any]
    ) -> None:
        """Test deleting a strategy."""
        # Create a strategy
        create_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = create_response.json()["id"]

        # Delete the strategy
        response = client.delete(f"/strategies/{strategy_id}")
        assert response.status_code == 204

        # Verify it's deleted
        response = client.get(f"/strategies/{strategy_id}")
        assert response.status_code == 404

    def test_delete_strategy_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent strategy."""
        response = client.delete("/strategies/99999")
        assert response.status_code == 404


class TestDatasetSnapshotCRUD:
    """Test suite for Dataset Snapshot CRUD operations."""

    def test_create_snapshot(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test creating a new dataset snapshot."""
        response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_dataset_snapshot_data["name"]
        assert data["version"] == sample_dataset_snapshot_data["version"]
        assert data["symbols"] == sample_dataset_snapshot_data["symbols"]
        assert "id" in data
        assert "created_at" in data

    def test_create_snapshot_duplicate_name(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test that creating a snapshot with duplicate name fails."""
        response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        assert response.status_code == 201

        response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_create_snapshot_minimal_data(self, client: TestClient) -> None:
        """Test creating a snapshot with minimal required data."""
        minimal_data = {
            "name": "minimal_snapshot",
            "version": "1.0.0",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-31T23:59:59",
            "symbols": ["SPY"],
        }
        response = client.post("/dataset-snapshots", json=minimal_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "minimal_snapshot"
        assert data["feature_schema_version"] is None

    def test_get_snapshot(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test getting a dataset snapshot by ID."""
        create_response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        snapshot_id = create_response.json()["id"]

        response = client.get(f"/dataset-snapshots/{snapshot_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == snapshot_id
        assert data["name"] == sample_dataset_snapshot_data["name"]

    def test_get_snapshot_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent snapshot."""
        response = client.get("/dataset-snapshots/99999")
        assert response.status_code == 404

    def test_list_snapshots(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test listing all dataset snapshots."""
        for i in range(3):
            data = {**sample_dataset_snapshot_data, "name": f"snapshot_{i}"}
            client.post("/dataset-snapshots", json=data)

        response = client.get("/dataset-snapshots")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_snapshots_pagination(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test pagination of snapshot listing."""
        for i in range(5):
            data = {**sample_dataset_snapshot_data, "name": f"snapshot_{i}"}
            client.post("/dataset-snapshots", json=data)

        response = client.get("/dataset-snapshots?skip=1&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["skip"] == 1
        assert data["limit"] == 2

    def test_filter_snapshots_by_version(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test filtering snapshots by version."""
        client.post("/dataset-snapshots", json={**sample_dataset_snapshot_data, "name": "snap_1", "version": "1.0.0"})
        client.post("/dataset-snapshots", json={**sample_dataset_snapshot_data, "name": "snap_2", "version": "2.0.0"})
        client.post("/dataset-snapshots", json={**sample_dataset_snapshot_data, "name": "snap_3", "version": "1.0.0"})

        response = client.get("/dataset-snapshots?version=1.0.0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert item["version"] == "1.0.0"

    def test_update_snapshot(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test updating a dataset snapshot."""
        create_response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        snapshot_id = create_response.json()["id"]

        update_data = {"version": "2.0.0", "row_count": 200000}
        response = client.put(f"/dataset-snapshots/{snapshot_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0.0"
        assert data["row_count"] == 200000

    def test_delete_snapshot(
        self, client: TestClient, sample_dataset_snapshot_data: dict[str, Any]
    ) -> None:
        """Test deleting a dataset snapshot."""
        create_response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        snapshot_id = create_response.json()["id"]

        response = client.delete(f"/dataset-snapshots/{snapshot_id}")
        assert response.status_code == 204

        response = client.get(f"/dataset-snapshots/{snapshot_id}")
        assert response.status_code == 404


class TestArtifactCRUD:
    """Test suite for Artifact CRUD operations."""

    def test_create_artifact(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_artifact_data: dict[str, Any],
    ) -> None:
        """Test creating a new artifact."""
        # Create strategy first
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        artifact_data = {**sample_artifact_data, "strategy_id": strategy_id}
        response = client.post("/artifacts", json=artifact_data)
        assert response.status_code == 201
        data = response.json()
        assert data["strategy_id"] == strategy_id
        assert data["artifact_type"] == sample_artifact_data["artifact_type"]

    def test_get_artifact(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_artifact_data: dict[str, Any],
    ) -> None:
        """Test getting an artifact by ID."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        artifact_data = {**sample_artifact_data, "strategy_id": strategy_id}
        create_response = client.post("/artifacts", json=artifact_data)
        artifact_id = create_response.json()["id"]

        response = client.get(f"/artifacts/{artifact_id}")
        assert response.status_code == 200
        assert response.json()["id"] == artifact_id

    def test_list_artifacts_by_strategy(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_artifact_data: dict[str, Any],
    ) -> None:
        """Test listing artifacts filtered by strategy."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        # Create multiple artifacts for the strategy
        for artifact_type in ["onnx", "json", "config"]:
            artifact_data = {
                **sample_artifact_data,
                "strategy_id": strategy_id,
                "artifact_type": artifact_type,
                "path": f"s3://artifacts/{artifact_type}",
            }
            client.post("/artifacts", json=artifact_data)

        response = client.get(f"/artifacts?strategy_id={strategy_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3

    def test_delete_artifact(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_artifact_data: dict[str, Any],
    ) -> None:
        """Test deleting an artifact."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        artifact_data = {**sample_artifact_data, "strategy_id": strategy_id}
        create_response = client.post("/artifacts", json=artifact_data)
        artifact_id = create_response.json()["id"]

        response = client.delete(f"/artifacts/{artifact_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        response = client.get(f"/artifacts/{artifact_id}")
        assert response.status_code == 404


class TestBacktestRunCRUD:
    """Test suite for Backtest Run CRUD operations."""

    def test_create_backtest_run(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a new backtest run."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {**sample_backtest_run_data, "strategy_id": strategy_id}
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["strategy_id"] == strategy_id
        assert data["sharpe"] == sample_backtest_run_data["sharpe"]

    def test_create_backtest_run_with_snapshot(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_dataset_snapshot_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a backtest run linked to a dataset snapshot."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        snapshot_response = client.post("/dataset-snapshots", json=sample_dataset_snapshot_data)
        snapshot_id = snapshot_response.json()["id"]

        run_data = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "dataset_snapshot_id": snapshot_id,
        }
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["dataset_snapshot_id"] == snapshot_id

    def test_list_backtest_runs_by_strategy(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test listing backtest runs filtered by strategy."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        # Create multiple runs
        for i in range(3):
            run_data = {**sample_backtest_run_data, "strategy_id": strategy_id, "sharpe": 1.0 + i * 0.5}
            client.post("/backtest-runs", json=run_data)

        response = client.get(f"/backtest-runs?strategy_id={strategy_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3


class TestGateCRUD:
    """Test suite for Gate CRUD operations."""

    def test_create_gate(
        self, client: TestClient, sample_gate_data: dict[str, Any]
    ) -> None:
        """Test creating a new gate."""
        response = client.post("/gates", json=sample_gate_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_gate_data["name"]
        assert data["gate_type"] == sample_gate_data["gate_type"]
        assert data["criteria"] == sample_gate_data["criteria"]

    def test_get_gate(
        self, client: TestClient, sample_gate_data: dict[str, Any]
    ) -> None:
        """Test getting a gate by ID."""
        create_response = client.post("/gates", json=sample_gate_data)
        gate_id = create_response.json()["id"]

        response = client.get(f"/gates/{gate_id}")
        assert response.status_code == 200
        assert response.json()["id"] == gate_id

    def test_list_gates(
        self, client: TestClient, sample_gate_data: dict[str, Any]
    ) -> None:
        """Test listing all gates."""
        for i in range(3):
            data = {**sample_gate_data, "name": f"gate_{i}"}
            client.post("/gates", json=data)

        response = client.get("/gates")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3

    def test_update_gate(
        self, client: TestClient, sample_gate_data: dict[str, Any]
    ) -> None:
        """Test updating a gate."""
        create_response = client.post("/gates", json=sample_gate_data)
        gate_id = create_response.json()["id"]

        update_data = {"is_active": False, "criteria": {"min_sharpe": 1.5}}
        response = client.put(f"/gates/{gate_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False
        assert data["criteria"]["min_sharpe"] == 1.5

    def test_delete_gate(
        self, client: TestClient, sample_gate_data: dict[str, Any]
    ) -> None:
        """Test deleting a gate."""
        create_response = client.post("/gates", json=sample_gate_data)
        gate_id = create_response.json()["id"]

        response = client.delete(f"/gates/{gate_id}")
        assert response.status_code == 204

        response = client.get(f"/gates/{gate_id}")
        assert response.status_code == 404


class TestPromotionCRUD:
    """Test suite for Promotion CRUD operations."""

    def test_create_promotion(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_gate_data: dict[str, Any],
    ) -> None:
        """Test creating a new promotion record."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        gate_response = client.post("/gates", json=sample_gate_data)
        gate_id = gate_response.json()["id"]

        promotion_data = {
            "strategy_id": strategy_id,
            "gate_id": gate_id,
            "from_state": "candidate",
            "to_state": "shadow",
            "passed": True,
            "evaluation_results": {"sharpe": 1.5, "passed_criteria": ["min_sharpe", "min_win_rate"]},
            "promoted_by": "optimizer",
        }
        response = client.post("/promotions", json=promotion_data)
        assert response.status_code == 201
        data = response.json()
        assert data["strategy_id"] == strategy_id
        assert data["passed"] is True

    def test_list_promotions_by_strategy(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_gate_data: dict[str, Any],
    ) -> None:
        """Test listing promotions filtered by strategy."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        gate_response = client.post("/gates", json=sample_gate_data)
        gate_id = gate_response.json()["id"]

        # Create multiple promotions
        promotions = [
            ("candidate", "shadow", True),
            ("shadow", "paper", True),
        ]
        for from_state, to_state, passed in promotions:
            promotion_data = {
                "strategy_id": strategy_id,
                "gate_id": gate_id,
                "from_state": from_state,
                "to_state": to_state,
                "passed": passed,
            }
            client.post("/promotions", json=promotion_data)

        response = client.get(f"/promotions?strategy_id={strategy_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
