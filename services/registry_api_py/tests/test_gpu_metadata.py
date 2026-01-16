"""GPU training metadata tests for BacktestRun - TDD approach."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestBacktestRunGPUMetadata:
    """Test suite for GPU training metadata in BacktestRun - TDD tests."""

    def test_create_backtest_run_with_gpu_metadata(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a backtest run with full GPU training metadata."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cuda",
            "gpu_backend": "xgboost_gpu",
            "cuda_version": "12.1",
            "driver_version": "535.104.05",
            "seed": 42,
            "determinism_flags": "CUDA_LAUNCH_BLOCKING=1,CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "training_time_sec": 125.5,
        }
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["strategy_id"] == strategy_id
        assert data["training_device"] == "cuda"
        assert data["gpu_backend"] == "xgboost_gpu"
        assert data["cuda_version"] == "12.1"
        assert data["driver_version"] == "535.104.05"
        assert data["seed"] == 42
        assert data["determinism_flags"] == "CUDA_LAUNCH_BLOCKING=1,CUBLAS_WORKSPACE_CONFIG=:4096:8"
        assert data["training_time_sec"] == 125.5

    def test_create_backtest_run_with_cpu_training(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a backtest run with CPU training device."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cpu",
            "gpu_backend": "cpu",
            "seed": 12345,
            "training_time_sec": 450.0,
        }
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["training_device"] == "cpu"
        assert data["gpu_backend"] == "cpu"
        assert data["cuda_version"] is None
        assert data["driver_version"] is None
        assert data["seed"] == 12345
        assert data["training_time_sec"] == 450.0

    def test_create_backtest_run_with_pytorch_cuda(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a backtest run with PyTorch CUDA backend."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cuda",
            "gpu_backend": "pytorch_cuda",
            "cuda_version": "11.8",
            "driver_version": "525.60.13",
            "seed": 999,
            "determinism_flags": "torch.use_deterministic_algorithms(True)",
            "training_time_sec": 89.25,
        }
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["gpu_backend"] == "pytorch_cuda"

    def test_filter_backtest_runs_by_training_device(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test filtering backtest runs by training device (cuda vs cpu)."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        cuda_run = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cuda",
            "gpu_backend": "xgboost_gpu",
        }
        cpu_run = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cpu",
            "gpu_backend": "cpu",
        }

        client.post("/backtest-runs", json=cuda_run)
        client.post("/backtest-runs", json=cpu_run)

        response = client.get(f"/backtest-runs?strategy_id={strategy_id}&training_device=cuda")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["training_device"] == "cuda"

        response = client.get(f"/backtest-runs?strategy_id={strategy_id}&training_device=cpu")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["training_device"] == "cpu"

    def test_create_backtest_run_without_gpu_metadata(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test creating a backtest run without any GPU metadata (backwards compatibility)."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {**sample_backtest_run_data, "strategy_id": strategy_id}
        response = client.post("/backtest-runs", json=run_data)
        assert response.status_code == 201
        data = response.json()
        assert data["training_device"] is None
        assert data["gpu_backend"] is None
        assert data["cuda_version"] is None
        assert data["driver_version"] is None
        assert data["seed"] is None
        assert data["determinism_flags"] is None
        assert data["training_time_sec"] is None

    def test_get_backtest_run_includes_gpu_metadata(
        self,
        client: TestClient,
        sample_strategy_data: dict[str, Any],
        sample_backtest_run_data: dict[str, Any],
    ) -> None:
        """Test that retrieving a backtest run includes GPU metadata fields."""
        strategy_response = client.post("/strategies", json=sample_strategy_data)
        strategy_id = strategy_response.json()["id"]

        run_data = {
            **sample_backtest_run_data,
            "strategy_id": strategy_id,
            "training_device": "cuda",
            "gpu_backend": "xgboost_gpu",
            "seed": 42,
            "training_time_sec": 100.0,
        }
        create_response = client.post("/backtest-runs", json=run_data)
        run_id = create_response.json()["id"]

        response = client.get(f"/backtest-runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["training_device"] == "cuda"
        assert data["gpu_backend"] == "xgboost_gpu"
        assert data["seed"] == 42
        assert data["training_time_sec"] == 100.0
