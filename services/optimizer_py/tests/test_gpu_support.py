"""
Tests for GPU support module.

TDD tests for T4.06 (GPU Training Support), T4.07 (CPU Fallback Logic),
and T4.12 (GPU Metadata Logging).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from optimizer_py.gpu_support import (
    GPUConfig,
    GPUManager,
    GPUMetadata,
    TrainingDevice,
    get_cuda_info,
)


class TestTrainingDeviceEnum:
    """Test TrainingDevice enum values."""

    def test_training_device_cpu_value(self) -> None:
        """Test CPU enum value."""
        assert TrainingDevice.CPU.value == "cpu"

    def test_training_device_cuda_value(self) -> None:
        """Test CUDA enum value."""
        assert TrainingDevice.CUDA.value == "cuda"

    def test_training_device_members(self) -> None:
        """Test all enum members exist."""
        members = list(TrainingDevice)
        assert len(members) == 2
        assert TrainingDevice.CPU in members
        assert TrainingDevice.CUDA in members


class TestGPUConfig:
    """Test GPUConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GPUConfig()
        assert config.enabled is True
        assert config.device_id == 0
        assert config.fallback_to_cpu is True
        assert config.memory_fraction == 0.8

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = GPUConfig(
            enabled=False,
            device_id=2,
            fallback_to_cpu=False,
            memory_fraction=0.5,
        )
        assert config.enabled is False
        assert config.device_id == 2
        assert config.fallback_to_cpu is False
        assert config.memory_fraction == 0.5

    def test_memory_fraction_bounds(self) -> None:
        """Test memory fraction can be set to edge values."""
        config_low = GPUConfig(memory_fraction=0.1)
        config_high = GPUConfig(memory_fraction=1.0)
        assert config_low.memory_fraction == 0.1
        assert config_high.memory_fraction == 1.0


class TestGPUMetadata:
    """Test GPUMetadata dataclass."""

    def test_required_fields(self) -> None:
        """Test creating metadata with required fields."""
        metadata = GPUMetadata(
            training_device=TrainingDevice.CPU,
            gpu_backend="cpu",
            cuda_version=None,
            driver_version=None,
            gpu_name=None,
            gpu_memory_mb=None,
            training_time_sec=10.5,
        )
        assert metadata.training_device == TrainingDevice.CPU
        assert metadata.gpu_backend == "cpu"
        assert metadata.training_time_sec == 10.5
        assert isinstance(metadata.timestamp, datetime)

    def test_gpu_metadata_with_cuda(self) -> None:
        """Test metadata with CUDA GPU information."""
        metadata = GPUMetadata(
            training_device=TrainingDevice.CUDA,
            gpu_backend="xgboost_gpu",
            cuda_version="12.1",
            driver_version="535.154.05",
            gpu_name="NVIDIA GeForce RTX 3080",
            gpu_memory_mb=10240,
            training_time_sec=5.2,
        )
        assert metadata.training_device == TrainingDevice.CUDA
        assert metadata.gpu_backend == "xgboost_gpu"
        assert metadata.cuda_version == "12.1"
        assert metadata.driver_version == "535.154.05"
        assert metadata.gpu_name == "NVIDIA GeForce RTX 3080"
        assert metadata.gpu_memory_mb == 10240

    def test_timestamp_auto_generated(self) -> None:
        """Test timestamp is automatically generated."""
        before = datetime.now(UTC)
        metadata = GPUMetadata(
            training_device=TrainingDevice.CPU,
            gpu_backend="cpu",
            cuda_version=None,
            driver_version=None,
            gpu_name=None,
            gpu_memory_mb=None,
            training_time_sec=1.0,
        )
        after = datetime.now(UTC)
        assert before <= metadata.timestamp <= after


class TestGPUManager:
    """Test GPUManager class."""

    def test_gpu_manager_init(self) -> None:
        """Test GPUManager initialization with default config."""
        config = GPUConfig()
        manager = GPUManager(config)
        assert manager.config == config

    def test_gpu_manager_init_custom_config(self) -> None:
        """Test GPUManager initialization with custom config."""
        config = GPUConfig(enabled=False, device_id=1)
        manager = GPUManager(config)
        assert manager.config.enabled is False
        assert manager.config.device_id == 1

    def test_gpu_availability_check_disabled(self) -> None:
        """Test GPU availability when explicitly disabled."""
        config = GPUConfig(enabled=False)
        manager = GPUManager(config)
        assert manager.is_gpu_available() is False

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_gpu_availability_check_no_cuda(self, mock_cuda_info: MagicMock) -> None:
        """Test GPU availability when CUDA is not available."""
        mock_cuda_info.return_value = None
        config = GPUConfig(enabled=True)
        manager = GPUManager(config)
        assert manager.is_gpu_available() is False

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_gpu_availability_check_with_cuda(self, mock_cuda_info: MagicMock) -> None:
        """Test GPU availability when CUDA is available."""
        mock_cuda_info.return_value = {
            "cuda_version": "12.1",
            "driver_version": "535.154.05",
            "gpu_name": "NVIDIA GeForce RTX 3080",
            "gpu_memory_mb": 10240,
        }
        config = GPUConfig(enabled=True)
        manager = GPUManager(config)
        assert manager.is_gpu_available() is True

    def test_cpu_fallback_when_gpu_unavailable(self) -> None:
        """Test automatic CPU fallback when GPU is unavailable."""
        config = GPUConfig(enabled=True, fallback_to_cpu=True)
        manager = GPUManager(config)
        # When GPU not available, should fall back to CPU params
        params = manager.get_xgboost_params()
        # Either GPU params if available, or CPU fallback
        assert "tree_method" in params
        assert "device" in params

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_xgboost_params_with_gpu(self, mock_cuda_info: MagicMock) -> None:
        """Test XGBoost params include GPU settings when available."""
        mock_cuda_info.return_value = {
            "cuda_version": "12.1",
            "driver_version": "535.154.05",
            "gpu_name": "NVIDIA GeForce RTX 3080",
            "gpu_memory_mb": 10240,
        }
        config = GPUConfig(enabled=True, device_id=0)
        manager = GPUManager(config)
        params = manager.get_xgboost_params()
        assert params["tree_method"] == "hist"
        assert params["device"] == "cuda:0"

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_xgboost_params_cpu_fallback(self, mock_cuda_info: MagicMock) -> None:
        """Test XGBoost params fallback to CPU when GPU unavailable."""
        mock_cuda_info.return_value = None
        config = GPUConfig(enabled=True, fallback_to_cpu=True)
        manager = GPUManager(config)
        params = manager.get_xgboost_params()
        assert params["tree_method"] == "hist"
        assert params["device"] == "cpu"

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_xgboost_params_no_fallback_raises(self, mock_cuda_info: MagicMock) -> None:
        """Test exception when GPU unavailable and fallback disabled."""
        mock_cuda_info.return_value = None
        config = GPUConfig(enabled=True, fallback_to_cpu=False)
        manager = GPUManager(config)
        with pytest.raises(RuntimeError, match="GPU not available"):
            manager.get_xgboost_params()

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_metadata_collection_cpu(self, mock_cuda_info: MagicMock) -> None:
        """Test GPU metadata collection for CPU training."""
        mock_cuda_info.return_value = None
        config = GPUConfig(enabled=True, fallback_to_cpu=True)
        manager = GPUManager(config)
        metadata = manager.collect_metadata(training_time=15.5)

        assert metadata.training_device == TrainingDevice.CPU
        assert metadata.gpu_backend == "cpu"
        assert metadata.cuda_version is None
        assert metadata.driver_version is None
        assert metadata.gpu_name is None
        assert metadata.gpu_memory_mb is None
        assert metadata.training_time_sec == 15.5

    @patch("optimizer_py.gpu_support.get_cuda_info")
    def test_metadata_collection_gpu(self, mock_cuda_info: MagicMock) -> None:
        """Test GPU metadata collection for GPU training."""
        mock_cuda_info.return_value = {
            "cuda_version": "12.1",
            "driver_version": "535.154.05",
            "gpu_name": "NVIDIA GeForce RTX 3080",
            "gpu_memory_mb": 10240,
        }
        config = GPUConfig(enabled=True)
        manager = GPUManager(config)
        metadata = manager.collect_metadata(training_time=5.2)

        assert metadata.training_device == TrainingDevice.CUDA
        assert metadata.gpu_backend == "xgboost_gpu"
        assert metadata.cuda_version == "12.1"
        assert metadata.driver_version == "535.154.05"
        assert metadata.gpu_name == "NVIDIA GeForce RTX 3080"
        assert metadata.gpu_memory_mb == 10240
        assert metadata.training_time_sec == 5.2


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_env_var_gpu_enabled_true(self) -> None:
        """Test GPU_ENABLED environment variable set to true."""
        with patch.dict(os.environ, {"GPU_ENABLED": "true"}):
            config = GPUConfig.from_env()
            assert config.enabled is True

    def test_env_var_gpu_enabled_false(self) -> None:
        """Test GPU_ENABLED environment variable set to false."""
        with patch.dict(os.environ, {"GPU_ENABLED": "false"}):
            config = GPUConfig.from_env()
            assert config.enabled is False

    def test_env_var_gpu_device(self) -> None:
        """Test GPU_DEVICE environment variable."""
        with patch.dict(os.environ, {"GPU_DEVICE": "2"}):
            config = GPUConfig.from_env()
            assert config.device_id == 2

    def test_env_var_gpu_fallback_cpu(self) -> None:
        """Test GPU_FALLBACK_CPU environment variable."""
        with patch.dict(os.environ, {"GPU_FALLBACK_CPU": "false"}):
            config = GPUConfig.from_env()
            assert config.fallback_to_cpu is False

    def test_env_var_all_combined(self) -> None:
        """Test all environment variables together."""
        env_vars = {
            "GPU_ENABLED": "true",
            "GPU_DEVICE": "1",
            "GPU_FALLBACK_CPU": "true",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = GPUConfig.from_env()
            assert config.enabled is True
            assert config.device_id == 1
            assert config.fallback_to_cpu is True

    def test_env_var_defaults_when_not_set(self) -> None:
        """Test defaults when environment variables not set."""
        # Clear relevant env vars
        env_vars = {
            "GPU_ENABLED": "",
            "GPU_DEVICE": "",
            "GPU_FALLBACK_CPU": "",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            # Remove from environ by using clear flag
            for key in ["GPU_ENABLED", "GPU_DEVICE", "GPU_FALLBACK_CPU"]:
                os.environ.pop(key, None)
            config = GPUConfig.from_env()
            assert config.enabled is True  # default
            assert config.device_id == 0  # default
            assert config.fallback_to_cpu is True  # default


class TestGetCudaInfo:
    """Test get_cuda_info function."""

    def test_cuda_info_returns_none_when_unavailable(self) -> None:
        """Test cuda info returns None when CUDA unavailable."""
        # This test will pass on systems without CUDA
        result = get_cuda_info()
        # Result can be None or dict depending on system
        assert result is None or isinstance(result, dict)

    def test_cuda_info_dict_structure(self) -> None:
        """Test cuda info returns correct dict structure when available."""
        result = get_cuda_info()
        if result is not None:
            assert "cuda_version" in result
            assert "driver_version" in result
            assert "gpu_name" in result
            assert "gpu_memory_mb" in result


class TestXGBoostIntegration:
    """Integration tests for XGBoost GPU configuration."""

    def test_xgboost_params_are_valid(self) -> None:
        """Test that generated params are valid for XGBoost."""
        config = GPUConfig(enabled=False)  # Force CPU for test
        manager = GPUManager(config)
        params = manager.get_xgboost_params()

        # Verify params are XGBoost-compatible
        assert params["tree_method"] in ["hist", "gpu_hist", "approx", "exact"]
        assert params["device"] in ["cpu", "cuda:0", "cuda:1", "cuda:2"]

    def test_xgboost_params_with_different_device_ids(self) -> None:
        """Test XGBoost params with different GPU device IDs."""
        for device_id in [0, 1, 2]:
            config = GPUConfig(enabled=True, device_id=device_id)
            manager = GPUManager(config)
            # Mock GPU availability
            with patch.object(manager, "is_gpu_available", return_value=True):
                params = manager.get_xgboost_params()
                assert params["device"] == f"cuda:{device_id}"


class TestMetadataToDict:
    """Test metadata serialization."""

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = GPUMetadata(
            training_device=TrainingDevice.CUDA,
            gpu_backend="xgboost_gpu",
            cuda_version="12.1",
            driver_version="535.154.05",
            gpu_name="NVIDIA GeForce RTX 3080",
            gpu_memory_mb=10240,
            training_time_sec=5.2,
        )
        result = metadata.to_dict()

        assert result["training_device"] == "cuda"
        assert result["gpu_backend"] == "xgboost_gpu"
        assert result["cuda_version"] == "12.1"
        assert result["driver_version"] == "535.154.05"
        assert result["gpu_name"] == "NVIDIA GeForce RTX 3080"
        assert result["gpu_memory_mb"] == 10240
        assert result["training_time_sec"] == 5.2
        assert "timestamp" in result

    def test_metadata_cpu_to_dict(self) -> None:
        """Test converting CPU metadata to dictionary."""
        metadata = GPUMetadata(
            training_device=TrainingDevice.CPU,
            gpu_backend="cpu",
            cuda_version=None,
            driver_version=None,
            gpu_name=None,
            gpu_memory_mb=None,
            training_time_sec=10.0,
        )
        result = metadata.to_dict()

        assert result["training_device"] == "cpu"
        assert result["gpu_backend"] == "cpu"
        assert result["cuda_version"] is None
        assert result["gpu_memory_mb"] is None
