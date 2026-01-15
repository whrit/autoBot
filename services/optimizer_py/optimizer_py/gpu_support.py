"""
GPU Support Module for XGBoost Training.

This module provides GPU acceleration support for XGBoost with graceful
CPU fallback. Implements T4.06 (GPU Training Support), T4.07 (CPU Fallback Logic),
and T4.12 (GPU Metadata Logging).

Environment Variables:
    GPU_ENABLED: "true" | "false" - Enable/disable GPU training (default: "true")
    GPU_DEVICE: Device ID string (default: "0")
    GPU_FALLBACK_CPU: "true" | "false" - Allow CPU fallback (default: "true")

Example:
    >>> config = GPUConfig.from_env()
    >>> manager = GPUManager(config)
    >>> params = manager.get_xgboost_params()
    >>> # Train with XGBoost using params
    >>> metadata = manager.collect_metadata(training_time=10.5)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class TrainingDevice(Enum):
    """Enumeration of training device types.

    Attributes:
        CPU: Train on CPU.
        CUDA: Train on NVIDIA CUDA GPU.
    """

    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class GPUConfig:
    """Configuration for GPU training.

    Attributes:
        enabled: Whether GPU training is enabled.
        device_id: CUDA device ID to use (0, 1, 2, etc.).
        fallback_to_cpu: Whether to fall back to CPU if GPU unavailable.
        memory_fraction: Fraction of GPU memory to use (0.0-1.0).
    """

    enabled: bool = True
    device_id: int = 0
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.8

    @classmethod
    def from_env(cls) -> GPUConfig:
        """Create GPUConfig from environment variables.

        Environment Variables:
            GPU_ENABLED: "true" | "false" (default: "true")
            GPU_DEVICE: Device ID as string (default: "0")
            GPU_FALLBACK_CPU: "true" | "false" (default: "true")

        Returns:
            GPUConfig instance configured from environment.

        Example:
            >>> os.environ["GPU_ENABLED"] = "false"
            >>> config = GPUConfig.from_env()
            >>> config.enabled
            False
        """
        enabled_str = os.environ.get("GPU_ENABLED", "true").lower()
        enabled = enabled_str in ("true", "1", "yes")

        device_str = os.environ.get("GPU_DEVICE", "0")
        try:
            device_id = int(device_str)
        except ValueError:
            device_id = 0

        fallback_str = os.environ.get("GPU_FALLBACK_CPU", "true").lower()
        fallback_to_cpu = fallback_str in ("true", "1", "yes")

        return cls(
            enabled=enabled,
            device_id=device_id,
            fallback_to_cpu=fallback_to_cpu,
        )


@dataclass
class GPUMetadata:
    """Metadata about GPU training execution.

    Records information about the training device and performance
    for audit and reproducibility purposes.

    Attributes:
        training_device: Device used for training (CPU or CUDA).
        gpu_backend: Backend identifier ("xgboost_gpu" | "cpu").
        cuda_version: CUDA toolkit version if GPU used.
        driver_version: NVIDIA driver version if GPU used.
        gpu_name: GPU model name if GPU used.
        gpu_memory_mb: GPU memory in MB if GPU used.
        training_time_sec: Training duration in seconds.
        timestamp: When metadata was collected.
    """

    training_device: TrainingDevice
    gpu_backend: str
    cuda_version: str | None
    driver_version: str | None
    gpu_name: str | None
    gpu_memory_mb: int | None
    training_time_sec: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "training_device": self.training_device.value,
            "gpu_backend": self.gpu_backend,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "training_time_sec": self.training_time_sec,
            "timestamp": self.timestamp.isoformat(),
        }


class GPUManager:
    """Manage GPU availability and XGBoost configuration.

    Handles GPU detection, XGBoost parameter generation with appropriate
    device settings, and metadata collection.

    Example:
        >>> config = GPUConfig(enabled=True, fallback_to_cpu=True)
        >>> manager = GPUManager(config)
        >>> if manager.is_gpu_available():
        ...     print("Training on GPU")
        >>> params = manager.get_xgboost_params()
        >>> # Use params for XGBoost training
        >>> metadata = manager.collect_metadata(training_time=10.5)
    """

    def __init__(self, config: GPUConfig) -> None:
        """Initialize GPUManager with configuration.

        Args:
            config: GPU configuration settings.
        """
        self.config = config
        self._cuda_info: dict[str, Any] | None = None
        self._checked_cuda = False

    def is_gpu_available(self) -> bool:
        """Check if CUDA GPU is available for training.

        Returns:
            True if GPU is enabled in config and CUDA is available.
        """
        if not self.config.enabled:
            return False

        if not self._checked_cuda:
            self._cuda_info = get_cuda_info()
            self._checked_cuda = True

        return self._cuda_info is not None

    def get_xgboost_params(self) -> dict[str, Any]:
        """Get XGBoost parameters with appropriate device settings.

        Returns GPU-accelerated parameters if GPU is available,
        otherwise returns CPU parameters (if fallback enabled).

        Returns:
            Dictionary of XGBoost parameters including tree_method and device.

        Raises:
            RuntimeError: If GPU unavailable and fallback_to_cpu is False.

        Example:
            >>> manager = GPUManager(GPUConfig())
            >>> params = manager.get_xgboost_params()
            >>> # params might be {"tree_method": "hist", "device": "cuda:0"}
            >>> # or {"tree_method": "hist", "device": "cpu"}
        """
        if self.is_gpu_available():
            return {
                "tree_method": "hist",
                "device": f"cuda:{self.config.device_id}",
            }

        if self.config.fallback_to_cpu:
            return {
                "tree_method": "hist",
                "device": "cpu",
            }

        raise RuntimeError(
            "GPU not available and CPU fallback is disabled. "
            "Set GPU_FALLBACK_CPU=true or ensure CUDA is properly installed."
        )

    def collect_metadata(self, training_time: float) -> GPUMetadata:
        """Collect GPU metadata after training.

        Args:
            training_time: Training duration in seconds.

        Returns:
            GPUMetadata with device information and training time.
        """
        if self.is_gpu_available() and self._cuda_info:
            return GPUMetadata(
                training_device=TrainingDevice.CUDA,
                gpu_backend="xgboost_gpu",
                cuda_version=self._cuda_info.get("cuda_version"),
                driver_version=self._cuda_info.get("driver_version"),
                gpu_name=self._cuda_info.get("gpu_name"),
                gpu_memory_mb=self._cuda_info.get("gpu_memory_mb"),
                training_time_sec=training_time,
            )

        return GPUMetadata(
            training_device=TrainingDevice.CPU,
            gpu_backend="cpu",
            cuda_version=None,
            driver_version=None,
            gpu_name=None,
            gpu_memory_mb=None,
            training_time_sec=training_time,
        )


def get_cuda_info() -> dict[str, Any] | None:
    """Get CUDA version and driver information if available.

    Attempts to detect CUDA availability and gather GPU information
    using nvidia-smi. Returns None if CUDA is not available or
    nvidia-smi cannot be executed.

    Returns:
        Dictionary with cuda_version, driver_version, gpu_name, gpu_memory_mb
        if CUDA is available, None otherwise.

    Example:
        >>> info = get_cuda_info()
        >>> if info:
        ...     print(f"CUDA {info['cuda_version']} on {info['gpu_name']}")
    """
    try:
        # Try to get GPU info from nvidia-smi
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            return None

        # Parse nvidia-smi output
        output = result.stdout.strip()
        if not output:
            return None

        # Parse first GPU info (name, memory, driver)
        parts = [p.strip() for p in output.split("\n")[0].split(",")]
        if len(parts) < 3:
            return None

        gpu_name = parts[0]
        try:
            gpu_memory_mb = int(float(parts[1]))
        except (ValueError, IndexError):
            gpu_memory_mb = None
        driver_version = parts[2]

        # Get CUDA version from nvcc or nvidia-smi
        cuda_version = _get_cuda_version()

        return {
            "cuda_version": cuda_version,
            "driver_version": driver_version,
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
        }

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # nvidia-smi not found or timeout
        return None


def _get_cuda_version() -> str | None:
    """Get CUDA toolkit version.

    Tries nvcc first, then nvidia-smi as fallback.

    Returns:
        CUDA version string or None if unavailable.
    """
    # Try nvcc first
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            # Parse "release X.Y" from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        version_part = parts[1].strip().split(",")[0].strip()
                        return version_part
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            # Parse "CUDA Version: X.Y" from output
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0]
                        return version
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None
