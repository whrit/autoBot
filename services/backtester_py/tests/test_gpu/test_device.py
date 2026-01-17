"""
Tests for GPU device detection and management.

TDD tests written BEFORE implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from backtester_py.gpu.device import DeviceType


class TestGetDevice:
    """Tests for get_device() function."""

    def test_get_device_returns_valid_type(self) -> None:
        """get_device() should return either 'cuda' or 'cpu'."""
        from backtester_py.gpu.device import get_device

        device = get_device()
        assert device in ("cuda", "cpu")

    def test_get_device_returns_cpu_when_cupy_unavailable(self) -> None:
        """get_device() should return 'cpu' when CuPy is not available."""
        from backtester_py.gpu import device as device_module

        with patch.dict("sys.modules", {"cupy": None}):
            # Force reimport to pick up mocked module
            with patch.object(device_module, "_detect_cuda", return_value=False):
                result = device_module.get_device()
                assert result == "cpu"

    def test_get_device_caches_result(self) -> None:
        """get_device() should cache its result for performance."""
        from backtester_py.gpu.device import get_device

        # Call twice and ensure same result (cached)
        result1 = get_device()
        result2 = get_device()
        assert result1 == result2


class TestIsCudaAvailable:
    """Tests for is_cuda_available() function."""

    def test_is_cuda_available_returns_bool(self) -> None:
        """is_cuda_available() should return a boolean."""
        from backtester_py.gpu.device import is_cuda_available

        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_is_cuda_available_consistent_with_get_device(self) -> None:
        """is_cuda_available() should be consistent with get_device()."""
        from backtester_py.gpu.device import get_device, is_cuda_available

        cuda_available = is_cuda_available()
        device = get_device()

        if cuda_available:
            assert device == "cuda"
        else:
            assert device == "cpu"


class TestToGpu:
    """Tests for to_gpu() function."""

    def test_to_gpu_accepts_numpy_array(self) -> None:
        """to_gpu() should accept numpy arrays."""
        from backtester_py.gpu.device import to_gpu

        data = np.array([1.0, 2.0, 3.0])
        result = to_gpu(data)
        assert result is not None

    def test_to_gpu_preserves_data(self) -> None:
        """to_gpu() should preserve array data."""
        from backtester_py.gpu.device import to_cpu, to_gpu

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gpu_data = to_gpu(data)
        cpu_data = to_cpu(gpu_data)

        np.testing.assert_array_equal(cpu_data, data)

    def test_to_gpu_preserves_dtype(self) -> None:
        """to_gpu() should preserve array dtype."""
        from backtester_py.gpu.device import to_cpu, to_gpu

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu_data = to_gpu(data)
        cpu_data = to_cpu(gpu_data)

        assert cpu_data.dtype == data.dtype

    def test_to_gpu_preserves_shape(self) -> None:
        """to_gpu() should preserve array shape."""
        from backtester_py.gpu.device import to_cpu, to_gpu

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        gpu_data = to_gpu(data)
        cpu_data = to_cpu(gpu_data)

        assert cpu_data.shape == data.shape

    def test_to_gpu_with_2d_array(self) -> None:
        """to_gpu() should handle 2D arrays."""
        from backtester_py.gpu.device import to_cpu, to_gpu

        data = np.random.randn(100, 50).astype(np.float64)
        gpu_data = to_gpu(data)
        cpu_data = to_cpu(gpu_data)

        np.testing.assert_array_almost_equal(cpu_data, data)


class TestToCpu:
    """Tests for to_cpu() function."""

    def test_to_cpu_with_numpy_array(self) -> None:
        """to_cpu() should handle numpy arrays (no-op)."""
        from backtester_py.gpu.device import to_cpu

        data = np.array([1.0, 2.0, 3.0])
        result = to_cpu(data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_to_cpu_returns_numpy_array(self) -> None:
        """to_cpu() should always return a numpy array."""
        from backtester_py.gpu.device import to_cpu, to_gpu

        data = np.array([1.0, 2.0, 3.0])
        gpu_data = to_gpu(data)
        cpu_data = to_cpu(gpu_data)

        assert isinstance(cpu_data, np.ndarray)


class TestGetArrayModule:
    """Tests for get_array_module() function."""

    def test_get_array_module_with_numpy(self) -> None:
        """get_array_module() should return numpy for numpy arrays."""
        from backtester_py.gpu.device import get_array_module

        data = np.array([1.0, 2.0, 3.0])
        xp = get_array_module(data)

        assert xp is np

    def test_get_array_module_operations_work(self) -> None:
        """get_array_module() should return module with working operations."""
        from backtester_py.gpu.device import get_array_module, to_gpu

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gpu_data = to_gpu(data)
        xp = get_array_module(gpu_data)

        # Test common operations
        result_sum = xp.sum(gpu_data)
        result_mean = xp.mean(gpu_data)

        assert float(result_sum) == pytest.approx(15.0)
        assert float(result_mean) == pytest.approx(3.0)


class TestDeviceContext:
    """Tests for DeviceContext context manager."""

    def test_device_context_enters_and_exits(self) -> None:
        """DeviceContext should enter and exit cleanly."""
        from backtester_py.gpu.device import DeviceContext

        with DeviceContext("cpu"):
            pass  # Should not raise

    def test_device_context_with_cpu(self) -> None:
        """DeviceContext should work with CPU device."""
        from backtester_py.gpu.device import DeviceContext

        with DeviceContext("cpu") as ctx:
            assert ctx.device == "cpu"

    def test_device_context_operations_work(self) -> None:
        """Operations inside DeviceContext should work."""
        from backtester_py.gpu.device import DeviceContext

        with DeviceContext("cpu") as ctx:
            data = np.array([1.0, 2.0, 3.0])
            result = ctx.to_device(data)
            assert result is not None


class TestDeviceInfo:
    """Tests for device information functions."""

    def test_get_device_info_returns_dict(self) -> None:
        """get_device_info() should return a dictionary."""
        from backtester_py.gpu.device import get_device_info

        info = get_device_info()
        assert isinstance(info, dict)

    def test_get_device_info_has_device_key(self) -> None:
        """get_device_info() should include 'device' key."""
        from backtester_py.gpu.device import get_device_info

        info = get_device_info()
        assert "device" in info

    def test_get_device_info_has_cuda_available_key(self) -> None:
        """get_device_info() should include 'cuda_available' key."""
        from backtester_py.gpu.device import get_device_info

        info = get_device_info()
        assert "cuda_available" in info
        assert isinstance(info["cuda_available"], bool)
