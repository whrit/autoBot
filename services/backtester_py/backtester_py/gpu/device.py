"""
GPU device detection and management module.

Provides utilities for detecting CUDA availability, managing device selection,
and transferring data between CPU and GPU with graceful fallback to CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    import cupy as cp


# Type definitions
DeviceType = Literal["cuda", "cpu"]
ArrayType = TypeVar("ArrayType", bound=np.ndarray)


# Module-level cache for CuPy import
_cupy_module: ModuleType | None = None
_cupy_available: bool | None = None


def _detect_cuda() -> bool:
    """
    Detect if CUDA is available via CuPy.

    Returns:
        True if CUDA is available, False otherwise.
    """
    global _cupy_available, _cupy_module

    if _cupy_available is not None:
        return _cupy_available

    try:
        import cupy as cp_import

        # Try to access device to verify CUDA works
        _ = cp_import.cuda.Device(0).compute_capability
        _cupy_module = cp_import
        _cupy_available = True
    except Exception:
        _cupy_available = False

    return _cupy_available


@lru_cache(maxsize=1)
def get_device() -> DeviceType:
    """
    Detect available device, preferring CUDA if available.

    Returns:
        'cuda' if CUDA is available, 'cpu' otherwise.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if _detect_cuda():
        return "cuda"
    return "cpu"


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    return _detect_cuda()


def _get_cupy() -> ModuleType:
    """
    Get CuPy module, raising ImportError if not available.

    Returns:
        CuPy module.

    Raises:
        ImportError: If CuPy is not available.
    """
    if not _detect_cuda():
        raise ImportError("CuPy is not available. GPU operations require CuPy.")
    assert _cupy_module is not None
    return _cupy_module


def to_gpu(data: np.ndarray) -> cp.ndarray | np.ndarray:
    """
    Move numpy array to GPU if available.

    Args:
        data: NumPy array to move to GPU.

    Returns:
        CuPy array if CUDA is available, otherwise the original NumPy array.

    Example:
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> gpu_data = to_gpu(data)
    """
    if _detect_cuda():
        cp = _get_cupy()
        return cp.asarray(data)
    return data


def to_cpu(data: cp.ndarray | np.ndarray) -> np.ndarray:
    """
    Move array from GPU to CPU.

    Args:
        data: CuPy or NumPy array.

    Returns:
        NumPy array on CPU.

    Example:
        >>> gpu_data = to_gpu(np.array([1.0, 2.0, 3.0]))
        >>> cpu_data = to_cpu(gpu_data)
        >>> isinstance(cpu_data, np.ndarray)
        True
    """
    if isinstance(data, np.ndarray):
        return data

    if _detect_cuda():
        cp = _get_cupy()
        if isinstance(data, cp.ndarray):
            result: np.ndarray = cp.asnumpy(data)
            return result

    # Fallback: try to convert using array protocol
    return np.asarray(data)


def get_array_module(data: cp.ndarray | np.ndarray) -> ModuleType:
    """
    Get the array module (numpy or cupy) for the given array.

    This enables writing code that works with both NumPy and CuPy arrays.

    Args:
        data: NumPy or CuPy array.

    Returns:
        numpy module if data is NumPy array, cupy module if CuPy array.

    Example:
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> xp = get_array_module(data)
        >>> result = xp.sum(data)  # Works with both numpy and cupy
    """
    if isinstance(data, np.ndarray):
        return np

    if _detect_cuda():
        cp = _get_cupy()
        if isinstance(data, cp.ndarray):
            return cp

        # Use CuPy's get_array_module if available
        return cp.get_array_module(data)  # type: ignore[no-any-return]

    return np


class DeviceContext:
    """
    Context manager for device-specific operations.

    Provides a convenient way to perform operations on a specific device
    with automatic data transfer.

    Attributes:
        device: The device type ('cuda' or 'cpu').

    Example:
        >>> with DeviceContext('cuda') as ctx:
        ...     gpu_data = ctx.to_device(np.array([1.0, 2.0, 3.0]))
        ...     result = gpu_data.sum()
    """

    def __init__(self, device: DeviceType | None = None) -> None:
        """
        Initialize DeviceContext.

        Args:
            device: Device type to use. If None, auto-detect.
        """
        if device is None:
            self._device = get_device()
        else:
            self._device = device

    @property
    def device(self) -> DeviceType:
        """Get the device type."""
        return self._device

    def __enter__(self) -> DeviceContext:
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context."""
        pass

    def to_device(self, data: np.ndarray) -> cp.ndarray | np.ndarray:
        """
        Move data to the context's device.

        Args:
            data: NumPy array to move.

        Returns:
            Array on the context's device.
        """
        if self._device == "cuda":
            return to_gpu(data)
        return data

    def to_numpy(self, data: cp.ndarray | np.ndarray) -> np.ndarray:
        """
        Move data back to CPU as numpy array.

        Args:
            data: Array to convert.

        Returns:
            NumPy array on CPU.
        """
        return to_cpu(data)

    def get_module(self) -> ModuleType:
        """
        Get the array module for this device.

        Returns:
            numpy for CPU, cupy for CUDA.
        """
        if self._device == "cuda" and _detect_cuda():
            return _get_cupy()
        return np


@contextmanager
def device_context(device: DeviceType | None = None) -> Generator[DeviceContext, None, None]:
    """
    Context manager factory for device-specific operations.

    Args:
        device: Device type to use. If None, auto-detect.

    Yields:
        DeviceContext instance.

    Example:
        >>> with device_context('cuda') as ctx:
        ...     data = ctx.to_device(np.array([1.0, 2.0, 3.0]))
    """
    ctx = DeviceContext(device)
    try:
        yield ctx
    finally:
        pass


def get_device_info() -> dict[str, Any]:
    """
    Get information about the current device configuration.

    Returns:
        Dictionary with device information including:
        - device: Current device type
        - cuda_available: Whether CUDA is available
        - cupy_version: CuPy version if available
        - cuda_version: CUDA runtime version if available
        - gpu_name: GPU device name if available
        - gpu_memory: Total GPU memory if available

    Example:
        >>> info = get_device_info()
        >>> print(f"Device: {info['device']}")
    """
    info: dict[str, Any] = {
        "device": get_device(),
        "cuda_available": is_cuda_available(),
    }

    if is_cuda_available():
        try:
            cp = _get_cupy()
            info["cupy_version"] = cp.__version__

            # Get CUDA runtime version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            info["cuda_version"] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

            # Get device info
            device = cp.cuda.Device(0)
            info["gpu_name"] = device.name
            info["gpu_memory_total"] = device.mem_info[1]
            info["gpu_memory_free"] = device.mem_info[0]

        except Exception as e:
            info["error"] = str(e)

    return info


def synchronize() -> None:
    """
    Synchronize GPU device if CUDA is available.

    This ensures all GPU operations are completed before continuing.
    No-op if CUDA is not available.
    """
    if is_cuda_available():
        cp = _get_cupy()
        cp.cuda.Device(0).synchronize()


def clear_memory() -> None:
    """
    Clear GPU memory pool if CUDA is available.

    This can help free GPU memory when it's running low.
    No-op if CUDA is not available.
    """
    if is_cuda_available():
        cp = _get_cupy()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()


__all__ = [
    "DeviceType",
    "DeviceContext",
    "device_context",
    "get_device",
    "is_cuda_available",
    "to_gpu",
    "to_cpu",
    "get_array_module",
    "get_device_info",
    "synchronize",
    "clear_memory",
]
