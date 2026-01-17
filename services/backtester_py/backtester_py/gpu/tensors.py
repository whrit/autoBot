"""
PyTorch tensor utilities for GPU-accelerated operations.

Provides utilities for working with PyTorch tensors, including
device management, conversion utilities, and common operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch


# Module-level cache for PyTorch import
_torch_module: Any | None = None
_torch_available: bool | None = None
_torch_cuda_available: bool | None = None


def _detect_torch() -> bool:
    """
    Detect if PyTorch is available.

    Returns:
        True if PyTorch is available, False otherwise.
    """
    global _torch_available, _torch_module

    if _torch_available is not None:
        return _torch_available

    try:
        import torch as torch_import

        _torch_module = torch_import
        _torch_available = True
    except ImportError:
        _torch_available = False

    return _torch_available


def _detect_torch_cuda() -> bool:
    """
    Detect if PyTorch CUDA is available.

    Returns:
        True if PyTorch CUDA is available, False otherwise.
    """
    global _torch_cuda_available

    if _torch_cuda_available is not None:
        return _torch_cuda_available

    if not _detect_torch():
        _torch_cuda_available = False
        return False

    assert _torch_module is not None
    _torch_cuda_available = _torch_module.cuda.is_available()
    return _torch_cuda_available


def _get_torch() -> Any:
    """
    Get PyTorch module, raising ImportError if not available.

    Returns:
        PyTorch module.

    Raises:
        ImportError: If PyTorch is not available.
    """
    if not _detect_torch():
        raise ImportError("PyTorch is not available.")
    return _torch_module


TorchDeviceType = Literal["cuda", "cpu"]


def get_torch_device() -> TorchDeviceType:
    """
    Get the best available PyTorch device.

    Returns:
        'cuda' if CUDA is available, 'cpu' otherwise.
    """
    if _detect_torch_cuda():
        return "cuda"
    return "cpu"


def is_torch_available() -> bool:
    """
    Check if PyTorch is available.

    Returns:
        True if PyTorch is available, False otherwise.
    """
    return _detect_torch()


def is_torch_cuda_available() -> bool:
    """
    Check if PyTorch CUDA is available.

    Returns:
        True if PyTorch CUDA is available, False otherwise.
    """
    return _detect_torch_cuda()


def numpy_to_tensor(
    data: np.ndarray,
    device: TorchDeviceType | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        data: NumPy array to convert.
        device: Target device ('cuda' or 'cpu'). Auto-detect if None.
        dtype: Target dtype. Uses appropriate default if None.

    Returns:
        PyTorch tensor on the specified device.

    Raises:
        ImportError: If PyTorch is not available.

    Example:
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> tensor = numpy_to_tensor(data, device='cuda')
    """
    torch = _get_torch()

    if device is None:
        device = get_torch_device()

    tensor = torch.from_numpy(data.copy())

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    return tensor.to(device=device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        NumPy array on CPU.

    Example:
        >>> tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> array = tensor_to_numpy(tensor)
    """
    if not _detect_torch():
        raise ImportError("PyTorch is not available.")

    # Move to CPU if necessary and convert
    if tensor.is_cuda:
        result: np.ndarray = tensor.detach().cpu().numpy()
        return result
    cpu_result: np.ndarray = tensor.detach().numpy()
    return cpu_result


def create_tensor(
    shape: tuple[int, ...],
    fill_value: float | None = None,
    device: TorchDeviceType | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Create a PyTorch tensor with optional fill value.

    Args:
        shape: Tensor shape.
        fill_value: Value to fill tensor with. If None, creates zeros.
        device: Target device. Auto-detect if None.
        dtype: Target dtype.

    Returns:
        PyTorch tensor on the specified device.

    Example:
        >>> tensor = create_tensor((3, 4), fill_value=1.0, device='cuda')
    """
    torch = _get_torch()

    if device is None:
        device = get_torch_device()

    if fill_value is None:
        tensor = torch.zeros(shape, device=device)
    else:
        tensor = torch.full(shape, fill_value, device=device)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    return tensor


def synchronize_torch() -> None:
    """
    Synchronize CUDA device if using PyTorch CUDA.

    Ensures all CUDA operations are completed before continuing.
    No-op if PyTorch or CUDA is not available.
    """
    if _detect_torch_cuda():
        torch = _get_torch()
        torch.cuda.synchronize()


def clear_torch_cache() -> None:
    """
    Clear PyTorch CUDA cache.

    Helps free GPU memory when it's running low.
    No-op if PyTorch or CUDA is not available.
    """
    if _detect_torch_cuda():
        torch = _get_torch()
        torch.cuda.empty_cache()


def get_torch_device_info() -> dict[str, Any]:
    """
    Get information about PyTorch and CUDA configuration.

    Returns:
        Dictionary with device information.
    """
    info: dict[str, Any] = {
        "torch_available": is_torch_available(),
        "cuda_available": is_torch_cuda_available(),
    }

    if is_torch_available():
        torch = _get_torch()
        info["torch_version"] = torch.__version__

        if is_torch_cuda_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["device_count"] = torch.cuda.device_count()

            if info["device_count"] > 0:
                info["current_device"] = torch.cuda.current_device()
                info["device_name"] = torch.cuda.get_device_name(0)

                # Memory info
                info["memory_allocated"] = torch.cuda.memory_allocated(0)
                info["memory_reserved"] = torch.cuda.memory_reserved(0)

    return info


class TensorContext:
    """
    Context manager for PyTorch tensor operations.

    Provides convenient device management for tensor operations.

    Example:
        >>> with TensorContext('cuda') as ctx:
        ...     tensor = ctx.to_tensor(np.array([1.0, 2.0, 3.0]))
        ...     result = tensor.sum()
    """

    def __init__(self, device: TorchDeviceType | None = None) -> None:
        """
        Initialize TensorContext.

        Args:
            device: Target device. Auto-detect if None.
        """
        self._device = device or get_torch_device()

    @property
    def device(self) -> TorchDeviceType:
        """Get the device type."""
        return self._device

    def __enter__(self) -> TensorContext:
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

    def to_tensor(
        self, data: np.ndarray, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """
        Convert numpy array to tensor on context's device.

        Args:
            data: NumPy array to convert.
            dtype: Target dtype.

        Returns:
            PyTorch tensor on context's device.
        """
        return numpy_to_tensor(data, device=self._device, dtype=dtype)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array.

        Args:
            tensor: PyTorch tensor.

        Returns:
            NumPy array on CPU.
        """
        return tensor_to_numpy(tensor)

    def zeros(
        self, shape: tuple[int, ...], dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Create zeros tensor on context's device."""
        return create_tensor(shape, fill_value=None, device=self._device, dtype=dtype)

    def ones(
        self, shape: tuple[int, ...], dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Create ones tensor on context's device."""
        return create_tensor(shape, fill_value=1.0, device=self._device, dtype=dtype)


def batch_numpy_to_tensors(
    arrays: list[np.ndarray],
    device: TorchDeviceType | None = None,
) -> list[torch.Tensor]:
    """
    Convert multiple NumPy arrays to PyTorch tensors efficiently.

    Args:
        arrays: List of NumPy arrays to convert.
        device: Target device. Auto-detect if None.

    Returns:
        List of PyTorch tensors.
    """
    if device is None:
        device = get_torch_device()

    return [numpy_to_tensor(arr, device=device) for arr in arrays]


def batch_tensors_to_numpy(tensors: list[torch.Tensor]) -> list[np.ndarray]:
    """
    Convert multiple PyTorch tensors to NumPy arrays.

    Args:
        tensors: List of PyTorch tensors.

    Returns:
        List of NumPy arrays.
    """
    return [tensor_to_numpy(t) for t in tensors]


__all__ = [
    "TorchDeviceType",
    "TensorContext",
    "get_torch_device",
    "is_torch_available",
    "is_torch_cuda_available",
    "numpy_to_tensor",
    "tensor_to_numpy",
    "create_tensor",
    "synchronize_torch",
    "clear_torch_cache",
    "get_torch_device_info",
    "batch_numpy_to_tensors",
    "batch_tensors_to_numpy",
]
