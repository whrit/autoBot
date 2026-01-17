"""
GPU acceleration module for backtesting operations.

This module provides GPU-accelerated operations for technical indicator
calculations and array operations, with automatic CPU fallback when
CUDA is not available.

Key Features:
- Automatic GPU detection and fallback to CPU
- CuPy-based array operations
- PyTorch tensor utilities
- GPU-accelerated technical indicators (SMA, EMA, RSI, MACD, etc.)

Example usage:
    >>> from backtester_py.gpu import get_device, is_cuda_available
    >>> from backtester_py.gpu import GpuArray, sma, ema, rsi
    >>>
    >>> # Check device availability
    >>> print(f"Using device: {get_device()}")
    >>> print(f"CUDA available: {is_cuda_available()}")
    >>>
    >>> # Use GPU arrays
    >>> import numpy as np
    >>> data = np.random.randn(1000)
    >>> arr = GpuArray(data)
    >>> result = arr.mean()
    >>>
    >>> # Calculate indicators
    >>> prices = np.random.randn(1000).cumsum() + 100
    >>> sma_values = sma(prices, window=20)
    >>> ema_values = ema(prices, window=20)
    >>> rsi_values = rsi(prices, window=14)
"""

from backtester_py.gpu.arrays import (
    GpuArray,
    arange,
    concatenate,
    empty,
    full,
    linspace,
    ones,
    stack,
    zeros,
)
from backtester_py.gpu.device import (
    DeviceContext,
    DeviceType,
    clear_memory,
    device_context,
    get_array_module,
    get_device,
    get_device_info,
    is_cuda_available,
    synchronize,
    to_cpu,
    to_gpu,
)
from backtester_py.gpu.indicators import (
    atr,
    bollinger_bands,
    compute_all_indicators,
    ema,
    macd,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rsi,
    sma,
)
from backtester_py.gpu.tensors import (
    TensorContext,
    TorchDeviceType,
    batch_numpy_to_tensors,
    batch_tensors_to_numpy,
    clear_torch_cache,
    create_tensor,
    get_torch_device,
    get_torch_device_info,
    is_torch_available,
    is_torch_cuda_available,
    numpy_to_tensor,
    synchronize_torch,
    tensor_to_numpy,
)

__all__ = [
    # Device management
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
    # Arrays
    "GpuArray",
    "zeros",
    "ones",
    "empty",
    "arange",
    "linspace",
    "full",
    "concatenate",
    "stack",
    # Tensors (PyTorch)
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
    # Indicators
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "rolling_std",
    "rolling_mean",
    "rolling_max",
    "rolling_min",
    "compute_all_indicators",
]

__version__ = "0.1.0"
