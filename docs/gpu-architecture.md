# GPU Architecture for autoBot Backtester

## Executive Summary

This document defines the GPU acceleration architecture for the autoBot backtester and feature builder services. The design provides a unified abstraction layer that enables seamless switching between CPU and GPU execution paths while maintaining compatibility with the existing Polars-based I/O pipeline.

---

## 1. Architecture Overview

### 1.1 Design Principles

1. **Separation of Concerns**: Polars handles I/O and data loading; CuPy/PyTorch handle numerical computation
2. **Graceful Degradation**: All GPU operations fall back to CPU when CUDA is unavailable
3. **Lazy Initialization**: GPU resources are only allocated when first used
4. **Batch-Oriented Processing**: Operations are designed for vectorized batch execution
5. **Memory Efficiency**: Explicit memory management with pooling and cleanup

### 1.2 Component Diagram (C4 Level 2)

```
+------------------------------------------------------------------+
|                        autoBot Services                           |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+    +-------------------+                   |
|  | backtester_py     |    | feature_builder_py|                   |
|  |                   |    |                   |                   |
|  | - main.py         |    | - bars.py         |                   |
|  | - walk_forward.py |    | - micro_bars.py   |                   |
|  | - metrics.py      |    | - joins.py        |                   |
|  | - engine.py       |    | - incremental.py  |                   |
|  +--------+----------+    +--------+----------+                   |
|           |                        |                              |
|           v                        v                              |
|  +------------------------------------------------+              |
|  |              libs/gpu_utils (NEW)               |              |
|  |                                                 |              |
|  |  +------------------------------------------+  |              |
|  |  |           DeviceManager (Singleton)       |  |              |
|  |  | - detect_device()                        |  |              |
|  |  | - get_backend()                          |  |              |
|  |  | - allocate_memory()                      |  |              |
|  |  +------------------------------------------+  |              |
|  |                                                 |              |
|  |  +------------------+  +--------------------+  |              |
|  |  | CuPyBackend      |  | PyTorchBackend     |  |              |
|  |  | (Numerical ops)  |  | (ML signals)       |  |              |
|  |  +------------------+  +--------------------+  |              |
|  |                                                 |              |
|  |  +------------------------------------------+  |              |
|  |  |           ArrayProtocol (ABC)             |  |              |
|  |  | - from_numpy() / to_numpy()              |  |              |
|  |  | - from_polars() / to_polars()            |  |              |
|  |  +------------------------------------------+  |              |
|  +------------------------------------------------+              |
|                          |                                        |
|                          v                                        |
|  +-------------------+  +-------------------+                     |
|  | Polars (I/O)      |  | NumPy (Fallback)  |                     |
|  | - scan_parquet()  |  | - CPU operations  |                     |
|  | - collect()       |  |                   |                     |
|  +-------------------+  +-------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
                              |
                              v
           +----------------------------------+
           |          CUDA Runtime            |
           | (Optional - GPU acceleration)    |
           +----------------------------------+
```

---

## 2. Module Structure

### 2.1 Directory Layout

```
libs/
  gpu_utils/
    __init__.py              # Public API exports
    device.py                # Device detection and management
    backends/
      __init__.py
      base.py                # Abstract base classes
      numpy_backend.py       # CPU fallback implementation
      cupy_backend.py        # CuPy GPU implementation
      torch_backend.py       # PyTorch GPU implementation
    array_ops.py             # Array operation wrappers
    tensor_ops.py            # Tensor operations for ML
    memory.py                # Memory pool management
    config.py                # Configuration dataclasses
    exceptions.py            # Custom exception types
```

### 2.2 Module Dependencies

```
gpu_utils
  |-- device.py (DeviceManager)
  |     |-- config.py (GPUConfig)
  |     |-- exceptions.py (GPUNotAvailableError)
  |
  |-- backends/
  |     |-- base.py (ComputeBackend ABC)
  |     |     |-- array_ops.py (operations)
  |     |     |-- memory.py (MemoryPool)
  |     |
  |     |-- numpy_backend.py (NumpyBackend)
  |     |-- cupy_backend.py (CuPyBackend)
  |     |-- torch_backend.py (TorchBackend)
  |
  |-- array_ops.py (high-level array operations)
  |-- tensor_ops.py (ML-specific tensor operations)
```

---

## 3. Interface Definitions

### 3.1 Abstract Base Classes

```python
# libs/gpu_utils/backends/base.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any
import numpy as np
import polars as pl

ArrayType = TypeVar('ArrayType')


class ComputeBackend(ABC, Generic[ArrayType]):
    """
    Abstract base class for compute backends (CPU/GPU).

    All numerical operations flow through this interface, allowing
    transparent switching between NumPy, CuPy, and PyTorch.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'numpy', 'cupy', 'torch')."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Device identifier (e.g., 'cpu', 'cuda:0')."""
        ...

    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """True if this backend uses GPU acceleration."""
        ...

    # ============ Array Creation ============

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayType:
        """Create array of zeros."""
        ...

    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayType:
        """Create array of ones."""
        ...

    @abstractmethod
    def empty(self, shape: tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayType:
        """Create uninitialized array."""
        ...

    @abstractmethod
    def from_numpy(self, arr: np.ndarray) -> ArrayType:
        """Convert NumPy array to backend array type."""
        ...

    @abstractmethod
    def to_numpy(self, arr: ArrayType) -> np.ndarray:
        """Convert backend array to NumPy array."""
        ...

    # ============ Polars Integration ============

    def from_polars_series(self, series: pl.Series) -> ArrayType:
        """Convert Polars Series to backend array."""
        return self.from_numpy(series.to_numpy())

    def to_polars_series(self, arr: ArrayType, name: str = "") -> pl.Series:
        """Convert backend array to Polars Series."""
        return pl.Series(name, self.to_numpy(arr))

    def from_polars_df(self, df: pl.DataFrame, columns: list[str] | None = None) -> ArrayType:
        """
        Convert Polars DataFrame columns to 2D backend array.

        Args:
            df: Source DataFrame
            columns: Columns to extract (all numeric if None)

        Returns:
            2D array with shape (n_rows, n_columns)
        """
        if columns is None:
            columns = [c for c in df.columns if df[c].dtype.is_numeric()]

        np_arr = df.select(columns).to_numpy()
        return self.from_numpy(np_arr)

    # ============ Mathematical Operations ============

    @abstractmethod
    def add(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Element-wise addition."""
        ...

    @abstractmethod
    def subtract(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Element-wise subtraction."""
        ...

    @abstractmethod
    def multiply(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Element-wise multiplication."""
        ...

    @abstractmethod
    def divide(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Element-wise division."""
        ...

    @abstractmethod
    def dot(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Matrix multiplication / dot product."""
        ...

    @abstractmethod
    def sqrt(self, arr: ArrayType) -> ArrayType:
        """Element-wise square root."""
        ...

    @abstractmethod
    def exp(self, arr: ArrayType) -> ArrayType:
        """Element-wise exponential."""
        ...

    @abstractmethod
    def log(self, arr: ArrayType) -> ArrayType:
        """Element-wise natural logarithm."""
        ...

    @abstractmethod
    def abs(self, arr: ArrayType) -> ArrayType:
        """Element-wise absolute value."""
        ...

    # ============ Reduction Operations ============

    @abstractmethod
    def sum(self, arr: ArrayType, axis: int | None = None) -> ArrayType | float:
        """Sum along axis (or total if axis=None)."""
        ...

    @abstractmethod
    def mean(self, arr: ArrayType, axis: int | None = None) -> ArrayType | float:
        """Mean along axis."""
        ...

    @abstractmethod
    def std(self, arr: ArrayType, axis: int | None = None, ddof: int = 0) -> ArrayType | float:
        """Standard deviation along axis."""
        ...

    @abstractmethod
    def min(self, arr: ArrayType, axis: int | None = None) -> ArrayType | float:
        """Minimum along axis."""
        ...

    @abstractmethod
    def max(self, arr: ArrayType, axis: int | None = None) -> ArrayType | float:
        """Maximum along axis."""
        ...

    @abstractmethod
    def cumsum(self, arr: ArrayType, axis: int = 0) -> ArrayType:
        """Cumulative sum along axis."""
        ...

    @abstractmethod
    def cumprod(self, arr: ArrayType, axis: int = 0) -> ArrayType:
        """Cumulative product along axis."""
        ...

    # ============ Rolling Window Operations ============

    @abstractmethod
    def rolling_mean(
        self,
        arr: ArrayType,
        window: int,
        min_periods: int = 1
    ) -> ArrayType:
        """Rolling mean with specified window size."""
        ...

    @abstractmethod
    def rolling_std(
        self,
        arr: ArrayType,
        window: int,
        min_periods: int = 1,
        ddof: int = 1
    ) -> ArrayType:
        """Rolling standard deviation."""
        ...

    @abstractmethod
    def rolling_max(
        self,
        arr: ArrayType,
        window: int,
        min_periods: int = 1
    ) -> ArrayType:
        """Rolling maximum."""
        ...

    @abstractmethod
    def rolling_min(
        self,
        arr: ArrayType,
        window: int,
        min_periods: int = 1
    ) -> ArrayType:
        """Rolling minimum."""
        ...

    # ============ Financial Operations ============

    def returns(self, prices: ArrayType) -> ArrayType:
        """
        Calculate simple returns from price series.

        Returns: (prices[1:] / prices[:-1]) - 1, with leading NaN
        """
        shifted = self._shift(prices, 1)
        return self.divide(prices, shifted) - 1.0

    def log_returns(self, prices: ArrayType) -> ArrayType:
        """
        Calculate log returns from price series.

        Returns: log(prices[1:] / prices[:-1]), with leading NaN
        """
        shifted = self._shift(prices, 1)
        return self.log(self.divide(prices, shifted))

    def drawdown(self, equity: ArrayType) -> ArrayType:
        """
        Calculate drawdown series from equity curve.

        Returns: 1 - (equity / running_max)
        """
        running_max = self._cummax(equity)
        return 1.0 - self.divide(equity, running_max)

    @abstractmethod
    def _shift(self, arr: ArrayType, periods: int) -> ArrayType:
        """Shift array by n periods (for internal use)."""
        ...

    @abstractmethod
    def _cummax(self, arr: ArrayType) -> ArrayType:
        """Cumulative maximum (for internal use)."""
        ...

    # ============ Memory Management ============

    @abstractmethod
    def synchronize(self) -> None:
        """
        Synchronize device operations (no-op for CPU).

        Call this before timing operations or transferring data.
        """
        ...

    @abstractmethod
    def free_memory(self, arr: ArrayType) -> None:
        """
        Explicitly free array memory.

        On GPU, this can help manage memory pressure.
        On CPU, relies on garbage collection.
        """
        ...

    @abstractmethod
    def memory_info(self) -> dict[str, int]:
        """
        Get memory usage information.

        Returns:
            Dict with 'used', 'free', 'total' in bytes
        """
        ...
```

### 3.2 Device Manager

```python
# libs/gpu_utils/device.py

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.base import ComputeBackend


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"  # Detect automatically


@dataclass(frozen=True)
class GPUConfig:
    """
    Configuration for GPU acceleration.

    Attributes:
        device_type: Target device (cpu, cuda, auto)
        device_id: CUDA device index (0, 1, 2, etc.)
        fallback_to_cpu: Whether to fall back to CPU if GPU unavailable
        memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        enable_memory_pool: Use memory pooling for allocations
        synchronize_on_transfer: Sync GPU before data transfers
    """
    device_type: DeviceType = DeviceType.AUTO
    device_id: int = 0
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.9
    enable_memory_pool: bool = True
    synchronize_on_transfer: bool = True

    @classmethod
    def from_env(cls) -> GPUConfig:
        """
        Create config from environment variables.

        Environment Variables:
            GPU_ENABLED: "true" | "false" (default: "true")
            GPU_DEVICE: Device ID (default: "0")
            GPU_FALLBACK_CPU: "true" | "false" (default: "true")
            GPU_MEMORY_FRACTION: 0.0-1.0 (default: "0.9")
        """
        enabled = os.environ.get("GPU_ENABLED", "true").lower() in ("true", "1")
        device_type = DeviceType.AUTO if enabled else DeviceType.CPU

        return cls(
            device_type=device_type,
            device_id=int(os.environ.get("GPU_DEVICE", "0")),
            fallback_to_cpu=os.environ.get("GPU_FALLBACK_CPU", "true").lower() in ("true", "1"),
            memory_fraction=float(os.environ.get("GPU_MEMORY_FRACTION", "0.9")),
        )


class DeviceManager:
    """
    Singleton manager for compute device selection.

    Handles lazy initialization, device detection, and backend creation.
    Thread-safe for use in parallel processing.

    Usage:
        >>> manager = DeviceManager.get_instance()
        >>> backend = manager.get_backend()
        >>> arr = backend.from_numpy(np.array([1, 2, 3]))
    """

    _instance: DeviceManager | None = None
    _lock = threading.Lock()

    def __init__(self, config: GPUConfig | None = None):
        """Initialize device manager (use get_instance() instead)."""
        self._config = config or GPUConfig.from_env()
        self._backend: ComputeBackend | None = None
        self._cuda_available: bool | None = None
        self._initialized = False

    @classmethod
    def get_instance(cls, config: GPUConfig | None = None) -> DeviceManager:
        """
        Get singleton instance of DeviceManager.

        Args:
            config: Optional configuration (only used on first call)

        Returns:
            DeviceManager singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance._backend is not None:
                    cls._instance._backend.synchronize()
                cls._instance = None

    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.

        Performs lazy detection on first call.
        """
        if self._cuda_available is None:
            self._cuda_available = self._detect_cuda()
        return self._cuda_available

    def _detect_cuda(self) -> bool:
        """Detect CUDA availability without importing heavy libraries."""
        # First check: environment variable override
        if os.environ.get("FORCE_CPU", "").lower() in ("true", "1"):
            return False

        # Second check: try importing cupy
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return True
        except (ImportError, Exception):
            pass

        # Third check: try importing torch
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, Exception):
            pass

        return False

    def get_backend(self, prefer: str = "cupy") -> ComputeBackend:
        """
        Get the compute backend based on configuration.

        Args:
            prefer: Preferred GPU backend ("cupy" or "torch")

        Returns:
            ComputeBackend instance (CuPy, PyTorch, or NumPy)

        Raises:
            GPUNotAvailableError: If GPU required but unavailable
        """
        if self._backend is not None:
            return self._backend

        with self._lock:
            if self._backend is not None:
                return self._backend

            self._backend = self._create_backend(prefer)
            return self._backend

    def _create_backend(self, prefer: str) -> ComputeBackend:
        """Create the appropriate backend."""
        from .backends.numpy_backend import NumpyBackend

        # If CPU explicitly requested
        if self._config.device_type == DeviceType.CPU:
            return NumpyBackend()

        # Try GPU if available
        if self.is_cuda_available():
            if prefer == "cupy":
                try:
                    from .backends.cupy_backend import CuPyBackend
                    return CuPyBackend(
                        device_id=self._config.device_id,
                        memory_fraction=self._config.memory_fraction,
                        enable_pool=self._config.enable_memory_pool,
                    )
                except ImportError:
                    pass

            if prefer == "torch" or prefer == "cupy":
                try:
                    from .backends.torch_backend import TorchBackend
                    return TorchBackend(
                        device_id=self._config.device_id,
                    )
                except ImportError:
                    pass

        # Fallback to CPU
        if self._config.fallback_to_cpu:
            return NumpyBackend()

        from .exceptions import GPUNotAvailableError
        raise GPUNotAvailableError(
            "GPU not available and CPU fallback is disabled. "
            "Set GPU_FALLBACK_CPU=true or install CUDA."
        )

    @property
    def device_info(self) -> dict[str, str | int | bool]:
        """Get information about the current device."""
        backend = self.get_backend()
        return {
            "backend": backend.name,
            "device": backend.device,
            "is_gpu": backend.is_gpu,
            "cuda_available": self.is_cuda_available(),
            **backend.memory_info(),
        }
```

### 3.3 Configuration

```python
# libs/gpu_utils/config.py

from dataclasses import dataclass, field
from enum import Enum


class BackendPreference(Enum):
    """Preferred GPU backend."""
    CUPY = "cupy"
    TORCH = "torch"
    AUTO = "auto"


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.

    Attributes:
        batch_size: Number of rows per batch
        prefetch: Number of batches to prefetch
        pin_memory: Pin CPU memory for faster GPU transfers
    """
    batch_size: int = 100_000
    prefetch: int = 2
    pin_memory: bool = True


@dataclass
class MemoryPoolConfig:
    """
    Configuration for GPU memory pooling.

    Attributes:
        initial_size_mb: Initial pool size in MB
        max_size_mb: Maximum pool size in MB (0 = unlimited)
        growth_factor: Pool growth multiplier
    """
    initial_size_mb: int = 256
    max_size_mb: int = 0
    growth_factor: float = 2.0


@dataclass
class ComputeConfig:
    """
    Complete configuration for GPU compute operations.

    Combines device, memory, and batch settings.
    """
    device_type: str = "auto"
    device_id: int = 0
    fallback_to_cpu: bool = True
    backend_preference: BackendPreference = BackendPreference.AUTO
    batch: BatchConfig = field(default_factory=BatchConfig)
    memory_pool: MemoryPoolConfig = field(default_factory=MemoryPoolConfig)
    enable_async: bool = True
    stream_count: int = 2
```

---

## 4. Integration Points

### 4.1 Backtester Integration

The backtester's metrics calculation (`/services/backtester_py/backtester_py/metrics.py`) currently uses NumPy directly. This will be refactored to use the GPU abstraction.

**Current Code:**
```python
# metrics.py (current)
def _calculate_sharpe(self, returns: np.ndarray) -> float:
    std = np.std(returns, ddof=1)
    return np.mean(returns) / std * np.sqrt(self.periods_per_year)
```

**GPU-Accelerated Code:**
```python
# metrics.py (with GPU support)
from libs.gpu_utils import DeviceManager

class MetricsCalculator:
    def __init__(self, ...):
        self._backend = DeviceManager.get_instance().get_backend()

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        # Convert to GPU array (no-op if CPU backend)
        gpu_returns = self._backend.from_numpy(returns)

        std = self._backend.std(gpu_returns, ddof=1)
        mean = self._backend.mean(gpu_returns)
        sharpe = mean / std * np.sqrt(self.periods_per_year)

        return float(self._backend.to_numpy(sharpe))
```

### 4.2 Feature Builder Integration

The feature builder's bar calculations (`/services/feature_builder_py/feature_builder_py/bars.py`) can leverage GPU for rolling operations.

**Integration Pattern:**
```python
# bars.py (with GPU support)
from libs.gpu_utils import DeviceManager

class StandardBarBuilder:
    def __init__(self, ...):
        self._backend = DeviceManager.get_instance().get_backend()

    def _calculate_atr(self, bars: pl.DataFrame) -> pl.DataFrame:
        # Extract price columns to GPU arrays
        high = self._backend.from_polars_series(bars["high"])
        low = self._backend.from_polars_series(bars["low"])
        close = self._backend.from_polars_series(bars["close"])
        prev_close = self._backend._shift(close, 1)

        # True Range calculation on GPU
        hl = self._backend.subtract(high, low)
        hc = self._backend.abs(self._backend.subtract(high, prev_close))
        lc = self._backend.abs(self._backend.subtract(low, prev_close))

        # Element-wise max (would need to add to backend)
        true_range = self._backend.maximum(hl, self._backend.maximum(hc, lc))

        # Rolling mean on GPU
        atr = self._backend.rolling_mean(true_range, self.atr_period)

        # Convert back to Polars
        return bars.with_columns([
            self._backend.to_polars_series(atr, "atr")
        ])
```

### 4.3 Walk-Forward Optimization

The parallel walk-forward optimizer can benefit from GPU acceleration for fold processing.

**Integration Pattern:**
```python
# walk_forward.py (with GPU support)
def _process_fold_worker(
    fold_input: ParallelFoldInput,
    evaluate_fn: Callable,
    use_gpu: bool = True,
) -> ParallelFoldOutput:
    """Process fold with optional GPU acceleration."""

    if use_gpu:
        # Each worker gets its own backend (thread-safe)
        from libs.gpu_utils import DeviceManager, GPUConfig

        config = GPUConfig(
            device_id=fold_input.fold_index % torch.cuda.device_count(),
            fallback_to_cpu=True,
        )
        manager = DeviceManager(config)  # Per-worker instance
        backend = manager.get_backend()

    # Rest of processing...
```

---

## 5. Memory Management Strategy

### 5.1 Memory Lifecycle

```
+------------------+
| Data Loading     |  <- Polars handles I/O (memory-mapped)
| (Polars)         |
+--------+---------+
         |
         v
+--------+---------+
| Transfer to GPU  |  <- Zero-copy where possible
| (from_numpy)     |
+--------+---------+
         |
         v
+--------+---------+
| GPU Computation  |  <- All operations on GPU
| (backend ops)    |
+--------+---------+
         |
         v
+--------+---------+
| Transfer to CPU  |  <- Only final results
| (to_numpy)       |
+--------+---------+
         |
         v
+--------+---------+
| Store Results    |  <- Polars handles output
| (Polars)         |
+------------------+
```

### 5.2 Memory Pool Architecture

```python
# libs/gpu_utils/memory.py

class MemoryPool:
    """
    GPU memory pool for efficient allocation.

    Reduces allocation overhead by reusing memory blocks.
    """

    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self._pool: dict[int, list[Any]] = {}  # size -> [free_blocks]
        self._allocated: int = 0
        self._peak: int = 0

    def allocate(self, size_bytes: int) -> Any:
        """
        Allocate memory from pool.

        Tries to reuse existing block, allocates new if none available.
        """
        # Round up to power of 2 for efficient pooling
        bucket_size = self._round_to_bucket(size_bytes)

        if bucket_size in self._pool and self._pool[bucket_size]:
            return self._pool[bucket_size].pop()

        # Allocate new block
        block = self._raw_allocate(bucket_size)
        self._allocated += bucket_size
        self._peak = max(self._peak, self._allocated)
        return block

    def free(self, block: Any, size_bytes: int) -> None:
        """Return block to pool for reuse."""
        bucket_size = self._round_to_bucket(size_bytes)
        if bucket_size not in self._pool:
            self._pool[bucket_size] = []
        self._pool[bucket_size].append(block)

    def clear(self) -> None:
        """Release all pooled memory."""
        self._pool.clear()
        self._allocated = 0
```

### 5.3 Memory Guidelines

| Scenario | Recommendation |
|----------|----------------|
| Single backtest | Use memory pool, batch_size=100K |
| Walk-forward (sequential) | Clear pool between folds |
| Walk-forward (parallel) | Each worker gets own pool |
| Very large dataset | Use streaming with small batches |
| Multi-GPU | Distribute by fold index |

---

## 6. Fallback Patterns

### 6.1 Graceful Degradation

```python
# Example: Graceful fallback with logging

import structlog
from libs.gpu_utils import DeviceManager, GPUNotAvailableError

logger = structlog.get_logger(__name__)

def compute_metrics(data: np.ndarray) -> dict:
    """Compute metrics with automatic GPU/CPU selection."""

    try:
        manager = DeviceManager.get_instance()
        backend = manager.get_backend()

        if backend.is_gpu:
            logger.info("compute_metrics_gpu", device=backend.device)
        else:
            logger.info("compute_metrics_cpu")

        # Operations use same API regardless of backend
        gpu_data = backend.from_numpy(data)
        mean = backend.mean(gpu_data)
        std = backend.std(gpu_data, ddof=1)

        return {
            "mean": float(backend.to_numpy(mean)),
            "std": float(backend.to_numpy(std)),
            "device": backend.device,
        }

    except GPUNotAvailableError:
        logger.warning("gpu_fallback", reason="GPU not available")
        # Pure NumPy fallback
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "device": "cpu",
        }
```

### 6.2 Fallback Decision Tree

```
Start
  |
  v
Is GPU_ENABLED=true?
  |
  +-- No --> Use NumPy backend
  |
  +-- Yes
        |
        v
      Is CUDA available?
        |
        +-- No
        |     |
        |     v
        |   Is GPU_FALLBACK_CPU=true?
        |     |
        |     +-- Yes --> Use NumPy backend (log warning)
        |     |
        |     +-- No --> Raise GPUNotAvailableError
        |
        +-- Yes
              |
              v
            Is CuPy installed?
              |
              +-- Yes --> Use CuPy backend
              |
              +-- No
                    |
                    v
                  Is PyTorch installed?
                    |
                    +-- Yes --> Use PyTorch backend
                    |
                    +-- No --> Use NumPy backend (log warning)
```

---

## 7. Performance Considerations

### 7.1 When GPU Helps

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Rolling mean (1M rows) | 150ms | 8ms | 18x |
| Matrix multiply (10K x 10K) | 2.5s | 45ms | 55x |
| Vectorized returns calc | 25ms | 2ms | 12x |
| Drawdown calculation | 80ms | 5ms | 16x |

### 7.2 When CPU is Better

- Small arrays (< 10,000 elements)
- Operations dominated by data transfer
- Simple element-wise operations on small data
- Operations requiring many small allocations

### 7.3 Optimization Guidelines

1. **Batch operations**: Combine multiple small operations
2. **Minimize transfers**: Keep data on GPU as long as possible
3. **Use streams**: Overlap computation and data transfer
4. **Pool memory**: Reuse allocations to avoid overhead

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `ComputeBackend` ABC
- [ ] Implement `NumpyBackend` (CPU fallback)
- [ ] Implement `DeviceManager` singleton
- [ ] Add configuration classes
- [ ] Unit tests for CPU backend

### Phase 2: CuPy Backend (Week 2)
- [ ] Implement `CuPyBackend`
- [ ] Memory pool integration
- [ ] Rolling window operations
- [ ] Integration tests with GPU

### Phase 3: PyTorch Backend (Week 3)
- [ ] Implement `TorchBackend`
- [ ] Tensor operations for ML signals
- [ ] Autograd support for gradient-based optimization
- [ ] Integration with existing PyTorch models

### Phase 4: Service Integration (Week 4)
- [ ] Integrate with `backtester_py/metrics.py`
- [ ] Integrate with `feature_builder_py/bars.py`
- [ ] Benchmark and optimize
- [ ] Documentation and examples

---

## 9. Testing Strategy

### 9.1 Test Matrix

| Test Type | CPU Backend | CuPy Backend | PyTorch Backend |
|-----------|-------------|--------------|-----------------|
| Unit tests | Required | Required | Required |
| Numerical accuracy | Required | Required | Required |
| Memory leak tests | Required | Required | Required |
| Fallback behavior | N/A | Required | Required |
| Performance benchmarks | Baseline | Required | Required |

### 9.2 Example Test

```python
# tests/test_backends.py

import pytest
import numpy as np
from libs.gpu_utils import DeviceManager
from libs.gpu_utils.backends.numpy_backend import NumpyBackend


@pytest.fixture
def backend():
    """Get available backend (GPU if available, else CPU)."""
    return DeviceManager.get_instance().get_backend()


def test_rolling_mean_accuracy(backend):
    """Verify rolling mean matches NumPy reference."""
    np.random.seed(42)
    data = np.random.randn(10000).astype(np.float64)
    window = 20

    # Backend calculation
    gpu_data = backend.from_numpy(data)
    result = backend.rolling_mean(gpu_data, window)
    result_np = backend.to_numpy(result)

    # NumPy reference
    import pandas as pd
    expected = pd.Series(data).rolling(window).mean().to_numpy()

    # Allow for floating point tolerance
    np.testing.assert_allclose(result_np, expected, rtol=1e-6, equal_nan=True)


def test_fallback_to_cpu():
    """Verify CPU fallback when GPU unavailable."""
    import os
    os.environ["FORCE_CPU"] = "true"

    DeviceManager.reset()
    manager = DeviceManager.get_instance()
    backend = manager.get_backend()

    assert isinstance(backend, NumpyBackend)
    assert not backend.is_gpu

    os.environ.pop("FORCE_CPU")
    DeviceManager.reset()
```

---

## 10. Appendix

### 10.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_ENABLED` | `"true"` | Enable GPU acceleration |
| `GPU_DEVICE` | `"0"` | CUDA device ID |
| `GPU_FALLBACK_CPU` | `"true"` | Allow CPU fallback |
| `GPU_MEMORY_FRACTION` | `"0.9"` | GPU memory fraction |
| `FORCE_CPU` | `"false"` | Force CPU backend (testing) |

### 10.2 Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
gpu = [
    "cupy-cuda12x>=13.0.0",  # For CUDA 12.x
    "torch>=2.0.0",
]

gpu-cuda11 = [
    "cupy-cuda11x>=13.0.0",  # For CUDA 11.x
    "torch>=2.0.0",
]
```

### 10.3 References

- [CuPy Documentation](https://docs.cupy.dev/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NumPy Array Interface](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [Polars User Guide](https://docs.pola.rs/user-guide/)
