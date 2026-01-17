"""
GPU array wrapper with CPU fallback.

Provides a unified interface for array operations that automatically
uses GPU (CuPy) when available, falling back to CPU (NumPy) otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np

from backtester_py.gpu.device import get_array_module, is_cuda_available, to_cpu, to_gpu

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cupy as cp

    # Type aliases for type checking only
    ArrayLike = np.ndarray | cp.ndarray | Sequence[float] | Sequence[int]
    ShapeLike = tuple[int, ...] | int
else:
    # Runtime type aliases (simpler, don't need actual types)
    ArrayLike = Any
    ShapeLike = Any


class GpuArray:
    """
    GPU-accelerated array wrapper with automatic CPU fallback.

    Provides a NumPy-like interface that automatically uses GPU operations
    when CuPy is available, falling back to NumPy otherwise.

    Attributes:
        data: The underlying array (CuPy or NumPy).

    Example:
        >>> arr = GpuArray(np.array([1.0, 2.0, 3.0]))
        >>> result = arr.sum()  # Uses GPU if available
        >>> cpu_result = arr.to_numpy()  # Always returns NumPy array
    """

    _data: Any

    def __init__(self, data: Any) -> None:
        """
        Initialize GpuArray.

        Args:
            data: Input data (numpy array, cupy array, or sequence).
        """
        if isinstance(data, GpuArray):
            self._data = data._data
        elif isinstance(data, np.ndarray):
            self._data = to_gpu(data)
        else:
            # Convert to numpy first, then to GPU
            self._data = to_gpu(np.asarray(data))

    @property
    def data(self) -> cp.ndarray | np.ndarray:
        """Get the underlying array."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Get array shape."""
        return tuple(self._data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get array dtype."""
        dtype_val: np.dtype[Any] = np.dtype(self._data.dtype)
        return dtype_val

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return int(self._data.ndim)

    @property
    def size(self) -> int:
        """Get total number of elements."""
        return int(self._data.size)

    def __len__(self) -> int:
        """Get length of first dimension."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation."""
        device = "cuda" if is_cuda_available() else "cpu"
        return f"GpuArray(shape={self.shape}, dtype={self.dtype}, device={device})"

    def to_numpy(self) -> np.ndarray:
        """
        Convert to NumPy array on CPU.

        Returns:
            NumPy array.
        """
        return to_cpu(self._data)

    def _wrap(self, result: cp.ndarray | np.ndarray) -> GpuArray:
        """Wrap result in GpuArray."""
        arr = GpuArray.__new__(GpuArray)
        arr._data = result
        return arr

    def _wrap_scalar(self, result: Any) -> float:
        """Convert scalar result to float."""
        return float(to_cpu(np.asarray(result)))

    # ========== Arithmetic Operations ==========

    def __add__(self, other: GpuArray | float | int) -> GpuArray:
        """Addition."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data + other._data)
        return self._wrap(self._data + other)

    def __radd__(self, other: float | int) -> GpuArray:
        """Right addition."""
        return self._wrap(other + self._data)

    def __sub__(self, other: GpuArray | float | int) -> GpuArray:
        """Subtraction."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data - other._data)
        return self._wrap(self._data - other)

    def __rsub__(self, other: float | int) -> GpuArray:
        """Right subtraction."""
        return self._wrap(other - self._data)

    def __mul__(self, other: GpuArray | float | int) -> GpuArray:
        """Multiplication."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data * other._data)
        return self._wrap(self._data * other)

    def __rmul__(self, other: float | int) -> GpuArray:
        """Right multiplication."""
        return self._wrap(other * self._data)

    def __truediv__(self, other: GpuArray | float | int) -> GpuArray:
        """Division."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data / other._data)
        return self._wrap(self._data / other)

    def __rtruediv__(self, other: float | int) -> GpuArray:
        """Right division."""
        return self._wrap(other / self._data)

    def __floordiv__(self, other: GpuArray | float | int) -> GpuArray:
        """Floor division."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data // other._data)
        return self._wrap(self._data // other)

    def __pow__(self, other: GpuArray | float | int) -> GpuArray:
        """Power."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data ** other._data)
        return self._wrap(self._data ** other)

    def __neg__(self) -> GpuArray:
        """Negation."""
        return self._wrap(-self._data)

    def __abs__(self) -> GpuArray:
        """Absolute value."""
        xp = get_array_module(self._data)
        return self._wrap(xp.abs(self._data))

    # ========== Comparison Operations ==========

    def __gt__(self, other: GpuArray | float | int) -> GpuArray:
        """Greater than."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data > other._data)
        return self._wrap(self._data > other)

    def __ge__(self, other: GpuArray | float | int) -> GpuArray:
        """Greater than or equal."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data >= other._data)
        return self._wrap(self._data >= other)

    def __lt__(self, other: GpuArray | float | int) -> GpuArray:
        """Less than."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data < other._data)
        return self._wrap(self._data < other)

    def __le__(self, other: GpuArray | float | int) -> GpuArray:
        """Less than or equal."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data <= other._data)
        return self._wrap(self._data <= other)

    def __eq__(self, other: object) -> GpuArray:  # type: ignore[override]
        """Equality."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data == other._data)
        return self._wrap(self._data == other)

    def __ne__(self, other: object) -> GpuArray:  # type: ignore[override]
        """Not equal."""
        if isinstance(other, GpuArray):
            return self._wrap(self._data != other._data)
        return self._wrap(self._data != other)

    # ========== Indexing ==========

    def __getitem__(
        self, key: SupportsIndex | slice | np.ndarray | GpuArray
    ) -> float | GpuArray:
        """Get item or slice."""
        if isinstance(key, GpuArray):
            key = key._data
        result = self._data[key]
        # If result is scalar, return as float
        if hasattr(result, 'shape') and result.shape == ():
            return self._wrap_scalar(result)
        if not hasattr(result, 'shape'):
            return float(result)
        return self._wrap(result)

    def __setitem__(
        self, key: SupportsIndex | slice | np.ndarray | GpuArray, value: float | GpuArray
    ) -> None:
        """Set item or slice."""
        if isinstance(key, GpuArray):
            key = key._data
        if isinstance(value, GpuArray):
            value = value._data
        self._data[key] = value

    # ========== Reduction Operations ==========

    def sum(self, axis: int | None = None, keepdims: bool = False) -> float | GpuArray:
        """
        Sum of array elements.

        Args:
            axis: Axis along which to sum.
            keepdims: Keep reduced dimensions.

        Returns:
            Sum result.
        """
        xp = get_array_module(self._data)
        result = xp.sum(self._data, axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> float | GpuArray:
        """
        Mean of array elements.

        Args:
            axis: Axis along which to compute mean.
            keepdims: Keep reduced dimensions.

        Returns:
            Mean result.
        """
        xp = get_array_module(self._data)
        result = xp.mean(self._data, axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def std(
        self, axis: int | None = None, ddof: int = 0, keepdims: bool = False
    ) -> float | GpuArray:
        """
        Standard deviation of array elements.

        Args:
            axis: Axis along which to compute std.
            ddof: Delta degrees of freedom.
            keepdims: Keep reduced dimensions.

        Returns:
            Standard deviation result.
        """
        xp = get_array_module(self._data)
        result = xp.std(self._data, axis=axis, ddof=ddof, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def var(
        self, axis: int | None = None, ddof: int = 0, keepdims: bool = False
    ) -> float | GpuArray:
        """
        Variance of array elements.

        Args:
            axis: Axis along which to compute variance.
            ddof: Delta degrees of freedom.
            keepdims: Keep reduced dimensions.

        Returns:
            Variance result.
        """
        xp = get_array_module(self._data)
        result = xp.var(self._data, axis=axis, ddof=ddof, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def min(self, axis: int | None = None, keepdims: bool = False) -> float | GpuArray:
        """
        Minimum of array elements.

        Args:
            axis: Axis along which to find minimum.
            keepdims: Keep reduced dimensions.

        Returns:
            Minimum result.
        """
        xp = get_array_module(self._data)
        result = xp.min(self._data, axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def max(self, axis: int | None = None, keepdims: bool = False) -> float | GpuArray:
        """
        Maximum of array elements.

        Args:
            axis: Axis along which to find maximum.
            keepdims: Keep reduced dimensions.

        Returns:
            Maximum result.
        """
        xp = get_array_module(self._data)
        result = xp.max(self._data, axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return self._wrap_scalar(result)
        return self._wrap(result)

    def argmin(self, axis: int | None = None) -> int | GpuArray:
        """Index of minimum element."""
        xp = get_array_module(self._data)
        result = xp.argmin(self._data, axis=axis)
        if axis is None:
            return int(to_cpu(np.asarray(result)))
        return self._wrap(result)

    def argmax(self, axis: int | None = None) -> int | GpuArray:
        """Index of maximum element."""
        xp = get_array_module(self._data)
        result = xp.argmax(self._data, axis=axis)
        if axis is None:
            return int(to_cpu(np.asarray(result)))
        return self._wrap(result)

    # ========== Math Functions ==========

    def sqrt(self) -> GpuArray:
        """Square root."""
        xp = get_array_module(self._data)
        return self._wrap(xp.sqrt(self._data))

    def exp(self) -> GpuArray:
        """Exponential."""
        xp = get_array_module(self._data)
        return self._wrap(xp.exp(self._data))

    def log(self) -> GpuArray:
        """Natural logarithm."""
        xp = get_array_module(self._data)
        return self._wrap(xp.log(self._data))

    def log10(self) -> GpuArray:
        """Base-10 logarithm."""
        xp = get_array_module(self._data)
        return self._wrap(xp.log10(self._data))

    def abs(self) -> GpuArray:
        """Absolute value."""
        xp = get_array_module(self._data)
        return self._wrap(xp.abs(self._data))

    def sin(self) -> GpuArray:
        """Sine."""
        xp = get_array_module(self._data)
        return self._wrap(xp.sin(self._data))

    def cos(self) -> GpuArray:
        """Cosine."""
        xp = get_array_module(self._data)
        return self._wrap(xp.cos(self._data))

    def tan(self) -> GpuArray:
        """Tangent."""
        xp = get_array_module(self._data)
        return self._wrap(xp.tan(self._data))

    # ========== Cumulative Operations ==========

    def cumsum(self, axis: int | None = None) -> GpuArray:
        """Cumulative sum."""
        xp = get_array_module(self._data)
        return self._wrap(xp.cumsum(self._data, axis=axis))

    def cumprod(self, axis: int | None = None) -> GpuArray:
        """Cumulative product."""
        xp = get_array_module(self._data)
        return self._wrap(xp.cumprod(self._data, axis=axis))

    def diff(self, n: int = 1, axis: int = -1) -> GpuArray:
        """Difference between consecutive elements."""
        xp = get_array_module(self._data)
        return self._wrap(xp.diff(self._data, n=n, axis=axis))

    # ========== Array Manipulation ==========

    def reshape(self, shape: ShapeLike) -> GpuArray:
        """Reshape array."""
        if isinstance(shape, int):
            shape = (shape,)
        return self._wrap(self._data.reshape(shape))

    def flatten(self) -> GpuArray:
        """Flatten to 1D."""
        return self._wrap(self._data.flatten())

    def ravel(self) -> GpuArray:
        """Return flattened array (may be view)."""
        return self._wrap(self._data.ravel())

    def transpose(self, axes: tuple[int, ...] | None = None) -> GpuArray:
        """Transpose array."""
        if axes is None:
            return self._wrap(self._data.T)
        return self._wrap(self._data.transpose(axes))

    @property
    def T(self) -> GpuArray:  # noqa: N802
        """Transpose."""
        return self._wrap(self._data.T)

    def roll(self, shift: int, axis: int | None = None) -> GpuArray:
        """Roll array elements."""
        xp = get_array_module(self._data)
        return self._wrap(xp.roll(self._data, shift, axis=axis))

    def clip(self, a_min: float | None, a_max: float | None) -> GpuArray:
        """Clip values to range."""
        xp = get_array_module(self._data)
        return self._wrap(xp.clip(self._data, a_min, a_max))

    def astype(self, dtype: np.dtype[Any] | type) -> GpuArray:
        """Cast to different dtype."""
        return self._wrap(self._data.astype(dtype))

    def copy(self) -> GpuArray:
        """Create a copy."""
        return self._wrap(self._data.copy())


# ========== Factory Functions ==========


def zeros(shape: ShapeLike, dtype: np.dtype[Any] | type = np.float64) -> GpuArray:
    """
    Create array filled with zeros.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        GpuArray filled with zeros.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        data = cp.zeros(shape, dtype=dtype)
    else:
        data = np.zeros(shape, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def ones(shape: ShapeLike, dtype: np.dtype[Any] | type = np.float64) -> GpuArray:
    """
    Create array filled with ones.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        GpuArray filled with ones.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        data = cp.ones(shape, dtype=dtype)
    else:
        data = np.ones(shape, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def empty(shape: ShapeLike, dtype: np.dtype[Any] | type = np.float64) -> GpuArray:
    """
    Create uninitialized array.

    Args:
        shape: Array shape.
        dtype: Data type.

    Returns:
        Uninitialized GpuArray.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        data = cp.empty(shape, dtype=dtype)
    else:
        data = np.empty(shape, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: np.dtype[Any] | type | None = None,
) -> GpuArray:
    """
    Create array with evenly spaced values.

    Args:
        start: Start value (or stop if stop is None).
        stop: Stop value (exclusive).
        step: Step between values.
        dtype: Data type.

    Returns:
        GpuArray with range values.
    """
    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        if stop is None:
            data = cp.arange(start, step=step, dtype=dtype)
        else:
            data = cp.arange(start, stop, step, dtype=dtype)
    else:
        if stop is None:
            data = np.arange(start, step=step, dtype=dtype)
        else:
            data = np.arange(start, stop, step, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    dtype: np.dtype[Any] | type = np.float64,
) -> GpuArray:
    """
    Create array with linearly spaced values.

    Args:
        start: Start value.
        stop: Stop value.
        num: Number of values.
        dtype: Data type.

    Returns:
        GpuArray with linearly spaced values.
    """
    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        data = cp.linspace(start, stop, num, dtype=dtype)
    else:
        data = np.linspace(start, stop, num, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def full(
    shape: ShapeLike,
    fill_value: float,
    dtype: np.dtype[Any] | type = np.float64,
) -> GpuArray:
    """
    Create array filled with specified value.

    Args:
        shape: Array shape.
        fill_value: Value to fill array with.
        dtype: Data type.

    Returns:
        GpuArray filled with specified value.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if is_cuda_available():
        from backtester_py.gpu.device import _get_cupy

        cp = _get_cupy()
        data = cp.full(shape, fill_value, dtype=dtype)
    else:
        data = np.full(shape, fill_value, dtype=dtype)

    arr = GpuArray.__new__(GpuArray)
    arr._data = data
    return arr


def concatenate(arrays: Sequence[GpuArray], axis: int = 0) -> GpuArray:
    """
    Concatenate arrays along axis.

    Args:
        arrays: Sequence of GpuArrays.
        axis: Axis along which to concatenate.

    Returns:
        Concatenated GpuArray.
    """
    data_arrays = [a._data for a in arrays]
    xp = get_array_module(data_arrays[0])
    result = xp.concatenate(data_arrays, axis=axis)

    arr = GpuArray.__new__(GpuArray)
    arr._data = result
    return arr


def stack(arrays: Sequence[GpuArray], axis: int = 0) -> GpuArray:
    """
    Stack arrays along new axis.

    Args:
        arrays: Sequence of GpuArrays.
        axis: Axis along which to stack.

    Returns:
        Stacked GpuArray.
    """
    data_arrays = [a._data for a in arrays]
    xp = get_array_module(data_arrays[0])
    result = xp.stack(data_arrays, axis=axis)

    arr = GpuArray.__new__(GpuArray)
    arr._data = result
    return arr


__all__ = [
    "GpuArray",
    "zeros",
    "ones",
    "empty",
    "arange",
    "linspace",
    "full",
    "concatenate",
    "stack",
]
