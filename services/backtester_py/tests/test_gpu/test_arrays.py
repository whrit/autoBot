"""
Tests for GPU array operations with CPU fallback.

TDD tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestGpuArray:
    """Tests for GpuArray class."""

    def test_gpu_array_from_numpy(self) -> None:
        """GpuArray should be creatable from numpy array."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, 2.0, 3.0])
        arr = GpuArray(data)

        assert arr is not None

    def test_gpu_array_to_numpy(self) -> None:
        """GpuArray should be convertible to numpy."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, 2.0, 3.0])
        arr = GpuArray(data)
        result = arr.to_numpy()

        np.testing.assert_array_equal(result, data)

    def test_gpu_array_shape(self) -> None:
        """GpuArray should expose shape property."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = GpuArray(data)

        assert arr.shape == (2, 2)

    def test_gpu_array_dtype(self) -> None:
        """GpuArray should expose dtype property."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr = GpuArray(data)

        assert arr.dtype == np.float32

    def test_gpu_array_len(self) -> None:
        """GpuArray should support len()."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = GpuArray(data)

        assert len(arr) == 5


class TestGpuArrayArithmetic:
    """Tests for GpuArray arithmetic operations."""

    def test_gpu_array_add(self) -> None:
        """GpuArray should support addition."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        b = GpuArray(np.array([4.0, 5.0, 6.0]))

        result = (a + b).to_numpy()
        expected = np.array([5.0, 7.0, 9.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_subtract(self) -> None:
        """GpuArray should support subtraction."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([4.0, 5.0, 6.0]))
        b = GpuArray(np.array([1.0, 2.0, 3.0]))

        result = (a - b).to_numpy()
        expected = np.array([3.0, 3.0, 3.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_multiply(self) -> None:
        """GpuArray should support multiplication."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        b = GpuArray(np.array([2.0, 3.0, 4.0]))

        result = (a * b).to_numpy()
        expected = np.array([2.0, 6.0, 12.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_divide(self) -> None:
        """GpuArray should support division."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([4.0, 6.0, 8.0]))
        b = GpuArray(np.array([2.0, 2.0, 2.0]))

        result = (a / b).to_numpy()
        expected = np.array([2.0, 3.0, 4.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_scalar_add(self) -> None:
        """GpuArray should support scalar addition."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        result = (a + 10.0).to_numpy()
        expected = np.array([11.0, 12.0, 13.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_scalar_multiply(self) -> None:
        """GpuArray should support scalar multiplication."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        result = (a * 2.0).to_numpy()
        expected = np.array([2.0, 4.0, 6.0])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayReductions:
    """Tests for GpuArray reduction operations."""

    def test_gpu_array_sum(self) -> None:
        """GpuArray should support sum()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a.sum()

        assert float(result) == pytest.approx(15.0)

    def test_gpu_array_mean(self) -> None:
        """GpuArray should support mean()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a.mean()

        assert float(result) == pytest.approx(3.0)

    def test_gpu_array_std(self) -> None:
        """GpuArray should support std()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a.std()

        # np.std with ddof=0 (population std)
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(result) == pytest.approx(expected)

    def test_gpu_array_min(self) -> None:
        """GpuArray should support min()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = a.min()

        assert float(result) == pytest.approx(1.0)

    def test_gpu_array_max(self) -> None:
        """GpuArray should support max()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = a.max()

        assert float(result) == pytest.approx(5.0)


class TestGpuArrayIndexing:
    """Tests for GpuArray indexing operations."""

    def test_gpu_array_getitem_single(self) -> None:
        """GpuArray should support single index access."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = float(a[2])

        assert result == pytest.approx(3.0)

    def test_gpu_array_getitem_slice(self) -> None:
        """GpuArray should support slice access."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a[1:4].to_numpy()
        expected = np.array([2.0, 3.0, 4.0])

        np.testing.assert_array_equal(result, expected)

    def test_gpu_array_getitem_negative(self) -> None:
        """GpuArray should support negative indexing."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = float(a[-1])

        assert result == pytest.approx(5.0)


class TestGpuArrayMath:
    """Tests for GpuArray math functions."""

    def test_gpu_array_sqrt(self) -> None:
        """GpuArray should support sqrt()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 4.0, 9.0, 16.0]))
        result = a.sqrt().to_numpy()
        expected = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_gpu_array_exp(self) -> None:
        """GpuArray should support exp()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([0.0, 1.0, 2.0]))
        result = a.exp().to_numpy()
        expected = np.exp([0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_gpu_array_log(self) -> None:
        """GpuArray should support log()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, np.e, np.e**2]))
        result = a.log().to_numpy()
        expected = np.array([0.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_gpu_array_abs(self) -> None:
        """GpuArray should support abs()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([-1.0, -2.0, 3.0, -4.0]))
        result = a.abs().to_numpy()
        expected = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayFactory:
    """Tests for GpuArray factory functions."""

    def test_zeros(self) -> None:
        """zeros() should create array of zeros."""
        from backtester_py.gpu.arrays import zeros

        arr = zeros((3, 4))
        result = arr.to_numpy()

        assert result.shape == (3, 4)
        assert np.all(result == 0.0)

    def test_ones(self) -> None:
        """ones() should create array of ones."""
        from backtester_py.gpu.arrays import ones

        arr = ones((2, 3))
        result = arr.to_numpy()

        assert result.shape == (2, 3)
        assert np.all(result == 1.0)

    def test_empty(self) -> None:
        """empty() should create uninitialized array with correct shape."""
        from backtester_py.gpu.arrays import empty

        arr = empty((5, 5))

        assert arr.shape == (5, 5)

    def test_arange(self) -> None:
        """arange() should create range array."""
        from backtester_py.gpu.arrays import arange

        arr = arange(0, 10, 2)
        result = arr.to_numpy()
        expected = np.array([0, 2, 4, 6, 8])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayAdvanced:
    """Tests for advanced GpuArray operations."""

    def test_cumsum(self) -> None:
        """GpuArray should support cumsum()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a.cumsum().to_numpy()
        expected = np.array([1.0, 3.0, 6.0, 10.0, 15.0])

        np.testing.assert_array_equal(result, expected)

    def test_cumprod(self) -> None:
        """GpuArray should support cumprod()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0]))
        result = a.cumprod().to_numpy()
        expected = np.array([1.0, 2.0, 6.0, 24.0])

        np.testing.assert_array_equal(result, expected)

    def test_diff(self) -> None:
        """GpuArray should support diff()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 3.0, 6.0, 10.0, 15.0]))
        result = a.diff().to_numpy()
        expected = np.array([2.0, 3.0, 4.0, 5.0])

        np.testing.assert_array_equal(result, expected)

    def test_roll(self) -> None:
        """GpuArray should support roll()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = a.roll(2).to_numpy()
        expected = np.array([4.0, 5.0, 1.0, 2.0, 3.0])

        np.testing.assert_array_equal(result, expected)

    def test_clip(self) -> None:
        """GpuArray should support clip()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 5.0, 10.0, 15.0, 20.0]))
        result = a.clip(5.0, 15.0).to_numpy()
        expected = np.array([5.0, 5.0, 10.0, 15.0, 15.0])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayReshape:
    """Tests for GpuArray reshape operations."""

    def test_reshape(self) -> None:
        """GpuArray should support reshape()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.arange(12))
        result = a.reshape((3, 4))

        assert result.shape == (3, 4)

    def test_flatten(self) -> None:
        """GpuArray should support flatten()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([[1, 2, 3], [4, 5, 6]]))
        result = a.flatten().to_numpy()
        expected = np.array([1, 2, 3, 4, 5, 6])

        np.testing.assert_array_equal(result, expected)

    def test_transpose(self) -> None:
        """GpuArray should support transpose()."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([[1, 2, 3], [4, 5, 6]]))
        result = a.transpose().to_numpy()
        expected = np.array([[1, 4], [2, 5], [3, 6]])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayComparison:
    """Tests for GpuArray comparison operations."""

    def test_greater_than(self) -> None:
        """GpuArray should support > comparison."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 5.0, 3.0, 7.0]))
        mask = (a > 4.0).to_numpy()
        expected = np.array([False, True, False, True])

        np.testing.assert_array_equal(mask, expected)

    def test_less_than(self) -> None:
        """GpuArray should support < comparison."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 5.0, 3.0, 7.0]))
        mask = (a < 4.0).to_numpy()
        expected = np.array([True, False, True, False])

        np.testing.assert_array_equal(mask, expected)

    def test_boolean_indexing(self) -> None:
        """GpuArray should support boolean indexing."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 5.0, 3.0, 7.0, 2.0]))
        mask = a > 3.0
        result = a[mask].to_numpy()
        expected = np.array([5.0, 7.0])

        np.testing.assert_array_equal(result, expected)


class TestGpuArrayEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array(self) -> None:
        """GpuArray should handle empty arrays."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([], dtype=np.float64))
        result = a.to_numpy()

        assert len(result) == 0
        assert result.dtype == np.float64

    def test_single_element(self) -> None:
        """GpuArray should handle single element arrays."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([42.0]))
        # Use to_numpy() to get the array, then check first element
        result = a.to_numpy()

        assert len(result) == 1
        assert result[0] == pytest.approx(42.0)

    def test_nan_handling(self) -> None:
        """GpuArray should preserve NaN values."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        a = GpuArray(data)
        result = a.to_numpy()

        assert np.isnan(result[1])
        assert np.isnan(result[3])
        assert result[0] == pytest.approx(1.0)
        assert result[4] == pytest.approx(5.0)

    def test_inf_handling(self) -> None:
        """GpuArray should preserve infinity values."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1.0, np.inf, -np.inf, 4.0])
        a = GpuArray(data)
        result = a.to_numpy()

        assert np.isinf(result[1]) and result[1] > 0
        assert np.isinf(result[2]) and result[2] < 0

    def test_nan_propagation_in_operations(self) -> None:
        """NaN should propagate through operations."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, np.nan, 3.0]))
        b = GpuArray(np.array([1.0, 2.0, 3.0]))

        result = (a + b).to_numpy()
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(6.0)

    def test_division_by_zero(self) -> None:
        """Division by zero should produce inf."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        b = GpuArray(np.array([1.0, 0.0, 1.0]))

        result = (a / b).to_numpy()
        assert result[0] == pytest.approx(1.0)
        assert np.isinf(result[1])
        assert result[2] == pytest.approx(3.0)

    def test_very_large_values(self) -> None:
        """GpuArray should handle very large values."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1e15, 2e15, 3e15], dtype=np.float64)
        a = GpuArray(data)
        result = a.to_numpy()

        np.testing.assert_array_almost_equal(result, data)

    def test_very_small_values(self) -> None:
        """GpuArray should handle very small values."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.array([1e-15, 2e-15, 3e-15], dtype=np.float64)
        a = GpuArray(data)
        result = a.to_numpy()

        np.testing.assert_array_almost_equal(result, data)

    def test_constant_array_std(self) -> None:
        """Standard deviation of constant array should be 0."""
        from backtester_py.gpu.arrays import full

        a = full(100, 42.0)
        result = float(a.std())

        assert result == pytest.approx(0.0, abs=1e-10)


class TestGpuArrayCpuGpuConsistency:
    """Tests verifying GPU and CPU produce identical results."""

    def test_addition_consistency(self) -> None:
        """GPU and CPU addition should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data_a = np.random.randn(1000).astype(np.float64)
        data_b = np.random.randn(1000).astype(np.float64)

        # CPU reference
        cpu_result = data_a + data_b

        # GPU computation
        gpu_a = GpuArray(data_a)
        gpu_b = GpuArray(data_b)
        gpu_result = (gpu_a + gpu_b).to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_multiplication_consistency(self) -> None:
        """GPU and CPU multiplication should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data_a = np.random.randn(1000).astype(np.float64)
        data_b = np.random.randn(1000).astype(np.float64)

        cpu_result = data_a * data_b

        gpu_a = GpuArray(data_a)
        gpu_b = GpuArray(data_b)
        gpu_result = (gpu_a * gpu_b).to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_division_consistency(self) -> None:
        """GPU and CPU division should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data_a = np.random.randn(1000).astype(np.float64) + 100  # Positive values
        data_b = np.random.randn(1000).astype(np.float64) + 10  # Positive non-zero

        cpu_result = data_a / data_b

        gpu_a = GpuArray(data_a)
        gpu_b = GpuArray(data_b)
        gpu_result = (gpu_a / gpu_b).to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_sum_consistency(self) -> None:
        """GPU and CPU sum should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(10000).astype(np.float64)

        cpu_result = np.sum(data)
        gpu_result = float(GpuArray(data).sum())

        assert gpu_result == pytest.approx(cpu_result, rel=1e-10)

    def test_mean_consistency(self) -> None:
        """GPU and CPU mean should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(10000).astype(np.float64)

        cpu_result = np.mean(data)
        gpu_result = float(GpuArray(data).mean())

        assert gpu_result == pytest.approx(cpu_result, rel=1e-10)

    def test_std_consistency(self) -> None:
        """GPU and CPU std should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(10000).astype(np.float64)

        cpu_result = np.std(data)
        gpu_result = float(GpuArray(data).std())

        assert gpu_result == pytest.approx(cpu_result, rel=1e-10)

    def test_cumsum_consistency(self) -> None:
        """GPU and CPU cumsum should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)

        cpu_result = np.cumsum(data)
        gpu_result = GpuArray(data).cumsum().to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_sqrt_consistency(self) -> None:
        """GPU and CPU sqrt should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.abs(np.random.randn(1000)).astype(np.float64) + 0.1

        cpu_result = np.sqrt(data)
        gpu_result = GpuArray(data).sqrt().to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_exp_consistency(self) -> None:
        """GPU and CPU exp should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)

        cpu_result = np.exp(data)
        gpu_result = GpuArray(data).exp().to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_log_consistency(self) -> None:
        """GPU and CPU log should match."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.abs(np.random.randn(1000)).astype(np.float64) + 0.1

        cpu_result = np.log(data)
        gpu_result = GpuArray(data).log().to_numpy()

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)


class TestGpuArrayLargeDatasets:
    """Tests for large dataset handling."""

    def test_large_array_creation(self) -> None:
        """Should handle creation of large arrays."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(1_000_000).astype(np.float64)
        a = GpuArray(data)

        assert len(a) == 1_000_000

    def test_large_array_operations(self) -> None:
        """Should handle operations on large arrays."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(100_000).astype(np.float64)

        a = GpuArray(data)
        b = GpuArray(data)

        # Should not raise memory errors
        result = ((a + b) * 2 - a).to_numpy()
        expected = (data + data) * 2 - data

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_large_reduction_operations(self) -> None:
        """Should handle reduction operations on large arrays."""
        from backtester_py.gpu.arrays import GpuArray

        np.random.seed(42)
        data = np.random.randn(1_000_000).astype(np.float64)

        a = GpuArray(data)

        cpu_sum = np.sum(data)
        gpu_sum = float(a.sum())

        # Tolerance relaxed for large reductions due to floating point
        assert gpu_sum == pytest.approx(cpu_sum, rel=1e-8)


class TestGpuArrayMemoryCleanup:
    """Tests for proper memory cleanup."""

    def test_array_cleanup_in_loop(self) -> None:
        """Arrays should be cleaned up properly in loops."""
        from backtester_py.gpu.arrays import GpuArray

        # Create many arrays in a loop
        for _ in range(100):
            data = np.random.randn(10000).astype(np.float64)
            a = GpuArray(data)
            _ = a.sum()
            del a

        # Should not raise memory error

    def test_intermediate_results_cleanup(self) -> None:
        """Intermediate results should be cleaned up."""
        from backtester_py.gpu.arrays import GpuArray

        data = np.random.randn(10000).astype(np.float64)

        for _ in range(50):
            a = GpuArray(data)
            # Chain of operations creates intermediates
            result = ((a + 1) * 2 - 0.5).sqrt().exp().to_numpy()
            del a

        # Should not raise memory error


class TestGpuArrayFallback:
    """Tests for graceful fallback when GPU unavailable."""

    def test_fallback_operations_work(self) -> None:
        """Operations should work regardless of GPU availability."""
        from backtester_py.gpu.arrays import GpuArray

        # This test should pass whether GPU is available or not
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = GpuArray(data)

        # All basic operations should work
        assert len(a) == 5
        assert a.shape == (5,)
        assert float(a.sum()) == pytest.approx(15.0)
        assert float(a.mean()) == pytest.approx(3.0)

        result = a.to_numpy()
        np.testing.assert_array_equal(result, data)

    def test_fallback_arithmetic(self) -> None:
        """Arithmetic should work in fallback mode."""
        from backtester_py.gpu.arrays import GpuArray

        a = GpuArray(np.array([1.0, 2.0, 3.0]))
        b = GpuArray(np.array([4.0, 5.0, 6.0]))

        result = (a + b).to_numpy()
        expected = np.array([5.0, 7.0, 9.0])

        np.testing.assert_array_equal(result, expected)
