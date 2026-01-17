"""
Pytest fixtures for GPU tests.

Provides reusable fixtures for testing GPU/CUDA functionality including:
- Sample market data generation
- CUDA availability detection
- Device management
- Test data patterns (trending, volatile, flat)
"""

from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CUDA Detection Fixtures
# ---------------------------------------------------------------------------


def _check_cuda_available() -> bool:
    """Check if CUDA is available via CuPy."""
    try:
        import cupy as cp

        # Try to get device count - this will fail if no CUDA
        device_count = cp.cuda.runtime.getDeviceCount()
        return device_count > 0
    except Exception:
        return False


@pytest.fixture
def cuda_available() -> bool:
    """Return whether CUDA is available."""
    return _check_cuda_available()


@pytest.fixture
def skip_if_no_cuda() -> None:
    """Skip test if CUDA is not available."""
    if not _check_cuda_available():
        pytest.skip("CUDA not available - skipping GPU test")


@pytest.fixture
def skip_if_cuda() -> None:
    """Skip test if CUDA IS available (for CPU-only fallback tests)."""
    if _check_cuda_available():
        pytest.skip("CUDA available - skipping CPU fallback test")


@pytest.fixture
def require_cupy():
    """Ensure CuPy is importable, skip otherwise."""
    try:
        import cupy as cp

        return cp
    except ImportError:
        pytest.skip("CuPy not installed - skipping test")


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prices() -> dict[str, np.ndarray]:
    """
    Generate sample OHLCV data with realistic market patterns.

    Returns:
        dict with 'open', 'high', 'low', 'close', 'volume' arrays
        Each array has 1000 data points.
    """
    np.random.seed(42)
    n = 1000

    # Generate a random walk for close prices starting at 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    # Generate high/low based on close with realistic volatility
    volatility = np.abs(np.random.randn(n) * 0.3)
    high = close + volatility
    low = close - volatility

    # Open is close from previous day with small gap
    open_prices = np.roll(close, 1) + np.random.randn(n) * 0.1
    open_prices[0] = close[0]

    # Volume follows a lognormal distribution
    volume = np.random.lognormal(mean=15, sigma=1, size=n).astype(np.float64)

    return {
        "open": open_prices.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "close": close.astype(np.float64),
        "volume": volume,
    }


@pytest.fixture
def small_sample_prices() -> dict[str, np.ndarray]:
    """Generate small sample data for quick tests (100 points)."""
    np.random.seed(42)
    n = 100

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volatility = np.abs(np.random.randn(n) * 0.3)

    return {
        "open": (np.roll(close, 1) + np.random.randn(n) * 0.1).astype(np.float64),
        "high": (close + volatility).astype(np.float64),
        "low": (close - volatility).astype(np.float64),
        "close": close.astype(np.float64),
        "volume": np.random.lognormal(mean=15, sigma=1, size=n).astype(np.float64),
    }


@pytest.fixture
def large_sample_prices() -> dict[str, np.ndarray]:
    """Generate large sample data for performance tests (100,000 points)."""
    np.random.seed(42)
    n = 100_000

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volatility = np.abs(np.random.randn(n) * 0.3)

    return {
        "open": (np.roll(close, 1) + np.random.randn(n) * 0.1).astype(np.float64),
        "high": (close + volatility).astype(np.float64),
        "low": (close - volatility).astype(np.float64),
        "close": close.astype(np.float64),
        "volume": np.random.lognormal(mean=15, sigma=1, size=n).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Edge Case Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_array() -> np.ndarray:
    """Return an empty numpy array."""
    return np.array([], dtype=np.float64)


@pytest.fixture
def single_value_array() -> np.ndarray:
    """Return an array with a single value."""
    return np.array([100.0], dtype=np.float64)


@pytest.fixture
def nan_array() -> np.ndarray:
    """Return an array containing NaN values."""
    arr = np.array([100.0, np.nan, 102.0, np.nan, 105.0, 104.0, np.nan], dtype=np.float64)
    return arr


@pytest.fixture
def inf_array() -> np.ndarray:
    """Return an array containing infinity values."""
    arr = np.array([100.0, np.inf, 102.0, -np.inf, 105.0], dtype=np.float64)
    return arr


@pytest.fixture
def constant_array() -> np.ndarray:
    """Return an array where all values are the same."""
    return np.full(100, 100.0, dtype=np.float64)


@pytest.fixture
def trending_up_array() -> np.ndarray:
    """Return a monotonically increasing array (strong uptrend)."""
    return np.linspace(100, 150, 100, dtype=np.float64)


@pytest.fixture
def trending_down_array() -> np.ndarray:
    """Return a monotonically decreasing array (strong downtrend)."""
    return np.linspace(150, 100, 100, dtype=np.float64)


@pytest.fixture
def volatile_array() -> np.ndarray:
    """Return a highly volatile array (alternating up/down)."""
    np.random.seed(42)
    base = np.linspace(100, 105, 100)
    noise = np.random.randn(100) * 5  # High volatility
    return (base + noise).astype(np.float64)


@pytest.fixture
def very_small_values() -> np.ndarray:
    """Return an array with very small values (near machine epsilon)."""
    return np.array([1e-10, 2e-10, 1.5e-10, 3e-10, 2.5e-10], dtype=np.float64)


@pytest.fixture
def very_large_values() -> np.ndarray:
    """Return an array with very large values."""
    return np.array([1e15, 2e15, 1.5e15, 3e15, 2.5e15], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tolerance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rtol() -> float:
    """Relative tolerance for floating-point comparisons."""
    return 1e-5


@pytest.fixture
def atol() -> float:
    """Absolute tolerance for floating-point comparisons."""
    return 1e-8


@pytest.fixture
def loose_rtol() -> float:
    """Looser relative tolerance for GPU vs CPU comparisons."""
    return 1e-4


@pytest.fixture
def loose_atol() -> float:
    """Looser absolute tolerance for GPU vs CPU comparisons."""
    return 1e-6


# ---------------------------------------------------------------------------
# Device Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_device_id() -> int:
    """Default GPU device ID for tests."""
    return 0


@pytest.fixture
def mock_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment where CUDA is not available."""
    # Mock CuPy import to fail
    import sys

    class MockCuPyImportError:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("No module named 'cupy'")

    monkeypatch.setitem(sys.modules, "cupy", None)


# ---------------------------------------------------------------------------
# Performance Tracking Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def benchmark_iterations() -> int:
    """Number of iterations for benchmark tests."""
    return 100


@pytest.fixture
def warmup_iterations() -> int:
    """Number of warmup iterations before benchmarking."""
    return 10
