# GPU/CUDA Research Findings for Trading Backtester

**Research Date:** 2026-01-16
**Libraries Researched:** CuPy, PyTorch
**Purpose:** GPU acceleration for vectorized backtesting operations

---

## Table of Contents

1. [CuPy API Patterns for Financial Calculations](#cupy-api-patterns)
2. [PyTorch Tensor Patterns for Vectorized Backtesting](#pytorch-tensor-patterns)
3. [Memory Management Best Practices](#memory-management)
4. [Rolling Window Implementations](#rolling-window-implementations)
5. [Performance Optimization Tips](#performance-optimization)
6. [Code Examples](#code-examples)

---

## CuPy API Patterns

CuPy provides a NumPy-compatible API for GPU-accelerated computing. It serves as a drop-in replacement for NumPy on NVIDIA CUDA platforms.

### Array Creation and Manipulation

```python
import cupy as cp
import numpy as np

# Create arrays directly on GPU
prices = cp.array([100.0, 101.5, 99.8, 102.3, 103.1])
zeros = cp.zeros((1000, 10), dtype=cp.float32)
ones = cp.ones((1000, 10), dtype=cp.float32)
range_arr = cp.arange(0, 1000, 1)
linspace_arr = cp.linspace(0, 1, 100)

# Transfer from NumPy (CPU to GPU)
numpy_prices = np.array([100.0, 101.5, 99.8, 102.3])
gpu_prices = cp.array(numpy_prices)  # Method 1
gpu_prices = cp.asarray(numpy_prices)  # Method 2 (preferred, no-op if already GPU)

# Transfer back to CPU
cpu_result = gpu_prices.get()  # Method 1
cpu_result = cp.asnumpy(gpu_prices)  # Method 2

# Array manipulation
reshaped = prices.reshape(5, 1)
transposed = zeros.T
concatenated = cp.concatenate([prices, prices])
```

### Financial Calculations with CuPy

```python
import cupy as cp

# Returns calculation
prices = cp.array([100.0, 101.0, 99.5, 102.0, 103.5])
returns = cp.diff(prices) / prices[:-1]  # Simple returns
log_returns = cp.diff(cp.log(prices))  # Log returns

# Cumulative operations (essential for backtesting)
cumulative_returns = cp.cumprod(1 + returns)
cumulative_sum = cp.cumsum(returns)
cumulative_max = cp.maximum.accumulate(prices)  # Running maximum (for drawdown)

# Drawdown calculation
running_max = cp.maximum.accumulate(prices)
drawdown = (prices - running_max) / running_max

# Statistical operations
mean_return = cp.mean(returns)
std_return = cp.std(returns)
sharpe = mean_return / std_return * cp.sqrt(252)  # Annualized Sharpe
```

### CPU/GPU Agnostic Code Pattern

```python
import numpy as np
import cupy as cp

def calculate_returns(prices):
    """Works with both NumPy and CuPy arrays."""
    xp = cp.get_array_module(prices)  # Dynamically select backend

    returns = xp.diff(prices) / prices[:-1]
    cumulative = xp.cumprod(1 + returns)

    return returns, cumulative

# Works with CPU (NumPy)
cpu_prices = np.array([100.0, 101.0, 102.0])
cpu_returns, cpu_cum = calculate_returns(cpu_prices)

# Works with GPU (CuPy)
gpu_prices = cp.array([100.0, 101.0, 102.0])
gpu_returns, gpu_cum = calculate_returns(gpu_prices)
```

---

## PyTorch Tensor Patterns

PyTorch provides powerful tensor operations with automatic GPU acceleration and optional gradient tracking (which we disable for backtesting).

### Tensor Creation and Device Management

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Create tensors on specific device
prices = torch.tensor([100.0, 101.5, 99.8, 102.3], device=device)
zeros = torch.zeros(1000, 10, device=device, dtype=torch.float32)
ones = torch.ones(1000, 10, device=device)
range_t = torch.arange(0, 1000, device=device)
linspace_t = torch.linspace(0, 1, 100, device=device)

# Transfer between devices
cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
gpu_tensor = cpu_tensor.cuda()  # Method 1
gpu_tensor = cpu_tensor.to('cuda:0')  # Method 2
back_to_cpu = gpu_tensor.cpu()

# From/to NumPy
import numpy as np
numpy_arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(numpy_arr)  # Shares memory!
back_to_numpy = tensor.numpy()  # Only works on CPU tensors
```

### Gradient-Free Computation (Critical for Backtesting)

```python
import torch

# Method 1: torch.no_grad() - Disables gradient tracking
with torch.no_grad():
    prices = torch.tensor([100.0, 101.0, 102.0], device='cuda')
    returns = torch.diff(prices) / prices[:-1]
    # No gradient graph is created - memory efficient

# Method 2: torch.inference_mode() - Even faster, more restrictive
with torch.inference_mode():
    prices = torch.tensor([100.0, 101.0, 102.0], device='cuda')
    returns = torch.diff(prices) / prices[:-1]
    # Best performance for inference-only workloads

# For entire functions/modules
@torch.inference_mode()
def calculate_strategy_returns(prices, signals):
    returns = torch.diff(prices) / prices[:-1]
    strategy_returns = returns * signals[:-1]
    return strategy_returns
```

### Financial Calculations with PyTorch

```python
import torch

@torch.inference_mode()
def backtest_vectorized(prices, signals, device='cuda'):
    """Vectorized backtesting on GPU."""
    prices = torch.tensor(prices, device=device, dtype=torch.float32)
    signals = torch.tensor(signals, device=device, dtype=torch.float32)

    # Calculate returns
    returns = torch.diff(prices) / prices[:-1]

    # Strategy returns
    strategy_returns = returns * signals[:-1]

    # Cumulative returns
    cumulative = torch.cumprod(1 + strategy_returns, dim=0)

    # Running maximum for drawdown
    running_max = torch.cummax(cumulative, dim=0)[0]
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = torch.min(drawdown)

    # Statistics
    total_return = cumulative[-1] - 1
    mean_return = torch.mean(strategy_returns)
    std_return = torch.std(strategy_returns)
    sharpe = mean_return / std_return * torch.sqrt(torch.tensor(252.0))

    return {
        'total_return': total_return.item(),
        'sharpe': sharpe.item(),
        'max_drawdown': max_drawdown.item(),
        'cumulative': cumulative.cpu().numpy()
    }
```

### Reduction Operations

```python
import torch

prices = torch.randn(1000, 100, device='cuda')  # 1000 assets, 100 time periods

# Sum along dimensions
total = prices.sum()  # Sum all elements
per_asset = prices.sum(dim=1)  # Sum per asset (across time)
per_period = prices.sum(dim=0)  # Sum per period (across assets)

# Mean, max, min with dimension
mean_per_asset = prices.mean(dim=1)
max_per_asset, max_indices = prices.max(dim=1)  # Also returns indices
min_per_asset = prices.min(dim=1)[0]

# Cumulative operations
cumsum = torch.cumsum(prices, dim=1)  # Cumulative sum along time
cumprod = torch.cumprod(prices, dim=1)  # Cumulative product
cummax = torch.cummax(prices, dim=1)[0]  # Running maximum
```

---

## Memory Management

### CuPy Memory Management

```python
import cupy as cp

# Access the default memory pool
mempool = cp.get_default_memory_pool()

# Check memory usage
used_bytes = mempool.used_bytes()
total_bytes = mempool.total_bytes()
print(f"GPU Memory: {used_bytes / 1e9:.2f} GB used / {total_bytes / 1e9:.2f} GB allocated")

# Free unused memory (keeps pool structure)
mempool.free_all_blocks()

# Pinned memory for faster CPU-GPU transfers
# Use when doing frequent transfers
pinned_mem = cp.cuda.alloc_pinned_memory(1000 * 8)  # 1000 float64s

# Device selection
n_devices = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(0).use()  # Use GPU 0

# Context manager for temporary device switch
with cp.cuda.Device(1):
    temp_array = cp.zeros(1000)  # Created on GPU 1
```

### PyTorch Memory Management

```python
import torch

# Check memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

# Clear cache
torch.cuda.empty_cache()

# Pinned memory for faster transfers
cpu_tensor = torch.zeros(1000, pin_memory=True)
gpu_tensor = cpu_tensor.cuda(non_blocking=True)  # Async transfer

# Device context
with torch.cuda.device(0):
    tensor = torch.zeros(1000)  # On GPU 0

# Memory-efficient operations
# Use in-place operations when possible
tensor = torch.randn(1000, device='cuda')
tensor.add_(1)  # In-place add (saves memory)
tensor.mul_(2)  # In-place multiply
```

### Best Practices for Memory Efficiency

```python
# 1. Process in batches to avoid OOM
def process_large_dataset(data, batch_size=10000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = cp.asarray(data[i:i+batch_size])
        result = process_batch(batch)
        results.append(cp.asnumpy(result))
        cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(results)

# 2. Delete intermediate results
large_array = cp.random.randn(10000, 10000)
result = cp.sum(large_array, axis=0)
del large_array  # Free memory immediately
cp.get_default_memory_pool().free_all_blocks()

# 3. Use appropriate dtypes
# float32 uses half the memory of float64
prices_f32 = cp.array(prices, dtype=cp.float32)
prices_f64 = cp.array(prices, dtype=cp.float64)  # 2x memory
```

---

## Rolling Window Implementations

### CuPy Rolling Window via Convolution

```python
import cupy as cp
from cupyx.scipy.signal import convolve

def rolling_mean_cupy(data, window):
    """Calculate rolling mean using convolution."""
    kernel = cp.ones(window) / window
    # 'valid' mode returns only complete windows
    return convolve(data, kernel, mode='valid')

def rolling_sum_cupy(data, window):
    """Calculate rolling sum using convolution."""
    kernel = cp.ones(window)
    return convolve(data, kernel, mode='valid')

def rolling_std_cupy(data, window):
    """Calculate rolling standard deviation."""
    # Rolling mean
    mean = rolling_mean_cupy(data, window)

    # Rolling mean of squares
    data_sq = data ** 2
    mean_sq = rolling_mean_cupy(data_sq, window)

    # Variance = E[X^2] - E[X]^2
    variance = mean_sq - mean ** 2
    return cp.sqrt(variance)

# Example usage
prices = cp.random.randn(10000)
ma_20 = rolling_mean_cupy(prices, 20)
std_20 = rolling_std_cupy(prices, 20)
```

### PyTorch Rolling Window via unfold

```python
import torch

def rolling_window_pytorch(data, window):
    """Create rolling windows using unfold."""
    # unfold(dimension, size, step)
    return data.unfold(0, window, 1)

def rolling_mean_pytorch(data, window):
    """Calculate rolling mean."""
    windows = data.unfold(0, window, 1)
    return windows.mean(dim=1)

def rolling_std_pytorch(data, window):
    """Calculate rolling standard deviation."""
    windows = data.unfold(0, window, 1)
    return windows.std(dim=1)

def rolling_max_pytorch(data, window):
    """Calculate rolling maximum."""
    windows = data.unfold(0, window, 1)
    return windows.max(dim=1)[0]

# Example usage
with torch.inference_mode():
    prices = torch.randn(10000, device='cuda')
    ma_20 = rolling_mean_pytorch(prices, 20)
    std_20 = rolling_std_pytorch(prices, 20)
    high_20 = rolling_max_pytorch(prices, 20)
```

### Exponential Moving Average (EMA)

```python
import cupy as cp
import torch

# CuPy EMA (iterative - consider using numba for better performance)
def ema_cupy(data, span):
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = cp.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

# PyTorch EMA (using scan-like operation)
def ema_pytorch(data, span):
    """Exponential moving average using PyTorch."""
    alpha = 2.0 / (span + 1)
    result = torch.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

# For production, consider using custom CUDA kernels for EMA
```

---

## Performance Optimization

### CuPy Performance Tips

```python
import cupy as cp

# 1. Use kernel fusion with @cp.fuse
@cp.fuse()
def fused_calculation(x, y):
    """Fused operations execute as single kernel."""
    return cp.sqrt(x ** 2 + y ** 2) * cp.exp(-x)

# 2. Custom CUDA kernels for complex operations
vector_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vector_add(const float* x, const float* y, float* z, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        z[tid] = x[tid] + y[tid];
    }
}
''', 'vector_add')

# Launch custom kernel
n = 1000000
x = cp.random.rand(n, dtype=cp.float32)
y = cp.random.rand(n, dtype=cp.float32)
z = cp.empty(n, dtype=cp.float32)

threads_per_block = 256
blocks = (n + threads_per_block - 1) // threads_per_block
vector_add_kernel((blocks,), (threads_per_block,), (x, y, z, n))

# 3. Avoid unnecessary transfers
# BAD: Multiple transfers
result = cp.asnumpy(gpu_array1) + cp.asnumpy(gpu_array2)

# GOOD: Compute on GPU, transfer once
result = cp.asnumpy(gpu_array1 + gpu_array2)

# 4. Use appropriate memory pool settings
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=4 * 1024**3)  # Limit to 4GB
```

### PyTorch Performance Tips

```python
import torch

# 1. Use torch.compile for automatic optimization (PyTorch 2.0+)
@torch.compile(backend="inductor")
def optimized_calculation(prices, signals):
    returns = torch.diff(prices) / prices[:-1]
    strategy_returns = returns * signals[:-1]
    cumulative = torch.cumprod(1 + strategy_returns, dim=0)
    return cumulative

# 2. Always use inference_mode for non-training
@torch.inference_mode()
def backtest(prices, signals):
    # All operations here skip gradient tracking
    pass

# 3. Use non_blocking transfers with pinned memory
cpu_data = torch.zeros(10000, pin_memory=True)
gpu_data = cpu_data.cuda(non_blocking=True)

# 4. Batch operations instead of loops
# BAD: Loop
results = []
for price in prices:
    results.append(calculate(price))

# GOOD: Vectorized
results = calculate_vectorized(prices)

# 5. Use appropriate dtypes
# float32 is usually sufficient and 2x faster than float64
prices = torch.tensor(data, dtype=torch.float32, device='cuda')
```

### Benchmarking Template

```python
import time
import cupy as cp
import torch

def benchmark_cupy(func, *args, n_runs=100, warmup=10):
    """Benchmark CuPy function."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        func(*args)
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000  # ms per iteration

def benchmark_pytorch(func, *args, n_runs=100, warmup=10):
    """Benchmark PyTorch function."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000  # ms per iteration
```

---

## Code Examples

### Complete Vectorized Backtest Example (CuPy)

```python
import cupy as cp
import numpy as np

class GPUBacktester:
    """GPU-accelerated backtester using CuPy."""

    def __init__(self, device_id=0):
        cp.cuda.Device(device_id).use()
        self.mempool = cp.get_default_memory_pool()

    def calculate_returns(self, prices):
        """Calculate simple and log returns."""
        simple_returns = cp.diff(prices) / prices[:-1]
        log_returns = cp.diff(cp.log(prices))
        return simple_returns, log_returns

    def calculate_signals(self, prices, short_window=20, long_window=50):
        """Simple moving average crossover signals."""
        from cupyx.scipy.signal import convolve

        short_kernel = cp.ones(short_window) / short_window
        long_kernel = cp.ones(long_window) / long_window

        short_ma = convolve(prices, short_kernel, mode='valid')
        long_ma = convolve(prices, long_kernel, mode='valid')

        # Align lengths
        offset = long_window - short_window
        short_ma = short_ma[offset:]

        # Generate signals: 1 for long, -1 for short, 0 for flat
        signals = cp.where(short_ma > long_ma, 1.0, -1.0)

        return signals

    def backtest(self, prices, signals):
        """Run vectorized backtest."""
        returns, _ = self.calculate_returns(prices)

        # Align signals with returns
        aligned_signals = signals[:len(returns)]
        aligned_returns = returns[-len(aligned_signals):]

        # Strategy returns
        strategy_returns = aligned_returns * aligned_signals

        # Cumulative returns
        cumulative = cp.cumprod(1 + strategy_returns)

        # Calculate metrics
        total_return = float(cumulative[-1] - 1)

        # Drawdown
        running_max = cp.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(cp.min(drawdown))

        # Sharpe ratio (annualized)
        mean_return = float(cp.mean(strategy_returns))
        std_return = float(cp.std(strategy_returns))
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'cumulative': cp.asnumpy(cumulative)
        }

    def cleanup(self):
        """Free GPU memory."""
        self.mempool.free_all_blocks()


# Usage
backtester = GPUBacktester()
prices = cp.random.randn(10000).cumsum() + 100  # Simulated price series
signals = backtester.calculate_signals(prices)
results = backtester.backtest(prices, signals)
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
backtester.cleanup()
```

### Complete Vectorized Backtest Example (PyTorch)

```python
import torch
import numpy as np

class TorchBacktester:
    """GPU-accelerated backtester using PyTorch."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    @torch.inference_mode()
    def calculate_returns(self, prices):
        """Calculate simple and log returns."""
        simple_returns = torch.diff(prices) / prices[:-1]
        log_returns = torch.diff(torch.log(prices))
        return simple_returns, log_returns

    @torch.inference_mode()
    def rolling_mean(self, data, window):
        """Calculate rolling mean using unfold."""
        windows = data.unfold(0, window, 1)
        return windows.mean(dim=1)

    @torch.inference_mode()
    def calculate_signals(self, prices, short_window=20, long_window=50):
        """Simple moving average crossover signals."""
        short_ma = self.rolling_mean(prices, short_window)
        long_ma = self.rolling_mean(prices, long_window)

        # Align lengths
        offset = long_window - short_window
        short_ma = short_ma[offset:]

        # Generate signals
        signals = torch.where(short_ma > long_ma,
                             torch.ones_like(short_ma),
                             -torch.ones_like(short_ma))
        return signals

    @torch.inference_mode()
    def backtest(self, prices, signals):
        """Run vectorized backtest."""
        prices = torch.tensor(prices, device=self.device, dtype=torch.float32)
        signals = signals.to(self.device)

        returns, _ = self.calculate_returns(prices)

        # Align signals with returns
        aligned_signals = signals[:len(returns)]
        aligned_returns = returns[-len(aligned_signals):]

        # Strategy returns
        strategy_returns = aligned_returns * aligned_signals

        # Cumulative returns
        cumulative = torch.cumprod(1 + strategy_returns, dim=0)

        # Calculate metrics
        total_return = (cumulative[-1] - 1).item()

        # Drawdown
        running_max = torch.cummax(cumulative, dim=0)[0]
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = torch.min(drawdown).item()

        # Sharpe ratio
        mean_return = torch.mean(strategy_returns).item()
        std_return = torch.std(strategy_returns).item()
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'cumulative': cumulative.cpu().numpy()
        }


# Usage
backtester = TorchBacktester()
prices = torch.randn(10000).cumsum() + 100
prices = prices.to(backtester.device)
signals = backtester.calculate_signals(prices)
results = backtester.backtest(prices.cpu().numpy(), signals)
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

---

## Summary of Recommendations

### When to Use CuPy
- Drop-in replacement for NumPy code
- Heavy use of scipy functions (signal processing, statistics)
- Need direct memory pool control
- Existing codebase uses NumPy conventions

### When to Use PyTorch
- Need torch.compile optimization (PyTorch 2.0+)
- Integration with ML models
- Complex tensor operations (unfold, advanced indexing)
- Better debugging tools (TensorBoard integration)

### General Guidelines
1. **Minimize CPU-GPU transfers** - They are the main bottleneck
2. **Use float32** - Sufficient precision for financial calculations, 2x memory savings
3. **Batch operations** - Process multiple securities/timeframes together
4. **Use inference mode** - Always disable gradients for backtesting
5. **Profile first** - Use NVIDIA Nsight or torch.profiler before optimizing
6. **Memory pools** - Reuse allocations to avoid fragmentation

---

## References

- [CuPy Documentation](https://docs.cupy.dev/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
