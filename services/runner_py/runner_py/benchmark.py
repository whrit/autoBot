"""
Performance Benchmarking (T6.10).

This module provides benchmarking tools to measure system performance
including latency, throughput, and memory usage.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for performance benchmarking.

    Attributes:
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations before measurement
    """

    iterations: int = 100
    warmup_iterations: int = 10


@dataclass
class BenchmarkResult:
    """
    Result of a performance benchmark.

    Attributes:
        component: Name of the benchmarked component
        avg_latency_ms: Average latency in milliseconds
        p50_latency_ms: 50th percentile latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        throughput_per_sec: Operations per second
        memory_mb: Memory usage in megabytes
    """

    component: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_sec: float
    memory_mb: float


class PerformanceBenchmark:
    """
    Benchmark system performance.

    Provides tools to measure latency, throughput, and memory usage
    of various system components.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """
        Initialize the PerformanceBenchmark.

        Args:
            config: Benchmark configuration. Uses defaults if not provided.
        """
        self.config = config or BenchmarkConfig()

    def benchmark_signal_generation(self) -> BenchmarkResult:
        """
        Benchmark signal generation latency.

        Simulates signal generation operations and measures performance.

        Returns:
            BenchmarkResult for signal generation
        """
        latencies = self._run_benchmark(self._simulate_signal_generation)
        memory = self._measure_memory()

        return self._create_result("signal_generation", latencies, memory)

    def benchmark_shadow_execution(self) -> BenchmarkResult:
        """
        Benchmark shadow execution latency.

        Simulates shadow execution operations and measures performance.

        Returns:
            BenchmarkResult for shadow execution
        """
        latencies = self._run_benchmark(self._simulate_shadow_execution)
        memory = self._measure_memory()

        return self._create_result("shadow_execution", latencies, memory)

    def benchmark_feature_refresh(self) -> BenchmarkResult:
        """
        Benchmark feature refresh latency.

        Simulates feature refresh operations and measures performance.

        Returns:
            BenchmarkResult for feature refresh
        """
        latencies = self._run_benchmark(self._simulate_feature_refresh)
        memory = self._measure_memory()

        return self._create_result("feature_refresh", latencies, memory)

    def run_full_benchmark(self) -> list[BenchmarkResult]:
        """
        Run all benchmarks.

        Returns:
            List of BenchmarkResults for all components
        """
        results = []

        # Signal generation benchmark
        results.append(self.benchmark_signal_generation())

        # Shadow execution benchmark
        results.append(self.benchmark_shadow_execution())

        # Feature refresh benchmark
        results.append(self.benchmark_feature_refresh())

        return results

    def generate_report(self, results: list[BenchmarkResult]) -> str:
        """
        Generate a benchmark report.

        Args:
            results: List of benchmark results

        Returns:
            Formatted benchmark report string
        """
        if not results:
            return "No benchmark results to report."

        lines = [
            "=" * 70,
            "PERFORMANCE BENCHMARK REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            f"Iterations: {self.config.iterations}",
            f"Warmup: {self.config.warmup_iterations}",
            "",
            "-" * 70,
            f"{'Component':<25} {'Avg (ms)':<10} {'P50 (ms)':<10} "
            f"{'P95 (ms)':<10} {'P99 (ms)':<10} {'TPS':<10}",
            "-" * 70,
        ]

        for result in results:
            lines.append(
                f"{result.component:<25} {result.avg_latency_ms:<10.2f} "
                f"{result.p50_latency_ms:<10.2f} {result.p95_latency_ms:<10.2f} "
                f"{result.p99_latency_ms:<10.2f} {result.throughput_per_sec:<10.1f}"
            )

        lines.extend([
            "-" * 70,
            "",
            "Memory Usage:",
        ])

        for result in results:
            lines.append(f"  {result.component}: {result.memory_mb:.1f} MB")

        lines.extend([
            "",
            "=" * 70,
            f"Total components benchmarked: {len(results)}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def _run_benchmark(self, operation: Callable[[], None]) -> list[float]:
        """
        Run a benchmark operation multiple times.

        Args:
            operation: Function to benchmark

        Returns:
            List of latencies in milliseconds
        """
        latencies: list[float] = []

        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            operation()

        # Force garbage collection before measurement
        gc.collect()

        # Measurement phase
        for _ in range(self.config.iterations):
            start = time.perf_counter()
            operation()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        return latencies

    def _create_result(
        self, component: str, latencies: list[float], memory_mb: float
    ) -> BenchmarkResult:
        """
        Create a BenchmarkResult from latency measurements.

        Args:
            component: Component name
            latencies: List of latencies in ms
            memory_mb: Memory usage in MB

        Returns:
            BenchmarkResult instance
        """
        latencies_arr = np.array(latencies)

        avg_latency = float(np.mean(latencies_arr))
        p50_latency = float(np.percentile(latencies_arr, 50))
        p95_latency = float(np.percentile(latencies_arr, 95))
        p99_latency = float(np.percentile(latencies_arr, 99))

        # Calculate throughput (ops/sec)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else float("inf")

        return BenchmarkResult(
            component=component,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_per_sec=throughput,
            memory_mb=memory_mb,
        )

    def _measure_memory(self) -> float:
        """
        Measure current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        # Get memory usage from sys.getsizeof for tracked objects
        # This is an approximation - for precise measurement use tracemalloc
        gc.collect()

        # Simple approximation using process memory
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return float(usage.ru_maxrss) / 1024.0  # Convert KB to MB on Linux
        except (ImportError, AttributeError):
            # Fallback for systems without resource module
            return 0.0

    def _simulate_signal_generation(self) -> None:
        """Simulate signal generation workload."""
        # Create sample feature data
        n_rows = 100
        base_time = datetime(2024, 1, 15, 9, 30, 0)

        df = pl.DataFrame({
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
            "symbol": ["SPY"] * n_rows,
            "price": np.random.uniform(450, 455, n_rows).tolist(),
            "sma_5": np.random.uniform(450, 455, n_rows).tolist(),
            "sma_20": np.random.uniform(448, 452, n_rows).tolist(),
        })

        # Simulate signal generation logic
        signals = df.with_columns([
            (pl.col("sma_5") > pl.col("sma_20")).alias("signal")
        ])

        # Force computation
        _ = signals["signal"].sum()

    def _simulate_shadow_execution(self) -> None:
        """Simulate shadow execution workload."""
        # Create sample signal with typed values
        target_notional: float = 10000.0
        signal = {
            "timestamp": datetime(2024, 1, 15, 9, 30, 0),
            "symbol": "SPY",
            "signal_type": "long",
            "strength": 0.8,
            "target_notional": target_notional,
        }

        # Simulate quote lookup with typed values
        bid_price: float = 450.10
        ask_price: float = 450.15
        _quote = {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": 1000,
            "ask_size": 1200,
        }

        # Simulate fill calculation
        spread_bps = (ask_price - bid_price) / bid_price * 10000
        slippage_bps = spread_bps * 0.3 + 0.5  # Simplified slippage model

        fill_price = ask_price * (1 + slippage_bps / 10000)
        quantity = target_notional / fill_price

        # Simulate fill record creation
        fill = {
            "timestamp": signal["timestamp"],
            "symbol": signal["symbol"],
            "side": "buy",
            "quantity": quantity,
            "price": fill_price,
            "slippage_bps": slippage_bps,
        }

        # Force computation
        _ = fill["quantity"]

    def _simulate_feature_refresh(self) -> None:
        """Simulate feature refresh workload."""
        n_rows = 500
        base_time = datetime(2024, 1, 15, 9, 30, 0)

        # Create sample price data
        prices = np.random.uniform(450, 455, n_rows)

        df = pl.DataFrame({
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
            "price": prices.tolist(),
        })

        # Simulate feature calculations
        features = df.with_columns([
            pl.col("price").rolling_mean(window_size=5).alias("sma_5"),
            pl.col("price").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("price").rolling_std(window_size=20).alias("volatility"),
            (pl.col("price") / pl.col("price").shift(1) - 1).alias("returns"),
        ])

        # Force computation
        _ = features["volatility"].sum()


__all__ = ["BenchmarkConfig", "BenchmarkResult", "PerformanceBenchmark"]
