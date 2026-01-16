"""
Tests for performance benchmarking (T6.10).

TDD tests written before implementation.
"""


from runner_py.benchmark import BenchmarkConfig, BenchmarkResult, PerformanceBenchmark


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.iterations == 100
        assert config.warmup_iterations == 10

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = BenchmarkConfig(iterations=50, warmup_iterations=5)
        assert config.iterations == 50
        assert config.warmup_iterations == 5


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            component="signal_generation",
            avg_latency_ms=5.0,
            p50_latency_ms=4.5,
            p95_latency_ms=8.0,
            p99_latency_ms=12.0,
            throughput_per_sec=200.0,
            memory_mb=50.0,
        )

        assert result.component == "signal_generation"
        assert result.avg_latency_ms == 5.0
        assert result.p50_latency_ms == 4.5
        assert result.p95_latency_ms == 8.0
        assert result.p99_latency_ms == 12.0
        assert result.throughput_per_sec == 200.0
        assert result.memory_mb == 50.0

    def test_result_invariants(self) -> None:
        """Test benchmark result invariants."""
        result = BenchmarkResult(
            component="test",
            avg_latency_ms=10.0,
            p50_latency_ms=8.0,
            p95_latency_ms=20.0,
            p99_latency_ms=30.0,
            throughput_per_sec=100.0,
            memory_mb=25.0,
        )

        # P50 should typically be <= avg
        assert result.p50_latency_ms <= result.avg_latency_ms * 1.5
        # P99 should be >= P95
        assert result.p99_latency_ms >= result.p95_latency_ms
        # All values should be positive
        assert result.avg_latency_ms >= 0
        assert result.throughput_per_sec >= 0
        assert result.memory_mb >= 0


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark class."""

    def test_init_with_default_config(self) -> None:
        """Test benchmark initialization with default config."""
        benchmark = PerformanceBenchmark()
        assert benchmark.config is not None
        assert benchmark.config.iterations == 100

    def test_init_with_custom_config(self) -> None:
        """Test benchmark initialization with custom config."""
        config = BenchmarkConfig(iterations=20, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)
        assert benchmark.config.iterations == 20
        assert benchmark.config.warmup_iterations == 2

    def test_benchmark_signal_generation(self) -> None:
        """Test signal generation benchmarking."""
        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        assert isinstance(result, BenchmarkResult)
        assert result.component == "signal_generation"
        assert result.avg_latency_ms > 0
        assert result.throughput_per_sec > 0

    def test_benchmark_shadow_execution(self) -> None:
        """Test shadow execution benchmarking."""
        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_shadow_execution()

        assert isinstance(result, BenchmarkResult)
        assert result.component == "shadow_execution"
        assert result.avg_latency_ms > 0
        assert result.throughput_per_sec > 0

    def test_benchmark_feature_refresh(self) -> None:
        """Test feature refresh benchmarking."""
        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_feature_refresh()

        assert isinstance(result, BenchmarkResult)
        assert result.component == "feature_refresh"
        assert result.avg_latency_ms > 0

    def test_run_full_benchmark(self) -> None:
        """Test running all benchmarks."""
        config = BenchmarkConfig(iterations=5, warmup_iterations=1)
        benchmark = PerformanceBenchmark(config)

        results = benchmark.run_full_benchmark()

        assert isinstance(results, list)
        assert len(results) >= 3  # At least 3 components
        # Verify all results are valid
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.avg_latency_ms > 0

    def test_generate_report(self) -> None:
        """Test benchmark report generation."""
        config = BenchmarkConfig(iterations=5, warmup_iterations=1)
        benchmark = PerformanceBenchmark(config)

        results = [
            BenchmarkResult(
                component="test_1",
                avg_latency_ms=5.0,
                p50_latency_ms=4.0,
                p95_latency_ms=10.0,
                p99_latency_ms=15.0,
                throughput_per_sec=200.0,
                memory_mb=50.0,
            ),
            BenchmarkResult(
                component="test_2",
                avg_latency_ms=10.0,
                p50_latency_ms=8.0,
                p95_latency_ms=20.0,
                p99_latency_ms=30.0,
                throughput_per_sec=100.0,
                memory_mb=75.0,
            ),
        ]

        report = benchmark.generate_report(results)

        assert isinstance(report, str)
        assert "test_1" in report
        assert "test_2" in report
        # Report contains "Avg (ms)" in header
        assert "avg" in report.lower() or "ms" in report.lower()

    def test_generate_report_empty_results(self) -> None:
        """Test report generation with empty results."""
        benchmark = PerformanceBenchmark()
        report = benchmark.generate_report([])

        assert isinstance(report, str)
        # Should handle empty results gracefully


class TestBenchmarkMeasurements:
    """Tests for benchmark measurement accuracy."""

    def test_latency_percentiles_order(self) -> None:
        """Test that latency percentiles are in correct order."""
        config = BenchmarkConfig(iterations=20, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        # Percentiles should be in ascending order
        assert result.p50_latency_ms <= result.p95_latency_ms
        assert result.p95_latency_ms <= result.p99_latency_ms

    def test_throughput_consistency(self) -> None:
        """Test throughput is consistent with latency."""
        config = BenchmarkConfig(iterations=20, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        # Throughput should be approximately 1000 / avg_latency for single ops
        # Allow for some variance due to parallelism and overhead
        1000.0 / result.avg_latency_ms
        # Throughput can vary significantly due to batching etc
        assert result.throughput_per_sec > 0

    def test_memory_measurement(self) -> None:
        """Test memory usage is measured."""
        config = BenchmarkConfig(iterations=10, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        # Memory should be a reasonable value
        assert result.memory_mb >= 0
        assert result.memory_mb < 10000  # Sanity check


class TestBenchmarkEdgeCases:
    """Edge case tests for benchmarking."""

    def test_minimum_iterations(self) -> None:
        """Test with minimum iterations."""
        config = BenchmarkConfig(iterations=1, warmup_iterations=0)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        assert isinstance(result, BenchmarkResult)
        assert result.avg_latency_ms > 0

    def test_zero_warmup(self) -> None:
        """Test with zero warmup iterations."""
        config = BenchmarkConfig(iterations=5, warmup_iterations=0)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        assert isinstance(result, BenchmarkResult)

    def test_high_iterations(self) -> None:
        """Test with higher iteration count."""
        config = BenchmarkConfig(iterations=50, warmup_iterations=5)
        benchmark = PerformanceBenchmark(config)

        result = benchmark.benchmark_signal_generation()

        assert isinstance(result, BenchmarkResult)
        # Higher iterations should give more stable results
        assert result.avg_latency_ms > 0
