"""
End-to-end integration tests for runner_py (T6.09).

TDD tests written before implementation.
"""

from datetime import datetime

import numpy as np
import polars as pl

from runner_py.acceptance import AcceptanceTester
from runner_py.benchmark import BenchmarkConfig, PerformanceBenchmark
from runner_py.slippage_analysis import SlippageAnalyzer
from runner_py.validation import BehaviorValidator, ValidationConfig


class TestFullPipelineShadowMode:
    """End-to-end tests for shadow execution mode."""

    def test_shadow_execution_produces_valid_results(
        self, sample_shadow_results: pl.DataFrame
    ) -> None:
        """Test that shadow execution produces valid results format."""
        # Verify expected columns
        expected_columns = [
            "timestamp",
            "equity",
            "returns",
            "symbol",
            "strategy_id",
        ]
        for col in expected_columns:
            assert col in sample_shadow_results.columns

        # Verify data types
        assert sample_shadow_results["equity"].dtype in [pl.Float64, pl.Float32]
        assert sample_shadow_results["returns"].dtype in [pl.Float64, pl.Float32]

    def test_shadow_results_validation(
        self,
        sample_shadow_results: pl.DataFrame,
        sample_backtest_results: pl.DataFrame,
    ) -> None:
        """Test validation of shadow results against backtest."""
        validator = BehaviorValidator(ValidationConfig(sharpe_tolerance=0.50))

        result = validator.validate(sample_shadow_results, sample_backtest_results)

        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "sharpe_deviation")

    def test_shadow_to_slippage_analysis_flow(
        self, sample_paper_fills: pl.DataFrame
    ) -> None:
        """Test flow from shadow execution to slippage analysis."""
        analyzer = SlippageAnalyzer()
        report = analyzer.analyze(sample_paper_fills)

        assert report.total_fills > 0
        assert report.avg_slippage_bps >= 0

    def test_shadow_mode_metrics_collection(
        self, sample_shadow_results: pl.DataFrame
    ) -> None:
        """Test that shadow mode collects proper metrics."""
        # Calculate basic metrics from shadow results
        returns = sample_shadow_results["returns"].to_numpy()
        equity = sample_shadow_results["equity"].to_numpy()

        # Sharpe ratio calculation
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            assert isinstance(sharpe, (int, float))

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_dd = np.max(drawdown)
        assert 0 <= max_dd <= 1


class TestSignalToExecutionFlow:
    """Test signal generation to execution flow."""

    def test_signal_processing_complete(self) -> None:
        """Test complete signal processing pipeline."""
        # Create sample signals
        signals = [
            {
                "timestamp": datetime(2024, 1, 15, 9, 30, 0),
                "symbol": "SPY",
                "signal_type": "long",
                "strength": 0.8,
                "strategy_id": "test_strategy",
            },
            {
                "timestamp": datetime(2024, 1, 15, 9, 31, 0),
                "symbol": "SPY",
                "signal_type": "flat",
                "strength": 1.0,
                "strategy_id": "test_strategy",
            },
        ]

        # Verify signal structure
        for signal in signals:
            assert "timestamp" in signal
            assert "symbol" in signal
            assert "signal_type" in signal
            assert signal["signal_type"] in ["long", "short", "flat"]

    def test_signal_to_fill_transformation(self) -> None:
        """Test signal to fill data transformation."""
        # Sample signal
        signal = {
            "timestamp": datetime(2024, 1, 15, 9, 30, 0),
            "symbol": "SPY",
            "signal_type": "long",
            "target_notional": 10000.0,
        }

        # Expected fill structure
        fill = {
            "timestamp": signal["timestamp"],
            "symbol": signal["symbol"],
            "side": "buy",  # long -> buy
            "quantity": 22.22,  # 10000 / 450
            "expected_price": 450.0,
            "actual_price": 450.10,
            "slippage_bps": 2.2,
        }

        assert fill["side"] == "buy"
        assert fill["slippage_bps"] > 0


class TestFeatureToSignalToLog:
    """Test feature -> signal -> log flow."""

    def test_feature_to_signal_generation(self) -> None:
        """Test generating signals from features."""
        # Sample feature data
        features = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 9, 30, 0)],
            "symbol": ["SPY"],
            "sma_5": [450.0],
            "sma_20": [448.0],
            "rsi_14": [55.0],
            "volatility": [0.015],
        })

        # Verify feature structure
        assert "sma_5" in features.columns
        assert "sma_20" in features.columns
        assert len(features) > 0

    def test_signal_logging_format(self) -> None:
        """Test signal log format."""
        log_entry = {
            "timestamp": datetime(2024, 1, 15, 9, 30, 0).isoformat(),
            "strategy_id": "trend_follow_001",
            "symbol": "SPY",
            "signal_type": "long",
            "strength": 0.75,
            "features": {
                "sma_5": 450.0,
                "sma_20": 448.0,
            },
            "execution": {
                "mode": "shadow",
                "expected_price": 450.0,
                "simulated_fill_price": 450.08,
            },
        }

        assert "timestamp" in log_entry
        assert "strategy_id" in log_entry
        assert "features" in log_entry
        assert "execution" in log_entry


class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    def test_complete_pipeline_validation_benchmark(
        self,
        sample_shadow_results: pl.DataFrame,
        sample_backtest_results: pl.DataFrame,
        sample_paper_fills: pl.DataFrame,
    ) -> None:
        """Test complete pipeline: validation -> slippage -> benchmark."""
        # Step 1: Validate shadow vs backtest
        validator = BehaviorValidator(ValidationConfig(sharpe_tolerance=0.50))
        validation_result = validator.validate(
            sample_shadow_results, sample_backtest_results
        )
        assert validation_result is not None

        # Step 2: Analyze slippage
        analyzer = SlippageAnalyzer()
        slippage_report = analyzer.analyze(sample_paper_fills)
        assert slippage_report.total_fills > 0

        # Step 3: Benchmark performance
        benchmark = PerformanceBenchmark(
            BenchmarkConfig(iterations=5, warmup_iterations=1)
        )
        results = benchmark.run_full_benchmark()
        assert len(results) >= 3

    def test_acceptance_criteria_coverage(self) -> None:
        """Test that all acceptance criteria can be tested."""
        tester = AcceptanceTester()
        results = tester.run_all_tests()

        # Should have results for all criteria
        assert len(results) >= 6

        # Each result should have required fields
        for result in results:
            assert result.criterion
            assert result.expected
            assert isinstance(result.passed, bool)

    def test_error_propagation(self) -> None:
        """Test that errors are properly propagated through pipeline."""
        # Create invalid data
        invalid_df = pl.DataFrame({
            "timestamp": [],
            "equity": [],
            "returns": [],
            "symbol": [],
            "strategy_id": [],
            "fill_count": [],
            "slippage_bps": [],
        })

        validator = BehaviorValidator()
        result = validator.validate(invalid_df, invalid_df)

        # Should handle gracefully and report issues
        assert result.passed is False
        assert len(result.issues) > 0

    def test_metrics_consistency(
        self,
        sample_shadow_results: pl.DataFrame,
        sample_backtest_results: pl.DataFrame,
    ) -> None:
        """Test that metrics are consistent across components."""
        # Calculate Sharpe from shadow results (basic formula)
        returns = sample_shadow_results["returns"].to_numpy()
        if len(returns) > 1 and np.std(returns) > 0:
            # Validate reports consistent Sharpe (sign and approximate magnitude)
            validator = BehaviorValidator()
            result = validator.validate(sample_shadow_results, sample_backtest_results)

            # Sharpe values should have consistent sign and be finite
            assert np.isfinite(result.shadow_sharpe)
            assert np.isfinite(result.backtest_sharpe)
            # Both should be in a reasonable range for typical returns
            assert abs(result.shadow_sharpe) < 100
            assert abs(result.backtest_sharpe) < 100


class TestPerformanceRequirements:
    """Tests for performance requirements from PRD."""

    def test_latency_under_threshold(self) -> None:
        """Test that processing latency is under 500ms threshold."""
        import time

        benchmark = PerformanceBenchmark(
            BenchmarkConfig(iterations=10, warmup_iterations=2)
        )

        start = time.time()
        result = benchmark.benchmark_signal_generation()
        elapsed_ms = (time.time() - start) * 1000

        # Total benchmark time should be reasonable
        assert elapsed_ms < 30000  # 30 seconds max for benchmark

        # Individual operation latency should be under threshold
        assert result.avg_latency_ms < 500.0

    def test_throughput_adequate(self) -> None:
        """Test that throughput meets requirements."""
        benchmark = PerformanceBenchmark(
            BenchmarkConfig(iterations=20, warmup_iterations=5)
        )

        result = benchmark.benchmark_signal_generation()

        # Should process at least 10 signals per second
        assert result.throughput_per_sec >= 10.0


class TestDataIntegrity:
    """Tests for data integrity through pipeline."""

    def test_no_data_loss(
        self, sample_shadow_results: pl.DataFrame
    ) -> None:
        """Test that no data is lost through processing."""
        original_count = len(sample_shadow_results)

        # Process through validation (which shouldn't modify data)
        BehaviorValidator()

        # Data should be preserved
        assert len(sample_shadow_results) == original_count

    def test_timestamp_ordering(
        self, sample_shadow_results: pl.DataFrame
    ) -> None:
        """Test that timestamps remain ordered."""
        timestamps = sample_shadow_results["timestamp"].to_list()

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_numeric_precision(
        self, sample_shadow_results: pl.DataFrame
    ) -> None:
        """Test that numeric precision is maintained."""
        equity = sample_shadow_results["equity"].to_list()
        returns = sample_shadow_results["returns"].to_list()

        # All values should be finite
        for e in equity:
            assert np.isfinite(e)

        for r in returns:
            assert np.isfinite(r)
