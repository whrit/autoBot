"""
Tests for paper slippage analysis (T6.08).

TDD tests written before implementation.
"""

import numpy as np
import polars as pl
import pytest

from runner_py.slippage_analysis import (
    SlippageAnalysisConfig,
    SlippageAnalyzer,
    SlippageReport,
)


class TestSlippageAnalysisConfig:
    """Tests for SlippageAnalysisConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SlippageAnalysisConfig()
        assert config.expected_slippage_bps == 2.0
        assert config.max_acceptable_ratio == 2.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SlippageAnalysisConfig(
            expected_slippage_bps=3.0,
            max_acceptable_ratio=1.5,
        )
        assert config.expected_slippage_bps == 3.0
        assert config.max_acceptable_ratio == 1.5


class TestSlippageAnalyzer:
    """Tests for SlippageAnalyzer class."""

    def test_init_with_default_config(self) -> None:
        """Test analyzer initialization with default config."""
        analyzer = SlippageAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.expected_slippage_bps == 2.0

    def test_init_with_custom_config(self) -> None:
        """Test analyzer initialization with custom config."""
        config = SlippageAnalysisConfig(expected_slippage_bps=5.0)
        analyzer = SlippageAnalyzer(config)
        assert analyzer.config.expected_slippage_bps == 5.0

    def test_analyze_basic(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test basic slippage analysis."""
        analyzer = SlippageAnalyzer()
        report = analyzer.analyze(sample_paper_fills)

        assert isinstance(report, SlippageReport)
        assert report.total_fills == len(sample_paper_fills)
        assert report.avg_slippage_bps > 0
        assert report.max_slippage_bps >= report.avg_slippage_bps
        assert report.min_slippage_bps <= report.avg_slippage_bps
        assert report.std_slippage_bps >= 0

    def test_analyze_within_acceptable(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test analysis when slippage is within acceptable range."""
        config = SlippageAnalysisConfig(
            expected_slippage_bps=5.0,
            max_acceptable_ratio=3.0,
        )
        analyzer = SlippageAnalyzer(config)
        report = analyzer.analyze(sample_paper_fills)

        # With generous config, should be within acceptable
        assert report.within_acceptable is True
        assert report.ratio_to_expected <= config.max_acceptable_ratio

    def test_analyze_outside_acceptable(self, high_slippage_fills: pl.DataFrame) -> None:
        """Test analysis when slippage exceeds acceptable range."""
        config = SlippageAnalysisConfig(
            expected_slippage_bps=1.0,
            max_acceptable_ratio=2.0,
        )
        analyzer = SlippageAnalyzer(config)
        report = analyzer.analyze(high_slippage_fills)

        # High slippage should exceed tight tolerance
        assert report.ratio_to_expected > config.max_acceptable_ratio
        assert report.within_acceptable is False

    def test_analyze_empty_fills(self, empty_fills: pl.DataFrame) -> None:
        """Test analysis with empty fills DataFrame."""
        analyzer = SlippageAnalyzer()
        report = analyzer.analyze(empty_fills)

        assert report.total_fills == 0
        assert report.avg_slippage_bps == 0.0
        assert report.max_slippage_bps == 0.0
        assert report.min_slippage_bps == 0.0
        assert report.std_slippage_bps == 0.0
        assert report.within_acceptable is True  # No fills = no violations

    def test_identify_outliers(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test outlier identification."""
        analyzer = SlippageAnalyzer()
        outliers = analyzer.identify_outliers(sample_paper_fills, threshold_bps=10.0)

        assert isinstance(outliers, list)
        # All outliers should have slippage > threshold
        for outlier in outliers:
            assert outlier["slippage_bps"] > 10.0

    def test_identify_outliers_high_threshold(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test outlier identification with high threshold."""
        analyzer = SlippageAnalyzer()
        # Very high threshold should return no outliers
        outliers = analyzer.identify_outliers(sample_paper_fills, threshold_bps=100.0)

        assert len(outliers) == 0

    def test_identify_outliers_low_threshold(self, high_slippage_fills: pl.DataFrame) -> None:
        """Test outlier identification with low threshold."""
        analyzer = SlippageAnalyzer()
        # Low threshold on high slippage should return many outliers
        outliers = analyzer.identify_outliers(high_slippage_fills, threshold_bps=5.0)

        assert len(outliers) > 0

    def test_analyze_by_symbol(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test breakdown by symbol."""
        analyzer = SlippageAnalyzer()
        by_symbol = analyzer.analyze_by_symbol(sample_paper_fills)

        assert isinstance(by_symbol, dict)
        # Should have entries for each unique symbol
        unique_symbols = sample_paper_fills["symbol"].unique().to_list()
        for symbol in unique_symbols:
            assert symbol in by_symbol
            assert "avg_slippage_bps" in by_symbol[symbol]
            assert "fill_count" in by_symbol[symbol]
            assert "max_slippage_bps" in by_symbol[symbol]

    def test_analyze_by_symbol_single_symbol(self) -> None:
        """Test symbol breakdown with single symbol."""
        from datetime import datetime

        analyzer = SlippageAnalyzer()
        single_symbol_fills = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 9, 30, 0)],
            "symbol": ["SPY"],
            "side": ["buy"],
            "quantity": [100.0],
            "expected_price": [450.0],
            "actual_price": [450.05],
            "slippage_bps": [1.1],
            "order_id": ["order_001"],
        })

        by_symbol = analyzer.analyze_by_symbol(single_symbol_fills)

        assert len(by_symbol) == 1
        assert "SPY" in by_symbol
        assert by_symbol["SPY"]["fill_count"] == 1

    def test_analyze_by_time(self, sample_paper_fills: pl.DataFrame) -> None:
        """Test breakdown by time of day."""
        analyzer = SlippageAnalyzer()
        by_hour = analyzer.analyze_by_time(sample_paper_fills)

        assert isinstance(by_hour, dict)
        # All hours should be valid (0-23)
        for hour in by_hour.keys():
            assert 0 <= hour <= 23
            assert "avg_slippage_bps" in by_hour[hour]
            assert "fill_count" in by_hour[hour]

    def test_analyze_by_time_empty(self, empty_fills: pl.DataFrame) -> None:
        """Test time breakdown with empty fills."""
        analyzer = SlippageAnalyzer()
        by_hour = analyzer.analyze_by_time(empty_fills)

        assert isinstance(by_hour, dict)
        assert len(by_hour) == 0


class TestSlippageReport:
    """Tests for SlippageReport dataclass."""

    def test_report_creation(self) -> None:
        """Test creating a slippage report."""
        report = SlippageReport(
            total_fills=100,
            avg_slippage_bps=2.5,
            max_slippage_bps=10.0,
            min_slippage_bps=0.5,
            std_slippage_bps=1.5,
            ratio_to_expected=1.25,
            within_acceptable=True,
            by_symbol={"SPY": {"avg_slippage_bps": 2.3, "fill_count": 60}},
            by_hour={9: {"avg_slippage_bps": 3.0, "fill_count": 20}},
            outliers=[],
        )

        assert report.total_fills == 100
        assert report.avg_slippage_bps == 2.5
        assert report.ratio_to_expected == 1.25
        assert report.within_acceptable is True
        assert "SPY" in report.by_symbol

    def test_report_with_outliers(self) -> None:
        """Test slippage report with outliers."""
        report = SlippageReport(
            total_fills=50,
            avg_slippage_bps=5.0,
            max_slippage_bps=25.0,
            min_slippage_bps=1.0,
            std_slippage_bps=5.0,
            ratio_to_expected=2.5,
            within_acceptable=False,
            by_symbol={},
            by_hour={},
            outliers=[
                {"order_id": "order_001", "slippage_bps": 25.0, "symbol": "AAPL"},
                {"order_id": "order_002", "slippage_bps": 20.0, "symbol": "SPY"},
            ],
        )

        assert report.within_acceptable is False
        assert len(report.outliers) == 2
        assert report.outliers[0]["slippage_bps"] == 25.0


class TestSlippageAnalysisEdgeCases:
    """Edge case tests for slippage analysis."""

    def test_analyze_single_fill(self) -> None:
        """Test analysis with single fill."""
        from datetime import datetime

        analyzer = SlippageAnalyzer()
        single_fill = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 9, 30, 0)],
            "symbol": ["SPY"],
            "side": ["buy"],
            "quantity": [100.0],
            "expected_price": [450.0],
            "actual_price": [450.10],
            "slippage_bps": [2.2],
            "order_id": ["order_001"],
        })

        report = analyzer.analyze(single_fill)

        assert report.total_fills == 1
        assert report.avg_slippage_bps == 2.2
        assert report.max_slippage_bps == 2.2
        assert report.min_slippage_bps == 2.2
        assert report.std_slippage_bps == 0.0  # Single value has no std

    def test_analyze_zero_slippage(self) -> None:
        """Test analysis with zero slippage fills."""
        from datetime import datetime

        analyzer = SlippageAnalyzer()
        zero_slippage = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 9, 30, 0)] * 5,
            "symbol": ["SPY"] * 5,
            "side": ["buy"] * 5,
            "quantity": [100.0] * 5,
            "expected_price": [450.0] * 5,
            "actual_price": [450.0] * 5,
            "slippage_bps": [0.0] * 5,
            "order_id": [f"order_{i}" for i in range(5)],
        })

        report = analyzer.analyze(zero_slippage)

        assert report.avg_slippage_bps == 0.0
        assert report.ratio_to_expected == 0.0
        assert report.within_acceptable is True

    def test_analyze_negative_slippage(self) -> None:
        """Test analysis handles negative slippage (price improvement)."""
        from datetime import datetime

        analyzer = SlippageAnalyzer()
        # Negative slippage = better than expected fill
        negative_slippage = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 9, 30, 0)],
            "symbol": ["SPY"],
            "side": ["buy"],
            "quantity": [100.0],
            "expected_price": [450.0],
            "actual_price": [449.95],
            "slippage_bps": [-1.1],  # Price improvement
            "order_id": ["order_001"],
        })

        report = analyzer.analyze(negative_slippage)

        # Should handle negative values correctly
        assert report.min_slippage_bps < 0
        assert isinstance(report.avg_slippage_bps, float)
