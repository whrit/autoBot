"""Tests for report generation module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from viz_utils import BacktestData, BacktestReport, TradeData
from viz_utils.report import BacktestMetrics, MetricDisplay, SectionConfig


class TestBacktestMetrics:
    """Tests for BacktestMetrics calculation."""

    def test_from_backtest_data(self, sample_backtest_data: BacktestData) -> None:
        """Test metrics calculation from backtest data."""
        metrics = BacktestMetrics.from_backtest_data(sample_backtest_data)

        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.num_trades == len(sample_backtest_data.trades)

    def test_metrics_reasonable_values(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test that metrics have reasonable values."""
        metrics = BacktestMetrics.from_backtest_data(sample_backtest_data)

        # Max drawdown should be between 0 and 1
        assert 0 <= metrics.max_drawdown <= 1

        # Win rate should be between 0 and 1
        assert 0 <= metrics.win_rate <= 1

        # Volatility should be positive
        assert metrics.volatility >= 0

    def test_metrics_with_risk_free_rate(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test metrics with non-zero risk-free rate."""
        metrics_rf = BacktestMetrics.from_backtest_data(
            sample_backtest_data, risk_free_rate=0.05
        )
        metrics_no_rf = BacktestMetrics.from_backtest_data(
            sample_backtest_data, risk_free_rate=0.0
        )

        # Sharpe should be lower with risk-free rate
        assert metrics_rf.sharpe_ratio <= metrics_no_rf.sharpe_ratio

    def test_metrics_empty_trades(self, minimal_backtest_data: BacktestData) -> None:
        """Test metrics with no trades."""
        metrics = BacktestMetrics.from_backtest_data(minimal_backtest_data)

        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0

    def test_metrics_all_winners(self) -> None:
        """Test metrics with all winning trades."""
        trades = [
            TradeData(
                entry_time=datetime(2023, 1, 3),
                exit_time=datetime(2023, 1, 4),
                symbol="AAPL",
                side="long",
                pnl=100.0,
                return_pct=0.01,
                notional=10000.0,
            )
            for _ in range(10)
        ]

        data = BacktestData(
            timestamps=[datetime(2023, 1, 3), datetime(2023, 1, 4)],
            equity=[100_000.0, 101_000.0],
            returns=[0.0, 0.01],
            trades=trades,
            initial_capital=100_000.0,
        )

        metrics = BacktestMetrics.from_backtest_data(data)

        assert metrics.win_rate == 1.0
        assert metrics.avg_loss == 0.0
        # Profit factor should be 0 (no losses to divide by)
        assert metrics.profit_factor == 0.0


class TestMetricDisplay:
    """Tests for MetricDisplay class."""

    def test_metric_display_creation(self) -> None:
        """Test MetricDisplay creation."""
        metric = MetricDisplay(
            label="Total Return",
            value="25.5%",
            positive=True,
        )

        assert metric.label == "Total Return"
        assert metric.value == "25.5%"
        assert metric.positive is True
        assert metric.negative is False

    def test_metric_display_negative(self) -> None:
        """Test MetricDisplay with negative value."""
        metric = MetricDisplay(
            label="Max Drawdown",
            value="-15.2%",
            negative=True,
        )

        assert metric.negative is True
        assert metric.positive is False


class TestSectionConfig:
    """Tests for SectionConfig class."""

    def test_chart_section(self) -> None:
        """Test chart section configuration."""
        section = SectionConfig(
            title="Equity Curve",
            type="chart",
            id="chart_1",
            chart_json='{"data": [], "layout": {}}',
        )

        assert section.title == "Equity Curve"
        assert section.type == "chart"
        assert section.id == "chart_1"

    def test_table_section(self) -> None:
        """Test table section configuration."""
        section = SectionConfig(
            title="Statistics",
            type="table",
            headers=["Metric", "Value"],
            rows=[["Sharpe", "1.5"], ["Max DD", "10%"]],
        )

        assert section.type == "table"
        assert section.headers is not None
        assert len(section.headers) == 2
        assert section.rows is not None
        assert len(section.rows) == 2


class TestBacktestReport:
    """Tests for BacktestReport class."""

    def test_report_creation(self, sample_backtest_data: BacktestData) -> None:
        """Test basic report creation."""
        report = BacktestReport(sample_backtest_data, title="Test Report")

        assert report.title == "Test Report"
        assert len(report.sections) == 0
        assert report.metrics is not None

    def test_add_equity_curve(self, sample_backtest_data: BacktestData) -> None:
        """Test adding equity curve to report."""
        report = BacktestReport(sample_backtest_data)
        result = report.add_equity_curve()

        assert result is report  # Method chaining
        assert len(report.sections) == 1
        assert report.sections[0].type == "chart"

    def test_add_monthly_returns(self, sample_backtest_data: BacktestData) -> None:
        """Test adding monthly returns heatmap."""
        report = BacktestReport(sample_backtest_data)
        report.add_monthly_returns()

        assert len(report.sections) == 1

    def test_add_rolling_metrics(self, sample_backtest_data: BacktestData) -> None:
        """Test adding rolling metrics charts."""
        report = BacktestReport(sample_backtest_data)
        report.add_rolling_metrics(sharpe_window=21, vol_window=10)

        assert len(report.sections) == 1
        assert report.sections[0].type == "two_charts"

    def test_add_trade_analysis(self, sample_backtest_data: BacktestData) -> None:
        """Test adding trade analysis charts."""
        report = BacktestReport(sample_backtest_data)
        report.add_trade_analysis()

        assert len(report.sections) == 1
        assert report.sections[0].type == "two_charts"

    def test_add_trade_duration(self, sample_backtest_data: BacktestData) -> None:
        """Test adding trade duration charts."""
        report = BacktestReport(sample_backtest_data)
        report.add_trade_duration()

        assert len(report.sections) == 1

    def test_add_drawdown_chart(self, sample_backtest_data: BacktestData) -> None:
        """Test adding drawdown chart."""
        report = BacktestReport(sample_backtest_data)
        report.add_drawdown_chart(min_drawdown=0.02)

        assert len(report.sections) == 1

    def test_add_var_analysis(self, sample_backtest_data: BacktestData) -> None:
        """Test adding VaR analysis."""
        report = BacktestReport(sample_backtest_data)
        report.add_var_analysis(confidence_levels=[0.95, 0.99])

        assert len(report.sections) == 1

    def test_add_statistics_table(self, sample_backtest_data: BacktestData) -> None:
        """Test adding statistics table."""
        report = BacktestReport(sample_backtest_data)
        report.add_statistics_table()

        assert len(report.sections) == 1
        assert report.sections[0].type == "table"
        assert report.sections[0].headers is not None
        assert report.sections[0].rows is not None

    def test_add_regime_returns(
        self,
        sample_backtest_data: BacktestData,
        sample_regimes: list[tuple[datetime, datetime, str]],
    ) -> None:
        """Test adding regime returns chart."""
        report = BacktestReport(sample_backtest_data)
        report.add_regime_returns(regimes=sample_regimes)

        assert len(report.sections) == 1

    def test_method_chaining(self, sample_backtest_data: BacktestData) -> None:
        """Test method chaining for adding sections."""
        report = (
            BacktestReport(sample_backtest_data)
            .add_equity_curve()
            .add_monthly_returns()
            .add_statistics_table()
        )

        assert len(report.sections) == 3

    def test_render_html(self, sample_backtest_data: BacktestData) -> None:
        """Test HTML rendering."""
        report = BacktestReport(sample_backtest_data, title="Test Report")
        report.add_equity_curve()
        report.add_statistics_table()

        html = report.render_html()

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "plotly" in html.lower()

    def test_save_html(self, sample_backtest_data: BacktestData) -> None:
        """Test saving report to HTML file."""
        report = BacktestReport(sample_backtest_data)
        report.add_equity_curve()
        report.add_statistics_table()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            report.save(path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_save_with_string_path(self, sample_backtest_data: BacktestData) -> None:
        """Test saving with string path."""
        report = BacktestReport(sample_backtest_data)
        report.add_equity_curve()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/report.html"
            report.save(path)

            assert Path(path).exists()

    def test_to_dict(self, sample_backtest_data: BacktestData) -> None:
        """Test exporting report to dictionary."""
        report = BacktestReport(sample_backtest_data, title="Test Report")

        data = report.to_dict()

        assert isinstance(data, dict)
        assert data["title"] == "Test Report"
        assert "data" in data
        assert "metrics" in data
        assert "timestamps" in data["data"]
        assert "equity" in data["data"]
        assert "trades" in data["data"]

    def test_to_dict_serializable(self, sample_backtest_data: BacktestData) -> None:
        """Test that to_dict output is JSON serializable."""
        report = BacktestReport(sample_backtest_data)
        data = report.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestBacktestReportIntegration:
    """Integration tests for BacktestReport."""

    def test_full_report_generation(self, sample_backtest_data: BacktestData) -> None:
        """Test generating a complete report with all sections."""
        report = BacktestReport(sample_backtest_data, title="Full Backtest Report")

        report.add_equity_curve(
            show_benchmark=True,
            show_drawdown=True,
            show_trades=True,
        )
        report.add_monthly_returns()
        report.add_rolling_metrics()
        report.add_trade_analysis()
        report.add_trade_duration()
        report.add_drawdown_chart()
        report.add_var_analysis()
        report.add_statistics_table()

        # Verify all sections added
        assert len(report.sections) == 8

        # Verify HTML can be generated
        html = report.render_html()
        assert len(html) > 1000  # Should be substantial

        # Verify can be saved
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full_report.html"
            report.save(path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_report_with_minimal_data(
        self, minimal_backtest_data: BacktestData
    ) -> None:
        """Test report generation with minimal data."""
        report = BacktestReport(minimal_backtest_data, title="Minimal Report")
        report.add_equity_curve()
        report.add_statistics_table()

        html = report.render_html()
        assert isinstance(html, str)

    def test_report_with_empty_data(self, empty_backtest_data: BacktestData) -> None:
        """Test report generation with empty data."""
        report = BacktestReport(empty_backtest_data, title="Empty Report")
        report.add_equity_curve()

        html = report.render_html()
        assert isinstance(html, str)

    def test_report_key_metrics_display(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test that key metrics are correctly displayed."""
        report = BacktestReport(sample_backtest_data)
        html = report.render_html()

        # Check for key metric labels
        assert "Total Return" in html
        assert "Sharpe Ratio" in html
        assert "Max Drawdown" in html
        assert "Win Rate" in html

    def test_report_date_range(self, sample_backtest_data: BacktestData) -> None:
        """Test that date range is displayed correctly."""
        report = BacktestReport(sample_backtest_data)
        html = report.render_html()

        # Check for date formatting
        assert "2023" in html  # Year should be present


class TestReportEdgeCases:
    """Edge case tests for report generation."""

    def test_report_single_trade(self) -> None:
        """Test report with single trade."""
        trades = [
            TradeData(
                entry_time=datetime(2023, 1, 3),
                exit_time=datetime(2023, 1, 4),
                symbol="AAPL",
                side="long",
                pnl=500.0,
                return_pct=0.05,
                notional=10000.0,
            )
        ]

        data = BacktestData(
            timestamps=[datetime(2023, 1, 3), datetime(2023, 1, 4)],
            equity=[100_000.0, 100_500.0],
            returns=[0.0, 0.005],
            trades=trades,
            initial_capital=100_000.0,
        )

        report = BacktestReport(data)
        report.add_trade_analysis()

        html = report.render_html()
        assert isinstance(html, str)

    def test_report_large_numbers(self) -> None:
        """Test report with large monetary values."""
        np.random.seed(42)
        n = 100
        timestamps = [datetime(2023, 1, 3) + timedelta(days=i) for i in range(n)]
        equity = [1_000_000_000.0]  # 1 billion

        for _ in range(n - 1):
            equity.append(equity[-1] * (1 + np.random.normal(0.001, 0.02)))

        data = BacktestData(
            timestamps=timestamps,
            equity=equity,
            returns=[0.0] + [
                (equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, n)
            ],
            trades=[],
            initial_capital=1_000_000_000.0,
        )

        report = BacktestReport(data)
        report.add_statistics_table()

        html = report.render_html()
        assert isinstance(html, str)

    def test_report_negative_returns(self) -> None:
        """Test report with significant losses."""
        timestamps = [datetime(2023, 1, i) for i in range(3, 13)]
        equity = [100_000.0]
        for _ in range(9):
            equity.append(equity[-1] * 0.95)  # 5% loss each day

        data = BacktestData(
            timestamps=timestamps,
            equity=equity,
            returns=[0.0] + [-0.05] * 9,
            trades=[],
            initial_capital=100_000.0,
        )

        report = BacktestReport(data)
        report.add_equity_curve()
        report.add_drawdown_chart()
        report.add_statistics_table()

        html = report.render_html()
        assert isinstance(html, str)
        # Metrics should show negative return
        assert report.metrics.total_return < 0
