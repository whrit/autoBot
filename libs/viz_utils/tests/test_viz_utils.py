"""
Comprehensive tests for viz_utils library.

Tests cover:
- Theme configuration and switching
- Chart creation and rendering
- Market data visualization
- Feature visualization
- Export functionality
"""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from viz_utils import (
    DARK_THEME,
    LIGHT_THEME,
    BarChart,
    CandlestickChart,
    FeatureCorrelationHeatmap,
    FeatureDistributionChart,
    FeatureImportanceChart,
    FeatureTimeSeriesChart,
    HeatmapChart,
    MarketChart,
    MultiChartReport,
    PriceLineChart,
    QuoteImbalanceHeatmap,
    RollingStatisticsChart,
    SimpleLineChart,
    SpreadChart,
    TradeImbalanceChart,
    create_correlation_heatmap,
    create_feature_distribution,
    create_feature_importance,
    create_market_chart,
    create_matplotlib_figure,
    create_ohlc_chart,
    create_report,
    get_theme,
    save_html,
    save_png,
    set_theme,
    to_base64,
    to_image_bytes,
    use_dark_mode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

    # Generate realistic OHLCV data
    opens = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = opens + np.abs(np.random.randn(n) * 0.3)
    lows = opens - np.abs(np.random.randn(n) * 0.3)
    closes = opens + np.random.randn(n) * 0.2
    volumes = np.random.randint(1000, 10000, n)

    return pd.DataFrame(
        {
            "bar_start": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def sample_quote_data() -> pd.DataFrame:
    """Generate sample quote data with spread."""
    np.random.seed(42)
    n = 100
    dates = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]

    midprice = 100 + np.cumsum(np.random.randn(n) * 0.1)
    spread = np.abs(np.random.randn(n) * 0.05) + 0.01

    return pd.DataFrame(
        {
            "bar_start": dates,
            "midprice": midprice,
            "spread": spread,
            "quote_imbalance": np.random.randn(n) * 0.5,
        }
    )


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n) * 2 + 1,
            "feature_3": np.random.exponential(1, n),
            "feature_4": np.random.uniform(-1, 1, n),
            "feature_5": np.random.randn(n) + np.random.randn(n) * 0.5,  # Correlated
        }
    )


@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """Generate sample time series feature data."""
    np.random.seed(42)
    n = 100
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame(
        {
            "timestamp": dates,
            "momentum": np.cumsum(np.random.randn(n) * 0.1),
            "volatility": np.abs(np.random.randn(n) * 0.5) + 0.1,
            "trend": np.linspace(-1, 1, n) + np.random.randn(n) * 0.1,
        }
    )


# =============================================================================
# Theme Tests
# =============================================================================


class TestThemes:
    """Tests for theme configuration."""

    def test_get_default_theme(self) -> None:
        """Test getting default theme."""
        theme = get_theme()
        assert theme is not None
        assert hasattr(theme, "colors")
        assert hasattr(theme, "font_family")

    def test_set_theme_light(self) -> None:
        """Test setting light theme."""
        set_theme("light")
        theme = get_theme()
        assert theme.name == "light"
        assert theme.colors.background == "#ffffff"

    def test_set_theme_dark(self) -> None:
        """Test setting dark theme."""
        set_theme("dark")
        theme = get_theme()
        assert theme.name == "dark"
        assert theme.colors.background == "#1e1e1e"

    def test_use_dark_mode(self) -> None:
        """Test dark mode toggle."""
        use_dark_mode(True)
        assert get_theme().name == "dark"

        use_dark_mode(False)
        assert get_theme().name == "light"

    def test_theme_to_plotly_layout(self) -> None:
        """Test converting theme to plotly layout."""
        layout = LIGHT_THEME.to_plotly_layout()
        assert "paper_bgcolor" in layout
        assert "plot_bgcolor" in layout
        assert "font" in layout

    def test_dark_theme_layout(self) -> None:
        """Test dark theme plotly layout."""
        layout = DARK_THEME.to_plotly_layout()
        assert layout["paper_bgcolor"] == "#252525"
        assert layout["plot_bgcolor"] == "#1e1e1e"


# =============================================================================
# Chart Base Tests
# =============================================================================


class TestBaseCharts:
    """Tests for base chart functionality."""

    def test_simple_line_chart(self, sample_time_series_data: pd.DataFrame) -> None:
        """Test simple line chart creation."""
        chart = SimpleLineChart(
            data=sample_time_series_data,
            x_col="timestamp",
            y_cols=["momentum", "volatility"],
            title="Test Line Chart",
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_bar_chart(self) -> None:
        """Test bar chart creation."""
        data = pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [10, 25, 15, 30]})

        chart = BarChart(
            data=data,
            x_col="category",
            y_col="value",
            title="Test Bar Chart",
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_heatmap_chart(self) -> None:
        """Test heatmap chart creation."""
        data = np.random.randn(5, 5)

        chart = HeatmapChart(
            data=data,
            x_labels=["A", "B", "C", "D", "E"],
            y_labels=["1", "2", "3", "4", "5"],
            title="Test Heatmap",
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_chart_annotation(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test adding annotation to chart."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        chart.add_annotation(
            x=sample_ohlcv_data["bar_start"].iloc[50],
            y=sample_ohlcv_data["close"].iloc[50],
            text="Test Point",
        )

        assert len(chart.figure.layout.annotations) == 1

    def test_chart_hline(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test adding horizontal line."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        chart.add_hline(y=100, annotation_text="Support")

        # Check that shape was added
        assert chart.figure.layout.shapes is not None


# =============================================================================
# Market Chart Tests
# =============================================================================


class TestMarketCharts:
    """Tests for market data charts."""

    def test_candlestick_chart(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test candlestick chart creation."""
        chart = CandlestickChart(
            data=sample_ohlcv_data,
            title="Test Candlestick",
            show_volume=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Candlestick + volume

    def test_candlestick_no_volume(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test candlestick chart without volume."""
        chart = CandlestickChart(
            data=sample_ohlcv_data,
            title="Test Candlestick",
            show_volume=False,
        )

        fig = chart.figure
        assert len(fig.data) == 1

    def test_candlestick_moving_average(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test adding moving average to candlestick chart."""
        chart = CandlestickChart(data=sample_ohlcv_data, show_volume=False)
        chart.add_moving_average(period=20, name="SMA20")

        assert len(chart.figure.data) == 2

    def test_candlestick_bollinger_bands(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test adding Bollinger Bands."""
        chart = CandlestickChart(data=sample_ohlcv_data, show_volume=False)
        chart.add_bollinger_bands(period=20, std_dev=2.0)

        # Should have candlestick + upper band + lower band
        assert len(chart.figure.data) == 3

    def test_market_chart(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test comprehensive market chart."""
        # Add spread column
        sample_ohlcv_data["spread"] = np.random.rand(len(sample_ohlcv_data)) * 0.1

        chart = MarketChart(data=sample_ohlcv_data, title="Market Analysis")
        chart.add_candlesticks()
        chart.add_volume()
        chart.add_spread_overlay()

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_create_ohlc_chart(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test convenience function for OHLC chart."""
        chart = create_ohlc_chart(sample_ohlcv_data, title="Test", show_volume=True)
        assert isinstance(chart, CandlestickChart)

    def test_create_market_chart(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test convenience function for market chart."""
        chart = create_market_chart(sample_ohlcv_data, title="Test")
        assert isinstance(chart, MarketChart)

    def test_price_line_chart(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test price line chart for multiple symbols."""
        # Create multi-symbol data
        data = sample_ohlcv_data.copy()
        data["SPY"] = data["close"]
        data["QQQ"] = data["close"] * 1.1

        chart = PriceLineChart(
            data=data,
            date_col="bar_start",
            price_cols=["SPY", "QQQ"],
            normalize=True,
        )

        fig = chart.figure
        assert len(fig.data) == 2

    def test_spread_chart(self, sample_quote_data: pd.DataFrame) -> None:
        """Test spread visualization chart."""
        chart = SpreadChart(
            data=sample_quote_data,
            date_col="bar_start",
            spread_col="spread",
            show_rolling_mean=True,
            rolling_window=10,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_quote_imbalance_heatmap(self, sample_quote_data: pd.DataFrame) -> None:
        """Test quote imbalance heatmap."""
        chart = QuoteImbalanceHeatmap(
            data=sample_quote_data,
            date_col="bar_start",
            imbalance_col="quote_imbalance",
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_trade_imbalance_chart(self) -> None:
        """Test trade imbalance chart."""
        n = 50
        data = pd.DataFrame(
            {
                "bar_start": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)],
                "buy_volume": np.random.randint(1000, 5000, n),
                "sell_volume": np.random.randint(1000, 5000, n),
            }
        )

        chart = TradeImbalanceChart(
            data=data,
            show_cumulative=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)


# =============================================================================
# Feature Chart Tests
# =============================================================================


class TestFeatureCharts:
    """Tests for feature visualization charts."""

    def test_feature_distribution_chart(self, sample_feature_data: pd.DataFrame) -> None:
        """Test feature distribution histograms."""
        chart = FeatureDistributionChart(
            data=sample_feature_data,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            bins=30,
            show_stats=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_feature_correlation_heatmap(self, sample_feature_data: pd.DataFrame) -> None:
        """Test feature correlation heatmap."""
        chart = FeatureCorrelationHeatmap(
            data=sample_feature_data,
            method="pearson",
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_feature_correlation_clustered(self, sample_feature_data: pd.DataFrame) -> None:
        """Test clustered correlation heatmap."""
        chart = FeatureCorrelationHeatmap(
            data=sample_feature_data,
            cluster=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_feature_time_series_chart(self, sample_time_series_data: pd.DataFrame) -> None:
        """Test feature time series visualization."""
        chart = FeatureTimeSeriesChart(
            data=sample_time_series_data,
            date_col="timestamp",
            feature_cols=["momentum", "volatility", "trend"],
            normalize=True,
        )

        fig = chart.figure
        assert len(fig.data) == 3

    def test_feature_time_series_stacked(self, sample_time_series_data: pd.DataFrame) -> None:
        """Test stacked feature time series."""
        chart = FeatureTimeSeriesChart(
            data=sample_time_series_data,
            date_col="timestamp",
            feature_cols=["momentum", "volatility"],
            stacked=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_rolling_statistics_chart(self, sample_time_series_data: pd.DataFrame) -> None:
        """Test rolling statistics visualization."""
        chart = RollingStatisticsChart(
            data=sample_time_series_data,
            date_col="timestamp",
            value_col="volatility",
            window=10,
            show_percentiles=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_feature_importance_chart(self) -> None:
        """Test feature importance bar chart."""
        features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importance = [0.25, 0.20, 0.18, 0.22, 0.15]
        errors = [0.02, 0.03, 0.01, 0.02, 0.01]

        chart = FeatureImportanceChart(
            feature_names=features,
            importance_values=importance,
            errors=errors,
            horizontal=True,
            sort=True,
        )

        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_feature_importance_top_n(self) -> None:
        """Test feature importance with top N filter."""
        features = ["f1", "f2", "f3", "f4", "f5"]
        importance = [0.1, 0.3, 0.2, 0.25, 0.15]

        chart = FeatureImportanceChart(
            feature_names=features,
            importance_values=importance,
            top_n=3,
        )

        fig = chart.figure
        # Should only show top 3 bars
        assert len(fig.data[0].y) == 3

    def test_create_feature_distribution(self, sample_feature_data: pd.DataFrame) -> None:
        """Test convenience function for distribution chart."""
        chart = create_feature_distribution(
            sample_feature_data,
            feature_cols=["feature_1", "feature_2"],
        )
        assert isinstance(chart, FeatureDistributionChart)

    def test_create_correlation_heatmap(self, sample_feature_data: pd.DataFrame) -> None:
        """Test convenience function for correlation heatmap."""
        chart = create_correlation_heatmap(sample_feature_data)
        assert isinstance(chart, FeatureCorrelationHeatmap)

    def test_create_feature_importance(self) -> None:
        """Test convenience function for importance chart."""
        chart = create_feature_importance(
            feature_names=["a", "b", "c"],
            importance_values=[0.3, 0.5, 0.2],
        )
        assert isinstance(chart, FeatureImportanceChart)


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for export functionality."""

    def test_save_html(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test saving chart to HTML."""
        chart = CandlestickChart(data=sample_ohlcv_data)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.html"
            result = save_html(chart, path)

            assert result.exists()
            assert result.stat().st_size > 0

            content = path.read_text()
            assert "<html>" in content.lower()

    def test_save_html_cdn(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test saving HTML with CDN plotly.js."""
        chart = CandlestickChart(data=sample_ohlcv_data)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_cdn.html"
            save_html(chart, path, include_plotlyjs="cdn")

            content = path.read_text()
            assert "cdn.plot.ly" in content

    def test_to_image_bytes(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test converting chart to image bytes."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        img_bytes = to_image_bytes(chart, format="png")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0

    def test_to_base64(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test converting chart to base64."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        b64 = to_base64(chart, format="png")
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_save_png(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test saving chart to PNG."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            result = save_png(chart, path)

            assert result.exists()
            assert result.stat().st_size > 0


# =============================================================================
# Multi-Chart Report Tests
# =============================================================================


class TestMultiChartReport:
    """Tests for multi-chart report functionality."""

    def test_create_report(self) -> None:
        """Test creating empty report."""
        report = create_report(title="Test Report", description="Test description")
        assert isinstance(report, MultiChartReport)
        assert report.title == "Test Report"

    def test_add_chart_to_report(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test adding chart to report."""
        report = create_report(title="Test Report")

        chart = CandlestickChart(data=sample_ohlcv_data)
        report.add_chart(chart, title="Price Chart", description="OHLCV data")

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "chart"

    def test_add_text_to_report(self) -> None:
        """Test adding text section to report."""
        report = create_report(title="Test Report")
        report.add_text("This is test content", title="Introduction")

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "text"

    def test_add_table_to_report(self) -> None:
        """Test adding table to report."""
        report = create_report(title="Test Report")
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report.add_table(data, title="Test Table")

        assert len(report.sections) == 1
        assert report.sections[0]["type"] == "table"

    def test_report_to_html(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test generating HTML from report."""
        report = create_report(title="Full Report")

        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        report.add_text("Introduction", title="Intro")
        report.add_chart(chart, title="Price Analysis")
        report.add_table({"Metric": ["A", "B"], "Value": [1, 2]}, title="Summary")

        html = report.to_html()
        assert "<html>" in html.lower()
        assert "Full Report" in html
        assert "Introduction" in html

    def test_report_save(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test saving report to file."""
        report = create_report(title="Test Report")

        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )
        report.add_chart(chart, title="Chart")

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            result = report.save(path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_report_chaining(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test method chaining on report."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        report = (
            create_report(title="Chained Report")
            .add_text("Intro")
            .add_chart(chart, title="Chart")
            .add_text("Conclusion")
        )

        assert len(report.sections) == 3


# =============================================================================
# Matplotlib Integration Tests
# =============================================================================


class TestMatplotlibIntegration:
    """Tests for matplotlib integration."""

    def test_create_matplotlib_figure(self) -> None:
        """Test creating matplotlib figure with theme."""
        fig, axes = create_matplotlib_figure(rows=2, cols=2)
        assert fig is not None
        assert axes is not None

    def test_matplotlib_figure_theme_applied(self) -> None:
        """Test that theme is applied to matplotlib figure."""
        set_theme("dark")
        fig, ax = create_matplotlib_figure()

        # Check that dark theme background is applied
        assert fig.get_facecolor() != (1.0, 1.0, 1.0, 1.0)

        # Reset to light
        set_theme("light")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        chart = SimpleLineChart(
            data=empty_df,
            x_col="timestamp",
            y_cols=["value"],
        )
        # Should not raise, just return empty figure
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_missing_columns(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test handling of missing columns."""
        chart = FeatureTimeSeriesChart(
            data=sample_ohlcv_data,
            date_col="bar_start",
            feature_cols=["nonexistent_column", "close"],
        )
        # Should not raise, just skip missing columns
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_none_data(self) -> None:
        """Test handling of None data."""
        chart = CandlestickChart(data=None)  # type: ignore
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_single_data_point(self) -> None:
        """Test handling of single data point."""
        data = pd.DataFrame(
            {
                "bar_start": [datetime.now()],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        chart = CandlestickChart(data=data, show_volume=True)
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_nan_values(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test handling of NaN values."""
        data = sample_ohlcv_data.copy()
        data.loc[10:20, "close"] = np.nan

        chart = SimpleLineChart(
            data=data,
            x_col="bar_start",
            y_cols=["close"],
        )
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_correlation_with_single_column(self) -> None:
        """Test correlation heatmap with single column."""
        data = pd.DataFrame({"single_col": np.random.randn(100)})

        chart = FeatureCorrelationHeatmap(
            data=data,
            feature_cols=["single_col"],
        )
        fig = chart.figure
        assert isinstance(fig, go.Figure)

    def test_update_layout_chaining(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that update_layout returns self for chaining."""
        chart = SimpleLineChart(
            data=sample_ohlcv_data,
            x_col="bar_start",
            y_cols=["close"],
        )

        result = chart.update_layout(title="Updated Title")
        assert result is chart
        assert chart.figure.layout.title.text == "Updated Title"
