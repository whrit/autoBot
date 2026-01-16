"""
Visualization Utilities Library for Autonomous Trading Engine.

This module provides comprehensive visualization tools including:

## Backtest Visualization
- Equity curve charts with benchmark comparison and drawdown
- Performance charts (monthly returns heatmap, rolling Sharpe, volatility)
- Trade analysis charts (win/loss distribution, P&L by symbol)
- Risk visualization (VaR/CVaR, drawdown periods, exposure heatmaps)
- Full HTML/PDF report generation

## Market Data Visualization
- Candlestick/OHLC charts with volume
- Price line charts with multiple symbols
- Spread visualization over time
- Quote imbalance heatmaps
- Trade imbalance visualization

## Feature Visualization
- Feature distribution histograms
- Feature correlation heatmaps
- Time series of features
- Rolling statistics plots
- Feature importance bar charts

## Export Utilities
- Save to HTML (interactive)
- Save to PNG/SVG (static)
- Multi-chart reports
- Notebook-friendly display

Example usage for backtest:
    >>> from viz_utils import BacktestReport, BacktestData
    >>>
    >>> data = BacktestData.from_polars(equity_curve, trades)
    >>> report = BacktestReport(data, title="My Strategy Backtest")
    >>> report.add_equity_curve()
    >>> report.save("backtest_report.html")

Example usage for market data:
    >>> from viz_utils import MarketChart, save_html, use_dark_mode
    >>>
    >>> use_dark_mode(True)
    >>> chart = MarketChart(df)
    >>> chart.add_candlesticks()
    >>> chart.add_volume()
    >>> chart.add_spread_overlay()
    >>> save_html(chart, "spy_analysis.html")

Example usage for features:
    >>> from viz_utils import create_correlation_heatmap, create_feature_importance
    >>>
    >>> heatmap = create_correlation_heatmap(feature_df)
    >>> heatmap.show()
    >>>
    >>> importance_chart = create_feature_importance(
    ...     feature_names=["vol", "momentum", "spread"],
    ...     importance_values=[0.4, 0.35, 0.25]
    ... )
    >>> importance_chart.show()
"""

# Backtest visualization (existing)
from .backtest import (
    BacktestData,
    EquityCurveChart,
    PerformanceCharts,
    RiskVisualization,
    TradeAnalysisCharts,
    TradeData,
)

# Core chart utilities (new)
from .charts import (
    BarChart,
    BaseChart,
    HeatmapChart,
    MultiPanelChart,
    SimpleLineChart,
    create_matplotlib_figure,
)

# Export utilities (new)
from .export import (
    MultiChartReport,
    create_report,
    display_inline,
    save_html,
    save_png,
    save_svg,
    to_base64,
    to_data_uri,
    to_image_bytes,
)

# Feature visualization (new)
from .features import (
    FeatureCorrelationHeatmap,
    FeatureDistributionChart,
    FeatureImportanceChart,
    FeatureTimeSeriesChart,
    RollingStatisticsChart,
    create_correlation_heatmap,
    create_feature_distribution,
    create_feature_importance,
    plot_correlation_matrix_matplotlib,
    plot_feature_distributions_matplotlib,
)

# Market data visualization (new)
from .market import (
    CandlestickChart,
    MarketChart,
    PriceLineChart,
    QuoteImbalanceHeatmap,
    SpreadChart,
    TradeImbalanceChart,
    create_market_chart,
    create_ohlc_chart,
)

# Report generation (existing)
from .report import (
    BacktestMetrics,
    BacktestReport,
    MetricDisplay,
    SectionConfig,
)

# Theme configuration (new)
from .themes import (
    DARK_THEME,
    LIGHT_THEME,
    ColorPalette,
    DarkColorPalette,
    ThemeConfig,
    get_theme,
    set_theme,
    use_dark_mode,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # ===== Backtest Visualization =====
    # Data containers
    "TradeData",
    "BacktestData",
    # Backtest chart classes
    "EquityCurveChart",
    "PerformanceCharts",
    "TradeAnalysisCharts",
    "RiskVisualization",
    # Backtest report generation
    "MetricDisplay",
    "SectionConfig",
    "BacktestMetrics",
    "BacktestReport",
    # ===== Theme Configuration =====
    "ColorPalette",
    "DarkColorPalette",
    "ThemeConfig",
    "LIGHT_THEME",
    "DARK_THEME",
    "get_theme",
    "set_theme",
    "use_dark_mode",
    # ===== Base Charts =====
    "BaseChart",
    "MultiPanelChart",
    "SimpleLineChart",
    "BarChart",
    "HeatmapChart",
    "create_matplotlib_figure",
    # ===== Market Data Charts =====
    "CandlestickChart",
    "MarketChart",
    "PriceLineChart",
    "SpreadChart",
    "QuoteImbalanceHeatmap",
    "TradeImbalanceChart",
    "create_ohlc_chart",
    "create_market_chart",
    # ===== Feature Charts =====
    "FeatureDistributionChart",
    "FeatureCorrelationHeatmap",
    "FeatureTimeSeriesChart",
    "RollingStatisticsChart",
    "FeatureImportanceChart",
    "create_feature_distribution",
    "create_correlation_heatmap",
    "create_feature_importance",
    "plot_feature_distributions_matplotlib",
    "plot_correlation_matrix_matplotlib",
    # ===== Export Utilities =====
    "save_html",
    "save_png",
    "save_svg",
    "to_image_bytes",
    "to_base64",
    "to_data_uri",
    "display_inline",
    "MultiChartReport",
    "create_report",
]
