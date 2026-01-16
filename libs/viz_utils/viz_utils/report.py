"""
Backtest Report Generation Module.

Provides comprehensive HTML and PDF report generation for backtest results:
- Combines multiple charts into a single report
- Includes summary statistics tables
- Supports export to HTML and PDF formats
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jinja2 import Template

from .backtest import (
    BacktestData,
    EquityCurveChart,
    PerformanceCharts,
    RiskVisualization,
    TradeAnalysisCharts,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


# HTML Template for the report
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary-color: #2196F3;
            --success-color: #4CAF50;
            --danger-color: #F44336;
            --warning-color: #FF9800;
            --text-color: #333;
            --bg-color: #f5f5f5;
            --card-bg: #fff;
            --border-color: #e0e0e0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), #1976D2);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }

        header .subtitle {
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: var(--card-bg);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid var(--primary-color);
        }

        .metric-card.positive {
            border-left-color: var(--success-color);
        }

        .metric-card.negative {
            border-left-color: var(--danger-color);
        }

        .metric-card .label {
            font-size: 0.85rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .metric-card .value.positive {
            color: var(--success-color);
        }

        .metric-card .value.negative {
            color: var(--danger-color);
        }

        .section {
            background: var(--card-bg);
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            overflow: hidden;
        }

        .section-header {
            background: #fafafa;
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .section-header h2 {
            font-size: 1.2rem;
            color: var(--text-color);
        }

        .section-content {
            padding: 20px;
        }

        .chart-container {
            width: 100%;
            min-height: 400px;
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
        }

        .stats-table th,
        .stats-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .stats-table th {
            background: #fafafa;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }

        .stats-table tr:hover {
            background: #fafafa;
        }

        .two-column {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }

        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            header {
                padding: 20px;
            }

            header h1 {
                font-size: 1.5rem;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .two-column {
                grid-template-columns: 1fr;
            }
        }

        @media print {
            body {
                background: white;
            }

            .section {
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid var(--border-color);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <div class="subtitle">
                {{ start_date }} to {{ end_date }} | Generated: {{ generated_date }}
            </div>
        </header>

        <div class="metrics-grid">
            {% for metric in key_metrics %}
            <div class="metric-card {{ 'positive' if metric.positive else 'negative' if metric.negative else '' }}">
                <div class="label">{{ metric.label }}</div>
                <div class="value {{ 'positive' if metric.positive else 'negative' if metric.negative else '' }}">
                    {{ metric.value }}
                </div>
            </div>
            {% endfor %}
        </div>

        {% for section in sections %}
        <div class="section">
            <div class="section-header">
                <h2>{{ section.title }}</h2>
            </div>
            <div class="section-content">
                {% if section.type == 'chart' %}
                <div class="chart-container" id="{{ section.id }}"></div>
                <script>
                    var data = {{ section.chart_json | safe }};
                    Plotly.newPlot('{{ section.id }}', data.data, data.layout, {responsive: true});
                </script>
                {% elif section.type == 'table' %}
                <table class="stats-table">
                    <thead>
                        <tr>
                            {% for header in section.headers %}
                            <th>{{ header }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in section.rows %}
                        <tr>
                            {% for cell in row %}
                            <td>{{ cell }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% elif section.type == 'two_charts' %}
                <div class="two-column">
                    <div class="chart-container" id="{{ section.chart1_id }}"></div>
                    <div class="chart-container" id="{{ section.chart2_id }}"></div>
                </div>
                <script>
                    var data1 = {{ section.chart1_json | safe }};
                    var data2 = {{ section.chart2_json | safe }};
                    Plotly.newPlot('{{ section.chart1_id }}', data1.data, data1.layout, {responsive: true});
                    Plotly.newPlot('{{ section.chart2_id }}', data2.data, data2.layout, {responsive: true});
                </script>
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <footer>
            <p>Generated by AutoBot Trading Engine | &copy; {{ year }}</p>
        </footer>
    </div>
</body>
</html>
"""


@dataclass
class MetricDisplay:
    """Display configuration for a metric."""

    label: str
    value: str
    positive: bool = False
    negative: bool = False


@dataclass
class SectionConfig:
    """Configuration for a report section."""

    title: str
    type: str
    id: str = ""
    chart_json: str = ""
    chart1_id: str = ""
    chart1_json: str = ""
    chart2_id: str = ""
    chart2_json: str = ""
    headers: list[str] | None = None
    rows: list[list[str]] | None = None


@dataclass
class BacktestMetrics:
    """Computed backtest metrics for display.

    Attributes:
        total_return: Total return percentage
        cagr: Compound annual growth rate
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Annualized Sortino ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / gross loss
        num_trades: Total number of trades
        avg_trade_pnl: Average P&L per trade
        avg_win: Average winning trade
        avg_loss: Average losing trade
        volatility: Annualized volatility
        calmar_ratio: CAGR / max drawdown
    """

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    volatility: float
    calmar_ratio: float

    @classmethod
    def from_backtest_data(
        cls,
        data: BacktestData,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> BacktestMetrics:
        """Calculate metrics from backtest data.

        Args:
            data: Backtest data container
            risk_free_rate: Annualized risk-free rate
            periods_per_year: Number of trading periods per year

        Returns:
            BacktestMetrics instance
        """
        returns_arr = np.array(data.returns, dtype=np.float64)
        returns_arr = returns_arr[~np.isnan(returns_arr)]

        equity_arr = np.array(data.equity, dtype=np.float64)

        # Total return
        total_return = (equity_arr[-1] - equity_arr[0]) / equity_arr[0] if len(equity_arr) > 1 else 0.0

        # CAGR
        n_periods = len(equity_arr) - 1
        years = n_periods / periods_per_year if n_periods > 0 else 1
        cagr = (equity_arr[-1] / equity_arr[0]) ** (1 / years) - 1 if equity_arr[0] > 0 and years > 0 else 0.0

        # Volatility
        volatility = np.std(returns_arr, ddof=1) * np.sqrt(periods_per_year) if len(returns_arr) > 1 else 0.0

        # Sharpe ratio
        rf_per_period = risk_free_rate / periods_per_year
        std = np.std(returns_arr, ddof=1)
        sharpe_ratio = (np.mean(returns_arr) - rf_per_period) / std * np.sqrt(periods_per_year) if std > 0 else 0.0

        # Sortino ratio
        downside_returns = returns_arr[returns_arr < rf_per_period]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
        sortino_ratio = (
            (np.mean(returns_arr) - rf_per_period) / downside_std * np.sqrt(periods_per_year)
            if downside_std > 0
            else 0.0
        )

        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (running_max - equity_arr) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Trade metrics
        num_trades = len(data.trades)
        pnls = [t.pnl for t in data.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        win_rate = len(winners) / num_trades if num_trades > 0 else 0.0
        avg_trade_pnl = np.mean(pnls) if pnls else 0.0
        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = abs(np.mean(losers)) if losers else 0.0

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Calmar ratio
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

        return cls(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
        )


class BacktestReport:
    """Comprehensive backtest report generator.

    Combines multiple chart types and statistics into a single HTML/PDF report.

    Example:
        >>> from viz_utils.backtest import BacktestData
        >>> from viz_utils.report import BacktestReport
        >>>
        >>> data = BacktestData.from_polars(equity_curve, trades)
        >>> report = BacktestReport(data, title="My Strategy")
        >>> report.add_equity_curve()
        >>> report.add_monthly_returns()
        >>> report.add_trade_analysis()
        >>> report.add_drawdown_chart()
        >>> report.save("backtest_report.html")
    """

    def __init__(
        self,
        data: BacktestData,
        title: str = "Backtest Report",
        risk_free_rate: float = 0.0,
    ) -> None:
        """Initialize the backtest report.

        Args:
            data: Backtest data container
            title: Report title
            risk_free_rate: Annualized risk-free rate for metrics calculation
        """
        self.data = data
        self.title = title
        self.risk_free_rate = risk_free_rate
        self.sections: list[SectionConfig] = []
        self._section_counter = 0

        # Initialize chart generators
        self._equity_chart = EquityCurveChart(data)
        self._perf_charts = PerformanceCharts(data)
        self._trade_charts = TradeAnalysisCharts(data)
        self._risk_charts = RiskVisualization(data)

        # Calculate metrics
        self.metrics = BacktestMetrics.from_backtest_data(data, risk_free_rate)

    def add_equity_curve(
        self,
        show_benchmark: bool = True,
        show_drawdown: bool = True,
        show_trades: bool = True,
        title: str = "Equity Curve",
    ) -> BacktestReport:
        """Add equity curve chart to the report.

        Args:
            show_benchmark: Whether to show benchmark overlay
            show_drawdown: Whether to show drawdown fill chart
            show_trades: Whether to show trade markers
            title: Section title

        Returns:
            Self for method chaining
        """
        fig = self._equity_chart.create(
            show_benchmark=show_benchmark,
            show_drawdown=show_drawdown,
            show_trades=show_trades,
            title="",
        )
        self._add_chart_section(title, fig)
        return self

    def add_monthly_returns(
        self,
        title: str = "Monthly Returns",
    ) -> BacktestReport:
        """Add monthly returns heatmap to the report.

        Args:
            title: Section title

        Returns:
            Self for method chaining
        """
        fig = self._perf_charts.monthly_returns_heatmap(title="")
        self._add_chart_section(title, fig)
        return self

    def add_rolling_metrics(
        self,
        sharpe_window: int = 63,
        vol_window: int = 21,
        title: str = "Rolling Performance Metrics",
    ) -> BacktestReport:
        """Add rolling Sharpe and volatility charts.

        Args:
            sharpe_window: Window for rolling Sharpe calculation
            vol_window: Window for rolling volatility calculation
            title: Section title

        Returns:
            Self for method chaining
        """
        sharpe_fig = self._perf_charts.rolling_sharpe(
            window=sharpe_window,
            title="Rolling Sharpe Ratio",
            risk_free_rate=self.risk_free_rate,
        )
        vol_fig = self._perf_charts.rolling_volatility(
            window=vol_window,
            title="Rolling Volatility",
        )
        self._add_two_charts_section(title, sharpe_fig, vol_fig)
        return self

    def add_trade_analysis(
        self,
        title: str = "Trade Analysis",
    ) -> BacktestReport:
        """Add trade analysis charts (win/loss distribution and P&L by symbol).

        Args:
            title: Section title

        Returns:
            Self for method chaining
        """
        dist_fig = self._trade_charts.win_loss_distribution(title="Win/Loss Distribution")
        pnl_fig = self._trade_charts.pnl_by_symbol(title="P&L by Symbol")
        self._add_two_charts_section(title, dist_fig, pnl_fig)
        return self

    def add_trade_duration(
        self,
        title: str = "Trade Duration Analysis",
    ) -> BacktestReport:
        """Add trade duration and return scatter charts.

        Args:
            title: Section title

        Returns:
            Self for method chaining
        """
        duration_fig = self._trade_charts.trade_duration_histogram(title="Trade Duration")
        scatter_fig = self._trade_charts.return_vs_duration_scatter(title="Return vs Duration")
        self._add_two_charts_section(title, duration_fig, scatter_fig)
        return self

    def add_drawdown_chart(
        self,
        min_drawdown: float = 0.05,
        title: str = "Drawdown Analysis",
    ) -> BacktestReport:
        """Add drawdown periods timeline.

        Args:
            min_drawdown: Minimum drawdown to highlight
            title: Section title

        Returns:
            Self for method chaining
        """
        fig = self._risk_charts.drawdown_periods_timeline(
            min_drawdown=min_drawdown,
            title="",
        )
        self._add_chart_section(title, fig)
        return self

    def add_var_analysis(
        self,
        confidence_levels: list[float] | None = None,
        title: str = "VaR/CVaR Analysis",
    ) -> BacktestReport:
        """Add VaR/CVaR visualization.

        Args:
            confidence_levels: List of confidence levels
            title: Section title

        Returns:
            Self for method chaining
        """
        fig = self._risk_charts.var_cvar_visualization(
            confidence_levels=confidence_levels,
            title="",
        )
        self._add_chart_section(title, fig)
        return self

    def add_statistics_table(
        self,
        title: str = "Performance Statistics",
    ) -> BacktestReport:
        """Add comprehensive statistics table.

        Args:
            title: Section title

        Returns:
            Self for method chaining
        """
        m = self.metrics

        headers = ["Metric", "Value", "Metric", "Value"]
        rows = [
            ["Total Return", f"{m.total_return:.2%}", "CAGR", f"{m.cagr:.2%}"],
            ["Sharpe Ratio", f"{m.sharpe_ratio:.2f}", "Sortino Ratio", f"{m.sortino_ratio:.2f}"],
            ["Max Drawdown", f"{m.max_drawdown:.2%}", "Volatility", f"{m.volatility:.2%}"],
            ["Win Rate", f"{m.win_rate:.1%}", "Profit Factor", f"{m.profit_factor:.2f}"],
            ["Num Trades", f"{m.num_trades:,}", "Avg Trade P&L", f"${m.avg_trade_pnl:,.0f}"],
            ["Avg Win", f"${m.avg_win:,.0f}", "Avg Loss", f"${m.avg_loss:,.0f}"],
            ["Calmar Ratio", f"{m.calmar_ratio:.2f}", "", ""],
        ]

        section = SectionConfig(
            title=title,
            type="table",
            id=f"table_{self._section_counter}",
            headers=headers,
            rows=rows,
        )
        self.sections.append(section)
        self._section_counter += 1

        return self

    def add_custom_chart(
        self,
        fig: go.Figure,
        title: str,
    ) -> BacktestReport:
        """Add a custom Plotly chart.

        Args:
            fig: Plotly Figure object
            title: Section title

        Returns:
            Self for method chaining
        """
        self._add_chart_section(title, fig)
        return self

    def add_regime_returns(
        self,
        regimes: list[tuple[datetime, datetime, str]],
        title: str = "Returns by Market Regime",
    ) -> BacktestReport:
        """Add cumulative returns by regime chart.

        Args:
            regimes: List of (start, end, regime_name) tuples
            title: Section title

        Returns:
            Self for method chaining
        """
        fig = self._perf_charts.cumulative_returns_by_regime(
            regimes=regimes,
            title="",
        )
        self._add_chart_section(title, fig)
        return self

    def _add_chart_section(self, title: str, fig: go.Figure) -> None:
        """Add a single chart section."""
        section = SectionConfig(
            title=title,
            type="chart",
            id=f"chart_{self._section_counter}",
            chart_json=fig.to_json(),
        )
        self.sections.append(section)
        self._section_counter += 1

    def _add_two_charts_section(
        self,
        title: str,
        fig1: go.Figure,
        fig2: go.Figure,
    ) -> None:
        """Add a two-column chart section."""
        section = SectionConfig(
            title=title,
            type="two_charts",
            chart1_id=f"chart_{self._section_counter}_1",
            chart1_json=fig1.to_json(),
            chart2_id=f"chart_{self._section_counter}_2",
            chart2_json=fig2.to_json(),
        )
        self.sections.append(section)
        self._section_counter += 1

    def _get_key_metrics(self) -> list[MetricDisplay]:
        """Get key metrics for display in header."""
        m = self.metrics

        return [
            MetricDisplay(
                label="Total Return",
                value=f"{m.total_return:.2%}",
                positive=m.total_return > 0,
                negative=m.total_return < 0,
            ),
            MetricDisplay(
                label="Sharpe Ratio",
                value=f"{m.sharpe_ratio:.2f}",
                positive=m.sharpe_ratio > 1,
            ),
            MetricDisplay(
                label="Max Drawdown",
                value=f"{m.max_drawdown:.2%}",
                negative=True,
            ),
            MetricDisplay(
                label="Win Rate",
                value=f"{m.win_rate:.1%}",
                positive=m.win_rate > 0.5,
            ),
            MetricDisplay(
                label="Profit Factor",
                value=f"{m.profit_factor:.2f}",
                positive=m.profit_factor > 1,
            ),
            MetricDisplay(
                label="Total Trades",
                value=f"{m.num_trades:,}",
            ),
        ]

    def render_html(self) -> str:
        """Render the report as HTML string.

        Returns:
            HTML string
        """
        template = Template(REPORT_TEMPLATE)

        # Get date range
        start_date = (
            self.data.timestamps[0].strftime("%Y-%m-%d")
            if isinstance(self.data.timestamps[0], datetime)
            else str(self.data.timestamps[0])
        )
        end_date = (
            self.data.timestamps[-1].strftime("%Y-%m-%d")
            if isinstance(self.data.timestamps[-1], datetime)
            else str(self.data.timestamps[-1])
        )

        # Convert MetricDisplay to dict for template
        key_metrics = [
            {
                "label": m.label,
                "value": m.value,
                "positive": m.positive,
                "negative": m.negative,
            }
            for m in self._get_key_metrics()
        ]

        # Convert sections to dict for template
        sections = []
        for s in self.sections:
            section_dict: dict[str, Any] = {
                "title": s.title,
                "type": s.type,
            }
            if s.type == "chart":
                section_dict["id"] = s.id
                section_dict["chart_json"] = s.chart_json
            elif s.type == "table":
                section_dict["headers"] = s.headers
                section_dict["rows"] = s.rows
            elif s.type == "two_charts":
                section_dict["chart1_id"] = s.chart1_id
                section_dict["chart1_json"] = s.chart1_json
                section_dict["chart2_id"] = s.chart2_id
                section_dict["chart2_json"] = s.chart2_json
            sections.append(section_dict)

        html = template.render(
            title=self.title,
            start_date=start_date,
            end_date=end_date,
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            year=datetime.now().year,
            key_metrics=key_metrics,
            sections=sections,
        )

        return html

    def save(self, path: str | Path) -> None:
        """Save the report to a file.

        Args:
            path: Output file path (.html or .pdf)
        """
        path = Path(path)
        html = self.render_html()

        if path.suffix.lower() == ".pdf":
            self._save_pdf(path, html)
        else:
            # Save as HTML
            path.write_text(html, encoding="utf-8")

    def _save_pdf(self, path: Path, html: str) -> None:
        """Save the report as PDF using WeasyPrint.

        Args:
            path: Output PDF path
            html: HTML content
        """
        try:
            from weasyprint import HTML as WeasyHTML
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF export. "
                "Install it with: pip install weasyprint"
            )

        # WeasyPrint needs absolute paths for resources
        WeasyHTML(string=html).write_pdf(path)

    def to_dict(self) -> dict[str, Any]:
        """Export report data as a dictionary.

        Returns:
            Dictionary containing all report data
        """
        return {
            "title": self.title,
            "data": {
                "timestamps": [
                    ts.isoformat() if isinstance(ts, datetime) else str(ts)
                    for ts in self.data.timestamps
                ],
                "equity": self.data.equity,
                "returns": self.data.returns,
                "trades": [
                    {
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "symbol": t.symbol,
                        "side": t.side,
                        "pnl": t.pnl,
                        "return_pct": t.return_pct,
                        "notional": t.notional,
                    }
                    for t in self.data.trades
                ],
            },
            "metrics": {
                "total_return": self.metrics.total_return,
                "cagr": self.metrics.cagr,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "num_trades": self.metrics.num_trades,
                "avg_trade_pnl": self.metrics.avg_trade_pnl,
                "avg_win": self.metrics.avg_win,
                "avg_loss": self.metrics.avg_loss,
                "volatility": self.metrics.volatility,
                "calmar_ratio": self.metrics.calmar_ratio,
            },
        }


# Export all classes
__all__ = [
    "MetricDisplay",
    "SectionConfig",
    "BacktestMetrics",
    "BacktestReport",
]
