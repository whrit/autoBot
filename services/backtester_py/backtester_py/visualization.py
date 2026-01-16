"""
Visualization module for backtester_py service.

Provides equity curve generation, trade summary charts, and HTML report export.
Uses Rich for console visualization and generates standalone HTML reports.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    import polars as pl

    from backtester_py.engine import BacktestResult, Fill


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""

    output_dir: Path = Path("reports")
    chart_width: int = 80
    chart_height: int = 20
    include_trades: bool = True
    include_metrics: bool = True
    include_equity_chart: bool = True


class ASCIIChart:
    """
    Simple ASCII chart generator for console output.

    Generates basic line charts using ASCII characters for
    displaying equity curves in the terminal.
    """

    CHARS = {
        "full": "\u2588",
        "upper_half": "\u2580",
        "lower_half": "\u2584",
        "empty": " ",
        "line": "\u2500",
        "vline": "\u2502",
        "corner_bl": "\u2514",
        "corner_tl": "\u250c",
        "corner_br": "\u2518",
        "corner_tr": "\u2510",
    }

    def __init__(self, width: int = 60, height: int = 15) -> None:
        """
        Initialize ASCII chart.

        Args:
            width: Chart width in characters
            height: Chart height in lines
        """
        self.width = width
        self.height = height

    def render(
        self,
        values: list[float],
        title: str = "",
        y_label: str = "",
    ) -> str:
        """
        Render a line chart as ASCII art.

        Args:
            values: List of numeric values to plot
            title: Chart title
            y_label: Y-axis label

        Returns:
            String containing the ASCII chart
        """
        if not values:
            return "No data to display"

        # Normalize values to chart height
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1.0

        # Sample values if too many for width
        if len(values) > self.width:
            step = len(values) / self.width
            sampled = [values[int(i * step)] for i in range(self.width)]
        else:
            sampled = values

        # Normalize to height
        normalized = [
            int((v - min_val) / value_range * (self.height - 1))
            for v in sampled
        ]

        # Build chart
        lines = []

        # Title
        if title:
            lines.append(title.center(self.width + 10))
            lines.append("")

        # Y-axis labels
        y_labels = [
            f"{max_val:>10.2f}",
            f"{(max_val + min_val) / 2:>10.2f}",
            f"{min_val:>10.2f}",
        ]

        # Chart body
        for row in range(self.height - 1, -1, -1):
            # Y-axis label
            if row == self.height - 1:
                line = y_labels[0] + " " + self.CHARS["vline"]
            elif row == self.height // 2:
                line = y_labels[1] + " " + self.CHARS["vline"]
            elif row == 0:
                line = y_labels[2] + " " + self.CHARS["vline"]
            else:
                line = " " * 10 + " " + self.CHARS["vline"]

            # Chart content
            for col, val in enumerate(normalized):
                if val == row:
                    line += "*"
                elif val > row:
                    line += self.CHARS["vline"]
                else:
                    line += " "

            lines.append(line)

        # X-axis
        x_axis = " " * 11 + self.CHARS["corner_bl"] + self.CHARS["line"] * len(normalized)
        lines.append(x_axis)

        return "\n".join(lines)


def generate_equity_chart(
    equity_curve: pl.DataFrame,
    console: Console,
    width: int = 60,
    height: int = 15,
) -> None:
    """
    Generate and display an equity curve chart.

    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns
        console: Rich console for output
        width: Chart width
        height: Chart height
    """
    if equity_curve.is_empty():
        console.print("[dim]No equity data to display[/dim]")
        return

    values = equity_curve["equity"].to_list()
    chart = ASCIIChart(width=width, height=height)

    chart_text = chart.render(
        values,
        title="Equity Curve",
        y_label="Equity ($)",
    )

    panel = Panel(
        chart_text,
        title="[header]Equity Curve[/header]",
        border_style="cyan",
    )
    console.print(panel)


def generate_trade_summary_table(
    fills: list[Fill],
    console: Console,
    max_rows: int = 20,
) -> None:
    """
    Generate and display a trade summary table.

    Args:
        fills: List of fill objects
        console: Rich console for output
        max_rows: Maximum rows to display
    """
    if not fills:
        console.print("[dim]No trades to display[/dim]")
        return

    table = Table(
        title="Trade Summary",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Time", style="dim")
    table.add_column("Symbol")
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Notional", justify="right")
    table.add_column("Slippage (bps)", justify="right")
    table.add_column("Commission", justify="right")

    display_fills = fills[-max_rows:] if len(fills) > max_rows else fills

    for fill in display_fills:
        side_style = "trade_buy" if fill.side == "buy" else "trade_sell"

        table.add_row(
            fill.timestamp.strftime("%Y-%m-%d %H:%M"),
            fill.symbol,
            Text(fill.side.upper(), style=side_style),
            f"{fill.quantity:,.2f}",
            f"${fill.price:,.2f}",
            f"${fill.notional:,.0f}",
            f"{fill.slippage_bps:.1f}",
            f"${fill.commission:.2f}",
        )

    if len(fills) > max_rows:
        table.add_row(
            f"... {len(fills) - max_rows} more trades ...",
            "", "", "", "", "", "", "",
        )

    console.print(table)


def generate_html_report(
    result: BacktestResult,
    metrics: dict[str, float],
    strategy_name: str,
    output_path: Path | None = None,
) -> str:
    """
    Generate an HTML report for backtest results.

    Args:
        result: BacktestResult object
        metrics: Dictionary of computed metrics
        strategy_name: Name of the strategy
        output_path: Optional path to save the report

    Returns:
        HTML content as string
    """
    # Prepare equity curve data for chart
    equity_data = []
    if not result.equity_curve.is_empty():
        timestamps = result.equity_curve["timestamp"].to_list()
        equities = result.equity_curve["equity"].to_list()
        equity_data = [
            {"x": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), "y": eq}
            for ts, eq in zip(timestamps, equities, strict=True)
        ]

    # Prepare trades data
    trades_data = [
        {
            "timestamp": f.timestamp.isoformat(),
            "symbol": f.symbol,
            "side": f.side,
            "quantity": f.quantity,
            "price": f.price,
            "notional": f.notional,
            "slippage_bps": f.slippage_bps,
            "commission": f.commission,
        }
        for f in result.fills[-100:]  # Limit to last 100 trades
    ]

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {html.escape(strategy_name)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #0f3460;
            --profit: #00d26a;
            --loss: #ff5252;
            --border: #2a2a4e;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: var(--bg-secondary);
            border-radius: 12px;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .metric-card {{
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border);
        }}

        .metric-value {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}

        .metric-value.profit {{
            color: var(--profit);
        }}

        .metric-value.loss {{
            color: var(--loss);
        }}

        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .chart-container {{
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}

        .chart-container h2 {{
            margin-bottom: 1rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--accent);
            font-weight: 600;
        }}

        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .buy {{ color: var(--profit); }}
        .sell {{ color: var(--loss); }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Backtest Report</h1>
            <p class="subtitle">Strategy: {html.escape(strategy_name)}</p>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'profit' if metrics.get('total_return', 0) >= 0 else 'loss'}">
                    {metrics.get('total_return', 0) * 100:.2f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('sharpe_ratio', 0):.2f}
                </div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value loss">
                    {metrics.get('max_drawdown', 0) * 100:.1f}%
                </div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('win_rate', 0) * 100:.1f}%
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {result.total_trades:,}
                </div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('profit_factor', 0):.2f}
                </div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Equity Curve</h2>
            <canvas id="equityChart" height="300"></canvas>
        </div>

        <div class="chart-container">
            <h2>Recent Trades</h2>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Notional</th>
                        <th>Slippage (bps)</th>
                        <th>Commission</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{t["timestamp"]}</td>
                        <td>{html.escape(t["symbol"])}</td>
                        <td class="{t["side"]}">{t["side"].upper()}</td>
                        <td>{t["quantity"]:,.2f}</td>
                        <td>${t["price"]:,.2f}</td>
                        <td>${t["notional"]:,.0f}</td>
                        <td>{t["slippage_bps"]:.1f}</td>
                        <td>${t["commission"]:.2f}</td>
                    </tr>
                    ''' for t in trades_data)}
                </tbody>
            </table>
        </div>

        <footer>
            <p>Generated by autoBot Backtester</p>
        </footer>
    </div>

    <script>
        const equityData = {json.dumps(equity_data)};

        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: equityData.map(d => d.x),
                datasets: [{{
                    label: 'Equity',
                    data: equityData.map(d => d.y),
                    borderColor: '#0f3460',
                    backgroundColor: 'rgba(15, 52, 96, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        ticks: {{
                            maxTicksLimit: 10,
                            color: '#aaa'
                        }},
                        grid: {{
                            color: '#2a2a4e'
                        }}
                    }},
                    y: {{
                        display: true,
                        ticks: {{
                            color: '#aaa'
                        }},
                        grid: {{
                            color: '#2a2a4e'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)

    return html_content


__all__ = [
    "VisualizationConfig",
    "ASCIIChart",
    "generate_equity_chart",
    "generate_trade_summary_table",
    "generate_html_report",
]
