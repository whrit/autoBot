"""
Backtest Visualization Module.

Provides comprehensive charting capabilities for backtest analysis including:
- Equity curve visualization with benchmark comparison
- Performance charts (monthly returns heatmap, rolling metrics)
- Trade analysis charts (distribution, duration, P&L by symbol)
- Risk visualization (drawdown, VaR/CVaR, exposure)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


@dataclass
class TradeData:
    """Trade data for visualization.

    Attributes:
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        symbol: Trading symbol
        side: Trade side (long or short)
        pnl: Profit/loss from the trade
        return_pct: Return percentage
        notional: Trade notional value
    """

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: Literal["long", "short"]
    pnl: float
    return_pct: float
    notional: float

    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.pnl > 0


@dataclass
class BacktestData:
    """Container for backtest visualization data.

    Attributes:
        timestamps: List of timestamps for equity curve
        equity: List of equity values
        returns: List of period returns
        benchmark_equity: Optional benchmark equity values
        benchmark_returns: Optional benchmark returns
        trades: List of TradeData objects
        initial_capital: Starting capital
    """

    timestamps: list[datetime]
    equity: list[float]
    returns: list[float]
    benchmark_equity: list[float] | None = None
    benchmark_returns: list[float] | None = None
    trades: list[TradeData] = field(default_factory=list)
    initial_capital: float = 100_000.0

    @classmethod
    def from_polars(
        cls,
        equity_curve: pl.DataFrame,
        trades: list[dict[str, Any]] | None = None,
        benchmark: pl.DataFrame | None = None,
        initial_capital: float = 100_000.0,
    ) -> BacktestData:
        """Create BacktestData from Polars DataFrames.

        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            trades: List of trade dictionaries
            benchmark: Optional benchmark DataFrame with 'timestamp' and 'equity'
            initial_capital: Starting capital

        Returns:
            BacktestData instance
        """
        timestamps = equity_curve["timestamp"].to_list()
        equity = equity_curve["equity"].to_list()

        # Calculate returns if not present
        if "returns" in equity_curve.columns:
            returns = equity_curve["returns"].to_list()
        else:
            equity_arr = np.array(equity, dtype=np.float64)
            returns_arr = np.diff(equity_arr) / equity_arr[:-1]
            returns = [0.0] + returns_arr.tolist()

        # Process benchmark
        benchmark_equity = None
        benchmark_returns = None
        if benchmark is not None:
            benchmark_equity = benchmark["equity"].to_list()
            if "returns" in benchmark.columns:
                benchmark_returns = benchmark["returns"].to_list()
            else:
                bench_arr = np.array(benchmark_equity, dtype=np.float64)
                bench_returns_arr = np.diff(bench_arr) / bench_arr[:-1]
                benchmark_returns = [0.0] + bench_returns_arr.tolist()

        # Process trades
        trade_list: list[TradeData] = []
        if trades:
            for t in trades:
                trade_list.append(
                    TradeData(
                        entry_time=t.get("entry_time", t.get("timestamp")),
                        exit_time=t.get("exit_time", t.get("timestamp")),
                        symbol=t.get("symbol", "UNKNOWN"),
                        side=t.get("side", "long"),
                        pnl=float(t.get("pnl", 0.0)),
                        return_pct=float(t.get("return_pct", 0.0)),
                        notional=float(t.get("notional", 0.0)),
                    )
                )

        return cls(
            timestamps=timestamps,
            equity=equity,
            returns=returns,
            benchmark_equity=benchmark_equity,
            benchmark_returns=benchmark_returns,
            trades=trade_list,
            initial_capital=initial_capital,
        )

    @classmethod
    def from_pandas(
        cls,
        equity_curve: pd.DataFrame,
        trades: list[dict[str, Any]] | None = None,
        benchmark: pd.DataFrame | None = None,
        initial_capital: float = 100_000.0,
    ) -> BacktestData:
        """Create BacktestData from Pandas DataFrames.

        Args:
            equity_curve: DataFrame with 'timestamp' (or index) and 'equity' columns
            trades: List of trade dictionaries
            benchmark: Optional benchmark DataFrame
            initial_capital: Starting capital

        Returns:
            BacktestData instance
        """
        # Handle timestamp column or index
        if "timestamp" in equity_curve.columns:
            timestamps = equity_curve["timestamp"].tolist()
        else:
            timestamps = equity_curve.index.tolist()

        equity = equity_curve["equity"].tolist()

        # Calculate returns
        if "returns" in equity_curve.columns:
            returns = equity_curve["returns"].tolist()
        else:
            equity_arr = np.array(equity, dtype=np.float64)
            returns_arr = np.diff(equity_arr) / equity_arr[:-1]
            returns = [0.0] + returns_arr.tolist()

        # Process benchmark
        benchmark_equity = None
        benchmark_returns = None
        if benchmark is not None:
            benchmark_equity = benchmark["equity"].tolist()
            if "returns" in benchmark.columns:
                benchmark_returns = benchmark["returns"].tolist()

        # Process trades
        trade_list: list[TradeData] = []
        if trades:
            for t in trades:
                trade_list.append(
                    TradeData(
                        entry_time=t.get("entry_time", t.get("timestamp")),
                        exit_time=t.get("exit_time", t.get("timestamp")),
                        symbol=t.get("symbol", "UNKNOWN"),
                        side=t.get("side", "long"),
                        pnl=float(t.get("pnl", 0.0)),
                        return_pct=float(t.get("return_pct", 0.0)),
                        notional=float(t.get("notional", 0.0)),
                    )
                )

        return cls(
            timestamps=timestamps,
            equity=equity,
            returns=returns,
            benchmark_equity=benchmark_equity,
            benchmark_returns=benchmark_returns,
            trades=trade_list,
            initial_capital=initial_capital,
        )


class EquityCurveChart:
    """Equity curve visualization with benchmark comparison and drawdown."""

    def __init__(self, data: BacktestData) -> None:
        """Initialize equity curve chart.

        Args:
            data: Backtest data container
        """
        self.data = data

    def create(
        self,
        show_benchmark: bool = True,
        show_drawdown: bool = True,
        show_trades: bool = True,
        height: int = 600,
        title: str = "Portfolio Equity Curve",
    ) -> go.Figure:
        """Create equity curve figure.

        Args:
            show_benchmark: Whether to show benchmark overlay
            show_drawdown: Whether to show drawdown fill chart below
            show_trades: Whether to show trade entry/exit markers
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Create subplots if showing drawdown
        if show_drawdown:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Drawdown"),
            )
        else:
            fig = go.Figure()

        # Main equity curve
        fig.add_trace(
            go.Scatter(
                x=self.data.timestamps,
                y=self.data.equity,
                mode="lines",
                name="Portfolio",
                line={"color": "#2196F3", "width": 2},
            ),
            row=1 if show_drawdown else None,
            col=1 if show_drawdown else None,
        )

        # Benchmark overlay
        if show_benchmark and self.data.benchmark_equity:
            # Normalize benchmark to same starting value
            bench_start = self.data.benchmark_equity[0]
            port_start = self.data.equity[0]
            normalized_bench = [
                b * port_start / bench_start for b in self.data.benchmark_equity
            ]

            fig.add_trace(
                go.Scatter(
                    x=self.data.timestamps,
                    y=normalized_bench,
                    mode="lines",
                    name="Benchmark",
                    line={"color": "#9E9E9E", "width": 1.5, "dash": "dash"},
                ),
                row=1 if show_drawdown else None,
                col=1 if show_drawdown else None,
            )

        # Trade markers
        if show_trades and self.data.trades:
            entry_times = [t.entry_time for t in self.data.trades]
            exit_times = [t.exit_time for t in self.data.trades]

            # Find corresponding equity values
            entry_values = self._get_equity_at_times(entry_times)
            exit_values = self._get_equity_at_times(exit_times)

            # Separate winners and losers
            winners = [t for t in self.data.trades if t.is_winner]
            losers = [t for t in self.data.trades if not t.is_winner]

            if winners:
                winner_entries = [t.entry_time for t in winners]
                winner_entry_vals = self._get_equity_at_times(winner_entries)
                fig.add_trace(
                    go.Scatter(
                        x=winner_entries,
                        y=winner_entry_vals,
                        mode="markers",
                        name="Winning Entry",
                        marker={"symbol": "triangle-up", "size": 10, "color": "#4CAF50"},
                    ),
                    row=1 if show_drawdown else None,
                    col=1 if show_drawdown else None,
                )

            if losers:
                loser_entries = [t.entry_time for t in losers]
                loser_entry_vals = self._get_equity_at_times(loser_entries)
                fig.add_trace(
                    go.Scatter(
                        x=loser_entries,
                        y=loser_entry_vals,
                        mode="markers",
                        name="Losing Entry",
                        marker={"symbol": "triangle-down", "size": 10, "color": "#F44336"},
                    ),
                    row=1 if show_drawdown else None,
                    col=1 if show_drawdown else None,
                )

        # Drawdown fill chart
        if show_drawdown:
            drawdown = self._calculate_drawdown()
            fig.add_trace(
                go.Scatter(
                    x=self.data.timestamps,
                    y=drawdown,
                    mode="lines",
                    fill="tozeroy",
                    name="Drawdown",
                    line={"color": "#F44336", "width": 1},
                    fillcolor="rgba(244, 67, 54, 0.3)",
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=height,
            showlegend=True,
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            hovermode="x unified",
            template="plotly_white",
        )

        if not show_drawdown:
            fig.update_layout(title=title)

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_yaxes(tickformat=".1%", row=2, col=1)

        return fig

    def _get_equity_at_times(self, times: list[datetime]) -> list[float]:
        """Get equity values at specific times."""
        values = []
        for t in times:
            # Find closest timestamp
            idx = min(
                range(len(self.data.timestamps)),
                key=lambda i: abs(
                    (self.data.timestamps[i] - t).total_seconds()
                    if isinstance(self.data.timestamps[i], datetime)
                    else float("inf")
                ),
            )
            values.append(self.data.equity[idx])
        return values

    def _calculate_drawdown(self) -> list[float]:
        """Calculate drawdown series."""
        equity_arr = np.array(self.data.equity, dtype=np.float64)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        return drawdown.tolist()


class PerformanceCharts:
    """Performance-related charts including monthly returns heatmap and rolling metrics."""

    def __init__(self, data: BacktestData) -> None:
        """Initialize performance charts.

        Args:
            data: Backtest data container
        """
        self.data = data

    def monthly_returns_heatmap(
        self,
        height: int = 400,
        title: str = "Monthly Returns Heatmap",
    ) -> go.Figure:
        """Create monthly returns heatmap calendar.

        Args:
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns()

        if not monthly_returns:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No monthly data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        # Get unique years and months
        years = sorted(set(yr for yr, _ in monthly_returns.keys()))
        months = list(range(1, 13))
        month_names = [calendar.month_abbr[m] for m in months]

        # Create matrix
        z = []
        text = []
        for year in years:
            row = []
            text_row = []
            for month in months:
                value = monthly_returns.get((year, month))
                if value is not None:
                    row.append(value * 100)  # Convert to percentage
                    text_row.append(f"{value * 100:.1f}%")
                else:
                    row.append(None)
                    text_row.append("")
            z.append(row)
            text.append(text_row)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=month_names,
                y=[str(y) for y in years],
                text=text,
                texttemplate="%{text}",
                colorscale=[
                    [0, "#F44336"],
                    [0.5, "#FFFFFF"],
                    [1, "#4CAF50"],
                ],
                zmid=0,
                colorbar={
                    "title": "Return %",
                    "tickformat": ".1f",
                },
            )
        )

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
        )

        return fig

    def rolling_sharpe(
        self,
        window: int = 63,
        height: int = 400,
        title: str = "Rolling Sharpe Ratio",
        risk_free_rate: float = 0.0,
    ) -> go.Figure:
        """Create rolling Sharpe ratio chart.

        Args:
            window: Rolling window size in periods
            height: Chart height in pixels
            title: Chart title
            risk_free_rate: Annualized risk-free rate

        Returns:
            Plotly Figure object
        """
        returns_arr = np.array(self.data.returns, dtype=np.float64)

        if len(returns_arr) < window:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for {window}-period rolling window",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        # Calculate rolling Sharpe
        rolling_mean = np.convolve(returns_arr, np.ones(window) / window, mode="valid")
        rolling_std = np.array(
            [
                np.std(returns_arr[i : i + window], ddof=1)
                for i in range(len(returns_arr) - window + 1)
            ]
        )

        # Annualize (assuming 252 trading days)
        rf_per_period = risk_free_rate / 252
        rolling_sharpe = (
            (rolling_mean - rf_per_period) / np.where(rolling_std > 0, rolling_std, np.nan)
        ) * np.sqrt(252)

        # Corresponding timestamps
        sharpe_timestamps = self.data.timestamps[window - 1 :]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sharpe_timestamps,
                y=rolling_sharpe,
                mode="lines",
                name=f"{window}-day Rolling Sharpe",
                line={"color": "#2196F3", "width": 2},
            )
        )

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="0",
            annotation_position="right",
        )

        fig.update_layout(
            height=height,
            title=title,
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def rolling_volatility(
        self,
        window: int = 21,
        height: int = 400,
        title: str = "Rolling Volatility",
        annualize: bool = True,
    ) -> go.Figure:
        """Create rolling volatility chart.

        Args:
            window: Rolling window size in periods
            height: Chart height in pixels
            title: Chart title
            annualize: Whether to annualize volatility

        Returns:
            Plotly Figure object
        """
        returns_arr = np.array(self.data.returns, dtype=np.float64)

        if len(returns_arr) < window:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for {window}-period rolling window",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        # Calculate rolling volatility
        rolling_vol = np.array(
            [
                np.std(returns_arr[i : i + window], ddof=1)
                for i in range(len(returns_arr) - window + 1)
            ]
        )

        if annualize:
            rolling_vol = rolling_vol * np.sqrt(252)

        vol_timestamps = self.data.timestamps[window - 1 :]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=vol_timestamps,
                y=rolling_vol,
                mode="lines",
                name=f"{window}-day Rolling Volatility",
                line={"color": "#FF9800", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(255, 152, 0, 0.2)",
            )
        )

        fig.update_layout(
            height=height,
            title=title,
            yaxis_title="Volatility" + (" (Annualized)" if annualize else ""),
            yaxis_tickformat=".1%",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def cumulative_returns_by_regime(
        self,
        regimes: list[tuple[datetime, datetime, str]],
        height: int = 400,
        title: str = "Cumulative Returns by Regime",
    ) -> go.Figure:
        """Create cumulative returns chart colored by market regime.

        Args:
            regimes: List of (start_time, end_time, regime_name) tuples
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate cumulative returns
        returns_arr = np.array(self.data.returns, dtype=np.float64)
        cum_returns = np.cumprod(1 + returns_arr) - 1

        fig = go.Figure()

        # Define regime colors
        regime_colors = {
            "bull": "rgba(76, 175, 80, 0.3)",
            "bear": "rgba(244, 67, 54, 0.3)",
            "neutral": "rgba(158, 158, 158, 0.3)",
            "high_vol": "rgba(255, 152, 0, 0.3)",
            "low_vol": "rgba(33, 150, 243, 0.3)",
        }

        # Add regime backgrounds
        for start, end, regime in regimes:
            color = regime_colors.get(regime.lower(), "rgba(158, 158, 158, 0.2)")
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                layer="below",
                line_width=0,
                annotation_text=regime,
                annotation_position="top left",
            )

        # Main cumulative returns line
        fig.add_trace(
            go.Scatter(
                x=self.data.timestamps,
                y=cum_returns,
                mode="lines",
                name="Cumulative Return",
                line={"color": "#2196F3", "width": 2},
            )
        )

        fig.update_layout(
            height=height,
            title=title,
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def _calculate_monthly_returns(self) -> dict[tuple[int, int], float]:
        """Calculate monthly returns from daily data."""
        monthly_returns: dict[tuple[int, int], float] = {}

        # Group returns by month
        monthly_data: dict[tuple[int, int], list[float]] = {}
        for ts, ret in zip(self.data.timestamps, self.data.returns, strict=False):
            if isinstance(ts, datetime):
                key = (ts.year, ts.month)
                if key not in monthly_data:
                    monthly_data[key] = []
                monthly_data[key].append(ret)

        # Calculate compounded monthly returns
        for key, returns in monthly_data.items():
            monthly_returns[key] = np.prod(1 + np.array(returns)) - 1

        return monthly_returns


class TradeAnalysisCharts:
    """Trade analysis charts for win/loss distribution, duration, and P&L analysis."""

    def __init__(self, data: BacktestData) -> None:
        """Initialize trade analysis charts.

        Args:
            data: Backtest data container
        """
        self.data = data

    def win_loss_distribution(
        self,
        height: int = 400,
        title: str = "Win/Loss Distribution",
        bins: int = 50,
    ) -> go.Figure:
        """Create win/loss distribution histogram.

        Args:
            height: Chart height in pixels
            title: Chart title
            bins: Number of histogram bins

        Returns:
            Plotly Figure object
        """
        if not self.data.trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        pnls = [t.pnl for t in self.data.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        fig = go.Figure()

        # Winners
        if winners:
            fig.add_trace(
                go.Histogram(
                    x=winners,
                    name="Winners",
                    marker_color="#4CAF50",
                    opacity=0.7,
                    nbinsx=bins // 2,
                )
            )

        # Losers
        if losers:
            fig.add_trace(
                go.Histogram(
                    x=losers,
                    name="Losers",
                    marker_color="#F44336",
                    opacity=0.7,
                    nbinsx=bins // 2,
                )
            )

        # Add statistics annotations
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0

        fig.add_annotation(
            text=(
                f"Win Rate: {win_rate:.1f}%<br>"
                f"Avg Win: ${avg_win:,.0f}<br>"
                f"Avg Loss: ${avg_loss:,.0f}<br>"
                f"Total: {len(pnls)} trades"
            ),
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            bordercolor="gray",
            borderwidth=1,
            borderpad=5,
            bgcolor="white",
            align="left",
        )

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white",
        )

        return fig

    def trade_duration_histogram(
        self,
        height: int = 400,
        title: str = "Trade Duration Distribution",
        bins: int = 30,
    ) -> go.Figure:
        """Create trade duration histogram.

        Args:
            height: Chart height in pixels
            title: Chart title
            bins: Number of histogram bins

        Returns:
            Plotly Figure object
        """
        if not self.data.trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        durations = [t.duration_hours for t in self.data.trades]
        winner_durations = [t.duration_hours for t in self.data.trades if t.is_winner]
        loser_durations = [t.duration_hours for t in self.data.trades if not t.is_winner]

        fig = go.Figure()

        if winner_durations:
            fig.add_trace(
                go.Histogram(
                    x=winner_durations,
                    name="Winners",
                    marker_color="#4CAF50",
                    opacity=0.7,
                    nbinsx=bins // 2,
                )
            )

        if loser_durations:
            fig.add_trace(
                go.Histogram(
                    x=loser_durations,
                    name="Losers",
                    marker_color="#F44336",
                    opacity=0.7,
                    nbinsx=bins // 2,
                )
            )

        # Statistics
        avg_duration = np.mean(durations)
        median_duration = np.median(durations)

        fig.add_annotation(
            text=(f"Mean: {avg_duration:.1f}h<br>" f"Median: {median_duration:.1f}h"),
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            bordercolor="gray",
            borderwidth=1,
            borderpad=5,
            bgcolor="white",
            align="left",
        )

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="Duration (hours)",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white",
        )

        return fig

    def pnl_by_symbol(
        self,
        height: int = 400,
        title: str = "P&L by Symbol",
    ) -> go.Figure:
        """Create P&L by symbol bar chart.

        Args:
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if not self.data.trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        # Aggregate P&L by symbol
        symbol_pnl: dict[str, float] = {}
        symbol_count: dict[str, int] = {}
        for t in self.data.trades:
            if t.symbol not in symbol_pnl:
                symbol_pnl[t.symbol] = 0.0
                symbol_count[t.symbol] = 0
            symbol_pnl[t.symbol] += t.pnl
            symbol_count[t.symbol] += 1

        # Sort by P&L
        sorted_symbols = sorted(symbol_pnl.keys(), key=lambda s: symbol_pnl[s], reverse=True)
        pnls = [symbol_pnl[s] for s in sorted_symbols]
        counts = [symbol_count[s] for s in sorted_symbols]

        # Colors based on positive/negative
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=sorted_symbols,
                y=pnls,
                marker_color=colors,
                text=[f"{c} trades" for c in counts],
                textposition="outside",
            )
        )

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="Symbol",
            yaxis_title="P&L ($)",
            template="plotly_white",
        )

        return fig

    def return_vs_duration_scatter(
        self,
        height: int = 400,
        title: str = "Return vs Holding Time",
    ) -> go.Figure:
        """Create scatter plot of return vs holding time.

        Args:
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if not self.data.trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        durations = [t.duration_hours for t in self.data.trades]
        returns = [t.return_pct for t in self.data.trades]
        symbols = [t.symbol for t in self.data.trades]
        is_winner = [t.is_winner for t in self.data.trades]

        colors = ["#4CAF50" if w else "#F44336" for w in is_winner]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=durations,
                y=returns,
                mode="markers",
                marker={"color": colors, "size": 10, "opacity": 0.6},
                text=symbols,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Duration: %{x:.1f}h<br>"
                    "Return: %{y:.2%}<extra></extra>"
                ),
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="Holding Time (hours)",
            yaxis_title="Return",
            yaxis_tickformat=".1%",
            template="plotly_white",
        )

        return fig


class RiskVisualization:
    """Risk-related visualizations including drawdown, VaR, and exposure."""

    def __init__(self, data: BacktestData) -> None:
        """Initialize risk visualization.

        Args:
            data: Backtest data container
        """
        self.data = data

    def drawdown_periods_timeline(
        self,
        height: int = 400,
        title: str = "Drawdown Periods",
        min_drawdown: float = 0.05,
    ) -> go.Figure:
        """Create drawdown periods timeline.

        Args:
            height: Chart height in pixels
            title: Chart title
            min_drawdown: Minimum drawdown to highlight (as decimal)

        Returns:
            Plotly Figure object
        """
        # Calculate drawdown series
        equity_arr = np.array(self.data.equity, dtype=np.float64)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (running_max - equity_arr) / running_max

        fig = go.Figure()

        # Main drawdown line
        fig.add_trace(
            go.Scatter(
                x=self.data.timestamps,
                y=-drawdown,  # Negative for visual representation
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line={"color": "#F44336", "width": 1},
                fillcolor="rgba(244, 67, 54, 0.3)",
            )
        )

        # Identify significant drawdown periods
        periods = self._identify_drawdown_periods(drawdown, min_drawdown)

        # Add annotations for major drawdowns
        for start_idx, end_idx, max_dd, max_dd_idx in periods:
            fig.add_annotation(
                x=self.data.timestamps[max_dd_idx],
                y=-max_dd,
                text=f"{max_dd:.1%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#F44336",
                font={"size": 10},
            )

        fig.update_layout(
            height=height,
            title=title,
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def var_cvar_visualization(
        self,
        confidence_levels: list[float] | None = None,
        height: int = 400,
        title: str = "VaR/CVaR Analysis",
    ) -> go.Figure:
        """Create VaR/CVaR visualization.

        Args:
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        returns_arr = np.array(self.data.returns, dtype=np.float64)
        returns_arr = returns_arr[~np.isnan(returns_arr)]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Return Distribution with VaR", "VaR/CVaR Summary"),
            column_widths=[0.6, 0.4],
        )

        # Histogram of returns
        fig.add_trace(
            go.Histogram(
                x=returns_arr,
                name="Returns",
                marker_color="#2196F3",
                opacity=0.7,
                nbinsx=50,
            ),
            row=1,
            col=1,
        )

        # Calculate VaR and CVaR for each confidence level
        var_values = []
        cvar_values = []
        colors = ["#FF9800", "#F44336"]

        for i, conf in enumerate(confidence_levels):
            alpha = 1 - conf
            var = np.percentile(returns_arr, alpha * 100)
            cvar = np.mean(returns_arr[returns_arr <= var])

            var_values.append(var)
            cvar_values.append(cvar)

            # Add VaR line
            fig.add_vline(
                x=var,
                line_dash="dash",
                line_color=colors[i % len(colors)],
                annotation_text=f"VaR{int(conf * 100)}%: {var:.2%}",
                annotation_position="top",
                row=1,
                col=1,
            )

        # Bar chart for VaR/CVaR summary
        x_labels = [f"{int(c * 100)}%" for c in confidence_levels]

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=[-v * 100 for v in var_values],  # Convert to positive percentage
                name="VaR",
                marker_color="#FF9800",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=[-c * 100 for c in cvar_values],
                name="CVaR",
                marker_color="#F44336",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=height,
            title=title,
            template="plotly_white",
            barmode="group",
        )

        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Confidence Level", row=1, col=2)
        fig.update_yaxes(title_text="Loss (%)", row=1, col=2)

        return fig

    def position_size_over_time(
        self,
        positions: list[tuple[datetime, dict[str, float]]],
        height: int = 400,
        title: str = "Position Size Over Time",
    ) -> go.Figure:
        """Create position size over time chart.

        Args:
            positions: List of (timestamp, {symbol: notional}) tuples
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if not positions:
            fig = go.Figure()
            fig.add_annotation(
                text="No position data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        timestamps = [p[0] for p in positions]

        # Get all symbols
        all_symbols: set[str] = set()
        for _, pos_dict in positions:
            all_symbols.update(pos_dict.keys())

        fig = go.Figure()

        # Stack area chart for each symbol
        for symbol in sorted(all_symbols):
            values = [p[1].get(symbol, 0.0) for p in positions]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines",
                    name=symbol,
                    stackgroup="positions",
                )
            )

        fig.update_layout(
            height=height,
            title=title,
            yaxis_title="Notional ($)",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def exposure_heatmap(
        self,
        exposures: list[tuple[datetime, dict[str, float]]],
        height: int = 400,
        title: str = "Exposure Heatmap",
    ) -> go.Figure:
        """Create exposure heatmap over time.

        Args:
            exposures: List of (timestamp, {exposure_type: value}) tuples
            height: Chart height in pixels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if not exposures:
            fig = go.Figure()
            fig.add_annotation(
                text="No exposure data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=height, title=title)
            return fig

        timestamps = [e[0] for e in exposures]

        # Get all exposure types
        all_types: set[str] = set()
        for _, exp_dict in exposures:
            all_types.update(exp_dict.keys())

        # Build matrix
        z = []
        for exp_type in sorted(all_types):
            row = [e[1].get(exp_type, 0.0) for e in exposures]
            z.append(row)

        # Format timestamps for display
        x_labels = [ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts) for ts in timestamps]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=list(sorted(all_types)),
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Exposure"},
            )
        )

        fig.update_layout(
            height=height,
            title=title,
            xaxis_title="Date",
            yaxis_title="Exposure Type",
            template="plotly_white",
        )

        return fig

    def _identify_drawdown_periods(
        self,
        drawdown: np.ndarray,
        min_drawdown: float,
    ) -> list[tuple[int, int, float, int]]:
        """Identify significant drawdown periods.

        Returns list of (start_idx, end_idx, max_drawdown, max_dd_idx) tuples.
        """
        periods: list[tuple[int, int, float, int]] = []
        in_drawdown = False
        start_idx = 0
        max_dd = 0.0
        max_dd_idx = 0

        for i, dd in enumerate(drawdown):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
                max_dd = dd
                max_dd_idx = i
            elif dd > 0 and in_drawdown:
                if dd > max_dd:
                    max_dd = dd
                    max_dd_idx = i
            elif dd == 0 and in_drawdown:
                if max_dd >= min_drawdown:
                    periods.append((start_idx, i, max_dd, max_dd_idx))
                in_drawdown = False
                max_dd = 0.0

        # Handle case where we end in a drawdown
        if in_drawdown and max_dd >= min_drawdown:
            periods.append((start_idx, len(drawdown) - 1, max_dd, max_dd_idx))

        return periods


# Export all classes
__all__ = [
    "TradeData",
    "BacktestData",
    "EquityCurveChart",
    "PerformanceCharts",
    "TradeAnalysisCharts",
    "RiskVisualization",
]
