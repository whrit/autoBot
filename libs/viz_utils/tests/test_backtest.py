"""Tests for backtest visualization module."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from viz_utils import BacktestData, TradeData
from viz_utils.backtest import (
    EquityCurveChart,
    PerformanceCharts,
    RiskVisualization,
    TradeAnalysisCharts,
)


class TestTradeData:
    """Tests for TradeData class."""

    def test_trade_data_creation(self) -> None:
        """Test basic TradeData creation."""
        trade = TradeData(
            entry_time=datetime(2023, 1, 3, 10, 0),
            exit_time=datetime(2023, 1, 3, 14, 0),
            symbol="AAPL",
            side="long",
            pnl=500.0,
            return_pct=0.025,
            notional=20000.0,
        )

        assert trade.symbol == "AAPL"
        assert trade.side == "long"
        assert trade.pnl == 500.0
        assert trade.is_winner is True

    def test_trade_duration_hours(self) -> None:
        """Test trade duration calculation."""
        trade = TradeData(
            entry_time=datetime(2023, 1, 3, 10, 0),
            exit_time=datetime(2023, 1, 3, 14, 30),
            symbol="AAPL",
            side="long",
            pnl=100.0,
            return_pct=0.01,
            notional=10000.0,
        )

        assert trade.duration_hours == 4.5

    def test_trade_is_winner_losing_trade(self) -> None:
        """Test is_winner for losing trade."""
        trade = TradeData(
            entry_time=datetime(2023, 1, 3),
            exit_time=datetime(2023, 1, 4),
            symbol="GOOGL",
            side="short",
            pnl=-200.0,
            return_pct=-0.02,
            notional=10000.0,
        )

        assert trade.is_winner is False

    def test_trade_zero_pnl(self) -> None:
        """Test trade with zero P&L."""
        trade = TradeData(
            entry_time=datetime(2023, 1, 3),
            exit_time=datetime(2023, 1, 4),
            symbol="MSFT",
            side="long",
            pnl=0.0,
            return_pct=0.0,
            notional=10000.0,
        )

        assert trade.is_winner is False  # Zero is not a winner


class TestBacktestData:
    """Tests for BacktestData class."""

    def test_from_polars(
        self,
        sample_polars_equity_curve: pl.DataFrame,
        sample_polars_benchmark: pl.DataFrame,
    ) -> None:
        """Test creation from Polars DataFrames."""
        trades = [
            {
                "entry_time": datetime(2023, 1, 5),
                "exit_time": datetime(2023, 1, 6),
                "symbol": "AAPL",
                "side": "long",
                "pnl": 500.0,
                "return_pct": 0.05,
                "notional": 10000.0,
            }
        ]

        data = BacktestData.from_polars(
            equity_curve=sample_polars_equity_curve,
            trades=trades,
            benchmark=sample_polars_benchmark,
            initial_capital=100_000.0,
        )

        assert len(data.timestamps) == 252
        assert len(data.equity) == 252
        assert len(data.returns) == 252
        assert data.benchmark_equity is not None
        assert len(data.benchmark_equity) == 252
        assert len(data.trades) == 1

    def test_from_polars_without_returns_column(
        self,
        sample_timestamps: list[datetime],
        sample_equity: list[float],
    ) -> None:
        """Test that returns are calculated when not provided."""
        df = pl.DataFrame(
            {
                "timestamp": sample_timestamps,
                "equity": sample_equity,
            }
        )

        data = BacktestData.from_polars(df)

        assert len(data.returns) == len(sample_equity)
        assert data.returns[0] == 0.0  # First return is 0

    def test_from_polars_without_benchmark(
        self,
        sample_polars_equity_curve: pl.DataFrame,
    ) -> None:
        """Test creation without benchmark."""
        data = BacktestData.from_polars(sample_polars_equity_curve)

        assert data.benchmark_equity is None
        assert data.benchmark_returns is None


class TestEquityCurveChart:
    """Tests for EquityCurveChart class."""

    def test_create_basic_chart(self, sample_backtest_data: BacktestData) -> None:
        """Test basic equity curve chart creation."""
        chart = EquityCurveChart(sample_backtest_data)
        fig = chart.create(
            show_benchmark=False,
            show_drawdown=False,
            show_trades=False,
        )

        assert fig is not None
        assert len(fig.data) >= 1  # At least portfolio line

    def test_create_with_benchmark(self, sample_backtest_data: BacktestData) -> None:
        """Test equity curve with benchmark overlay."""
        chart = EquityCurveChart(sample_backtest_data)
        fig = chart.create(
            show_benchmark=True,
            show_drawdown=False,
            show_trades=False,
        )

        # Should have portfolio and benchmark traces
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Portfolio" in trace_names
        assert "Benchmark" in trace_names

    def test_create_with_drawdown(self, sample_backtest_data: BacktestData) -> None:
        """Test equity curve with drawdown subplot."""
        chart = EquityCurveChart(sample_backtest_data)
        fig = chart.create(
            show_benchmark=False,
            show_drawdown=True,
            show_trades=False,
        )

        # Should have drawdown trace
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Drawdown" in trace_names

    def test_create_with_trades(self, sample_backtest_data: BacktestData) -> None:
        """Test equity curve with trade markers."""
        chart = EquityCurveChart(sample_backtest_data)
        fig = chart.create(
            show_benchmark=False,
            show_drawdown=False,
            show_trades=True,
        )

        # Should have trade entry markers
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert any("Entry" in name for name in trace_names)

    def test_create_full_chart(self, sample_backtest_data: BacktestData) -> None:
        """Test equity curve with all features."""
        chart = EquityCurveChart(sample_backtest_data)
        fig = chart.create(
            show_benchmark=True,
            show_drawdown=True,
            show_trades=True,
            height=800,
            title="Test Equity Curve",
        )

        assert fig is not None
        assert len(fig.data) >= 3  # Multiple traces

    def test_create_minimal_data(self, minimal_backtest_data: BacktestData) -> None:
        """Test with minimal data."""
        chart = EquityCurveChart(minimal_backtest_data)
        fig = chart.create()

        assert fig is not None


class TestPerformanceCharts:
    """Tests for PerformanceCharts class."""

    def test_monthly_returns_heatmap(self, sample_backtest_data: BacktestData) -> None:
        """Test monthly returns heatmap creation."""
        charts = PerformanceCharts(sample_backtest_data)
        fig = charts.monthly_returns_heatmap()

        assert fig is not None
        assert len(fig.data) >= 1

    def test_monthly_returns_heatmap_empty_data(
        self, empty_backtest_data: BacktestData
    ) -> None:
        """Test monthly returns heatmap with insufficient data."""
        charts = PerformanceCharts(empty_backtest_data)
        fig = charts.monthly_returns_heatmap()

        assert fig is not None  # Should return figure with annotation

    def test_rolling_sharpe(self, sample_backtest_data: BacktestData) -> None:
        """Test rolling Sharpe ratio chart."""
        charts = PerformanceCharts(sample_backtest_data)
        fig = charts.rolling_sharpe(window=63)

        assert fig is not None
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert any("Sharpe" in name for name in trace_names)

    def test_rolling_sharpe_insufficient_data(
        self, minimal_backtest_data: BacktestData
    ) -> None:
        """Test rolling Sharpe with insufficient data."""
        charts = PerformanceCharts(minimal_backtest_data)
        fig = charts.rolling_sharpe(window=63)

        assert fig is not None  # Should return figure with annotation

    def test_rolling_volatility(self, sample_backtest_data: BacktestData) -> None:
        """Test rolling volatility chart."""
        charts = PerformanceCharts(sample_backtest_data)
        fig = charts.rolling_volatility(window=21, annualize=True)

        assert fig is not None

    def test_rolling_volatility_not_annualized(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test rolling volatility without annualization."""
        charts = PerformanceCharts(sample_backtest_data)
        fig = charts.rolling_volatility(window=21, annualize=False)

        assert fig is not None

    def test_cumulative_returns_by_regime(
        self,
        sample_backtest_data: BacktestData,
        sample_regimes: list[tuple[datetime, datetime, str]],
    ) -> None:
        """Test cumulative returns by regime chart."""
        charts = PerformanceCharts(sample_backtest_data)
        fig = charts.cumulative_returns_by_regime(regimes=sample_regimes)

        assert fig is not None


class TestTradeAnalysisCharts:
    """Tests for TradeAnalysisCharts class."""

    def test_win_loss_distribution(self, sample_backtest_data: BacktestData) -> None:
        """Test win/loss distribution histogram."""
        charts = TradeAnalysisCharts(sample_backtest_data)
        fig = charts.win_loss_distribution(bins=30)

        assert fig is not None
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Winners" in trace_names or "Losers" in trace_names

    def test_win_loss_distribution_no_trades(
        self, minimal_backtest_data: BacktestData
    ) -> None:
        """Test win/loss distribution with no trades."""
        charts = TradeAnalysisCharts(minimal_backtest_data)
        fig = charts.win_loss_distribution()

        assert fig is not None  # Should show annotation

    def test_trade_duration_histogram(self, sample_backtest_data: BacktestData) -> None:
        """Test trade duration histogram."""
        charts = TradeAnalysisCharts(sample_backtest_data)
        fig = charts.trade_duration_histogram(bins=20)

        assert fig is not None

    def test_pnl_by_symbol(self, sample_backtest_data: BacktestData) -> None:
        """Test P&L by symbol bar chart."""
        charts = TradeAnalysisCharts(sample_backtest_data)
        fig = charts.pnl_by_symbol()

        assert fig is not None
        assert len(fig.data) >= 1

    def test_pnl_by_symbol_no_trades(
        self, empty_backtest_data: BacktestData
    ) -> None:
        """Test P&L by symbol with no trades."""
        charts = TradeAnalysisCharts(empty_backtest_data)
        fig = charts.pnl_by_symbol()

        assert fig is not None

    def test_return_vs_duration_scatter(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test return vs duration scatter plot."""
        charts = TradeAnalysisCharts(sample_backtest_data)
        fig = charts.return_vs_duration_scatter()

        assert fig is not None
        assert len(fig.data) >= 1


class TestRiskVisualization:
    """Tests for RiskVisualization class."""

    def test_drawdown_periods_timeline(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test drawdown periods timeline."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.drawdown_periods_timeline(min_drawdown=0.02)

        assert fig is not None
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Drawdown" in trace_names

    def test_var_cvar_visualization(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test VaR/CVaR visualization."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.var_cvar_visualization(confidence_levels=[0.95, 0.99])

        assert fig is not None
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "VaR" in trace_names or "CVaR" in trace_names

    def test_var_cvar_default_confidence(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test VaR/CVaR with default confidence levels."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.var_cvar_visualization()

        assert fig is not None

    def test_position_size_over_time(
        self,
        sample_backtest_data: BacktestData,
        sample_positions: list[tuple[datetime, dict[str, float]]],
    ) -> None:
        """Test position size over time chart."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.position_size_over_time(positions=sample_positions)

        assert fig is not None

    def test_position_size_empty_positions(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test position size with no data."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.position_size_over_time(positions=[])

        assert fig is not None

    def test_exposure_heatmap(
        self,
        sample_backtest_data: BacktestData,
        sample_exposures: list[tuple[datetime, dict[str, float]]],
    ) -> None:
        """Test exposure heatmap."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.exposure_heatmap(exposures=sample_exposures)

        assert fig is not None

    def test_exposure_heatmap_empty(
        self, sample_backtest_data: BacktestData
    ) -> None:
        """Test exposure heatmap with no data."""
        charts = RiskVisualization(sample_backtest_data)
        fig = charts.exposure_heatmap(exposures=[])

        assert fig is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_data_point(self) -> None:
        """Test with single data point."""
        data = BacktestData(
            timestamps=[datetime(2023, 1, 3)],
            equity=[100_000.0],
            returns=[0.0],
            trades=[],
            initial_capital=100_000.0,
        )

        chart = EquityCurveChart(data)
        fig = chart.create()
        assert fig is not None

    def test_all_winning_trades(self) -> None:
        """Test with all winning trades."""
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

        charts = TradeAnalysisCharts(data)
        fig = charts.win_loss_distribution()
        assert fig is not None

    def test_all_losing_trades(self) -> None:
        """Test with all losing trades."""
        trades = [
            TradeData(
                entry_time=datetime(2023, 1, 3),
                exit_time=datetime(2023, 1, 4),
                symbol="AAPL",
                side="long",
                pnl=-100.0,
                return_pct=-0.01,
                notional=10000.0,
            )
            for _ in range(10)
        ]

        data = BacktestData(
            timestamps=[datetime(2023, 1, 3), datetime(2023, 1, 4)],
            equity=[100_000.0, 99_000.0],
            returns=[0.0, -0.01],
            trades=trades,
            initial_capital=100_000.0,
        )

        charts = TradeAnalysisCharts(data)
        fig = charts.win_loss_distribution()
        assert fig is not None

    def test_zero_volatility_returns(self) -> None:
        """Test with constant returns (zero volatility)."""
        timestamps = [datetime(2023, 1, i) for i in range(3, 13)]
        data = BacktestData(
            timestamps=timestamps,
            equity=[100_000.0 + i * 100 for i in range(10)],
            returns=[0.001] * 10,
            trades=[],
            initial_capital=100_000.0,
        )

        charts = PerformanceCharts(data)
        fig = charts.rolling_sharpe(window=5)
        assert fig is not None

    def test_negative_equity(self) -> None:
        """Test with negative equity (margin call scenario)."""
        timestamps = [datetime(2023, 1, i) for i in range(3, 8)]
        data = BacktestData(
            timestamps=timestamps,
            equity=[100_000.0, 50_000.0, 10_000.0, -5_000.0, -10_000.0],
            returns=[0.0, -0.5, -0.8, -1.5, -1.0],
            trades=[],
            initial_capital=100_000.0,
        )

        chart = EquityCurveChart(data)
        fig = chart.create()
        assert fig is not None

    def test_large_dataset(self) -> None:
        """Test with large dataset (5 years of daily data)."""
        np.random.seed(42)
        n = 252 * 5  # 5 years of daily data

        timestamps = [datetime(2018, 1, 1) + timedelta(days=i) for i in range(n)]
        returns = np.random.normal(0.0003, 0.015, n)

        equity = [100_000.0]
        for ret in returns[1:]:
            equity.append(equity[-1] * (1 + ret))

        data = BacktestData(
            timestamps=timestamps,
            equity=equity,
            returns=returns.tolist(),
            trades=[],
            initial_capital=100_000.0,
        )

        chart = EquityCurveChart(data)
        fig = chart.create()
        assert fig is not None

        perf = PerformanceCharts(data)
        fig2 = perf.monthly_returns_heatmap()
        assert fig2 is not None
