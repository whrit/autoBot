"""
Metrics computation module for backtesting evaluation.

This module provides comprehensive performance metrics calculation including:
- Sharpe ratio (annualized)
- Sortino ratio (downside deviation only)
- Maximum drawdown and duration
- Profit factor
- Win rate and average win/loss
- Total return and CAGR
- Calmar ratio

Implementation follows financial industry standards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class BacktestMetrics:
    """
    Comprehensive backtest performance metrics.

    All metrics are calculated from equity curve and trade history.
    Immutable dataclass for safe sharing across analysis pipeline.

    Attributes:
        sharpe_ratio: Annualized Sharpe ratio (excess return / volatility)
        sortino_ratio: Annualized Sortino ratio (uses downside deviation)
        max_drawdown: Maximum drawdown as decimal (0.15 = 15%)
        max_drawdown_duration: Duration of max drawdown in bars
        profit_factor: Gross profit / gross loss
        win_rate: Percentage of winning trades (0.55 = 55%)
        avg_win: Average winning trade P&L
        avg_loss: Average losing trade P&L (positive number)
        total_return: Total return as decimal (0.25 = 25%)
        cagr: Compound annual growth rate
        calmar_ratio: CAGR / max drawdown
        num_trades: Total number of trades
    """

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_return: float
    cagr: float
    calmar_ratio: float
    num_trades: int


class MetricsCalculator:
    """
    Calculator for comprehensive backtest metrics.

    Provides standardized calculation of all performance metrics from
    equity curve and trade history.

    Attributes:
        risk_free_rate: Annualized risk-free rate for Sharpe calculation
        periods_per_year: Trading periods per year (252 for daily, 252*6.5*60 for minute)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize MetricsCalculator.

        Args:
            risk_free_rate: Annualized risk-free rate (default 0.0)
            periods_per_year: Number of trading periods per year (default 252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate(
        self,
        equity_curve: pl.DataFrame,
        trades: list[dict[str, float | str]],
    ) -> BacktestMetrics:
        """
        Calculate all backtest metrics.

        Args:
            equity_curve: DataFrame with columns: timestamp, equity, returns
            trades: List of trade dicts with 'pnl' key (and optionally 'side', 'notional')

        Returns:
            BacktestMetrics with all calculated metrics
        """

        # Extract returns as numpy array
        if "returns" in equity_curve.columns:
            returns = equity_curve["returns"].to_numpy().astype(np.float64)
        else:
            equity = equity_curve["equity"].to_numpy().astype(np.float64)
            returns = np.diff(equity) / equity[:-1]
            returns = np.insert(returns, 0, 0.0)

        equity = equity_curve["equity"].to_numpy().astype(np.float64)

        # Calculate basic return metrics
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        total_return = self._calculate_total_return(equity)
        cagr = self._calculate_cagr(equity)

        # Calculate drawdown metrics
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity)

        # Calculate trade-based metrics
        trade_metrics = self._calculate_trade_metrics(trades)

        # Calculate Calmar ratio
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0.0

        return BacktestMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            profit_factor=trade_metrics["profit_factor"],
            win_rate=trade_metrics["win_rate"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            total_return=total_return,
            cagr=cagr,
            calmar_ratio=calmar_ratio,
            num_trades=int(trade_metrics["num_trades"]),
        )

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean(returns) - rf/periods) / std(returns) * sqrt(periods)

        Args:
            returns: Array of period returns

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        std = np.std(returns, ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0

        # Convert annual risk-free rate to per-period rate
        rf_per_period = self.risk_free_rate / self.periods_per_year

        excess_return = np.mean(returns) - rf_per_period
        sharpe = excess_return / std * np.sqrt(self.periods_per_year)

        return float(sharpe)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """
        Calculate annualized Sortino ratio.

        Sortino = (mean(returns) - rf/periods) / downside_std * sqrt(periods)

        Uses only negative returns for downside deviation.

        Args:
            returns: Array of period returns

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        # Calculate downside deviation (only negative returns)
        rf_per_period = self.risk_free_rate / self.periods_per_year
        downside_returns = returns[returns < rf_per_period]

        if len(downside_returns) == 0:
            # No downside returns - return high value (but not infinite)
            mean_return = np.mean(returns)
            if mean_return > rf_per_period:
                return 10.0  # Cap at 10 for practical purposes
            return 0.0

        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        excess_return = np.mean(returns) - rf_per_period
        sortino = excess_return / downside_std * np.sqrt(self.periods_per_year)

        return float(sortino)

    def _calculate_drawdown(self, equity: np.ndarray) -> tuple[float, int]:
        """
        Calculate maximum drawdown and its duration.

        MDD = max(1 - equity / running_max)

        Args:
            equity: Array of equity values

        Returns:
            Tuple of (max_drawdown, max_drawdown_duration_bars)
        """
        if len(equity) < 2:
            return 0.0, 0

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown at each point
        drawdown = 1 - equity / running_max

        # Handle edge cases
        drawdown = np.where(np.isnan(drawdown), 0.0, drawdown)
        drawdown = np.where(np.isinf(drawdown), 0.0, drawdown)

        max_drawdown = float(np.max(drawdown))

        # Calculate drawdown duration
        max_drawdown_duration = self._calculate_drawdown_duration(equity, running_max)

        return max_drawdown, max_drawdown_duration

    def _calculate_drawdown_duration(
        self, equity: np.ndarray, running_max: np.ndarray
    ) -> int:
        """
        Calculate the duration of the maximum drawdown.

        Args:
            equity: Array of equity values
            running_max: Array of running maximum equity

        Returns:
            Duration of maximum drawdown in bars
        """
        if len(equity) < 2:
            return 0

        # Find periods in drawdown (equity < running max)
        in_drawdown = equity < running_max

        if not np.any(in_drawdown):
            return 0

        # Find drawdown periods
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_total_return(self, equity: np.ndarray) -> float:
        """
        Calculate total return.

        Args:
            equity: Array of equity values

        Returns:
            Total return as decimal
        """
        if len(equity) < 2:
            return 0.0

        initial = equity[0]
        final = equity[-1]

        if initial == 0:
            return 0.0

        return float((final - initial) / initial)

    def _calculate_cagr(self, equity: np.ndarray) -> float:
        """
        Calculate compound annual growth rate.

        CAGR = (final/initial)^(periods_per_year/n_periods) - 1

        Args:
            equity: Array of equity values

        Returns:
            CAGR as decimal
        """
        if len(equity) < 2:
            return 0.0

        initial = equity[0]
        final = equity[-1]

        if initial <= 0 or final <= 0:
            return 0.0

        n_periods = len(equity) - 1
        years = n_periods / self.periods_per_year

        if years <= 0:
            return 0.0

        cagr = (final / initial) ** (1 / years) - 1

        return float(cagr)

    def _calculate_trade_metrics(
        self, trades: list[dict[str, float | str]]
    ) -> dict[str, float | int]:
        """
        Calculate trade-based metrics.

        Args:
            trades: List of trade dicts with 'pnl' key

        Returns:
            Dict with profit_factor, win_rate, avg_win, avg_loss, num_trades
        """
        if not trades:
            return {
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "num_trades": 0,
            }

        pnls = [float(t.get("pnl", 0.0)) for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Win rate
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        # Average win/loss
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = abs(float(np.mean(losses))) if losses else 0.0

        return {
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "num_trades": len(trades),
        }


__all__ = ["BacktestMetrics", "MetricsCalculator"]
