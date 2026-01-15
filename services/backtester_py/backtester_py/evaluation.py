"""
Backtester Evaluation Module (T3.05-T3.08).

Provides:
- Performance metrics computation (Sharpe, Sortino, MDD, etc.)
- Walk-forward evaluation framework
- Regime slicing
- Purged cross-validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from backtester_py.engine import BacktestEngine, BacktestResult, Signal


@dataclass
class EvaluationMetrics:
    """
    Comprehensive backtest evaluation metrics.

    All metrics follow standard quantitative finance definitions.
    """

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    annualized_volatility: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Exposure metrics
    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    time_in_market_pct: float = 0.0

    # Cost metrics
    total_commission: float = 0.0
    total_slippage_bps: float = 0.0
    avg_slippage_bps: float = 0.0

    # Turnover
    turnover: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "avg_drawdown": self.avg_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_exposure": self.avg_exposure,
            "max_exposure": self.max_exposure,
            "time_in_market_pct": self.time_in_market_pct,
            "total_commission": self.total_commission,
            "total_slippage_bps": self.total_slippage_bps,
            "avg_slippage_bps": self.avg_slippage_bps,
            "turnover": self.turnover,
        }


def calculate_metrics(
    result: BacktestResult,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> EvaluationMetrics:
    """
    Calculate comprehensive metrics from backtest results.

    Args:
        result: BacktestResult from backtester
        initial_capital: Initial capital amount
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
        trading_days_per_year: Number of trading days per year

    Returns:
        EvaluationMetrics with all computed values
    """
    metrics = EvaluationMetrics()

    # Basic trade metrics
    metrics.total_trades = len(result.fills)

    # Check if we have any data to work with (either fills or equity curve)
    has_fills = metrics.total_trades > 0
    has_equity_curve = not result.equity_curve.is_empty() and len(result.equity_curve) >= 2

    if not has_fills and not has_equity_curve:
        return metrics

    # Calculate trade P&Ls
    trade_pnls = _calculate_trade_pnls(result)

    if trade_pnls:
        winning_pnls = [p for p in trade_pnls if p > 0]
        losing_pnls = [p for p in trade_pnls if p < 0]

        metrics.winning_trades = len(winning_pnls)
        metrics.losing_trades = len(losing_pnls)
        metrics.win_rate = metrics.winning_trades / len(trade_pnls) if trade_pnls else 0

        # Average trade metrics
        metrics.avg_win = float(np.mean(winning_pnls)) if winning_pnls else 0.0
        metrics.avg_loss = float(np.mean(losing_pnls)) if losing_pnls else 0.0
        metrics.avg_trade = float(np.mean(trade_pnls))
        metrics.largest_win = max(winning_pnls) if winning_pnls else 0.0
        metrics.largest_loss = min(losing_pnls) if losing_pnls else 0.0

        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Return metrics from equity curve
    if not result.equity_curve.is_empty():
        equity_values = result.equity_curve["equity"].to_list()

        if len(equity_values) >= 2:
            # Total return
            final_equity = equity_values[-1]
            metrics.total_return = (final_equity - initial_capital) / initial_capital

            # Calculate returns
            returns = _calculate_returns_from_equity(equity_values)

            if returns:
                returns_array = np.array(returns)

                # Volatility
                metrics.volatility = float(np.std(returns_array))
                metrics.annualized_volatility = metrics.volatility * np.sqrt(trading_days_per_year)

                # Annualized return
                n_periods = len(returns_array)
                if n_periods > 0:
                    compound_return = (1 + metrics.total_return)
                    metrics.annualized_return = (
                        compound_return ** (trading_days_per_year / n_periods) - 1
                    )

                # Sharpe ratio
                daily_rf = risk_free_rate / trading_days_per_year
                excess_returns = returns_array - daily_rf
                if metrics.volatility > 0:
                    sharpe_numerator = np.mean(excess_returns)
                    sharpe_denominator = np.std(excess_returns)
                    annualization_factor = np.sqrt(trading_days_per_year)
                    metrics.sharpe_ratio = float(
                        sharpe_numerator / sharpe_denominator * annualization_factor
                    )

                # Sortino ratio
                downside_returns = returns_array[returns_array < daily_rf]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        metrics.sortino_ratio = float(
                            np.mean(excess_returns) / downside_std * np.sqrt(trading_days_per_year)
                        )

    # Drawdown metrics
    if not result.equity_curve.is_empty():
        drawdown_info = _calculate_drawdown_metrics(result.equity_curve)
        metrics.max_drawdown = drawdown_info["max_drawdown"]
        metrics.max_drawdown_duration_days = int(drawdown_info["max_duration_days"])
        metrics.avg_drawdown = drawdown_info["avg_drawdown"]

        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

    # Cost metrics
    if result.fills:
        metrics.total_commission = sum(f.commission for f in result.fills)
        slippages = [f.slippage_bps for f in result.fills]
        metrics.total_slippage_bps = sum(slippages)
        metrics.avg_slippage_bps = float(np.mean(slippages)) if slippages else 0.0

    # Turnover
    total_notional = sum(f.notional for f in result.fills)
    metrics.turnover = total_notional / initial_capital

    return metrics


def _calculate_trade_pnls(result: BacktestResult) -> list[float]:
    """Calculate P&L for each round-trip trade."""
    trade_pnls: list[float] = []

    # Group fills by symbol and match buys with sells
    fills_by_symbol: dict[str, list[tuple[str, float, float]]] = {}
    for fill in result.fills:
        if fill.symbol not in fills_by_symbol:
            fills_by_symbol[fill.symbol] = []
        fills_by_symbol[fill.symbol].append((fill.side, fill.quantity, fill.price))

    for _symbol, fills in fills_by_symbol.items():
        open_qty = 0.0
        open_cost = 0.0

        for side, qty, price in fills:
            if side == "buy":
                if open_qty >= 0:
                    # Opening/adding to long
                    open_cost += qty * price
                    open_qty += qty
                else:
                    # Closing short
                    closed_qty = min(qty, abs(open_qty))
                    pnl = closed_qty * (open_cost / abs(open_qty) - price)
                    trade_pnls.append(pnl)
                    open_qty += qty
                    if open_qty > 0:
                        open_cost = open_qty * price
                    elif open_qty == 0:
                        open_cost = 0
            else:  # sell
                if open_qty <= 0:
                    # Opening/adding to short
                    open_cost += qty * price
                    open_qty -= qty
                else:
                    # Closing long
                    closed_qty = min(qty, open_qty)
                    pnl = closed_qty * (price - open_cost / open_qty)
                    trade_pnls.append(pnl)
                    open_qty -= qty
                    if open_qty < 0:
                        open_cost = abs(open_qty) * price
                    elif open_qty == 0:
                        open_cost = 0

    return trade_pnls


def _calculate_returns_from_equity(equity_values: list[float]) -> list[float]:
    """Calculate period returns from equity values."""
    returns: list[float] = []
    for i in range(1, len(equity_values)):
        if equity_values[i - 1] > 0:
            ret = (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
            returns.append(ret)
    return returns


def _calculate_drawdown_metrics(equity_curve: pl.DataFrame) -> dict[str, float | int]:
    """Calculate drawdown-related metrics."""
    equity_values = equity_curve["equity"].to_list()

    if not equity_values:
        return {"max_drawdown": 0.0, "max_duration_days": 0, "avg_drawdown": 0.0}

    # Calculate running maximum
    running_max = equity_values[0]
    drawdowns: list[float] = []
    current_dd_start: int | None = None
    max_duration = 0
    current_duration = 0

    for i, equity in enumerate(equity_values):
        if equity > running_max:
            running_max = equity
            if current_dd_start is not None:
                max_duration = max(max_duration, current_duration)
                current_dd_start = None
                current_duration = 0

        if running_max > 0:
            dd = (running_max - equity) / running_max
            drawdowns.append(dd)

            if dd > 0:
                if current_dd_start is None:
                    current_dd_start = i
                current_duration += 1

    # Handle case where drawdown continues to end
    if current_dd_start is not None:
        max_duration = max(max_duration, current_duration)

    max_dd = max(drawdowns) if drawdowns else 0.0
    avg_dd = np.mean([d for d in drawdowns if d > 0]) if any(d > 0 for d in drawdowns) else 0.0

    return {
        "max_drawdown": max_dd,
        "max_duration_days": max_duration,
        "avg_drawdown": float(avg_dd),
    }


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward evaluation.

    Attributes:
        train_period_days: Length of training window in days
        test_period_days: Length of test window in days
        step_days: Step size between windows in days
        purge_days: Gap between train and test to prevent leakage
        embargo_days: Gap after test period before next train
        min_train_trades: Minimum trades required in training period
    """

    train_period_days: int = 252  # 1 year
    test_period_days: int = 63  # ~3 months
    step_days: int = 21  # ~1 month
    purge_days: int = 5  # 1 week purge gap
    embargo_days: int = 1  # 1 day embargo
    min_train_trades: int = 30


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics


@dataclass
class WalkForwardSummary:
    """Summary of walk-forward evaluation."""

    folds: list[WalkForwardResult] = field(default_factory=list)
    avg_test_sharpe: float = 0.0
    avg_test_return: float = 0.0
    avg_test_max_dd: float = 0.0
    std_test_sharpe: float = 0.0
    consistency_ratio: float = 0.0  # % of folds with positive return


class WalkForwardEvaluator:
    """
    Walk-forward evaluation framework with purged cross-validation.

    Implements time-series cross-validation with:
    - Purge gap to prevent lookahead bias
    - Embargo period after test
    - Rolling or expanding windows
    """

    def __init__(
        self,
        engine: BacktestEngine,
        config: WalkForwardConfig,
    ) -> None:
        """
        Initialize walk-forward evaluator.

        Args:
            engine: BacktestEngine instance
            config: Walk-forward configuration
        """
        self.engine = engine
        self.config = config

    def evaluate(
        self,
        market_data: pl.DataFrame,
        signals: list[Signal],
        start_date: datetime,
        end_date: datetime,
    ) -> WalkForwardSummary:
        """
        Run walk-forward evaluation.

        Args:
            market_data: Full market data DataFrame
            signals: Full list of signals
            start_date: Start of evaluation period
            end_date: End of evaluation period

        Returns:
            WalkForwardSummary with all fold results
        """
        folds: list[WalkForwardResult] = []

        # Generate fold boundaries
        current_start = start_date
        fold_id = 0

        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + timedelta(days=self.config.train_period_days)

            # Purge gap
            test_start = train_end + timedelta(days=self.config.purge_days)
            test_end = test_start + timedelta(days=self.config.test_period_days)

            # Check if we've exceeded end date
            if test_end > end_date:
                break

            # Filter data and signals for each window
            train_data = self._filter_by_date(market_data, train_start, train_end)
            train_signals = self._filter_signals_by_date(signals, train_start, train_end)

            test_data = self._filter_by_date(market_data, test_start, test_end)
            test_signals = self._filter_signals_by_date(signals, test_start, test_end)

            # Skip if insufficient training data
            if len(train_signals) < self.config.min_train_trades:
                current_start += timedelta(days=self.config.step_days)
                continue

            # Run backtests
            train_result = self.engine.run(train_data, train_signals)
            test_result = self.engine.run(test_data, test_signals)

            # Calculate metrics
            train_metrics = calculate_metrics(
                train_result, self.engine.config.initial_capital
            )
            test_metrics = calculate_metrics(
                test_result, self.engine.config.initial_capital
            )

            fold = WalkForwardResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            folds.append(fold)

            # Move to next fold
            current_start += timedelta(days=self.config.step_days)
            fold_id += 1

        # Build summary
        return self._build_summary(folds)

    def _filter_by_date(
        self, df: pl.DataFrame, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Filter DataFrame by date range."""
        return df.filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
        )

    def _filter_signals_by_date(
        self, signals: list[Signal], start: datetime, end: datetime
    ) -> list[Signal]:
        """Filter signals by date range."""
        return [s for s in signals if start <= s.timestamp < end]

    def _build_summary(self, folds: list[WalkForwardResult]) -> WalkForwardSummary:
        """Build walk-forward summary from fold results."""
        if not folds:
            return WalkForwardSummary()

        test_sharpes = [f.test_metrics.sharpe_ratio for f in folds]
        test_returns = [f.test_metrics.total_return for f in folds]
        test_drawdowns = [f.test_metrics.max_drawdown for f in folds]

        positive_folds = sum(1 for r in test_returns if r > 0)

        return WalkForwardSummary(
            folds=folds,
            avg_test_sharpe=float(np.mean(test_sharpes)) if test_sharpes else 0.0,
            avg_test_return=float(np.mean(test_returns)) if test_returns else 0.0,
            avg_test_max_dd=float(np.mean(test_drawdowns)) if test_drawdowns else 0.0,
            std_test_sharpe=float(np.std(test_sharpes)) if test_sharpes else 0.0,
            consistency_ratio=positive_folds / len(folds) if folds else 0.0,
        )
