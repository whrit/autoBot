"""
Labeler Service for Autonomous Equities Trading Engine.

This service defines taker-realistic prediction targets by:
- Calculating forward returns at configurable horizons (T2.06)
- Converting returns to direction labels with no-trade band (T2.07)
- Computing net-of-spread returns accounting for transaction costs (T2.08)

Key Features:
- Uses midprice/microprice (NOT last trade) for accurate returns
- Configurable horizons: 60s, 300s, 900s (default)
- No-trade band filters unprofitable signals
- Net returns account for spread crossing costs

Example usage:
    >>> from labeler_py import ForwardReturnsCalculator, DirectionLabeler, NetOfSpreadCalculator
    >>>
    >>> # Calculate forward returns
    >>> returns_calc = ForwardReturnsCalculator(horizons=[60, 300, 900])
    >>> returns_df = returns_calc.calculate(quotes_df, decision_times, "SPY")
    >>>
    >>> # Calculate net-of-spread returns
    >>> net_calc = NetOfSpreadCalculator()
    >>> net_returns_df = net_calc.calculate_net_returns(returns_df)
    >>>
    >>> # Label directions with no-trade band
    >>> labeler = DirectionLabeler(no_trade_threshold=0.0005)
    >>> labeled_df = labeler.label_returns(net_returns_df)
"""

from labeler_py.labels import DirectionLabeler
from labeler_py.net_returns import NetOfSpreadCalculator
from labeler_py.returns import ForwardReturnsCalculator

__all__ = [
    "DirectionLabeler",
    "ForwardReturnsCalculator",
    "NetOfSpreadCalculator",
]

__version__ = "0.1.0"
