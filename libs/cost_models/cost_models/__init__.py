"""
Cost Models Library - Slippage and Transaction Cost Modeling.

This library provides realistic transaction cost estimation for taker-style
trading strategies. It models slippage based on spread, order size relative
to book depth, and short-term volatility.

Slippage formula:
    slippage_bps = a*spread_bps + b*(order/book)*100 + c*vol*100

Where:
    - a = spread_coef (spread impact coefficient)
    - b = size_coef (size impact coefficient)
    - c = vol_coef (volatility impact coefficient)

Example usage:
    >>> from cost_models import SlippageModel, TransactionCostModel
    >>> slippage = SlippageModel(spread_coef=1.0, size_coef=0.5, vol_coef=2.0)
    >>> model = TransactionCostModel(slippage, fixed_cost_bps=0.35)
    >>> fill = model.calculate_fill_price(
    ...     side="buy", ask_price=100.01, bid_price=100.00,
    ...     order_notional=50000, book_notional=500000, short_term_vol=0.001
    ... )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SlippageModel:
    """
    Slippage model based on spread, order size, and volatility.

    The model calculates expected slippage in basis points using a linear
    combination of three factors:
    - Spread impact: proportional to the current bid-ask spread
    - Size impact: proportional to order size relative to top-of-book liquidity
    - Volatility impact: proportional to short-term realized volatility

    Formula:
        slippage_bps = spread_coef * spread_bps
                     + size_coef * (order_notional / book_notional) * 100
                     + vol_coef * short_term_vol * 100

    Attributes:
        spread_coef: Coefficient for spread impact (default: 1.0)
        size_coef: Coefficient for size impact (default: 1.0)
        vol_coef: Coefficient for volatility impact (default: 1.0)
        max_size_impact_bps: Maximum size impact in bps (default: 50.0)
    """

    spread_coef: float = 1.0
    size_coef: float = 1.0
    vol_coef: float = 1.0
    max_size_impact_bps: float = 50.0

    def calculate_slippage_bps(
        self,
        spread_bps: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate expected slippage in basis points.

        Args:
            spread_bps: Current bid-ask spread in basis points
            order_notional: Order size in notional terms (price * quantity)
            book_notional: Top-of-book notional (best bid/ask size * price)
            short_term_vol: Short-term realized volatility (e.g., 30-second)

        Returns:
            Expected slippage in basis points (always non-negative)
        """
        # Spread component
        spread_impact = self.spread_coef * spread_bps

        # Size impact component (capped for thin books or invalid book size)
        if book_notional > 0:
            size_ratio = order_notional / book_notional
            size_impact = self.size_coef * size_ratio * 100
        else:
            # Invalid or empty book - use max impact
            size_impact = self.max_size_impact_bps

        # Cap size impact at maximum
        size_impact = min(size_impact, self.max_size_impact_bps)

        # Volatility component
        vol_impact = self.vol_coef * short_term_vol * 100

        return spread_impact + size_impact + vol_impact


@dataclass
class TransactionCostModel:
    """
    Full transaction cost model including slippage and fixed costs.

    This model combines the SlippageModel with fixed costs (commissions, fees)
    to provide complete fill price estimation for both buy and sell orders.

    For buy orders: fill_price = ask_price * (1 + total_cost_bps / 10000)
    For sell orders: fill_price = bid_price * (1 - total_cost_bps / 10000)

    Attributes:
        slippage_model: SlippageModel instance for slippage calculation
        fixed_cost_bps: Fixed commission/fees in basis points (default: 0.0)
    """

    slippage_model: SlippageModel
    fixed_cost_bps: float = field(default=0.0)

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate expected fill price including all costs.

        Args:
            side: Order side - "buy" or "sell"
            ask_price: Current best ask price
            bid_price: Current best bid price
            order_notional: Order size in notional terms
            book_notional: Top-of-book notional
            short_term_vol: Short-term realized volatility

        Returns:
            Expected fill price accounting for slippage and costs
        """
        midprice = (ask_price + bid_price) / 2
        spread_bps = (ask_price - bid_price) / midprice * 10000

        slippage_bps = self.slippage_model.calculate_slippage_bps(
            spread_bps=spread_bps,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        total_cost_bps = slippage_bps + self.fixed_cost_bps

        if side == "buy":
            # Buy: pay ask price + additional slippage
            return ask_price * (1 + total_cost_bps / 10000)
        else:
            # Sell: receive bid price - slippage
            return bid_price * (1 - total_cost_bps / 10000)

    def calculate_round_trip_cost_bps(
        self,
        spread_bps: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate total round-trip cost in basis points.

        This represents the cost of buying and then selling (or vice versa),
        useful for evaluating strategy profitability thresholds.

        Args:
            spread_bps: Current bid-ask spread in basis points
            order_notional: Order size in notional terms
            book_notional: Top-of-book notional
            short_term_vol: Short-term realized volatility

        Returns:
            Total round-trip cost in basis points
        """
        one_way = self.slippage_model.calculate_slippage_bps(
            spread_bps, order_notional, book_notional, short_term_vol
        )
        return 2 * (one_way + self.fixed_cost_bps)


def load_cost_model(config_path: str | Path) -> TransactionCostModel:
    """
    Load a transaction cost model from a JSON configuration file.

    The configuration file should contain the following fields (all optional,
    defaults will be used for missing fields):
    - spread_coef: Spread impact coefficient (default: 1.0)
    - size_coef: Size impact coefficient (default: 1.0)
    - vol_coef: Volatility impact coefficient (default: 1.0)
    - max_size_impact_bps: Maximum size impact cap (default: 50.0)
    - fixed_cost_bps: Fixed costs in bps (default: 0.0)

    Example config:
        {
            "version": "cm_v1",
            "spread_coef": 1.5,
            "size_coef": 0.8,
            "vol_coef": 2.0,
            "fixed_cost_bps": 0.35
        }

    Args:
        config_path: Path to JSON config file (str or Path)

    Returns:
        Configured TransactionCostModel instance

    Raises:
        FileNotFoundError: If config file does not exist
        json.JSONDecodeError: If config file contains invalid JSON
    """
    with open(config_path) as f:
        config = json.load(f)

    slippage_model = SlippageModel(
        spread_coef=config.get("spread_coef", 1.0),
        size_coef=config.get("size_coef", 1.0),
        vol_coef=config.get("vol_coef", 1.0),
        max_size_impact_bps=config.get("max_size_impact_bps", 50.0),
    )

    return TransactionCostModel(
        slippage_model=slippage_model,
        fixed_cost_bps=config.get("fixed_cost_bps", 0.0),
    )


# Convenience exports
__all__ = [
    "SlippageModel",
    "TransactionCostModel",
    "load_cost_model",
]
