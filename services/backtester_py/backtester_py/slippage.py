"""
Slippage Integration Module (T3.03).

Wraps the cost_models library to provide slippage calculations
for the backtester. Implements the slippage formula:
    slippage_bps = a*spread + b*(order/book) + c*vol

Example usage:
    >>> from backtester_py.slippage import SlippageIntegrator
    >>> from cost_models import TransactionCostModel, SlippageModel
    >>> integrator = SlippageIntegrator(cost_model)
    >>> slippage = integrator.calculate_slippage("buy", 150.0, 150.02, 10000, 150000, 0.001)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cost_models import TransactionCostModel


class SlippageIntegrator:
    """
    Integrates the cost_models library for slippage calculations.

    Wraps TransactionCostModel to provide a simplified interface for
    the backtester's fill simulation.

    Attributes:
        cost_model: The underlying TransactionCostModel instance.
    """

    def __init__(self, cost_model: TransactionCostModel) -> None:
        """
        Initialize the slippage integrator.

        Args:
            cost_model: TransactionCostModel instance from cost_models library.
        """
        self.cost_model = cost_model

    def calculate_slippage(
        self,
        side: Literal["buy", "sell"],
        bid_price: float,
        ask_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> float:
        """
        Calculate expected slippage in basis points.

        Uses the cost model formula:
            slippage_bps = a*spread_bps + b*(order/book)*100 + c*vol*100

        Args:
            side: Order side ("buy" or "sell").
            bid_price: Current best bid price.
            ask_price: Current best ask price.
            order_notional: Order size in notional terms.
            book_notional: Top-of-book notional (best bid/ask size * price).
            short_term_vol: Short-term realized volatility (e.g., 30-second).

        Returns:
            Expected slippage in basis points (always non-negative).
        """
        if bid_price <= 0 or ask_price <= 0:
            return 0.0

        # Calculate spread in basis points
        midprice = (ask_price + bid_price) / 2
        spread_bps = (ask_price - bid_price) / midprice * 10000

        return float(
            self.cost_model.slippage_model.calculate_slippage_bps(
                spread_bps=spread_bps,
                order_notional=order_notional,
                book_notional=book_notional,
                short_term_vol=short_term_vol,
            )
        )

    def apply_slippage(
        self,
        side: Literal["buy", "sell"],
        base_price: float,
        slippage_bps: float,
    ) -> float:
        """
        Apply slippage to a base price.

        For buy orders: price increases (worse fill)
        For sell orders: price decreases (worse fill)

        Args:
            side: Order side ("buy" or "sell").
            base_price: The base price (ask for buy, bid for sell).
            slippage_bps: Slippage in basis points.

        Returns:
            Adjusted fill price.
        """
        if side == "buy":
            # Buy: pay more due to slippage
            return base_price * (1 + slippage_bps / 10000)
        else:
            # Sell: receive less due to slippage
            return base_price * (1 - slippage_bps / 10000)

    def calculate_fill_price(
        self,
        side: Literal["buy", "sell"],
        bid_price: float,
        ask_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float,
    ) -> tuple[float, float]:
        """
        Calculate expected fill price including slippage.

        Convenience method that combines slippage calculation
        and application in one call.

        Args:
            side: Order side ("buy" or "sell").
            bid_price: Current best bid price.
            ask_price: Current best ask price.
            order_notional: Order size in notional terms.
            book_notional: Top-of-book notional.
            short_term_vol: Short-term realized volatility.

        Returns:
            Tuple of (fill_price, slippage_bps).
        """
        slippage_bps = self.calculate_slippage(
            side=side,
            bid_price=bid_price,
            ask_price=ask_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        # Add fixed costs
        total_cost_bps = slippage_bps + self.cost_model.fixed_cost_bps

        # Apply to base price
        base_price = ask_price if side == "buy" else bid_price
        fill_price = self.apply_slippage(side, base_price, total_cost_bps)

        return fill_price, total_cost_bps


__all__ = ["SlippageIntegrator"]
