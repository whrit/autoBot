"""
Fill Simulation Module (T3.02).

Implements realistic taker execution simulation:
- Buy orders fill at ASK price + slippage
- Sell orders fill at BID price - slippage
- No fills if quote is stale or crossed
- Tracks fill latency

Example usage:
    >>> from backtester_py.fills import FillSimulator
    >>> simulator = FillSimulator(cost_model)
    >>> result = simulator.simulate_fill("buy", 150.0, 150.02, 1000, 900, 10000, 0.001, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from backtester_py.slippage import SlippageIntegrator

if TYPE_CHECKING:
    from cost_models import TransactionCostModel


@dataclass
class FillResult:
    """
    Result of a fill simulation attempt.

    Attributes:
        filled: Whether the order was filled.
        fill_price: The fill price (None if not filled).
        fill_time: The time of fill (None if not filled).
        slippage_bps: Slippage in basis points (None if not filled).
        latency_ms: Quote-to-order latency in milliseconds.
        reject_reason: Reason for rejection (None if filled).
    """

    filled: bool
    fill_price: float | None = None
    fill_time: datetime | None = None
    slippage_bps: float | None = None
    latency_ms: float | None = None
    reject_reason: str | None = None


class FillSimulator:
    """
    Simulates realistic taker-style order fills.

    Implements quote-based fill logic where:
    - Buy orders fill at the ask price plus slippage
    - Sell orders fill at the bid price minus slippage
    - Orders are rejected if quotes are stale or the market is crossed

    Attributes:
        cost_model: Transaction cost model for slippage calculation.
        max_quote_age_seconds: Maximum age of quotes to accept (default: 5.0).
        slippage_integrator: Wrapper for slippage calculations.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        max_quote_age_seconds: float = 5.0,
    ) -> None:
        """
        Initialize the fill simulator.

        Args:
            cost_model: TransactionCostModel for slippage calculations.
            max_quote_age_seconds: Maximum quote age to accept for fills.
        """
        self.cost_model = cost_model
        self.max_quote_age_seconds = max_quote_age_seconds
        self.slippage_integrator = SlippageIntegrator(cost_model)

    def simulate_fill(
        self,
        side: Literal["buy", "sell"],
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        order_notional: float,
        short_term_vol: float,
        order_time: datetime,
        quote_time: datetime,
    ) -> FillResult:
        """
        Simulate an order fill based on current quotes.

        Args:
            side: Order side ("buy" or "sell").
            bid_price: Current best bid price.
            ask_price: Current best ask price.
            bid_size: Size available at best bid.
            ask_size: Size available at best ask.
            order_notional: Order size in notional terms.
            short_term_vol: Short-term realized volatility.
            order_time: Time the order was placed.
            quote_time: Time of the quote data.

        Returns:
            FillResult with fill details or rejection reason.
        """
        # Calculate latency
        latency_ms = self._calculate_latency_ms(order_time, quote_time)

        # Check for stale quote
        if self._is_quote_stale(order_time, quote_time):
            return FillResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason="stale_quote",
            )

        # Check for crossed market
        if self._is_market_crossed(bid_price, ask_price):
            return FillResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason="crossed_market",
            )

        # Check for liquidity
        if not self._has_liquidity(side, bid_size, ask_size):
            return FillResult(
                filled=False,
                latency_ms=latency_ms,
                reject_reason="no_liquidity",
            )

        # Calculate book notional
        book_notional = ask_price * ask_size if side == "buy" else bid_price * bid_size

        # Calculate fill price with slippage
        fill_price, slippage_bps = self.slippage_integrator.calculate_fill_price(
            side=side,
            bid_price=bid_price,
            ask_price=ask_price,
            order_notional=order_notional,
            book_notional=book_notional,
            short_term_vol=short_term_vol,
        )

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_time=order_time,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        )

    def _calculate_latency_ms(
        self, order_time: datetime, quote_time: datetime
    ) -> float:
        """Calculate latency between quote and order in milliseconds."""
        # Ensure both are timezone-aware
        if order_time.tzinfo is None:
            order_time = order_time.replace(tzinfo=UTC)
        if quote_time.tzinfo is None:
            quote_time = quote_time.replace(tzinfo=UTC)

        delta = order_time - quote_time
        return delta.total_seconds() * 1000

    def _is_quote_stale(
        self, order_time: datetime, quote_time: datetime
    ) -> bool:
        """Check if quote is stale (older than max age)."""
        latency_ms = self._calculate_latency_ms(order_time, quote_time)
        return latency_ms > self.max_quote_age_seconds * 1000

    def _is_market_crossed(self, bid_price: float, ask_price: float) -> bool:
        """Check if market is crossed (bid > ask)."""
        return bid_price > ask_price

    def _has_liquidity(
        self, side: Literal["buy", "sell"], bid_size: float, ask_size: float
    ) -> bool:
        """Check if there's liquidity on the relevant side."""
        if side == "buy":
            return ask_size > 0
        else:
            return bid_size > 0


__all__ = ["FillSimulator", "FillResult"]
