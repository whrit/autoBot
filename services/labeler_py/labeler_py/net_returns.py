"""
Net-of-Spread Returns Module (T2.08).

Calculates returns net of expected transaction costs (spread crossing).
A trade is only profitable if the return exceeds the spread + slippage costs.

Example usage:
    >>> calculator = NetOfSpreadCalculator(half_spread_factor=1.0)
    >>> net_returns_df = calculator.calculate_net_returns(returns_df)
"""

from __future__ import annotations

import polars as pl


class NetOfSpreadCalculator:
    """
    Calculate net-of-spread returns.

    For taker-style trading, each trade incurs a cost of crossing the spread.
    This calculator adjusts gross returns by subtracting the half-spread cost
    (one-way transaction cost).

    The half-spread cost represents the immediate loss from executing at
    ask (for buys) or bid (for sells) rather than at midprice.

    Attributes:
        half_spread_factor: Multiplier for half-spread cost. Default 1.0.
                           Use >1.0 to account for additional slippage.
    """

    def __init__(self, half_spread_factor: float = 1.0) -> None:
        """
        Initialize NetOfSpreadCalculator.

        Args:
            half_spread_factor: Multiplier for the half-spread cost.
                               1.0 = just the half spread
                               1.5 = half spread + 50% slippage buffer
        """
        self.half_spread_factor = half_spread_factor

    def calculate_spread_cost(self, bid_price: float, ask_price: float) -> float:
        """
        Calculate the one-way spread cost as a fraction of price.

        The spread cost is half the bid-ask spread divided by the midprice,
        representing the cost of crossing from mid to the execution price.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price

        Returns:
            Spread cost as a decimal (e.g., 0.0001 = 1 basis point)
        """
        midprice = (bid_price + ask_price) / 2
        half_spread = (ask_price - bid_price) / 2
        return (half_spread / midprice) * self.half_spread_factor

    def calculate_net_return(
        self,
        gross_return: float,
        bid_price: float,
        ask_price: float,
    ) -> float:
        """
        Calculate net return after spread costs.

        Net return = gross_return - spread_cost

        For a round-trip trade (entry + exit), the total cost would be
        2x the half-spread cost. This function calculates for one leg.

        Args:
            gross_return: The raw forward return
            bid_price: Best bid price at decision time
            ask_price: Best ask price at decision time

        Returns:
            Net return after subtracting spread cost
        """
        spread_cost = self.calculate_spread_cost(bid_price, ask_price)
        return gross_return - spread_cost

    def calculate_net_returns(self, returns_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate net returns for a DataFrame of forward returns.

        Requires the DataFrame to have columns:
            - fwd_return_mid: Gross forward return
            - bid_price: Best bid at decision time
            - ask_price: Best ask at decision time

        Args:
            returns_df: DataFrame with returns and quote data

        Returns:
            DataFrame with added "fwd_return_net" column
        """
        factor = self.half_spread_factor

        return returns_df.with_columns(
            [
                (
                    pl.col("fwd_return_mid")
                    - (
                        (pl.col("ask_price") - pl.col("bid_price"))
                        / (pl.col("bid_price") + pl.col("ask_price"))
                        * factor
                    )
                ).alias("fwd_return_net")
            ]
        )
