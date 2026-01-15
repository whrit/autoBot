"""
Forward Returns Calculation Module (T2.06).

Calculates forward returns at configurable horizons using midprice or microprice.
Critical: Uses quote-based prices (midprice/microprice), NOT last trade price,
for accurate taker-realistic returns.

Example usage:
    >>> calculator = ForwardReturnsCalculator(horizons=[60, 300, 900])
    >>> returns_df = calculator.calculate(
    ...     quotes_df=quotes,
    ...     decision_times=decision_times,
    ...     symbol="SPY",
    ... )
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


class ForwardReturnsCalculator:
    """
    Calculate forward returns at configurable horizons.

    Uses midprice or microprice from quotes (NOT last trade price) for
    accurate taker-realistic return calculations.

    Forward return formula:
        return = (future_price - current_price) / current_price

    Attributes:
        horizons: List of horizon periods in seconds (default: [60, 300, 900])
    """

    def __init__(self, horizons: Sequence[int] | None = None) -> None:
        """
        Initialize ForwardReturnsCalculator.

        Args:
            horizons: List of forward-looking horizons in seconds.
                      Default is [60, 300, 900] (1min, 5min, 15min).

        Raises:
            ValueError: If any horizon is not positive.
        """
        if horizons is None:
            horizons = [60, 300, 900]

        self.horizons = list(horizons)

        # Validate horizons
        for h in self.horizons:
            if h <= 0:
                raise ValueError(f"Horizons must be positive integers, got {h}")

    def _calculate_midprice(self, bid_price: float, ask_price: float) -> float:
        """
        Calculate midprice (arithmetic mean of bid and ask).

        Args:
            bid_price: Best bid price
            ask_price: Best ask price

        Returns:
            Midprice: (bid + ask) / 2
        """
        return (bid_price + ask_price) / 2

    def _calculate_microprice(
        self,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        """
        Calculate microprice (size-weighted midprice).

        The microprice weights the bid/ask prices by the opposite side's size,
        giving a more realistic estimate of fair value based on order book imbalance.

        Formula:
            microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

        Falls back to midprice when total size is zero.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Size at best bid
            ask_size: Size at best ask

        Returns:
            Microprice, or midprice if total size is zero
        """
        total_size = bid_size + ask_size
        if total_size == 0:
            return self._calculate_midprice(bid_price, ask_price)

        return (bid_price * ask_size + ask_price * bid_size) / total_size

    def _calculate_forward_return(
        self, current_price: float, future_price: float
    ) -> float:
        """
        Calculate simple forward return.

        Args:
            current_price: Price at decision time
            future_price: Price at horizon

        Returns:
            Forward return: (future - current) / current
        """
        return (future_price - current_price) / current_price

    def calculate(
        self,
        quotes_df: pl.DataFrame,
        decision_times: Sequence[datetime],
        symbol: str,
        use_microprice: bool = False,
    ) -> pl.DataFrame:
        """
        Calculate forward returns for given decision times and horizons.

        Args:
            quotes_df: DataFrame with columns:
                - symbol: str
                - ts_event: datetime (timezone-aware)
                - bid_price: float
                - ask_price: float
                - bid_size: float
                - ask_size: float
            decision_times: List of decision timestamps
            symbol: Symbol to filter quotes for
            use_microprice: If True, use microprice; if False, use midprice (default)

        Returns:
            DataFrame with columns:
                - symbol: str
                - decision_ts: datetime
                - horizon: int (seconds)
                - fwd_return_mid: float
        """
        if len(decision_times) == 0 or quotes_df.is_empty():
            return pl.DataFrame(
                {
                    "symbol": pl.Series([], dtype=pl.Utf8),
                    "decision_ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "fwd_return_mid": pl.Series([], dtype=pl.Float64),
                }
            )

        # Filter quotes for the symbol
        symbol_quotes = quotes_df.filter(pl.col("symbol") == symbol)

        if symbol_quotes.is_empty():
            return pl.DataFrame(
                {
                    "symbol": pl.Series([], dtype=pl.Utf8),
                    "decision_ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "fwd_return_mid": pl.Series([], dtype=pl.Float64),
                }
            )

        # Sort by timestamp for efficient lookups
        symbol_quotes = symbol_quotes.sort("ts_event")

        # Calculate price column based on method
        if use_microprice:
            symbol_quotes = symbol_quotes.with_columns(
                [
                    (
                        (
                            pl.col("bid_price") * pl.col("ask_size")
                            + pl.col("ask_price") * pl.col("bid_size")
                        )
                        / (pl.col("bid_size") + pl.col("ask_size"))
                    )
                    .fill_null((pl.col("bid_price") + pl.col("ask_price")) / 2)
                    .alias("price")
                ]
            )
        else:
            symbol_quotes = symbol_quotes.with_columns(
                [((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("price")]
            )

        # Build results
        results: list[dict[str, str | datetime | int | float]] = []

        for decision_ts in decision_times:
            # Find quote at decision time (or nearest before)
            current_quotes = symbol_quotes.filter(pl.col("ts_event") <= decision_ts)
            if current_quotes.is_empty():
                continue

            current_row = current_quotes.tail(1)
            current_price: float = current_row["price"][0]

            for horizon in self.horizons:
                future_ts = decision_ts + timedelta(seconds=horizon)

                # Find quote at future time (or nearest before)
                future_quotes = symbol_quotes.filter(pl.col("ts_event") <= future_ts)
                if future_quotes.is_empty():
                    continue

                # Check if we have data at or after the horizon timestamp
                # (ensuring we actually have forward-looking data)
                last_quote_ts_result = future_quotes["ts_event"].max()
                if last_quote_ts_result is None:
                    continue

                # Cast to datetime for comparison
                last_quote_ts = datetime.fromisoformat(
                    str(last_quote_ts_result)
                ) if not isinstance(last_quote_ts_result, datetime) else last_quote_ts_result

                min_required_ts = decision_ts + timedelta(seconds=horizon - 1)
                if last_quote_ts < min_required_ts:
                    # Not enough forward data for this horizon
                    continue

                future_row = future_quotes.tail(1)
                future_price: float = future_row["price"][0]

                fwd_return = self._calculate_forward_return(current_price, future_price)

                results.append(
                    {
                        "symbol": symbol,
                        "decision_ts": decision_ts,
                        "horizon": horizon,
                        "fwd_return_mid": fwd_return,
                    }
                )

        if not results:
            return pl.DataFrame(
                {
                    "symbol": pl.Series([], dtype=pl.Utf8),
                    "decision_ts": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "fwd_return_mid": pl.Series([], dtype=pl.Float64),
                }
            )

        return pl.DataFrame(results)
