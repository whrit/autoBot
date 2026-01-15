"""
Direction Labeling Module (T2.07).

Converts forward returns to direction labels (-1, 0, +1) with a configurable
no-trade band. Returns within the no-trade threshold are labeled 0 to filter
out noise and unprofitable trades.

Example usage:
    >>> labeler = DirectionLabeler(no_trade_threshold=0.0005)
    >>> labeled_df = labeler.label_returns(returns_df)
"""

from __future__ import annotations

import polars as pl


class DirectionLabeler:
    """
    Label forward returns with direction indicators.

    Converts continuous forward returns to discrete direction labels:
        - +1: Long signal (return > threshold)
        -  0: No-trade (return within +/- threshold)
        - -1: Short signal (return < -threshold)

    The no-trade band filters out small returns that would be unprofitable
    after transaction costs.

    Attributes:
        no_trade_threshold: Minimum absolute return to trigger a trade signal.
                           Default is 0.0005 (5 basis points).
    """

    def __init__(self, no_trade_threshold: float = 0.0005) -> None:
        """
        Initialize DirectionLabeler.

        Args:
            no_trade_threshold: Minimum absolute return required for a trade
                               signal. Returns within +/- this threshold are
                               labeled as 0 (no-trade). Default is 0.0005 (5 bps).

        Raises:
            ValueError: If threshold is negative.
        """
        if no_trade_threshold < 0:
            raise ValueError(
                f"no_trade_threshold must be non-negative, got {no_trade_threshold}"
            )

        self.no_trade_threshold = no_trade_threshold

    def label_direction(self, forward_return: float) -> int:
        """
        Label a single forward return with direction.

        Args:
            forward_return: The forward return value

        Returns:
            Direction label:
                +1 if forward_return > threshold
                -1 if forward_return < -threshold
                 0 otherwise (within no-trade band)
        """
        if forward_return > self.no_trade_threshold:
            return 1
        elif forward_return < -self.no_trade_threshold:
            return -1
        else:
            return 0

    def label_returns(
        self,
        returns_df: pl.DataFrame,
        return_column: str = "fwd_return_mid",
    ) -> pl.DataFrame:
        """
        Label all returns in a DataFrame with directions.

        Args:
            returns_df: DataFrame containing forward returns
            return_column: Name of the column containing returns to label.
                          Default is "fwd_return_mid".

        Returns:
            DataFrame with added "direction" column containing labels (-1, 0, +1)
        """
        threshold = self.no_trade_threshold

        return returns_df.with_columns(
            [
                pl.when(pl.col(return_column) > threshold)
                .then(pl.lit(1))
                .when(pl.col(return_column) < -threshold)
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .alias("direction")
            ]
        )
