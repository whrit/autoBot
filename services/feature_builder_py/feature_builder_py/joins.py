"""
As-Of Joiner (T2.03).

Implements point-in-time correct joins to prevent lookahead bias.
Features must only use data available at decision time.
"""

from datetime import timedelta
from uuid import uuid4

import polars as pl


class AsOfJoiner:
    """
    Perform as-of joins to prevent lookahead bias.

    As-of joins ensure that for each decision timestamp, only data
    that was available at that time is used (no future data leakage).
    """

    def __init__(
        self,
        tolerance: timedelta | None = None,
        strict: bool = False,
    ) -> None:
        """
        Initialize the as-of joiner.

        Args:
            tolerance: Maximum time difference for valid join.
                      If None, join to most recent available data.
            strict: If True, raise ValueError when no matching data found.
        """
        self.tolerance = tolerance
        self.strict = strict

    def join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on: str,
        by: str,
        suffix: str = "",
    ) -> pl.DataFrame:
        """
        Perform an as-of join.

        For each row in left, find the most recent row in right
        where right[by] <= left[on].

        Args:
            left: Left DataFrame (contains decision timestamps).
            right: Right DataFrame (contains feature data).
            on: Column name in left for decision timestamps.
            by: Column name in right for feature timestamps.
            suffix: Suffix to add to right columns (except by column).

        Returns:
            Joined DataFrame with all columns from left and right.

        Raises:
            ValueError: If strict=True and no matching data found.
        """
        if left.is_empty():
            return self._add_empty_columns(left, right, by, suffix)

        if right.is_empty():
            return self._add_null_columns(left, right, by, suffix)

        # Ensure both DataFrames are sorted by their time columns
        left_sorted = left.sort(on)
        right_sorted = right.sort(by)

        # Get column names from right (excluding join column)
        right_columns = [c for c in right_sorted.columns if c != by]

        # Create unique temporary name for join column to avoid conflicts
        unique_id = str(uuid4())[:8]
        temp_join_col = f"__asof_join_key_{unique_id}"

        # Rename the join column in right to the temporary name
        right_sorted = right_sorted.rename({by: temp_join_col})

        # Now rename other columns to avoid conflicts
        rename_map = {}
        final_names = {}

        for col in right_columns:
            # Generate temporary and final names
            temp_name = f"__temp_{col}_{unique_id}"
            final_name = f"{col}{suffix}" if suffix else col

            # Check if final name would conflict with existing columns
            if final_name in left_sorted.columns:
                final_name = f"{col}_{unique_id}"

            rename_map[col] = temp_name
            final_names[temp_name] = final_name

        if rename_map:
            right_sorted = right_sorted.rename(rename_map)

        # Perform as-of join using polars join_asof
        # coalesce=True means the join key from right is dropped
        if self.tolerance is not None:
            result = left_sorted.join_asof(
                right_sorted,
                left_on=on,
                right_on=temp_join_col,
                strategy="backward",
                tolerance=self.tolerance,
                coalesce=True,
            )
        else:
            result = left_sorted.join_asof(
                right_sorted,
                left_on=on,
                right_on=temp_join_col,
                strategy="backward",
                coalesce=True,
            )

        # Rename temporary columns to final names
        if final_names:
            result = result.rename(final_names)

        # Check for missing joins in strict mode
        if self.strict:
            joined_cols = list(final_names.values())
            for col in joined_cols:
                if col in result.columns and result[col].null_count() > 0:
                    raise ValueError(
                        f"As-of join has missing values in column '{col}'. "
                        "No matching data found for some decision timestamps."
                    )

        return result

    def join_multi(
        self,
        left: pl.DataFrame,
        sources: list[tuple[pl.DataFrame, str, str]],
        on: str,
    ) -> pl.DataFrame:
        """
        Perform multiple as-of joins sequentially.

        Args:
            left: Left DataFrame (contains decision timestamps).
            sources: List of (DataFrame, timestamp_col, suffix) tuples.
            on: Column name in left for decision timestamps.

        Returns:
            DataFrame with all sources joined.
        """
        result = left

        for right_df, by_col, suffix in sources:
            result = self.join(result, right_df, on=on, by=by_col, suffix=suffix)

        return result

    def _add_empty_columns(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        by: str,
        suffix: str,
    ) -> pl.DataFrame:
        """Add empty columns from right to empty left DataFrame."""
        for col in right.columns:
            if col != by:
                col_name = f"{col}{suffix}" if suffix else col
                if col_name not in left.columns:
                    dtype = right.schema[col]
                    left = left.with_columns(
                        pl.lit(None).cast(dtype).alias(col_name)
                    )
        return left

    def _add_null_columns(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        by: str,
        suffix: str,
    ) -> pl.DataFrame:
        """Add null columns from right to left DataFrame."""
        for col in right.columns:
            if col != by:
                col_name = f"{col}{suffix}" if suffix else col
                if col_name not in left.columns:
                    dtype = right.schema[col]
                    left = left.with_columns(
                        pl.lit(None).cast(dtype).alias(col_name)
                    )
        return left

    def validate_no_leakage(
        self,
        result: pl.DataFrame,
        decision_col: str,
        feature_time_col: str,
    ) -> bool:
        """
        Validate that no future data leakage occurred.

        Args:
            result: Joined DataFrame.
            decision_col: Column with decision timestamps.
            feature_time_col: Column with feature timestamps.

        Returns:
            True if no leakage detected.

        Raises:
            ValueError: If future data leakage detected.
        """
        if feature_time_col not in result.columns:
            return True

        # Check that all feature timestamps are <= decision timestamps
        leakage_mask = result.filter(
            pl.col(feature_time_col) > pl.col(decision_col)
        )

        if len(leakage_mask) > 0:
            raise ValueError(
                f"Future data leakage detected: {len(leakage_mask)} rows have "
                f"feature timestamps after decision timestamps."
            )

        return True
