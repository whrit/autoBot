"""
Signal Logger for logging trading signals to Parquet files.

This module provides signal logging capabilities for analysis and
backtesting comparison.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from runner_py.types import (
    ShadowExecutionResult,
    Signal,
)

logger = structlog.get_logger(__name__)


@dataclass
class SignalLogConfig:
    """
    Configuration for signal logging.

    Attributes:
        log_path: Directory path for storing log files
        buffer_size: Number of signals to buffer before flushing
        include_market_data: Whether to include market data in logs
    """

    log_path: Path
    buffer_size: int = 100
    include_market_data: bool = True


class SignalLogger:
    """
    Log signals to Parquet files for analysis.

    This logger buffers signals in memory and periodically flushes
    them to Parquet files for efficient storage and analysis.
    """

    def __init__(self, config: SignalLogConfig) -> None:
        """
        Initialize signal logger.

        Args:
            config: Signal log configuration
        """
        self.config = config
        self._buffer: list[dict[str, Any]] = []

        # Ensure log directory exists
        self.config.log_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "signal_logger_initialized",
            log_path=str(config.log_path),
            buffer_size=config.buffer_size,
        )

    def log_signal(
        self,
        signal: Signal,
        execution_result: ShadowExecutionResult | None = None,
    ) -> None:
        """
        Log a signal with optional execution result.

        Args:
            signal: The signal to log
            execution_result: Optional execution result for the signal
        """
        entry = self._create_log_entry(signal, execution_result)
        self._buffer.append(entry)

        # Auto-flush if buffer is full
        if len(self._buffer) >= self.config.buffer_size:
            self.flush()

    def log_batch(
        self,
        signals: list[tuple[Signal, ShadowExecutionResult | None]],
    ) -> None:
        """
        Log multiple signals at once.

        Args:
            signals: List of (signal, execution_result) tuples
        """
        for signal, result in signals:
            entry = self._create_log_entry(signal, result)
            self._buffer.append(entry)

        # Auto-flush if buffer is full
        if len(self._buffer) >= self.config.buffer_size:
            self.flush()

    def _create_log_entry(
        self,
        signal: Signal,
        execution_result: ShadowExecutionResult | None,
    ) -> dict[str, Any]:
        """Create a log entry from signal and result."""
        entry: dict[str, Any] = {
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "direction": signal.signal_type.value,
            "strength": signal.strength,
            "confidence": signal.strength,  # Alias
            "target_notional": signal.target_notional,
            "strategy_id": signal.strategy_id,
            "metadata": str(signal.metadata),
        }

        if execution_result is not None:
            entry["success"] = execution_result.success
            entry["execution_mode"] = execution_result.mode.value
            entry["message"] = execution_result.message

            if execution_result.fill is not None:
                entry["simulated_price"] = execution_result.fill.simulated_price
                entry["simulated_slippage_bps"] = execution_result.fill.simulated_slippage_bps
                entry["would_have_executed"] = execution_result.fill.would_have_executed
                entry["rejection_reason"] = execution_result.fill.rejection_reason
                entry["fill_timestamp"] = execution_result.fill.timestamp
            else:
                entry["simulated_price"] = None
                entry["simulated_slippage_bps"] = None
                entry["would_have_executed"] = None
                entry["rejection_reason"] = None
                entry["fill_timestamp"] = None
        else:
            entry["success"] = None
            entry["execution_mode"] = None
            entry["message"] = None
            entry["simulated_price"] = None
            entry["simulated_slippage_bps"] = None
            entry["would_have_executed"] = None
            entry["rejection_reason"] = None
            entry["fill_timestamp"] = None

        return entry

    def flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return

        # Create DataFrame from buffer
        df = pl.DataFrame(self._buffer)

        # Generate filename with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"signals_{timestamp}.parquet"
        filepath = self.config.log_path / filename

        # Write to Parquet
        df.write_parquet(filepath)

        logger.info(
            "signals_flushed",
            filepath=str(filepath),
            count=len(self._buffer),
        )

        # Clear buffer
        self._buffer.clear()

    def read_logs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_buffer: bool = False,
    ) -> pl.DataFrame:
        """
        Read logged signals within time range.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            include_buffer: Whether to include unflushed buffer

        Returns:
            DataFrame of logged signals
        """
        # Get all parquet files
        files = sorted(self.config.log_path.glob("signals_*.parquet"))

        if not files and not (include_buffer and self._buffer):
            # Return empty DataFrame with correct schema
            return self._empty_dataframe()

        # Read and concatenate all files
        dfs = []
        for file in files:
            try:
                df = pl.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.warning("failed_to_read_parquet", file=str(file), error=str(e))

        # Include buffer if requested
        if include_buffer and self._buffer:
            buffer_df = pl.DataFrame(self._buffer)
            dfs.append(buffer_df)

        if not dfs:
            return self._empty_dataframe()

        # Concatenate all DataFrames
        combined = pl.concat(dfs, how="diagonal")

        # Apply time filters
        if start_time is not None:
            combined = combined.filter(pl.col("timestamp") >= start_time)
        if end_time is not None:
            combined = combined.filter(pl.col("timestamp") <= end_time)

        return combined.sort("timestamp")

    def _empty_dataframe(self) -> pl.DataFrame:
        """Create an empty DataFrame with the correct schema."""
        return pl.DataFrame({
            "timestamp": [],
            "symbol": [],
            "direction": [],
            "strength": [],
            "confidence": [],
            "target_notional": [],
            "strategy_id": [],
            "metadata": [],
            "success": [],
            "execution_mode": [],
            "message": [],
            "simulated_price": [],
            "simulated_slippage_bps": [],
            "would_have_executed": [],
            "rejection_reason": [],
            "fill_timestamp": [],
        }).cast({
            "timestamp": pl.Datetime("us", "UTC"),
            "symbol": pl.Utf8,
            "direction": pl.Utf8,
            "strength": pl.Float64,
            "confidence": pl.Float64,
            "target_notional": pl.Float64,
            "strategy_id": pl.Utf8,
            "metadata": pl.Utf8,
            "success": pl.Boolean,
            "execution_mode": pl.Utf8,
            "message": pl.Utf8,
            "simulated_price": pl.Float64,
            "simulated_slippage_bps": pl.Float64,
            "would_have_executed": pl.Boolean,
            "rejection_reason": pl.Utf8,
            "fill_timestamp": pl.Datetime("us", "UTC"),
        })

    def get_signal_stats(self) -> dict[str, Any]:
        """
        Get statistics about logged signals.

        Returns:
            Dictionary of signal statistics
        """
        # Read all logs including buffer
        df = self.read_logs(include_buffer=True)

        if len(df) == 0:
            return {
                "total_signals": 0,
                "by_direction": {},
                "by_symbol": {},
                "by_strategy": {},
            }

        # Calculate statistics
        stats: dict[str, Any] = {
            "total_signals": len(df),
        }

        # By direction
        direction_counts = df.group_by("direction").len()
        stats["by_direction"] = {
            row["direction"]: row["len"]
            for row in direction_counts.to_dicts()
        }

        # By symbol
        symbol_counts = df.group_by("symbol").len()
        stats["by_symbol"] = {
            row["symbol"]: row["len"]
            for row in symbol_counts.to_dicts()
        }

        # By strategy
        strategy_counts = df.group_by("strategy_id").len()
        stats["by_strategy"] = {
            row["strategy_id"]: row["len"]
            for row in strategy_counts.to_dicts()
        }

        # Success rate if available
        if "success" in df.columns:
            success_df = df.filter(pl.col("success").is_not_null())
            if len(success_df) > 0:
                success_count = success_df.filter(pl.col("success")).height
                stats["success_rate"] = success_count / len(success_df)

        return stats
