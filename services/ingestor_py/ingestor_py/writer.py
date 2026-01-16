"""
Parquet Writer - Write market data to partitioned Parquet files.

Partitioning scheme: lake/raw/{data_type}/dt=YYYY-MM-DD/symbol=XXX/
Includes comprehensive logging for file writes, partition creation,
and compression statistics.

Memory optimizations:
- Configurable batch sizes for processing large datasets
- Write buffering to reduce I/O operations
- Multiple compression codec options
- Memory usage logging for debugging
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from ingestor_py.logging_config import format_bytes, get_logger

logger = get_logger(__name__)

# Type variable for generic record types
T = TypeVar("T", bound=dict[str, object])


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        batch_size: Number of records to process in each batch.
                   Lower values reduce peak memory but increase I/O.
        buffer_size: Number of records to buffer before writing.
                    Set to 0 for immediate writes.
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'lz4', 'none').
        compression_level: Compression level (codec-dependent).
        enable_memory_logging: Log memory usage during processing.
        gc_after_batch: Run garbage collection after each batch.
    """
    batch_size: int = 10_000
    buffer_size: int = 5_000
    compression: str = "snappy"
    compression_level: int | None = None
    enable_memory_logging: bool = False
    gc_after_batch: bool = False


@dataclass
class MemoryStats:
    """Track memory usage during processing.

    Attributes:
        peak_memory_bytes: Peak memory usage in bytes.
        current_memory_bytes: Current memory usage in bytes.
        records_processed: Total records processed.
        batches_processed: Total batches processed.
    """
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    records_processed: int = 0
    batches_processed: int = 0

    def update(self) -> None:
        """Update memory stats with current process memory."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.current_memory_bytes = usage.ru_maxrss * 1024  # Convert KB to bytes
            self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)
        except (ImportError, AttributeError):
            # resource module not available on all platforms
            pass


def iter_batches(
    records: Sequence[T],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """Iterate over records in batches.

    Args:
        records: Sequence of records to batch.
        batch_size: Size of each batch.

    Yields:
        List of records for each batch.
    """
    total = len(records)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield list(records[start:end])


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes.
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB
    except (ImportError, AttributeError):
        return 0.0

# Schema definitions for each data type
TRADE_SCHEMA = pa.schema(
    [
        ("ts_event", pa.timestamp("us", tz="UTC")),
        ("ts_recv", pa.timestamp("us", tz="UTC")),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("exchange", pa.string()),
        ("conditions", pa.string()),
    ]
)

QUOTE_SCHEMA = pa.schema(
    [
        ("ts_event", pa.timestamp("us", tz="UTC")),
        ("ts_recv", pa.timestamp("us", tz="UTC")),
        ("bid_price", pa.float64()),
        ("bid_size", pa.float64()),
        ("ask_price", pa.float64()),
        ("ask_size", pa.float64()),
        ("bid_exchange", pa.string()),
        ("ask_exchange", pa.string()),
        ("conditions", pa.string()),
    ]
)

BAR_SCHEMA = pa.schema(
    [
        ("ts_event", pa.timestamp("us", tz="UTC")),
        ("ts_recv", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("vwap", pa.float64()),
    ]
)


def _extract_date(ts: datetime) -> str:
    """Extract date string from timestamp."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.strftime("%Y-%m-%d")


def _group_by_partition(
    records: Sequence[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Group records by (date, symbol) partition keys."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}

    for record in records:
        ts_event = record["ts_event"]
        if not isinstance(ts_event, datetime):
            continue

        date_str = _extract_date(ts_event)
        symbol = str(record.get("symbol", "UNKNOWN"))
        key = (date_str, symbol)

        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    return groups


class ParquetWriter:
    """Write market data to partitioned Parquet files.

    Directory structure:
        base_path/raw/{data_type}/dt=YYYY-MM-DD/symbol=XXX/data.parquet

    Includes logging for:
    - File writes with path and size
    - Partition creation
    - Compression statistics
    - Memory usage (optional)

    Memory optimizations:
    - Configurable batch sizes for large datasets
    - Write buffering to reduce I/O
    - Multiple compression codec support
    - Optional garbage collection after batches
    """

    # Supported compression codecs
    VALID_COMPRESSION = {"snappy", "gzip", "zstd", "lz4", "none"}

    def __init__(
        self,
        base_path: Path | str,
        batch_config: BatchConfig | None = None,
    ) -> None:
        """Initialize the writer.

        Args:
            base_path: Base path for the data lake.
            batch_config: Optional batch processing configuration.
                         If None, uses default BatchConfig.
        """
        self.base_path = Path(base_path)
        self.batch_config = batch_config or BatchConfig()
        self._total_bytes_written = 0
        self._total_records_written = 0
        self._partitions_created: set[str] = set()
        self._memory_stats = MemoryStats()

        # Write buffer for batching I/O
        self._write_buffer: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        self._buffer_count = 0

        # Validate compression codec
        if self.batch_config.compression not in self.VALID_COMPRESSION:
            raise ValueError(
                f"Invalid compression: {self.batch_config.compression}. "
                f"Valid options: {self.VALID_COMPRESSION}"
            )

        logger.info(
            "parquet_writer_initialized",
            base_path=str(base_path),
            compression=self.batch_config.compression,
            batch_size=self.batch_config.batch_size,
            buffer_size=self.batch_config.buffer_size,
            memory_logging=self.batch_config.enable_memory_logging,
        )

    def _write_partition(
        self,
        records: list[dict[str, object]],
        data_type: str,
        date_str: str,
        symbol: str,
        schema: pa.Schema,
    ) -> int:
        """Write records to a partitioned Parquet file.

        Args:
            records: List of records to write.
            data_type: Type of data ('trades', 'quotes', 'bars_provider').
            date_str: Date string (YYYY-MM-DD).
            symbol: Stock symbol.
            schema: PyArrow schema for the data.

        Returns:
            Number of bytes written to the file.
        """
        # Build partition path
        partition_path = (
            self.base_path / "raw" / data_type / f"dt={date_str}" / f"symbol={symbol}"
        )

        # Check if this is a new partition
        partition_key = str(partition_path)
        is_new_partition = partition_key not in self._partitions_created

        if is_new_partition:
            partition_path.mkdir(parents=True, exist_ok=True)
            self._partitions_created.add(partition_key)

            logger.debug(
                "partition_created",
                path=partition_key,
                data_type=data_type,
                date=date_str,
                symbol=symbol,
            )

        # Prepare data for PyArrow (exclude symbol, already in partition)
        columns: dict[str, list[object]] = {col.name: [] for col in schema}

        for record in records:
            for col in schema:
                col_name = col.name
                value = record.get(col_name)

                # Handle datetime conversion - ensure timezone-aware
                if (
                    pa.types.is_timestamp(col.type)
                    and isinstance(value, datetime)
                    and value.tzinfo is None
                ):
                    value = value.replace(tzinfo=UTC)
                columns[col_name].append(value)

        # Create table and write
        table = pa.table(columns, schema=schema)
        file_path = partition_path / "data.parquet"

        # Prepare compression settings
        compression: str | None = self.batch_config.compression
        if compression == "none":
            compression = None

        compression_kwargs: dict[str, object] = {"compression": compression}
        if self.batch_config.compression_level is not None and compression:
            compression_kwargs["compression_level"] = self.batch_config.compression_level

        # Write with compression
        pq.write_table(
            table,
            file_path,
            **compression_kwargs,
        )

        # Get file size
        file_size = os.path.getsize(file_path)
        self._total_bytes_written += file_size
        self._total_records_written += len(records)

        # Calculate compression ratio (approximate)
        # Estimate uncompressed size based on schema
        estimated_uncompressed = len(records) * sum(
            8 if pa.types.is_floating(col.type) or pa.types.is_integer(col.type) else 50
            for col in schema
        )
        compression_ratio = (
            estimated_uncompressed / file_size if file_size > 0 else 1.0
        )

        logger.debug(
            "parquet_file_written",
            path=str(file_path),
            records=len(records),
            file_size=format_bytes(file_size),
            compression_ratio=f"{compression_ratio:.2f}x",
            data_type=data_type,
            symbol=symbol,
            date=date_str,
        )

        return file_size

    def write_trades(self, trades: Sequence[dict[str, object]]) -> int:
        """Write trades to partitioned Parquet files.

        Args:
            trades: Sequence of trade records.

        Returns:
            Total bytes written across all partitions.
        """
        if not trades:
            return 0

        total_bytes = 0
        groups = _group_by_partition(trades)

        log = logger.bind(data_type="trades", record_count=len(trades))
        log.debug("writing_trades", partitions=len(groups))

        for (date_str, symbol), records in groups.items():
            bytes_written = self._write_partition(
                records=records,
                data_type="trades",
                date_str=date_str,
                symbol=symbol,
                schema=TRADE_SCHEMA,
            )
            total_bytes += bytes_written

        log.info(
            "trades_written",
            partitions=len(groups),
            total_bytes=format_bytes(total_bytes),
        )

        return total_bytes

    def write_quotes(self, quotes: Sequence[dict[str, object]]) -> int:
        """Write quotes to partitioned Parquet files.

        Args:
            quotes: Sequence of quote records.

        Returns:
            Total bytes written across all partitions.
        """
        if not quotes:
            return 0

        total_bytes = 0
        groups = _group_by_partition(quotes)

        log = logger.bind(data_type="quotes", record_count=len(quotes))
        log.debug("writing_quotes", partitions=len(groups))

        for (date_str, symbol), records in groups.items():
            bytes_written = self._write_partition(
                records=records,
                data_type="quotes",
                date_str=date_str,
                symbol=symbol,
                schema=QUOTE_SCHEMA,
            )
            total_bytes += bytes_written

        log.info(
            "quotes_written",
            partitions=len(groups),
            total_bytes=format_bytes(total_bytes),
        )

        return total_bytes

    def write_bars(self, bars: Sequence[dict[str, object]]) -> int:
        """Write bars to partitioned Parquet files.

        Args:
            bars: Sequence of bar records.

        Returns:
            Total bytes written across all partitions.
        """
        if not bars:
            return 0

        total_bytes = 0
        groups = _group_by_partition(bars)

        log = logger.bind(data_type="bars", record_count=len(bars))
        log.debug("writing_bars", partitions=len(groups))

        for (date_str, symbol), records in groups.items():
            bytes_written = self._write_partition(
                records=records,
                data_type="bars_provider",
                date_str=date_str,
                symbol=symbol,
                schema=BAR_SCHEMA,
            )
            total_bytes += bytes_written

        log.info(
            "bars_written",
            partitions=len(groups),
            total_bytes=format_bytes(total_bytes),
        )

        return total_bytes

    def _add_to_buffer(
        self,
        records: list[dict[str, object]],
        data_type: str,
        date_str: str,
        symbol: str,
        schema: pa.Schema,
    ) -> int:
        """Add records to the write buffer.

        Args:
            records: Records to buffer.
            data_type: Type of data ('trades', 'quotes', 'bars_provider').
            date_str: Date string (YYYY-MM-DD).
            symbol: Stock symbol.
            schema: PyArrow schema for the data.

        Returns:
            Number of bytes written if buffer was flushed, otherwise 0.
        """
        key = (data_type, date_str, symbol)
        if key not in self._write_buffer:
            self._write_buffer[key] = []

        self._write_buffer[key].extend(records)
        self._buffer_count += len(records)

        bytes_written = 0

        # Flush if buffer exceeds threshold
        if self._buffer_count >= self.batch_config.buffer_size > 0:
            bytes_written = self._flush_buffer(schema)

        return bytes_written

    def _flush_buffer(self, schema: pa.Schema | None = None) -> int:
        """Flush all buffered records to disk.

        Args:
            schema: Optional schema to use for writing.
                   If None, infers schema from data type.

        Returns:
            Total bytes written during flush.
        """
        if not self._write_buffer:
            return 0

        total_bytes = 0

        for (data_type, date_str, symbol), records in self._write_buffer.items():
            if not records:
                continue

            # Determine schema based on data type
            write_schema = schema
            if write_schema is None:
                if data_type == "trades":
                    write_schema = TRADE_SCHEMA
                elif data_type == "quotes":
                    write_schema = QUOTE_SCHEMA
                elif data_type == "bars_provider":
                    write_schema = BAR_SCHEMA
                else:
                    logger.warning(
                        "unknown_data_type_in_buffer",
                        data_type=data_type,
                        records=len(records),
                    )
                    continue

            bytes_written = self._write_partition(
                records=records,
                data_type=data_type,
                date_str=date_str,
                symbol=symbol,
                schema=write_schema,
            )
            total_bytes += bytes_written

        # Clear buffer
        self._write_buffer.clear()
        self._buffer_count = 0

        # Optional memory cleanup
        if self.batch_config.gc_after_batch:
            gc.collect()

        # Log memory usage if enabled
        if self.batch_config.enable_memory_logging:
            self._memory_stats.update()
            logger.debug(
                "buffer_flushed_memory_stats",
                memory_mb=get_memory_usage_mb(),
                peak_memory_bytes=self._memory_stats.peak_memory_bytes,
                records_processed=self._memory_stats.records_processed,
            )

        return total_bytes

    def write_trades_batched(
        self,
        trades: Iterator[dict[str, object]] | Sequence[dict[str, object]],
    ) -> int:
        """Write trades using batch processing for memory efficiency.

        Processes records in configurable batches to limit peak memory usage.
        Useful for very large datasets that don't fit in memory.

        Args:
            trades: Iterator or sequence of trade records.

        Returns:
            Total bytes written across all partitions.
        """
        total_bytes = 0
        batch: list[dict[str, object]] = []

        # Convert to iterator if sequence
        trade_iter = iter(trades)

        for record in trade_iter:
            batch.append(record)

            if len(batch) >= self.batch_config.batch_size:
                # Process batch
                bytes_written = self.write_trades(batch)
                total_bytes += bytes_written

                # Update memory stats
                self._memory_stats.records_processed += len(batch)
                self._memory_stats.batches_processed += 1

                if self.batch_config.enable_memory_logging:
                    self._memory_stats.update()
                    logger.debug(
                        "batch_processed",
                        batch_number=self._memory_stats.batches_processed,
                        records_in_batch=len(batch),
                        total_records=self._memory_stats.records_processed,
                        memory_mb=get_memory_usage_mb(),
                    )

                # Clear batch and optionally run GC
                batch = []
                if self.batch_config.gc_after_batch:
                    gc.collect()

        # Process remaining records
        if batch:
            total_bytes += self.write_trades(batch)
            self._memory_stats.records_processed += len(batch)
            self._memory_stats.batches_processed += 1

        return total_bytes

    def write_quotes_batched(
        self,
        quotes: Iterator[dict[str, object]] | Sequence[dict[str, object]],
    ) -> int:
        """Write quotes using batch processing for memory efficiency.

        Args:
            quotes: Iterator or sequence of quote records.

        Returns:
            Total bytes written across all partitions.
        """
        total_bytes = 0
        batch: list[dict[str, object]] = []
        quote_iter = iter(quotes)

        for record in quote_iter:
            batch.append(record)

            if len(batch) >= self.batch_config.batch_size:
                bytes_written = self.write_quotes(batch)
                total_bytes += bytes_written
                self._memory_stats.records_processed += len(batch)
                self._memory_stats.batches_processed += 1

                if self.batch_config.enable_memory_logging:
                    self._memory_stats.update()
                    logger.debug(
                        "batch_processed",
                        data_type="quotes",
                        batch_number=self._memory_stats.batches_processed,
                        records_in_batch=len(batch),
                        memory_mb=get_memory_usage_mb(),
                    )

                batch = []
                if self.batch_config.gc_after_batch:
                    gc.collect()

        if batch:
            total_bytes += self.write_quotes(batch)
            self._memory_stats.records_processed += len(batch)
            self._memory_stats.batches_processed += 1

        return total_bytes

    def write_bars_batched(
        self,
        bars: Iterator[dict[str, object]] | Sequence[dict[str, object]],
    ) -> int:
        """Write bars using batch processing for memory efficiency.

        Args:
            bars: Iterator or sequence of bar records.

        Returns:
            Total bytes written across all partitions.
        """
        total_bytes = 0
        batch: list[dict[str, object]] = []
        bar_iter = iter(bars)

        for record in bar_iter:
            batch.append(record)

            if len(batch) >= self.batch_config.batch_size:
                bytes_written = self.write_bars(batch)
                total_bytes += bytes_written
                self._memory_stats.records_processed += len(batch)
                self._memory_stats.batches_processed += 1

                if self.batch_config.enable_memory_logging:
                    self._memory_stats.update()
                    logger.debug(
                        "batch_processed",
                        data_type="bars",
                        batch_number=self._memory_stats.batches_processed,
                        records_in_batch=len(batch),
                        memory_mb=get_memory_usage_mb(),
                    )

                batch = []
                if self.batch_config.gc_after_batch:
                    gc.collect()

        if batch:
            total_bytes += self.write_bars(batch)
            self._memory_stats.records_processed += len(batch)
            self._memory_stats.batches_processed += 1

        return total_bytes

    def flush(self) -> int:
        """Flush any remaining buffered records.

        Should be called after completing all writes to ensure
        all data is persisted.

        Returns:
            Total bytes written during flush.
        """
        return self._flush_buffer()

    def get_stats(self) -> dict[str, object]:
        """Get writer statistics.

        Returns:
            Dictionary with total bytes written, records written, and partitions created.
        """
        return {
            "total_bytes_written": self._total_bytes_written,
            "total_records_written": self._total_records_written,
            "partitions_created": len(self._partitions_created),
        }

    def get_memory_stats(self) -> dict[str, object]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory usage metrics.
        """
        self._memory_stats.update()
        return {
            "peak_memory_bytes": self._memory_stats.peak_memory_bytes,
            "peak_memory_mb": self._memory_stats.peak_memory_bytes / (1024 * 1024),
            "current_memory_mb": get_memory_usage_mb(),
            "records_processed": self._memory_stats.records_processed,
            "batches_processed": self._memory_stats.batches_processed,
            "buffer_count": self._buffer_count,
        }
