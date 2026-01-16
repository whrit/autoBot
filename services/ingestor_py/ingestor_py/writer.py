"""
Parquet Writer - Write market data to partitioned Parquet files.

Partitioning scheme: lake/raw/{data_type}/dt=YYYY-MM-DD/symbol=XXX/
Includes comprehensive logging for file writes, partition creation,
and compression statistics.
"""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ingestor_py.logging_config import format_bytes, get_logger

logger = get_logger(__name__)

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
    """

    # Default compression codec
    COMPRESSION = "snappy"

    def __init__(self, base_path: Path | str) -> None:
        """Initialize the writer.

        Args:
            base_path: Base path for the data lake.
        """
        self.base_path = Path(base_path)
        self._total_bytes_written = 0
        self._total_records_written = 0
        self._partitions_created: set[str] = set()

        logger.info(
            "parquet_writer_initialized",
            base_path=str(base_path),
            compression=self.COMPRESSION,
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

        # Write with compression
        pq.write_table(
            table,
            file_path,
            compression=self.COMPRESSION,
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
