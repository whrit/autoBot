"""
Parquet Writer - Write market data to partitioned Parquet files.

Partitioning scheme: lake/raw/{data_type}/dt=YYYY-MM-DD/symbol=XXX/
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
    """

    def __init__(self, base_path: Path | str) -> None:
        """Initialize the writer.

        Args:
            base_path: Base path for the data lake.
        """
        self.base_path = Path(base_path)

    def _write_partition(
        self,
        records: list[dict[str, object]],
        data_type: str,
        date_str: str,
        symbol: str,
        schema: pa.Schema,
    ) -> None:
        """Write records to a partitioned Parquet file.

        Args:
            records: List of records to write.
            data_type: Type of data ('trades', 'quotes', 'bars_provider').
            date_str: Date string (YYYY-MM-DD).
            symbol: Stock symbol.
            schema: PyArrow schema for the data.
        """
        # Build partition path
        partition_path = (
            self.base_path / "raw" / data_type / f"dt={date_str}" / f"symbol={symbol}"
        )
        partition_path.mkdir(parents=True, exist_ok=True)

        # Prepare data for PyArrow (exclude symbol, already in partition)
        columns: dict[str, list[object]] = {
            col.name: [] for col in schema
        }

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
        pq.write_table(table, partition_path / "data.parquet")

    def write_trades(self, trades: Sequence[dict[str, object]]) -> None:
        """Write trades to partitioned Parquet files.

        Args:
            trades: Sequence of trade records.
        """
        if not trades:
            return

        groups = _group_by_partition(trades)

        for (date_str, symbol), records in groups.items():
            self._write_partition(
                records=records,
                data_type="trades",
                date_str=date_str,
                symbol=symbol,
                schema=TRADE_SCHEMA,
            )

    def write_quotes(self, quotes: Sequence[dict[str, object]]) -> None:
        """Write quotes to partitioned Parquet files.

        Args:
            quotes: Sequence of quote records.
        """
        if not quotes:
            return

        groups = _group_by_partition(quotes)

        for (date_str, symbol), records in groups.items():
            self._write_partition(
                records=records,
                data_type="quotes",
                date_str=date_str,
                symbol=symbol,
                schema=QUOTE_SCHEMA,
            )

    def write_bars(self, bars: Sequence[dict[str, object]]) -> None:
        """Write bars to partitioned Parquet files.

        Args:
            bars: Sequence of bar records.
        """
        if not bars:
            return

        groups = _group_by_partition(bars)

        for (date_str, symbol), records in groups.items():
            self._write_partition(
                records=records,
                data_type="bars_provider",
                date_str=date_str,
                symbol=symbol,
                schema=BAR_SCHEMA,
            )
