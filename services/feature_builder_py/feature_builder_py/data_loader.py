"""
Optimized Data Loader for Large Parquet Datasets.

Provides lazy evaluation, streaming, and batch processing capabilities
for memory-efficient processing of large trade and quote datasets.

Key Features:
- Lazy scanning with scan_parquet() for memory efficiency
- Batch/chunked processing for datasets larger than memory
- Memory-mapped file access for optimal I/O
- Progress logging for long-running operations
- Streaming support for continuous data processing
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Iterator

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for data loading operations."""

    # Batch processing settings
    batch_size: int = 1_000_000  # Rows per batch for chunked processing

    # Memory settings
    use_memory_map: bool = True  # Use memory-mapped I/O
    low_memory: bool = False  # Trade memory for speed

    # Parallel processing
    n_rows: int | None = None  # Limit total rows (None = all)

    # Column filtering
    columns: list[str] | None = None  # Only load specific columns

    # Date filtering for partitioned data
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Progress settings
    enable_progress: bool = True
    progress_callback: Callable[[int, int, str], None] | None = None


@dataclass
class DataStats:
    """Statistics about loaded data."""

    total_rows: int = 0
    total_files: int = 0
    memory_bytes: int = 0
    load_time_seconds: float = 0.0
    batches_processed: int = 0
    columns: list[str] = field(default_factory=list)


class LazyDataLoader:
    """
    Memory-efficient data loader using Polars lazy evaluation.

    Uses scan_parquet() instead of read_parquet() for lazy evaluation,
    which allows query optimization and minimal memory usage.
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        """
        Initialize the lazy data loader.

        Args:
            config: Loading configuration. Uses defaults if None.
        """
        self.config = config or LoaderConfig()
        self._log = logger.bind(component="LazyDataLoader")

    def scan_parquet(
        self,
        path: Path | str,
        **kwargs,
    ) -> pl.LazyFrame:
        """
        Lazily scan a parquet file or directory.

        Uses scan_parquet() for lazy evaluation - no data is loaded
        until .collect() is called. This enables query optimization
        and predicate pushdown.

        Args:
            path: Path to parquet file or directory.
            **kwargs: Additional arguments passed to pl.scan_parquet().

        Returns:
            LazyFrame for deferred computation.
        """
        path = Path(path)

        self._log.debug(
            "scanning_parquet",
            path=str(path),
            memory_map=self.config.use_memory_map,
            low_memory=self.config.low_memory,
        )

        # Build scan options
        scan_opts = {
            "memory_map": self.config.use_memory_map,
            "low_memory": self.config.low_memory,
            "n_rows": self.config.n_rows,
        }
        scan_opts.update(kwargs)

        if path.is_dir():
            # Scan all parquet files in directory
            pattern = str(path / "*.parquet")
            lf = pl.scan_parquet(pattern, **scan_opts)
        else:
            lf = pl.scan_parquet(path, **scan_opts)

        # Apply column filtering if specified
        if self.config.columns:
            lf = lf.select(self.config.columns)

        return lf

    def scan_trades(
        self,
        path: Path | str,
        symbol: str | None = None,
    ) -> pl.LazyFrame:
        """
        Lazily scan trade data with optimized settings.

        Args:
            path: Path to trades parquet file or directory.
            symbol: Optional symbol to filter on.

        Returns:
            LazyFrame of trade data.
        """
        lf = self.scan_parquet(path)

        # Apply date filtering with predicate pushdown
        if self.config.start_date:
            lf = lf.filter(pl.col("ts_event") >= self.config.start_date)
        if self.config.end_date:
            lf = lf.filter(pl.col("ts_event") <= self.config.end_date)

        # Apply symbol filter if specified
        if symbol and "symbol" in lf.collect_schema().names():
            lf = lf.filter(pl.col("symbol") == symbol)

        return lf

    def scan_quotes(
        self,
        path: Path | str,
        symbol: str | None = None,
    ) -> pl.LazyFrame:
        """
        Lazily scan quote data with optimized settings.

        Args:
            path: Path to quotes parquet file or directory.
            symbol: Optional symbol to filter on.

        Returns:
            LazyFrame of quote data.
        """
        lf = self.scan_parquet(path)

        # Apply date filtering with predicate pushdown
        if self.config.start_date:
            lf = lf.filter(pl.col("ts_event") >= self.config.start_date)
        if self.config.end_date:
            lf = lf.filter(pl.col("ts_event") <= self.config.end_date)

        # Apply symbol filter if specified
        if symbol and "symbol" in lf.collect_schema().names():
            lf = lf.filter(pl.col("symbol") == symbol)

        return lf

    def collect_with_progress(
        self,
        lf: pl.LazyFrame,
        description: str = "Processing",
    ) -> pl.DataFrame:
        """
        Collect a LazyFrame with progress logging.

        Args:
            lf: LazyFrame to collect.
            description: Description for progress logging.

        Returns:
            Collected DataFrame.
        """
        import time

        start = time.perf_counter()
        self._log.info(f"{description}_started")

        # Collect with streaming if available
        try:
            # Use streaming engine for large datasets
            df = lf.collect(streaming=True)
        except Exception:
            # Fallback to regular collection
            df = lf.collect()

        elapsed = time.perf_counter() - start

        self._log.info(
            f"{description}_completed",
            rows=len(df),
            columns=len(df.columns),
            elapsed_sec=round(elapsed, 3),
        )

        if self.config.progress_callback:
            self.config.progress_callback(len(df), len(df), description)

        return df


class BatchProcessor:
    """
    Process large datasets in batches for memory efficiency.

    Splits large parquet files into manageable chunks and processes
    them sequentially, maintaining state between batches.
    """

    def __init__(
        self,
        batch_size: int = 1_000_000,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            batch_size: Number of rows per batch.
            progress_callback: Optional callback for progress updates.
        """
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self._log = logger.bind(component="BatchProcessor")

    def iter_batches(
        self,
        path: Path | str,
        sort_by: str | None = "ts_event",
    ) -> Generator[tuple[int, pl.DataFrame], None, None]:
        """
        Iterate over a parquet file in batches.

        Yields batches of data without loading the entire file into memory.

        Args:
            path: Path to parquet file.
            sort_by: Column to sort batches by (ensures temporal order).

        Yields:
            Tuple of (batch_index, DataFrame) for each batch.
        """
        path = Path(path)

        # Get total row count first
        lf = pl.scan_parquet(path)
        total_rows = lf.select(pl.len()).collect().item()

        self._log.info(
            "batch_iteration_started",
            path=str(path),
            total_rows=total_rows,
            batch_size=self.batch_size,
            num_batches=(total_rows + self.batch_size - 1) // self.batch_size,
        )

        batch_idx = 0
        offset = 0

        while offset < total_rows:
            # Scan with offset and limit
            batch_lf = pl.scan_parquet(path)

            if sort_by:
                batch_lf = batch_lf.sort(sort_by)

            batch_df = batch_lf.slice(offset, self.batch_size).collect()

            if batch_df.is_empty():
                break

            if self.progress_callback:
                self.progress_callback(
                    offset + len(batch_df),
                    total_rows,
                    f"Batch {batch_idx + 1}",
                )

            self._log.debug(
                "batch_yielded",
                batch_idx=batch_idx,
                rows=len(batch_df),
                offset=offset,
            )

            yield batch_idx, batch_df

            offset += len(batch_df)
            batch_idx += 1

        self._log.info(
            "batch_iteration_completed",
            batches_processed=batch_idx,
            total_rows_processed=offset,
        )

    def process_batches(
        self,
        path: Path | str,
        process_fn: Callable[[pl.DataFrame], pl.DataFrame],
        output_path: Path | str | None = None,
        sort_by: str | None = "ts_event",
    ) -> pl.DataFrame | None:
        """
        Process a parquet file in batches and optionally save results.

        Args:
            path: Path to input parquet file.
            process_fn: Function to apply to each batch.
            output_path: Optional path to save results (appended per batch).
            sort_by: Column to sort batches by.

        Returns:
            Concatenated results if no output_path, None otherwise.
        """
        results = []

        for batch_idx, batch_df in self.iter_batches(path, sort_by):
            # Process the batch
            processed = process_fn(batch_df)

            if output_path:
                # Append to output file
                output_path = Path(output_path)
                if batch_idx == 0:
                    processed.write_parquet(output_path)
                else:
                    # Append by reading, concatenating, and rewriting
                    # (Polars doesn't support append mode yet)
                    existing = pl.read_parquet(output_path)
                    combined = pl.concat([existing, processed])
                    combined.write_parquet(output_path)
            else:
                results.append(processed)

        if results:
            return pl.concat(results)
        return None


class StreamingLoader:
    """
    Streaming data loader for continuous data processing.

    Supports reading data in a streaming fashion, useful for
    real-time processing or very large datasets.
    """

    def __init__(
        self,
        chunk_size: int = 100_000,
    ) -> None:
        """
        Initialize the streaming loader.

        Args:
            chunk_size: Number of rows per chunk.
        """
        self.chunk_size = chunk_size
        self._log = logger.bind(component="StreamingLoader")

    def stream_parquet(
        self,
        path: Path | str,
        columns: list[str] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """
        Stream a parquet file in chunks.

        Args:
            path: Path to parquet file.
            columns: Optional list of columns to load.

        Yields:
            DataFrame chunks.
        """
        path = Path(path)

        # Use PyArrow for streaming reads
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path, memory_map=True)

            for batch in parquet_file.iter_batches(
                batch_size=self.chunk_size,
                columns=columns,
            ):
                # Convert Arrow batch to Polars DataFrame
                df = pl.from_arrow(batch)
                yield df

        except ImportError:
            # Fallback to batch reading
            self._log.warning("pyarrow_not_available_using_fallback")

            lf = pl.scan_parquet(path)
            if columns:
                lf = lf.select(columns)

            total_rows = lf.select(pl.len()).collect().item()
            offset = 0

            while offset < total_rows:
                chunk = lf.slice(offset, self.chunk_size).collect()
                if chunk.is_empty():
                    break
                yield chunk
                offset += len(chunk)


def estimate_memory_usage(path: Path | str) -> dict:
    """
    Estimate memory usage for a parquet file.

    Args:
        path: Path to parquet file.

    Returns:
        Dictionary with memory estimates.
    """
    path = Path(path)
    file_size = path.stat().st_size

    # Scan to get schema and row count
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    row_count = lf.select(pl.len()).collect().item()

    # Estimate in-memory size (typically 2-10x compressed size)
    estimated_memory = file_size * 3  # Conservative estimate

    return {
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "row_count": row_count,
        "column_count": len(schema),
        "estimated_memory_bytes": estimated_memory,
        "estimated_memory_mb": estimated_memory / (1024 * 1024),
        "columns": list(schema.names()),
        "dtypes": {name: str(dtype) for name, dtype in schema.items()},
    }


def get_parquet_metadata(path: Path | str) -> dict:
    """
    Get metadata from a parquet file without loading data.

    Args:
        path: Path to parquet file.

    Returns:
        Dictionary with parquet metadata.
    """
    try:
        import pyarrow.parquet as pq

        path = Path(path)
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata

        return {
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "num_row_groups": metadata.num_row_groups,
            "created_by": metadata.created_by,
            "format_version": str(metadata.format_version),
            "row_groups": [
                {
                    "num_rows": metadata.row_group(i).num_rows,
                    "total_byte_size": metadata.row_group(i).total_byte_size,
                }
                for i in range(metadata.num_row_groups)
            ],
        }
    except ImportError:
        # Fallback without PyArrow
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
        row_count = lf.select(pl.len()).collect().item()

        return {
            "num_rows": row_count,
            "num_columns": len(schema),
            "columns": list(schema.names()),
        }


def scan_lake_directory(
    lake_path: Path | str,
    symbol: str,
    data_type: str = "trades",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pl.LazyFrame:
    """
    Scan a data lake directory with optimized settings.

    Assumes a directory structure like:
    lake/{data_type}/{symbol}/*.parquet
    or
    lake/{symbol}_{data_type}.parquet

    Args:
        lake_path: Path to lake directory.
        symbol: Symbol to load.
        data_type: Type of data ('trades' or 'quotes').
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        LazyFrame with filtered data.
    """
    lake_path = Path(lake_path)

    # Try different path patterns
    patterns = [
        lake_path / data_type / symbol / "*.parquet",
        lake_path / f"{symbol}_{data_type}.parquet",
        lake_path / f"{symbol}_{data_type}/*.parquet",
        lake_path / symbol / f"{data_type}.parquet",
        lake_path / symbol / f"{data_type}/*.parquet",
    ]

    lf = None
    for pattern in patterns:
        if pattern.parent.exists():
            try:
                if "*" in str(pattern):
                    lf = pl.scan_parquet(str(pattern), memory_map=True)
                elif pattern.exists():
                    lf = pl.scan_parquet(pattern, memory_map=True)
                break
            except Exception:
                continue

    if lf is None:
        # Try loading from single file with symbol filter
        all_parquet = list(lake_path.glob("**/*.parquet"))
        if all_parquet:
            lf = pl.scan_parquet(
                [str(p) for p in all_parquet],
                memory_map=True,
            )
            if "symbol" in lf.collect_schema().names():
                lf = lf.filter(pl.col("symbol") == symbol)
        else:
            raise FileNotFoundError(
                f"No parquet files found for {symbol} {data_type} in {lake_path}"
            )

    # Apply date filters with predicate pushdown
    if start_date:
        lf = lf.filter(pl.col("ts_event") >= start_date)
    if end_date:
        lf = lf.filter(pl.col("ts_event") <= end_date)

    return lf
