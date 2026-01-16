"""
Feature Builder Service for Autonomous Equities Trading Engine.

This module provides multi-timeframe feature construction including:
- Standard OHLCV bars (1m, 5m, 15m)
- Microstructure bars (5s, 15s, 30s)
- As-of joins for point-in-time correctness
- Decision frame construction
- Incremental processing support

Optimized for Large Dataset Processing:
- Lazy evaluation with scan_parquet() for memory efficiency
- Batch processing for datasets larger than available memory
- Streaming/chunked reading of parquet files
- Progress logging for long-running computations
- Memory-mapped file access for optimal I/O performance
"""

from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.data_loader import (
    BatchProcessor,
    LazyDataLoader,
    LoaderConfig,
    StreamingLoader,
    estimate_memory_usage,
    get_parquet_metadata,
    scan_lake_directory,
)
from feature_builder_py.decision_frame import DecisionFrameBuilder
from feature_builder_py.incremental import IncrementalProcessor
from feature_builder_py.joins import AsOfJoiner
from feature_builder_py.main import FeatureBuilderCLI, LargeDatasetBuilder, main
from feature_builder_py.micro_bars import MicrostructureBarBuilder

__all__ = [
    # Core builders
    "StandardBarBuilder",
    "MicrostructureBarBuilder",
    "AsOfJoiner",
    "DecisionFrameBuilder",
    "IncrementalProcessor",
    # CLI
    "FeatureBuilderCLI",
    "LargeDatasetBuilder",
    "main",
    # Data loading utilities
    "LazyDataLoader",
    "LoaderConfig",
    "BatchProcessor",
    "StreamingLoader",
    "estimate_memory_usage",
    "get_parquet_metadata",
    "scan_lake_directory",
]
