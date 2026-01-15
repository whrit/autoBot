"""
Feature Builder Service for Autonomous Equities Trading Engine.

This module provides multi-timeframe feature construction including:
- Standard OHLCV bars (1m, 5m, 15m)
- Microstructure bars (5s, 15s, 30s)
- As-of joins for point-in-time correctness
- Decision frame construction
- Incremental processing support
"""

from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.decision_frame import DecisionFrameBuilder
from feature_builder_py.incremental import IncrementalProcessor
from feature_builder_py.joins import AsOfJoiner
from feature_builder_py.micro_bars import MicrostructureBarBuilder

__all__ = [
    "StandardBarBuilder",
    "MicrostructureBarBuilder",
    "AsOfJoiner",
    "DecisionFrameBuilder",
    "IncrementalProcessor",
]
