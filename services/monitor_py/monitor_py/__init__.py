"""Monitor service for autonomous trading engine.

This package provides monitoring capabilities for:
- Feed health monitoring (T5.07)
- Slippage tracking (T5.08)
- Drawdown alerts (T5.09)
- Prometheus metrics export (T5.10)
- Rollback trigger logic (T5.11)
"""

from monitor_py.drawdown import DrawdownConfig, DrawdownMonitor, DrawdownStatus
from monitor_py.feed_health import (
    FeedHealthConfig,
    FeedHealthMonitor,
    FeedHealthStatus,
    FeedStatus,
)
from monitor_py.metrics import MetricsExporter
from monitor_py.rollback import RollbackAction, RollbackConfig, RollbackManager
from monitor_py.slippage import SlippageConfig, SlippageStats, SlippageTracker

__all__ = [
    # Feed Health
    "FeedHealthConfig",
    "FeedHealthMonitor",
    "FeedHealthStatus",
    "FeedStatus",
    # Slippage
    "SlippageConfig",
    "SlippageStats",
    "SlippageTracker",
    # Drawdown
    "DrawdownConfig",
    "DrawdownMonitor",
    "DrawdownStatus",
    # Metrics
    "MetricsExporter",
    # Rollback
    "RollbackAction",
    "RollbackConfig",
    "RollbackManager",
]

__version__ = "0.1.0"
