"""Main entry point for monitor service."""

from typing import Any

import structlog

from monitor_py import (
    DrawdownConfig,
    DrawdownMonitor,
    FeedHealthConfig,
    FeedHealthMonitor,
    MetricsExporter,
    RollbackConfig,
    RollbackManager,
    SlippageConfig,
    SlippageTracker,
)

logger = structlog.get_logger(__name__)


def create_monitoring_stack(
    initial_equity: float = 100000.0,
) -> dict[str, Any]:
    """Create a complete monitoring stack.

    Args:
        initial_equity: Initial portfolio equity.

    Returns:
        Dictionary with all monitoring components.
    """
    return {
        "feed_monitor": FeedHealthMonitor(FeedHealthConfig()),
        "slippage_tracker": SlippageTracker(SlippageConfig()),
        "drawdown_monitor": DrawdownMonitor(DrawdownConfig(), initial_equity=initial_equity),
        "metrics_exporter": MetricsExporter(),
        "rollback_manager": RollbackManager(RollbackConfig()),
    }


def main() -> None:
    """Main entry point."""
    logger.info("monitor_py service started")
    stack = create_monitoring_stack()
    logger.info(
        "monitoring_stack_created",
        components=list(stack.keys()),
    )


if __name__ == "__main__":
    main()
