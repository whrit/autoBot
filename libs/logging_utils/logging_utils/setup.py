"""
Core Logging Setup with structlog and Rich console rendering.

Provides:
- Structured logging with timestamps, log level, caller info
- Rich console rendering with colors
- JSON file output for production
- Context binding (symbol, service, task_id)
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme
from structlog.types import EventDict, Processor, WrappedLogger


# Custom theme for trading-specific log styles
TRADING_THEME = Theme({
    "log.time": "cyan",
    "log.level.info": "green",
    "log.level.warning": "yellow",
    "log.level.error": "red bold",
    "log.level.critical": "red bold reverse",
    "log.level.debug": "dim",
    "symbol": "magenta bold",
    "service": "blue",
    "task_id": "cyan dim",
    "metric": "green",
    "duration": "yellow",
})


@dataclass
class LogConfig:
    """Configuration for the logging system."""

    level: str = "INFO"
    console_output: bool = True
    json_output: bool = False
    json_file: Path | None = None
    show_caller: bool = True
    show_timestamp: bool = True
    service_name: str = "autoBot"
    colors: bool = True
    tree_format: bool = True  # Enable tree-style context rendering


# Global console instance
_console: Console | None = None
_config: LogConfig | None = None


def get_console() -> Console:
    """Get or create the Rich console instance."""
    global _console
    if _console is None:
        _console = Console(theme=TRADING_THEME, stderr=True)
    return _console


def _add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO timestamp to log event."""
    event_dict["timestamp"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
    return event_dict


def _add_caller_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add caller module and line number."""
    # structlog already adds _record with caller info
    record = event_dict.get("_record")
    if record:
        event_dict["caller"] = f"{record.module}:{record.lineno}"
    return event_dict


def _format_context_tree(context: dict[str, Any]) -> str:
    """Format context as a tree structure for display."""
    if not context:
        return ""

    lines = []
    items = list(context.items())

    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        prefix = "\u2514\u2500\u2500" if is_last else "\u251c\u2500\u2500"  # "---" or "|--"

        # Format the value nicely
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, list) and len(value) <= 5:
            formatted_value = str(value)
        elif isinstance(value, list):
            formatted_value = f"[{len(value)} items]"
        else:
            formatted_value = str(value)

        lines.append(f"           {prefix} {key}: {formatted_value}")

    return "\n" + "\n".join(lines)


class TreeContextRenderer:
    """Render log context as an indented tree structure."""

    def __call__(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> str:
        """Render the event dict as formatted string with tree context."""
        # Extract core fields
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", "INFO").upper()
        event = event_dict.pop("event", "")

        # Remove internal fields
        event_dict.pop("_record", None)
        event_dict.pop("caller", None)

        # Build the main log line
        level_colors = {
            "DEBUG": "dim",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red bold",
            "CRITICAL": "red bold reverse",
        }
        level_style = level_colors.get(level, "white")

        # Format: [HH:MM:SS] LEVEL  Message
        main_line = f"[{timestamp}] [{level_style}]{level:5}[/]  {event}"

        # Format remaining context as tree
        if event_dict:
            context_tree = _format_context_tree(event_dict)
            return main_line + context_tree

        return main_line


class RichStructlogHandler(logging.Handler):
    """Custom handler that renders structlog output with Rich."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__()
        self.console = console or get_console()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using Rich console."""
        try:
            msg = self.format(record)
            # Parse the message for Rich markup
            self.console.print(msg, markup=True, highlight=False)
        except Exception:
            self.handleError(record)


def configure_logging(config: LogConfig | None = None) -> None:
    """Configure the structured logging system.

    Args:
        config: Logging configuration. Uses defaults if not provided.
    """
    global _config, _console

    if config is None:
        config = LogConfig()

    _config = config
    _console = Console(
        theme=TRADING_THEME,
        stderr=True,
        force_terminal=config.colors,
    )

    # Build processor chain
    processors: list[Processor] = [
        structlog.stdlib.add_log_level,
    ]

    if config.show_timestamp:
        processors.append(_add_timestamp)

    if config.show_caller:
        processors.append(_add_caller_info)

    # Add context merging
    processors.extend([
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ])

    # Final renderer based on config
    if config.json_output:
        processors.append(structlog.processors.JSONRenderer())
    elif config.tree_format:
        processors.append(TreeContextRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=config.colors))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use Rich handler
    log_level = getattr(logging, config.level.upper())

    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Add Rich-based handler for console output
    if config.console_output:
        handler = RichStructlogHandler(_console)
        handler.setLevel(log_level)
        root_logger.addHandler(handler)

    # Add JSON file handler if configured
    if config.json_output and config.json_file:
        config.json_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.json_file)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with optional initial context.

    Args:
        name: Logger name (typically __name__).
        **initial_context: Initial context values to bind.

    Returns:
        Bound structlog logger.
    """
    global _config

    # Auto-configure if not already done
    if _config is None:
        configure_logging()

    logger = structlog.get_logger(name)

    if initial_context:
        logger = logger.bind(**initial_context)

    return logger


def bind_context(**context: Any) -> structlog.stdlib.BoundLogger:
    """Bind context to the current logger.

    Commonly used context keys:
    - symbol: Trading symbol (e.g., 'SPY', 'QQQ')
    - service: Service name (e.g., 'ingestor', 'backtester')
    - task_id: Unique task identifier
    - operation: Current operation name

    Args:
        **context: Context key-value pairs to bind.

    Returns:
        Logger with bound context.
    """
    logger = structlog.get_logger()
    return logger.bind(**context)


# Convenience functions for quick setup
def setup_console_logging(
    level: str = "INFO",
    service: str = "autoBot",
    tree_format: bool = True,
) -> structlog.stdlib.BoundLogger:
    """Quick setup for console logging with Rich output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        service: Service name for context.
        tree_format: Enable tree-style context rendering.

    Returns:
        Configured logger.
    """
    configure_logging(LogConfig(
        level=level,
        console_output=True,
        json_output=False,
        service_name=service,
        tree_format=tree_format,
    ))
    return get_logger(service=service)


def setup_production_logging(
    log_file: Path,
    level: str = "INFO",
    service: str = "autoBot",
) -> structlog.stdlib.BoundLogger:
    """Quick setup for production logging with JSON file output.

    Args:
        log_file: Path to JSON log file.
        level: Log level.
        service: Service name for context.

    Returns:
        Configured logger.
    """
    configure_logging(LogConfig(
        level=level,
        console_output=True,
        json_output=True,
        json_file=log_file,
        service_name=service,
        colors=False,  # No colors in production
        tree_format=False,  # JSON format instead
    ))
    return get_logger(service=service)
