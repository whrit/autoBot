"""Tests for logging setup module."""

import logging
import tempfile
from pathlib import Path

import pytest
import structlog

from logging_utils.setup import (
    LogConfig,
    TreeContextRenderer,
    bind_context,
    configure_logging,
    get_console,
    get_logger,
    setup_console_logging,
    setup_production_logging,
)


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.console_output is True
        assert config.json_output is False
        assert config.json_file is None
        assert config.show_caller is True
        assert config.show_timestamp is True
        assert config.service_name == "autoBot"
        assert config.colors is True
        assert config.tree_format is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LogConfig(
            level="DEBUG",
            console_output=False,
            json_output=True,
            json_file=Path("/tmp/test.log"),
            service_name="test_service",
        )
        assert config.level == "DEBUG"
        assert config.console_output is False
        assert config.json_output is True
        assert config.json_file == Path("/tmp/test.log")
        assert config.service_name == "test_service"


class TestGetConsole:
    """Tests for get_console function."""

    def test_returns_console(self) -> None:
        """Test that get_console returns a Console instance."""
        console = get_console()
        assert console is not None

    def test_console_is_singleton(self) -> None:
        """Test that the same console instance is returned."""
        console1 = get_console()
        console2 = get_console()
        assert console1 is console2


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_default_configuration(self) -> None:
        """Test default logging configuration."""
        configure_logging()
        logger = structlog.get_logger("test")
        assert logger is not None

    def test_custom_level(self) -> None:
        """Test configuration with custom log level."""
        configure_logging(LogConfig(level="DEBUG"))
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_warning_level(self) -> None:
        """Test configuration with WARNING level."""
        configure_logging(LogConfig(level="WARNING"))
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_without_name(self) -> None:
        """Test getting logger without name."""
        configure_logging()
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """Test getting logger with name."""
        configure_logging()
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_with_context(self) -> None:
        """Test getting logger with initial context."""
        configure_logging()
        logger = get_logger("test", service="test_service", symbol="SPY")
        assert logger is not None


class TestBindContext:
    """Tests for bind_context function."""

    def test_bind_single_context(self) -> None:
        """Test binding single context value."""
        configure_logging()
        logger = bind_context(symbol="SPY")
        assert logger is not None

    def test_bind_multiple_context(self) -> None:
        """Test binding multiple context values."""
        configure_logging()
        logger = bind_context(
            symbol="SPY",
            service="ingestor",
            task_id="task-123",
        )
        assert logger is not None


class TestTreeContextRenderer:
    """Tests for TreeContextRenderer class."""

    def test_render_simple_event(self) -> None:
        """Test rendering a simple event."""
        renderer = TreeContextRenderer()
        event_dict = {
            "timestamp": "12:00:00",
            "level": "info",
            "event": "Test message",
        }
        result = renderer(None, "info", event_dict)
        assert "Test message" in result
        assert "12:00:00" in result
        assert "INFO" in result

    def test_render_with_context(self) -> None:
        """Test rendering event with context."""
        renderer = TreeContextRenderer()
        event_dict = {
            "timestamp": "12:00:00",
            "level": "info",
            "event": "Test message",
            "symbol": "SPY",
            "count": 100,
        }
        result = renderer(None, "info", event_dict)
        assert "Test message" in result
        assert "symbol" in result
        assert "SPY" in result
        assert "count" in result
        assert "100" in result

    def test_render_with_float_context(self) -> None:
        """Test rendering event with float context value."""
        renderer = TreeContextRenderer()
        event_dict = {
            "timestamp": "12:00:00",
            "level": "info",
            "event": "Test",
            "rate": 3.14159,
        }
        result = renderer(None, "info", event_dict)
        assert "3.14" in result

    def test_render_with_list_context(self) -> None:
        """Test rendering event with list context value."""
        renderer = TreeContextRenderer()
        event_dict = {
            "timestamp": "12:00:00",
            "level": "info",
            "event": "Test",
            "symbols": ["SPY", "QQQ"],
        }
        result = renderer(None, "info", event_dict)
        assert "symbols" in result

    def test_render_different_levels(self) -> None:
        """Test rendering different log levels."""
        renderer = TreeContextRenderer()

        levels = ["debug", "info", "warning", "error", "critical"]
        for level in levels:
            event_dict = {
                "timestamp": "12:00:00",
                "level": level,
                "event": "Test",
            }
            result = renderer(None, level, event_dict)
            assert level.upper() in result


class TestSetupConvenienceFunctions:
    """Tests for convenience setup functions."""

    def test_setup_console_logging(self) -> None:
        """Test setup_console_logging function."""
        logger = setup_console_logging(level="DEBUG", service="test")
        assert logger is not None

    def test_setup_production_logging(self) -> None:
        """Test setup_production_logging function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_production_logging(log_file, level="INFO", service="test")
            assert logger is not None


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_log_info_message(self) -> None:
        """Test logging an info message."""
        configure_logging(LogConfig(level="INFO"))
        logger = get_logger("test")
        # Should not raise
        logger.info("Test info message")

    def test_log_with_exception(self) -> None:
        """Test logging with exception info."""
        configure_logging(LogConfig(level="INFO"))
        logger = get_logger("test")
        try:
            raise ValueError("Test error")
        except ValueError:
            # Should not raise
            logger.exception("Error occurred")

    def test_context_inheritance(self) -> None:
        """Test that context is inherited correctly."""
        configure_logging()
        logger = get_logger("test", service="parent")
        child_logger = logger.bind(operation="child_op")
        assert child_logger is not None
