"""
Tests for Signal Logger (T6.02).

Tests signal logging to Parquet files for analysis.
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from runner_py.shadow import ShadowConfig, ShadowExecutor
from runner_py.signal_log import SignalLogConfig, SignalLogger
from runner_py.types import (
    ExecutionMode,
    MarketData,
    ShadowExecutionResult,
    ShadowFill,
    Signal,
    SignalDirection,
)


@pytest.fixture
def temp_log_dir() -> Path:
    """Create temporary directory for signal logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def signal_log_config(temp_log_dir: Path) -> SignalLogConfig:
    """Create test signal log config."""
    return SignalLogConfig(
        log_path=temp_log_dir / "signals",
        buffer_size=10,
        include_market_data=True,
    )


@pytest.fixture
def sample_signal() -> Signal:
    """Create a sample signal."""
    return Signal(
        timestamp=datetime.now(UTC),
        symbol="SPY",
        signal_type=SignalDirection.LONG,
        strength=0.8,
        target_notional=5000.0,
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_execution_result(sample_signal: Signal) -> ShadowExecutionResult:
    """Create a sample execution result."""
    fill = ShadowFill(
        signal=sample_signal,
        simulated_price=450.15,
        simulated_slippage_bps=2.0,
        timestamp=datetime.now(UTC),
        would_have_executed=True,
    )
    return ShadowExecutionResult(
        mode=ExecutionMode.SHADOW,
        signal=sample_signal,
        fill=fill,
        success=True,
        message="Shadow execution successful",
    )


class TestSignalLogConfig:
    """Tests for SignalLogConfig dataclass."""

    def test_default_values(self, temp_log_dir: Path) -> None:
        """Test default config values."""
        config = SignalLogConfig(log_path=temp_log_dir)
        assert config.buffer_size == 100
        assert config.include_market_data is True

    def test_custom_values(self, temp_log_dir: Path) -> None:
        """Test custom config values."""
        config = SignalLogConfig(
            log_path=temp_log_dir,
            buffer_size=50,
            include_market_data=False,
        )
        assert config.buffer_size == 50
        assert config.include_market_data is False


class TestSignalLoggerInit:
    """Tests for SignalLogger initialization."""

    def test_init_creates_logger(self, signal_log_config: SignalLogConfig) -> None:
        """Test logger initialization."""
        logger = SignalLogger(signal_log_config)
        assert logger.config == signal_log_config
        assert len(logger._buffer) == 0

    def test_init_creates_directory(self, signal_log_config: SignalLogConfig) -> None:
        """Test logger creates log directory if it doesn't exist."""
        _ = SignalLogger(signal_log_config)
        assert signal_log_config.log_path.exists()


class TestSignalLoggerLogSignal:
    """Tests for logging individual signals."""

    def test_log_signal(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
    ) -> None:
        """Test logging a single signal."""
        logger = SignalLogger(signal_log_config)
        logger.log_signal(sample_signal)

        assert len(logger._buffer) == 1

    def test_log_signal_with_result(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
        sample_execution_result: ShadowExecutionResult,
    ) -> None:
        """Test logging signal with execution result."""
        logger = SignalLogger(signal_log_config)
        logger.log_signal(sample_signal, sample_execution_result)

        assert len(logger._buffer) == 1
        # The logged entry should include execution result
        entry = logger._buffer[0]
        assert entry["success"] is True
        assert entry["simulated_price"] == pytest.approx(450.15)

    def test_log_signal_without_result(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
    ) -> None:
        """Test logging signal without execution result."""
        logger = SignalLogger(signal_log_config)
        logger.log_signal(sample_signal, None)

        assert len(logger._buffer) == 1
        entry = logger._buffer[0]
        assert entry["success"] is None  # No result

    def test_log_multiple_signals(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test logging multiple signals."""
        logger = SignalLogger(signal_log_config)

        for i in range(5):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.5 + i * 0.1,
                target_notional=1000.0 * (i + 1),
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        assert len(logger._buffer) == 5


class TestSignalLoggerLogBatch:
    """Tests for batch logging."""

    def test_log_batch(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test logging a batch of signals."""
        logger = SignalLogger(signal_log_config)

        signals = []
        for i in range(5):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.5 + i * 0.1,
                target_notional=1000.0 * (i + 1),
                strategy_id="test-strategy",
            )
            signals.append((signal, None))

        logger.log_batch(signals)
        assert len(logger._buffer) == 5

    def test_log_batch_with_results(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test logging batch with execution results."""
        logger = SignalLogger(signal_log_config)

        batch: list[tuple[Signal, ShadowExecutionResult | None]] = []
        for i in range(3):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            fill = ShadowFill(
                signal=signal,
                simulated_price=450.0 + i,
                simulated_slippage_bps=2.0,
                timestamp=datetime.now(UTC),
                would_have_executed=True,
            )
            result = ShadowExecutionResult(
                mode=ExecutionMode.SHADOW,
                signal=signal,
                fill=fill,
                success=True,
                message="OK",
            )
            batch.append((signal, result))

        logger.log_batch(batch)
        assert len(logger._buffer) == 3


class TestSignalLoggerFlush:
    """Tests for flushing buffer to disk."""

    def test_flush_to_disk(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
    ) -> None:
        """Test flushing buffer to disk."""
        logger = SignalLogger(signal_log_config)

        # Add some signals
        for _ in range(5):
            logger.log_signal(sample_signal)

        # Flush to disk
        logger.flush()

        # Buffer should be empty
        assert len(logger._buffer) == 0

        # File should exist
        files = list(signal_log_config.log_path.glob("*.parquet"))
        assert len(files) > 0

    def test_flush_empty_buffer(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test flushing empty buffer is a no-op."""
        logger = SignalLogger(signal_log_config)
        logger.flush()  # Should not raise

        # No files created
        files = list(signal_log_config.log_path.glob("*.parquet"))
        assert len(files) == 0

    def test_auto_flush_on_buffer_full(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test automatic flush when buffer is full."""
        config = SignalLogConfig(
            log_path=temp_log_dir / "signals",
            buffer_size=5,  # Small buffer for testing
        )
        logger = SignalLogger(config)

        # Add more signals than buffer size
        for i in range(7):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        # Should have auto-flushed once
        files = list(config.log_path.glob("*.parquet"))
        assert len(files) >= 1

        # Buffer should have remaining signals
        assert len(logger._buffer) == 2  # 7 - 5 = 2


class TestSignalLoggerReadLogs:
    """Tests for reading logged signals."""

    def test_read_logs(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test reading logged signals."""
        logger = SignalLogger(signal_log_config)

        # Log some signals
        for i in range(5):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        logger.flush()

        # Read them back
        df = logger.read_logs()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_read_logs_with_time_range(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test reading logs within time range."""
        logger = SignalLogger(signal_log_config)

        base_time = datetime.now(UTC)
        for i in range(10):
            signal = Signal(
                timestamp=base_time + timedelta(hours=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        logger.flush()

        # Read with time filter
        start = base_time + timedelta(hours=2)
        end = base_time + timedelta(hours=5)
        df = logger.read_logs(start_time=start, end_time=end)

        assert len(df) >= 3  # Hours 2, 3, 4 at minimum

    def test_read_logs_empty(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test reading empty logs."""
        logger = SignalLogger(signal_log_config)
        df = logger.read_logs()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_read_logs_includes_buffer(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
    ) -> None:
        """Test reading logs includes unflushed buffer."""
        logger = SignalLogger(signal_log_config)

        # Add signals but don't flush
        logger.log_signal(sample_signal)
        logger.log_signal(sample_signal)

        # Read should include buffered signals
        df = logger.read_logs(include_buffer=True)
        assert len(df) == 2


class TestSignalLoggerStats:
    """Tests for signal statistics."""

    def test_get_signal_stats(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test getting signal statistics."""
        logger = SignalLogger(signal_log_config)

        # Log various signals
        directions = [SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT]
        for i, direction in enumerate(directions * 3):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=direction,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        logger.flush()

        stats = logger.get_signal_stats()

        assert "total_signals" in stats
        assert stats["total_signals"] == 9
        assert "by_direction" in stats
        assert "by_symbol" in stats

    def test_get_signal_stats_empty(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test statistics with no signals."""
        logger = SignalLogger(signal_log_config)
        stats = logger.get_signal_stats()

        assert stats["total_signals"] == 0

    def test_stats_by_symbol(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test statistics grouped by symbol."""
        logger = SignalLogger(signal_log_config)

        symbols = ["SPY", "QQQ", "AAPL"]
        for i, symbol in enumerate(symbols * 2):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol=symbol,
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id="test-strategy",
            )
            logger.log_signal(signal)

        logger.flush()
        stats = logger.get_signal_stats()

        assert "by_symbol" in stats
        assert "SPY" in stats["by_symbol"]
        assert stats["by_symbol"]["SPY"] == 2


class TestSignalLoggerIntegration:
    """Integration tests for signal logger with shadow executor."""

    def test_log_shadow_execution(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test logging shadow execution results."""
        logger = SignalLogger(signal_log_config)

        shadow_config = ShadowConfig(
            strategy_id="test-strategy",
            symbols=["SPY"],
            max_position_size=10000.0,
        )
        executor = ShadowExecutor(shadow_config)

        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="SPY",
            signal_type=SignalDirection.LONG,
            strength=0.8,
            target_notional=5000.0,
            strategy_id="test-strategy",
        )
        market_data = MarketData(
            symbol="SPY",
            timestamp=datetime.now(UTC),
            bid=450.00,
            ask=450.10,
            last=450.05,
        )

        result = executor.execute(signal, market_data)
        logger.log_signal(signal, result)
        logger.flush()

        df = logger.read_logs()
        assert len(df) == 1
        assert df["symbol"][0] == "SPY"
        assert df["success"][0] is True

    def test_log_multiple_strategy_signals(
        self,
        signal_log_config: SignalLogConfig,
    ) -> None:
        """Test logging signals from multiple strategies."""
        logger = SignalLogger(signal_log_config)

        strategies = ["trend-follow", "mean-revert", "volatility"]
        for i, strategy in enumerate(strategies):
            signal = Signal(
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                symbol="SPY",
                signal_type=SignalDirection.LONG,
                strength=0.8,
                target_notional=5000.0,
                strategy_id=strategy,
            )
            logger.log_signal(signal)

        logger.flush()

        df = logger.read_logs()
        assert len(df) == 3

        stats = logger.get_signal_stats()
        assert "by_strategy" in stats
        assert len(stats["by_strategy"]) == 3


class TestSignalLoggerParquetSchema:
    """Tests for Parquet file schema."""

    def test_parquet_schema(
        self,
        signal_log_config: SignalLogConfig,
        sample_signal: Signal,
        sample_execution_result: ShadowExecutionResult,
    ) -> None:
        """Test Parquet file has correct schema."""
        logger = SignalLogger(signal_log_config)
        logger.log_signal(sample_signal, sample_execution_result)
        logger.flush()

        files = list(signal_log_config.log_path.glob("*.parquet"))
        assert len(files) == 1

        df = pl.read_parquet(files[0])

        # Check required columns exist
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "direction" in df.columns
        assert "strength" in df.columns
        assert "strategy_id" in df.columns
        assert "success" in df.columns
        assert "simulated_price" in df.columns
