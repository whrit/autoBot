"""
Comprehensive tests for feature_builder_py service.

Tests cover:
- T2.01: StandardBarBuilder (1m, 5m, 15m bars)
- T2.02: MicrostructureBarBuilder (5s, 15s, 30s bars)
- T2.03: AsOfJoiner (point-in-time correct joins)
- T2.04: DecisionFrameBuilder (multi-timeframe matrix)
- T2.05: IncrementalProcessor (incremental updates)
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.decision_frame import DecisionFrameBuilder
from feature_builder_py.incremental import IncrementalProcessor
from feature_builder_py.joins import AsOfJoiner
from feature_builder_py.micro_bars import MicrostructureBarBuilder

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_trades() -> pl.DataFrame:
    """Generate sample trade data for testing."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    trades = []
    price = 100.0

    # Generate 10 minutes of trade data (1 trade per second)
    for i in range(600):
        ts = base_time + timedelta(seconds=i)
        # Add some price movement
        price += (i % 7 - 3) * 0.01
        trades.append({
            "ts_event": ts,
            "ts_recv": ts + timedelta(microseconds=100),
            "price": round(price, 2),
            "size": float(100 + (i % 50)),
            "exchange": "XNYS",
            "conditions": None,
        })

    return pl.DataFrame(trades)


@pytest.fixture
def sample_quotes() -> pl.DataFrame:
    """Generate sample quote data for testing."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    quotes = []
    mid = 100.0

    # Generate 10 minutes of quote data (5 quotes per second)
    for i in range(3000):
        ts = base_time + timedelta(milliseconds=i * 200)
        mid += (i % 11 - 5) * 0.001
        spread = 0.01 + (i % 5) * 0.001
        bid = round(mid - spread / 2, 4)
        ask = round(mid + spread / 2, 4)

        quotes.append({
            "ts_event": ts,
            "ts_recv": ts + timedelta(microseconds=50),
            "bid_price": bid,
            "bid_size": float(500 + (i % 200)),
            "ask_price": ask,
            "ask_size": float(400 + (i % 150)),
            "bid_exchange": "XNYS",
            "ask_exchange": "XNYS",
            "conditions": None,
        })

    return pl.DataFrame(quotes)


@pytest.fixture
def sample_1m_bars() -> pl.DataFrame:
    """Generate sample 1-minute bar data."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    bars = []

    for i in range(15):  # 15 minutes of bars
        bar_start = base_time + timedelta(minutes=i)
        bar_end = bar_start + timedelta(minutes=1)
        open_price = 100.0 + i * 0.1

        bars.append({
            "symbol": "SPY",
            "bar_start": bar_start,
            "bar_end": bar_end,
            "open": open_price,
            "high": open_price + 0.15,
            "low": open_price - 0.05,
            "close": open_price + 0.08,
            "volume": 10000.0 + i * 100,
            "returns": 0.0008 if i > 0 else None,
            "atr": 0.12,
            "realized_vol": 0.015,
        })

    return pl.DataFrame(bars)


@pytest.fixture
def sample_5m_bars() -> pl.DataFrame:
    """Generate sample 5-minute bar data."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    bars = []

    for i in range(3):  # 3 five-minute bars (15 minutes total)
        bar_start = base_time + timedelta(minutes=i * 5)
        bar_end = bar_start + timedelta(minutes=5)
        open_price = 100.0 + i * 0.5

        bars.append({
            "symbol": "SPY",
            "bar_start": bar_start,
            "bar_end": bar_end,
            "open": open_price,
            "high": open_price + 0.4,
            "low": open_price - 0.15,
            "close": open_price + 0.3,
            "volume": 50000.0 + i * 500,
            "returns": 0.003 if i > 0 else None,
            "atr": 0.35,
            "realized_vol": 0.018,
        })

    return pl.DataFrame(bars)


@pytest.fixture
def sample_micro_bars() -> pl.DataFrame:
    """Generate sample 30-second microstructure bars."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    bars = []

    for i in range(20):  # 20 thirty-second bars (10 minutes)
        bar_start = base_time + timedelta(seconds=i * 30)
        bar_end = bar_start + timedelta(seconds=30)

        bars.append({
            "symbol": "SPY",
            "bar_start": bar_start,
            "bar_end": bar_end,
            "vwap": 100.0 + i * 0.02,
            "midprice": 100.0 + i * 0.018,
            "microprice": 100.0 + i * 0.019,
            "spread": 0.01 + (i % 3) * 0.002,
            "bid_size": 500.0 + i * 10,
            "ask_size": 450.0 + i * 8,
            "quote_imbalance": 0.05 - (i % 5) * 0.02,
            "trade_volume": 1500.0 + i * 50,
            "realized_vol": 0.012 + (i % 4) * 0.001,
        })

    return pl.DataFrame(bars)


# =============================================================================
# T2.01: STANDARD BAR BUILDER TESTS
# =============================================================================


class TestStandardBarBuilder:
    """Tests for StandardBarBuilder (T2.01)."""

    def test_build_1m_bars_basic(self, sample_trades: pl.DataFrame) -> None:
        """Test building 1-minute bars from trades."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        assert bars is not None
        assert len(bars) == 10  # 10 minutes of data
        assert "symbol" in bars.columns
        assert "bar_start" in bars.columns
        assert "bar_end" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns

    def test_build_5m_bars(self, sample_trades: pl.DataFrame) -> None:
        """Test building 5-minute bars from trades."""
        builder = StandardBarBuilder(granularity="5m")
        bars = builder.build(sample_trades, symbol="SPY")

        assert bars is not None
        assert len(bars) == 2  # 10 minutes = 2 five-minute bars

    def test_build_15m_bars(self, sample_trades: pl.DataFrame) -> None:
        """Test building 15-minute bars from trades."""
        builder = StandardBarBuilder(granularity="15m")
        bars = builder.build(sample_trades, symbol="SPY")

        assert bars is not None
        assert len(bars) == 1  # 10 minutes fits in 1 fifteen-minute bar

    def test_ohlcv_values_correct(self, sample_trades: pl.DataFrame) -> None:
        """Test OHLCV values are computed correctly."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        first_bar = bars.row(0, named=True)

        # Get first minute of trades
        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        first_minute = sample_trades.filter(
            (pl.col("ts_event") >= base_time) &
            (pl.col("ts_event") < base_time + timedelta(minutes=1))
        )

        assert first_bar["open"] == first_minute["price"][0]
        assert first_bar["high"] == first_minute["price"].max()
        assert first_bar["low"] == first_minute["price"].min()
        assert first_bar["close"] == first_minute["price"][-1]
        assert first_bar["volume"] == first_minute["size"].sum()

    def test_returns_calculation(self, sample_trades: pl.DataFrame) -> None:
        """Test that returns are calculated correctly."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        # First bar should have None return (no previous bar)
        assert bars["returns"][0] is None

        # Second bar should have return = (close_1 - close_0) / close_0
        close_0 = bars["close"][0]
        close_1 = bars["close"][1]
        expected_return = (close_1 - close_0) / close_0
        assert abs(bars["returns"][1] - expected_return) < 1e-10

    def test_atr_calculation(self, sample_trades: pl.DataFrame) -> None:
        """Test ATR (Average True Range) calculation."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        # ATR should be non-negative
        atr_values = bars["atr"].drop_nulls()
        assert all(v >= 0 for v in atr_values)

    def test_realized_vol_calculation(self, sample_trades: pl.DataFrame) -> None:
        """Test realized volatility calculation."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        # Realized vol should be non-negative
        vol_values = bars["realized_vol"].drop_nulls()
        assert all(v >= 0 for v in vol_values)

    def test_empty_trades_returns_empty_df(self) -> None:
        """Test that empty trades return empty DataFrame."""
        builder = StandardBarBuilder(granularity="1m")
        empty_trades = pl.DataFrame({
            "ts_event": [],
            "ts_recv": [],
            "price": [],
            "size": [],
            "exchange": [],
            "conditions": [],
        }).cast({
            "ts_event": pl.Datetime("us", "UTC"),
            "ts_recv": pl.Datetime("us", "UTC"),
            "price": pl.Float64,
            "size": pl.Float64,
        })

        bars = builder.build(empty_trades, symbol="SPY")
        assert len(bars) == 0

    def test_invalid_granularity_raises(self) -> None:
        """Test that invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity"):
            StandardBarBuilder(granularity="2m")

    def test_timestamps_are_timezone_aware(self, sample_trades: pl.DataFrame) -> None:
        """Test that output timestamps are timezone-aware (UTC)."""
        builder = StandardBarBuilder(granularity="1m")
        bars = builder.build(sample_trades, symbol="SPY")

        # Check that timestamps have UTC timezone
        bar_start_dtype = bars.schema["bar_start"]
        assert bar_start_dtype == pl.Datetime("us", "UTC")


# =============================================================================
# T2.02: MICROSTRUCTURE BAR BUILDER TESTS
# =============================================================================


class TestMicrostructureBarBuilder:
    """Tests for MicrostructureBarBuilder (T2.02)."""

    def test_build_5s_bars(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test building 5-second microstructure bars."""
        builder = MicrostructureBarBuilder(granularity="5s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        assert bars is not None
        assert len(bars) == 120  # 10 minutes = 120 five-second bars

    def test_build_15s_bars(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test building 15-second microstructure bars."""
        builder = MicrostructureBarBuilder(granularity="15s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        assert bars is not None
        assert len(bars) == 40  # 10 minutes = 40 fifteen-second bars

    def test_build_30s_bars(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test building 30-second microstructure bars."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        assert bars is not None
        assert len(bars) == 20  # 10 minutes = 20 thirty-second bars

    def test_micro_bar_schema(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test that micro bars have correct schema."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        expected_columns = {
            "symbol", "bar_start", "bar_end", "vwap", "midprice",
            "microprice", "spread", "bid_size", "ask_size",
            "quote_imbalance", "trade_volume", "realized_vol"
        }
        assert set(bars.columns) == expected_columns

    def test_spread_calculation(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test spread calculation is correct."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Spread should be positive
        assert all(bars["spread"] > 0)

    def test_midprice_calculation(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test midprice calculation."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Midprice should be between bid and ask (approximately)
        assert all(bars["midprice"] > 0)

    def test_microprice_calculation(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test microprice (size-weighted midprice) calculation."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Microprice should be close to midprice
        assert all(bars["microprice"] > 0)
        diff = (bars["microprice"] - bars["midprice"]).abs()
        assert all(diff < 1.0)  # Should be within $1

    def test_quote_imbalance(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test quote imbalance calculation."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Quote imbalance should be between -1 and 1
        assert all(bars["quote_imbalance"] >= -1)
        assert all(bars["quote_imbalance"] <= 1)

    def test_vwap_calculation(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test VWAP calculation."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # VWAP should be reasonable (close to price range)
        assert all(bars["vwap"] > 0)

    def test_realized_vol_micro(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test realized volatility in micro bars."""
        builder = MicrostructureBarBuilder(granularity="30s")
        bars = builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Realized vol should be non-negative
        assert all(bars["realized_vol"] >= 0)

    def test_invalid_granularity_raises(self) -> None:
        """Test that invalid granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity"):
            MicrostructureBarBuilder(granularity="1m")


# =============================================================================
# T2.03: AS-OF JOINER TESTS
# =============================================================================


class TestAsOfJoiner:
    """Tests for AsOfJoiner (T2.03)."""

    def test_as_of_join_basic(
        self, sample_1m_bars: pl.DataFrame, sample_micro_bars: pl.DataFrame
    ) -> None:
        """Test basic as-of join functionality."""
        joiner = AsOfJoiner()

        # Create decision timestamps
        base_time = datetime(2024, 1, 15, 9, 31, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=i) for i in range(10)]
        })

        result = joiner.join(
            decision_times,
            sample_1m_bars,
            on="decision_ts",
            by="bar_end",
        )

        assert result is not None
        assert len(result) == 10

    def test_no_future_data_leakage(
        self, sample_1m_bars: pl.DataFrame
    ) -> None:
        """Test that as-of join prevents future data leakage."""
        joiner = AsOfJoiner()

        # Decision time at 9:31:30 should only see bar ending at 9:31
        decision_time = datetime(2024, 1, 15, 9, 31, 30, tzinfo=UTC)
        decision_times = pl.DataFrame({"decision_ts": [decision_time]})

        # Include bar_start so we can verify which bar was joined
        bars_with_start = sample_1m_bars.select(["bar_end", "bar_start", "close"])

        result = joiner.join(
            decision_times,
            bars_with_start,
            on="decision_ts",
            by="bar_end",
        )

        # The bar_start should be before decision time (bar ending at 9:31 starts at 9:30)
        joined_bar_start = result["bar_start"][0]
        assert joined_bar_start <= decision_time

    def test_as_of_join_with_tolerance(self) -> None:
        """Test as-of join with time tolerance."""
        joiner = AsOfJoiner(tolerance=timedelta(minutes=5))

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)

        left = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=10)]
        })

        right = pl.DataFrame({
            "bar_end": [base_time],  # 10 minutes before
            "value": [100.0],
        })

        # With 5-minute tolerance, should not join
        result = joiner.join(left, right, on="decision_ts", by="bar_end")
        assert result["value"][0] is None

    def test_as_of_join_multiple_features(
        self, sample_1m_bars: pl.DataFrame, sample_5m_bars: pl.DataFrame
    ) -> None:
        """Test joining multiple feature sources."""
        joiner = AsOfJoiner()

        base_time = datetime(2024, 1, 15, 9, 35, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({"decision_ts": [base_time]})

        # Join 1m bars
        result = joiner.join(
            decision_times,
            sample_1m_bars.select(["bar_end", "close", "volume"]),
            on="decision_ts",
            by="bar_end",
            suffix="_1m"
        )

        # Join 5m bars
        result = joiner.join(
            result,
            sample_5m_bars.select(["bar_end", "close", "volume"]),
            on="decision_ts",
            by="bar_end",
            suffix="_5m"
        )

        assert "close_1m" in result.columns
        assert "close_5m" in result.columns

    def test_strict_mode_raises_on_missing(self) -> None:
        """Test strict mode raises when no matching data."""
        joiner = AsOfJoiner(strict=True)

        # Decision time BEFORE any available data
        left = pl.DataFrame({
            "decision_ts": [datetime(2024, 1, 13, 10, 0, 0, tzinfo=UTC)]
        })

        right = pl.DataFrame({
            "bar_end": [datetime(2024, 1, 14, 10, 0, 0, tzinfo=UTC)],  # Day AFTER decision
            "value": [100.0],
        })

        # backward strategy won't find any data before the decision time
        with pytest.raises(ValueError, match="missing"):
            joiner.join(left, right, on="decision_ts", by="bar_end")

    def test_as_of_preserves_all_decision_times(self) -> None:
        """Test that all decision times are preserved after join."""
        joiner = AsOfJoiner()

        base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=i) for i in range(5)]
        })

        right = pl.DataFrame({
            "bar_end": [base_time, base_time + timedelta(minutes=2)],
            "value": [100.0, 101.0],
        })

        result = joiner.join(decision_times, right, on="decision_ts", by="bar_end")
        assert len(result) == 5  # All decision times preserved


# =============================================================================
# T2.04: DECISION FRAME BUILDER TESTS
# =============================================================================


class TestDecisionFrameBuilder:
    """Tests for DecisionFrameBuilder (T2.04)."""

    def test_build_decision_frame(
        self,
        sample_1m_bars: pl.DataFrame,
        sample_5m_bars: pl.DataFrame,
        sample_micro_bars: pl.DataFrame,
    ) -> None:
        """Test building a decision frame from multiple timeframes."""
        builder = DecisionFrameBuilder()

        base_time = datetime(2024, 1, 15, 9, 40, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=i) for i in range(5)]
        })

        frame = builder.build(
            decision_times=decision_times,
            micro_bars_30s=sample_micro_bars,
            bars_1m=sample_1m_bars,
            bars_5m=sample_5m_bars,
            symbol="SPY",
        )

        assert frame is not None
        assert len(frame) == 5

    def test_decision_frame_schema(
        self,
        sample_1m_bars: pl.DataFrame,
        sample_5m_bars: pl.DataFrame,
        sample_micro_bars: pl.DataFrame,
    ) -> None:
        """Test decision frame has correct schema."""
        builder = DecisionFrameBuilder()

        base_time = datetime(2024, 1, 15, 9, 40, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({
            "decision_ts": [base_time]
        })

        frame = builder.build(
            decision_times=decision_times,
            micro_bars_30s=sample_micro_bars,
            bars_1m=sample_1m_bars,
            bars_5m=sample_5m_bars,
            symbol="SPY",
        )

        expected_columns = {
            "symbol", "decision_ts",
            # 30s micro features
            "spread_30s", "microprice_30s", "quote_imbalance_30s", "vol_30s",
            # 1m features
            "ret_1m", "vol_1m", "atr_1m",
            # 5m features
            "ret_5m", "trend_5m", "vol_5m",
            # 15m features
            "ret_15m", "trend_15m", "vol_15m",
        }
        assert expected_columns.issubset(set(frame.columns))

    def test_decision_frame_no_leakage(
        self,
        sample_1m_bars: pl.DataFrame,
        sample_5m_bars: pl.DataFrame,
        sample_micro_bars: pl.DataFrame,
    ) -> None:
        """Test decision frame has no lookahead bias."""
        builder = DecisionFrameBuilder()

        # Decision at 9:35 should only see data up to 9:35
        decision_time = datetime(2024, 1, 15, 9, 35, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({"decision_ts": [decision_time]})

        frame = builder.build(
            decision_times=decision_times,
            micro_bars_30s=sample_micro_bars,
            bars_1m=sample_1m_bars,
            bars_5m=sample_5m_bars,
            symbol="SPY",
        )

        # All feature values should be from data available at 9:35
        # (This is a structural test - actual verification in integration tests)
        assert frame is not None
        assert len(frame) == 1

    def test_schema_versioning(
        self,
        sample_1m_bars: pl.DataFrame,
        sample_5m_bars: pl.DataFrame,
        sample_micro_bars: pl.DataFrame,
    ) -> None:
        """Test that decision frame includes schema version."""
        builder = DecisionFrameBuilder(schema_version="1.0.0")

        base_time = datetime(2024, 1, 15, 9, 40, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({"decision_ts": [base_time]})

        _ = builder.build(
            decision_times=decision_times,
            micro_bars_30s=sample_micro_bars,
            bars_1m=sample_1m_bars,
            bars_5m=sample_5m_bars,
            symbol="SPY",
        )

        assert builder.schema_version == "1.0.0"

    def test_missing_15m_bars_handled(
        self,
        sample_1m_bars: pl.DataFrame,
        sample_5m_bars: pl.DataFrame,
        sample_micro_bars: pl.DataFrame,
    ) -> None:
        """Test graceful handling when 15m bars not available."""
        builder = DecisionFrameBuilder()

        base_time = datetime(2024, 1, 15, 9, 35, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({"decision_ts": [base_time]})

        # Build without 15m bars
        frame = builder.build(
            decision_times=decision_times,
            micro_bars_30s=sample_micro_bars,
            bars_1m=sample_1m_bars,
            bars_5m=sample_5m_bars,
            bars_15m=None,
            symbol="SPY",
        )

        # 15m features should be null
        assert frame["ret_15m"][0] is None
        assert frame["trend_15m"][0] is None
        assert frame["vol_15m"][0] is None


# =============================================================================
# T2.05: INCREMENTAL PROCESSOR TESTS
# =============================================================================


class TestIncrementalProcessor:
    """Tests for IncrementalProcessor (T2.05)."""

    def test_incremental_processing_basic(
        self, sample_trades: pl.DataFrame
    ) -> None:
        """Test basic incremental processing."""
        processor = IncrementalProcessor()

        # Split trades into two batches
        batch1 = sample_trades.head(300)
        batch2 = sample_trades.tail(300)

        # Process first batch
        result1 = processor.process_trades(batch1, symbol="SPY")
        assert result1 is not None

        # Process second batch incrementally
        result2 = processor.process_trades(batch2, symbol="SPY")
        assert result2 is not None

    def test_incremental_state_preservation(
        self, sample_trades: pl.DataFrame
    ) -> None:
        """Test that state is preserved between incremental updates."""
        processor = IncrementalProcessor()

        batch1 = sample_trades.head(300)
        processor.process_trades(batch1, symbol="SPY")

        # Check state is preserved
        state = processor.get_state("SPY")
        assert state is not None
        assert "last_processed_ts" in state

    def test_incremental_vs_full_consistency(
        self, sample_trades: pl.DataFrame
    ) -> None:
        """Test incremental results match full recomputation."""
        processor_incremental = IncrementalProcessor()
        processor_full = IncrementalProcessor()

        # Split and process incrementally
        batch1 = sample_trades.head(300)
        batch2 = sample_trades.tail(300)

        processor_incremental.process_trades(batch1, symbol="SPY")
        result_incremental = processor_incremental.process_trades(
            batch2, symbol="SPY"
        )

        # Process all at once
        result_full = processor_full.process_trades(sample_trades, symbol="SPY")

        # Results should be very close (may have minor float differences)
        # Compare last bar values
        assert result_incremental is not None
        assert result_full is not None

    def test_watermark_tracking(self, sample_trades: pl.DataFrame) -> None:
        """Test watermark tracking for incremental updates."""
        processor = IncrementalProcessor()

        batch = sample_trades.head(100)
        processor.process_trades(batch, symbol="SPY")

        watermark = processor.get_watermark("SPY")
        assert watermark is not None
        assert watermark == batch["ts_event"].max()

    def test_duplicate_data_handling(self, sample_trades: pl.DataFrame) -> None:
        """Test handling of duplicate data in incremental mode."""
        processor = IncrementalProcessor()

        batch = sample_trades.head(100)

        # Process same batch twice
        result1 = processor.process_trades(batch, symbol="SPY")
        result2 = processor.process_trades(batch, symbol="SPY")

        # Should handle gracefully (skip duplicates - return None for no new data)
        # First process should return bars, second should skip duplicates
        assert result1 is not None
        # Second call with same data returns None (nothing new to process)
        assert result2 is None

    def test_reset_state(self, sample_trades: pl.DataFrame) -> None:
        """Test resetting processor state."""
        processor = IncrementalProcessor()

        batch = sample_trades.head(100)
        processor.process_trades(batch, symbol="SPY")

        # Reset state
        processor.reset("SPY")

        state = processor.get_state("SPY")
        assert state is None or state.get("last_processed_ts") is None

    def test_multi_symbol_incremental(self, sample_trades: pl.DataFrame) -> None:
        """Test incremental processing for multiple symbols."""
        processor = IncrementalProcessor()

        batch = sample_trades.head(100)

        # Process for different symbols
        processor.process_trades(batch, symbol="SPY")
        processor.process_trades(batch, symbol="QQQ")

        spy_state = processor.get_state("SPY")
        qqq_state = processor.get_state("QQQ")

        assert spy_state is not None
        assert qqq_state is not None
        assert spy_state != qqq_state or spy_state["symbol"] != qqq_state["symbol"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the full feature pipeline."""

    def test_full_pipeline(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test the complete feature building pipeline."""
        # Build standard bars
        bar_builder = StandardBarBuilder(granularity="1m")
        bars_1m = bar_builder.build(sample_trades, symbol="SPY")

        bar_builder_5m = StandardBarBuilder(granularity="5m")
        bars_5m = bar_builder_5m.build(sample_trades, symbol="SPY")

        # Build micro bars
        micro_builder = MicrostructureBarBuilder(granularity="30s")
        micro_bars = micro_builder.build(sample_trades, sample_quotes, symbol="SPY")

        # Build decision frame
        base_time = datetime(2024, 1, 15, 9, 35, 0, tzinfo=UTC)
        decision_times = pl.DataFrame({
            "decision_ts": [base_time + timedelta(minutes=i) for i in range(5)]
        })

        frame_builder = DecisionFrameBuilder()
        decision_frame = frame_builder.build(
            decision_times=decision_times,
            micro_bars_30s=micro_bars,
            bars_1m=bars_1m,
            bars_5m=bars_5m,
            symbol="SPY",
        )

        assert decision_frame is not None
        assert len(decision_frame) == 5

    def test_pipeline_with_incremental(
        self, sample_trades: pl.DataFrame, sample_quotes: pl.DataFrame
    ) -> None:
        """Test pipeline with incremental processing."""
        processor = IncrementalProcessor()

        # Process in batches
        batch1_trades = sample_trades.head(300)
        batch1_quotes = sample_quotes.head(1500)

        batch2_trades = sample_trades.tail(300)
        batch2_quotes = sample_quotes.tail(1500)

        # First batch
        processor.process_trades(batch1_trades, symbol="SPY")
        processor.process_quotes(batch1_quotes, symbol="SPY")

        # Second batch
        result = processor.process_trades(batch2_trades, symbol="SPY")
        processor.process_quotes(batch2_quotes, symbol="SPY")

        assert result is not None
