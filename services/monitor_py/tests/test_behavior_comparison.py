"""Tests for Shadow vs Backtest Behavior Comparison."""

from datetime import UTC, datetime, timedelta

import pytest

from monitor_py.behavior_comparison import (
    BehaviorComparator,
    BehaviorConfig,
    BehaviorStats,
    ExecutionRecord,
    SignalRecord,
)


class TestBehaviorConfig:
    def test_default_config(self) -> None:
        config = BehaviorConfig()
        assert config.signal_alignment_threshold == 0.95
        assert config.timing_tolerance_ms == 100.0
        assert config.drift_warning_threshold == 0.05
        assert config.drift_critical_threshold == 0.10
        assert config.window_size == 1000

    def test_custom_config(self) -> None:
        config = BehaviorConfig(
            signal_alignment_threshold=0.99,
            timing_tolerance_ms=50.0,
            drift_warning_threshold=0.03,
            drift_critical_threshold=0.08,
            window_size=500,
        )
        assert config.signal_alignment_threshold == 0.99
        assert config.timing_tolerance_ms == 50.0


class TestSignalRecord:
    def test_signal_record_buy(self) -> None:
        record = SignalRecord(signal_id="SIG001", symbol="AAPL", signal_type="buy",
                              strength=0.85, timestamp=datetime.now(UTC), source="backtest")
        assert record.signal_type == "buy"
        assert record.strength == 0.85
        assert record.source == "backtest"

    def test_signal_record_sell(self) -> None:
        record = SignalRecord(signal_id="SIG002", symbol="AAPL", signal_type="sell",
                              strength=0.70, timestamp=datetime.now(UTC), source="shadow")
        assert record.signal_type == "sell"
        assert record.source == "shadow"


class TestExecutionRecord:
    def test_execution_record(self) -> None:
        record = ExecutionRecord(execution_id="EXE001", signal_id="SIG001", symbol="AAPL",
                                 side="buy", quantity=100.0, price=150.0,
                                 timestamp=datetime.now(UTC), source="backtest")
        assert record.side == "buy"
        assert record.notional_value == pytest.approx(15000.0, rel=0.01)


class TestBehaviorComparator:
    def test_record_backtest_signal(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        record = comparator.record_backtest_signal("SIG001", "AAPL", "buy", 0.85)
        assert record.source == "backtest"
        assert record.signal_type == "buy"

    def test_record_shadow_signal(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        record = comparator.record_shadow_signal("SIG001", "AAPL", "buy", 0.85)
        assert record.source == "shadow"

    def test_record_backtest_execution(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        record = comparator.record_backtest_execution(
            "EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0
        )
        assert record.source == "backtest"

    def test_record_shadow_execution(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        record = comparator.record_shadow_execution("EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0)
        assert record.source == "shadow"

    def test_get_stats_empty(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        stats = comparator.get_stats()
        assert stats.total_signals == 0
        assert stats.aligned_signals == 0
        assert stats.signal_alignment_rate == 0.0
        assert stats.drift_score == 0.0
        assert stats.alert_level == "none"

    def test_signal_alignment_perfect(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        now = datetime.now(UTC)
        for i in range(10):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.signal_alignment_rate == pytest.approx(1.0, rel=0.01)
        assert stats.alert_level == "none"

    def test_signal_alignment_partial(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        now = datetime.now(UTC)
        for i in range(8):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG008", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG008", "AAPL", "sell", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG009", "AAPL", "buy", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.signal_alignment_rate == pytest.approx(0.80, rel=0.01)

    def test_timing_comparison(self) -> None:
        config = BehaviorConfig(timing_tolerance_ms=100.0)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        comparator.record_backtest_execution(
            "EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0, timestamp=now
        )
        comparator.record_shadow_execution(
            "EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0,
            timestamp=now + timedelta(milliseconds=50),
        )
        stats = comparator.get_stats()
        assert stats.avg_timing_drift_ms == pytest.approx(50.0, rel=0.1)
        assert stats.timing_within_tolerance is True

    def test_timing_drift_exceeded(self) -> None:
        config = BehaviorConfig(timing_tolerance_ms=100.0)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        comparator.record_backtest_execution(
            "EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0, timestamp=now
        )
        comparator.record_shadow_execution(
            "EXE001", "SIG001", "AAPL", "buy", 100.0, 150.0,
            timestamp=now + timedelta(milliseconds=200),
        )
        stats = comparator.get_stats()
        assert stats.avg_timing_drift_ms == pytest.approx(200.0, rel=0.1)
        assert stats.timing_within_tolerance is False

    def test_drift_score_calculation(self) -> None:
        config = BehaviorConfig(drift_warning_threshold=0.05, drift_critical_threshold=0.10)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(8):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG008", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG008", "AAPL", "buy", 0.5, timestamp=now)
        comparator.record_backtest_signal("SIG009", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG009", "AAPL", "sell", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.drift_score > 0.0

    def test_alert_level_warning(self) -> None:
        config = BehaviorConfig(drift_warning_threshold=0.05, drift_critical_threshold=0.10)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(93):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        for i in range(7):
            comparator.record_backtest_signal(f"SIG{93+i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{93+i:03d}", "AAPL", "sell", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.alert_level == "warning"

    def test_alert_level_critical(self) -> None:
        config = BehaviorConfig(drift_warning_threshold=0.05, drift_critical_threshold=0.10)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(85):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        for i in range(15):
            comparator.record_backtest_signal(f"SIG{85+i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{85+i:03d}", "AAPL", "sell", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.alert_level == "critical"

    def test_should_alert_false(self) -> None:
        config = BehaviorConfig(drift_warning_threshold=0.05)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(10):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        assert comparator.should_alert() is False

    def test_should_alert_true(self) -> None:
        config = BehaviorConfig(drift_warning_threshold=0.05)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(5):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        for i in range(5):
            comparator.record_backtest_signal(f"SIG{5+i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{5+i:03d}", "AAPL", "sell", 0.8, timestamp=now)
        assert comparator.should_alert() is True

    def test_window_size_limit(self) -> None:
        config = BehaviorConfig(window_size=5)
        comparator = BehaviorComparator(config)
        now = datetime.now(UTC)
        for i in range(3):
            comparator.record_backtest_signal(f"SIG{i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{i:03d}", "AAPL", "sell", 0.8, timestamp=now)
        for i in range(7):
            comparator.record_backtest_signal(f"SIG{3+i:03d}", "AAPL", "buy", 0.8, timestamp=now)
            comparator.record_shadow_signal(f"SIG{3+i:03d}", "AAPL", "buy", 0.8, timestamp=now)
        stats = comparator.get_stats()
        assert stats.total_signals == 5
        assert stats.signal_alignment_rate == pytest.approx(1.0, rel=0.01)

    def test_get_stats_by_symbol(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        now = datetime.now(UTC)
        comparator.record_backtest_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG002", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG002", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG003", "GOOGL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG003", "GOOGL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG004", "GOOGL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG004", "GOOGL", "sell", 0.8, timestamp=now)
        aapl_stats = comparator.get_stats(symbol="AAPL")
        assert aapl_stats.total_signals == 2
        assert aapl_stats.signal_alignment_rate == pytest.approx(1.0, rel=0.01)
        googl_stats = comparator.get_stats(symbol="GOOGL")
        assert googl_stats.total_signals == 2
        assert googl_stats.signal_alignment_rate == pytest.approx(0.5, rel=0.01)

    def test_get_divergence_report(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        now = datetime.now(UTC)
        comparator.record_backtest_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_backtest_signal("SIG002", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG002", "AAPL", "sell", 0.8, timestamp=now)
        report = comparator.get_divergence_report()
        assert len(report) == 1
        assert report[0]["signal_id"] == "SIG002"
        assert report[0]["backtest_type"] == "buy"
        assert report[0]["shadow_type"] == "sell"

    def test_reset_stats(self) -> None:
        comparator = BehaviorComparator(BehaviorConfig())
        now = datetime.now(UTC)
        comparator.record_backtest_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        comparator.record_shadow_signal("SIG001", "AAPL", "buy", 0.8, timestamp=now)
        assert comparator.get_stats().total_signals == 1
        comparator.reset()
        assert comparator.get_stats().total_signals == 0


class TestBehaviorStats:
    def test_stats_creation(self) -> None:
        stats = BehaviorStats(
            total_signals=100,
            aligned_signals=95,
            signal_alignment_rate=0.95,
            total_executions=80,
            aligned_executions=78,
            execution_alignment_rate=0.975,
            avg_timing_drift_ms=50.0,
            max_timing_drift_ms=150.0,
            timing_within_tolerance=True,
            drift_score=0.03,
            alert_level="none",
        )
        assert stats.total_signals == 100
        assert stats.signal_alignment_rate == 0.95
        assert stats.alert_level == "none"
