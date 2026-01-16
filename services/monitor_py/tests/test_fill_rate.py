"""Tests for Fill Rate Monitoring."""

from datetime import UTC, datetime

import pytest

from monitor_py.fill_rate import FillEvent, FillRateConfig, FillRateStats, FillRateTracker


class TestFillRateConfig:
    def test_default_config(self) -> None:
        config = FillRateConfig()
        assert config.expected_fill_rate == 0.95
        assert config.warning_threshold == 0.90
        assert config.critical_threshold == 0.80
        assert config.window_size == 100
        assert config.latency_warning_ms == 500.0

    def test_custom_config(self) -> None:
        config = FillRateConfig(expected_fill_rate=0.98, warning_threshold=0.95,
                                critical_threshold=0.85, window_size=200, latency_warning_ms=250.0)
        assert config.expected_fill_rate == 0.98


class TestFillEvent:
    def test_fill_event_full_fill(self) -> None:
        event = FillEvent(order_id="ORD001", symbol="AAPL", requested_qty=100.0,
                          filled_qty=100.0, fill_time_ms=150.0, timestamp=datetime.now(UTC))
        assert event.fill_ratio == 1.0
        assert event.is_full_fill is True
        assert event.is_partial_fill is False

    def test_fill_event_partial_fill(self) -> None:
        event = FillEvent(order_id="ORD002", symbol="AAPL", requested_qty=100.0,
                          filled_qty=75.0, fill_time_ms=200.0, timestamp=datetime.now(UTC))
        assert event.fill_ratio == pytest.approx(0.75, rel=0.01)
        assert event.is_partial_fill is True

    def test_fill_event_no_fill(self) -> None:
        event = FillEvent(order_id="ORD003", symbol="AAPL", requested_qty=100.0,
                          filled_qty=0.0, fill_time_ms=0.0, timestamp=datetime.now(UTC))
        assert event.fill_ratio == 0.0
        assert event.is_full_fill is False


class TestFillRateTracker:
    def test_record_fill_full(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        event = tracker.record_fill("ORD001", "AAPL", 100.0, 100.0, 150.0)
        assert event.fill_ratio == 1.0

    def test_record_fill_partial(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        event = tracker.record_fill("ORD002", "AAPL", 100.0, 50.0, 200.0)
        assert event.fill_ratio == pytest.approx(0.5, rel=0.01)

    def test_get_stats_empty(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        stats = tracker.get_stats()
        assert stats.total_orders == 0
        assert stats.fill_rate == 0.0
        assert stats.alert_level == "none"

    def test_get_stats_with_data(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        for i in range(8):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 150.0)
        tracker.record_fill("ORD008", "AAPL", 100.0, 50.0, 200.0)
        tracker.record_fill("ORD009", "AAPL", 100.0, 0.0, 0.0)
        stats = tracker.get_stats()
        assert stats.total_orders == 10
        assert stats.full_fills == 8
        assert stats.partial_fills == 1
        assert stats.unfilled == 1
        assert stats.fill_rate == pytest.approx(0.85, rel=0.01)

    def test_fill_rate_calculation(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        tracker.record_fill("ORD001", "AAPL", 100.0, 100.0, 100.0)
        tracker.record_fill("ORD002", "AAPL", 100.0, 50.0, 100.0)
        tracker.record_fill("ORD003", "AAPL", 100.0, 75.0, 100.0)
        stats = tracker.get_stats()
        assert stats.fill_rate == pytest.approx(0.75, rel=0.01)

    def test_alert_level_none(self) -> None:
        config = FillRateConfig(warning_threshold=0.90, critical_threshold=0.80)
        tracker = FillRateTracker(config)
        for i in range(10):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        assert tracker.get_stats().alert_level == "none"

    def test_alert_level_warning(self) -> None:
        config = FillRateConfig(warning_threshold=0.90, critical_threshold=0.80)
        tracker = FillRateTracker(config)
        for i in range(85):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        for i in range(15):
            tracker.record_fill(f"ORD{85+i:03d}", "AAPL", 100.0, 0.0, 0.0)
        assert tracker.get_stats().alert_level == "warning"

    def test_alert_level_critical(self) -> None:
        config = FillRateConfig(warning_threshold=0.90, critical_threshold=0.80)
        tracker = FillRateTracker(config)
        for i in range(75):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        for i in range(25):
            tracker.record_fill(f"ORD{75+i:03d}", "AAPL", 100.0, 0.0, 0.0)
        assert tracker.get_stats().alert_level == "critical"

    def test_should_alert_false(self) -> None:
        config = FillRateConfig(warning_threshold=0.90)
        tracker = FillRateTracker(config)
        for i in range(10):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        assert tracker.should_alert() is False

    def test_should_alert_true(self) -> None:
        config = FillRateConfig(warning_threshold=0.90)
        tracker = FillRateTracker(config)
        for i in range(5):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        for i in range(5):
            tracker.record_fill(f"ORD{5+i:03d}", "AAPL", 100.0, 0.0, 0.0)
        assert tracker.should_alert() is True

    def test_window_size_limit(self) -> None:
        config = FillRateConfig(window_size=5)
        tracker = FillRateTracker(config)
        for i in range(3):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 0.0, 0.0)
        for i in range(7):
            tracker.record_fill(f"ORD{3+i:03d}", "AAPL", 100.0, 100.0, 100.0)
        stats = tracker.get_stats()
        assert stats.total_orders == 5
        assert stats.fill_rate == pytest.approx(1.0, rel=0.01)

    def test_latency_metrics(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        tracker.record_fill("ORD001", "AAPL", 100.0, 100.0, 100.0)
        tracker.record_fill("ORD002", "AAPL", 100.0, 100.0, 200.0)
        tracker.record_fill("ORD003", "AAPL", 100.0, 100.0, 300.0)
        stats = tracker.get_stats()
        assert stats.avg_latency_ms == pytest.approx(200.0, rel=0.01)
        assert stats.max_latency_ms == pytest.approx(300.0, rel=0.01)

    def test_latency_warning(self) -> None:
        config = FillRateConfig(latency_warning_ms=250.0)
        tracker = FillRateTracker(config)
        for i in range(5):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 300.0)
        assert tracker.get_stats().latency_warning is True

    def test_get_stats_by_symbol(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        tracker.record_fill("ORD001", "AAPL", 100.0, 100.0, 100.0)
        tracker.record_fill("ORD002", "AAPL", 100.0, 50.0, 100.0)
        tracker.record_fill("ORD003", "GOOGL", 100.0, 100.0, 100.0)
        tracker.record_fill("ORD004", "GOOGL", 100.0, 100.0, 100.0)
        aapl_stats = tracker.get_stats(symbol="AAPL")
        assert aapl_stats.total_orders == 2
        assert aapl_stats.fill_rate == pytest.approx(0.75, rel=0.01)
        googl_stats = tracker.get_stats(symbol="GOOGL")
        assert googl_stats.fill_rate == pytest.approx(1.0, rel=0.01)

    def test_reset_stats(self) -> None:
        tracker = FillRateTracker(FillRateConfig())
        for i in range(5):
            tracker.record_fill(f"ORD{i:03d}", "AAPL", 100.0, 100.0, 100.0)
        assert tracker.get_stats().total_orders == 5
        tracker.reset()
        assert tracker.get_stats().total_orders == 0


class TestFillRateStats:
    def test_stats_creation(self) -> None:
        stats = FillRateStats(total_orders=100, full_fills=85, partial_fills=10,
                              unfilled=5, fill_rate=0.90, avg_latency_ms=150.0,
                              max_latency_ms=500.0, alert_level="none", latency_warning=False)
        assert stats.total_orders == 100
        assert stats.fill_rate == 0.90
