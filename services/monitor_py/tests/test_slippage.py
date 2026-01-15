"""Tests for Slippage Tracking (T5.08)."""

import pytest

from monitor_py.slippage import SlippageConfig, SlippageStats, SlippageTracker


class TestSlippageConfig:
    """Test SlippageConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SlippageConfig()
        assert config.expected_slippage_bps == 2.0
        assert config.alert_threshold_multiplier == 2.0
        assert config.window_size == 100

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SlippageConfig(
            expected_slippage_bps=5.0,
            alert_threshold_multiplier=3.0,
            window_size=50,
        )
        assert config.expected_slippage_bps == 5.0
        assert config.alert_threshold_multiplier == 3.0
        assert config.window_size == 50


class TestSlippageTracker:
    """Test SlippageTracker functionality."""

    def test_record_fill_buy(self) -> None:
        """Test recording a buy fill with slippage."""
        tracker = SlippageTracker(SlippageConfig())
        # Expected: 100.00, Actual: 100.02 (2 bps slippage)
        slippage = tracker.record_fill(
            expected_price=100.00,
            actual_price=100.02,
            side="buy",
        )
        assert slippage == pytest.approx(2.0, rel=0.01)

    def test_record_fill_sell(self) -> None:
        """Test recording a sell fill with slippage."""
        tracker = SlippageTracker(SlippageConfig())
        # For sell: Expected 100.00, Actual 99.98 (2 bps slippage - worse price)
        slippage = tracker.record_fill(
            expected_price=100.00,
            actual_price=99.98,
            side="sell",
        )
        assert slippage == pytest.approx(2.0, rel=0.01)

    def test_record_fill_negative_slippage_buy(self) -> None:
        """Test recording a buy fill with price improvement."""
        tracker = SlippageTracker(SlippageConfig())
        # Expected: 100.00, Actual: 99.98 (better price for buy)
        slippage = tracker.record_fill(
            expected_price=100.00,
            actual_price=99.98,
            side="buy",
        )
        assert slippage == pytest.approx(-2.0, rel=0.01)

    def test_record_fill_negative_slippage_sell(self) -> None:
        """Test recording a sell fill with price improvement."""
        tracker = SlippageTracker(SlippageConfig())
        # For sell: Expected 100.00, Actual 100.02 (better price for sell)
        slippage = tracker.record_fill(
            expected_price=100.00,
            actual_price=100.02,
            side="sell",
        )
        assert slippage == pytest.approx(-2.0, rel=0.01)

    def test_get_stats_empty(self) -> None:
        """Test getting stats when no fills recorded."""
        tracker = SlippageTracker(SlippageConfig())
        stats = tracker.get_stats()
        assert stats.sample_count == 0
        assert stats.avg_slippage_bps == 0.0
        assert stats.max_slippage_bps == 0.0
        assert stats.alert_triggered is False

    def test_get_stats_with_data(self) -> None:
        """Test getting stats after recording fills."""
        config = SlippageConfig(expected_slippage_bps=2.0)
        tracker = SlippageTracker(config)

        # Record some fills
        tracker.record_fill(100.00, 100.01, "buy")  # 1 bps
        tracker.record_fill(100.00, 100.02, "buy")  # 2 bps
        tracker.record_fill(100.00, 100.03, "buy")  # 3 bps

        stats = tracker.get_stats()
        assert stats.sample_count == 3
        assert stats.avg_slippage_bps == pytest.approx(2.0, rel=0.01)
        assert stats.max_slippage_bps == pytest.approx(3.0, rel=0.01)
        assert stats.expected_slippage_bps == 2.0

    def test_should_alert_false(self) -> None:
        """Test no alert when slippage is normal."""
        config = SlippageConfig(expected_slippage_bps=2.0, alert_threshold_multiplier=2.0)
        tracker = SlippageTracker(config)

        # Record fills with slippage at expected level
        for _ in range(10):
            tracker.record_fill(100.00, 100.02, "buy")  # 2 bps

        assert tracker.should_alert() is False

    def test_should_alert_true(self) -> None:
        """Test alert when slippage exceeds threshold."""
        config = SlippageConfig(expected_slippage_bps=2.0, alert_threshold_multiplier=2.0)
        tracker = SlippageTracker(config)

        # Record fills with high slippage (5 bps = 2.5x expected)
        for _ in range(10):
            tracker.record_fill(100.00, 100.05, "buy")  # 5 bps

        assert tracker.should_alert() is True

    def test_window_size_limit(self) -> None:
        """Test that window size is respected."""
        config = SlippageConfig(window_size=5)
        tracker = SlippageTracker(config)

        # Record more fills than window size
        for i in range(10):
            tracker.record_fill(100.00, 100.00 + (i + 1) * 0.01, "buy")

        stats = tracker.get_stats()
        # Should only keep last 5 fills (6-10 bps)
        assert stats.sample_count == 5
        # Average should be around (6+7+8+9+10)/5 = 8 bps
        assert stats.avg_slippage_bps == pytest.approx(8.0, rel=0.1)

    def test_ratio_to_expected(self) -> None:
        """Test ratio calculation to expected slippage."""
        config = SlippageConfig(expected_slippage_bps=2.0)
        tracker = SlippageTracker(config)

        # Record fills with 4 bps slippage (2x expected)
        for _ in range(5):
            tracker.record_fill(100.00, 100.04, "buy")

        stats = tracker.get_stats()
        assert stats.ratio_to_expected == pytest.approx(2.0, rel=0.01)

    def test_alert_triggered_in_stats(self) -> None:
        """Test alert_triggered flag in stats."""
        config = SlippageConfig(expected_slippage_bps=2.0, alert_threshold_multiplier=2.0)
        tracker = SlippageTracker(config)

        # High slippage
        for _ in range(5):
            tracker.record_fill(100.00, 100.05, "buy")  # 5 bps

        stats = tracker.get_stats()
        assert stats.alert_triggered is True


class TestSlippageStats:
    """Test SlippageStats dataclass."""

    def test_stats_creation(self) -> None:
        """Test SlippageStats can be created."""
        stats = SlippageStats(
            avg_slippage_bps=2.5,
            max_slippage_bps=5.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=1.25,
            sample_count=100,
            alert_triggered=False,
        )
        assert stats.avg_slippage_bps == 2.5
        assert stats.sample_count == 100
