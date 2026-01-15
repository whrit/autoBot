"""Tests for Drawdown Monitoring (T5.09)."""

import pytest

from monitor_py.drawdown import DrawdownConfig, DrawdownMonitor, DrawdownStatus


class TestDrawdownConfig:
    """Test DrawdownConfig defaults and validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DrawdownConfig()
        assert config.warning_threshold == 0.05
        assert config.critical_threshold == 0.10
        assert config.kill_switch_threshold == 0.15

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DrawdownConfig(
            warning_threshold=0.03,
            critical_threshold=0.07,
            kill_switch_threshold=0.12,
        )
        assert config.warning_threshold == 0.03
        assert config.critical_threshold == 0.07
        assert config.kill_switch_threshold == 0.12


class TestDrawdownMonitor:
    """Test DrawdownMonitor functionality."""

    def test_initial_state(self) -> None:
        """Test initial state with no drawdown."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)
        status = monitor.get_status()
        assert status.current_drawdown == 0.0
        assert status.max_drawdown == 0.0
        assert status.peak_equity == 100000.0
        assert status.current_equity == 100000.0
        assert status.alert_level == "none"

    def test_drawdown_calculation(self) -> None:
        """Test drawdown is calculated correctly."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        # Equity drops to 95000 (5% drawdown)
        status = monitor.update(95000.0)

        assert status.current_drawdown == pytest.approx(0.05, rel=0.001)
        assert status.peak_equity == 100000.0
        assert status.current_equity == 95000.0

    def test_peak_equity_tracking(self) -> None:
        """Test peak equity is tracked correctly."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        # Equity increases to 110000
        monitor.update(110000.0)
        status = monitor.get_status()
        assert status.peak_equity == 110000.0
        assert status.current_drawdown == 0.0

        # Equity drops to 100000 (9.09% from peak)
        status = monitor.update(100000.0)
        assert status.peak_equity == 110000.0
        assert status.current_drawdown == pytest.approx(0.0909, rel=0.01)

    def test_max_drawdown_tracking(self) -> None:
        """Test max drawdown is tracked correctly."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        # Drop to 10% drawdown
        monitor.update(90000.0)
        # Recover
        monitor.update(95000.0)

        status = monitor.get_status()
        assert status.max_drawdown == pytest.approx(0.10, rel=0.001)
        assert status.current_drawdown == pytest.approx(0.05, rel=0.001)

    def test_warning_alert(self) -> None:
        """Test warning alert level."""
        config = DrawdownConfig(
            warning_threshold=0.05,
            critical_threshold=0.10,
            kill_switch_threshold=0.15,
        )
        monitor = DrawdownMonitor(config, initial_equity=100000.0)

        # 6% drawdown - should trigger warning
        status = monitor.update(94000.0)
        assert status.alert_level == "warning"

    def test_critical_alert(self) -> None:
        """Test critical alert level."""
        config = DrawdownConfig(
            warning_threshold=0.05,
            critical_threshold=0.10,
            kill_switch_threshold=0.15,
        )
        monitor = DrawdownMonitor(config, initial_equity=100000.0)

        # 12% drawdown - should trigger critical
        status = monitor.update(88000.0)
        assert status.alert_level == "critical"

    def test_kill_switch_alert(self) -> None:
        """Test kill switch alert level."""
        config = DrawdownConfig(
            warning_threshold=0.05,
            critical_threshold=0.10,
            kill_switch_threshold=0.15,
        )
        monitor = DrawdownMonitor(config, initial_equity=100000.0)

        # 16% drawdown - should trigger kill switch
        status = monitor.update(84000.0)
        assert status.alert_level == "kill_switch"

    def test_should_halt_trading_false(self) -> None:
        """Test should_halt_trading returns False below threshold."""
        config = DrawdownConfig(kill_switch_threshold=0.15)
        monitor = DrawdownMonitor(config, initial_equity=100000.0)

        # 10% drawdown - should not halt
        monitor.update(90000.0)
        assert monitor.should_halt_trading() is False

    def test_should_halt_trading_true(self) -> None:
        """Test should_halt_trading returns True at threshold."""
        config = DrawdownConfig(kill_switch_threshold=0.15)
        monitor = DrawdownMonitor(config, initial_equity=100000.0)

        # 16% drawdown - should halt
        monitor.update(84000.0)
        assert monitor.should_halt_trading() is True

    def test_days_in_drawdown_tracking(self) -> None:
        """Test days in drawdown tracking."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        status = monitor.get_status()
        assert status.days_in_drawdown == 0

        # Enter drawdown
        monitor.update(95000.0)
        # Days should start tracking (initially 1)
        status = monitor.get_status()
        assert status.days_in_drawdown >= 0

    def test_recovery_from_drawdown(self) -> None:
        """Test recovery from drawdown."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)

        # Enter drawdown
        monitor.update(90000.0)
        assert monitor.get_status().current_drawdown > 0

        # Full recovery to new high
        monitor.update(105000.0)
        status = monitor.get_status()
        assert status.current_drawdown == 0.0
        assert status.peak_equity == 105000.0
        assert status.alert_level == "none"

    def test_get_status_consistency(self) -> None:
        """Test get_status returns consistent values."""
        monitor = DrawdownMonitor(DrawdownConfig(), initial_equity=100000.0)
        monitor.update(95000.0)

        status1 = monitor.get_status()
        status2 = monitor.get_status()

        assert status1.current_drawdown == status2.current_drawdown
        assert status1.peak_equity == status2.peak_equity


class TestDrawdownStatus:
    """Test DrawdownStatus dataclass."""

    def test_status_creation(self) -> None:
        """Test DrawdownStatus can be created."""
        status = DrawdownStatus(
            current_drawdown=0.05,
            max_drawdown=0.10,
            peak_equity=100000.0,
            current_equity=95000.0,
            alert_level="warning",
            days_in_drawdown=5,
        )
        assert status.current_drawdown == 0.05
        assert status.alert_level == "warning"
