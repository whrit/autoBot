"""Tests for Turnover Analysis."""

from datetime import UTC, datetime, timedelta

import pytest

from monitor_py.turnover import TradeRecord, TurnoverAnalyzer, TurnoverConfig, TurnoverStats


class TestTurnoverConfig:
    def test_default_config(self) -> None:
        config = TurnoverConfig()
        assert config.daily_turnover_warning == 0.20
        assert config.daily_turnover_critical == 0.50
        assert config.weekly_turnover_warning == 1.0
        assert config.cost_per_turnover_bps == 5.0
        assert config.window_days == 30

    def test_custom_config(self) -> None:
        config = TurnoverConfig(
            daily_turnover_warning=0.10,
            daily_turnover_critical=0.30,
            weekly_turnover_warning=0.5,
            cost_per_turnover_bps=10.0,
            window_days=60,
        )
        assert config.daily_turnover_warning == 0.10
        assert config.cost_per_turnover_bps == 10.0


class TestTradeRecord:
    def test_trade_record_buy(self) -> None:
        record = TradeRecord(trade_id="TRD001", symbol="AAPL", side="buy", quantity=100.0,
                             price=150.0, timestamp=datetime.now(UTC))
        assert record.notional_value == pytest.approx(15000.0, rel=0.01)
        assert record.side == "buy"

    def test_trade_record_sell(self) -> None:
        record = TradeRecord(trade_id="TRD002", symbol="AAPL", side="sell", quantity=50.0,
                             price=155.0, timestamp=datetime.now(UTC))
        assert record.notional_value == pytest.approx(7750.0, rel=0.01)


class TestTurnoverAnalyzer:
    def test_record_trade(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        record = analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 150.0)
        assert record.notional_value == pytest.approx(15000.0, rel=0.01)

    def test_get_stats_empty(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        stats = analyzer.get_stats()
        assert stats.total_trades == 0
        assert stats.daily_turnover == 0.0
        assert stats.weekly_turnover == 0.0
        assert stats.total_turnover == 0.0
        assert stats.alert_level == "none"

    def test_daily_turnover_calculation(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        stats = analyzer.get_stats()
        assert stats.daily_turnover == pytest.approx(0.10, rel=0.01)

    def test_weekly_turnover_calculation(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        now = datetime.now(UTC)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0, timestamp=now)
        analyzer.record_trade(
            "TRD002", "AAPL", "sell", 100.0, 100.0, timestamp=now - timedelta(days=1)
        )
        analyzer.record_trade(
            "TRD003", "GOOGL", "buy", 50.0, 100.0, timestamp=now - timedelta(days=3)
        )
        stats = analyzer.get_stats()
        assert stats.weekly_turnover == pytest.approx(0.25, rel=0.01)

    def test_alert_level_none(self) -> None:
        config = TurnoverConfig(daily_turnover_warning=0.20, daily_turnover_critical=0.50)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        assert analyzer.get_stats().alert_level == "none"

    def test_alert_level_warning(self) -> None:
        config = TurnoverConfig(daily_turnover_warning=0.20, daily_turnover_critical=0.50)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 200.0, 150.0)
        assert analyzer.get_stats().alert_level == "warning"

    def test_alert_level_critical(self) -> None:
        config = TurnoverConfig(daily_turnover_warning=0.20, daily_turnover_critical=0.50)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 400.0, 150.0)
        assert analyzer.get_stats().alert_level == "critical"

    def test_cost_attribution(self) -> None:
        config = TurnoverConfig(cost_per_turnover_bps=10.0)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        stats = analyzer.get_stats()
        assert stats.estimated_cost == pytest.approx(10.0, rel=0.1)

    def test_should_alert_false(self) -> None:
        config = TurnoverConfig(daily_turnover_warning=0.20)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 50.0, 100.0)
        assert analyzer.should_alert() is False

    def test_should_alert_true(self) -> None:
        config = TurnoverConfig(daily_turnover_warning=0.20)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 500.0, 100.0)
        assert analyzer.should_alert() is True

    def test_window_days_limit(self) -> None:
        config = TurnoverConfig(window_days=7)
        analyzer = TurnoverAnalyzer(config, portfolio_value=100000.0)
        now = datetime.now(UTC)
        analyzer.record_trade(
            "TRD001", "AAPL", "buy", 100.0, 100.0, timestamp=now - timedelta(days=10)
        )
        analyzer.record_trade("TRD002", "AAPL", "buy", 50.0, 100.0, timestamp=now)
        stats = analyzer.get_stats()
        assert stats.total_turnover == pytest.approx(0.05, rel=0.01)

    def test_update_portfolio_value(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        stats1 = analyzer.get_stats()
        assert stats1.daily_turnover == pytest.approx(0.10, rel=0.01)
        analyzer.update_portfolio_value(200000.0)
        stats2 = analyzer.get_stats()
        assert stats2.daily_turnover == pytest.approx(0.05, rel=0.01)

    def test_get_stats_by_symbol(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        analyzer.record_trade("TRD002", "GOOGL", "buy", 50.0, 100.0)
        aapl_stats = analyzer.get_stats(symbol="AAPL")
        assert aapl_stats.total_trades == 1
        googl_stats = analyzer.get_stats(symbol="GOOGL")
        assert googl_stats.total_trades == 1

    def test_get_daily_breakdown(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        now = datetime.now(UTC)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0, timestamp=now)
        analyzer.record_trade(
            "TRD002", "AAPL", "sell", 50.0, 100.0, timestamp=now - timedelta(days=1)
        )
        breakdown = analyzer.get_daily_breakdown(days=3)
        assert len(breakdown) == 3
        assert breakdown[0]["turnover"] == pytest.approx(0.10, rel=0.01)
        assert breakdown[1]["turnover"] == pytest.approx(0.05, rel=0.01)

    def test_reset_stats(self) -> None:
        analyzer = TurnoverAnalyzer(TurnoverConfig(), portfolio_value=100000.0)
        analyzer.record_trade("TRD001", "AAPL", "buy", 100.0, 100.0)
        assert analyzer.get_stats().total_trades == 1
        analyzer.reset()
        assert analyzer.get_stats().total_trades == 0


class TestTurnoverStats:
    def test_stats_creation(self) -> None:
        stats = TurnoverStats(total_trades=100, total_volume=500000.0, daily_turnover=0.15,
                              weekly_turnover=0.75, total_turnover=2.5, estimated_cost=250.0,
                              alert_level="warning")
        assert stats.total_trades == 100
        assert stats.daily_turnover == 0.15
        assert stats.alert_level == "warning"
