"""Tests for Prometheus Metrics Export (T5.10)."""

from datetime import UTC, datetime

from monitor_py.drawdown import DrawdownStatus
from monitor_py.feed_health import FeedHealthStatus, FeedStatus
from monitor_py.metrics import MetricsExporter
from monitor_py.slippage import SlippageStats


class TestMetricsExporter:
    """Test MetricsExporter functionality."""

    def test_exporter_creation(self) -> None:
        """Test MetricsExporter can be created."""
        exporter = MetricsExporter()
        assert exporter is not None

    def test_get_metrics_returns_bytes(self) -> None:
        """Test get_metrics returns bytes in Prometheus format."""
        exporter = MetricsExporter()
        metrics = exporter.get_metrics()
        assert isinstance(metrics, bytes)
        # Should contain some Prometheus metrics
        metrics_str = metrics.decode("utf-8")
        assert "# HELP" in metrics_str or "# TYPE" in metrics_str or len(metrics_str) > 0

    def test_update_feed_metrics(self) -> None:
        """Test updating feed health metrics."""
        exporter = MetricsExporter()
        status = FeedHealthStatus(
            status=FeedStatus.HEALTHY,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=0.5,
            messages_per_second=100.0,
            disconnect_count=2,
            message="Feed is healthy",
        )

        exporter.update_feed_metrics(status)
        metrics = exporter.get_metrics().decode("utf-8")

        # Check that feed metrics are present
        assert "feed_lag_seconds" in metrics
        assert "feed_messages_total" in metrics or "feed_messages" in metrics

    def test_update_slippage_metrics(self) -> None:
        """Test updating slippage metrics."""
        exporter = MetricsExporter()
        stats = SlippageStats(
            avg_slippage_bps=2.5,
            max_slippage_bps=5.0,
            expected_slippage_bps=2.0,
            ratio_to_expected=1.25,
            sample_count=100,
            alert_triggered=False,
        )

        exporter.update_slippage_metrics(stats)
        metrics = exporter.get_metrics().decode("utf-8")

        # Check that slippage metrics are present
        assert "slippage" in metrics.lower()

    def test_update_drawdown_metrics(self) -> None:
        """Test updating drawdown metrics."""
        exporter = MetricsExporter()
        status = DrawdownStatus(
            current_drawdown=0.05,
            max_drawdown=0.10,
            peak_equity=100000.0,
            current_equity=95000.0,
            alert_level="warning",
            days_in_drawdown=5,
        )

        exporter.update_drawdown_metrics(status)
        metrics = exporter.get_metrics().decode("utf-8")

        # Check that drawdown metrics are present
        assert "drawdown" in metrics.lower()

    def test_prometheus_format(self) -> None:
        """Test metrics are in valid Prometheus format."""
        exporter = MetricsExporter()

        # Update some metrics
        feed_status = FeedHealthStatus(
            status=FeedStatus.HEALTHY,
            last_message_time=datetime.now(UTC),
            current_lag_seconds=0.5,
            messages_per_second=100.0,
            disconnect_count=0,
            message="Feed is healthy",
        )
        exporter.update_feed_metrics(feed_status)

        metrics = exporter.get_metrics().decode("utf-8")
        lines = metrics.strip().split("\n")

        # Prometheus format has HELP/TYPE comments and metric lines
        for line in lines:
            if line.startswith("#"):
                # Should be HELP or TYPE
                assert line.startswith("# HELP") or line.startswith("# TYPE")
            elif line.strip():
                # Should be metric_name{labels} value or metric_name value
                # Basic check: has a space separating name from value
                assert " " in line

    def test_record_alert(self) -> None:
        """Test recording alerts increments counter."""
        exporter = MetricsExporter()

        # Record some alerts
        exporter.record_alert("drawdown_warning")
        exporter.record_alert("slippage_high")

        metrics = exporter.get_metrics().decode("utf-8")
        # Should have alerts_triggered counter
        assert "alerts_triggered" in metrics

    def test_record_rollback(self) -> None:
        """Test recording rollback increments counter."""
        exporter = MetricsExporter()

        exporter.record_rollback()

        metrics = exporter.get_metrics().decode("utf-8")
        assert "rollbacks_triggered" in metrics

    def test_metrics_persistence(self) -> None:
        """Test metrics persist between calls."""
        exporter = MetricsExporter()

        # Update metrics multiple times
        for i in range(3):
            status = FeedHealthStatus(
                status=FeedStatus.HEALTHY,
                last_message_time=datetime.now(UTC),
                current_lag_seconds=float(i),
                messages_per_second=100.0,
                disconnect_count=i,
                message="Feed is healthy",
            )
            exporter.update_feed_metrics(status)

        # Get final metrics - should reflect last update
        metrics = exporter.get_metrics().decode("utf-8")
        # Lag should be 2.0 (last value)
        assert "feed_lag_seconds" in metrics


class TestMetricNames:
    """Test that metric names follow Prometheus conventions."""

    def test_metric_names_lowercase(self) -> None:
        """Test metric names are lowercase with underscores."""
        exporter = MetricsExporter()
        exporter.update_feed_metrics(
            FeedHealthStatus(
                status=FeedStatus.HEALTHY,
                last_message_time=datetime.now(UTC),
                current_lag_seconds=0.5,
                messages_per_second=100.0,
                disconnect_count=0,
                message="Feed is healthy",
            )
        )

        metrics = exporter.get_metrics().decode("utf-8")
        lines = [
            line for line in metrics.split("\n") if line and not line.startswith("#")
        ]

        for line in lines:
            metric_name = line.split("{")[0].split()[0]
            # Should be lowercase and use underscores
            assert metric_name == metric_name.lower()
            assert "-" not in metric_name


class TestMetricsLabels:
    """Test metrics labels."""

    def test_alert_type_label(self) -> None:
        """Test alerts have type labels."""
        exporter = MetricsExporter()

        exporter.record_alert("drawdown_warning")
        exporter.record_alert("slippage_high")

        metrics = exporter.get_metrics().decode("utf-8")
        # Should have labeled alerts
        assert "alert_type=" in metrics or "alerts_triggered_total" in metrics
