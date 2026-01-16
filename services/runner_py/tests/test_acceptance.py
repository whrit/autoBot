"""
Tests for Phase 1 acceptance testing (T6.12).

TDD tests written before implementation.
"""

import pytest

from runner_py.acceptance import AcceptanceCriteria, AcceptanceResult, AcceptanceTester


class TestAcceptanceCriteria:
    """Tests for AcceptanceCriteria dataclass."""

    def test_default_criteria(self) -> None:
        """Test default acceptance criteria match PRD."""
        criteria = AcceptanceCriteria()

        # From PRD requirements
        assert criteria.historical_data_days == 30
        assert criteria.realtime_latency_ms == 500.0
        assert criteria.feature_zero_leakage is True
        assert criteria.backtest_sharpe_tolerance == 0.05
        assert criteria.shadow_match_tolerance == 0.10
        assert criteria.paper_slippage_max_ratio == 2.0

    def test_custom_criteria(self) -> None:
        """Test custom acceptance criteria."""
        criteria = AcceptanceCriteria(
            historical_data_days=60,
            realtime_latency_ms=250.0,
            feature_zero_leakage=True,
            backtest_sharpe_tolerance=0.03,
            shadow_match_tolerance=0.05,
            paper_slippage_max_ratio=1.5,
        )

        assert criteria.historical_data_days == 60
        assert criteria.realtime_latency_ms == 250.0
        assert criteria.shadow_match_tolerance == 0.05


class TestAcceptanceResult:
    """Tests for AcceptanceResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating an acceptance result."""
        result = AcceptanceResult(
            criterion="Historical Data Ingestion",
            expected=">=30 days",
            actual="45 days",
            passed=True,
            details="SPY: 45 days, QQQ: 42 days",
        )

        assert result.criterion == "Historical Data Ingestion"
        assert result.expected == ">=30 days"
        assert result.actual == "45 days"
        assert result.passed is True
        assert "SPY" in result.details

    def test_failed_result(self) -> None:
        """Test creating a failed acceptance result."""
        result = AcceptanceResult(
            criterion="Real-time Latency",
            expected="<500ms",
            actual="750ms",
            passed=False,
            details="Average latency exceeded threshold",
        )

        assert result.passed is False
        assert "exceeded" in result.details


class TestAcceptanceTester:
    """Tests for AcceptanceTester class."""

    def test_init_with_default_criteria(self) -> None:
        """Test tester initialization with default criteria."""
        tester = AcceptanceTester()
        assert tester.criteria is not None
        assert tester.criteria.historical_data_days == 30

    def test_init_with_custom_criteria(self) -> None:
        """Test tester initialization with custom criteria."""
        criteria = AcceptanceCriteria(historical_data_days=60)
        tester = AcceptanceTester(criteria)
        assert tester.criteria.historical_data_days == 60

    def test_historical_data_ingestion(self) -> None:
        """Test historical data ingestion acceptance."""
        tester = AcceptanceTester()
        result = tester.test_historical_data_ingestion()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Historical Data Ingestion"
        assert "30" in result.expected
        assert isinstance(result.passed, bool)

    def test_realtime_latency(self) -> None:
        """Test real-time latency acceptance."""
        tester = AcceptanceTester()
        result = tester.test_realtime_latency()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Real-time Latency"
        assert "500" in result.expected or "ms" in result.expected.lower()
        assert isinstance(result.passed, bool)

    def test_feature_pipeline_no_leakage(self) -> None:
        """Test feature pipeline leakage acceptance."""
        tester = AcceptanceTester()
        result = tester.test_feature_pipeline_no_leakage()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Feature Pipeline No Leakage"
        assert "leakage" in result.expected.lower() or "zero" in result.expected.lower()
        assert isinstance(result.passed, bool)

    def test_backtest_realism(self) -> None:
        """Test backtest realism acceptance."""
        tester = AcceptanceTester()
        result = tester.test_backtest_realism()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Backtest Realism"
        assert isinstance(result.passed, bool)

    def test_shadow_matches_backtest(self) -> None:
        """Test shadow vs backtest match acceptance."""
        tester = AcceptanceTester()
        result = tester.test_shadow_matches_backtest()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Shadow Matches Backtest"
        assert "10" in result.expected or "%" in result.expected
        assert isinstance(result.passed, bool)

    def test_paper_slippage(self) -> None:
        """Test paper slippage acceptance."""
        tester = AcceptanceTester()
        result = tester.test_paper_slippage()

        assert isinstance(result, AcceptanceResult)
        assert result.criterion == "Paper Slippage"
        assert "2x" in result.expected or "2" in result.expected
        assert isinstance(result.passed, bool)

    def test_run_all_tests(self) -> None:
        """Test running all acceptance tests."""
        tester = AcceptanceTester()
        results = tester.run_all_tests()

        assert isinstance(results, list)
        assert len(results) >= 6  # At least 6 acceptance criteria
        for result in results:
            assert isinstance(result, AcceptanceResult)

    def test_generate_report(self) -> None:
        """Test acceptance report generation."""
        tester = AcceptanceTester()
        results = [
            AcceptanceResult(
                criterion="Test Criterion 1",
                expected=">=30",
                actual="45",
                passed=True,
                details="Test passed",
            ),
            AcceptanceResult(
                criterion="Test Criterion 2",
                expected="<500",
                actual="750",
                passed=False,
                details="Test failed",
            ),
        ]

        report = tester.generate_report(results)

        assert isinstance(report, str)
        assert "Test Criterion 1" in report
        assert "Test Criterion 2" in report
        assert "PASS" in report.upper() or "pass" in report.lower()
        assert "FAIL" in report.upper() or "fail" in report.lower()

    def test_generate_report_all_pass(self) -> None:
        """Test report generation when all tests pass."""
        tester = AcceptanceTester()
        results = [
            AcceptanceResult(
                criterion="Test 1",
                expected=">=30",
                actual="45",
                passed=True,
                details="OK",
            ),
            AcceptanceResult(
                criterion="Test 2",
                expected="<500",
                actual="250",
                passed=True,
                details="OK",
            ),
        ]

        report = tester.generate_report(results)

        assert isinstance(report, str)
        # Report should indicate overall success
        assert "2/2" in report or "100" in report or "all" in report.lower()

    def test_generate_report_all_fail(self) -> None:
        """Test report generation when all tests fail."""
        tester = AcceptanceTester()
        results = [
            AcceptanceResult(
                criterion="Test 1",
                expected=">=30",
                actual="15",
                passed=False,
                details="Insufficient",
            ),
            AcceptanceResult(
                criterion="Test 2",
                expected="<500",
                actual="750",
                passed=False,
                details="Too slow",
            ),
        ]

        report = tester.generate_report(results)

        assert isinstance(report, str)
        # Report should indicate overall failure
        assert "0/2" in report or "fail" in report.lower()


class TestAcceptanceIntegration:
    """Integration tests for acceptance testing."""

    def test_all_criteria_covered(self) -> None:
        """Test that all PRD criteria are covered."""
        tester = AcceptanceTester()
        results = tester.run_all_tests()

        # Extract criterion names
        criteria_names = [r.criterion for r in results]

        # All major criteria from PRD should be covered
        expected_criteria = [
            "Historical Data",
            "Latency",
            "Leakage",
            "Backtest",
            "Shadow",
            "Slippage",
        ]

        for expected in expected_criteria:
            found = any(expected.lower() in c.lower() for c in criteria_names)
            assert found, f"Missing criterion: {expected}"

    def test_results_have_details(self) -> None:
        """Test that all results have meaningful details."""
        tester = AcceptanceTester()
        results = tester.run_all_tests()

        for result in results:
            assert result.details, f"Missing details for: {result.criterion}"
            assert len(result.details) > 0

    def test_expected_values_match_prd(self) -> None:
        """Test that expected values match PRD requirements."""
        criteria = AcceptanceCriteria()

        # PRD requirements
        assert criteria.historical_data_days >= 30
        assert criteria.realtime_latency_ms <= 500.0
        assert criteria.shadow_match_tolerance <= 0.10  # 10%
        assert criteria.paper_slippage_max_ratio <= 2.0
