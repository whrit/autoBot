"""
Tests for shadow vs backtest behavior validation (T6.07).

TDD tests written before implementation.
"""

import polars as pl

from runner_py.validation import BehaviorValidator, ValidationConfig, ValidationResult


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.sharpe_tolerance == 0.10
        assert config.fill_rate_tolerance == 0.05
        assert config.slippage_tolerance_bps == 1.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ValidationConfig(
            sharpe_tolerance=0.15,
            fill_rate_tolerance=0.08,
            slippage_tolerance_bps=2.0,
        )
        assert config.sharpe_tolerance == 0.15
        assert config.fill_rate_tolerance == 0.08
        assert config.slippage_tolerance_bps == 2.0


class TestBehaviorValidator:
    """Tests for BehaviorValidator class."""

    def test_init_with_default_config(self) -> None:
        """Test validator initialization with default config."""
        validator = BehaviorValidator()
        assert validator.config is not None
        assert validator.config.sharpe_tolerance == 0.10

    def test_init_with_custom_config(self) -> None:
        """Test validator initialization with custom config."""
        config = ValidationConfig(sharpe_tolerance=0.20)
        validator = BehaviorValidator(config)
        assert validator.config.sharpe_tolerance == 0.20

    def test_validate_within_tolerance(
        self, sample_shadow_results: pl.DataFrame, sample_backtest_results: pl.DataFrame
    ) -> None:
        """Test validation passes when results are within tolerance."""
        config = ValidationConfig(sharpe_tolerance=0.50)  # Wide tolerance
        validator = BehaviorValidator(config)

        result = validator.validate(sample_shadow_results, sample_backtest_results)

        assert isinstance(result, ValidationResult)
        assert result.strategy_id == "strategy_001"
        assert isinstance(result.shadow_sharpe, float)
        assert isinstance(result.backtest_sharpe, float)
        # With wide tolerance, should pass
        assert result.within_tolerance is True
        assert result.passed is True

    def test_validate_outside_tolerance(
        self,
        divergent_shadow_results: pl.DataFrame,
        sample_backtest_results: pl.DataFrame,
    ) -> None:
        """Test validation fails when results exceed tolerance."""
        config = ValidationConfig(sharpe_tolerance=0.05)  # Tight tolerance
        validator = BehaviorValidator(config)

        result = validator.validate(divergent_shadow_results, sample_backtest_results)

        assert isinstance(result, ValidationResult)
        # Divergent results should fail tight tolerance
        assert result.sharpe_deviation > config.sharpe_tolerance
        assert result.within_tolerance is False
        assert result.passed is False
        assert len(result.issues) > 0

    def test_compare_sharpe_within_tolerance(self) -> None:
        """Test Sharpe ratio comparison within tolerance."""
        config = ValidationConfig(sharpe_tolerance=0.10)
        validator = BehaviorValidator(config)

        # 5% deviation should be within 10% tolerance
        shadow_sharpe = 1.5
        backtest_sharpe = 1.425  # 5% lower

        deviation, within = validator.compare_sharpe(shadow_sharpe, backtest_sharpe)

        assert abs(deviation - 0.05) < 0.01  # ~5% deviation
        assert within is True

    def test_compare_sharpe_outside_tolerance(self) -> None:
        """Test Sharpe ratio comparison outside tolerance."""
        config = ValidationConfig(sharpe_tolerance=0.05)
        validator = BehaviorValidator(config)

        # 15% deviation should exceed 5% tolerance
        shadow_sharpe = 2.0
        backtest_sharpe = 1.7  # 15% lower

        deviation, within = validator.compare_sharpe(shadow_sharpe, backtest_sharpe)

        assert deviation > 0.10  # More than 10% deviation
        assert within is False

    def test_compare_sharpe_zero_backtest(self) -> None:
        """Test Sharpe comparison when backtest Sharpe is zero."""
        validator = BehaviorValidator()

        deviation, within = validator.compare_sharpe(1.5, 0.0)

        # Should handle zero gracefully
        assert deviation == float("inf") or deviation > 1.0
        assert within is False

    def test_compare_fill_rates_within_tolerance(self) -> None:
        """Test fill rate comparison within tolerance."""
        config = ValidationConfig(fill_rate_tolerance=0.05)
        validator = BehaviorValidator(config)

        shadow_fill_rate = 0.95
        backtest_fill_rate = 0.93  # 2% lower

        deviation, within = validator.compare_fill_rates(shadow_fill_rate, backtest_fill_rate)

        assert deviation < 0.05
        assert within is True

    def test_compare_fill_rates_outside_tolerance(self) -> None:
        """Test fill rate comparison outside tolerance."""
        config = ValidationConfig(fill_rate_tolerance=0.02)
        validator = BehaviorValidator(config)

        shadow_fill_rate = 0.95
        backtest_fill_rate = 0.88  # 7% lower

        deviation, within = validator.compare_fill_rates(shadow_fill_rate, backtest_fill_rate)

        assert deviation > 0.05
        assert within is False

    def test_analyze_slippage_deviation(self) -> None:
        """Test slippage deviation analysis."""
        validator = BehaviorValidator()

        shadow_slippage = [1.0, 1.5, 2.0, 1.2, 1.8]
        backtest_slippage = [1.1, 1.4, 2.1, 1.3, 1.7]

        deviation_bps = validator.analyze_slippage_deviation(shadow_slippage, backtest_slippage)

        # Should be a small deviation in bps
        assert isinstance(deviation_bps, float)
        assert deviation_bps >= 0

    def test_analyze_slippage_deviation_empty_lists(self) -> None:
        """Test slippage deviation with empty lists."""
        validator = BehaviorValidator()

        deviation_bps = validator.analyze_slippage_deviation([], [])

        assert deviation_bps == 0.0

    def test_analyze_slippage_deviation_mismatched_lengths(self) -> None:
        """Test slippage deviation with mismatched list lengths."""
        validator = BehaviorValidator()

        shadow_slippage = [1.0, 1.5, 2.0]
        backtest_slippage = [1.1, 1.4]

        # Should handle gracefully, using shorter length
        deviation_bps = validator.analyze_slippage_deviation(shadow_slippage, backtest_slippage)
        assert isinstance(deviation_bps, float)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(
            strategy_id="test_strategy",
            shadow_sharpe=1.5,
            backtest_sharpe=1.4,
            sharpe_deviation=0.07,
            within_tolerance=True,
            fill_rate_shadow=0.95,
            fill_rate_backtest=0.93,
            avg_slippage_deviation_bps=0.5,
            issues=[],
            passed=True,
        )

        assert result.strategy_id == "test_strategy"
        assert result.shadow_sharpe == 1.5
        assert result.backtest_sharpe == 1.4
        assert result.sharpe_deviation == 0.07
        assert result.within_tolerance is True
        assert result.passed is True
        assert len(result.issues) == 0

    def test_result_with_issues(self) -> None:
        """Test validation result with issues."""
        result = ValidationResult(
            strategy_id="test_strategy",
            shadow_sharpe=2.0,
            backtest_sharpe=1.0,
            sharpe_deviation=0.50,
            within_tolerance=False,
            fill_rate_shadow=0.80,
            fill_rate_backtest=0.95,
            avg_slippage_deviation_bps=5.0,
            issues=["Sharpe ratio deviation exceeds tolerance", "Fill rate deviation too high"],
            passed=False,
        )

        assert result.passed is False
        assert len(result.issues) == 2
        assert "Sharpe ratio" in result.issues[0]


class TestValidationEdgeCases:
    """Edge case tests for validation."""

    def test_validate_empty_dataframes(self) -> None:
        """Test validation with empty DataFrames."""
        validator = BehaviorValidator()

        empty_df = pl.DataFrame({
            "timestamp": [],
            "equity": [],
            "returns": [],
            "symbol": [],
            "strategy_id": [],
            "fill_count": [],
            "slippage_bps": [],
        })

        result = validator.validate(empty_df, empty_df)

        assert isinstance(result, ValidationResult)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_single_row(self) -> None:
        """Test validation with single row DataFrames."""
        validator = BehaviorValidator()

        single_row = pl.DataFrame({
            "timestamp": ["2024-01-15 09:30:00"],
            "equity": [100000.0],
            "returns": [0.001],
            "symbol": ["SPY"],
            "strategy_id": ["test"],
            "fill_count": [1],
            "slippage_bps": [1.0],
        })

        result = validator.validate(single_row, single_row)

        assert isinstance(result, ValidationResult)
        # Should handle edge case gracefully

    def test_validate_negative_sharpe(self) -> None:
        """Test validation with negative Sharpe ratios."""
        validator = BehaviorValidator()

        deviation, within = validator.compare_sharpe(-0.5, -0.6)

        # Should handle negative values correctly
        assert isinstance(deviation, float)
        assert isinstance(within, bool)
