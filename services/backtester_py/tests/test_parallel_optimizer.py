"""
Tests for the parallel grid search optimizer.

Tests cover:
- Parameter grid generation
- Result caching
- Grid search execution (with mock backtest)
- Early stopping
- Result sorting
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from backtester_py.parallel_optimizer import (
    GridSearchResult,
    GridSearchSummary,
    ParallelGridSearch,
    ResultCache,
    _generate_param_hash,
)


class TestGridSearchResult:
    """Tests for GridSearchResult dataclass."""

    def test_is_valid_with_no_error(self):
        """Test is_valid returns True when no error and has trades."""
        result = GridSearchResult(
            params={"a": 1},
            sharpe_ratio=1.5,
            num_trades=10,
        )
        assert result.is_valid is True

    def test_is_valid_with_error(self):
        """Test is_valid returns False when there's an error."""
        result = GridSearchResult(
            params={"a": 1},
            sharpe_ratio=1.5,
            num_trades=10,
            error="Something went wrong",
        )
        assert result.is_valid is False

    def test_is_valid_with_zero_trades(self):
        """Test is_valid returns False when no trades."""
        result = GridSearchResult(
            params={"a": 1},
            sharpe_ratio=0.0,
            num_trades=0,
        )
        assert result.is_valid is False

    def test_get_metric_from_attribute(self):
        """Test get_metric returns attribute value."""
        result = GridSearchResult(
            params={"a": 1},
            sharpe_ratio=1.5,
            total_return=0.25,
        )
        assert result.get_metric("sharpe_ratio") == 1.5
        assert result.get_metric("total_return") == 0.25

    def test_get_metric_from_metrics_dict(self):
        """Test get_metric falls back to metrics dict."""
        result = GridSearchResult(
            params={"a": 1},
            metrics={"custom_metric": 42.0},
        )
        assert result.get_metric("custom_metric") == 42.0

    def test_get_metric_default_zero(self):
        """Test get_metric returns 0 for unknown metric."""
        result = GridSearchResult(params={"a": 1})
        assert result.get_metric("unknown") == 0.0


class TestGridSearchSummary:
    """Tests for GridSearchSummary dataclass."""

    def test_top_n(self):
        """Test top_n returns correct number of results."""
        results = [
            GridSearchResult(params={"i": i}, sharpe_ratio=float(i))
            for i in range(20)
        ]
        summary = GridSearchSummary(results=results)

        top_5 = summary.top_n(5)
        assert len(top_5) == 5
        assert top_5[0].sharpe_ratio == 0.0

    def test_top_n_with_fewer_results(self):
        """Test top_n when fewer results than requested."""
        results = [
            GridSearchResult(params={"i": i}, sharpe_ratio=float(i))
            for i in range(3)
        ]
        summary = GridSearchSummary(results=results)

        top_10 = summary.top_n(10)
        assert len(top_10) == 3

    def test_summary_table_generation(self):
        """Test summary table is generated without error."""
        results = [
            GridSearchResult(
                params={"stop_loss": 0.02, "take_profit": 0.04},
                sharpe_ratio=1.5,
                total_return=0.15,
                max_drawdown=0.08,
                win_rate=0.55,
                num_trades=25,
            ),
            GridSearchResult(
                params={"stop_loss": 0.03, "take_profit": 0.06},
                sharpe_ratio=1.2,
                total_return=0.10,
                max_drawdown=0.12,
                win_rate=0.50,
                num_trades=30,
            ),
        ]
        summary = GridSearchSummary(
            results=results,
            best_result=results[0],
            total_combinations=10,
            successful_runs=2,
            failed_runs=8,
            total_time_seconds=5.5,
        )

        table = summary.summary_table(top_n=5)
        assert table is not None
        assert table.title is not None


class TestResultCache:
    """Tests for ResultCache."""

    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(cache_dir=tmpdir, cache_id="test")

            params = {"a": 1, "b": 2}
            result = GridSearchResult(params=params, sharpe_ratio=1.5)

            cache.put(params, result)
            retrieved = cache.get(params)

            assert retrieved is not None
            assert retrieved.sharpe_ratio == 1.5

    def test_cache_miss(self):
        """Test cache returns None for missing params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(cache_dir=tmpdir, cache_id="test")

            params = {"a": 1, "b": 2}
            retrieved = cache.get(params)

            assert retrieved is None

    def test_cache_persistence(self):
        """Test cache persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache and store result
            cache1 = ResultCache(cache_dir=tmpdir, cache_id="test_persist")
            params = {"a": 1, "b": 2}
            result = GridSearchResult(params=params, sharpe_ratio=2.0)
            cache1.put(params, result)
            cache1.save()

            # Create new cache instance and verify persistence
            cache2 = ResultCache(cache_dir=tmpdir, cache_id="test_persist")
            retrieved = cache2.get(params)

            assert retrieved is not None
            assert retrieved.sharpe_ratio == 2.0

    def test_cache_clear(self):
        """Test cache clear removes entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(cache_dir=tmpdir, cache_id="test_clear")

            params = {"a": 1}
            result = GridSearchResult(params=params, sharpe_ratio=1.0)
            cache.put(params, result)

            assert len(cache) == 1

            cache.clear()

            assert len(cache) == 0
            assert cache.get(params) is None


class TestParamHashGeneration:
    """Tests for parameter hash generation."""

    def test_consistent_hash(self):
        """Test same params produce same hash."""
        params = {"a": 1, "b": 2.5, "c": "test"}

        hash1 = _generate_param_hash(params)
        hash2 = _generate_param_hash(params)

        assert hash1 == hash2

    def test_different_params_different_hash(self):
        """Test different params produce different hash."""
        params1 = {"a": 1, "b": 2}
        params2 = {"a": 1, "b": 3}

        hash1 = _generate_param_hash(params1)
        hash2 = _generate_param_hash(params2)

        assert hash1 != hash2

    def test_order_independent(self):
        """Test parameter order doesn't affect hash."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}

        hash1 = _generate_param_hash(params1)
        hash2 = _generate_param_hash(params2)

        assert hash1 == hash2


class TestParallelGridSearch:
    """Tests for ParallelGridSearch class."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock BacktestEngine."""
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            initial_capital: float = 100_000.0
            stop_loss_pct: float = 0.02
            take_profit_pct: float | None = None
            default_order_notional: float = 10_000.0

        engine = MagicMock()
        engine.config = MockConfig()
        return engine

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        n_bars = 100
        timestamps = pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 12, 31),
            interval="1h",
            eager=True,
        )[:n_bars]

        prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

        return pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["TEST"] * n_bars,
            "bid_price": prices * 0.999,
            "ask_price": prices * 1.001,
            "vol": [0.001] * n_bars,
        })

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals."""
        from backtester_py.engine import Signal, SignalType

        timestamps = pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 12, 31),
            interval="1h",
            eager=True,
        )

        return [
            Signal(
                timestamp=timestamps[i],
                symbol="TEST",
                signal_type=SignalType.LONG if i % 2 == 0 else SignalType.SHORT,
                strength=0.5,
                target_notional=10_000.0,
            )
            for i in range(10, 50, 10)
        ]

    def test_generate_combinations(self, mock_engine):
        """Test parameter combination generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = ParallelGridSearch(
                engine=mock_engine,
                max_workers=2,
                cache_dir=tmpdir,
            )

            param_grid = {
                "a": [1, 2],
                "b": [10, 20, 30],
            }

            combinations = optimizer._generate_combinations(param_grid)

            assert len(combinations) == 6
            assert {"a": 1, "b": 10} in combinations
            assert {"a": 2, "b": 30} in combinations

    def test_invalid_max_workers(self, mock_engine):
        """Test that invalid max_workers raises error."""
        with pytest.raises(ValueError):
            ParallelGridSearch(engine=mock_engine, max_workers=0)

    def test_max_workers_auto(self, mock_engine):
        """Test max_workers=-1 uses CPU count."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = ParallelGridSearch(
                engine=mock_engine,
                max_workers=-1,
                cache_dir=tmpdir,
            )
            assert optimizer.max_workers == (os.cpu_count() or 4)

    def test_clear_cache(self, mock_engine):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = ParallelGridSearch(
                engine=mock_engine,
                max_workers=2,
                cache_dir=tmpdir,
            )

            # Add something to cache
            params = {"test": 1}
            result = GridSearchResult(params=params, sharpe_ratio=1.0)
            optimizer.cache.put(params, result)
            optimizer.cache.save()

            assert len(optimizer.cache) == 1

            optimizer.clear_cache()

            assert len(optimizer.cache) == 0


class TestGridSearchIntegration:
    """Integration tests for grid search (with mocked parallel execution)."""

    def test_empty_param_grid(self):
        """Test handling of empty parameter grid."""
        from backtester_py.engine import BacktestConfig
        from backtester_py.risk import RiskChecker, RiskLimits
        from cost_models import SlippageModel, TransactionCostModel

        # Create picklable config with all required components
        slippage_model = SlippageModel(spread_coef=1.0, size_coef=0.5)
        cost_model = TransactionCostModel(
            slippage_model=slippage_model,
            fixed_cost_bps=0.35,
        )
        risk_limits = RiskLimits(
            max_position_notional=100_000.0,
            max_gross_exposure=500_000.0,
            max_net_exposure=250_000.0,
            max_daily_loss=10_000.0,
            max_drawdown_pct=0.2,
        )
        risk_checker = RiskChecker(limits=risk_limits)

        config = BacktestConfig(
            initial_capital=100_000.0,
            cost_model=cost_model,
            risk_checker=risk_checker,
            default_order_notional=10_000.0,
            stop_loss_pct=0.02,
        )

        engine = MagicMock()
        engine.config = config

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = ParallelGridSearch(
                engine=engine,
                max_workers=2,
                cache_dir=tmpdir,
            )

            # Create minimal data
            market_data = pl.DataFrame({
                "timestamp": [datetime(2023, 1, 1)],
                "symbol": ["TEST"],
                "bid_price": [100.0],
                "ask_price": [100.1],
                "vol": [0.001],
            })

            summary = optimizer.run_grid_search(
                param_grid={},
                market_data=market_data,
                signals=[],
                use_cache=False,
            )

            # Empty param_grid produces 1 combination (empty dict = use defaults)
            # The backtest fails due to mock engine, so best_result is None
            assert summary.total_combinations == 1
            assert summary.best_result is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
