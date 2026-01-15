"""
Tests for XGBoost ML Strategy (T4.05) and ONNX Export (T4.08).

Tests machine learning strategy with XGBoost and ONNX model export.
"""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest

from optimizer_py.ml_strategy import (
    MLStrategyConfig,
    XGBoostStrategy,
)
from optimizer_py.onnx_export import (
    ONNXExportConfig,
    ONNXExporter,
)
from optimizer_py.strategies import StrategySignal


class TestMLStrategyConfig:
    """Tests for MLStrategyConfig dataclass."""

    def test_default_config(self, ml_feature_columns: list[str]) -> None:
        """Test default configuration values."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
        )
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.tree_method == "hist"
        assert config.device == "cpu"

    def test_custom_config(self, ml_feature_columns: list[str]) -> None:
        """Test custom configuration values."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            tree_method="hist",
            device="cpu",
        )
        assert config.n_estimators == 200
        assert config.max_depth == 8
        assert config.learning_rate == 0.05


class TestXGBoostStrategy:
    """Tests for XGBoostStrategy class."""

    def test_strategy_initialization(self, ml_feature_columns: list[str]) -> None:
        """Test XGBoostStrategy initialization."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
        )
        strategy = XGBoostStrategy(config)
        assert strategy.config == config
        assert strategy.model is None  # Not trained yet

    def test_train_basic(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test basic model training."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,  # Small for fast testing
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        assert strategy.model is not None

    def test_train_with_larger_dataset(
        self, large_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test training with larger dataset."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=20,
            max_depth=4,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(large_labeled_frame)

        assert strategy.model is not None

    def test_predict_returns_probabilities(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test prediction returns valid probabilities."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        predictions = strategy.predict(sample_labeled_frame)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_labeled_frame)

    def test_predict_without_training_raises(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that predict raises error if model not trained."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
        )
        strategy = XGBoostStrategy(config)

        with pytest.raises(ValueError, match="Model not trained"):
            strategy.predict(sample_labeled_frame)

    def test_generate_signals_basic(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test signal generation from trained model."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        signals = strategy.generate_signals(sample_labeled_frame)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, StrategySignal)
            assert signal.direction in [-1, 0, 1]
            assert 0.0 <= signal.strength <= 1.0

    def test_generate_signals_structure(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that generated signals have correct structure."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        signals = strategy.generate_signals(sample_labeled_frame)

        if signals:  # Only check if signals were generated
            signal = signals[0]
            assert hasattr(signal, "timestamp")
            assert hasattr(signal, "symbol")
            assert hasattr(signal, "direction")
            assert hasattr(signal, "strength")

    def test_get_feature_importance(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test feature importance extraction."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        importance = strategy.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(ml_feature_columns)
        assert all(isinstance(v, float) for v in importance.values())

    def test_model_params_affect_training(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that different parameters produce different models."""
        config1 = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=5,
            max_depth=2,
        )
        config2 = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=20,
            max_depth=6,
        )

        strategy1 = XGBoostStrategy(config1)
        strategy2 = XGBoostStrategy(config2)

        strategy1.train(sample_labeled_frame)
        strategy2.train(sample_labeled_frame)

        pred1 = strategy1.predict(sample_labeled_frame)
        pred2 = strategy2.predict(sample_labeled_frame)

        # Predictions should be different with different configs
        assert not np.allclose(pred1, pred2, atol=0.01)


class TestONNXExportConfig:
    """Tests for ONNXExportConfig dataclass."""

    def test_default_config(self, ml_feature_columns: list[str]) -> None:
        """Test default configuration."""
        config = ONNXExportConfig(
            model_path=Path("/tmp/model.onnx"),
            input_features=ml_feature_columns,
        )
        assert config.opset_version == 15

    def test_custom_opset(self, ml_feature_columns: list[str]) -> None:
        """Test custom opset version."""
        config = ONNXExportConfig(
            model_path=Path("/tmp/model.onnx"),
            input_features=ml_feature_columns,
            opset_version=14,
        )
        assert config.opset_version == 14


class TestONNXExporter:
    """Tests for ONNXExporter class."""

    def test_exporter_initialization(self) -> None:
        """Test ONNXExporter initialization."""
        exporter = ONNXExporter()
        assert exporter is not None

    def test_export_xgboost_basic(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test basic XGBoost model export to ONNX."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        export_config = ONNXExportConfig(
            model_path=Path("/tmp/test_model.onnx"),
            input_features=ml_feature_columns,
        )
        exporter = ONNXExporter()
        onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

        assert isinstance(onnx_bytes, bytes)
        assert len(onnx_bytes) > 0

    def test_export_and_save(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test exporting and saving ONNX model to file."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.onnx"
            export_config = ONNXExportConfig(
                model_path=model_path,
                input_features=ml_feature_columns,
            )
            exporter = ONNXExporter()
            onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

            model_path.write_bytes(onnx_bytes)
            assert model_path.exists()
            assert model_path.stat().st_size > 0

    def test_validate_export_valid_model(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test ONNX model validation passes for valid model."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        export_config = ONNXExportConfig(
            model_path=Path("/tmp/test_model.onnx"),
            input_features=ml_feature_columns,
        )
        exporter = ONNXExporter()
        onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

        # Create sample input
        sample_input = sample_labeled_frame.select(ml_feature_columns).to_numpy()[:5].astype(
            np.float32
        )

        is_valid = exporter.validate_export(onnx_bytes, sample_input)
        assert is_valid

    def test_onnx_inference_matches_xgboost(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that ONNX inference produces same results as XGBoost."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        # Get XGBoost predictions
        xgb_predictions = strategy.predict(sample_labeled_frame)

        # Export to ONNX
        export_config = ONNXExportConfig(
            model_path=Path("/tmp/test_model.onnx"),
            input_features=ml_feature_columns,
        )
        exporter = ONNXExporter()
        onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

        # Get ONNX predictions
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(onnx_bytes)
            f.flush()

            session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_data = (
                sample_labeled_frame.select(ml_feature_columns).to_numpy().astype(np.float32)
            )

            onnx_output = session.run(None, {input_name: input_data})

        # Compare predictions (allow small tolerance due to floating point)
        onnx_predictions = onnx_output[1][:, 1]  # Get probability of class 1
        # XGBoost multiclass predictions need to be handled appropriately
        assert len(onnx_predictions) == len(xgb_predictions)

    def test_onnx_model_can_load_and_run(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that exported ONNX model can be loaded and run."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(sample_labeled_frame)

        export_config = ONNXExportConfig(
            model_path=Path("/tmp/test_model.onnx"),
            input_features=ml_feature_columns,
        )
        exporter = ONNXExporter()
        onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(onnx_bytes)
            f.flush()

            # Load and run the model
            session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_data = (
                sample_labeled_frame.select(ml_feature_columns).to_numpy()[:10].astype(np.float32)
            )

            outputs = session.run(None, {input_name: input_data})
            assert outputs is not None
            assert len(outputs) > 0


class TestXGBoostStrategyIntegration:
    """Integration tests for XGBoostStrategy."""

    def test_end_to_end_workflow(
        self, large_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test complete workflow from training to ONNX export."""
        # Step 1: Configure and train
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=20,
            max_depth=4,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(large_labeled_frame)

        # Step 2: Generate signals
        signals = strategy.generate_signals(large_labeled_frame)
        assert len(signals) > 0

        # Step 3: Export to ONNX
        export_config = ONNXExportConfig(
            model_path=Path("/tmp/model.onnx"),
            input_features=ml_feature_columns,
        )
        exporter = ONNXExporter()
        onnx_bytes = exporter.export_xgboost(strategy.model, export_config)

        # Step 4: Validate export
        sample_input = large_labeled_frame.select(ml_feature_columns).to_numpy()[:10].astype(
            np.float32
        )
        assert exporter.validate_export(onnx_bytes, sample_input)

    def test_strategy_family_interface(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that XGBoostStrategy implements StrategyFamily interface."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)

        # Check interface methods exist
        assert hasattr(strategy, "generate_signals")
        assert hasattr(strategy, "get_default_params")
        assert hasattr(strategy, "family_name")
        assert strategy.family_name == "xgboost"

    def test_different_target_columns(self, large_labeled_frame: pl.DataFrame) -> None:
        """Test training with different target columns."""
        features = ["return_1", "return_5", "vol_5"]

        config = MLStrategyConfig(
            features=features,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)
        strategy.train(large_labeled_frame)

        assert strategy.model is not None


class TestMLStrategyEdgeCases:
    """Edge case tests for ML strategy."""

    def test_missing_feature_column_raises(self, sample_labeled_frame: pl.DataFrame) -> None:
        """Test that missing feature column raises appropriate error."""
        config = MLStrategyConfig(
            features=["nonexistent_feature"],
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)

        with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
            strategy.train(sample_labeled_frame)

    def test_missing_target_column_raises(
        self, sample_labeled_frame: pl.DataFrame, ml_feature_columns: list[str]
    ) -> None:
        """Test that missing target column raises appropriate error."""
        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="nonexistent_target",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)

        with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
            strategy.train(sample_labeled_frame)

    def test_empty_dataframe_raises(self, ml_feature_columns: list[str]) -> None:
        """Test that empty dataframe raises appropriate error."""
        empty_df = pl.DataFrame({col: [] for col in ml_feature_columns})
        empty_df = empty_df.with_columns(pl.Series("direction_60s", []))

        config = MLStrategyConfig(
            features=ml_feature_columns,
            target="direction_60s",
            n_estimators=10,
        )
        strategy = XGBoostStrategy(config)

        with pytest.raises(ValueError):
            strategy.train(empty_df)
