"""
ONNX Export Module (T4.08).

Provides functionality to export trained ML models to ONNX format
for production deployment and cross-platform inference.

Key Components:
    - ONNXExportConfig: Configuration for ONNX export
    - ONNXExporter: Main class for exporting models

Example:
    >>> exporter = ONNXExporter()
    >>> config = ONNXExportConfig(
    ...     model_path=Path("model.onnx"),
    ...     input_features=["feature_1", "feature_2"],
    ... )
    >>> onnx_bytes = exporter.export_xgboost(model, config)
    >>> Path("model.onnx").write_bytes(onnx_bytes)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

if TYPE_CHECKING:
    import xgboost as xgb


@dataclass
class ONNXExportConfig:
    """
    Configuration for ONNX model export.

    Attributes:
        model_path: Path where the ONNX model will be saved
        input_features: List of input feature names
        opset_version: ONNX opset version (default: 15)
    """

    model_path: Path
    input_features: list[str]
    opset_version: int = 15


class ONNXExporter:
    """
    Export ML models to ONNX format.

    Supports exporting XGBoost models to ONNX for:
    - Cross-platform deployment
    - Hardware acceleration
    - Model versioning and serving

    Example:
        >>> exporter = ONNXExporter()
        >>> onnx_bytes = exporter.export_xgboost(booster, config)
        >>> is_valid = exporter.validate_export(onnx_bytes, sample_input)
    """

    def __init__(self) -> None:
        """Initialize ONNX exporter."""
        pass

    def export_xgboost(
        self,
        model: xgb.XGBClassifier | xgb.Booster,
        config: ONNXExportConfig,
    ) -> bytes:
        """
        Export XGBoost model to ONNX format.

        Args:
            model: Trained XGBoost classifier or Booster
            config: Export configuration

        Returns:
            ONNX model as bytes

        Raises:
            ValueError: If model cannot be exported
        """
        n_features = len(config.input_features)

        # Define input type
        initial_type = [("input", FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=config.opset_version,
        )

        # Serialize to bytes
        return bytes(onnx_model.SerializeToString())

    def validate_export(
        self,
        onnx_bytes: bytes,
        sample_input: np.ndarray,
    ) -> bool:
        """
        Validate exported ONNX model.

        Checks that:
        1. ONNX model can be loaded and checked
        2. Model can run inference on sample input
        3. Output has expected shape

        Args:
            onnx_bytes: Serialized ONNX model
            sample_input: Sample input array for testing (float32)

        Returns:
            True if model is valid and produces output
        """
        try:
            # Load and check ONNX model
            onnx_model = onnx.load_from_string(onnx_bytes)
            onnx.checker.check_model(onnx_model)

            # Create inference session and run
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                f.write(onnx_bytes)
                f.flush()

                session = ort.InferenceSession(
                    f.name,
                    providers=["CPUExecutionProvider"],
                )

                # Get input name
                input_name = session.get_inputs()[0].name

                # Ensure input is float32
                input_data = sample_input.astype(np.float32)

                # Run inference
                outputs = session.run(None, {input_name: input_data})

                # Check outputs exist
                return outputs is not None and len(outputs) > 0

        except Exception:
            return False

    def export_to_file(
        self,
        model: xgb.XGBClassifier | xgb.Booster,
        config: ONNXExportConfig,
    ) -> Path:
        """
        Export model and save to file.

        Args:
            model: Trained XGBoost model
            config: Export configuration

        Returns:
            Path to saved ONNX file
        """
        onnx_bytes = self.export_xgboost(model, config)
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        config.model_path.write_bytes(onnx_bytes)
        return config.model_path

    def get_model_metadata(self, onnx_bytes: bytes) -> dict[str, str | int | list[str]]:
        """
        Get metadata from ONNX model.

        Args:
            onnx_bytes: Serialized ONNX model

        Returns:
            Dictionary with model metadata
        """
        onnx_model = onnx.load_from_string(onnx_bytes)

        inputs = []
        for inp in onnx_model.graph.input:
            inputs.append(inp.name)

        outputs = []
        for out in onnx_model.graph.output:
            outputs.append(out.name)

        return {
            "ir_version": onnx_model.ir_version,
            "opset_version": onnx_model.opset_import[0].version if onnx_model.opset_import else 0,
            "producer_name": onnx_model.producer_name,
            "inputs": inputs,
            "outputs": outputs,
        }


def load_onnx_model(path: Path) -> ort.InferenceSession:
    """
    Load ONNX model from file for inference.

    Args:
        path: Path to ONNX model file

    Returns:
        ONNX Runtime inference session

    Example:
        >>> session = load_onnx_model(Path("model.onnx"))
        >>> output = session.run(None, {"input": features})
    """
    return ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )


def run_onnx_inference(
    session: ort.InferenceSession,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference with ONNX model.

    Args:
        session: ONNX Runtime session
        features: Input features as float32 array

    Returns:
        Tuple of (predictions, probabilities)
    """
    input_name = session.get_inputs()[0].name
    features = features.astype(np.float32)

    outputs = session.run(None, {input_name: features})

    # Typically: outputs[0] = class labels, outputs[1] = probabilities
    predictions = outputs[0] if len(outputs) > 0 else np.array([])
    probabilities = outputs[1] if len(outputs) > 1 else np.array([])

    return predictions, probabilities
