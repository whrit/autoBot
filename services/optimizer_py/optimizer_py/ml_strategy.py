"""
XGBoost ML Strategy Implementation (T4.05).

Provides machine learning-based trading strategy using XGBoost
for signal generation based on trained classification models.

Key Components:
    - MLStrategyConfig: Configuration for ML strategy
    - XGBoostStrategy: XGBoost-based trading strategy

Example:
    >>> config = MLStrategyConfig(
    ...     features=["rsi_14", "macd", "volume_imbalance"],
    ...     target="direction_60s",
    ...     n_estimators=100,
    ... )
    >>> strategy = XGBoostStrategy(config)
    >>> strategy.train(labeled_data)
    >>> signals = strategy.generate_signals(decision_frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl
import xgboost as xgb

from optimizer_py.strategies import StrategyFamily, StrategySignal


@dataclass
class MLStrategyConfig:
    """
    Configuration for XGBoost ML Strategy.

    Attributes:
        features: List of feature column names to use for prediction
        target: Target column name for training (e.g., "direction_60s")
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        tree_method: XGBoost tree method ("hist" or "gpu_hist")
        device: Training device ("cpu" or "cuda")
        random_state: Random seed for reproducibility
        signal_threshold: Probability threshold for generating signals
    """

    features: list[str]
    target: str
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    tree_method: str = "hist"
    device: str = "cpu"
    random_state: int = 42
    signal_threshold: float = 0.4
    extra_params: dict[str, Any] = field(default_factory=dict)


class XGBoostStrategy(StrategyFamily):
    """
    XGBoost-based machine learning trading strategy.

    Trains a gradient boosting classifier on historical data to predict
    trading direction, then generates signals based on model predictions.

    The model predicts probabilities for three classes:
        - Class 0 (-1): Short signal
        - Class 1 (0): No signal / flat
        - Class 2 (+1): Long signal

    Example:
        >>> config = MLStrategyConfig(
        ...     features=["rsi_14", "macd", "vol_5"],
        ...     target="direction_60s",
        ...     n_estimators=50,
        ... )
        >>> strategy = XGBoostStrategy(config)
        >>> strategy.train(labeled_frame)
        >>> signals = strategy.generate_signals(decision_frame)
    """

    family_name: ClassVar[str] = "xgboost"

    def __init__(self, config: MLStrategyConfig) -> None:
        """
        Initialize XGBoost strategy.

        Args:
            config: ML strategy configuration
        """
        self.config = config
        self.model: xgb.XGBClassifier | None = None
        self._feature_importance: dict[str, float] = {}

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        """Get default parameters for XGBoost strategy."""
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "device": "cpu",
        }

    def train(self, train_data: pl.DataFrame) -> None:
        """
        Train the XGBoost model on labeled data.

        Args:
            train_data: DataFrame containing feature columns and target column

        Raises:
            ValueError: If training data is empty or missing columns
            KeyError: If feature or target columns are missing
        """
        if len(train_data) == 0:
            raise ValueError("Training data is empty")

        # Extract features and target
        features_arr = train_data.select(self.config.features).to_numpy()
        targets = train_data.select(self.config.target).to_numpy().ravel()

        # Validate data
        if features_arr.shape[0] == 0:
            raise ValueError("No samples in training data")

        # Map target values to class indices (handles -1, 0, 1 -> 0, 1, 2)
        targets_mapped = targets.copy()
        targets_mapped[targets == -1] = 0
        targets_mapped[targets == 0] = 1
        targets_mapped[targets == 1] = 2

        # Configure XGBoost
        params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "tree_method": self.config.tree_method,
            "device": self.config.device,
            "random_state": self.config.random_state,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        params.update(self.config.extra_params)

        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(features_arr, targets_mapped)

        # Store feature importance (convert to Python float)
        if hasattr(self.model, "feature_importances_"):
            self._feature_importance = {
                feat: float(imp)
                for feat, imp in zip(
                    self.config.features, self.model.feature_importances_, strict=True
                )
            }

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        """
        Generate predictions from feature data.

        Args:
            features: DataFrame containing feature columns

        Returns:
            Array of predicted class indices (0, 1, 2)

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features_arr = features.select(self.config.features).to_numpy()
        result: np.ndarray = self.model.predict(features_arr)
        return result

    def predict_proba(self, features: pl.DataFrame) -> np.ndarray:
        """
        Generate probability predictions from feature data.

        Args:
            features: DataFrame containing feature columns

        Returns:
            Array of shape (n_samples, 3) with class probabilities

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features_arr = features.select(self.config.features).to_numpy()
        return self.model.predict_proba(features_arr)

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """
        Generate trading signals from a decision frame.

        Uses the trained model to predict direction probabilities,
        then generates signals for predictions above the threshold.

        Args:
            decision_frame: DataFrame with feature columns, timestamp, and symbol

        Returns:
            List of StrategySignal objects

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions
        probabilities = self.predict_proba(decision_frame)

        # Get timestamps and symbols
        timestamps = decision_frame["timestamp"].to_list()
        symbols = decision_frame["symbol"].to_list()

        signals: list[StrategySignal] = []
        threshold = self.config.signal_threshold

        for i in range(len(probabilities)):
            # probabilities: [P(short), P(flat), P(long)]
            p_short, p_flat, p_long = probabilities[i]

            # Determine direction and strength
            direction = 0
            strength = 0.0

            if p_long > threshold and p_long > p_short:
                direction = 1
                strength = float(p_long)
            elif p_short > threshold and p_short > p_long:
                direction = -1
                strength = float(p_short)
            else:
                # No strong signal, skip
                continue

            signals.append(
                StrategySignal(
                    timestamp=timestamps[i],
                    symbol=symbols[i],
                    direction=direction,
                    strength=strength,
                    metadata={
                        "p_short": float(p_short),
                        "p_flat": float(p_flat),
                        "p_long": float(p_long),
                    },
                )
            )

        return signals

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self._feature_importance.copy()

    def get_booster(self) -> xgb.Booster | None:
        """
        Get the underlying XGBoost Booster object.

        Returns:
            XGBoost Booster or None if not trained
        """
        if self.model is None:
            return None
        return self.model.get_booster()
