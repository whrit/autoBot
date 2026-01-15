"""
Common Types Library for Autonomous Equities Trading Engine.

This module provides all Pydantic models needed across the system, including:
- Market data models (Trade, Quote, Bar, MicroBar)
- Feature models (DecisionFrame, Label)
- Strategy and configuration models
- Artifact models for persistence

All schemas match the authoritative definitions in PRD.md.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# ENUMS
# =============================================================================


class LifecycleState(str, Enum):
    """Strategy lifecycle states for promotion pipeline."""

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PAPER = "paper"
    PROMOTED = "promoted"
    RETIRED = "retired"


class RunType(str, Enum):
    """Execution run types."""

    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class ArtifactKind(str, Enum):
    """Types of strategy artifacts."""

    JSON_RULES = "json_rules"
    ONNX_MODEL = "onnx_model"


# =============================================================================
# MARKET DATA MODELS
# =============================================================================


class Trade(BaseModel):
    """
    Raw trade tick from exchange.

    Matches schema: lake/raw/trades/
    Both timestamps required for reproducibility and latency analysis.
    """

    symbol: str
    ts_event: datetime = Field(..., description="Exchange timestamp")
    ts_recv: datetime = Field(..., description="Local receipt timestamp")
    price: float = Field(..., gt=0, description="Trade price (must be positive)")
    size: float = Field(..., gt=0, description="Trade size (must be positive)")
    exchange: str
    conditions: str | None = Field(default=None, description="Trade conditions")


class Quote(BaseModel):
    """
    NBBO quote snapshot.

    Matches schema: lake/raw/quotes/
    Provides computed properties for spread, midprice, and microprice.
    """

    symbol: str
    ts_event: datetime = Field(..., description="Exchange timestamp")
    ts_recv: datetime = Field(..., description="Local receipt timestamp")
    bid_price: float = Field(..., gt=0, description="Best bid price")
    bid_size: float = Field(..., ge=0, description="Best bid size")
    ask_price: float = Field(..., gt=0, description="Best ask price")
    ask_size: float = Field(..., ge=0, description="Best ask size")
    exchange: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @computed_field  # type: ignore[prop-decorator]
    @property
    def midprice(self) -> float:
        """Calculate midprice (arithmetic mean of bid and ask)."""
        return (self.bid_price + self.ask_price) / 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def microprice(self) -> float:
        """
        Calculate size-weighted midprice.

        Falls back to midprice when total size is zero.
        """
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return self.midprice
        return (self.bid_price * self.ask_size + self.ask_price * self.bid_size) / total_size


class Bar(BaseModel):
    """
    Standard OHLCV bar.

    Matches schema: lake/features/bars/
    Granularities: 1m, 5m, 15m
    """

    symbol: str
    bar_start: datetime
    bar_end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    returns: float | None = None
    atr: float | None = None
    realized_vol: float | None = None


class MicroBar(BaseModel):
    """
    Microstructure bar derived from quotes/trades.

    Matches schema: lake/features/micro_bars/
    Granularities: 5s, 15s, 30s
    """

    symbol: str
    bar_start: datetime
    bar_end: datetime
    vwap: float
    midprice: float
    microprice: float
    spread: float
    bid_size: float
    ask_size: float
    quote_imbalance: float
    trade_volume: float
    realized_vol: float


class DecisionFrame(BaseModel):
    """
    Multi-timeframe feature vector for decision making.

    Matches schema: lake/features/decision_frame/
    Contains features from 30s microstructure, 1m, 5m, and 15m timeframes.
    """

    symbol: str
    decision_ts: datetime

    # Microstructure features (30s)
    spread_30s: float
    microprice_30s: float
    quote_imbalance_30s: float
    vol_30s: float

    # 1m features
    ret_1m: float
    vol_1m: float
    atr_1m: float

    # 5m features
    ret_5m: float
    trend_5m: float
    vol_5m: float

    # 15m features
    ret_15m: float
    trend_15m: float
    vol_15m: float


class Label(BaseModel):
    """
    Prediction target label.

    Matches schema: lake/labels/
    Direction is constrained to -1, 0, or +1.
    """

    symbol: str
    decision_ts: datetime
    horizon: int = Field(..., description="Horizon in seconds")
    fwd_return_mid: float
    fwd_return_net: float = Field(..., description="Net of half-spread")
    direction: int = Field(..., ge=-1, le=1, description="Direction: -1, 0, or +1")


# =============================================================================
# STRATEGY & CONFIGURATION MODELS
# =============================================================================


class TrainingMetadata(BaseModel):
    """
    GPU/CPU training metadata for reproducibility.

    Records device, backend, versions, and determinism settings.
    """

    training_device: str = Field(default="cpu", description="cpu | cuda")
    gpu_backend: str | None = Field(default=None, description="xgboost_gpu | pytorch_cuda | cpu")
    cuda_version: str | None = None
    driver_version: str | None = None
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)


class SignalLogic(BaseModel):
    """Signal generation logic configuration."""

    type: str = Field(..., description="Signal type: threshold, ml, etc.")
    expression: str | None = Field(default=None, description="Expression for threshold signals")
    threshold: float | None = Field(default=None, description="Threshold value")


class SizingConfig(BaseModel):
    """Position sizing configuration."""

    type: str = Field(..., description="Sizing type: vol_target, fixed, etc.")
    target_vol: float | None = Field(default=None, description="Target volatility for vol_target")
    max_notional: float = Field(default=50000.0, description="Maximum notional position")


class ExecutionConfig(BaseModel):
    """Order execution configuration."""

    order_type: str = Field(default="marketable_limit")
    limit_offset_bps: float = Field(default=2.0, description="Limit offset in basis points")
    time_in_force: str = Field(default="IOC", description="Time in force: IOC, GTC, etc.")


class Strategy(BaseModel):
    """
    Strategy definition.

    Complete specification for a trading strategy including signal logic,
    sizing, execution, and optional training metadata.
    """

    strategy_id: str
    strategy_family: str
    feature_schema_version: str
    label_horizon_sec: int
    cost_model_version: str
    risk_profile_id: str
    symbols: list[str]
    decision_interval_sec: int = Field(default=60)
    signal_logic: SignalLogic
    sizing: SizingConfig
    execution: ExecutionConfig
    training_metadata: TrainingMetadata | None = None


class CostModelConfig(BaseModel):
    """
    Slippage/cost model configuration.

    Implements: slippage_bps = a*spread + b*(order/book) + c*vol + fixed
    """

    version: str
    spread_coef: float = Field(default=1.0, description="Coefficient 'a' for spread")
    size_coef: float = Field(default=1.0, description="Coefficient 'b' for size impact")
    vol_coef: float = Field(default=1.0, description="Coefficient 'c' for volatility")
    fixed_cost_bps: float = Field(default=0.0, description="Fixed cost in basis points")


class RiskProfile(BaseModel):
    """
    Risk limits configuration.

    Defines position limits, exposure limits, and loss limits.
    """

    profile_id: str
    max_position_notional: float
    max_gross_exposure: float
    max_net_exposure: float
    max_daily_loss: float
    max_drawdown: float
    slippage_tolerance_bps: float = Field(default=10.0)


# =============================================================================
# ARTIFACT MODELS
# =============================================================================


class CalibrationConfig(BaseModel):
    """Calibration configuration for ONNX models."""

    expected_return_scale: float = Field(default=1.0)
    confidence_threshold: float = Field(default=0.6)


class InferencePreferences(BaseModel):
    """Inference preferences for ONNX models."""

    prefer_gpu: bool = Field(default=False)
    cpu_fallback: bool = Field(default=True)


class OnnxMetadata(BaseModel):
    """
    ONNX artifact sidecar metadata.

    Matches the .json sidecar file for ONNX model artifacts.
    """

    onnx_version: str
    feature_schema_version: str
    label_horizon_sec: int
    cost_model_version: str
    risk_profile_id: str
    calibration: CalibrationConfig
    training_metadata: TrainingMetadata
    inference_preferences: InferencePreferences


class Artifact(BaseModel):
    """
    Strategy artifact wrapper.

    Can represent either JSON rules or ONNX model artifacts.
    Matches the authoritative artifact contract in PRD.md.
    """

    artifact_version: str
    artifact_kind: ArtifactKind
    strategy_id: str
    strategy_family: str
    feature_schema_version: str
    label_horizon_sec: int
    cost_model_version: str
    risk_profile_id: str
    symbols: list[str]
    decision_interval_sec: int = Field(default=60)
    signal_logic: SignalLogic
    sizing: SizingConfig
    execution: ExecutionConfig
    training_metadata: TrainingMetadata | None = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LifecycleState",
    "RunType",
    "ArtifactKind",
    # Market Data Models
    "Trade",
    "Quote",
    "Bar",
    "MicroBar",
    "DecisionFrame",
    "Label",
    # Strategy & Configuration Models
    "TrainingMetadata",
    "SignalLogic",
    "SizingConfig",
    "ExecutionConfig",
    "Strategy",
    "CostModelConfig",
    "RiskProfile",
    # Artifact Models
    "CalibrationConfig",
    "InferencePreferences",
    "OnnxMetadata",
    "Artifact",
]
