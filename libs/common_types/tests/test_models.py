"""
Comprehensive tests for common_types library.

TDD London School: Write tests FIRST, then implement models to make them pass.
These tests verify all Pydantic models needed across the Autonomous Equities Trading Engine.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from common_types import (
    # Artifact Models
    Artifact,
    ArtifactKind,
    Bar,
    CalibrationConfig,
    CostModelConfig,
    DecisionFrame,
    ExecutionConfig,
    InferencePreferences,
    Label,
    # Enums
    LifecycleState,
    MicroBar,
    OnnxMetadata,
    Quote,
    RiskProfile,
    RunType,
    SignalLogic,
    SizingConfig,
    Strategy,
    # Market Data Models
    Trade,
    # Strategy & Configuration Models
    TrainingMetadata,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestLifecycleState:
    """Tests for LifecycleState enum."""

    def test_all_states_exist(self) -> None:
        """All required lifecycle states must be defined."""
        assert LifecycleState.CANDIDATE == "candidate"
        assert LifecycleState.SHADOW == "shadow"
        assert LifecycleState.PAPER == "paper"
        assert LifecycleState.PROMOTED == "promoted"
        assert LifecycleState.RETIRED == "retired"

    def test_is_string_enum(self) -> None:
        """LifecycleState should be usable as string."""
        state = LifecycleState.CANDIDATE
        assert state == "candidate"
        assert isinstance(state.value, str)


class TestRunType:
    """Tests for RunType enum."""

    def test_all_run_types_exist(self) -> None:
        """All required run types must be defined."""
        assert RunType.BACKTEST == "backtest"
        assert RunType.SHADOW == "shadow"
        assert RunType.PAPER == "paper"
        assert RunType.LIVE == "live"


class TestArtifactKind:
    """Tests for ArtifactKind enum."""

    def test_all_artifact_kinds_exist(self) -> None:
        """All required artifact kinds must be defined."""
        assert ArtifactKind.JSON_RULES == "json_rules"
        assert ArtifactKind.ONNX_MODEL == "onnx_model"


# =============================================================================
# MARKET DATA MODEL TESTS
# =============================================================================


class TestTrade:
    """Tests for Trade model (raw trade tick from exchange)."""

    @pytest.fixture
    def valid_trade_data(self) -> dict[str, Any]:
        """Fixture providing valid trade data."""
        return {
            "symbol": "SPY",
            "ts_event": datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            "ts_recv": datetime(2024, 1, 15, 9, 30, 0, 100000, tzinfo=UTC),
            "price": 475.50,
            "size": 100.0,
            "exchange": "NYSE",
            "conditions": "regular",
        }

    def test_trade_creation(self, valid_trade_data: dict[str, Any]) -> None:
        """Test valid trade creation with all required fields."""
        trade = Trade(**valid_trade_data)
        assert trade.symbol == "SPY"
        assert trade.price == 475.50
        assert trade.size == 100.0
        assert trade.exchange == "NYSE"
        assert trade.conditions == "regular"

    def test_trade_timestamps_required(self) -> None:
        """Both ts_event and ts_recv must be present."""
        with pytest.raises(ValueError):
            Trade(
                symbol="SPY",
                price=475.50,
                size=100.0,
                exchange="NYSE",
            )

    def test_trade_price_must_be_positive(self, valid_trade_data: dict[str, Any]) -> None:
        """Price must be greater than 0."""
        valid_trade_data["price"] = 0
        with pytest.raises(ValueError):
            Trade(**valid_trade_data)

        valid_trade_data["price"] = -10.0
        with pytest.raises(ValueError):
            Trade(**valid_trade_data)

    def test_trade_size_must_be_positive(self, valid_trade_data: dict[str, Any]) -> None:
        """Size must be greater than 0."""
        valid_trade_data["size"] = 0
        with pytest.raises(ValueError):
            Trade(**valid_trade_data)

        valid_trade_data["size"] = -100.0
        with pytest.raises(ValueError):
            Trade(**valid_trade_data)

    def test_trade_conditions_optional(self, valid_trade_data: dict[str, Any]) -> None:
        """Conditions field is optional."""
        del valid_trade_data["conditions"]
        trade = Trade(**valid_trade_data)
        assert trade.conditions is None

    def test_trade_serialization(self, valid_trade_data: dict[str, Any]) -> None:
        """Trade should serialize to dict correctly."""
        trade = Trade(**valid_trade_data)
        data = trade.model_dump()
        assert data["symbol"] == "SPY"
        assert data["price"] == 475.50


class TestQuote:
    """Tests for Quote model (NBBO quote snapshot)."""

    @pytest.fixture
    def valid_quote_data(self) -> dict[str, Any]:
        """Fixture providing valid quote data."""
        return {
            "symbol": "SPY",
            "ts_event": datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            "ts_recv": datetime(2024, 1, 15, 9, 30, 0, 100000, tzinfo=UTC),
            "bid_price": 475.40,
            "bid_size": 500.0,
            "ask_price": 475.60,
            "ask_size": 300.0,
            "exchange": "NASDAQ",
        }

    def test_quote_creation(self, valid_quote_data: dict[str, Any]) -> None:
        """Test valid quote creation with all required fields."""
        quote = Quote(**valid_quote_data)
        assert quote.symbol == "SPY"
        assert quote.bid_price == 475.40
        assert quote.ask_price == 475.60
        assert quote.exchange == "NASDAQ"

    def test_spread_calculation(self, valid_quote_data: dict[str, Any]) -> None:
        """spread = ask_price - bid_price."""
        quote = Quote(**valid_quote_data)
        expected_spread = 475.60 - 475.40  # 0.20
        assert abs(quote.spread - expected_spread) < 1e-10

    def test_midprice_calculation(self, valid_quote_data: dict[str, Any]) -> None:
        """midprice = (bid_price + ask_price) / 2."""
        quote = Quote(**valid_quote_data)
        expected_midprice = (475.40 + 475.60) / 2  # 475.50
        assert abs(quote.midprice - expected_midprice) < 1e-10

    def test_microprice_calculation(self, valid_quote_data: dict[str, Any]) -> None:
        """microprice = size-weighted midprice."""
        quote = Quote(**valid_quote_data)
        # microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        # = (475.40 * 300 + 475.60 * 500) / (500 + 300)
        # = (142620 + 237800) / 800 = 380420 / 800 = 475.525
        expected_microprice = (475.40 * 300.0 + 475.60 * 500.0) / (500.0 + 300.0)
        assert abs(quote.microprice - expected_microprice) < 1e-10

    def test_microprice_zero_size_fallback(self, valid_quote_data: dict[str, Any]) -> None:
        """microprice falls back to midprice when sizes are zero."""
        valid_quote_data["bid_size"] = 0.0
        valid_quote_data["ask_size"] = 0.0
        quote = Quote(**valid_quote_data)
        assert quote.microprice == quote.midprice

    def test_quote_bid_price_must_be_positive(self, valid_quote_data: dict[str, Any]) -> None:
        """bid_price must be greater than 0."""
        valid_quote_data["bid_price"] = 0
        with pytest.raises(ValueError):
            Quote(**valid_quote_data)

    def test_quote_ask_price_must_be_positive(self, valid_quote_data: dict[str, Any]) -> None:
        """ask_price must be greater than 0."""
        valid_quote_data["ask_price"] = -10.0
        with pytest.raises(ValueError):
            Quote(**valid_quote_data)

    def test_quote_bid_size_must_be_non_negative(self, valid_quote_data: dict[str, Any]) -> None:
        """bid_size must be >= 0."""
        valid_quote_data["bid_size"] = -100.0
        with pytest.raises(ValueError):
            Quote(**valid_quote_data)

    def test_quote_ask_size_must_be_non_negative(self, valid_quote_data: dict[str, Any]) -> None:
        """ask_size must be >= 0."""
        valid_quote_data["ask_size"] = -50.0
        with pytest.raises(ValueError):
            Quote(**valid_quote_data)


class TestBar:
    """Tests for Bar model (standard OHLCV bar)."""

    @pytest.fixture
    def valid_bar_data(self) -> dict[str, Any]:
        """Fixture providing valid bar data."""
        return {
            "symbol": "SPY",
            "bar_start": datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            "bar_end": datetime(2024, 1, 15, 9, 31, 0, tzinfo=UTC),
            "open": 475.00,
            "high": 476.00,
            "low": 474.50,
            "close": 475.80,
            "volume": 50000.0,
            "returns": 0.00168,
            "atr": 0.50,
            "realized_vol": 0.015,
        }

    def test_bar_creation(self, valid_bar_data: dict[str, Any]) -> None:
        """Test valid bar creation with all fields."""
        bar = Bar(**valid_bar_data)
        assert bar.symbol == "SPY"
        assert bar.open == 475.00
        assert bar.high == 476.00
        assert bar.low == 474.50
        assert bar.close == 475.80
        assert bar.volume == 50000.0

    def test_bar_optional_fields(self, valid_bar_data: dict[str, Any]) -> None:
        """returns, atr, realized_vol are optional."""
        del valid_bar_data["returns"]
        del valid_bar_data["atr"]
        del valid_bar_data["realized_vol"]
        bar = Bar(**valid_bar_data)
        assert bar.returns is None
        assert bar.atr is None
        assert bar.realized_vol is None

    def test_bar_serialization(self, valid_bar_data: dict[str, Any]) -> None:
        """Bar should serialize to dict correctly."""
        bar = Bar(**valid_bar_data)
        data = bar.model_dump()
        assert data["symbol"] == "SPY"
        assert data["close"] == 475.80


class TestMicroBar:
    """Tests for MicroBar model (microstructure bar derived from quotes/trades)."""

    @pytest.fixture
    def valid_microbar_data(self) -> dict[str, Any]:
        """Fixture providing valid microbar data."""
        return {
            "symbol": "SPY",
            "bar_start": datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            "bar_end": datetime(2024, 1, 15, 9, 30, 30, tzinfo=UTC),
            "vwap": 475.45,
            "midprice": 475.50,
            "microprice": 475.52,
            "spread": 0.04,
            "bid_size": 10000.0,
            "ask_size": 8000.0,
            "quote_imbalance": 0.111,
            "trade_volume": 5000.0,
            "realized_vol": 0.012,
        }

    def test_microbar_creation(self, valid_microbar_data: dict[str, Any]) -> None:
        """Test valid microbar creation with all fields."""
        microbar = MicroBar(**valid_microbar_data)
        assert microbar.symbol == "SPY"
        assert microbar.vwap == 475.45
        assert microbar.midprice == 475.50
        assert microbar.microprice == 475.52
        assert microbar.spread == 0.04
        assert microbar.quote_imbalance == 0.111


class TestDecisionFrame:
    """Tests for DecisionFrame model (multi-timeframe feature vector)."""

    @pytest.fixture
    def valid_decision_frame_data(self) -> dict[str, Any]:
        """Fixture providing valid decision frame data."""
        return {
            "symbol": "SPY",
            "decision_ts": datetime(2024, 1, 15, 9, 31, 0, tzinfo=UTC),
            # Microstructure features (30s)
            "spread_30s": 0.04,
            "microprice_30s": 475.52,
            "quote_imbalance_30s": 0.111,
            "vol_30s": 0.012,
            # 1m features
            "ret_1m": 0.0015,
            "vol_1m": 0.015,
            "atr_1m": 0.50,
            # 5m features
            "ret_5m": 0.003,
            "trend_5m": 0.8,
            "vol_5m": 0.018,
            # 15m features
            "ret_15m": 0.005,
            "trend_15m": 0.6,
            "vol_15m": 0.022,
        }

    def test_decision_frame_creation(self, valid_decision_frame_data: dict[str, Any]) -> None:
        """Test valid decision frame creation with all fields."""
        df = DecisionFrame(**valid_decision_frame_data)
        assert df.symbol == "SPY"
        assert df.spread_30s == 0.04
        assert df.ret_1m == 0.0015
        assert df.trend_5m == 0.8
        assert df.vol_15m == 0.022

    def test_decision_frame_all_timeframes_present(
        self, valid_decision_frame_data: dict[str, Any]
    ) -> None:
        """All timeframe features must be present."""
        df = DecisionFrame(**valid_decision_frame_data)
        # 30s microstructure
        assert hasattr(df, "spread_30s")
        assert hasattr(df, "microprice_30s")
        assert hasattr(df, "quote_imbalance_30s")
        assert hasattr(df, "vol_30s")
        # 1m
        assert hasattr(df, "ret_1m")
        assert hasattr(df, "vol_1m")
        assert hasattr(df, "atr_1m")
        # 5m
        assert hasattr(df, "ret_5m")
        assert hasattr(df, "trend_5m")
        assert hasattr(df, "vol_5m")
        # 15m
        assert hasattr(df, "ret_15m")
        assert hasattr(df, "trend_15m")
        assert hasattr(df, "vol_15m")


class TestLabel:
    """Tests for Label model (prediction target label)."""

    @pytest.fixture
    def valid_label_data(self) -> dict[str, Any]:
        """Fixture providing valid label data."""
        return {
            "symbol": "SPY",
            "decision_ts": datetime(2024, 1, 15, 9, 31, 0, tzinfo=UTC),
            "horizon": 60,
            "fwd_return_mid": 0.0015,
            "fwd_return_net": 0.0012,
            "direction": 1,
        }

    def test_label_creation(self, valid_label_data: dict[str, Any]) -> None:
        """Test valid label creation."""
        label = Label(**valid_label_data)
        assert label.symbol == "SPY"
        assert label.horizon == 60
        assert label.fwd_return_mid == 0.0015
        assert label.fwd_return_net == 0.0012
        assert label.direction == 1

    def test_label_direction_values(self, valid_label_data: dict[str, Any]) -> None:
        """direction must be -1, 0, or +1."""
        # Valid values
        for direction in [-1, 0, 1]:
            valid_label_data["direction"] = direction
            label = Label(**valid_label_data)
            assert label.direction == direction

    def test_label_invalid_direction(self, valid_label_data: dict[str, Any]) -> None:
        """direction outside [-1, 1] should fail."""
        valid_label_data["direction"] = 2
        with pytest.raises(ValueError):
            Label(**valid_label_data)

        valid_label_data["direction"] = -2
        with pytest.raises(ValueError):
            Label(**valid_label_data)


# =============================================================================
# STRATEGY & CONFIGURATION MODEL TESTS
# =============================================================================


class TestTrainingMetadata:
    """Tests for TrainingMetadata model."""

    def test_training_metadata_defaults(self) -> None:
        """Test default values."""
        metadata = TrainingMetadata()
        assert metadata.training_device == "cpu"
        assert metadata.gpu_backend is None
        assert metadata.cuda_version is None
        assert metadata.driver_version is None
        assert metadata.seed == 42
        assert metadata.deterministic is False

    def test_training_metadata_gpu_config(self) -> None:
        """Test GPU configuration."""
        metadata = TrainingMetadata(
            training_device="cuda",
            gpu_backend="xgboost_gpu",
            cuda_version="12.1",
            driver_version="535.86.05",
            seed=1337,
            deterministic=True,
        )
        assert metadata.training_device == "cuda"
        assert metadata.gpu_backend == "xgboost_gpu"
        assert metadata.cuda_version == "12.1"


class TestSignalLogic:
    """Tests for SignalLogic model."""

    def test_signal_logic_threshold(self) -> None:
        """Test threshold signal logic."""
        logic = SignalLogic(
            type="threshold",
            expression="ret_5m > theta",
            threshold=0.0015,
        )
        assert logic.type == "threshold"
        assert logic.expression == "ret_5m > theta"
        assert logic.threshold == 0.0015

    def test_signal_logic_ml(self) -> None:
        """Test ML signal logic."""
        logic = SignalLogic(type="ml")
        assert logic.type == "ml"
        assert logic.expression is None
        assert logic.threshold is None


class TestSizingConfig:
    """Tests for SizingConfig model."""

    def test_sizing_config_vol_target(self) -> None:
        """Test volatility target sizing."""
        sizing = SizingConfig(
            type="vol_target",
            target_vol=0.10,
            max_notional=50000.0,
        )
        assert sizing.type == "vol_target"
        assert sizing.target_vol == 0.10
        assert sizing.max_notional == 50000.0

    def test_sizing_config_defaults(self) -> None:
        """Test default max_notional."""
        sizing = SizingConfig(type="fixed")
        assert sizing.max_notional == 50000.0


class TestExecutionConfig:
    """Tests for ExecutionConfig model."""

    def test_execution_config_defaults(self) -> None:
        """Test default execution config."""
        exec_config = ExecutionConfig()
        assert exec_config.order_type == "marketable_limit"
        assert exec_config.limit_offset_bps == 2.0
        assert exec_config.time_in_force == "IOC"

    def test_execution_config_custom(self) -> None:
        """Test custom execution config."""
        exec_config = ExecutionConfig(
            order_type="limit",
            limit_offset_bps=5.0,
            time_in_force="GTC",
        )
        assert exec_config.order_type == "limit"
        assert exec_config.limit_offset_bps == 5.0
        assert exec_config.time_in_force == "GTC"


class TestStrategy:
    """Tests for Strategy model."""

    @pytest.fixture
    def valid_strategy_data(self) -> dict[str, Any]:
        """Fixture providing valid strategy data."""
        return {
            "strategy_id": "trend_5m_v3",
            "strategy_family": "trend",
            "feature_schema_version": "decision_frame_v1",
            "label_horizon_sec": 300,
            "cost_model_version": "cm_v1",
            "risk_profile_id": "rp_default",
            "symbols": ["SPY", "QQQ"],
            "decision_interval_sec": 60,
            "signal_logic": {
                "type": "threshold",
                "expression": "ret_5m > theta",
                "threshold": 0.0015,
            },
            "sizing": {
                "type": "vol_target",
                "target_vol": 0.10,
                "max_notional": 50000.0,
            },
            "execution": {
                "order_type": "marketable_limit",
                "limit_offset_bps": 2.0,
                "time_in_force": "IOC",
            },
        }

    def test_strategy_creation(self, valid_strategy_data: dict[str, Any]) -> None:
        """Test valid strategy creation."""
        strategy = Strategy(**valid_strategy_data)
        assert strategy.strategy_id == "trend_5m_v3"
        assert strategy.strategy_family == "trend"
        assert strategy.symbols == ["SPY", "QQQ"]
        assert strategy.signal_logic.type == "threshold"
        assert strategy.sizing.target_vol == 0.10

    def test_strategy_with_training_metadata(self, valid_strategy_data: dict[str, Any]) -> None:
        """Test strategy with GPU training metadata."""
        valid_strategy_data["training_metadata"] = {
            "training_device": "cuda",
            "gpu_backend": "xgboost_gpu",
            "seed": 1337,
        }
        strategy = Strategy(**valid_strategy_data)
        assert strategy.training_metadata is not None
        assert strategy.training_metadata.training_device == "cuda"

    def test_strategy_defaults(self, valid_strategy_data: dict[str, Any]) -> None:
        """Test default values."""
        strategy = Strategy(**valid_strategy_data)
        assert strategy.decision_interval_sec == 60
        assert strategy.training_metadata is None


class TestCostModelConfig:
    """Tests for CostModelConfig model."""

    def test_cost_model_creation(self) -> None:
        """Test cost model creation."""
        cost_model = CostModelConfig(
            version="cm_v1",
            spread_coef=1.2,
            size_coef=0.8,
            vol_coef=1.5,
            fixed_cost_bps=0.5,
        )
        assert cost_model.version == "cm_v1"
        assert cost_model.spread_coef == 1.2
        assert cost_model.size_coef == 0.8
        assert cost_model.vol_coef == 1.5
        assert cost_model.fixed_cost_bps == 0.5

    def test_cost_model_defaults(self) -> None:
        """Test default coefficients."""
        cost_model = CostModelConfig(version="cm_v1")
        assert cost_model.spread_coef == 1.0
        assert cost_model.size_coef == 1.0
        assert cost_model.vol_coef == 1.0
        assert cost_model.fixed_cost_bps == 0.0


class TestRiskProfile:
    """Tests for RiskProfile model."""

    @pytest.fixture
    def valid_risk_profile_data(self) -> dict[str, Any]:
        """Fixture providing valid risk profile data."""
        return {
            "profile_id": "rp_default",
            "max_position_notional": 100000.0,
            "max_gross_exposure": 500000.0,
            "max_net_exposure": 200000.0,
            "max_daily_loss": 10000.0,
            "max_drawdown": 0.05,
            "slippage_tolerance_bps": 10.0,
        }

    def test_risk_profile_creation(self, valid_risk_profile_data: dict[str, Any]) -> None:
        """Test risk profile creation."""
        profile = RiskProfile(**valid_risk_profile_data)
        assert profile.profile_id == "rp_default"
        assert profile.max_position_notional == 100000.0
        assert profile.max_gross_exposure == 500000.0
        assert profile.max_drawdown == 0.05

    def test_risk_profile_defaults(self, valid_risk_profile_data: dict[str, Any]) -> None:
        """Test default slippage tolerance."""
        del valid_risk_profile_data["slippage_tolerance_bps"]
        profile = RiskProfile(**valid_risk_profile_data)
        assert profile.slippage_tolerance_bps == 10.0


# =============================================================================
# ARTIFACT MODEL TESTS
# =============================================================================


class TestCalibrationConfig:
    """Tests for CalibrationConfig model."""

    def test_calibration_config_creation(self) -> None:
        """Test calibration config creation."""
        calibration = CalibrationConfig(
            expected_return_scale=1.0,
            confidence_threshold=0.6,
        )
        assert calibration.expected_return_scale == 1.0
        assert calibration.confidence_threshold == 0.6


class TestInferencePreferences:
    """Tests for InferencePreferences model."""

    def test_inference_preferences_defaults(self) -> None:
        """Test default inference preferences."""
        prefs = InferencePreferences()
        assert prefs.prefer_gpu is False
        assert prefs.cpu_fallback is True


class TestOnnxMetadata:
    """Tests for OnnxMetadata model (ONNX artifact sidecar)."""

    @pytest.fixture
    def valid_onnx_metadata_data(self) -> dict[str, Any]:
        """Fixture providing valid ONNX metadata."""
        return {
            "onnx_version": "1.15",
            "feature_schema_version": "decision_frame_v1",
            "label_horizon_sec": 60,
            "cost_model_version": "cm_v1",
            "risk_profile_id": "rp_default",
            "calibration": {
                "expected_return_scale": 1.0,
                "confidence_threshold": 0.6,
            },
            "training_metadata": {
                "training_device": "cuda",
                "gpu_backend": "xgboost_gpu",
                "seed": 1337,
            },
            "inference_preferences": {
                "prefer_gpu": False,
                "cpu_fallback": True,
            },
        }

    def test_onnx_metadata_creation(self, valid_onnx_metadata_data: dict[str, Any]) -> None:
        """Test ONNX metadata creation."""
        metadata = OnnxMetadata(**valid_onnx_metadata_data)
        assert metadata.onnx_version == "1.15"
        assert metadata.feature_schema_version == "decision_frame_v1"
        assert metadata.calibration.confidence_threshold == 0.6
        assert metadata.training_metadata.training_device == "cuda"


class TestArtifact:
    """Tests for Artifact model (strategy artifact wrapper)."""

    @pytest.fixture
    def valid_artifact_data(self) -> dict[str, Any]:
        """Fixture providing valid artifact data."""
        return {
            "artifact_version": "1.1",
            "artifact_kind": "json_rules",
            "strategy_id": "trend_5m_v3",
            "strategy_family": "trend",
            "feature_schema_version": "decision_frame_v1",
            "label_horizon_sec": 300,
            "cost_model_version": "cm_v1",
            "risk_profile_id": "rp_default",
            "symbols": ["SPY", "QQQ"],
            "decision_interval_sec": 60,
            "signal_logic": {
                "type": "threshold",
                "expression": "ret_5m > theta",
                "threshold": 0.0015,
            },
            "sizing": {
                "type": "vol_target",
                "target_vol": 0.10,
                "max_notional": 50000.0,
            },
            "execution": {
                "order_type": "marketable_limit",
                "limit_offset_bps": 2.0,
                "time_in_force": "IOC",
            },
        }

    def test_artifact_creation(self, valid_artifact_data: dict[str, Any]) -> None:
        """Test artifact creation."""
        artifact = Artifact(**valid_artifact_data)
        assert artifact.artifact_version == "1.1"
        assert artifact.artifact_kind == ArtifactKind.JSON_RULES
        assert artifact.strategy_id == "trend_5m_v3"

    def test_artifact_with_training_metadata(self, valid_artifact_data: dict[str, Any]) -> None:
        """Test artifact with training metadata."""
        valid_artifact_data["training_metadata"] = {
            "training_device": "cuda",
            "gpu_backend": "xgboost_gpu",
            "cuda_version": "12.1",
            "seed": 1337,
        }
        artifact = Artifact(**valid_artifact_data)
        assert artifact.training_metadata is not None
        assert artifact.training_metadata.cuda_version == "12.1"

    def test_artifact_json_serialization(self, valid_artifact_data: dict[str, Any]) -> None:
        """Test artifact JSON serialization."""
        artifact = Artifact(**valid_artifact_data)
        json_str = artifact.model_dump_json()
        assert "trend_5m_v3" in json_str
        assert "json_rules" in json_str


# =============================================================================
# SERIALIZATION AND VALIDATION TESTS
# =============================================================================


class TestModelSerialization:
    """Tests for model serialization and deserialization."""

    def test_trade_round_trip(self) -> None:
        """Test Trade serialization round-trip."""
        original = Trade(
            symbol="SPY",
            ts_event=datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            ts_recv=datetime(2024, 1, 15, 9, 30, 0, 100000, tzinfo=UTC),
            price=475.50,
            size=100.0,
            exchange="NYSE",
        )
        data = original.model_dump()
        restored = Trade(**data)
        assert restored.symbol == original.symbol
        assert restored.price == original.price
        assert restored.ts_event == original.ts_event

    def test_quote_round_trip(self) -> None:
        """Test Quote serialization round-trip."""
        original = Quote(
            symbol="SPY",
            ts_event=datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            ts_recv=datetime(2024, 1, 15, 9, 30, 0, 100000, tzinfo=UTC),
            bid_price=475.40,
            bid_size=500.0,
            ask_price=475.60,
            ask_size=300.0,
            exchange="NASDAQ",
        )
        data = original.model_dump()
        restored = Quote(**data)
        assert restored.symbol == original.symbol
        assert restored.spread == original.spread
        assert restored.microprice == original.microprice

    def test_strategy_json_serialization(self) -> None:
        """Test Strategy JSON serialization."""
        strategy = Strategy(
            strategy_id="test_strategy",
            strategy_family="trend",
            feature_schema_version="v1",
            label_horizon_sec=60,
            cost_model_version="cm_v1",
            risk_profile_id="rp_default",
            symbols=["SPY"],
            signal_logic=SignalLogic(type="threshold", threshold=0.001),
            sizing=SizingConfig(type="fixed"),
            execution=ExecutionConfig(),
        )
        json_str = strategy.model_dump_json()
        assert "test_strategy" in json_str
        assert "trend" in json_str


class TestModelValidation:
    """Tests for model validation edge cases."""

    def test_empty_symbol_list_in_strategy(self) -> None:
        """Strategy should accept empty symbol list (validation happens elsewhere)."""
        strategy = Strategy(
            strategy_id="test",
            strategy_family="trend",
            feature_schema_version="v1",
            label_horizon_sec=60,
            cost_model_version="cm_v1",
            risk_profile_id="rp_default",
            symbols=[],
            signal_logic=SignalLogic(type="threshold"),
            sizing=SizingConfig(type="fixed"),
            execution=ExecutionConfig(),
        )
        assert strategy.symbols == []

    def test_negative_horizon_in_label(self) -> None:
        """Label horizon should accept any integer (business validation elsewhere)."""
        label = Label(
            symbol="SPY",
            decision_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            horizon=-60,  # Negative, but model accepts it
            fwd_return_mid=0.001,
            fwd_return_net=0.0008,
            direction=0,
        )
        assert label.horizon == -60

    def test_extreme_values_in_decision_frame(self) -> None:
        """DecisionFrame should accept extreme numeric values."""
        df = DecisionFrame(
            symbol="SPY",
            decision_ts=datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC),
            spread_30s=1e-10,
            microprice_30s=1e6,
            quote_imbalance_30s=-0.999,
            vol_30s=0.0,
            ret_1m=-0.5,
            vol_1m=1.0,
            atr_1m=100.0,
            ret_5m=0.5,
            trend_5m=-1.0,
            vol_5m=0.0001,
            ret_15m=0.0,
            trend_15m=0.0,
            vol_15m=10.0,
        )
        assert df.spread_30s == 1e-10
        assert df.microprice_30s == 1e6
