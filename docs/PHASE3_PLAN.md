# Phase 3 Plan: Scale, Strategy Library & Regime Allocation

**Autonomous Equities Taker Trading Engine**

**Document Version:** 1.0
**Created:** 2026-01-15
**Status:** Planning
**Estimated Duration:** 18 weeks (9 sprints)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategy Library Architecture](#2-strategy-library-architecture)
3. [Regime Classifier Design](#3-regime-classifier-design)
4. [Capital Allocator Design](#4-capital-allocator-design)
5. [Scalable Pipeline Design](#5-scalable-pipeline-design)
6. [Sprint Plan](#6-sprint-plan)
7. [Task Tracking Table](#7-task-tracking-table)
8. [Integration Points](#8-integration-points)
9. [Risk Register](#9-risk-register)
10. [Governance Framework](#10-governance-framework)
11. [Definition of Done](#11-definition-of-done)

---

## 1. Executive Summary

### 1.1 Phase 3 Objective

Transform the autonomous trading engine into a **self-allocating strategy factory** that:

- Runs multiple concurrent strategies across different families
- Dynamically allocates capital based on regime detection and live performance
- Scales data ingestion via Rust for high-throughput tick loads
- Maintains rigorous anti-overfit governance
- Enables safe live experimentation with automatic rollback

### 1.2 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Concurrent Strategies | >= 4 active families | Production deployment count |
| Regime Classification Accuracy | >= 75% | Out-of-sample regime prediction |
| Allocation Improvement | >= 10% Sharpe improvement vs best single strategy | Walk-forward comparison |
| Data Pipeline Throughput | >= 100,000 ticks/sec | Rust ingestor benchmark |
| Anti-Overfit Gate Pass Rate | >= 90% of live strategies | Deflated Sharpe validation |
| Canary Rollback Latency | < 500ms | Automated rollback timing |
| Operational Uptime | >= 99.5% | Production monitoring |

### 1.3 Key Deliverables

1. **Multi-Strategy Framework** - Concurrent execution of 4+ strategy families
2. **Regime Classifier** - HMM/ML-based market regime detection
3. **Capital Allocator** - Dynamic, drawdown-aware allocation engine
4. **Rust Tick Ingestor** - High-performance data pipeline
5. **Anti-Overfit Governance** - Purged CV, deflated Sharpe, stability tests
6. **Live Experimentation Framework** - Shadow + canary with auto-rollback
7. **Operational Maturity** - K8s deployment, DR, runbooks

---

## 2. Strategy Library Architecture

### 2.1 Strategy Family Definitions

| Family ID | Name | Description | Signal Horizon | Expected Regime |
|-----------|------|-------------|----------------|-----------------|
| `TREND` | Trend Following | Momentum-based directional strategies | 5m - 1h | Trending |
| `MREV` | Mean Reversion | Statistical reversion to mean | 1m - 15m | Ranging |
| `MICRO` | Microstructure | Quote imbalance, spread dynamics | 5s - 1m | High liquidity |
| `MLPRED` | ML Predictors | XGBoost/Neural network predictions | 1m - 15m | Regime-agnostic |

### 2.2 Strategy Family Specifications

#### 2.2.1 Trend Family (`TREND`)

```yaml
family: TREND
variants:
  - trend_momentum_5m
  - trend_breakout_15m
  - trend_channel_1h

features_used:
  - ret_5m, ret_15m, trend_5m, trend_15m
  - atr_5m, vol_15m

signal_logic:
  type: threshold | model
  entry: trend_strength > theta_entry
  exit: trend_reversal OR stop_loss

execution:
  order_type: marketable_limit
  sizing: vol_target | fixed_notional

regime_affinity:
  trending: 1.5x weight
  ranging: 0.3x weight
  reverting: 0.1x weight
```

#### 2.2.2 Mean Reversion Family (`MREV`)

```yaml
family: MREV
variants:
  - mrev_zscore_1m
  - mrev_bollinger_5m
  - mrev_rsi_15m

features_used:
  - spread_30s, microprice_30s
  - ret_1m, vol_1m, atr_1m

signal_logic:
  type: threshold
  entry: zscore > theta_entry OR zscore < -theta_entry
  exit: mean_cross OR time_stop

execution:
  order_type: marketable_limit
  sizing: vol_target

regime_affinity:
  trending: 0.2x weight
  ranging: 1.5x weight
  reverting: 1.2x weight
```

#### 2.2.3 Microstructure Family (`MICRO`)

```yaml
family: MICRO
variants:
  - micro_imbalance_5s
  - micro_spread_fade_15s
  - micro_momentum_confirm_30s

features_used:
  - quote_imbalance_30s, spread_30s
  - microprice_30s, vol_30s

signal_logic:
  type: model | threshold
  entry: imbalance_signal > theta AND liquidity_ok
  exit: imbalance_reversal OR time_limit

execution:
  order_type: marketable_limit
  sizing: liquidity_aware

regime_affinity:
  thin_liquidity: 0.1x weight
  normal_liquidity: 1.0x weight
  thick_liquidity: 1.3x weight
```

#### 2.2.4 ML Predictor Family (`MLPRED`)

```yaml
family: MLPRED
variants:
  - mlpred_xgb_1m
  - mlpred_xgb_5m
  - mlpred_neural_15m

features_used:
  - ALL decision_frame features
  - derived: cross-symbol correlations

signal_logic:
  type: model
  model_format: ONNX
  entry: predicted_return > theta AND confidence > conf_thresh
  exit: predicted_reversal OR stop_loss

training:
  device_preference: cuda | auto
  backend: xgboost_gpu | pytorch_cuda
  cv_method: purged_kfold

execution:
  order_type: marketable_limit
  sizing: confidence_weighted

regime_affinity:
  # Learned from training data
  adaptive: true
```

### 2.3 Concurrent Execution Model

```
                    +------------------------+
                    |   Strategy Orchestrator |
                    +------------------------+
                              |
          +-------------------+-------------------+
          |         |         |         |        |
    +-----v---+ +---v-----+ +-v-------+ +---v----+
    |  TREND  | |  MREV   | |  MICRO  | | MLPRED |
    | Runner  | | Runner  | | Runner  | | Runner |
    +---------+ +---------+ +---------+ +--------+
          |         |         |         |
          +-------------------+-------------------+
                              |
                    +------------------------+
                    |   Signal Aggregator    |
                    +------------------------+
                              |
                    +------------------------+
                    |   Capital Allocator    |
                    +------------------------+
                              |
                    +------------------------+
                    |   Order Manager        |
                    +------------------------+
```

### 2.4 Strategy Lifecycle Management

| State | Description | Entry Criteria | Exit Criteria |
|-------|-------------|----------------|---------------|
| `CANDIDATE` | Backtested, pending shadow | Backtest metrics pass | Shadow promotion OR rejection |
| `SHADOW` | Live signals, no orders | Shadow pass | Paper promotion OR rollback |
| `PAPER` | Paper account execution | Paper pass | Canary promotion OR rollback |
| `CANARY` | Small live allocation (5-10%) | Canary pass | Full promotion OR rollback |
| `ACTIVE` | Full production allocation | Continuous monitoring | Regime change OR performance decay |
| `PAUSED` | Temporarily disabled | Regime unfavorable OR limit breach | Regime favorable |
| `RETIRED` | Permanently disabled | Persistent underperformance | N/A |

### 2.5 Strategy Versioning

```json
{
  "strategy_id": "trend_momentum_5m_v3",
  "family": "TREND",
  "version": {
    "major": 3,
    "minor": 2,
    "patch": 1,
    "hash": "a1b2c3d4"
  },
  "artifact_path": "artifacts/trend/trend_momentum_5m_v3.2.1.json",
  "feature_schema": "decision_frame_v2",
  "promoted_at": "2026-01-15T10:00:00Z",
  "trained_on_data": "snapshot_20260114"
}
```

---

## 3. Regime Classifier Design

### 3.1 Feature Inputs for Regime Detection

#### 3.1.1 Volatility Features

| Feature | Calculation | Window |
|---------|-------------|--------|
| `vol_realized_1d` | Std(log returns) annualized | 1 day |
| `vol_realized_5d` | Std(log returns) annualized | 5 days |
| `vol_ratio` | vol_1d / vol_5d | - |
| `vix_proxy` | ATR(15m) / close | Rolling 20 bars |
| `vol_regime_zscore` | (vol_1d - vol_20d_ma) / vol_20d_std | - |

#### 3.1.2 Trend Features

| Feature | Calculation | Window |
|---------|-------------|--------|
| `trend_strength` | ADX(14) | 14 bars |
| `trend_direction` | Sign(EMA(20) - EMA(50)) | - |
| `trend_consistency` | % bars in trend direction | 20 bars |
| `momentum_zscore` | Standardized momentum | 60 bars |
| `breakout_distance` | (price - 20d_high) / ATR | - |

#### 3.1.3 Liquidity Features

| Feature | Calculation | Window |
|---------|-------------|--------|
| `spread_median` | Median bid-ask spread | 1 hour |
| `spread_zscore` | Standardized spread | 5 days |
| `depth_ratio` | bid_size / ask_size | Rolling |
| `volume_ratio` | volume_1h / volume_1h_ma | - |
| `trade_intensity` | trades_per_minute / median | - |

### 3.2 Classification Model Architecture

#### Option A: Hidden Markov Model (HMM)

```python
class RegimeHMM:
    """
    Hidden Markov Model for regime classification.

    States: K regimes (typically K=4-6)
    Observations: Feature vectors X(t)
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.transition_matrix = None  # K x K
        self.emission_params = None    # K x (mu, sigma)

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Fit HMM using Baum-Welch (unsupervised) or supervised initialization.
        """
        pass

    def predict(self, features: np.ndarray) -> RegimeOutput:
        """
        Returns: regime_id, regime_probs, transition_prob
        """
        pass
```

#### Option B: ML Classifier (XGBoost + Calibration)

```python
class RegimeClassifier:
    """
    Supervised regime classification with probability calibration.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model = XGBClassifier(
            tree_method="gpu_hist" if GPU_ENABLED else "hist",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05
        )
        self.calibrator = CalibratedClassifierCV(
            self.model, method="isotonic", cv=5
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train with purged cross-validation.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated regime probabilities.
        """
        pass
```

### 3.3 Regime Output Schema

```python
@dataclass
class RegimeOutput:
    """Output from regime classifier at each decision timestamp."""

    timestamp: datetime

    # Primary classification
    regime_id: int              # 0 to K-1
    regime_confidence: float    # 0.0 to 1.0

    # Dimensional regimes
    vol_regime: Literal["low", "medium", "high"]
    vol_percentile: float       # 0-100

    trend_regime: Literal["trending", "ranging", "reverting"]
    trend_strength: float       # 0.0 to 1.0

    liquidity_regime: Literal["thin", "normal", "thick"]
    liquidity_score: float      # 0.0 to 1.0

    # Transition probabilities
    regime_stability: float     # Prob of staying in current regime
    transition_probs: Dict[int, float]  # Prob of transitioning to each regime
```

### 3.4 Regime Transition Handling

```yaml
regime_transition_policy:
  # Minimum bars before allowing regime switch
  min_regime_duration: 20  # bars (1m bars = 20 minutes)

  # Confidence threshold for regime switch
  switch_confidence_threshold: 0.7

  # Smoothing: EMA of regime probabilities
  prob_smoothing_alpha: 0.3

  # Hysteresis: extra confidence needed to switch back
  hysteresis_penalty: 0.1

  # Actions on regime change
  on_regime_change:
    - recompute_strategy_weights
    - check_position_limits
    - log_regime_transition
    - notify_monitor
```

### 3.5 Regime Definitions

| Regime ID | Vol | Trend | Liquidity | Typical Market Conditions |
|-----------|-----|-------|-----------|---------------------------|
| 0 | Low | Ranging | Normal | Quiet consolidation |
| 1 | Low | Trending | Normal | Steady trend |
| 2 | Medium | Ranging | Normal | Choppy, uncertain |
| 3 | Medium | Trending | Normal | Strong directional move |
| 4 | High | Reverting | Thin | Crisis, mean-reversion opportunities |
| 5 | High | Trending | Thick | Breakout with volume |

---

## 4. Capital Allocator Design

### 4.1 Allocation Algorithm

#### 4.1.1 Core Formula

```
W(t) = f(Regime(t), Performance(t), Risk(t))

Where:
- W(t) = vector of strategy weights at time t
- Regime(t) = current regime classification
- Performance(t) = recent strategy performance metrics
- Risk(t) = current risk state (drawdown, exposure)
```

#### 4.1.2 Weight Calculation

```python
def compute_weights(
    regime: RegimeOutput,
    strategy_metrics: Dict[str, StrategyMetrics],
    risk_state: RiskState,
    config: AllocatorConfig
) -> Dict[str, float]:
    """
    Compute strategy allocation weights.

    Steps:
    1. Base weights from regime affinity
    2. Adjust by recent performance (Sharpe, drawdown)
    3. Apply risk constraints
    4. Normalize to sum = 1.0
    """

    weights = {}

    for strategy_id, metrics in strategy_metrics.items():
        # Step 1: Regime affinity weight
        regime_weight = get_regime_affinity(strategy_id, regime)

        # Step 2: Performance adjustment
        sharpe_mult = sharpe_multiplier(metrics.rolling_sharpe_5d)
        dd_mult = drawdown_multiplier(metrics.current_drawdown)
        perf_weight = regime_weight * sharpe_mult * dd_mult

        # Step 3: Risk constraint
        if risk_state.portfolio_drawdown > config.dd_reduce_threshold:
            perf_weight *= config.dd_reduction_factor

        weights[strategy_id] = perf_weight

    # Step 4: Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights
```

### 4.2 Regime x Performance Weighting

#### 4.2.1 Regime Affinity Matrix

| Strategy Family | Low Vol + Ranging | Low Vol + Trending | High Vol + Ranging | High Vol + Trending |
|-----------------|-------------------|--------------------|--------------------|---------------------|
| TREND | 0.3 | 1.5 | 0.4 | 1.2 |
| MREV | 1.4 | 0.3 | 1.0 | 0.2 |
| MICRO | 1.2 | 1.0 | 0.5 | 0.6 |
| MLPRED | 1.0 | 1.0 | 0.8 | 0.8 |

#### 4.2.2 Performance Multipliers

```python
def sharpe_multiplier(rolling_sharpe: float) -> float:
    """
    Multiplier based on recent Sharpe ratio.

    Sharpe < 0.0: 0.2x (reduce allocation)
    Sharpe 0.0-0.5: 0.5x
    Sharpe 0.5-1.0: 1.0x (base)
    Sharpe 1.0-1.5: 1.3x (increase allocation)
    Sharpe > 1.5: 1.5x (cap)
    """
    if rolling_sharpe < 0.0:
        return 0.2
    elif rolling_sharpe < 0.5:
        return 0.5
    elif rolling_sharpe < 1.0:
        return 1.0
    elif rolling_sharpe < 1.5:
        return 1.3
    else:
        return 1.5

def drawdown_multiplier(current_dd: float) -> float:
    """
    Multiplier based on current strategy drawdown.

    DD < 3%: 1.0x
    DD 3-6%: 0.7x
    DD 6-10%: 0.4x
    DD > 10%: 0.1x (near-pause)
    """
    if current_dd < 0.03:
        return 1.0
    elif current_dd < 0.06:
        return 0.7
    elif current_dd < 0.10:
        return 0.4
    else:
        return 0.1
```

### 4.3 Rebalancing Frequency

```yaml
rebalancing_policy:
  # Time-based rebalancing
  scheduled_interval: 15m  # Check every 15 minutes

  # Event-driven rebalancing triggers
  event_triggers:
    - regime_change
    - strategy_promotion
    - strategy_rollback
    - drawdown_breach
    - significant_pnl_change  # |daily_pnl| > 2%

  # Rate limiting
  min_rebalance_interval: 5m  # Minimum time between rebalances
  max_daily_rebalances: 20

  # Turnover constraints
  max_weight_change_per_rebalance: 0.3  # Max 30% weight shift
  min_weight_change_threshold: 0.02     # Ignore changes < 2%
```

### 4.4 Drawdown-Aware Allocation

```python
@dataclass
class RiskState:
    """Current risk state for allocation decisions."""

    # Portfolio level
    portfolio_drawdown: float
    portfolio_var_95: float
    daily_pnl: float

    # Strategy level
    strategy_drawdowns: Dict[str, float]
    strategy_var_95: Dict[str, float]

    # Limits
    max_portfolio_drawdown: float = 0.12
    max_strategy_drawdown: float = 0.08
    daily_loss_limit: float = 0.04

class DrawdownAwareAllocator:
    """
    Reduces allocation as drawdown approaches limits.
    """

    def __init__(self, config: AllocatorConfig):
        self.config = config

    def apply_drawdown_constraints(
        self,
        weights: Dict[str, float],
        risk_state: RiskState
    ) -> Dict[str, float]:
        """
        Apply drawdown-based constraints to weights.
        """

        adjusted = {}

        for strategy_id, weight in weights.items():
            strat_dd = risk_state.strategy_drawdowns.get(strategy_id, 0.0)

            # Strategy-level drawdown reduction
            if strat_dd > self.config.strategy_dd_warn:
                reduction = self._compute_reduction(
                    strat_dd,
                    self.config.strategy_dd_warn,
                    self.config.strategy_dd_max
                )
                weight *= (1 - reduction)

            adjusted[strategy_id] = weight

        # Portfolio-level drawdown reduction
        if risk_state.portfolio_drawdown > self.config.portfolio_dd_warn:
            global_reduction = self._compute_reduction(
                risk_state.portfolio_drawdown,
                self.config.portfolio_dd_warn,
                self.config.portfolio_dd_max
            )
            adjusted = {k: v * (1 - global_reduction) for k, v in adjusted.items()}

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _compute_reduction(
        self,
        current: float,
        warn_level: float,
        max_level: float
    ) -> float:
        """Linear reduction from warn to max."""
        if current >= max_level:
            return 0.9  # Near-full reduction
        return (current - warn_level) / (max_level - warn_level) * 0.9
```

### 4.5 Allocator Output Schema

```python
@dataclass
class AllocationOutput:
    """Output from capital allocator."""

    timestamp: datetime

    # Target weights
    strategy_weights: Dict[str, float]  # Sum = 1.0
    cash_weight: float                   # Reserved cash

    # Position targets (in notional)
    strategy_notionals: Dict[str, float]
    total_allocated_notional: float

    # Rebalancing info
    weight_changes: Dict[str, float]     # Delta from previous
    rebalance_trigger: str               # Why we rebalanced

    # Risk metrics at allocation time
    portfolio_var_95: float
    expected_portfolio_sharpe: float
    regime_at_allocation: int
```

---

## 5. Scalable Pipeline Design

### 5.1 Rust Ingestor Architecture

```
                        +------------------+
                        |   Alpaca WS      |
                        |   (SIP/IEX)      |
                        +--------+---------+
                                 |
                                 v
                    +------------+------------+
                    |   Rust Tick Ingestor   |
                    |   - Zero-copy parsing  |
                    |   - Ring buffer        |
                    |   - Batch aggregation  |
                    +------------+------------+
                                 |
            +--------------------+--------------------+
            |                    |                    |
            v                    v                    v
    +-------+------+    +-------+------+    +--------+------+
    | Parquet      |    | Feature      |    | Real-time     |
    | Writer       |    | Publisher    |    | Signal Feed   |
    | (Arrow IPC)  |    | (ZeroMQ)     |    | (WebSocket)   |
    +--------------+    +--------------+    +---------------+
```

### 5.2 Rust Ingestor Service Specification

```rust
// services/ingestor_rs/src/main.rs

/// High-performance tick ingestor for Alpaca streams.
///
/// Performance targets:
/// - Throughput: 100,000+ ticks/sec sustained
/// - Latency: < 1ms from receipt to publish
/// - Memory: < 500MB for 1-hour buffer

pub struct TickIngestor {
    /// WebSocket connection to Alpaca
    ws_client: AlpacaWebSocket,

    /// Ring buffer for incoming ticks
    tick_buffer: RingBuffer<RawTick>,

    /// Arrow record batch builder
    batch_builder: ArrowBatchBuilder,

    /// Output channels
    parquet_writer: ParquetWriter,
    feature_publisher: ZmqPublisher,
    signal_feed: WebSocketServer,

    /// Configuration
    config: IngestorConfig,
}

#[derive(Debug, Clone)]
pub struct IngestorConfig {
    /// Symbols to subscribe
    pub symbols: Vec<String>,

    /// Batch size for Parquet writes
    pub batch_size: usize,  // Default: 10,000

    /// Flush interval (ms)
    pub flush_interval_ms: u64,  // Default: 1000

    /// Ring buffer capacity
    pub buffer_capacity: usize,  // Default: 1_000_000

    /// Compression for Parquet
    pub compression: Compression,  // Default: Zstd(3)
}
```

### 5.3 Async Feature Computation

```python
# services/feature_builder_py/async_builder.py

import asyncio
from typing import AsyncIterator
import pyarrow as pa
import zmq.asyncio

class AsyncFeatureBuilder:
    """
    Async feature computation with backpressure handling.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.context = zmq.asyncio.Context()
        self.tick_subscriber = None
        self.feature_publisher = None

    async def start(self):
        """Start async feature computation pipeline."""

        # Subscribe to tick feed from Rust ingestor
        self.tick_subscriber = self.context.socket(zmq.SUB)
        self.tick_subscriber.connect(self.config.tick_feed_url)
        self.tick_subscriber.subscribe(b"")

        # Publish computed features
        self.feature_publisher = self.context.socket(zmq.PUB)
        self.feature_publisher.bind(self.config.feature_feed_url)

        # Start computation loop
        await self._computation_loop()

    async def _computation_loop(self):
        """Main computation loop with batching."""

        batch = []
        batch_start = asyncio.get_event_loop().time()

        while True:
            try:
                # Receive with timeout
                msg = await asyncio.wait_for(
                    self.tick_subscriber.recv(),
                    timeout=0.1
                )

                # Parse Arrow IPC message
                tick_batch = pa.ipc.read_record_batch(
                    pa.BufferReader(msg)
                )
                batch.append(tick_batch)

                # Flush if batch size reached or timeout
                now = asyncio.get_event_loop().time()
                if (len(batch) >= self.config.batch_size or
                    now - batch_start > self.config.batch_timeout_sec):

                    features = await self._compute_features(batch)
                    await self._publish_features(features)

                    batch = []
                    batch_start = now

            except asyncio.TimeoutError:
                # Flush partial batch on timeout
                if batch:
                    features = await self._compute_features(batch)
                    await self._publish_features(features)
                    batch = []
                    batch_start = asyncio.get_event_loop().time()

    async def _compute_features(
        self,
        batches: list[pa.RecordBatch]
    ) -> pa.RecordBatch:
        """
        Compute features from tick batches.
        Runs CPU-bound work in executor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Default executor
            self._compute_features_sync,
            batches
        )
```

### 5.4 Optional GPU Acceleration Path (RAPIDS/cuDF)

```python
# services/feature_builder_py/gpu_features.py

from typing import Optional
import pyarrow as pa

# GPU acceleration is optional - graceful fallback to CPU
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUFeatureBuilder:
    """
    GPU-accelerated feature computation using RAPIDS cuDF.

    Only used when:
    1. GPU_FEATURE_ACCELERATION=true in config
    2. RAPIDS is installed
    3. CUDA device is available
    4. Batch size > 100,000 (threshold for GPU benefit)
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.use_gpu = (
            GPU_AVAILABLE and
            config.gpu_feature_acceleration and
            self._check_cuda_device()
        )

    def compute_features(
        self,
        ticks: pa.Table
    ) -> pa.Table:
        """
        Compute features, using GPU if available and beneficial.
        """

        if self.use_gpu and len(ticks) > self.config.gpu_batch_threshold:
            return self._compute_gpu(ticks)
        else:
            return self._compute_cpu(ticks)

    def _compute_gpu(self, ticks: pa.Table) -> pa.Table:
        """GPU-accelerated feature computation."""

        # Convert to cuDF
        gdf = cudf.DataFrame.from_arrow(ticks)

        # GPU vectorized operations
        gdf['spread'] = gdf['ask_price'] - gdf['bid_price']
        gdf['midprice'] = (gdf['ask_price'] + gdf['bid_price']) / 2
        gdf['microprice'] = (
            gdf['bid_price'] * gdf['ask_size'] +
            gdf['ask_price'] * gdf['bid_size']
        ) / (gdf['bid_size'] + gdf['ask_size'])

        # Rolling computations on GPU
        gdf['vol_30s'] = gdf.groupby('symbol')['midprice'].transform(
            lambda x: x.pct_change().rolling(30).std()
        )

        # Convert back to Arrow
        return gdf.to_arrow()

    def _compute_cpu(self, ticks: pa.Table) -> pa.Table:
        """CPU fallback feature computation."""
        # Standard pandas/numpy implementation
        pass
```

### 5.5 Data Flow Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Alpaca SIP      |     |   Rust Ingestor   |     |   Parquet Lake    |
|   WebSocket       | --> |   (Zero-copy)     | --> |   (Immutable)     |
+-------------------+     +-------------------+     +-------------------+
                                   |
                                   v (ZeroMQ IPC)
                          +-------------------+
                          |  Feature Builder  |
                          |  (Async Python)   |
                          +-------------------+
                                   |
                    +--------------+--------------+
                    |              |              |
                    v              v              v
            +-------+---+   +------+----+   +----+------+
            | Feature   |   | Signal    |   | Monitor   |
            | Store     |   | Router    |   | Feed      |
            | (Parquet) |   | (ZeroMQ)  |   | (Metrics) |
            +-----------+   +-----------+   +-----------+
```

---

## 6. Sprint Plan

### 6.1 Overview Timeline

| Sprint | Duration | Focus Area | Key Deliverables |
|--------|----------|------------|------------------|
| 1 | Weeks 1-2 | Multi-Strategy Framework | Strategy orchestrator, lifecycle management |
| 2 | Weeks 3-4 | Strategy Families | TREND, MREV, MICRO, MLPRED implementations |
| 3 | Weeks 5-6 | Regime Classifier | HMM/ML classifier, feature extraction |
| 4 | Weeks 7-8 | Capital Allocator | Allocation engine, rebalancing |
| 5 | Weeks 9-10 | Rust Ingestor | High-performance tick ingestion |
| 6 | Weeks 11-12 | Anti-Overfit Governance | Purged CV, deflated Sharpe, stability |
| 7 | Weeks 13-14 | Live Experimentation | Shadow, canary, auto-rollback |
| 8 | Weeks 15-16 | Operational Maturity | K8s, DR, runbooks |
| 9 | Weeks 17-18 | Production Hardening | Integration testing, performance tuning |

---

### 6.2 Sprint 1: Multi-Strategy Framework (Weeks 1-2)

**Goal:** Build the foundation for concurrent strategy execution.

#### Tasks

- [ ] **S1-001**: Design strategy orchestrator architecture
- [ ] **S1-002**: Implement strategy family registry schema
- [ ] **S1-003**: Build concurrent strategy runner framework
- [ ] **S1-004**: Create signal aggregator service
- [ ] **S1-005**: Implement strategy lifecycle state machine
- [ ] **S1-006**: Build strategy versioning system
- [ ] **S1-007**: Create strategy configuration loader
- [ ] **S1-008**: Implement inter-strategy isolation
- [ ] **S1-009**: Add strategy-level metrics collection
- [ ] **S1-010**: Write unit tests for orchestrator

#### Acceptance Criteria

- [ ] Orchestrator can spawn 4+ strategy runners concurrently
- [ ] Strategy lifecycle transitions work correctly
- [ ] Signal aggregation produces combined output
- [ ] No cross-strategy state leakage
- [ ] Unit test coverage >= 80%

---

### 6.3 Sprint 2: Strategy Families Implementation (Weeks 3-4)

**Goal:** Implement all four strategy families.

#### Tasks

- [ ] **S2-001**: Implement TREND family base class
- [ ] **S2-002**: Build trend_momentum_5m variant
- [ ] **S2-003**: Build trend_breakout_15m variant
- [ ] **S2-004**: Implement MREV family base class
- [ ] **S2-005**: Build mrev_zscore_1m variant
- [ ] **S2-006**: Build mrev_bollinger_5m variant
- [ ] **S2-007**: Implement MICRO family base class
- [ ] **S2-008**: Build micro_imbalance_5s variant
- [ ] **S2-009**: Implement MLPRED family base class
- [ ] **S2-010**: Build mlpred_xgb_1m with GPU training
- [ ] **S2-011**: Create family-specific backtests
- [ ] **S2-012**: Validate all families against evaluation protocol

#### Acceptance Criteria

- [ ] All 4 families implemented with >= 2 variants each
- [ ] Each variant passes backtest acceptance criteria
- [ ] GPU training works for MLPRED family
- [ ] CPU fallback verified
- [ ] Documentation complete for each family

---

### 6.4 Sprint 3: Regime Classifier (Weeks 5-6)

**Goal:** Build and validate regime classification system.

#### Tasks

- [ ] **S3-001**: Design regime feature extraction pipeline
- [ ] **S3-002**: Implement volatility regime features
- [ ] **S3-003**: Implement trend regime features
- [ ] **S3-004**: Implement liquidity regime features
- [ ] **S3-005**: Build HMM regime classifier
- [ ] **S3-006**: Build ML (XGBoost) regime classifier
- [ ] **S3-007**: Implement regime output schema
- [ ] **S3-008**: Build regime transition handler
- [ ] **S3-009**: Create regime classifier backtester
- [ ] **S3-010**: Train and validate on historical data
- [ ] **S3-011**: Implement regime monitoring dashboard
- [ ] **S3-012**: Write regime classifier tests

#### Acceptance Criteria

- [ ] Regime classifier accuracy >= 75% out-of-sample
- [ ] Regime transitions are smooth (no rapid flipping)
- [ ] All 6 regimes are distinguishable
- [ ] Classifier latency < 10ms
- [ ] Historical regime labels validated by manual review

---

### 6.5 Sprint 4: Capital Allocator (Weeks 7-8)

**Goal:** Implement dynamic capital allocation engine.

#### Tasks

- [ ] **S4-001**: Design allocator architecture
- [ ] **S4-002**: Implement regime affinity matrix
- [ ] **S4-003**: Build performance multiplier functions
- [ ] **S4-004**: Implement drawdown-aware constraints
- [ ] **S4-005**: Build rebalancing engine
- [ ] **S4-006**: Create allocation output schema
- [ ] **S4-007**: Implement turnover constraints
- [ ] **S4-008**: Build allocation backtester
- [ ] **S4-009**: Validate allocator vs single-strategy baseline
- [ ] **S4-010**: Implement allocation audit logging
- [ ] **S4-011**: Create allocator configuration system
- [ ] **S4-012**: Write allocator unit and integration tests

#### Acceptance Criteria

- [ ] Allocator Sharpe >= 90% of best single strategy
- [ ] Allocator MDD <= best MDD + 2%
- [ ] Turnover <= 1.25x average constituent turnover
- [ ] Weight stability: avg daily L1 change <= 0.25
- [ ] All allocation decisions are auditable

---

### 6.6 Sprint 5: Rust Tick Ingestor (Weeks 9-10)

**Goal:** Build high-performance Rust tick ingestion service.

#### Tasks

- [ ] **S5-001**: Set up Rust project structure
- [ ] **S5-002**: Implement Alpaca WebSocket client
- [ ] **S5-003**: Build zero-copy tick parser
- [ ] **S5-004**: Implement ring buffer for ticks
- [ ] **S5-005**: Build Arrow record batch builder
- [ ] **S5-006**: Implement Parquet writer with compression
- [ ] **S5-007**: Build ZeroMQ publisher for features
- [ ] **S5-008**: Implement real-time WebSocket feed
- [ ] **S5-009**: Add configuration management
- [ ] **S5-010**: Build health check endpoint
- [ ] **S5-011**: Performance benchmark (target: 100k ticks/sec)
- [ ] **S5-012**: Integration test with Python feature builder

#### Acceptance Criteria

- [ ] Throughput >= 100,000 ticks/sec sustained
- [ ] Latency < 1ms (receipt to publish)
- [ ] Memory < 500MB for 1-hour buffer
- [ ] Zero data loss under normal operation
- [ ] Clean shutdown with data flush

---

### 6.7 Sprint 6: Anti-Overfit Governance (Weeks 11-12)

**Goal:** Implement rigorous anti-overfit framework.

#### Tasks

- [ ] **S6-001**: Implement purged k-fold cross-validation
- [ ] **S6-002**: Build gap-aware train/test splitting
- [ ] **S6-003**: Implement deflated Sharpe ratio calculation
- [ ] **S6-004**: Build cost sensitivity tests
- [ ] **S6-005**: Implement latency sensitivity tests
- [ ] **S6-006**: Build universe subset stability tests
- [ ] **S6-007**: Create promotion gate validator
- [ ] **S6-008**: Implement incumbent comparison logic
- [ ] **S6-009**: Build overfit detection alerts
- [ ] **S6-010**: Create governance audit reports
- [ ] **S6-011**: Document governance framework
- [ ] **S6-012**: Write governance test suite

#### Acceptance Criteria

- [ ] Purged CV prevents all lookahead leakage
- [ ] Deflated Sharpe implemented per Lopez de Prado
- [ ] Cost sensitivity: Sharpe >= 0.4 at 1.5x costs
- [ ] Parameter sensitivity: Sharpe drop <= 35%
- [ ] 90%+ of promoted strategies pass all governance gates

---

### 6.8 Sprint 7: Live Experimentation (Weeks 13-14)

**Goal:** Build shadow and canary deployment framework.

#### Tasks

- [ ] **S7-001**: Design experimentation architecture
- [ ] **S7-002**: Build always-on shadow runner
- [ ] **S7-003**: Implement shadow signal comparison
- [ ] **S7-004**: Build canary deployment manager
- [ ] **S7-005**: Implement gradual allocation ramping
- [ ] **S7-006**: Build auto-rollback triggers
- [ ] **S7-007**: Implement drawdown-based rollback
- [ ] **S7-008**: Implement slippage anomaly detection
- [ ] **S7-009**: Build order reject monitoring
- [ ] **S7-010**: Implement feed instability detection
- [ ] **S7-011**: Create experiment tracking dashboard
- [ ] **S7-012**: Write experimentation tests

#### Acceptance Criteria

- [ ] Shadow runs continuously for all candidates
- [ ] Canary allocation: 5-10% of target
- [ ] Auto-rollback latency < 500ms
- [ ] All rollback triggers working correctly
- [ ] Experiment metrics fully tracked

---

### 6.9 Sprint 8: Operational Maturity (Weeks 15-16)

**Goal:** Production-grade deployment and operations.

#### Tasks

- [ ] **S8-001**: Design Kubernetes deployment architecture
- [ ] **S8-002**: Create Helm charts for all services
- [ ] **S8-003**: Implement secrets management (Vault/K8s secrets)
- [ ] **S8-004**: Build disaster recovery procedures
- [ ] **S8-005**: Implement replayable event logs
- [ ] **S8-006**: Create automated runbooks
- [ ] **S8-007**: Build service health dashboards
- [ ] **S8-008**: Implement alerting rules
- [ ] **S8-009**: Create incident response procedures
- [ ] **S8-010**: Build backup and restore procedures
- [ ] **S8-011**: Document operational procedures
- [ ] **S8-012**: Conduct disaster recovery drill

#### Acceptance Criteria

- [ ] All services deployable via Helm
- [ ] Secrets never in plaintext
- [ ] RTO < 30 minutes for full recovery
- [ ] Event logs replayable for debugging
- [ ] Runbooks cover all common scenarios

---

### 6.10 Sprint 9: Production Hardening (Weeks 17-18)

**Goal:** Final integration testing and performance optimization.

#### Tasks

- [ ] **S9-001**: Full system integration testing
- [ ] **S9-002**: End-to-end latency optimization
- [ ] **S9-003**: Memory usage optimization
- [ ] **S9-004**: Load testing (2x expected volume)
- [ ] **S9-005**: Chaos engineering tests
- [ ] **S9-006**: Security audit
- [ ] **S9-007**: Performance regression testing
- [ ] **S9-008**: Documentation review and update
- [ ] **S9-009**: Stakeholder demo and review
- [ ] **S9-010**: Production deployment checklist
- [ ] **S9-011**: Go-live readiness assessment
- [ ] **S9-012**: Phase 3 completion sign-off

#### Acceptance Criteria

- [ ] All integration tests passing
- [ ] Latency targets met under load
- [ ] No memory leaks in 24-hour test
- [ ] System survives chaos tests
- [ ] Security audit passed
- [ ] Production deployment checklist complete

---

## 7. Task Tracking Table

### 7.1 Full Task List

| Task ID | Description | Component | Sprint | Status | Dependencies | Estimate |
|---------|-------------|-----------|--------|--------|--------------|----------|
| S1-001 | Design strategy orchestrator architecture | orchestrator | 1 | [ ] Pending | - | 2d |
| S1-002 | Implement strategy family registry schema | registry | 1 | [ ] Pending | S1-001 | 1d |
| S1-003 | Build concurrent strategy runner framework | runner | 1 | [ ] Pending | S1-001 | 3d |
| S1-004 | Create signal aggregator service | aggregator | 1 | [ ] Pending | S1-003 | 2d |
| S1-005 | Implement strategy lifecycle state machine | orchestrator | 1 | [ ] Pending | S1-001 | 2d |
| S1-006 | Build strategy versioning system | registry | 1 | [ ] Pending | S1-002 | 1d |
| S1-007 | Create strategy configuration loader | config | 1 | [ ] Pending | S1-002 | 1d |
| S1-008 | Implement inter-strategy isolation | runner | 1 | [ ] Pending | S1-003 | 2d |
| S1-009 | Add strategy-level metrics collection | monitor | 1 | [ ] Pending | S1-003 | 1d |
| S1-010 | Write unit tests for orchestrator | tests | 1 | [ ] Pending | S1-003 | 2d |
| S2-001 | Implement TREND family base class | strategies | 2 | [ ] Pending | S1-003 | 2d |
| S2-002 | Build trend_momentum_5m variant | strategies | 2 | [ ] Pending | S2-001 | 1d |
| S2-003 | Build trend_breakout_15m variant | strategies | 2 | [ ] Pending | S2-001 | 1d |
| S2-004 | Implement MREV family base class | strategies | 2 | [ ] Pending | S1-003 | 2d |
| S2-005 | Build mrev_zscore_1m variant | strategies | 2 | [ ] Pending | S2-004 | 1d |
| S2-006 | Build mrev_bollinger_5m variant | strategies | 2 | [ ] Pending | S2-004 | 1d |
| S2-007 | Implement MICRO family base class | strategies | 2 | [ ] Pending | S1-003 | 2d |
| S2-008 | Build micro_imbalance_5s variant | strategies | 2 | [ ] Pending | S2-007 | 1d |
| S2-009 | Implement MLPRED family base class | strategies | 2 | [ ] Pending | S1-003 | 2d |
| S2-010 | Build mlpred_xgb_1m with GPU training | strategies | 2 | [ ] Pending | S2-009 | 2d |
| S2-011 | Create family-specific backtests | backtester | 2 | [ ] Pending | S2-001, S2-004, S2-007, S2-009 | 2d |
| S2-012 | Validate all families against eval protocol | backtester | 2 | [ ] Pending | S2-011 | 1d |
| S3-001 | Design regime feature extraction pipeline | regime | 3 | [ ] Pending | - | 1d |
| S3-002 | Implement volatility regime features | regime | 3 | [ ] Pending | S3-001 | 1d |
| S3-003 | Implement trend regime features | regime | 3 | [ ] Pending | S3-001 | 1d |
| S3-004 | Implement liquidity regime features | regime | 3 | [ ] Pending | S3-001 | 1d |
| S3-005 | Build HMM regime classifier | regime | 3 | [ ] Pending | S3-002, S3-003, S3-004 | 3d |
| S3-006 | Build ML (XGBoost) regime classifier | regime | 3 | [ ] Pending | S3-002, S3-003, S3-004 | 2d |
| S3-007 | Implement regime output schema | regime | 3 | [ ] Pending | S3-005 | 1d |
| S3-008 | Build regime transition handler | regime | 3 | [ ] Pending | S3-007 | 2d |
| S3-009 | Create regime classifier backtester | regime | 3 | [ ] Pending | S3-005, S3-006 | 2d |
| S3-010 | Train and validate on historical data | regime | 3 | [ ] Pending | S3-009 | 2d |
| S3-011 | Implement regime monitoring dashboard | monitor | 3 | [ ] Pending | S3-007 | 1d |
| S3-012 | Write regime classifier tests | tests | 3 | [ ] Pending | S3-005, S3-006 | 1d |
| S4-001 | Design allocator architecture | allocator | 4 | [ ] Pending | S3-007 | 2d |
| S4-002 | Implement regime affinity matrix | allocator | 4 | [ ] Pending | S4-001 | 1d |
| S4-003 | Build performance multiplier functions | allocator | 4 | [ ] Pending | S4-001 | 1d |
| S4-004 | Implement drawdown-aware constraints | allocator | 4 | [ ] Pending | S4-001 | 2d |
| S4-005 | Build rebalancing engine | allocator | 4 | [ ] Pending | S4-002, S4-003, S4-004 | 2d |
| S4-006 | Create allocation output schema | allocator | 4 | [ ] Pending | S4-001 | 1d |
| S4-007 | Implement turnover constraints | allocator | 4 | [ ] Pending | S4-005 | 1d |
| S4-008 | Build allocation backtester | allocator | 4 | [ ] Pending | S4-005 | 2d |
| S4-009 | Validate allocator vs single-strategy baseline | allocator | 4 | [ ] Pending | S4-008 | 2d |
| S4-010 | Implement allocation audit logging | allocator | 4 | [ ] Pending | S4-005 | 1d |
| S4-011 | Create allocator configuration system | allocator | 4 | [ ] Pending | S4-001 | 1d |
| S4-012 | Write allocator unit and integration tests | tests | 4 | [ ] Pending | S4-005 | 2d |
| S5-001 | Set up Rust project structure | ingestor_rs | 5 | [ ] Pending | - | 1d |
| S5-002 | Implement Alpaca WebSocket client | ingestor_rs | 5 | [ ] Pending | S5-001 | 2d |
| S5-003 | Build zero-copy tick parser | ingestor_rs | 5 | [ ] Pending | S5-002 | 2d |
| S5-004 | Implement ring buffer for ticks | ingestor_rs | 5 | [ ] Pending | S5-001 | 2d |
| S5-005 | Build Arrow record batch builder | ingestor_rs | 5 | [ ] Pending | S5-003 | 2d |
| S5-006 | Implement Parquet writer with compression | ingestor_rs | 5 | [ ] Pending | S5-005 | 2d |
| S5-007 | Build ZeroMQ publisher for features | ingestor_rs | 5 | [ ] Pending | S5-005 | 1d |
| S5-008 | Implement real-time WebSocket feed | ingestor_rs | 5 | [ ] Pending | S5-003 | 2d |
| S5-009 | Add configuration management | ingestor_rs | 5 | [ ] Pending | S5-001 | 1d |
| S5-010 | Build health check endpoint | ingestor_rs | 5 | [ ] Pending | S5-001 | 1d |
| S5-011 | Performance benchmark (100k ticks/sec) | ingestor_rs | 5 | [ ] Pending | S5-006 | 2d |
| S5-012 | Integration test with Python feature builder | tests | 5 | [ ] Pending | S5-007 | 2d |
| S6-001 | Implement purged k-fold cross-validation | governance | 6 | [ ] Pending | - | 2d |
| S6-002 | Build gap-aware train/test splitting | governance | 6 | [ ] Pending | S6-001 | 1d |
| S6-003 | Implement deflated Sharpe ratio calculation | governance | 6 | [ ] Pending | - | 2d |
| S6-004 | Build cost sensitivity tests | governance | 6 | [ ] Pending | - | 1d |
| S6-005 | Implement latency sensitivity tests | governance | 6 | [ ] Pending | - | 1d |
| S6-006 | Build universe subset stability tests | governance | 6 | [ ] Pending | - | 2d |
| S6-007 | Create promotion gate validator | governance | 6 | [ ] Pending | S6-001, S6-003 | 2d |
| S6-008 | Implement incumbent comparison logic | governance | 6 | [ ] Pending | S6-007 | 1d |
| S6-009 | Build overfit detection alerts | governance | 6 | [ ] Pending | S6-007 | 1d |
| S6-010 | Create governance audit reports | governance | 6 | [ ] Pending | S6-007 | 2d |
| S6-011 | Document governance framework | docs | 6 | [ ] Pending | S6-007 | 1d |
| S6-012 | Write governance test suite | tests | 6 | [ ] Pending | S6-007 | 2d |
| S7-001 | Design experimentation architecture | experiment | 7 | [ ] Pending | - | 1d |
| S7-002 | Build always-on shadow runner | experiment | 7 | [ ] Pending | S7-001 | 2d |
| S7-003 | Implement shadow signal comparison | experiment | 7 | [ ] Pending | S7-002 | 2d |
| S7-004 | Build canary deployment manager | experiment | 7 | [ ] Pending | S7-001 | 2d |
| S7-005 | Implement gradual allocation ramping | experiment | 7 | [ ] Pending | S7-004 | 1d |
| S7-006 | Build auto-rollback triggers | experiment | 7 | [ ] Pending | S7-004 | 2d |
| S7-007 | Implement drawdown-based rollback | experiment | 7 | [ ] Pending | S7-006 | 1d |
| S7-008 | Implement slippage anomaly detection | experiment | 7 | [ ] Pending | S7-006 | 1d |
| S7-009 | Build order reject monitoring | experiment | 7 | [ ] Pending | S7-006 | 1d |
| S7-010 | Implement feed instability detection | experiment | 7 | [ ] Pending | S7-006 | 1d |
| S7-011 | Create experiment tracking dashboard | monitor | 7 | [ ] Pending | S7-002, S7-004 | 2d |
| S7-012 | Write experimentation tests | tests | 7 | [ ] Pending | S7-006 | 2d |
| S8-001 | Design Kubernetes deployment architecture | infra | 8 | [ ] Pending | - | 2d |
| S8-002 | Create Helm charts for all services | infra | 8 | [ ] Pending | S8-001 | 3d |
| S8-003 | Implement secrets management | infra | 8 | [ ] Pending | S8-001 | 2d |
| S8-004 | Build disaster recovery procedures | infra | 8 | [ ] Pending | S8-001 | 2d |
| S8-005 | Implement replayable event logs | infra | 8 | [ ] Pending | - | 2d |
| S8-006 | Create automated runbooks | docs | 8 | [ ] Pending | S8-001 | 2d |
| S8-007 | Build service health dashboards | monitor | 8 | [ ] Pending | S8-001 | 2d |
| S8-008 | Implement alerting rules | monitor | 8 | [ ] Pending | S8-007 | 1d |
| S8-009 | Create incident response procedures | docs | 8 | [ ] Pending | S8-006 | 1d |
| S8-010 | Build backup and restore procedures | infra | 8 | [ ] Pending | S8-004 | 1d |
| S8-011 | Document operational procedures | docs | 8 | [ ] Pending | S8-006 | 1d |
| S8-012 | Conduct disaster recovery drill | infra | 8 | [ ] Pending | S8-004, S8-010 | 1d |
| S9-001 | Full system integration testing | tests | 9 | [ ] Pending | All prior | 3d |
| S9-002 | End-to-end latency optimization | performance | 9 | [ ] Pending | S9-001 | 2d |
| S9-003 | Memory usage optimization | performance | 9 | [ ] Pending | S9-001 | 2d |
| S9-004 | Load testing (2x expected volume) | tests | 9 | [ ] Pending | S9-001 | 2d |
| S9-005 | Chaos engineering tests | tests | 9 | [ ] Pending | S9-001 | 2d |
| S9-006 | Security audit | security | 9 | [ ] Pending | S9-001 | 2d |
| S9-007 | Performance regression testing | tests | 9 | [ ] Pending | S9-002, S9-003 | 1d |
| S9-008 | Documentation review and update | docs | 9 | [ ] Pending | S9-001 | 2d |
| S9-009 | Stakeholder demo and review | milestone | 9 | [ ] Pending | S9-001 | 1d |
| S9-010 | Production deployment checklist | infra | 9 | [ ] Pending | S9-001 | 1d |
| S9-011 | Go-live readiness assessment | milestone | 9 | [ ] Pending | S9-009 | 1d |
| S9-012 | Phase 3 completion sign-off | milestone | 9 | [ ] Pending | S9-011 | 0.5d |

### 7.2 Task Summary by Component

| Component | Task Count | Total Estimate (days) |
|-----------|------------|----------------------|
| orchestrator | 3 | 6 |
| registry | 2 | 2 |
| runner | 2 | 5 |
| aggregator | 1 | 2 |
| config | 1 | 1 |
| monitor | 5 | 7 |
| strategies | 10 | 15 |
| backtester | 3 | 5 |
| regime | 11 | 17 |
| allocator | 11 | 17 |
| ingestor_rs | 12 | 20 |
| governance | 11 | 17 |
| experiment | 11 | 17 |
| infra | 8 | 14 |
| tests | 10 | 18 |
| docs | 4 | 5 |
| performance | 3 | 5 |
| security | 1 | 2 |
| milestone | 3 | 2.5 |
| **TOTAL** | **108** | **~168 days** |

---

## 8. Integration Points

### 8.1 Phase 1 Dependencies

| Phase 3 Component | Phase 1 Dependency | Integration Method |
|-------------------|-------------------|-------------------|
| Strategy Families | `backtester_py` | Backtest execution, metrics |
| Strategy Families | `feature_builder_py` | Feature schema, decision frame |
| Strategy Families | `labeler_py` | Label schema, horizons |
| Regime Classifier | `feature_builder_py` | Volatility, trend features |
| Capital Allocator | `backtester_py` | Historical performance data |
| All | `registry_api_py` | Strategy registration, audit |
| All | `monitor_py` | Metrics, alerts, dashboards |

### 8.2 Phase 2 Dependencies

| Phase 3 Component | Phase 2 Dependency | Integration Method |
|-------------------|-------------------|-------------------|
| Strategy Runner | `execution_engine_rs` | Order submission, fills |
| Auto-Rollback | `execution_engine_rs` | Position management |
| Canary Deployment | `control_plane` | Allocation updates |
| Live Experimentation | `control_plane` | Strategy promotion |
| All | Risk checks | Pre-trade, post-trade validation |

### 8.3 Cross-Service Communication

```yaml
# Inter-service communication patterns

# Feature Builder -> Strategy Runner
tick_features:
  protocol: ZeroMQ PUB/SUB
  format: Arrow IPC
  latency_target: < 5ms

# Regime Classifier -> Capital Allocator
regime_updates:
  protocol: gRPC (or REST for simplicity)
  format: JSON
  frequency: On regime change + 15m intervals

# Capital Allocator -> Execution Engine
allocation_updates:
  protocol: gRPC
  format: Protobuf
  latency_target: < 10ms

# Monitor -> All Services
health_checks:
  protocol: HTTP/REST
  format: JSON
  frequency: 10s intervals

# Registry -> All Services
artifact_sync:
  protocol: REST + S3-compatible
  format: JSON + ONNX
  trigger: On promotion
```

### 8.4 Data Flow Integration

```
Phase 1 Services                    Phase 3 Services
================                    ================

ingestor_py
    |
    v
feature_builder_py ----------------> Regime Classifier
    |                                      |
    v                                      v
labeler_py                          Capital Allocator
    |                                      |
    v                                      v
backtester_py <------------------- Strategy Families
    |                                      |
    v                                      v
optimizer_py                        Multi-Strategy Runner
    |                                      |
    v                                      |
registry_api_py <-------------------------------+
    |
    v
monitor_py <--------------------------- All Phase 3 Services


Phase 2 Services
================

execution_engine_rs <------------- Multi-Strategy Runner
    |                                      ^
    v                                      |
control_plane ---------------------------->+
```

---

## 9. Risk Register

### 9.1 Technical Risks

| Risk ID | Description | Impact | Probability | Mitigation | Owner |
|---------|-------------|--------|-------------|------------|-------|
| R-001 | Regime classifier accuracy below target | High | Medium | Multiple classifier approaches (HMM + ML), ensemble | ML Lead |
| R-002 | Rust ingestor performance below 100k/sec | High | Low | Early profiling, consider tokio optimizations | Rust Lead |
| R-003 | GPU acceleration not providing expected speedup | Medium | Medium | CPU fallback always available, profile before committing | ML Lead |
| R-004 | Inter-strategy interference causing bugs | High | Medium | Strong isolation, extensive testing | Platform Lead |
| R-005 | Allocator causing excessive turnover | Medium | Medium | Turnover constraints, rate limiting | Quant Lead |
| R-006 | Memory pressure from concurrent strategies | Medium | Medium | Resource limits, backpressure handling | Platform Lead |

### 9.2 Operational Risks

| Risk ID | Description | Impact | Probability | Mitigation | Owner |
|---------|-------------|--------|-------------|------------|-------|
| R-007 | K8s deployment complexity | Medium | High | Incremental rollout, staging environment | DevOps Lead |
| R-008 | Disaster recovery untested | High | Medium | Regular DR drills, automated procedures | DevOps Lead |
| R-009 | Secrets exposure | Critical | Low | Vault integration, no plaintext, audit logging | Security Lead |
| R-010 | Runbook coverage incomplete | Medium | Medium | Comprehensive documentation, incident reviews | DevOps Lead |

### 9.3 Research Risks

| Risk ID | Description | Impact | Probability | Mitigation | Owner |
|---------|-------------|--------|-------------|------------|-------|
| R-011 | Strategies overfit to historical data | Critical | Medium | Purged CV, deflated Sharpe, stability tests | Quant Lead |
| R-012 | Regime model fails in novel market conditions | High | Medium | Conservative allocation, continuous monitoring | Quant Lead |
| R-013 | Allocator underperforms vs simple baseline | Medium | Low | Fallback to equal-weight, continuous A/B testing | Quant Lead |
| R-014 | Strategy families correlated, reducing diversification | Medium | Medium | Correlation monitoring, family independence tests | Quant Lead |

### 9.4 Risk Mitigation Schedule

| Sprint | Risk Reviews | Key Mitigations |
|--------|--------------|-----------------|
| 1-2 | R-004, R-006 | Isolation testing, memory profiling |
| 3-4 | R-001, R-005 | Classifier benchmarks, turnover analysis |
| 5-6 | R-002, R-011 | Rust benchmarks, governance implementation |
| 7-8 | R-007, R-008, R-009, R-010 | K8s staging, DR drill, security audit |
| 9 | All | Final risk assessment, sign-off |

---

## 10. Governance Framework

### 10.1 Anti-Overfit Rules

#### 10.1.1 Purged Cross-Validation

```python
class PurgedKFold:
    """
    K-Fold cross-validation with purging to prevent leakage.

    Purge window: Remove observations within purge_gap of test set
    to prevent autocorrelation leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: timedelta = timedelta(days=5),
        embargo_gap: timedelta = timedelta(days=1)
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(self, X: pd.DataFrame, timestamps: pd.Series):
        """
        Generate purged train/test indices.

        For each fold:
        1. Define test window
        2. Purge: Remove train samples within purge_gap before test start
        3. Embargo: Remove train samples within embargo_gap after test end
        """
        pass
```

#### 10.1.2 Deflated Sharpe Ratio

```python
def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    variance_sharpe: float,
    expected_max_sharpe_null: float = 0.0
) -> float:
    """
    Calculate deflated Sharpe ratio per Lopez de Prado.

    Adjusts for multiple testing: If you test N strategies,
    the best one's Sharpe is biased upward.

    Args:
        sharpe: Observed Sharpe ratio
        n_trials: Number of strategies tested
        variance_sharpe: Variance of Sharpe estimates
        expected_max_sharpe_null: Expected max Sharpe under null

    Returns:
        Probability that observed Sharpe > expected under null
    """
    from scipy.stats import norm

    # Expected maximum Sharpe under null hypothesis
    # E[max(SR)] = sqrt(2*log(n_trials)) * variance_sharpe
    expected_max = expected_max_sharpe_null
    if n_trials > 1:
        expected_max = (
            (1 - np.euler_gamma) * norm.ppf(1 - 1/n_trials) +
            np.euler_gamma * norm.ppf(1 - 1/(n_trials * np.e))
        ) * np.sqrt(variance_sharpe)

    # Deflated Sharpe = P(SR > E[max(SR)|H0])
    deflated = norm.cdf(
        (sharpe - expected_max) / np.sqrt(variance_sharpe)
    )

    return deflated
```

### 10.2 Promotion Criteria

#### 10.2.1 Candidate to Shadow

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Net Sharpe (walk-forward test) | >= 0.8 | Annualized |
| Max Drawdown | <= 12% | Peak-to-trough |
| Profit Factor | >= 1.15 | Gross profit / gross loss |
| Worst Day | >= -4.0% | Single day return |
| Turnover | <= 4.0x / day | Notional / equity |
| Cost Sensitivity (1.5x) | Sharpe >= 0.4 | Under 1.5x cost multiplier |
| Parameter Sensitivity | Sharpe drop <= 35% | Under +/- 10% params |
| Positive Windows | >= 60% | Rolling 1-month windows |
| Deflated Sharpe | p > 0.95 | Adjusted for multiple testing |

#### 10.2.2 Shadow to Paper

| Criterion | Threshold | Duration |
|-----------|-----------|----------|
| Signal Rate | +/- 25% vs backtest | 5 days |
| Feature PSI (critical) | <= 0.2 | Per feature |
| Data Quality | Missing < 0.5%, gaps <= 5s | 5 days |
| Est. Slippage | <= backtest p75 + 5bps | 5 days |

#### 10.2.3 Paper to Canary

| Criterion | Threshold | Duration |
|-----------|-----------|----------|
| Realized Slippage | <= backtest median + 7bps | 10 days |
| Fill Rate | >= 95% | 10 days |
| Rolling 5-day Return | >= -1.5% | Continuous |
| Paper Sharpe | >= 0.0 | 10 days |

#### 10.2.4 Canary to Full

| Criterion | Threshold | Duration |
|-----------|-----------|----------|
| Live Slippage | <= paper median + 5bps | 10 days |
| Live MDD | <= 5% | 10 days |
| Live Sharpe | >= 0.3 | 10 days |
| Operational Incidents | 0 | 10 days |

### 10.3 Incumbent Comparison Rules

```yaml
incumbent_comparison:
  # New strategy must beat incumbent
  sharpe_improvement: +0.2  # OR
  mdd_improvement: -2%      # with similar Sharpe (within 0.1)

  # Comparison windows
  test_windows:
    - full_test_set
    - high_vol_regime
    - low_vol_regime
    - trending_regime
    - ranging_regime

  # All windows must show improvement in at least one metric
  min_windows_passed: 4 / 5
```

### 10.4 Governance Audit Trail

```json
{
  "promotion_decision": {
    "decision_id": "promo_20260115_001",
    "timestamp": "2026-01-15T14:30:00Z",
    "strategy_id": "trend_momentum_5m_v3",
    "from_state": "SHADOW",
    "to_state": "PAPER",

    "metrics": {
      "sharpe_net": 1.05,
      "max_drawdown": 0.078,
      "profit_factor": 1.32,
      "deflated_sharpe_p": 0.97,
      "cost_sensitivity_sharpe_1_5x": 0.52
    },

    "gate_results": {
      "sharpe_threshold": {"passed": true, "value": 1.05, "threshold": 0.8},
      "mdd_threshold": {"passed": true, "value": 0.078, "threshold": 0.12},
      "deflated_sharpe": {"passed": true, "value": 0.97, "threshold": 0.95}
    },

    "incumbent_comparison": {
      "incumbent_id": "trend_momentum_5m_v2",
      "sharpe_delta": +0.25,
      "mdd_delta": -0.01,
      "passed": true
    },

    "approved_by": "governance_engine_v1",
    "data_snapshot": "snapshot_20260114"
  }
}
```

---

## 11. Definition of Done

### 11.1 Phase 3 Completion Criteria

#### 11.1.1 Multi-Strategy Framework

- [ ] 4+ strategy families implemented (TREND, MREV, MICRO, MLPRED)
- [ ] Each family has >= 2 validated variants
- [ ] Concurrent execution with < 100ms latency overhead
- [ ] No inter-strategy state leakage (verified by testing)
- [ ] Strategy lifecycle management fully operational
- [ ] All strategies pass evaluation protocol

#### 11.1.2 Regime Classifier

- [ ] Regime classifier accuracy >= 75% out-of-sample
- [ ] 6 distinct regimes classifiable
- [ ] Transition smoothing prevents rapid flipping
- [ ] Classifier latency < 10ms
- [ ] Historical regime labels validated
- [ ] Monitoring dashboard operational

#### 11.1.3 Capital Allocator

- [ ] Allocator Sharpe >= 90% of best single strategy
- [ ] Allocator MDD <= best single MDD + 2%
- [ ] Turnover <= 1.25x average constituent
- [ ] Weight stability: avg L1 change <= 0.25
- [ ] All allocation decisions auditable
- [ ] Drawdown-aware reduction working correctly

#### 11.1.4 Scalable Pipeline

- [ ] Rust ingestor: >= 100,000 ticks/sec sustained
- [ ] Latency: < 1ms receipt to publish
- [ ] Memory: < 500MB for 1-hour buffer
- [ ] Zero data loss under normal operation
- [ ] Python feature builder async integration working
- [ ] (Optional) GPU acceleration path documented

#### 11.1.5 Anti-Overfit Governance

- [ ] Purged CV implemented and validated
- [ ] Deflated Sharpe ratio calculation verified
- [ ] Cost sensitivity tests: Sharpe >= 0.4 at 1.5x
- [ ] Parameter sensitivity: Sharpe drop <= 35%
- [ ] 90%+ of promoted strategies pass all gates
- [ ] Governance audit reports complete

#### 11.1.6 Live Experimentation

- [ ] Shadow runner continuous for all candidates
- [ ] Canary deployment: 5-10% allocation working
- [ ] Auto-rollback latency < 500ms
- [ ] All rollback triggers (DD, slippage, rejects, feed) working
- [ ] Experiment tracking dashboard operational

#### 11.1.7 Operational Maturity

- [ ] All services deployable via Helm
- [ ] Secrets management (no plaintext)
- [ ] RTO < 30 minutes for full recovery
- [ ] Event logs replayable
- [ ] Runbooks cover all common scenarios
- [ ] DR drill completed successfully
- [ ] Uptime >= 99.5% in staging

#### 11.1.8 Production Readiness

- [ ] All integration tests passing
- [ ] Latency targets met under 2x load
- [ ] No memory leaks in 24-hour test
- [ ] Chaos tests passed
- [ ] Security audit passed
- [ ] Documentation complete and reviewed
- [ ] Stakeholder sign-off obtained

### 11.2 Acceptance Sign-Off

| Role | Sign-Off Criteria | Signed |
|------|-------------------|--------|
| Tech Lead | All technical criteria met | [ ] |
| Quant Lead | Strategy/allocator performance validated | [ ] |
| DevOps Lead | Operational maturity complete | [ ] |
| Security Lead | Security audit passed | [ ] |
| Product Owner | Business requirements satisfied | [ ] |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Regime** | Market state characterized by volatility, trend, and liquidity conditions |
| **Purged CV** | Cross-validation that removes samples near test set to prevent leakage |
| **Deflated Sharpe** | Sharpe ratio adjusted for multiple testing bias |
| **Canary Deployment** | Small-scale live deployment for validation before full rollout |
| **HNSW** | Hierarchical Navigable Small World - graph-based approximate nearest neighbor search |
| **MDD** | Maximum Drawdown - largest peak-to-trough decline |
| **PSI** | Population Stability Index - measure of distribution shift |
| **Taker** | Market participant who crosses the spread (lifts offers / hits bids) |

---

## Appendix B: References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge Elements.
3. Chan, E. (2013). *Algorithmic Trading*. Wiley.
4. Alpaca Markets API Documentation: https://alpaca.markets/docs/
5. Apache Arrow Documentation: https://arrow.apache.org/docs/

---

*Document generated: 2026-01-15*
*Next review: Sprint 1 kickoff*
