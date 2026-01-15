---

# PRD.md — Autonomous Equities Taker Trading Engine

**Alpaca • Multi-TimeFrame • Microstructure-Aware • Strategy-Library + Regime Allocation • CUDA GPU Acceleration where applicable**

---

## 1. Purpose & Scope

Build a **production-grade autonomous equities taker engine** that continuously:

> **ingests → builds features → generates strategies → backtests → selects → shadows → paper trades → promotes → trades → monitors → rolls back**

The system is **self-iterating**, **policy-bounded**, and **audit-safe**.

This PRD covers **all three phases**, with:

- **quotes + trades ingestion from Day 1**
- a **multi-strategy, regime-aware allocator** as the end state
- **CUDA GPU acceleration for ML training and large candidate sweeps where applicable**, with mandatory CPU fallback.

---

## 2. Non-Negotiable Requirements

### 2.1 Reproducibility

- Same code + same data snapshot ⇒ identical results
- All backtests reference immutable dataset versions
- GPU runs must record:
  - device (`cpu|cuda`)
  - backend (`xgboost_gpu|pytorch_cuda|cpu`)
  - CUDA/driver versions
  - seeds and determinism flags

### 2.2 No Leakage

- All joins are **as-of**
- Features’ end timestamps ≤ decision timestamps
- Corporate actions applied before feature generation

### 2.3 Realistic Taker Execution

- Quote-based fills (bid/ask)
- Slippage depends on spread, volatility, depth
- No mid-price fantasy fills

### 2.4 Safety & Governance

- Shadow → Paper → Canary → Full promotion
- Automated rollback
- Full audit trail (including GPU training/inference metadata)

---

## 3. Phase Overview

### Phase 1 — Python-First Autonomous Loop (GPU training optional)

- Ingest **trades + quotes**
- Build **microstructure-aware bars**
- Run **multi-strategy library**
- Decisions on **1-minute clock**
- Shadow + paper only
- GPU acceleration used primarily for:
  - ML model training during candidate sweeps (e.g., XGBoost GPU)
  - (Optional) ML inference for heavy models (usually not needed in Phase 1)

### Phase 2 — Rust Execution Runtime

- Rust execution engine
- Live trading + strict risk checks
- Python remains research/control plane
- Optional GPU inference support for ONNX (only if warranted)

### Phase 3 — Strategy Factory + Regime Allocation

- Multiple concurrent strategies
- Regime classifier
- Capital allocator
- Tick/L2-ready ingestion
- GPU acceleration increasingly valuable for:
  - large-scale ML sweeps
  - more complex models (neural nets)
  - high-throughput research runs

---

## 4. System Architecture (All Phases)

### Services

- `ingestor_py`
- `feature_builder_py`
- `labeler_py`
- `backtester_py`
- `optimizer_py`
- `registry_api_py`
- `monitor_py`
- `execution_engine_rs` (Phase 2+)

### Shared Infrastructure

- Object storage (Parquet)
- Postgres (registry, metrics, audit)
- Alpaca REST + WebSockets

---

## 5. Phase 1 — Detailed Requirements

### 5.1 Data Ingestion (Option A)

**Raw, immutable ingestion of:**

- Trades (ticks)
- Quotes (NBBO)
- Provider bars (validation only)

Both timestamps required:

- `ts_event` (exchange)
- `ts_recv` (local receipt)

---

## 5.2 GPU Acceleration Requirements (Phase 1+)

### GPU applicability (where it is used)

- Candidate **ML training** inside `optimizer`:
  - Prefer XGBoost GPU (`gpu_hist`) when enabled and available.
  - Future: PyTorch CUDA for neural models.
- Optional ONNX inference on GPU (rarely required for initial tree/linear models).

### GPU policy controls (must exist)

- `GPU_ENABLED` (true/false)
- `GPU_DEVICE` (e.g., `cuda:0`)
- `GPU_BACKEND` (`xgboost_gpu`, `pytorch_cuda`, `cpu`)
- `GPU_DETERMINISTIC` (true/false)
- `GPU_FALLBACK_CPU` (true/false, default true)

### Registry/audit logging (must record)

- training_device (`cpu|cuda`)
- backend used
- CUDA runtime & driver versions (if cuda)
- seed and determinism flags
- per-candidate training time

---

## 6. Exact Parquet Schemas (Authoritative)

### 6.1 Trades (`lake/raw/trades/`)

```text
symbol: string
ts_event: timestamp[ns]
ts_recv: timestamp[ns]
price: float64
size: float64
exchange: string
conditions: string
````

**Partitioning:**

```ini
dt=YYYY-MM-DD/symbol=SPY/
```

### 6.2 Quotes (`lake/raw/quotes/`)

```text
symbol: string
ts_event: timestamp[ns]
ts_recv: timestamp[ns]
bid_price: float64
bid_size: float64
ask_price: float64
ask_size: float64
exchange: string
```

### 6.3 Microstructure Bars (`lake/features/micro_bars/`)

```text
symbol: string
bar_start: timestamp[ns]
bar_end: timestamp[ns]
vwap: float64
midprice: float64
microprice: float64
spread: float64
bid_size: float64
ask_size: float64
quote_imbalance: float64
trade_volume: float64
realized_vol: float64
```

Granularities: **5s**, **15s**, **30s**

### 6.4 Timeframe Bars (`lake/features/bars/`)

```text
symbol: string
bar_start: timestamp[ns]
bar_end: timestamp[ns]
open: float64
high: float64
low: float64
close: float64
volume: float64
returns: float64
atr: float64
realized_vol: float64
```

Granularities: **1m**, **5m**, **15m**

### 6.5 Decision-Frame Feature Matrix (`lake/features/decision_frame/`)

```text
symbol: string
decision_ts: timestamp[ns]

# Microstructure
spread_30s: float64
microprice_30s: float64
quote_imbalance_30s: float64
vol_30s: float64

# 1m
ret_1m: float64
vol_1m: float64
atr_1m: float64

# 5m
ret_5m: float64
trend_5m: float64
vol_5m: float64

# 15m
ret_15m: float64
trend_15m: float64
vol_15m: float64
```

---

## 7. Labeling Schema (`lake/labels/`)

```text
symbol: string
decision_ts: timestamp[ns]
horizon: int32            # seconds
fwd_return_mid: float64
fwd_return_net: float64   # net of half-spread
direction: int8           # -1, 0, +1
```

---

## 8. Backtest Execution Model (Phase 1)

### Fill Price

* **Buy:** ask + slippage
* **Sell:** bid − slippage

### Slippage Model

```makefile
slippage_bps =
  a * spread_bps +
  b * (order_notional / top_of_book_notional) +
  c * short_term_vol
```

Coefficients `a,b,c` are fixed per cost model version.

> GPU note: backtesting is CPU by default; GPU backtest acceleration is optional later if large-scale vectorization is implemented.

---

## 9. Strategy Model (Approach 2)

### Strategy Library

Each strategy is defined by:

* family (trend, mean-reversion, microstructure)
* feature subset
* model or rule
* execution policy
* training configuration:

  * `training_device_preference` (cpu|cuda|auto)
  * `backend` (xgboost|sklearn|pytorch)

Multiple strategies may be active simultaneously.

---

## 10. Artifact Contracts (Authoritative)

### 10.1 JSON Strategy Artifact (Rules / Simple Models)

(unchanged shape; must include training metadata fields)

```json
{
  "artifact_version": "1.1",
  "strategy_id": "trend_5m_v3",
  "strategy_family": "trend",
  "feature_schema_version": "decision_frame_v1",
  "label_horizon_sec": 300,
  "cost_model_version": "cm_v1",
  "risk_profile_id": "rp_default",
  "symbols": ["SPY", "QQQ"],
  "decision_interval_sec": 60,

  "training_metadata": {
    "training_device": "cuda",
    "gpu_backend": "xgboost_gpu",
    "cuda_version": "12.x",
    "driver_version": "xxx.xx",
    "seed": 1337,
    "deterministic": false
  },

  "signal_logic": {
    "type": "threshold",
    "expression": "ret_5m > theta",
    "theta": 0.0015
  },

  "sizing": {
    "type": "vol_target",
    "target_vol": 0.10,
    "max_notional": 50000
  },

  "execution": {
    "order_type": "marketable_limit",
    "limit_offset_bps": 2,
    "time_in_force": "IOC"
  }
}
```

### 10.2 ONNX Artifact Contract (ML Models)

Inputs and outputs are unchanged; metadata is expanded to capture GPU training provenance.

**Inputs (exact order):**

```csharp
[spread_30s,
 microprice_30s,
 quote_imbalance_30s,
 vol_30s,
 ret_1m,
 vol_1m,
 atr_1m,
 ret_5m,
 trend_5m,
 vol_5m,
 ret_15m,
 trend_15m,
 vol_15m]
```

**Outputs:**

```nginx
expected_return
confidence_score
```

**Sidecar Metadata (`.json`):**

```json
{
  "onnx_version": "1.15",
  "feature_schema_version": "decision_frame_v1",
  "label_horizon_sec": 60,
  "cost_model_version": "cm_v1",
  "risk_profile_id": "rp_default",
  "calibration": {
    "expected_return_scale": 1.0,
    "confidence_threshold": 0.6
  },
  "training_metadata": {
    "training_device": "cuda",
    "gpu_backend": "xgboost_gpu",
    "cuda_version": "12.x",
    "driver_version": "xxx.xx",
    "seed": 1337,
    "deterministic": false
  },
  "inference_preferences": {
    "prefer_gpu": false,
    "cpu_fallback": true
  }
}
```

---

## 11. Phase 2 — Execution & Risk (Summary)

* Rust execution engine
* Pre-trade risk:

  * max position
  * max gross/net
  * max daily loss
* Post-trade validation:

  * slippage
  * fill rate
* Kill switch:

  * cancel all orders
  * halt trading
* Optional: ONNX GPU inference (only if needed), with CPU fallback

---

## 12. Phase 3 — Regime Classifier & Allocator (Detailed)

(unchanged math; GPU primarily impacts training throughput for ML strategies and large sweeps)

---

## 13. Monitoring & Rollback

(unchanged; must also monitor GPU inference latency if enabled)

---

## 14. Repo Layout (Final)

(unchanged)

---

## 15. Final Locked-In Decisions

* Quotes + trades ingested from Day 1
* Microstructure-aware bars
* 1-minute decision clock initially
* Strategy library + regime allocator
* Python research, Rust execution
* CUDA GPU acceleration used where it provides material benefit (primarily ML training), with CPU fallback

---