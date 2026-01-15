---

# PRD.md — Autonomous Equities Taker Trading Engine

**Alpaca • Multi-Timeframe • Microstructure-Aware • Strategy-Library + Regime Allocation**

---

## 1. Purpose & Scope

Build a **production-grade autonomous equities taker engine** that continuously:

> **ingests → builds features → generates strategies → backtests → selects → shadows → paper trades → promotes → trades → monitors → rolls back**

The system is **self-iterating**, **policy-bounded**, and **audit-safe**.

This PRD covers **all three phases**, with **quotes + trades ingestion from Day 1** and a **multi-strategy, regime-aware allocator** as the end state.

---

## 2. Non-Negotiable Requirements

### 2.1 Reproducibility

* Same code + same data snapshot ⇒ identical results
* All backtests reference immutable dataset versions

### 2.2 No Leakage

* All joins are **as-of**
* Features’ end timestamps ≤ decision timestamps
* Corporate actions applied before feature generation

### 2.3 Realistic Taker Execution

* Quote-based fills (bid/ask)
* Slippage depends on spread, volatility, depth
* No mid-price fantasy fills

### 2.4 Safety & Governance

* Shadow → Paper → Canary → Full promotion
* Automated rollback
* Full audit trail

---

## 3. Phase Overview

### Phase 1 — Python-First Autonomous Loop

* Ingest **trades + quotes**
* Build **microstructure-aware bars**
* Run **multi-strategy library**
* Decisions on **1-minute clock**
* Shadow + paper only

### Phase 2 — Rust Execution Runtime

* Rust execution engine
* Live trading + strict risk checks
* Python remains research/control plane

### Phase 3 — Strategy Factory + Regime Allocation

* Multiple concurrent strategies
* Regime classifier
* Capital allocator
* Tick/L2-ready ingestion

---

## 4. System Architecture (All Phases)

### Services

* `ingestor_py`
* `feature_builder_py`
* `labeler_py`
* `backtester_py`
* `optimizer_py`
* `registry_api_py`
* `monitor_py`
* `execution_engine_rs` (Phase 2+)

### Shared Infrastructure

* Object storage (Parquet)
* Postgres (registry, metrics, audit)
* Alpaca REST + WebSockets

---

## 5. Phase 1 — Detailed Requirements

### 5.1 Data Ingestion (Option A)

**Raw, immutable ingestion of:**

* Trades (ticks)
* Quotes (NBBO)
* Provider bars (validation only)

Both timestamps required:

* `ts_event` (exchange)
* `ts_recv` (local receipt)

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
```

Partitioning:

```
dt=YYYY-MM-DD/symbol=SPY/
```

---

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

---

### 6.3 Microstructure Bars (`lake/features/micro_bars/`)

(Generated from trades + quotes)

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

Granularities:

* 5s
* 15s
* 30s

---

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

Granularities:

* 1m
* 5m
* 15m

---

### 6.5 Decision-Frame Feature Matrix (`lake/features/decision_frame/`)

(As-of joined, final model input)

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
horizon: int32   # seconds
fwd_return_mid: float64
fwd_return_net: float64   # net of half-spread
direction: int8           # -1, 0, +1
```

---

## 8. Backtest Execution Model (Phase 1)

### Fill Price

* Buy: `ask + slippage`
* Sell: `bid - slippage`

### Slippage Model

```
slippage_bps =
  a * spread_bps +
  b * (order_notional / top_of_book_notional) +
  c * short_term_vol
```

Coefficients `a,b,c` are fixed per cost model version.

---

## 9. Strategy Model (Approach 2)

### Strategy Library

Each strategy is defined by:

* family (trend, mean-reversion, microstructure)
* feature subset
* model or rule
* execution policy

Multiple strategies may be active simultaneously.

---

## 10. Artifact Contracts (Authoritative)

### 10.1 JSON Strategy Artifact (Rules / Simple Models)

```json
{
  "artifact_version": "1.0",
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

---

### 10.2 ONNX Artifact Contract (ML Models)

**Inputs (exact order):**

```
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

```
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

---

## 12. Phase 3 — Regime Classifier & Allocator (Detailed)

### 12.1 Regime State Vector

At time `t`, compute:

```
R(t) = [
  σ_short(t),        # 1m realized vol
  σ_long(t),         # 15m realized vol
  trend_strength(t), # |ret_15m| / σ_15m
  spread_z(t),       # z-score of spread
  liquidity_z(t)     # z-score of top-of-book depth
]
```

---

### 12.2 Regime Classification

Discrete regimes via clustering or rules:

| Regime         | Conditions                    |
| -------------- | ----------------------------- |
| Trending       | trend_strength > τ₁           |
| Mean-Reverting | trend_strength < τ₂ & low vol |
| Volatile       | σ_short / σ_long > τ₃         |
| Illiquid       | spread_z > τ₄                 |

Regime can be **soft** (probabilities) or **hard** (labels).

---

### 12.3 Strategy Expected Utility

For strategy *i* in regime *r*:

```
U_i(r) = E[PnL_i | r] - λ * Var(PnL_i | r)
```

Estimated from rolling live + paper performance.

---

### 12.4 Capital Allocation (Core Math)

Solve:

```
maximize_w  Σ w_i * U_i(r)
subject to:
  Σ w_i = 1
  w_i ≥ 0
  portfolio_drawdown ≤ D_max
  portfolio_vol ≤ V_max
```

Practical solution:

* Softmax over utilities with caps
* Or risk-parity-scaled weights:

```
w_i ∝ U_i(r) / σ_i
```

---

### 12.5 Live Adaptation

* Recompute allocation daily or on regime change
* New strategies enter at minimal weight (canary)
* Poor performers decay automatically

---

## 13. Monitoring & Rollback

Auto-rollback on:

* drawdown breach
* slippage spike
* rejection surge
* feed instability

Rollback target:

* last known good artifact set + weights

---

## 14. Repo Layout (Final)

```text
services/
  ingestor_py/
  feature_builder_py/
  labeler_py/
  backtester_py/
  optimizer_py/
  registry_api_py/
  monitor_py/
  execution_engine_rs/
libs/
  common_types/
  cost_models/
  risk_models/
infra/
  docker-compose.yml
  k8s/
docs/
  PRD.md
  EVAL_PROTOCOL.md
  PROMOTION_GATES.md
  RUNBOOKS.md
```

---

## 15. Final Locked-In Decisions

* **Quotes + trades ingested from Day 1**
* **Microstructure-aware bars**
* **1-minute decision clock initially**
* **Strategy library + regime allocator**
* **Python research, Rust execution**

---
