---

# Autonomous Equities Taker Trading Engine

**Alpaca-powered • Multi-timeframe • Microstructure-aware • Self-iterating**

## Overview

This document defines a **production-shaped, autonomous equities taker engine** that continuously:

> **ingests → generates strategies → backtests → selects → shadows → paper trades → promotes → trades → monitors → rolls back**

The system is designed to **iterate without manual intervention**, while remaining **statistically disciplined, auditable, and safe**.

Key architectural decisions baked in from the start:

* **Option A**: ingest *raw trades + quotes* and derive **microstructure-aware bars** (instead of naïve bar-only or full tick-by-tick execution).
* **Approach 2**: run a **strategy library + regime-based allocator**, rather than endlessly mutating a single strategy.

---

## System Goals (All Phases)

### What “Autonomous” Means (Precisely)

* The system **automatically generates strategy candidates**, evaluates them rigorously, and promotes winners.
* Humans define:

  * allowed strategy families
  * risk limits
  * evaluation protocol
  * promotion gates
* Humans do **not**:

  * tune parameters manually
  * decide which model to ship
  * intervene during normal iteration

This is **automation with policy**, not an unconstrained optimizer.

---

### Non-Negotiables

* **Reproducibility**

  * Same code + same data snapshot → identical backtest results
* **No Leakage**

  * Strict as-of joins
  * Feature timestamps ≤ decision timestamps
  * Corporate actions handled correctly
* **Realistic Execution Modeling**

  * Quote-based fills (bid/ask)
  * Spread + liquidity-aware slippage
* **Promotion Gates**

  * Shadow → Paper → Canary → Full
  * Automatic rollback
* **Full Audit Trail**

  * Why a strategy was promoted
  * What data it used
  * What trades it generated

---

## Alpaca Integration Points (All Phases)

### Trading & Account

* Paper trading uses the same API schema as live, with different credentials/endpoints.
* Orders use Alpaca’s standard `/orders` lifecycle.

### Streaming (Two WebSocket Classes)

1. **Trading Updates Stream**

   * Orders, fills, position updates, account state
   * Used by runner + monitor

2. **Market Data Streams**

   * Trades
   * Quotes (NBBO)
   * Bars
   * (Optional later: news)

> **Important**:
>
> * Free plan → IEX feed
> * Paid plan → SIP consolidated tape
>   Architecture supports either without changes.

---

# Phase 1 — MVP Closed Loop (Python-first, Microstructure-aware)

## Objective

Deliver a **fully autonomous loop** with:

* quotes + trades ingestion
* multi-timeframe feature construction
* strategy library evaluation
* shadow + paper execution
* realistic taker fills

No live capital yet.

---

## Phase 1 Core Design Choice (Critical)

### Option A: Tick/Quote Ingestion → Microstructure-Aware Bars

* **Raw trades + quotes are ingested and stored**
* Strategies **do NOT operate on every tick**
* Decisions occur on **fixed decision clocks** (initially 1-minute)
* Microstructure signals are derived from short windows (5s–60s)

This yields:

* realistic slippage modeling
* liquidity-aware signals
* fast iteration
* low risk of execution optimism

---

## Phase 1 Services

### A. `ingestor` (Python)

**Purpose:** Capture raw market truth.

* Pull historical:

  * trades
  * quotes
  * bars (for validation only)
* Subscribe to real-time WebSocket streams
* Write **immutable Parquet** data

**Outputs**

* `lake/raw/trades/`
* `lake/raw/quotes/`
* `lake/raw/bars_provider/`
* `universe/` table (symbols, exchange, tradability)

**Key Fields**

* `ts_event` (exchange timestamp)
* `ts_recv` (receipt timestamp)
* prices, sizes, symbol

> Both timestamps are mandatory to support latency modeling and leakage detection.

---

### B. `feature_builder` (Python)

**Purpose:** Build multi-timeframe, as-of-safe features.

#### Feature Classes

1. **Microstructure (from quotes + trades)**

   * spread
   * midprice / microprice
   * quote imbalance
   * top-of-book depth
   * trade imbalance (tick rule proxy)
   * short-horizon realized volatility

2. **Derived Micro-Bars**

   * 5s / 15s / 30s bars built from ticks
   * VWAP, range, volume

3. **Traditional Bars**

   * 1m / 5m / 15m OHLCV
   * momentum, mean reversion, volatility

#### Multi-Timeframe Join (As-Of)

At each decision timestamp `t`:

```
X(t) = [
  micro_30s_features ending ≤ t,
  bar_1m_features ending ≤ t,
  bar_5m_features ending ≤ t,
  bar_15m_features ending ≤ t
]
```

No future data is ever visible.

---

### C. `labeler` (Python)

**Purpose:** Define taker-realistic prediction targets.

* Forward returns at configurable horizons
* Direction labels with no-trade band
* Optional labels net of half-spread
* Uses midprice / microprice (not last trade)

---

### D. `backtester` (Python)

**Purpose:** Evaluate strategies under realistic taker execution.

#### Execution Model (Phase 1)

* Orders cross the spread:

  * buy → ask + slippage
  * sell → bid − slippage
* Slippage is a function of:

  * spread
  * volatility
  * order size vs top-of-book size

#### Includes

* transaction costs
* position limits
* stop logic
* realistic fill assumptions

This is **not** a tick-by-tick simulator, but is **vastly more realistic than mid-price fills**.

---

### E. `optimizer` (Python)

**Purpose:** Generate and evaluate candidate strategies.

#### Candidate Space

* Multiple strategy families:

  * momentum
  * mean reversion
  * volatility breakouts
  * ML predictors
* Each family has:

  * parameter sweeps
  * model variants
  * execution policies

#### Evaluation

* Walk-forward:

  * train → validate → test
* Regime slices:

  * volatility buckets
  * liquidity buckets
  * trend states
* Complexity penalties
* Cost sensitivity tests

---

### F. `registry` (Python API + DB)

**Purpose:** System memory + audit log.

Stores:

* strategy family + version
* feature schema version
* data snapshot IDs
* parameters / artifacts
* backtest metrics
* promotion decisions
* lifecycle state:

  * candidate → shadow → paper → promoted → retired

---

### G. `runner` (Python, Shadow + Paper)

**Purpose:** Execute strategies without live risk.

* Consumes real-time quotes/trades
* Generates signals on decision clock
* Shadow:

  * logs signals only
* Paper:

  * submits to Alpaca paper account
* Confirms fills via trading websocket

---

### H. `monitor` (Python)

**Purpose:** Detect failure modes early.

Monitors:

* fill rate
* slippage vs expectation
* drawdown
* turnover
* feed lag / disconnects

Triggers:

* alerts
* automatic rollback to last good artifact

---

## Phase 1 Data & Storage

### Object Storage

* `lake/raw/` (immutable)
* `lake/features/` (versioned)
* `lake/labels/` (versioned)
* `artifacts/` (JSON / ONNX later)

### Postgres

* strategy registry
* metrics
* promotion decisions
* full audit trail

---

## Phase 1 Autonomous Loop

Runs daily or weekly:

1. Ingest new ticks + quotes
2. Build incremental features
3. Generate strategy candidates
4. Walk-forward backtests
5. Stress tests (costs, delays)
6. Rank within **strategy families**
7. Promote top K to shadow
8. Promote to paper only if:

   * shadow behavior matches backtest
   * paper slippage & drawdown acceptable
9. **No live trading in Phase 1**

---

# Phase 2 — Production Execution + Safe Promotion

## Objective

Move the always-on trading runtime to **Rust**, while keeping research in Python.

---

## Phase 2 Services

### A. `execution_engine` (Rust)

**Purpose:** Deterministic, safe live execution.

Responsibilities:

* consume market data streams
* maintain state (positions, orders)
* load promoted artifacts
* generate signals
* enforce pre-trade risk
* submit orders
* process fills via trading websocket

---

### B. `control_plane` (Python API)

* Strategy promotion
* Capital allocation
* Kill switch
* Configuration updates

---

### C. Promotion Gates

* Shadow (Rust)
* Paper (Alpaca)
* Canary (small live capital)
* Full live

---

### D. Telemetry

* decision latency
* websocket lag
* order latency
* slippage distributions
* alerts on anomalies

---

## Artifact Contract (Python → Rust)

### Supported Formats

* **JSON** (rules, params, thresholds)
* **ONNX** (ML inference)

Each artifact includes:

* feature schema version
* label horizon
* cost model version
* risk profile ID

---

## Risk Rules (Taker-Specific)

* marketable limit orders
* TIF policies
* max position
* max exposure
* max daily loss
* slippage bounds
* kill switch (immediate cancel + halt)

---

# Phase 3 — Scale, Strategy Library & Regime Allocation

## Objective

Turn the system into a **self-allocating strategy factory**.

---

## Multi-Strategy Library (Approach 2)

Run concurrently:

* trend strategies
* mean reversion strategies
* microstructure confirmation strategies
* ML-based predictors

---

## Regime-Based Allocation

Allocate capital based on:

* recent live performance
* volatility regime
* liquidity regime
* trend strength
* drawdown constraints

This replaces brittle “best single model” logic.

---

## Scalable Data Pipeline

* Arrow / Parquet everywhere
* Incremental computation
* Rust ingestion for heavy tick loads
* Async feature consumers

---

## Anti-Overfit Governance

* purged CV
* deflated Sharpe
* stability under:

  * cost changes
  * latency changes
  * universe subsets
* promotion must beat incumbent across slices

---

## Live Experimentation

* always-on shadow for new candidates
* automatic canary deployment
* auto-rollback on:

  * drawdown
  * slippage
  * rejects
  * feed instability

---

## Operational Maturity

* secrets management
* systemd / Kubernetes
* disaster recovery
* replayable event logs
* automated runbooks

---

## Suggested Repo Layout

```
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
  EVAL_PROTOCOL.md
  PROMOTION_GATES.md
  RUNBOOKS.md
```

---
