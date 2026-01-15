# Phase 1 Plan — MVP Closed Loop

**Autonomous Equities Taker Trading Engine**

---

## 1. Executive Summary

### Objective
Deliver a **fully autonomous trading loop** with quotes + trades ingestion, multi-timeframe feature construction, strategy library evaluation, and shadow + paper execution — **without live capital**.

### Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Historical data ingestion | ≥30 days trades + quotes for SPY, QQQ |
| 2 | Real-time streaming | <500ms latency from Alpaca WebSocket |
| 3 | Feature pipeline | Multi-timeframe as-of joins with zero leakage |
| 4 | Backtest realism | Quote-based fills with slippage model |
| 5 | Strategy evaluation | Walk-forward with purged CV |
| 6 | Shadow execution | Signal logging matches backtest behavior |
| 7 | Paper execution | Alpaca paper fills within 2x expected slippage |

### Scope

**In Scope:**
- Python services: ingestor, feature_builder, labeler, backtester, optimizer, registry_api, monitor
- Alpaca integration via Alpaca-py Python package
- GPU-accelerated ML training (optional, with CPU fallback)
- Shadow + paper execution modes
- PostgreSQL registry + MinIO object storage

**Out of Scope:**
- Live capital deployment
- Rust execution engine (Phase 2)
- Multi-strategy regime allocation (Phase 3)
- L2/full order book data

---

## 2. Service Implementation Breakdown

### 2.1 ingestor_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Capture raw market truth from Alpaca |
| **Complexity** | L (Large) |
| **Est. Hours** | 60-80 |

**Key Deliverables:**
- [x] Alpaca REST client for historical trades, quotes, bars
- [x] WebSocket client for real-time streaming
- [x] Parquet writer with partitioning (`dt=YYYY-MM-DD/symbol=XXX/`)
- [x] Both timestamps: `ts_event` (exchange) + `ts_recv` (local)
- [ ] Universe table management
- [x] Backfill orchestration

**Dependencies:** common_types, alpaca-py, pyarrow

**Outputs:**
- `lake/raw/trades/`
- `lake/raw/quotes/`
- `lake/raw/bars_provider/`
- `universe/` table

---

### 2.2 feature_builder_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Build multi-timeframe, as-of-safe features |
| **Complexity** | XL (Extra Large) |
| **Est. Hours** | 80-100 |

**Key Deliverables:**
- [x] Microstructure bars (5s, 15s, 30s): spread, midprice, microprice, quote_imbalance, trade_imbalance, realized_vol
- [x] Standard bars (1m, 5m, 15m): OHLCV, returns, ATR, volatility
- [x] Decision-frame feature matrix with as-of joins
- [x] Incremental computation support
- [x] Feature schema versioning

**Dependencies:** ingestor_py outputs, polars/pandas, common_types

**Outputs:**
- `lake/features/micro_bars/`
- `lake/features/bars/`
- `lake/features/decision_frame/`

---

### 2.3 labeler_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Define taker-realistic prediction targets |
| **Complexity** | M (Medium) |
| **Est. Hours** | 30-40 |

**Key Deliverables:**
- [x] Forward return calculation at configurable horizons (60s, 300s, 900s)
- [x] Direction labels (-1, 0, +1) with configurable no-trade band
- [x] Net-of-spread return calculation
- [x] Midprice/microprice-based returns (not last trade)

**Dependencies:** feature_builder_py outputs

**Outputs:**
- `lake/labels/`

---

### 2.4 backtester_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Evaluate strategies under realistic taker execution |
| **Complexity** | XL (Extra Large) |
| **Est. Hours** | 80-100 |

**Key Deliverables:**
- [x] Quote-based fill simulation (buy @ ask + slippage, sell @ bid - slippage)
- [x] Slippage model: `slippage_bps = a*spread + b*(order/book) + c*vol`
- [x] Transaction cost modeling
- [x] Position limits + stop logic
- [x] Walk-forward evaluation framework
- [x] Regime-sliced evaluation (volatility, liquidity, trend buckets)
- [x] Metrics computation (Sharpe, Sortino, MDD, profit factor, win rate)

**Dependencies:** feature_builder_py, labeler_py, cost_models, risk_models

---

### 2.5 optimizer_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Generate and evaluate candidate strategies |
| **Complexity** | XL (Extra Large) |
| **Est. Hours** | 100-120 |

**Key Deliverables:**
- [x] Strategy family definitions (trend, mean-reversion, volatility, ML)
- [x] Parameter sweep framework
- [x] ML model training (XGBoost with optional GPU: `tree_method=gpu_hist`)
- [x] ONNX export for ML models
- [ ] Complexity penalties
- [x] Cost sensitivity tests
- [x] GPU policy controls (`GPU_ENABLED`, `GPU_DEVICE`, `GPU_FALLBACK_CPU`)
- [x] Candidate ranking within families

**Dependencies:** backtester_py, registry_api_py

**GPU Metadata Recording:**
- `training_device`: cpu | cuda
- `gpu_backend`: xgboost_gpu | pytorch_cuda | cpu
- `cuda_version`, `driver_version`
- `seed`, `determinism_flags`
- `training_time_sec`

---

### 2.6 registry_api_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | System memory + audit log |
| **Complexity** | L (Large) |
| **Est. Hours** | 60-80 |

**Key Deliverables:**
- [x] Full CRUD for all registry tables (strategies, artifacts, runs, gates, promotions)
- [x] Dataset snapshot management
- [ ] Feature schema versioning
- [x] Artifact upload/download (JSON + ONNX)
- [x] Gate evaluation endpoints
- [x] Promotion state management (candidate → shadow → paper)
- [x] Audit log recording
- [ ] GPU training metadata storage

**Dependencies:** PostgreSQL, MinIO, common_types

---

### 2.7 monitor_py

| Attribute | Value |
|-----------|-------|
| **Purpose** | Detect failure modes early |
| **Complexity** | L (Large) |
| **Est. Hours** | 50-70 |

**Key Deliverables:**
- [ ] Fill rate monitoring
- [ ] Slippage vs expectation tracking
- [ ] Drawdown monitoring
- [ ] Turnover analysis
- [ ] Feed lag / disconnect detection
- [ ] Alert generation (Prometheus metrics)
- [ ] Automatic rollback triggers
- [ ] Shadow vs backtest behavior comparison

**Dependencies:** registry_api_py, runner (shadow/paper)

---

## 3. Sprint Plan

### Overview

| Sprint | Focus | Duration | Est. Hours |
|--------|-------|----------|------------|
| S1 | Foundation | 2 weeks | 80 |
| S2 | Feature Pipeline | 2 weeks | 90 |
| S3 | Backtesting Engine | 2 weeks | 100 |
| S4 | Optimization Loop | 2 weeks | 110 |
| S5 | Registry & Monitoring | 2 weeks | 100 |
| S6 | Integration & Autonomous Loop | 2 weeks | 120 |
| **Total** | | **12 weeks** | **~600 hrs** |

---

### Sprint 1: Foundation (Weeks 1-2)

**Goal:** Infrastructure + data ingestion pipeline

| Task ID | Task | Est. Hrs | Status |
|---------|------|----------|--------|
| T1.01 | Deploy infrastructure (docker-compose up) | 2 | [x] |
| T1.02 | Apply database schema (registry_schema.sql) | 2 | [x] |
| T1.03 | Implement common_types library (Pydantic models) | 16 | [x] |
| T1.04 | Implement cost_models library | 8 | [x] |
| T1.05 | Implement risk_models library | 8 | [x] |
| T1.06 | ingestor: Alpaca REST client (historical) | 12 | [x] |
| T1.07 | ingestor: Parquet writer with partitioning | 8 | [x] |
| T1.08 | ingestor: Historical backfill for SPY, QQQ (30 days) | 8 | [x] |
| T1.09 | ingestor: WebSocket client (real-time) | 12 | [x] |
| T1.10 | Unit tests for ingestor | 4 | [x] |

**Milestone M1:** Historical data ingested, real-time streaming operational

---

### Sprint 2: Feature Pipeline (Weeks 3-4)

**Goal:** Multi-timeframe feature construction

| Task ID | Task | Est. Hrs | Deps | Status |
|---------|------|----------|------|--------|
| T2.01 | feature_builder: Standard bars (1m, 5m, 15m) | 16 | T1.08 | [x] |
| T2.02 | feature_builder: Microstructure bars (5s, 15s, 30s) | 20 | T1.08 | [x] |
| T2.03 | feature_builder: As-of join logic | 12 | T2.01, T2.02 | [x] |
| T2.04 | feature_builder: Decision-frame matrix | 10 | T2.03 | [x] |
| T2.05 | feature_builder: Incremental computation | 8 | T2.04 | [x] |
| T2.06 | labeler: Forward returns calculation | 8 | T2.04 | [x] |
| T2.07 | labeler: Direction labels with no-trade band | 6 | T2.06 | [x] |
| T2.08 | labeler: Net-of-spread returns | 4 | T2.07 | [x] |
| T2.09 | Unit tests for feature_builder | 4 | T2.05 | [x] |
| T2.10 | Unit tests for labeler | 2 | T2.08 | [x] |

**Milestone M2:** Feature pipeline operational, labels generated

---

### Sprint 3: Backtesting Engine (Weeks 5-6)

**Goal:** Realistic taker execution simulation

| Task ID | Task | Est. Hrs | Deps | Status |
|---------|------|----------|------|--------|
| T3.01 | backtester: Core simulation loop | 16 | T2.04 | [x] |
| T3.02 | backtester: Quote-based fill logic | 12 | T3.01 | [x] |
| T3.03 | backtester: Slippage model integration | 8 | T3.02, T1.04 | [x] |
| T3.04 | backtester: Position limits + stops | 8 | T3.03 | [x] |
| T3.05 | backtester: Walk-forward framework | 12 | T3.04 | [x] |
| T3.06 | backtester: Metrics computation | 10 | T3.05 | [x] |
| T3.07 | backtester: Regime slicing (vol, liquidity, trend) | 12 | T3.06 | [x] |
| T3.08 | backtester: Purged CV implementation | 10 | T3.07 | [x] |
| T3.09 | Integration tests (feature → backtest) | 8 | T3.08 | [x] |
| T3.10 | Backtest validation vs provider bars | 4 | T3.09 | [x] |

**Milestone M3:** Backtester produces realistic metrics

---

### Sprint 4: Optimization Loop (Weeks 7-8)

**Goal:** Strategy generation with GPU support

| Task ID | Task | Est. Hrs | Deps | Status |
|---------|------|----------|------|--------|
| T4.01 | optimizer: Strategy family framework | 12 | T3.08 | [x] |
| T4.02 | optimizer: Trend strategy implementation | 10 | T4.01 | [x] |
| T4.03 | optimizer: Mean-reversion strategy | 10 | T4.01 | [x] |
| T4.04 | optimizer: Parameter sweep framework | 12 | T4.02, T4.03 | [x] |
| T4.05 | optimizer: XGBoost ML strategy | 14 | T4.04 | [x] |
| T4.06 | optimizer: GPU training support | 10 | T4.05 | [x] |
| T4.07 | optimizer: CPU fallback logic | 6 | T4.06 | [x] |
| T4.08 | optimizer: ONNX export | 8 | T4.05 | [x] |
| T4.09 | optimizer: Candidate ranking | 8 | T4.08 | [x] |
| T4.10 | optimizer: Cost sensitivity tests | 8 | T4.09 | [x] |
| T4.11 | Unit tests for optimizer | 6 | T4.10 | [x] |
| T4.12 | GPU metadata logging | 6 | T4.06 | [x] |

**Milestone M4:** Optimizer generates and ranks candidates

---

### Sprint 5: Registry & Monitoring (Weeks 9-10)

**Goal:** System memory, audit, and monitoring

| Task ID | Task | Est. Hrs | Deps | Status |
|---------|------|----------|------|--------|
| T5.01 | registry: Expand FastAPI endpoints (CRUD) | 16 | M4 | [x] |
| T5.02 | registry: Dataset snapshot management | 8 | T5.01 | [x] |
| T5.03 | registry: Artifact upload/download | 10 | T5.02 | [x] |
| T5.04 | registry: Gate evaluation endpoints | 10 | T5.03 | [x] |
| T5.05 | registry: Promotion state machine | 8 | T5.04 | [x] |
| T5.06 | registry: Audit log implementation | 6 | T5.05 | [x] |
| T5.07 | monitor: Feed health monitoring | 8 | T1.09 | [x] |
| T5.08 | monitor: Slippage tracking | 8 | T5.05 | [x] |
| T5.09 | monitor: Drawdown alerts | 6 | T5.08 | [x] |
| T5.10 | monitor: Prometheus metrics export | 8 | T5.09 | [x] |
| T5.11 | monitor: Rollback trigger logic | 8 | T5.10 | [x] |
| T5.12 | Integration tests (registry + monitor) | 4 | T5.11 | [x] |

**Milestone M5:** Registry fully operational, monitoring active

---

### Sprint 6: Integration & Autonomous Loop (Weeks 11-12)

**Goal:** Complete autonomous loop with shadow/paper execution

| Task ID | Task | Est. Hrs | Deps | Status |
|---------|------|----------|------|--------|
| T6.01 | runner: Shadow execution mode | 16 | M5 | [ ] |
| T6.02 | runner: Signal logging | 8 | T6.01 | [ ] |
| T6.03 | runner: Paper execution (Alpaca paper API) | 16 | T6.02 | [ ] |
| T6.04 | runner: Fill confirmation via WebSocket | 10 | T6.03 | [ ] |
| T6.05 | Autonomous loop: Daily/weekly orchestration | 12 | T6.04 | [ ] |
| T6.06 | Autonomous loop: Shadow → paper promotion | 10 | T6.05 | [ ] |
| T6.07 | Shadow vs backtest behavior validation | 12 | T6.06 | [ ] |
| T6.08 | Paper slippage analysis | 8 | T6.07 | [ ] |
| T6.09 | End-to-end integration tests | 12 | T6.08 | [ ] |
| T6.10 | Performance benchmarking | 8 | T6.09 | [ ] |
| T6.11 | Documentation update | 4 | T6.10 | [ ] |
| T6.12 | Phase 1 acceptance testing | 4 | T6.11 | [ ] |

**Milestone M6:** Phase 1 complete — autonomous loop operational

---

## 4. Complete Task Tracking

| ID | Task | Service | Sprint | Hours | Deps | Status |
|----|------|---------|--------|-------|------|--------|
| T1.01 | Deploy infrastructure | infra | S1 | 2 | - | [x] |
| T1.02 | Apply database schema | infra | S1 | 2 | T1.01 | [x] |
| T1.03 | common_types library | libs | S1 | 16 | - | [x] |
| T1.04 | cost_models library | libs | S1 | 8 | T1.03 | [x] |
| T1.05 | risk_models library | libs | S1 | 8 | T1.03 | [x] |
| T1.06 | Alpaca REST client | ingestor | S1 | 12 | T1.03 | [x] |
| T1.07 | Parquet writer | ingestor | S1 | 8 | T1.06 | [x] |
| T1.08 | Historical backfill | ingestor | S1 | 8 | T1.07 | [x] |
| T1.09 | WebSocket client | ingestor | S1 | 12 | T1.06 | [x] |
| T1.10 | Ingestor tests | ingestor | S1 | 4 | T1.09 | [x] |
| T2.01 | Standard bars | feature_builder | S2 | 16 | T1.08 | [x] |
| T2.02 | Microstructure bars | feature_builder | S2 | 20 | T1.08 | [x] |
| T2.03 | As-of join logic | feature_builder | S2 | 12 | T2.01,T2.02 | [x] |
| T2.04 | Decision-frame matrix | feature_builder | S2 | 10 | T2.03 | [x] |
| T2.05 | Incremental computation | feature_builder | S2 | 8 | T2.04 | [x] |
| T2.06 | Forward returns | labeler | S2 | 8 | T2.04 | [x] |
| T2.07 | Direction labels | labeler | S2 | 6 | T2.06 | [x] |
| T2.08 | Net-of-spread returns | labeler | S2 | 4 | T2.07 | [x] |
| T2.09 | Feature builder tests | feature_builder | S2 | 4 | T2.05 | [x] |
| T2.10 | Labeler tests | labeler | S2 | 2 | T2.08 | [x] |
| T3.01 | Core simulation loop | backtester | S3 | 16 | T2.04 | [x] |
| T3.02 | Quote-based fills | backtester | S3 | 12 | T3.01 | [x] |
| T3.03 | Slippage model | backtester | S3 | 8 | T3.02,T1.04 | [x] |
| T3.04 | Position limits | backtester | S3 | 8 | T3.03 | [x] |
| T3.05 | Walk-forward framework | backtester | S3 | 12 | T3.04 | [x] |
| T3.06 | Metrics computation | backtester | S3 | 10 | T3.05 | [x] |
| T3.07 | Regime slicing | backtester | S3 | 12 | T3.06 | [x] |
| T3.08 | Purged CV | backtester | S3 | 10 | T3.07 | [x] |
| T3.09 | Integration tests | backtester | S3 | 8 | T3.08 | [x] |
| T3.10 | Validation vs provider | backtester | S3 | 4 | T3.09 | [x] |
| T4.01 | Strategy family framework | optimizer | S4 | 12 | T3.08 | [x] |
| T4.02 | Trend strategy | optimizer | S4 | 10 | T4.01 | [x] |
| T4.03 | Mean-reversion strategy | optimizer | S4 | 10 | T4.01 | [x] |
| T4.04 | Parameter sweep | optimizer | S4 | 12 | T4.02,T4.03 | [x] |
| T4.05 | XGBoost ML strategy | optimizer | S4 | 14 | T4.04 | [x] |
| T4.06 | GPU training support | optimizer | S4 | 10 | T4.05 | [x] |
| T4.07 | CPU fallback | optimizer | S4 | 6 | T4.06 | [x] |
| T4.08 | ONNX export | optimizer | S4 | 8 | T4.05 | [x] |
| T4.09 | Candidate ranking | optimizer | S4 | 8 | T4.08 | [x] |
| T4.10 | Cost sensitivity tests | optimizer | S4 | 8 | T4.09 | [x] |
| T4.11 | Optimizer tests | optimizer | S4 | 6 | T4.10 | [x] |
| T4.12 | GPU metadata logging | optimizer | S4 | 6 | T4.06 | [x] |
| T5.01 | Registry CRUD endpoints | registry_api | S5 | 16 | M4 | [x] |
| T5.02 | Dataset snapshots | registry_api | S5 | 8 | T5.01 | [x] |
| T5.03 | Artifact upload/download | registry_api | S5 | 10 | T5.02 | [x] |
| T5.04 | Gate evaluation | registry_api | S5 | 10 | T5.03 | [x] |
| T5.05 | Promotion state machine | registry_api | S5 | 8 | T5.04 | [x] |
| T5.06 | Audit log | registry_api | S5 | 6 | T5.05 | [x] |
| T5.07 | Feed health monitoring | monitor | S5 | 8 | T1.09 | [x] |
| T5.08 | Slippage tracking | monitor | S5 | 8 | T5.05 | [x] |
| T5.09 | Drawdown alerts | monitor | S5 | 6 | T5.08 | [x] |
| T5.10 | Prometheus metrics | monitor | S5 | 8 | T5.09 | [x] |
| T5.11 | Rollback triggers | monitor | S5 | 8 | T5.10 | [x] |
| T5.12 | Registry/monitor tests | monitor | S5 | 4 | T5.11 | [x] |
| T6.01 | Shadow execution mode | runner | S6 | 16 | M5 | [ ] |
| T6.02 | Signal logging | runner | S6 | 8 | T6.01 | [ ] |
| T6.03 | Paper execution | runner | S6 | 16 | T6.02 | [ ] |
| T6.04 | Fill confirmation | runner | S6 | 10 | T6.03 | [ ] |
| T6.05 | Autonomous orchestration | runner | S6 | 12 | T6.04 | [ ] |
| T6.06 | Shadow→paper promotion | runner | S6 | 10 | T6.05 | [ ] |
| T6.07 | Shadow vs backtest validation | runner | S6 | 12 | T6.06 | [ ] |
| T6.08 | Paper slippage analysis | runner | S6 | 8 | T6.07 | [ ] |
| T6.09 | E2E integration tests | runner | S6 | 12 | T6.08 | [ ] |
| T6.10 | Performance benchmarking | runner | S6 | 8 | T6.09 | [ ] |
| T6.11 | Documentation update | docs | S6 | 4 | T6.10 | [ ] |
| T6.12 | Acceptance testing | qa | S6 | 4 | T6.11 | [ ] |

---

## 5. Milestones & Gates

| Milestone | Sprint End | Gate Criteria |
|-----------|------------|---------------|
| **M1** | S1 | Historical data ingested (30 days SPY, QQQ); real-time streaming <500ms latency |
| **M2** | S2 | Feature pipeline produces decision frames; zero leakage verified |
| **M3** | S3 | Backtester reproduces known benchmark within 5% Sharpe tolerance |
| **M4** | S4 | Optimizer generates ≥10 candidates per family; GPU training functional |
| **M5** | S5 | Registry stores full audit trail; monitoring detects feed outages |
| **M6** | S6 | Autonomous loop runs unattended for 5 days; shadow matches backtest ±10% |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Alpaca API rate limits | High | Medium | Implement backoff, request batching |
| R2 | Data quality issues (gaps, outliers) | High | Medium | Validate against provider bars, gap detection |
| R3 | Feature leakage undetected | Critical | Low | Automated as-of validation tests |
| R4 | Backtest overfitting | High | Medium | Purged CV, deflated Sharpe, cost sensitivity |
| R5 | GPU driver compatibility | Medium | Low | CPU fallback mandatory, version pinning |
| R6 | WebSocket disconnections | Medium | Medium | Auto-reconnect with exponential backoff |
| R7 | Database schema migration issues | Medium | Low | Alembic migrations, rollback scripts |
| R8 | Slippage model inaccuracy | High | Medium | Calibrate against paper fills iteratively |
| R9 | Team velocity variance | Medium | Medium | 20% sprint buffer, scope flexibility |
| R10 | Third-party dependency breaking changes | Medium | Low | Pin versions, integration tests |

---

## 7. Definition of Done

### Phase 1 Complete When:

**Data Pipeline:**
- [ ] 30+ days historical trades + quotes ingested
- [ ] Real-time streaming operational <500ms latency
- [ ] All Parquet schemas match PRD.md exactly
- [ ] Feature pipeline zero-leakage certified

**Evaluation Engine:**
- [ ] Backtester uses quote-based fills
- [ ] Slippage model calibrated
- [ ] Walk-forward with purged CV implemented
- [ ] Metrics match EVAL_PROTOCOL.md thresholds

**Optimization:**
- [ ] ≥3 strategy families implemented
- [ ] GPU training with CPU fallback functional
- [ ] ONNX export working
- [ ] Candidate ranking operational

**Operations:**
- [ ] Registry stores complete audit trail
- [ ] Monitoring detects anomalies and triggers alerts
- [ ] Shadow execution matches backtest ±10%
- [ ] Paper execution fills within 2x expected slippage

**Quality:**
- [ ] ≥80% test coverage
- [ ] All type hints passing mypy
- [ ] Documentation updated
- [ ] No critical security vulnerabilities

---

## 8. Appendix

### A. Service Dependencies

```
common_types ─┬─► ingestor_py ─► feature_builder_py ─► labeler_py
              │                         │
cost_models ──┴─────────────────────────┼──► backtester_py ─► optimizer_py
                                        │                          │
risk_models ────────────────────────────┘                          │
                                                                   ▼
                                        registry_api_py ◄──────────┘
                                               │
                                               ▼
                                          monitor_py
                                               │
                                               ▼
                                        runner (shadow/paper)
```

### B. Data Flow

```
Alpaca API ──► ingestor ──► lake/raw/ ──► feature_builder ──► lake/features/
                                                    │
                                                    ▼
                                               labeler ──► lake/labels/
                                                    │
                                                    ▼
                backtester ◄── optimizer ──► registry ──► monitor
                    │                            │
                    ▼                            ▼
               metrics                    shadow/paper runner
```

### C. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Data | Polars, PyArrow, Parquet |
| ML | XGBoost, scikit-learn, ONNX |
| API | FastAPI, Pydantic |
| Database | PostgreSQL 16, SQLAlchemy |
| Object Store | MinIO (S3-compatible) |
| Broker | alpaca-py |
| Monitoring | Prometheus, structlog |
| Queue | Celery, Redis |

---

*Document Version: 1.0*
*Last Updated: Phase 1 Planning*
