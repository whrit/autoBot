# autoBot - Autonomous Equities Taker Trading Engine

**Phase 1 Trading Engine Documentation Hub**

---

## Project Overview

autoBot is a **production-grade autonomous equities taker trading engine** that continuously executes a self-iterating trading loop:

```
ingest -> build features -> generate strategies -> backtest -> select -> shadow -> paper trade -> monitor -> iterate
```

### What is Phase 1?

Phase 1 delivers a **fully autonomous closed-loop system** with:

- **Quotes + trades ingestion** from Alpaca API
- **Multi-timeframe microstructure-aware feature construction**
- **Strategy library evaluation** (trend, mean-reversion, ML)
- **Shadow + paper execution** with realistic taker fills
- **GPU-accelerated ML training** with CPU fallback

**No live capital is deployed in Phase 1** - this is a research and validation environment.

### Key Features

| Feature | Description |
|---------|-------------|
| **Microstructure-Aware** | Derives signals from quotes + trades, not just OHLCV bars |
| **Multi-Timeframe** | Combines 5s/15s/30s micro-bars with 1m/5m/15m standard bars |
| **Quote-Based Fills** | Realistic taker execution (buy @ ask, sell @ bid + slippage) |
| **Walk-Forward Testing** | Purged cross-validation with regime slicing |
| **GPU Acceleration** | XGBoost GPU training with transparent CPU fallback |
| **Full Audit Trail** | Every decision tracked in PostgreSQL registry |
| **Automatic Promotion** | Shadow -> Paper with configurable gates |

---

## Architecture

```
                                    +------------------+
                                    |   Alpaca API     |
                                    | (REST + WebSocket)|
                                    +--------+---------+
                                             |
                         +-------------------v-------------------+
                         |            ingestor_py               |
                         |  - Historical trades/quotes/bars     |
                         |  - Real-time WebSocket streaming     |
                         |  - Parquet writer with partitioning  |
                         +-------------------+-------------------+
                                             |
                                             v
                    +------------------------+------------------------+
                    |                                                 |
          +---------v---------+                            +----------v----------+
          | lake/raw/trades/  |                            | lake/raw/quotes/    |
          | lake/raw/quotes/  |                            | lake/raw/bars/      |
          +-------------------+                            +---------------------+
                    |                                                 |
                    +------------------------+------------------------+
                                             |
                         +-------------------v-------------------+
                         |         feature_builder_py           |
                         |  - Microstructure bars (5s/15s/30s)  |
                         |  - Standard bars (1m/5m/15m)         |
                         |  - Decision-frame as-of joins        |
                         +-------------------+-------------------+
                                             |
                    +------------------------+------------------------+
                    |                                                 |
          +---------v---------+                            +----------v----------+
          | lake/features/    |                            |    labeler_py       |
          | micro_bars/       |                            | - Forward returns   |
          | bars/             |                            | - Direction labels  |
          | decision_frame/   |                            | - Net-of-spread     |
          +-------------------+                            +----------+----------+
                                                                      |
                                                           +----------v----------+
                                                           |   lake/labels/      |
                                                           +----------+----------+
                                                                      |
                         +--------------------------------------------+
                         |
          +--------------v--------------+
          |       backtester_py         |
          | - Quote-based fills         |
          | - Slippage modeling         |
          | - Walk-forward evaluation   |
          | - Regime slicing            |
          | - Metrics computation       |
          +--------------+--------------+
                         |
          +--------------v--------------+
          |       optimizer_py          |
          | - Strategy families         |
          | - Parameter sweeps          |
          | - XGBoost GPU training      |
          | - ONNX export               |
          | - Candidate ranking         |
          +--------------+--------------+
                         |
          +--------------v--------------+
          |      registry_api_py        |
          | - Strategy CRUD             |
          | - Artifact storage          |
          | - Promotion management      |
          | - Audit logging             |
          +--------------+--------------+
                         |
          +--------------v--------------+        +--------------+
          |        runner_py            |<------>|  monitor_py  |
          | - Shadow execution          |        | - Health     |
          | - Paper execution           |        | - Slippage   |
          | - Signal generation         |        | - Drawdown   |
          | - Fill confirmation         |        | - Rollback   |
          +-----------------------------+        +--------------+
```

---

## Documentation Index

### Getting Started

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](./QUICKSTART.md) | Step-by-step setup guide for local development |
| [CONFIGURATION.md](./CONFIGURATION.md) | Environment variables and configuration options |

### Architecture & Design

| Document | Description |
|----------|-------------|
| [PRD.md](./PRD.md) | Product Requirements Document - system goals and specifications |
| [OUTLINE.md](./OUTLINE.md) | High-level system overview and design decisions |
| [PHASE1_PLAN.md](./PHASE1_PLAN.md) | Sprint plan, task tracking, and implementation status |

### Technical Reference

| Document | Description |
|----------|-------------|
| [PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md) | Data flow from ingestion to execution |
| [SERVICE_REFERENCE.md](./SERVICE_REFERENCE.md) | Detailed service documentation |
| [API_REFERENCE.md](./API_REFERENCE.md) | Registry API endpoints and schemas |
| [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) | Evaluation metrics and promotion gates |

### Operations

| Document | Description |
|----------|-------------|
| [OPERATIONS.md](./OPERATIONS.md) | Production deployment and monitoring |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Contributing guidelines and development workflow |

---

## Technology Stack

### Core Languages & Frameworks

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | All Phase 1 services |
| **FastAPI** | 0.128+ | Registry API server |
| **Pydantic** | 2.12+ | Data validation and settings |

### Data Processing

| Technology | Version | Purpose |
|------------|---------|---------|
| **Polars** | 1.37+ | High-performance DataFrames |
| **PyArrow** | 22.0+ | Parquet I/O and Arrow memory |
| **Pandas** | 2.3+ | Data manipulation (legacy support) |
| **NumPy** | 2.4+ | Numerical operations |

### Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| **XGBoost** | 3.1+ | Gradient boosting (GPU-accelerated) |
| **scikit-learn** | 1.8+ | ML utilities and preprocessing |
| **ONNX** | 1.20+ | Model serialization format |
| **ONNX Runtime** | 1.23+ | Model inference |

### Database & Storage

| Technology | Version | Purpose |
|------------|---------|---------|
| **PostgreSQL** | 16+ | Registry, metrics, audit trail |
| **SQLAlchemy** | 2.0+ | ORM and database abstraction |
| **MinIO** | - | S3-compatible object storage |
| **Parquet** | - | Columnar data lake format |

### Broker Integration

| Technology | Version | Purpose |
|------------|---------|---------|
| **alpaca-py** | 0.43+ | Trading API client |

### Infrastructure

| Technology | Version | Purpose |
|------------|---------|---------|
| **Docker** | - | Containerization |
| **Celery** | 5.6+ | Task queue (async jobs) |
| **Redis** | 7.1+ | Message broker |
| **Prometheus** | - | Metrics collection |
| **structlog** | 25.5+ | Structured logging |

### Development Tools

| Technology | Version | Purpose |
|------------|---------|---------|
| **uv** | - | Package management |
| **pytest** | 9.0+ | Testing framework |
| **mypy** | 1.19+ | Static type checking |
| **ruff** | 0.14+ | Linting and formatting |

---

## Project Status

### Phase 1 Completion: 100%

All six sprints have been completed with full test coverage.

| Sprint | Focus | Status |
|--------|-------|--------|
| S1 | Foundation (Infrastructure + Ingestion) | Complete |
| S2 | Feature Pipeline (Multi-timeframe) | Complete |
| S3 | Backtesting Engine (Realistic fills) | Complete |
| S4 | Optimization Loop (GPU training) | Complete |
| S5 | Registry & Monitoring | Complete |
| S6 | Integration & Autonomous Loop | Complete |

### Test Coverage Summary

| Service | Tests | Status |
|---------|-------|--------|
| ingestor_py | 59 | Passing |
| feature_builder_py | 56 | Passing |
| labeler_py | 39 | Passing |
| backtester_py | 103 | Passing |
| optimizer_py | 244 | Passing |
| registry_api_py | 203 | Passing |
| monitor_py | 143 | Passing |
| runner_py | 300 | Passing |
| **Total** | **1,091** | **All Passing** |

### Code Quality

| Check | Status |
|-------|--------|
| Lint (ruff) | Passing |
| Types (mypy) | Passing (77 source files) |
| Tests (pytest) | 1,091 tests passing |

### Known Limitations

1. **No Live Capital** - Phase 1 is shadow/paper only
2. **IEX Data Feed** - Free tier uses IEX (not SIP consolidated tape)
3. **Single Symbol Focus** - Optimized for SPY/QQQ initially
4. **CPU Backtesting** - GPU acceleration for backtest not yet implemented

---

## Project Structure

```
autoBot/
+-- services/                    # Microservices
|   +-- ingestor_py/            # Market data ingestion
|   +-- feature_builder_py/     # Feature construction
|   +-- labeler_py/             # Target labeling
|   +-- backtester_py/          # Strategy backtesting
|   +-- optimizer_py/           # Strategy optimization
|   +-- registry_api_py/        # Central registry API
|   +-- monitor_py/             # System monitoring
|   +-- runner_py/              # Shadow/paper execution
|
+-- libs/                        # Shared libraries
|   +-- common_types/           # Pydantic models
|   +-- cost_models/            # Transaction cost models
|   +-- risk_models/            # Risk management
|
+-- infra/                       # Infrastructure
|   +-- docker-compose.yml      # Local development stack
|
+-- db/                          # Database
|   +-- registry_schema.sql     # PostgreSQL schema
|
+-- docs/                        # Documentation
|   +-- README.md               # This file
|   +-- PRD.md                  # Product requirements
|   +-- PHASE1_PLAN.md          # Implementation plan
|   +-- EVAL_PROTOCOL.md        # Evaluation gates
|
+-- lake/                        # Data lake (generated)
|   +-- raw/                    # Immutable raw data
|   +-- features/               # Computed features
|   +-- labels/                 # Prediction targets
|
+-- pyproject.toml              # Project configuration
+-- .env.example                # Environment template
```

---

## Quick Start

```bash
# 1. Start infrastructure
docker compose -f infra/docker-compose.yml up -d

# 2. Configure environment
cp .env.example .env
# Edit .env with your Alpaca API keys

# 3. Create virtual environment and install
uv venv
uv sync --extra dev --extra data --extra services --extra db --extra telemetry

# 4. Run database migrations
uv run alembic upgrade head

# 5. Start the registry API
uv run --package registry_api_py python -m registry_api_py.app

# 6. Run tests
uv run pytest
```

See [QUICKSTART.md](./QUICKSTART.md) for detailed setup instructions.

---

## Future Phases

### Phase 2 - Rust Execution Runtime

- Rust execution engine for live trading
- Strict pre-trade risk checks
- Live capital deployment
- Optional GPU inference support

### Phase 3 - Strategy Factory & Regime Allocation

- Multiple concurrent strategies
- Regime-based capital allocation
- Neural network strategies
- L2/full order book data

---

## License

Proprietary - All Rights Reserved

---

## Contact

- **Author**: Whrit
- **Repository**: Private

---

*Document Version: 1.0*
*Last Updated: 2026-01-15*
