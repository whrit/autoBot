# Autonomous Equities Taker Engine (Phase 1 Scaffold)

This repo is a **Phase 1** scaffold for an autonomous equities taker engine using **Alpaca**.

Key properties:
- Ingest **trades + quotes** (ticks) into an immutable Parquet lake
- Build **microstructure-aware multi-timeframe features**
- Evaluate a **strategy library**
- **Shadow + paper** execute with quote-based taker fill modeling
- Full auditability through Postgres registry

## Quickstart (local)

### 1) Start infra
```bash
docker compose -f infra/docker-compose.yml up -d
```

### 2) Create env
```bash
cp .env.example .env
```

### 3) Create venv + install
```bash
uv venv
uv sync --extra dev --extra data --extra services --extra db --extra telemetry
```

### 4) Run registry API (example)
```bash
uv run --package registry_api_py python -m registry_api_py.app
```

> Note: this scaffold intentionally includes minimal code stubs. Fill in service logic incrementally.
