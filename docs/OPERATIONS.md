# Operations Guide - Phase 1 Trading Engine

**Autonomous Equities Taker Trading Engine - Production Operations**

---

## Table of Contents

1. [Running the Pipeline](#1-running-the-pipeline)
2. [Monitoring & Observability](#2-monitoring--observability)
3. [Common Operations](#3-common-operations)
4. [Health Checks](#4-health-checks)
5. [Backup & Recovery](#5-backup--recovery)
6. [Troubleshooting Guide](#6-troubleshooting-guide)

---

## 1. Running the Pipeline

### 1.1 Starting Infrastructure

#### Prerequisites

- Docker and Docker Compose installed
- Alpaca API credentials (paper trading account)
- Python 3.12+ with virtual environment

#### Start Core Services

```bash
# Navigate to infrastructure directory
cd /home/bw/Projects/autoBot/infra

# Start all infrastructure services
docker-compose up -d

# Verify services are running
docker-compose ps
```

**Services Started:**
| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL | 5432 | Registry database (qt_registry) |
| MinIO | 9000/9001 | Object storage for artifacts |
| Redis | 6379 | Task queue backend |

#### Environment Variables

Create a `.env` file or export these variables:

```bash
# Alpaca Credentials (Required)
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"

# Data Lake Configuration
export LAKE_PATH="./lake"

# Symbols to Process
export SYMBOLS="SPY,QQQ"

# Database
export DATABASE_URL="postgresql://qt:qt@localhost:5432/qt_registry"

# MinIO
export MINIO_ENDPOINT="localhost:9000"
export MINIO_ACCESS_KEY="minio"
export MINIO_SECRET_KEY="minio123"

# Registry API
export ARTIFACT_STORAGE_PATH="./artifacts"
```

### 1.2 Running Historical Backfill

The ingestor service supports backfilling historical data from Alpaca.

```bash
# Set environment variables
export INGESTOR_MODE="backfill"
export BACKFILL_DAYS=30
export SYMBOLS="SPY,QQQ"

# Run backfill
cd /home/bw/Projects/autoBot/services/ingestor_py
python -m ingestor_py.main
```

**Backfill Options:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKFILL_DAYS` | 30 | Number of days to backfill |
| `SYMBOLS` | SPY,QQQ | Comma-separated symbol list |
| `LAKE_PATH` | ./lake | Path to data lake directory |

**Data Types Backfilled:**
- Trades: `lake/raw/trades/dt=YYYY-MM-DD/symbol=XXX/`
- Quotes: `lake/raw/quotes/dt=YYYY-MM-DD/symbol=XXX/`
- Bars: `lake/raw/bars/dt=YYYY-MM-DD/symbol=XXX/`

### 1.3 Starting Real-Time Streaming

```bash
# Set environment variables
export INGESTOR_MODE="stream"
export SYMBOLS="SPY,QQQ"

# Start streaming
cd /home/bw/Projects/autoBot/services/ingestor_py
python -m ingestor_py.main
```

**Streaming Behavior:**
- Connects to Alpaca WebSocket (IEX feed by default)
- Buffers data in batches of 1000 records
- Writes to Parquet files on buffer flush
- Auto-reconnects with exponential backoff (max 10 attempts)
- Captures both `ts_event` (exchange) and `ts_recv` (local receipt)

### 1.4 Running Feature Generation

```bash
cd /home/bw/Projects/autoBot/services/feature_builder_py

# Run feature builder (processes data from lake)
python -m feature_builder_py.main
```

**Feature Outputs:**
| Feature Type | Granularity | Location |
|--------------|-------------|----------|
| Standard Bars | 1m, 5m, 15m | `lake/features/bars/` |
| Microstructure Bars | 5s, 15s, 30s | `lake/features/micro_bars/` |
| Decision Frame | as-of joined | `lake/features/decision_frame/` |

**Incremental Processing:**

For streaming data, use the `IncrementalProcessor`:

```python
from feature_builder_py.incremental import IncrementalProcessor

processor = IncrementalProcessor()

# Process new trades
bars = processor.process_trades(new_trades_df, symbol="SPY")

# Process new quotes
micro_bars = processor.process_quotes(new_quotes_df, symbol="SPY")

# Check watermark (last processed timestamp)
watermark = processor.get_watermark("SPY")
```

### 1.5 Running Backtests

```bash
cd /home/bw/Projects/autoBot/services/backtester_py

# Run backtester
python -m backtester_py.main
```

**Programmatic Backtest:**

```python
from backtester_py.engine import BacktestEngine, BacktestConfig, Signal, SignalType
from cost_models import TransactionCostModel
from risk_models import RiskChecker

# Configure backtest
config = BacktestConfig(
    initial_capital=100_000.0,
    cost_model=TransactionCostModel(
        spread_multiplier=0.5,
        impact_multiplier=0.1,
        volatility_multiplier=0.2,
        fixed_cost_bps=0.1,
    ),
    risk_checker=RiskChecker(
        max_position_notional=50_000.0,
        max_total_notional=200_000.0,
        max_daily_loss=5_000.0,
        max_drawdown_pct=0.15,
    ),
    stop_loss_pct=0.02,
)

# Run backtest
engine = BacktestEngine(config)
result = engine.run(market_data_df, signals_list)

# Access results
print(f"Total P&L: {result.total_pnl}")
print(f"Total Trades: {result.total_trades}")
print(f"Risk Violations: {result.risk_violations}")
```

### 1.6 Running Optimization

```bash
cd /home/bw/Projects/autoBot/services/optimizer_py

# Run optimizer
python -m optimizer_py.main
```

**Parameter Sweep Example:**

```python
from optimizer_py.sweep import ParameterSweep, SweepConfig
from optimizer_py.strategies import TrendStrategy

config = SweepConfig(
    strategy_family="trend",
    parameter_grid={
        "lookback": [10, 20, 50, 100],
        "threshold": [0.5, 1.0, 1.5],
    },
    n_splits=5,
    metric="sharpe",
)

sweep = ParameterSweep(strategy_class=TrendStrategy, config=config)
result = sweep.grid_search(decision_frame)

print(f"Best Params: {result.best_params}")
print(f"Best Sharpe: {result.best_score:.3f}")
```

**GPU Training (Optional):**

```bash
# Enable GPU for XGBoost
export GPU_ENABLED=true
export GPU_DEVICE=0
export GPU_FALLBACK_CPU=true
```

### 1.7 Starting Shadow/Paper Execution

#### Shadow Mode (No Real Orders)

```python
from runner_py.shadow import ShadowExecutor, ShadowConfig
from runner_py.types import Signal, SignalDirection, MarketData

config = ShadowConfig(
    strategy_id="trend_v1",
    symbols=["SPY", "QQQ"],
    max_position_size=10_000.0,
    slippage_model_bps=2.0,
)

executor = ShadowExecutor(config)

# Execute signal in shadow mode
result = executor.execute(signal, market_data)

# Check simulated P&L
print(f"Total P&L: {executor.get_total_pnl()}")
print(f"Trade History: {executor.get_trade_history()}")
```

#### Autonomous Orchestration

```python
from runner_py.orchestrator import (
    AutonomousOrchestrator,
    OrchestratorConfig,
    TaskType,
    ScheduleFrequency,
)
import asyncio

config = OrchestratorConfig(
    market_open=time(9, 30),
    market_close=time(16, 0),
    feature_refresh_interval_minutes=5,
    signal_generation_interval_minutes=1,
)

orchestrator = AutonomousOrchestrator(config)

# Register custom task handlers
orchestrator.register_task(
    task_type=TaskType.FEATURE_REFRESH,
    handler=my_feature_refresh_function,
    frequency=ScheduleFrequency.MINUTELY,
)

# Start autonomous loop
asyncio.run(orchestrator.start())
```

---

## 2. Monitoring & Observability

### 2.1 Prometheus Metrics Endpoints

The monitor service exposes Prometheus-compatible metrics:

```python
from monitor_py.metrics import MetricsExporter

exporter = MetricsExporter()

# Get metrics in Prometheus format
metrics_bytes = exporter.get_metrics()
```

**Available Metrics:**

| Metric Name | Type | Description |
|-------------|------|-------------|
| `feed_lag_seconds` | Gauge | Current WebSocket feed lag |
| `feed_messages_per_second` | Gauge | Message rate |
| `feed_disconnects_total` | Counter | Total disconnection count |
| `feed_status` | Gauge | 0=disconnected, 1=degraded, 2=healthy |
| `slippage_bps` | Histogram | Fill slippage distribution |
| `slippage_avg_bps` | Gauge | Average slippage |
| `slippage_max_bps` | Gauge | Maximum slippage |
| `slippage_ratio` | Gauge | Actual/expected slippage ratio |
| `drawdown_current` | Gauge | Current drawdown percentage |
| `drawdown_max` | Gauge | Maximum drawdown percentage |
| `equity_current` | Gauge | Current portfolio equity |
| `equity_peak` | Gauge | Peak portfolio equity |
| `days_in_drawdown` | Gauge | Days in current drawdown |
| `alerts_triggered_total` | Counter | Total alerts by type |
| `rollbacks_triggered_total` | Counter | Total rollbacks |

### 2.2 Key Metrics to Watch

#### Feed Health

```python
from monitor_py.feed_health import FeedHealthMonitor, FeedHealthConfig

monitor = FeedHealthMonitor(FeedHealthConfig(
    max_lag_seconds=5.0,
    stale_threshold_seconds=30.0,
    disconnect_threshold_seconds=60.0,
))

# Record incoming messages
monitor.record_message(timestamp)

# Check status
status = monitor.get_status()
print(f"Status: {status.status.value}")
print(f"Lag: {status.current_lag_seconds}s")
print(f"Messages/sec: {status.messages_per_second}")
```

#### Slippage Tracking

```python
from monitor_py.slippage import SlippageTracker, SlippageConfig

tracker = SlippageTracker(SlippageConfig(
    expected_slippage_bps=2.0,
    alert_threshold_multiplier=2.0,
    window_size=100,
))

# Record fills
slippage_bps = tracker.record_fill(
    expected_price=450.50,
    actual_price=450.55,
    side="buy"
)

# Get statistics
stats = tracker.get_stats()
print(f"Avg Slippage: {stats.avg_slippage_bps:.2f} bps")
print(f"Ratio to Expected: {stats.ratio_to_expected:.2f}x")
print(f"Alert Triggered: {stats.alert_triggered}")
```

#### Drawdown Monitoring

```python
from monitor_py.drawdown import DrawdownMonitor, DrawdownConfig

monitor = DrawdownMonitor(
    DrawdownConfig(
        warning_threshold=0.05,    # 5%
        critical_threshold=0.10,   # 10%
        kill_switch_threshold=0.15 # 15%
    ),
    initial_equity=100_000.0
)

# Update with current equity
status = monitor.update(equity=95_000.0)

print(f"Current Drawdown: {status.current_drawdown:.1%}")
print(f"Alert Level: {status.alert_level}")
print(f"Days in Drawdown: {status.days_in_drawdown}")

# Check kill switch
if monitor.should_halt_trading():
    print("KILL SWITCH TRIGGERED - HALT TRADING")
```

### 2.3 Alert Thresholds

| Alert Type | Default Threshold | Action |
|------------|-------------------|--------|
| Feed Degraded | >5s lag | Log warning |
| Feed Disconnected | >30s no messages | Alert + reconnect |
| Slippage Warning | >2x expected | Log warning |
| Slippage Critical | >3x expected | Trigger rollback |
| Drawdown Warning | >5% | Log warning |
| Drawdown Critical | >10% | Alert operator |
| Drawdown Kill Switch | >15% | Halt trading + rollback |
| Feed Disconnects | >5 in window | Trigger rollback |

### 2.4 Structured Logging with structlog

All services use `structlog` for structured JSON logging:

```python
import structlog

logger = structlog.get_logger(__name__)

# Example log entries
logger.info(
    "trade_executed",
    symbol="SPY",
    side="buy",
    price=450.55,
    quantity=100,
    slippage_bps=2.5,
)

logger.warning(
    "slippage_elevated",
    avg_slippage_bps=5.2,
    threshold=4.0,
    action="monitoring",
)

logger.error(
    "feed_disconnected",
    disconnect_count=3,
    last_message_time="2024-01-15T10:30:00Z",
)
```

**Log Analysis Patterns:**

```bash
# Search for trade executions
grep '"event": "trade_executed"' logs/*.log | jq .

# Find slippage warnings
grep '"event": "slippage_elevated"' logs/*.log | jq -r '.avg_slippage_bps'

# Count disconnections
grep '"event": "feed_disconnected"' logs/*.log | wc -l

# Filter by log level
grep '"level": "error"' logs/*.log | jq .
```

---

## 3. Common Operations

### 3.1 Adding New Symbols to Universe

1. **Update configuration:**

```yaml
# configs/universe.yaml
symbols: [SPY, QQQ, IWM, DIA]  # Add new symbols
```

2. **Run backfill for new symbols:**

```bash
export SYMBOLS="IWM,DIA"
export BACKFILL_DAYS=30
export INGESTOR_MODE="backfill"

python -m ingestor_py.main
```

3. **Restart streaming with new symbols:**

```bash
export SYMBOLS="SPY,QQQ,IWM,DIA"
export INGESTOR_MODE="stream"

python -m ingestor_py.main
```

### 3.2 Re-running Backfill for Specific Dates

```python
from datetime import datetime, UTC
from ingestor_py.backfill import BackfillOrchestrator

orchestrator = BackfillOrchestrator(
    api_key="your-key",
    api_secret="your-secret",
    lake_path="./lake",
)

# Backfill specific date range
result = orchestrator.backfill(
    symbols=["SPY", "QQQ"],
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 1, 15, tzinfo=UTC),
    data_types=["trades", "quotes", "bars"],
)

print(f"Trades: {result['trades']['count']} records")
print(f"Quotes: {result['quotes']['count']} records")
print(f"Bars: {result['bars']['count']} records")
```

### 3.3 Promoting a Strategy from Shadow to Paper

#### Via Registry API

```bash
# Check strategy current state
curl http://localhost:8080/strategies/1

# Evaluate gates before promotion
curl -X POST http://localhost:8080/strategies/1/evaluate-gates \
  -H "Content-Type: application/json" \
  -d '{
    "sharpe": 0.75,
    "max_drawdown": 0.08,
    "num_trades": 150,
    "win_rate": 0.52
  }'

# Promote to shadow (requires all gates passed)
curl -X POST http://localhost:8080/api/v1/strategies/1/promote \
  -H "Content-Type: application/json" \
  -d '{"target_state": "shadow"}'

# After shadow validation period, promote to paper
curl -X POST http://localhost:8080/api/v1/strategies/1/promote \
  -H "Content-Type: application/json" \
  -d '{"target_state": "paper"}'
```

#### Via Python SDK

```python
from registry_api_py.promotions import PromotionManager, PromotionRequirements
from registry_api_py.database import get_db

manager = PromotionManager(
    requirements=PromotionRequirements(
        to_shadow={
            "min_sharpe": 0.5,
            "max_drawdown": 0.15,
            "min_trades": 100,
        },
        to_paper={
            "shadow_days": 5,
            "shadow_sharpe": 0.4,
            "no_alerts": True,
        },
    )
)

db = next(get_db())

# Check requirements
reqs_met = manager.check_requirements(strategy_id=1, target_state="shadow", db=db)
print(f"Requirements: {reqs_met}")

# Promote if all requirements met
if all(reqs_met.values()):
    result = manager.promote(strategy_id=1, target_state="shadow", db=db)
    print(f"Promotion: {result.message}")
```

**Strategy Lifecycle States:**

```
CANDIDATE ──► SHADOW ──► PAPER
    │           │          │
    └───────────┴──────────┴──► RETIRED
```

### 3.4 Rolling Back a Failed Strategy

```python
from registry_api_py.promotions import PromotionManager

manager = PromotionManager()
db = next(get_db())

# Demote from paper to shadow
result = manager.demote(
    strategy_id=1,
    reason="Slippage exceeded threshold (3.5x expected)",
    db=db,
)

print(f"Demoted: {result.from_state.value} -> {result.to_state.value}")
```

**Automatic Rollback via Monitor:**

```python
from monitor_py.rollback import RollbackManager, RollbackConfig

manager = RollbackManager(RollbackConfig(
    max_drawdown_trigger=0.15,
    max_slippage_ratio_trigger=3.0,
    feed_disconnect_trigger=5,
    cooldown_minutes=30,
    registry_url="http://localhost:8080",
))

# Check conditions and trigger rollback if needed
action = manager.check_rollback_conditions(
    drawdown_status=drawdown_monitor.get_status(),
    slippage_stats=slippage_tracker.get_stats(),
    feed_status=feed_monitor.get_status(),
    strategy_id=1,
)

if action and action.triggered:
    success = manager.execute_rollback(action)
    print(f"Rollback executed: {success}")
```

### 3.5 Clearing and Rebuilding Features

```bash
# Clear existing features
rm -rf lake/features/micro_bars/*
rm -rf lake/features/bars/*
rm -rf lake/features/decision_frame/*

# Rebuild from raw data
python -m feature_builder_py.main --full-rebuild
```

**Programmatic Rebuild:**

```python
from feature_builder_py.incremental import IncrementalProcessor
from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.micro_bars import MicrostructureBarBuilder
import polars as pl

# Reset processor state
processor = IncrementalProcessor()
processor.reset_all()

# Load raw data
trades = pl.read_parquet("lake/raw/trades/**/*.parquet")
quotes = pl.read_parquet("lake/raw/quotes/**/*.parquet")

# Rebuild features per symbol
for symbol in ["SPY", "QQQ"]:
    symbol_trades = trades.filter(pl.col("symbol") == symbol)
    symbol_quotes = quotes.filter(pl.col("symbol") == symbol)

    bars = processor.process_trades(symbol_trades, symbol)
    micro_bars = processor.process_quotes(symbol_quotes, symbol)

    if bars is not None:
        bars.write_parquet(f"lake/features/bars/{symbol}.parquet")
    if micro_bars is not None:
        micro_bars.write_parquet(f"lake/features/micro_bars/{symbol}.parquet")
```

---

## 4. Health Checks

### 4.1 Service Health Endpoints

#### Registry API

```bash
# Health check
curl http://localhost:8080/health
# Response: {"status": "ok"}

# List strategies
curl http://localhost:8080/strategies?limit=10

# List active promotions
curl http://localhost:8080/promotions?passed=true
```

### 4.2 Database Connectivity

```bash
# PostgreSQL health check
docker exec -it infra-postgres-1 pg_isready -U qt -d qt_registry
# Response: localhost:5432 - accepting connections

# Direct connection test
psql postgresql://qt:qt@localhost:5432/qt_registry -c "SELECT 1;"
```

**Python Health Check:**

```python
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://qt:qt@localhost:5432/qt_registry")

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database: HEALTHY")
except Exception as e:
    print(f"Database: UNHEALTHY - {e}")
```

### 4.3 MinIO Storage Health

```bash
# MinIO health check via mc client
mc admin info local

# Or via HTTP
curl -I http://localhost:9000/minio/health/live
# Response: HTTP/1.1 200 OK
```

**Python Health Check:**

```python
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False,
)

try:
    buckets = client.list_buckets()
    print(f"MinIO: HEALTHY - {len(buckets)} buckets")
except Exception as e:
    print(f"MinIO: UNHEALTHY - {e}")
```

### 4.4 WebSocket Connection Status

```python
from monitor_py.feed_health import FeedHealthMonitor, FeedStatus

monitor = FeedHealthMonitor()
status = monitor.get_status()

if status.status == FeedStatus.HEALTHY:
    print(f"WebSocket: HEALTHY - lag {status.current_lag_seconds:.2f}s")
elif status.status == FeedStatus.DEGRADED:
    print(f"WebSocket: DEGRADED - lag {status.current_lag_seconds:.2f}s")
else:
    print(f"WebSocket: DISCONNECTED - {status.message}")
```

### 4.5 Comprehensive Health Check Script

```python
#!/usr/bin/env python3
"""Comprehensive health check for Phase 1 trading engine."""

import sys
from datetime import datetime, UTC

def check_postgres():
    from sqlalchemy import create_engine, text
    try:
        engine = create_engine("postgresql://qt:qt@localhost:5432/qt_registry")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as e:
        return False, str(e)

def check_minio():
    from minio import Minio
    try:
        client = Minio("localhost:9000", "minio", "minio123", secure=False)
        client.list_buckets()
        return True, "Connected"
    except Exception as e:
        return False, str(e)

def check_redis():
    import redis
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True, "Connected"
    except Exception as e:
        return False, str(e)

def check_registry_api():
    import httpx
    try:
        response = httpx.get("http://localhost:8080/health", timeout=5.0)
        if response.status_code == 200:
            return True, "Healthy"
        return False, f"Status {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    print(f"Health Check - {datetime.now(UTC).isoformat()}")
    print("=" * 50)

    checks = [
        ("PostgreSQL", check_postgres),
        ("MinIO", check_minio),
        ("Redis", check_redis),
        ("Registry API", check_registry_api),
    ]

    all_healthy = True
    for name, check_fn in checks:
        try:
            healthy, message = check_fn()
        except Exception as e:
            healthy, message = False, str(e)

        status = "OK" if healthy else "FAIL"
        print(f"{name:20} [{status:4}] {message}")
        all_healthy = all_healthy and healthy

    print("=" * 50)
    if all_healthy:
        print("Overall: ALL SYSTEMS HEALTHY")
        sys.exit(0)
    else:
        print("Overall: SOME SYSTEMS UNHEALTHY")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 5. Backup & Recovery

### 5.1 Database Backup Procedures

#### Automated Backup Script

```bash
#!/bin/bash
# backup_postgres.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/qt_registry_${TIMESTAMP}.sql.gz"

# Create backup directory if needed
mkdir -p ${BACKUP_DIR}

# Dump and compress
docker exec infra-postgres-1 \
  pg_dump -U qt qt_registry | gzip > ${BACKUP_FILE}

# Verify backup
if [ -f "${BACKUP_FILE}" ] && [ -s "${BACKUP_FILE}" ]; then
    echo "Backup successful: ${BACKUP_FILE}"
    # Keep last 7 days of backups
    find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete
else
    echo "Backup failed!"
    exit 1
fi
```

#### Restore from Backup

```bash
# Stop dependent services first
docker-compose stop registry_api monitor

# Restore database
gunzip -c /backups/postgres/qt_registry_20240115_120000.sql.gz | \
  docker exec -i infra-postgres-1 psql -U qt -d qt_registry

# Restart services
docker-compose start registry_api monitor
```

### 5.2 Data Lake Backup

#### Backup to S3/MinIO

```bash
#!/bin/bash
# backup_lake.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="backups"

# Sync data lake to backup bucket
mc mirror --overwrite \
  ./lake \
  local/${BACKUP_BUCKET}/lake_${TIMESTAMP}/

# Verify
mc ls local/${BACKUP_BUCKET}/lake_${TIMESTAMP}/
```

#### Restore Data Lake

```bash
# Restore from MinIO backup
mc mirror --overwrite \
  local/backups/lake_20240115_120000/ \
  ./lake/
```

### 5.3 Artifact Backup

```bash
#!/bin/bash
# backup_artifacts.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/artifacts"

# Backup artifacts directory
tar -czvf ${BACKUP_DIR}/artifacts_${TIMESTAMP}.tar.gz ./artifacts/

# Backup MinIO artifacts bucket
mc mirror local/artifacts ${BACKUP_DIR}/minio_artifacts_${TIMESTAMP}/
```

### 5.4 Disaster Recovery Steps

#### Complete System Recovery

1. **Restore Infrastructure:**

```bash
# Start infrastructure
cd /home/bw/Projects/autoBot/infra
docker-compose up -d

# Wait for services to be ready
sleep 30
```

2. **Restore Database:**

```bash
# Find latest backup
LATEST_BACKUP=$(ls -t /backups/postgres/*.sql.gz | head -1)

# Restore
gunzip -c ${LATEST_BACKUP} | \
  docker exec -i infra-postgres-1 psql -U qt -d qt_registry
```

3. **Restore Data Lake:**

```bash
# Find latest backup
LATEST_LAKE=$(ls -dt /backups/lake_* | head -1)

# Restore
mc mirror --overwrite ${LATEST_LAKE} ./lake/
```

4. **Restore Artifacts:**

```bash
# Restore local artifacts
tar -xzvf /backups/artifacts/artifacts_latest.tar.gz -C ./

# Restore MinIO artifacts
mc mirror /backups/minio_artifacts_latest/ local/artifacts/
```

5. **Verify Recovery:**

```bash
# Run health checks
python health_check.py

# Verify data integrity
python -c "
import polars as pl
trades = pl.read_parquet('lake/raw/trades/**/*.parquet')
print(f'Trades: {len(trades)} records')
quotes = pl.read_parquet('lake/raw/quotes/**/*.parquet')
print(f'Quotes: {len(quotes)} records')
"
```

---

## 6. Troubleshooting Guide

### 6.1 Common Error Messages and Solutions

#### Alpaca API Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Invalid API credentials | Verify ALPACA_API_KEY and ALPACA_API_SECRET |
| `429 Too Many Requests` | Rate limit exceeded | Implement backoff, reduce request frequency |
| `403 Forbidden` | Paper vs Live API mismatch | Ensure using paper trading endpoint |
| `WebSocket closed` | Connection timeout | Auto-reconnect is built-in; check network |

#### Database Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `connection refused` | PostgreSQL not running | `docker-compose up -d postgres` |
| `relation does not exist` | Missing schema | Run `alembic upgrade head` |
| `unique constraint violation` | Duplicate entry | Check for existing record before insert |
| `deadlock detected` | Concurrent transactions | Implement retry logic |

#### Data Lake Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No such file or directory` | Missing partition | Verify backfill completed; run backfill |
| `Parquet read error` | Corrupt file | Delete and re-backfill specific date |
| `Schema mismatch` | Version incompatibility | Rebuild features from raw data |

### 6.2 Debug Logging

Enable debug logging for detailed diagnostics:

```python
import structlog
import logging

# Set root logger to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Configure structlog for debug output
structlog.configure(
    wrapper_class=structlog.stdlib.BoundLogger,
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
)
```

**Service-Specific Debug:**

```bash
# Ingestor debug
export LOG_LEVEL=DEBUG
python -m ingestor_py.main 2>&1 | tee ingestor_debug.log

# Backtester debug
export LOG_LEVEL=DEBUG
python -m backtester_py.main 2>&1 | tee backtester_debug.log
```

### 6.3 Performance Issues

#### Slow Backfill

**Symptoms:** Historical data ingestion taking hours

**Solutions:**
1. Reduce date range per batch
2. Process symbols in parallel
3. Check API rate limits

```python
# Parallel backfill
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_backfill(symbols, days=30):
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                backfill_symbol,
                symbol,
                days
            )
            for symbol in symbols
        ]
        await asyncio.gather(*tasks)
```

#### High Memory Usage

**Symptoms:** OOM errors during feature computation

**Solutions:**
1. Process data in chunks
2. Use streaming/lazy evaluation
3. Increase swap space

```python
# Chunk processing with Polars lazy evaluation
import polars as pl

# Use lazy API for large datasets
lazy_df = pl.scan_parquet("lake/raw/trades/**/*.parquet")

# Process in batches
for batch in lazy_df.collect(streaming=True):
    process_batch(batch)
```

#### Slow Backtests

**Symptoms:** Single backtest taking >5 minutes

**Solutions:**
1. Reduce data granularity
2. Use vectorized operations
3. Pre-compute features

```python
# Pre-compute decision frame
decision_frame = pl.read_parquet("lake/features/decision_frame/*.parquet")

# Cache frequently used data
MARKET_DATA_CACHE = {}

def get_market_data(symbol, date):
    key = f"{symbol}_{date}"
    if key not in MARKET_DATA_CACHE:
        MARKET_DATA_CACHE[key] = load_market_data(symbol, date)
    return MARKET_DATA_CACHE[key]
```

### 6.4 WebSocket Reconnection Issues

**Symptoms:** Frequent disconnections, data gaps

**Solutions:**

1. Check network stability
2. Verify API credentials are not expired
3. Review reconnection logs

```python
from ingestor_py.streaming import RealtimeStreamer

# Increase reconnection attempts
RealtimeStreamer.MAX_RECONNECT_ATTEMPTS = 20
RealtimeStreamer.RECONNECT_BASE_DELAY = 2.0  # Start with 2s delay

# Monitor reconnection count
streamer = RealtimeStreamer(...)
print(f"Reconnect attempts: {streamer.reconnect_attempts}")
```

### 6.5 Diagnostic Commands

```bash
# Check Docker container status
docker-compose ps
docker-compose logs --tail=100 postgres
docker-compose logs --tail=100 minio

# Check disk space
df -h ./lake/
du -sh ./lake/raw/*

# Check data file counts
find ./lake/raw/trades -name "*.parquet" | wc -l
find ./lake/raw/quotes -name "*.parquet" | wc -l

# Verify Parquet file integrity
python -c "
import polars as pl
try:
    df = pl.read_parquet('lake/raw/trades/dt=2024-01-15/symbol=SPY/*.parquet')
    print(f'Records: {len(df)}')
    print(f'Columns: {df.columns}')
except Exception as e:
    print(f'Error: {e}')
"

# Check database tables
docker exec -it infra-postgres-1 psql -U qt -d qt_registry -c "
SELECT table_name, pg_size_pretty(pg_total_relation_size(quote_ident(table_name)))
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC;
"

# Monitor memory usage
docker stats --no-stream

# Check process resource usage
ps aux | grep python
```

---

## Appendix: Quick Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPACA_API_KEY` | Yes | - | Alpaca API key |
| `ALPACA_API_SECRET` | Yes | - | Alpaca API secret |
| `LAKE_PATH` | No | ./lake | Data lake directory |
| `SYMBOLS` | No | SPY,QQQ | Comma-separated symbols |
| `INGESTOR_MODE` | No | stream | backfill or stream |
| `BACKFILL_DAYS` | No | 30 | Days to backfill |
| `DATABASE_URL` | No | postgresql://qt:qt@localhost:5432/qt_registry | Database connection |
| `ARTIFACT_STORAGE_PATH` | No | ./artifacts | Artifact storage path |
| `GPU_ENABLED` | No | false | Enable GPU training |
| `GPU_DEVICE` | No | 0 | GPU device ID |
| `GPU_FALLBACK_CPU` | No | true | Fall back to CPU if GPU unavailable |
| `LOG_LEVEL` | No | INFO | Logging level |

### Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| PostgreSQL | 5432 | TCP |
| MinIO API | 9000 | HTTP |
| MinIO Console | 9001 | HTTP |
| Redis | 6379 | TCP |
| Registry API | 8080 | HTTP |

### Default Thresholds

| Metric | Warning | Critical | Kill Switch |
|--------|---------|----------|-------------|
| Feed Lag | 5s | 30s | 60s |
| Slippage Ratio | 2x | 3x | - |
| Drawdown | 5% | 10% | 15% |
| Feed Disconnects | - | - | 5 |

---

*Document Version: 1.0*
*Last Updated: 2026-01-15*
