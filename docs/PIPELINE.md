# autoBot End-to-End Pipeline Documentation

Complete guide for running the automated trading pipeline from data ingestion to backtesting and optimization.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration Reference](#configuration-reference)
- [Cost Models](#cost-models)
- [Risk Management](#risk-management)
- [Walk-Forward Optimization](#walk-forward-optimization)
- [Parallel Processing](#parallel-processing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The autoBot pipeline consists of four stages:

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│ INGEST  │───▶│ FEATURES │───▶│ BACKTEST  │───▶│ OPTIMIZE │
└─────────┘    └──────────┘    └───────────┘    └──────────┘
     │              │               │                │
     ▼              ▼               ▼                ▼
  lake/raw/    lake/features/   Results &       Best params
  (Hive fmt)   (Parquet)        Metrics         & configs
```

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **ingest** | Fetch market data from providers | API credentials | `lake/raw/` (Hive partitioned) |
| **features** | Build technical indicators | Raw OHLCV data | `lake/features/` (Parquet) |
| **backtest** | Run strategy simulation | Features + Signals | Performance metrics |
| **optimize** | Grid search parameters | Backtest engine | Optimal parameters |

---

## Quick Start

### Prerequisites

```bash
# Ensure you're in the autoBot root directory
cd /home/bw/Projects/autoBot

# Install dependencies (if not already done)
uv sync
```

### Run Full Pipeline

```bash
# Default: SPY, QQQ with 30 days of data
uv run python scripts/run_pipeline.py

# Skip ingestion (use existing data)
uv run python scripts/run_pipeline.py --skip-ingest

# Custom symbols
uv run python scripts/run_pipeline.py --symbols SPY,QQQ,AAPL,MSFT --skip-ingest
```

### Run Individual Stages

```bash
# Ingest only
uv run python scripts/run_pipeline.py ingest --days 60

# Features only
uv run python scripts/run_pipeline.py features

# Backtest only
uv run python scripts/run_pipeline.py backtest

# Optimize only
uv run python scripts/run_pipeline.py optimize
```

---

## Pipeline Stages

### Stage 1: Data Ingestion

Fetches OHLCV data from market data providers and stores in Hive-partitioned format.

```bash
uv run python scripts/run_pipeline.py ingest [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 30 | Number of days to backfill |
| `--symbols` | SPY,QQQ | Comma-separated symbols |

**Output Structure:**
```
lake/raw/bars_provider/
├── dt=2024-01-15/
│   ├── symbol=SPY/
│   │   └── data.parquet
│   └── symbol=QQQ/
│       └── data.parquet
└── dt=2024-01-16/
    └── ...
```

**Direct Usage:**
```bash
uv run -m ingestor_py.main backfill --days 30 --symbols SPY,QQQ
```

---

### Stage 2: Feature Building

Computes technical indicators and derived features from raw OHLCV data.

```bash
uv run python scripts/run_pipeline.py features [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | SPY,QQQ | Symbols to process |
| `--lake` | lake | Data lake path |

**Features Generated:**
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, volatility metrics
- Volume indicators
- Custom micro-features (30s granularity)

**Direct Usage:**
```bash
uv run -m feature_builder_py.main SPY QQQ \
  --data-dir lake \
  --output-dir lake/features \
  --large-dataset
```

---

### Stage 3: Backtesting

Runs strategy simulation with configurable cost models and risk management.

```bash
uv run python scripts/run_pipeline.py backtest [OPTIONS]
```

**Core Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | SPY,QQQ | Symbols to backtest |
| `--lake` | lake | Data lake path |
| `--strategy` | Momentum | Strategy name |
| `--start-date` | None | Start date (YYYY-MM-DD) |
| `--end-date` | None | End date (YYYY-MM-DD) |

**Cost Model Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--cost-model` | fixed | Model type: `fixed`, `volume`, `zero` |
| `--fee-rate` | 0.001 | Fee rate (0.001 = 0.1%) |
| `--slippage-rate` | 0.0005 | Slippage rate for fixed model |

**Risk Management Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--max-position-pct` | 0.10 | Max position as % of capital (10%) |
| `--max-drawdown-pct` | 0.20 | Max drawdown limit (20%) |

**Walk-Forward Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--walk-forward` | False | Enable walk-forward optimization |
| `--train-size` | 500 | Training window size (bars) |
| `--test-size` | 100 | Test window size (bars) |
| `--parallel` | False | Enable parallel processing |
| `--workers` | auto | Number of workers (default: CPU count) |

**Direct Usage:**
```bash
uv run -m backtester_py.main \
  --lake lake \
  --symbols SPY,QQQ \
  --strategy Momentum \
  --cost-model fixed \
  --fee-rate 0.001 \
  --max-position-pct 0.10 \
  --demo
```

---

### Stage 4: Optimization

Grid search over strategy parameters to find optimal configuration.

```bash
uv run python scripts/run_pipeline.py optimize [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | SPY,QQQ | Symbols to optimize |
| `--lake` | lake | Data lake path |
| `--workers` | auto | Parallel workers |

**Direct Usage:**
```bash
uv run -m backtester_py.main \
  --lake lake \
  --symbols SPY \
  --grid-search \
  --grid-metric sharpe_ratio \
  --grid-workers 8
```

---

## Configuration Reference

### Full CLI Reference

```bash
uv run python scripts/run_pipeline.py [STAGES] [OPTIONS]

STAGES:
  ingest      Run data ingestion
  features    Run feature building
  backtest    Run backtesting
  optimize    Run parameter optimization
  (none)      Run all stages

GENERAL OPTIONS:
  --symbols SYMBOLS         Comma-separated symbols (default: SPY,QQQ)
  --days DAYS               Lookback days for ingestion (default: 30)
  --lake PATH               Data lake path (default: lake)
  --skip-ingest             Skip data ingestion stage
  --workers N               Parallel workers (-1 = auto)
  --start-date YYYY-MM-DD   Start date filter
  --end-date YYYY-MM-DD     End date filter

BACKTEST OPTIONS:
  --walk-forward            Enable walk-forward optimization
  --train-size N            Training window size (default: 500)
  --test-size N             Test window size (default: 100)

COST MODEL OPTIONS:
  --cost-model TYPE         fixed | volume | zero (default: fixed)
  --fee-rate RATE           Fee rate decimal (default: 0.001)
  --slippage-rate RATE      Slippage rate (default: 0.0005)

RISK MANAGEMENT OPTIONS:
  --max-position-pct PCT    Max position % (default: 0.10)
  --max-drawdown-pct PCT    Max drawdown % (default: 0.20)
```

---

## Cost Models

The backtester includes three production-ready cost models:

### Fixed Cost Model (Default)

Simple fixed-percentage fees and slippage.

```bash
--cost-model fixed --fee-rate 0.001 --slippage-rate 0.0005
```

**Formula:**
```
commission = price × quantity × fee_rate
slippage = price × slippage_rate
fill_price = price + slippage (buy) or price - slippage (sell)
```

**Use Cases:**
- Most retail brokers
- Simple simulations
- Quick prototyping

### Volume Impact Cost Model

Market impact based on volume participation with square-root law.

```bash
--cost-model volume --fee-rate 0.001
```

**Formula:**
```
participation = min(quantity / volume, volume_limit)
impact = impact_factor × √(participation) × volatility
slippage = price × impact
```

**Use Cases:**
- Large orders
- Institutional trading
- Realistic HFT simulation

### Zero Cost Model

No transaction costs (for benchmarking).

```bash
--cost-model zero
```

**Use Cases:**
- Strategy development
- Baseline comparison
- Signal quality testing

### Cost Model Comparison

| Model | Fee Rate | Slippage | Best For |
|-------|----------|----------|----------|
| `fixed` | 0.1% | 0.05% | Retail trading |
| `volume` | 0.1% | Dynamic | Institutional |
| `zero` | 0% | 0% | Benchmarking |

---

## Risk Management

### Risk Checker

Pre-trade validation against configurable limits:

| Rule | Default | Description |
|------|---------|-------------|
| `max_position_pct` | 10% | Single position size limit |
| `max_order_value` | $100,000 | Maximum order value |
| `max_daily_trades` | 100 | Daily trade count limit |
| `max_drawdown_pct` | 20% | Maximum portfolio drawdown |

### Configuration Examples

**Conservative (Low Risk):**
```bash
--max-position-pct 0.05 --max-drawdown-pct 0.10
```

**Moderate:**
```bash
--max-position-pct 0.10 --max-drawdown-pct 0.20
```

**Aggressive:**
```bash
--max-position-pct 0.20 --max-drawdown-pct 0.30
```

### Position Sizing

The `PositionSizer` calculates optimal position size based on:

```python
risk_amount = account_balance × risk_per_trade_pct  # e.g., 2%
shares = risk_amount / stop_distance
position_size = min(shares × price, max_position_pct × account)
```

---

## Walk-Forward Optimization

Rolling window validation that prevents overfitting.

### How It Works

```
|------ Train ------|-- Test --|
                    |------ Train ------|-- Test --|
                                        |------ Train ------|-- Test --|
```

1. Train on window 1 → Test on window 1
2. Roll forward
3. Train on window 2 → Test on window 2
4. Repeat until data exhausted
5. Aggregate out-of-sample results

### Configuration

```bash
uv run python scripts/run_pipeline.py backtest \
  --walk-forward \
  --train-size 500 \
  --test-size 100 \
  --parallel \
  --workers 8
```

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `train-size` | 252-500 | ~1-2 years of daily bars |
| `test-size` | 21-63 | ~1-3 months of daily bars |
| `workers` | CPU count | Parallel fold processing |

### Performance

With parallel processing enabled:
- **11 workers**: ~3.73x speedup
- **155 folds**: ~2.9 seconds total

---

## Parallel Processing

### Automatic Detection

```bash
--workers -1  # Auto-detect CPU count (default)
--workers 8   # Explicit worker count
```

### What's Parallelized

| Component | Parallel | Notes |
|-----------|----------|-------|
| Walk-forward folds | ✅ | ProcessPoolExecutor |
| Grid search trials | ✅ | Per-worker batches |
| Feature building | ✅ | Polars streaming |
| Data ingestion | ❌ | API rate limited |

### Memory Considerations

```bash
# For large datasets, reduce workers to manage memory
--workers 4

# Monitor memory usage
htop  # or: watch -n 1 free -h
```

---

## Examples

### Example 1: Quick Development Backtest

```bash
# Fast iteration with zero costs
uv run python scripts/run_pipeline.py backtest \
  --symbols SPY \
  --cost-model zero \
  --skip-ingest
```

### Example 2: Realistic Production Simulation

```bash
# Full pipeline with realistic settings
uv run python scripts/run_pipeline.py \
  --symbols SPY,QQQ,IWM \
  --days 60 \
  --cost-model volume \
  --fee-rate 0.0015 \
  --max-position-pct 0.08 \
  --max-drawdown-pct 0.15
```

### Example 3: Walk-Forward Validation

```bash
# Rigorous out-of-sample testing
uv run python scripts/run_pipeline.py backtest \
  --symbols SPY \
  --walk-forward \
  --train-size 252 \
  --test-size 63 \
  --parallel \
  --workers 8 \
  --cost-model fixed \
  --fee-rate 0.001
```

### Example 4: Parameter Optimization

```bash
# Grid search for optimal parameters
uv run python scripts/run_pipeline.py optimize \
  --symbols SPY \
  --workers 12
```

### Example 5: Multi-Symbol with Date Range

```bash
# Specific date range across multiple symbols
uv run python scripts/run_pipeline.py backtest \
  --symbols SPY,QQQ,AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --cost-model fixed \
  --max-position-pct 0.05
```

### Example 6: Conservative Risk Profile

```bash
# Low-risk configuration
uv run python scripts/run_pipeline.py backtest \
  --symbols SPY \
  --cost-model fixed \
  --fee-rate 0.002 \
  --slippage-rate 0.001 \
  --max-position-pct 0.03 \
  --max-drawdown-pct 0.10
```

---

## Troubleshooting

### Common Issues

#### "No data specified"

```
Error: No data specified. Use --data, --lake, or --demo
```

**Solution:** Ensure you have data in the lake or use `--demo`:
```bash
# Check if data exists
ls -la lake/raw/bars_provider/

# Run ingestion first
uv run python scripts/run_pipeline.py ingest --days 30
```

#### "Symbol not found"

```
Warning: No data found for symbol XYZ
```

**Solution:** Verify symbol is available from your data provider:
```bash
# Check available symbols
ls lake/raw/bars_provider/*/symbol=*/
```

#### Memory Issues with Large Datasets

```
MemoryError or system becomes unresponsive
```

**Solution:** Reduce parallelism or use streaming:
```bash
# Reduce workers
--workers 2

# Or run sequentially
# (remove --parallel flag)
```

#### Walk-Forward Too Slow

**Solution:** Enable parallel processing:
```bash
uv run python scripts/run_pipeline.py backtest --walk-forward --parallel --workers 8
```

#### Pickling Errors in Parallel Mode

```
Can't pickle local object
```

**Solution:** This was fixed by moving mock classes to module level. Ensure you're using the latest code.

### Verification Commands

```bash
# Check backtester help
cd services/backtester_py
uv run python -m backtester_py.main --help

# Run model tests
uv run pytest tests/test_models/ -v

# Check imports
uv run python -c "from backtester_py.models import *; print('OK')"

# Dry run (demo mode)
uv run python -m backtester_py.main --demo --cost-model fixed
```

### Log Locations

| Component | Log Location |
|-----------|--------------|
| Pipeline | stdout |
| Backtester | stdout (rich console) |
| Feature builder | stdout |
| Ingestor | stdout |

---

## Architecture

### Module Structure

```
autoBot/
├── scripts/
│   └── run_pipeline.py          # Pipeline orchestrator
├── lake/                         # Data lake (Hive partitioned)
│   ├── raw/                      # Raw OHLCV data
│   └── features/                 # Computed features
└── services/
    ├── ingestor_py/              # Data ingestion
    ├── feature_builder_py/       # Feature engineering
    ├── backtester_py/            # Backtesting engine
    │   └── backtester_py/
    │       ├── main.py           # CLI entry point
    │       ├── walk_forward.py   # Walk-forward optimization
    │       ├── models/           # Cost & risk models
    │       │   ├── cost_model.py
    │       │   └── risk_checker.py
    │       └── gpu/              # GPU acceleration
    └── optimizer_py/             # Parameter optimization
```

### Data Flow

```
[Market Data APIs]
        │
        ▼
[ingestor_py] ──────▶ lake/raw/ (Hive: dt=YYYY-MM-DD/symbol=XXX)
        │
        ▼
[feature_builder_py] ▶ lake/features/ (Parquet)
        │
        ▼
[backtester_py] ─────▶ Performance Metrics
        │
        ▼
[optimizer_py] ──────▶ Optimal Parameters
```

---

## API Reference

### Cost Model Classes

```python
from backtester_py.models import (
    CostModel,          # Abstract base class
    FixedCostModel,     # Fixed % fees
    VolumeImpactCostModel,  # Volume-dependent
    ZeroCostModel,      # No costs
    create_cost_model,  # Factory function
)

# Create via factory
cost_model = create_cost_model(
    model_type="fixed",
    fee_rate=0.001,
    slippage_rate=0.0005,
)

# Or directly
cost_model = FixedCostModel(
    fee_rate=Decimal("0.001"),
    slippage_rate=Decimal("0.0005"),
)
```

### Risk Management Classes

```python
from backtester_py.models import (
    RiskChecker,        # Order validation
    PositionSizer,      # Position sizing
    RiskManager,        # Combined
    create_risk_checker,  # Factory function
)

# Create via factory
risk_checker = create_risk_checker(
    max_position_pct=0.10,
    max_drawdown_pct=0.20,
)

# Check order
violations = risk_checker.check_order(
    symbol="SPY",
    side="buy",
    notional=Decimal("10000"),
    portfolio_value=Decimal("100000"),
    current_drawdown=0.05,
)

if not violations:
    # Order is valid
    pass
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial pipeline |
| 1.1 | 2024-01 | Added Hive partition support |
| 1.2 | 2024-01 | Added parallel walk-forward |
| 1.3 | 2024-01 | Added GPU acceleration |
| 1.4 | 2024-01 | Added production cost models & risk management |
