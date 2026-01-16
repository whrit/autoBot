# Phase 1 Trading Engine Configuration Reference

This document provides comprehensive configuration options for the Phase 1 autonomous equities trading engine.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Service Configuration Classes](#service-configuration-classes)
3. [Docker Compose Configuration](#docker-compose-configuration)
4. [Data Lake Structure](#data-lake-structure)
5. [Environment-Specific Recommendations](#environment-specific-recommendations)

---

## Environment Variables

### Alpaca API Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ALPACA_API_KEY_ID` | Alpaca API key identifier | - | Yes |
| `ALPACA_API_SECRET_KEY` | Alpaca API secret key | - | Yes |
| `ALPACA_ENV` | Trading environment (`paper` or `live`) | `paper` | No |

**Example:**
```bash
ALPACA_API_KEY_ID=your_key
ALPACA_API_SECRET_KEY=your_secret
ALPACA_ENV=paper
```

**Notes:**
- Always use `paper` for development and testing
- The `live` setting should only be used after thorough validation
- API keys can be obtained from [Alpaca Dashboard](https://app.alpaca.markets)

---

### Database Configuration (PostgreSQL)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PGHOST` | PostgreSQL server hostname | `localhost` | Yes |
| `PGPORT` | PostgreSQL server port | `5432` | No |
| `PGDATABASE` | Database name | `qt_registry` | Yes |
| `PGUSER` | Database username | `qt` | Yes |
| `PGPASSWORD` | Database password | `qt` | Yes |

**Example:**
```bash
PGHOST=localhost
PGPORT=5432
PGDATABASE=qt_registry
PGUSER=qt
PGPASSWORD=qt
```

---

### MinIO/S3 Data Lake Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `S3_ENDPOINT_URL` | MinIO/S3 endpoint URL | `http://localhost:9000` | Yes |
| `S3_ACCESS_KEY` | S3 access key ID | `minio` | Yes |
| `S3_SECRET_KEY` | S3 secret access key | `minio123` | Yes |
| `S3_BUCKET` | Bucket name for data lake | `lake` | Yes |

**Example:**
```bash
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET=lake
```

---

### Redis Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis server hostname | `localhost` | No |
| `REDIS_PORT` | Redis server port | `6379` | No |

---

### GPU Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GPU_ENABLED` | Enable GPU training | `true` | No |
| `GPU_DEVICE` | CUDA device ID | `0` | No |
| `GPU_FALLBACK_CPU` | Allow CPU fallback if GPU unavailable | `true` | No |

**Example:**
```bash
GPU_ENABLED=true
GPU_DEVICE=0
GPU_FALLBACK_CPU=true
```

**Valid Values:**
- `GPU_ENABLED`: `true`, `false`, `1`, `0`, `yes`, `no`
- `GPU_DEVICE`: Any valid CUDA device ID (typically `0`, `1`, `2`, etc.)
- `GPU_FALLBACK_CPU`: `true`, `false`, `1`, `0`, `yes`, `no`

---

### Registry API Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REGISTRY_BIND` | Bind address for Registry API | `0.0.0.0` | No |
| `REGISTRY_PORT` | Port for Registry API | `8080` | No |

---

## Service Configuration Classes

### Ingestor Service

#### AlpacaDataClient Configuration

```python
class AlpacaDataClient:
    """Client for fetching historical market data from Alpaca."""

    def __init__(
        self,
        api_key: str,              # Alpaca API key (required)
        api_secret: str,           # Alpaca API secret (required)
        feed: str = "iex",         # Data feed: 'iex' (free) or 'sip' (paid)
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | - | Alpaca API key |
| `api_secret` | `str` | - | Alpaca API secret |
| `feed` | `str` | `"iex"` | Data feed source (`iex` = free delayed, `sip` = paid real-time) |

---

#### RealtimeStreamer Configuration

```python
class RealtimeStreamer:
    """Real-time market data streaming via WebSocket."""

    # Class constants
    BATCH_SIZE = 1000                  # Batch size for parquet writes
    MAX_RECONNECT_ATTEMPTS = 10        # Maximum reconnection attempts
    RECONNECT_BASE_DELAY = 1.0         # Base delay for exponential backoff (seconds)

    def __init__(
        self,
        api_key: str,                   # Alpaca API key (required)
        api_secret: str,                # Alpaca API secret (required)
        lake_path: Path | str,          # Path to data lake directory (required)
        symbols: list[str],             # Symbols to stream (required)
        feed: str = "iex",              # Data feed source
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | - | Alpaca API key |
| `api_secret` | `str` | - | Alpaca API secret |
| `lake_path` | `Path \| str` | - | Base path for data lake storage |
| `symbols` | `list[str]` | - | List of symbols to stream |
| `feed` | `str` | `"iex"` | Data feed source |

---

### Feature Builder Service

#### StandardBarBuilder Configuration

```python
class StandardBarBuilder:
    """Build standard OHLCV bars from trade data."""

    def __init__(
        self,
        granularity: str = "1m",        # Bar interval
        atr_period: int = 14,           # ATR calculation period
    ) -> None
```

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `granularity` | `str` | `"1m"` | `"1m"`, `"5m"`, `"15m"` | Bar time interval |
| `atr_period` | `int` | `14` | Any positive integer | Period for ATR and volatility calculations |

**Output Columns:**
- `symbol`, `bar_start`, `bar_end`, `open`, `high`, `low`, `close`, `volume`, `returns`, `atr`, `realized_vol`

---

#### MicrostructureBarBuilder Configuration

```python
class MicrostructureBarBuilder:
    """Build high-frequency microstructure bars."""

    def __init__(
        self,
        granularity: str = "30s",       # Bar interval
    ) -> None
```

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `granularity` | `str` | `"30s"` | `"5s"`, `"15s"`, `"30s"` | Microstructure bar interval |

**Output Columns:**
- `symbol`, `bar_start`, `bar_end`, `vwap`, `midprice`, `microprice`, `spread`, `bid_size`, `ask_size`, `quote_imbalance`, `trade_imbalance`, `trade_volume`, `realized_vol`

---

### Labeler Service

#### ForwardReturnsCalculator Configuration

```python
class ForwardReturnsCalculator:
    """Calculate forward returns at configurable horizons."""

    def __init__(
        self,
        horizons: Sequence[int] | None = None,  # Horizon periods in seconds
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizons` | `Sequence[int] \| None` | `[60, 300, 900]` | Forward-looking horizons in seconds |

**Default Horizons:**
- `60` seconds (1 minute)
- `300` seconds (5 minutes)
- `900` seconds (15 minutes)

---

#### DirectionLabeler Configuration

```python
class DirectionLabeler:
    """Label forward returns with direction indicators."""

    def __init__(
        self,
        no_trade_threshold: float = 0.0005,  # Minimum return for trade signal
    ) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `no_trade_threshold` | `float` | `0.0005` | Minimum absolute return to trigger trade (5 basis points) |

**Label Mapping:**
- `+1`: Long signal (return > threshold)
- `0`: No-trade (return within +/- threshold)
- `-1`: Short signal (return < -threshold)

---

### Backtester Service

#### BacktestConfig

```python
@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float              # Starting capital in dollars (required)
    cost_model: TransactionCostModel    # Transaction cost model (required)
    risk_checker: RiskChecker           # Risk checker instance (required)
    default_order_notional: float = 10_000.0    # Default order size
    stop_loss_pct: float = 0.02                 # Stop-loss percentage (2%)
    take_profit_pct: float | None = None        # Take-profit percentage
    book_notional_default: float = 500_000.0    # Default book notional
    short_term_vol_default: float = 0.001       # Default volatility
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | `float` | - | Starting capital in USD |
| `cost_model` | `TransactionCostModel` | - | Transaction cost model instance |
| `risk_checker` | `RiskChecker` | - | Risk checker instance |
| `default_order_notional` | `float` | `10,000.0` | Default order size in USD |
| `stop_loss_pct` | `float` | `0.02` | Stop-loss threshold (2% = 0.02) |
| `take_profit_pct` | `float \| None` | `None` | Take-profit threshold (optional) |
| `book_notional_default` | `float` | `500,000.0` | Default top-of-book notional |
| `short_term_vol_default` | `float` | `0.001` | Default short-term volatility |

---

### Cost Models Library

#### SlippageModel Configuration

```python
@dataclass
class SlippageModel:
    """Slippage model based on spread, order size, and volatility."""

    spread_coef: float = 1.0            # Spread impact coefficient
    size_coef: float = 1.0              # Size impact coefficient
    vol_coef: float = 1.0               # Volatility impact coefficient
    max_size_impact_bps: float = 50.0   # Maximum size impact cap
```

**Slippage Formula:**
```
slippage_bps = spread_coef * spread_bps
             + size_coef * (order_notional / book_notional) * 100
             + vol_coef * short_term_vol * 100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spread_coef` | `float` | `1.0` | Coefficient for spread impact |
| `size_coef` | `float` | `1.0` | Coefficient for size impact |
| `vol_coef` | `float` | `1.0` | Coefficient for volatility impact |
| `max_size_impact_bps` | `float` | `50.0` | Maximum size impact in basis points |

---

#### TransactionCostModel Configuration

```python
@dataclass
class TransactionCostModel:
    """Full transaction cost model including slippage and fixed costs."""

    slippage_model: SlippageModel       # Slippage model instance (required)
    fixed_cost_bps: float = 0.0         # Fixed costs in basis points
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slippage_model` | `SlippageModel` | - | Slippage model instance |
| `fixed_cost_bps` | `float` | `0.0` | Fixed commission/fees in basis points |

**JSON Configuration File Format:**
```json
{
    "version": "cm_v1",
    "spread_coef": 1.5,
    "size_coef": 0.8,
    "vol_coef": 2.0,
    "max_size_impact_bps": 50.0,
    "fixed_cost_bps": 0.35
}
```

---

### Risk Models Library

#### RiskLimits Configuration

```python
@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_notional: float        # Max single position value (required)
    max_gross_exposure: float           # Max total absolute exposure (required)
    max_net_exposure: float             # Max net long/short exposure (required)
    max_daily_loss: float               # Max daily loss allowed (required)
    max_drawdown_pct: float             # Max drawdown as decimal (required)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `max_position_notional` | `float` | > 0 | Maximum notional for single position |
| `max_gross_exposure` | `float` | > 0 | Maximum total absolute exposure |
| `max_net_exposure` | `float` | > 0 | Maximum net long/short exposure |
| `max_daily_loss` | `float` | > 0 | Maximum daily loss threshold |
| `max_drawdown_pct` | `float` | 0-1 | Maximum drawdown percentage |

---

### Optimizer Service

#### GPUConfig

```python
@dataclass
class GPUConfig:
    """Configuration for GPU training."""

    enabled: bool = True               # Whether GPU training is enabled
    device_id: int = 0                 # CUDA device ID
    fallback_to_cpu: bool = True       # Allow CPU fallback
    memory_fraction: float = 0.8       # GPU memory fraction to use
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable GPU training |
| `device_id` | `int` | `0` | CUDA device ID (0, 1, 2, etc.) |
| `fallback_to_cpu` | `bool` | `True` | Fall back to CPU if GPU unavailable |
| `memory_fraction` | `float` | `0.8` | Fraction of GPU memory to use (0.0-1.0) |

---

#### RankingConfig

```python
@dataclass
class RankingConfig:
    """Configuration for candidate ranking."""

    metric: RankingMetric = RankingMetric.COMPOSITE
    min_trades: int = 30
    max_drawdown_threshold: float = 0.20
    sharpe_weight: float = 0.4
    sortino_weight: float = 0.3
    profit_factor_weight: float = 0.2
    drawdown_penalty_weight: float = 0.1
    complexity_config: ComplexityPenaltyConfig | None = None
    apply_complexity_penalty: bool = True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `RankingMetric` | `COMPOSITE` | Primary ranking metric |
| `min_trades` | `int` | `30` | Minimum trades required |
| `max_drawdown_threshold` | `float` | `0.20` | Maximum acceptable drawdown (20%) |
| `sharpe_weight` | `float` | `0.4` | Weight for Sharpe ratio |
| `sortino_weight` | `float` | `0.3` | Weight for Sortino ratio |
| `profit_factor_weight` | `float` | `0.2` | Weight for profit factor |
| `drawdown_penalty_weight` | `float` | `0.1` | Penalty weight for drawdown |
| `apply_complexity_penalty` | `bool` | `True` | Apply complexity penalties |

**Ranking Metrics:**
- `SHARPE`: Annualized Sharpe ratio
- `SORTINO`: Annualized Sortino ratio
- `CALMAR`: Calmar ratio (Sharpe / max_drawdown)
- `PROFIT_FACTOR`: Gross profit / gross loss
- `COMPOSITE`: Weighted composite score

---

### Monitor Service

#### FillRateConfig

```python
@dataclass
class FillRateConfig:
    """Configuration for fill rate tracking."""

    expected_fill_rate: float = 0.95
    warning_threshold: float = 0.90
    critical_threshold: float = 0.80
    window_size: int = 100
    latency_warning_ms: float = 500.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_fill_rate` | `float` | `0.95` | Expected fill rate (95%) |
| `warning_threshold` | `float` | `0.90` | Fill rate warning threshold (90%) |
| `critical_threshold` | `float` | `0.80` | Fill rate critical threshold (80%) |
| `window_size` | `int` | `100` | Rolling window size for tracking |
| `latency_warning_ms` | `float` | `500.0` | Latency warning threshold (ms) |

---

#### TurnoverConfig

```python
@dataclass
class TurnoverConfig:
    """Configuration for turnover analysis."""

    daily_turnover_warning: float = 0.20
    daily_turnover_critical: float = 0.50
    weekly_turnover_warning: float = 1.0
    cost_per_turnover_bps: float = 5.0
    window_days: int = 30
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `daily_turnover_warning` | `float` | `0.20` | Daily turnover warning (20% of portfolio) |
| `daily_turnover_critical` | `float` | `0.50` | Daily turnover critical (50% of portfolio) |
| `weekly_turnover_warning` | `float` | `1.0` | Weekly turnover warning (100% of portfolio) |
| `cost_per_turnover_bps` | `float` | `5.0` | Estimated cost per turnover unit (bps) |
| `window_days` | `int` | `30` | Trade history window in days |

---

#### BehaviorConfig

```python
@dataclass
class BehaviorConfig:
    """Configuration for shadow vs backtest behavior comparison."""

    signal_alignment_threshold: float = 0.95
    timing_tolerance_ms: float = 100.0
    drift_warning_threshold: float = 0.05
    drift_critical_threshold: float = 0.10
    window_size: int = 1000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal_alignment_threshold` | `float` | `0.95` | Required signal alignment rate (95%) |
| `timing_tolerance_ms` | `float` | `100.0` | Acceptable timing drift (ms) |
| `drift_warning_threshold` | `float` | `0.05` | Drift warning threshold (5%) |
| `drift_critical_threshold` | `float` | `0.10` | Drift critical threshold (10%) |
| `window_size` | `int` | `1000` | Rolling window size for comparison |

---

### Runner Service

#### ShadowConfig

```python
@dataclass
class ShadowConfig:
    """Configuration for shadow execution."""

    strategy_id: str                    # Strategy identifier (required)
    symbols: list[str]                  # Allowed symbols (required)
    max_position_size: float = 10000.0  # Max position notional
    slippage_model_bps: float = 2.0     # Slippage in basis points
    check_risk_limits: bool = True      # Enable risk checks
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy_id` | `str` | - | Unique strategy identifier |
| `symbols` | `list[str]` | - | List of allowed trading symbols |
| `max_position_size` | `float` | `10,000.0` | Maximum position notional value |
| `slippage_model_bps` | `float` | `2.0` | Slippage to apply (basis points) |
| `check_risk_limits` | `bool` | `True` | Enable pre-trade risk checks |

---

#### PaperConfig

```python
@dataclass
class PaperConfig:
    """Configuration for paper trading executor."""

    api_key: str                        # Alpaca API key (required)
    api_secret: str                     # Alpaca API secret (required)
    paper: bool = True                  # Must be True (enforced)
    max_position_value: float = 10000.0 # Max position notional
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | - | Alpaca API key |
| `api_secret` | `str` | - | Alpaca API secret |
| `paper` | `bool` | `True` | Paper trading mode (always True) |
| `max_position_value` | `float` | `10,000.0` | Maximum position value in USD |

**CRITICAL:** The `paper` parameter is always enforced as `True` to prevent accidental live trading.

---

## Docker Compose Configuration

### Services Overview

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: qt_registry
      POSTGRES_USER: qt
      POSTGRES_PASSWORD: qt
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:RELEASE.2024-10-13T13-34-11Z
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"      # S3 API
      - "9001:9001"      # Console UI
    volumes:
      - minio_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  minio_data:
```

### Service Details

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `postgres` | `postgres:16-alpine` | `5432` | Strategy registry database |
| `minio` | `minio/minio:RELEASE.2024-10-13` | `9000`, `9001` | S3-compatible data lake storage |
| `redis` | `redis:7-alpine` | `6379` | Caching and pub/sub messaging |

### Port Mappings

| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| `5432` | PostgreSQL | TCP | Database connections |
| `6379` | Redis | TCP | Cache and messaging |
| `8080` | Registry API | HTTP | Strategy registry REST API |
| `9000` | MinIO | HTTP | S3-compatible API |
| `9001` | MinIO | HTTP | Web console UI |

### Volume Mounts

| Volume | Container Path | Purpose |
|--------|----------------|---------|
| `postgres_data` | `/var/lib/postgresql/data` | Persistent database storage |
| `minio_data` | `/data` | Persistent object storage |

---

## Data Lake Structure

### Directory Layout

```
lake/
├── raw/                              # Raw market data
│   ├── trades/                       # Tick-level trades
│   │   └── dt=YYYY-MM-DD/
│   │       └── symbol=XXX/
│   │           └── data.parquet
│   ├── quotes/                       # Quote data (NBBO)
│   │   └── dt=YYYY-MM-DD/
│   │       └── symbol=XXX/
│   │           └── data.parquet
│   └── bars_provider/                # Provider bars
│       └── dt=YYYY-MM-DD/
│           └── symbol=XXX/
│               └── data.parquet
│
├── features/                         # Computed features
│   ├── bars_1m/                      # 1-minute OHLCV bars
│   ├── bars_5m/                      # 5-minute OHLCV bars
│   ├── bars_15m/                     # 15-minute OHLCV bars
│   ├── micro_5s/                     # 5-second microstructure bars
│   ├── micro_15s/                    # 15-second microstructure bars
│   ├── micro_30s/                    # 30-second microstructure bars
│   └── decision_frames/              # Combined decision frames
│
└── labels/                           # Forward returns and labels
    ├── returns_60s/                  # 60-second forward returns
    ├── returns_300s/                 # 300-second forward returns
    └── returns_900s/                 # 900-second forward returns
```

### Partitioning Scheme

**Partition Keys:**
- `dt`: Date partition (`YYYY-MM-DD` format)
- `symbol`: Stock ticker symbol

**File Naming:**
- All data files are named `data.parquet` within their partition directory

**Example Path:**
```
lake/raw/trades/dt=2025-01-15/symbol=AAPL/data.parquet
```

### Parquet Schemas

#### Trade Schema
```python
TRADE_SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("us", tz="UTC")),
    ("ts_recv", pa.timestamp("us", tz="UTC")),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("exchange", pa.string()),
    ("conditions", pa.string()),
])
```

#### Quote Schema
```python
QUOTE_SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("us", tz="UTC")),
    ("ts_recv", pa.timestamp("us", tz="UTC")),
    ("bid_price", pa.float64()),
    ("bid_size", pa.float64()),
    ("ask_price", pa.float64()),
    ("ask_size", pa.float64()),
    ("bid_exchange", pa.string()),
    ("ask_exchange", pa.string()),
    ("conditions", pa.string()),
])
```

#### Bar Schema
```python
BAR_SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("us", tz="UTC")),
    ("ts_recv", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("trade_count", pa.int64()),
    ("vwap", pa.float64()),
])
```

---

## Environment-Specific Recommendations

### Development Environment

```bash
# .env.development
ALPACA_API_KEY_ID=your_paper_key
ALPACA_API_SECRET_KEY=your_paper_secret
ALPACA_ENV=paper

# Database
PGHOST=localhost
PGPORT=5432
PGDATABASE=qt_registry_dev
PGUSER=qt
PGPASSWORD=qt_dev

# MinIO
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET=lake-dev

# GPU (disable for faster iteration)
GPU_ENABLED=false
GPU_FALLBACK_CPU=true
```

**Recommended Configuration Values:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `initial_capital` | `100,000` | Sufficient for testing |
| `max_position_notional` | `10,000` | Limit exposure per position |
| `stop_loss_pct` | `0.02` | 2% stop-loss |
| `feed` | `iex` | Free data feed |
| `GPU_ENABLED` | `false` | Faster development cycles |

---

### Test Environment

```bash
# .env.test
ALPACA_API_KEY_ID=test_key
ALPACA_API_SECRET_KEY=test_secret
ALPACA_ENV=paper

# Database (use separate test database)
PGHOST=localhost
PGPORT=5432
PGDATABASE=qt_registry_test
PGUSER=qt_test
PGPASSWORD=qt_test

# MinIO
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET=lake-test

# GPU
GPU_ENABLED=false
GPU_FALLBACK_CPU=true
```

**Recommended Configuration Values:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `initial_capital` | `50,000` | Smaller for faster tests |
| `min_trades` | `10` | Lower threshold for test data |
| `window_size` | `50` | Smaller windows for tests |
| `max_reconnect_attempts` | `3` | Faster failure in tests |

---

### Production Environment

```bash
# .env.production
ALPACA_API_KEY_ID=your_production_key
ALPACA_API_SECRET_KEY=your_production_secret
ALPACA_ENV=paper  # Start with paper, graduate to live

# Database (production cluster)
PGHOST=prod-db.example.com
PGPORT=5432
PGDATABASE=qt_registry
PGUSER=qt_prod
PGPASSWORD=<secure-password>

# MinIO/S3 (production storage)
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY=<aws-access-key>
S3_SECRET_KEY=<aws-secret-key>
S3_BUCKET=qt-data-lake-prod

# GPU
GPU_ENABLED=true
GPU_DEVICE=0
GPU_FALLBACK_CPU=true
```

**Recommended Configuration Values:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `initial_capital` | Actual capital | Real money |
| `max_position_notional` | Based on risk tolerance | Production limits |
| `max_gross_exposure` | 2x capital | Conservative leverage |
| `max_drawdown_pct` | `0.10` | 10% max drawdown |
| `max_daily_loss` | `0.02 * capital` | 2% daily loss limit |
| `feed` | `sip` | Real-time data for production |
| `expected_fill_rate` | `0.98` | Higher expectations |
| `signal_alignment_threshold` | `0.99` | Stricter alignment |

---

### Configuration Checklist

Before deploying to any environment:

- [ ] Verify `ALPACA_ENV=paper` for all non-production environments
- [ ] Ensure database credentials are environment-specific
- [ ] Confirm S3/MinIO bucket names are unique per environment
- [ ] Set appropriate risk limits for the environment
- [ ] Enable GPU only when needed (save costs in dev/test)
- [ ] Review monitoring thresholds match expected behavior
- [ ] Test rollback procedures work correctly
- [ ] Verify all secrets are stored securely (not in code)

---

## Quick Reference

### Commonly Modified Settings

| Setting | Location | Impact |
|---------|----------|--------|
| `ALPACA_ENV` | Environment | Paper vs live trading |
| `feed` | Client/Streamer | Data quality and cost |
| `initial_capital` | BacktestConfig | Simulation scale |
| `max_position_notional` | RiskLimits | Position sizing |
| `stop_loss_pct` | BacktestConfig | Loss management |
| `no_trade_threshold` | DirectionLabeler | Signal filtering |
| `GPU_ENABLED` | Environment | Training performance |

### Default Values Summary

| Category | Parameter | Default |
|----------|-----------|---------|
| Data Feed | `feed` | `iex` (free) |
| Bars | `granularity` | `1m` |
| Micro Bars | `granularity` | `30s` |
| Labeler | `horizons` | `[60, 300, 900]` |
| Labeler | `no_trade_threshold` | `0.0005` (5 bps) |
| Backtest | `default_order_notional` | `$10,000` |
| Backtest | `stop_loss_pct` | `0.02` (2%) |
| Slippage | All coefficients | `1.0` |
| Ranking | `min_trades` | `30` |
| Fill Rate | `expected_fill_rate` | `0.95` (95%) |
| Shadow | `slippage_model_bps` | `2.0` |
| Paper | `max_position_value` | `$10,000` |
