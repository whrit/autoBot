# Service Reference - AutoBot Phase 1

This document provides comprehensive documentation for all 8 Phase 1 services in the AutoBot autonomous equities trading engine.

## Table of Contents

1. [ingestor_py - Market Data Ingestion](#1-ingestor_py---market-data-ingestion)
2. [feature_builder_py - Feature Construction](#2-feature_builder_py---feature-construction)
3. [labeler_py - Target Labeling](#3-labeler_py---target-labeling)
4. [backtester_py - Backtest Simulation](#4-backtester_py---backtest-simulation)
5. [optimizer_py - Strategy Optimization](#5-optimizer_py---strategy-optimization)
6. [registry_api_py - Strategy Registry API](#6-registry_api_py---strategy-registry-api)
7. [monitor_py - Performance Monitoring](#7-monitor_py---performance-monitoring)
8. [runner_py - Execution Runner](#8-runner_py---execution-runner)

---

## 1. ingestor_py - Market Data Ingestion

### Purpose and Responsibilities

The ingestor service fetches historical and real-time market data from the Alpaca API. It handles:

- Historical data backfilling for trades, quotes, and bars
- Real-time streaming data via WebSocket
- Data persistence to Parquet files
- Universe management for tracking tradeable symbols

### Key Classes and Methods

#### `AlpacaDataClient` (client.py)

```python
class AlpacaDataClient:
    """Client for fetching historical market data from Alpaca."""

    def __init__(self, api_key: str, api_secret: str, feed: str = "iex") -> None
    def get_trades(self, symbol: str, start: datetime, end: datetime) -> Iterator[dict]
    def get_quotes(self, symbol: str, start: datetime, end: datetime) -> Iterator[dict]
    def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: TimeFrame) -> Iterator[dict]
```

#### `ParquetWriter` (writer.py)

```python
class ParquetWriter:
    """Write market data to Parquet files with partitioning."""

    def __init__(self, base_path: Path, partition_by: str = "date")
    def write_trades(self, symbol: str, trades: list[dict]) -> Path
    def write_quotes(self, symbol: str, quotes: list[dict]) -> Path
    def write_bars(self, symbol: str, bars: list[dict], granularity: str) -> Path
```

#### `BackfillManager` (backfill.py)

```python
class BackfillManager:
    """Manage historical data backfilling."""

    def __init__(self, client: AlpacaDataClient, writer: ParquetWriter)
    def backfill_symbol(self, symbol: str, start: datetime, end: datetime, data_types: list[str])
    def backfill_universe(self, symbols: list[str], start: datetime, end: datetime)
```

#### `StreamingClient` (streaming.py)

```python
class StreamingClient:
    """Real-time market data streaming via WebSocket."""

    def __init__(self, api_key: str, api_secret: str, feed: str = "iex")
    async def subscribe_trades(self, symbols: list[str], callback: Callable)
    async def subscribe_quotes(self, symbols: list[str], callback: Callable)
    async def start(self) -> None
    async def stop(self) -> None
```

#### `UniverseManager` (universe.py)

```python
class UniverseManager:
    """Manage the universe of tradeable symbols."""

    def __init__(self, db_path: Path)
    def add_symbol(self, symbol: str, metadata: dict) -> None
    def remove_symbol(self, symbol: str) -> None
    def get_active_symbols(self) -> list[str]
    def update_metadata(self, symbol: str, metadata: dict) -> None
```

### Input/Output Data Formats

**Trade Record:**
```python
{
    "symbol": str,           # e.g., "AAPL"
    "ts_event": datetime,    # Trade timestamp
    "ts_recv": datetime,     # Receive timestamp
    "price": float,          # Trade price
    "size": float,           # Trade size
    "exchange": str,         # Exchange code
    "conditions": str,       # Trade conditions
}
```

**Quote Record:**
```python
{
    "symbol": str,
    "ts_event": datetime,
    "ts_recv": datetime,
    "bid_price": float,
    "bid_size": float,
    "ask_price": float,
    "ask_size": float,
    "bid_exchange": str,
    "ask_exchange": str,
    "conditions": str,
}
```

**Bar Record:**
```python
{
    "symbol": str,
    "ts_event": datetime,
    "ts_recv": datetime,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "trade_count": int,
    "vwap": float,
}
```

### Configuration Options

```python
# Environment variables
ALPACA_API_KEY = "your-api-key"
ALPACA_API_SECRET = "your-api-secret"

# Client configuration
feed = "iex"  # "iex" (free) or "sip" (paid)
base_path = Path("./data/market")
partition_by = "date"  # Parquet partitioning strategy
```

### Running Tests

```bash
cd services/ingestor_py
pytest tests/ -v

# Run specific test file
pytest tests/test_client.py -v

# Run with coverage
pytest tests/ --cov=ingestor_py --cov-report=term-missing
```

### Example Usage

```python
from datetime import datetime, UTC, timedelta
from pathlib import Path
from ingestor_py.client import AlpacaDataClient
from ingestor_py.writer import ParquetWriter
from ingestor_py.backfill import BackfillManager

# Initialize client
client = AlpacaDataClient(
    api_key="your-key",
    api_secret="your-secret",
    feed="iex"
)

# Initialize writer
writer = ParquetWriter(base_path=Path("./data"))

# Create backfill manager
backfill = BackfillManager(client=client, writer=writer)

# Backfill last 30 days of data
end = datetime.now(UTC)
start = end - timedelta(days=30)
backfill.backfill_symbol("SPY", start, end, data_types=["trades", "quotes"])
```

---

## 2. feature_builder_py - Feature Construction

### Purpose and Responsibilities

The feature builder constructs multi-timeframe features from raw market data:

- Standard OHLCV bars at 1m, 5m, 15m intervals
- Micro-structure features (trade imbalance, order flow)
- Multi-timeframe feature joins
- Decision frame construction for model training
- Incremental feature updates

### Key Classes and Methods

#### `StandardBarBuilder` (bars.py)

```python
class StandardBarBuilder:
    """Build standard OHLCV bars from trade data."""

    VALID_GRANULARITIES = {"1m", "5m", "15m"}

    def __init__(self, granularity: str = "1m", atr_period: int = 14)
    def build(self, trades: pl.DataFrame, symbol: str) -> pl.DataFrame
```

**Output columns:** `symbol`, `bar_start`, `bar_end`, `open`, `high`, `low`, `close`, `volume`, `returns`, `atr`, `realized_vol`

#### `MicroBarBuilder` (micro_bars.py)

```python
class MicroBarBuilder:
    """Build micro-structure features from tick data."""

    def __init__(self, granularity: str = "1m")
    def build(self, trades: pl.DataFrame, quotes: pl.DataFrame, symbol: str) -> pl.DataFrame
```

**Output columns:** `symbol`, `bar_start`, `trade_imbalance`, `volume_imbalance`, `spread_mean`, `spread_vol`, `microprice_vol`

#### `MultiTimeframeJoiner` (joins.py)

```python
class MultiTimeframeJoiner:
    """Join features across multiple timeframes."""

    def __init__(self, timeframes: list[str] = ["1m", "5m", "15m"])
    def join(self, features_dict: dict[str, pl.DataFrame]) -> pl.DataFrame
```

#### `DecisionFrameBuilder` (decision_frame.py)

```python
class DecisionFrameBuilder:
    """Build decision frames for ML training."""

    def __init__(self, feature_cols: list[str], label_col: str = "direction")
    def build(self, features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame
```

#### `IncrementalFeatureBuilder` (incremental.py)

```python
class IncrementalFeatureBuilder:
    """Incrementally update features as new data arrives."""

    def __init__(self, bar_builder: StandardBarBuilder, micro_builder: MicroBarBuilder)
    def update(self, new_trades: pl.DataFrame, new_quotes: pl.DataFrame) -> pl.DataFrame
```

### Input/Output Data Formats

**Input (Trades DataFrame):**
```
ts_event: datetime
ts_recv: datetime
price: float
size: float
exchange: str
conditions: str
```

**Output (Standard Bars):**
```
symbol: str
bar_start: datetime
bar_end: datetime
open: float
high: float
low: float
close: float
volume: float
returns: float
atr: float
realized_vol: float
```

**Output (Micro Features):**
```
symbol: str
bar_start: datetime
trade_imbalance: float     # Buy volume - sell volume / total
volume_imbalance: float    # Normalized order flow
spread_mean: float         # Average bid-ask spread
spread_vol: float          # Spread volatility
microprice_vol: float      # Microprice volatility
```

### Configuration Options

```python
# Bar builder configuration
granularity = "1m"  # "1m", "5m", "15m"
atr_period = 14     # ATR calculation window

# Decision frame configuration
feature_cols = [
    "returns", "atr", "realized_vol",
    "trade_imbalance", "volume_imbalance",
    "spread_mean", "spread_vol"
]
label_col = "direction"
```

### Running Tests

```bash
cd services/feature_builder_py
pytest tests/ -v

# Run with markers
pytest tests/ -v -m "not slow"
```

### Example Usage

```python
import polars as pl
from feature_builder_py.bars import StandardBarBuilder
from feature_builder_py.micro_bars import MicroBarBuilder
from feature_builder_py.joins import MultiTimeframeJoiner

# Load raw data
trades = pl.read_parquet("data/trades/SPY.parquet")
quotes = pl.read_parquet("data/quotes/SPY.parquet")

# Build 1-minute bars
bar_builder_1m = StandardBarBuilder(granularity="1m")
bars_1m = bar_builder_1m.build(trades, "SPY")

# Build 5-minute bars
bar_builder_5m = StandardBarBuilder(granularity="5m")
bars_5m = bar_builder_5m.build(trades, "SPY")

# Build micro features
micro_builder = MicroBarBuilder(granularity="1m")
micro_features = micro_builder.build(trades, quotes, "SPY")

# Join multi-timeframe features
joiner = MultiTimeframeJoiner()
combined = joiner.join({
    "1m": bars_1m,
    "5m": bars_5m,
    "micro": micro_features
})
```

---

## 3. labeler_py - Target Labeling

### Purpose and Responsibilities

The labeler service creates target labels for ML training:

- Forward returns calculation at configurable horizons
- Direction labels (long/short/flat)
- Net returns accounting for transaction costs
- Quote-based pricing (midprice/microprice)

### Key Classes and Methods

#### `ForwardReturnsCalculator` (returns.py)

```python
class ForwardReturnsCalculator:
    """Calculate forward returns at configurable horizons."""

    def __init__(self, horizons: Sequence[int] = [60, 300, 900])
    def calculate(
        self,
        quotes_df: pl.DataFrame,
        decision_times: Sequence[datetime],
        symbol: str,
        use_microprice: bool = False
    ) -> pl.DataFrame
```

#### `DirectionLabeler` (labels.py)

```python
class DirectionLabeler:
    """Generate direction labels from returns."""

    def __init__(
        self,
        long_threshold: float = 0.001,
        short_threshold: float = -0.001
    )
    def label(self, returns_df: pl.DataFrame) -> pl.DataFrame
```

#### `NetReturnsCalculator` (net_returns.py)

```python
class NetReturnsCalculator:
    """Calculate net returns after transaction costs."""

    def __init__(self, cost_bps: float = 5.0)
    def calculate(self, returns_df: pl.DataFrame) -> pl.DataFrame
```

### Input/Output Data Formats

**Input (Quotes DataFrame):**
```
symbol: str
ts_event: datetime
bid_price: float
ask_price: float
bid_size: float
ask_size: float
```

**Output (Forward Returns):**
```
symbol: str
decision_ts: datetime
horizon: int          # Horizon in seconds (60, 300, 900)
fwd_return_mid: float # Forward return using midprice
```

**Output (Direction Labels):**
```
symbol: str
decision_ts: datetime
horizon: int
fwd_return_mid: float
direction: str        # "long", "short", or "flat"
```

### Configuration Options

```python
# Forward returns horizons (in seconds)
horizons = [60, 300, 900]  # 1min, 5min, 15min

# Direction labeling thresholds
long_threshold = 0.001   # +0.1% for long
short_threshold = -0.001 # -0.1% for short

# Transaction cost for net returns
cost_bps = 5.0  # 5 basis points round-trip
```

### Running Tests

```bash
cd services/labeler_py
pytest tests/ -v
```

### Example Usage

```python
from datetime import datetime, timedelta
import polars as pl
from labeler_py.returns import ForwardReturnsCalculator
from labeler_py.labels import DirectionLabeler
from labeler_py.net_returns import NetReturnsCalculator

# Load quotes
quotes = pl.read_parquet("data/quotes/SPY.parquet")

# Generate decision times (every minute)
start = datetime(2024, 1, 15, 9, 30)
decision_times = [start + timedelta(minutes=i) for i in range(390)]

# Calculate forward returns at multiple horizons
calculator = ForwardReturnsCalculator(horizons=[60, 300, 900])
returns_df = calculator.calculate(quotes, decision_times, "SPY")

# Generate direction labels
labeler = DirectionLabeler(long_threshold=0.001, short_threshold=-0.001)
labeled = labeler.label(returns_df)

# Calculate net returns after costs
net_calc = NetReturnsCalculator(cost_bps=5.0)
net_returns = net_calc.calculate(labeled)
```

---

## 4. backtester_py - Backtest Simulation

### Purpose and Responsibilities

The backtester provides realistic trading simulation:

- Quote-based execution (buy at ask, sell at bid)
- Slippage modeling (Almgren-Chriss market impact)
- Position and risk management
- Performance metrics calculation
- Walk-forward and purged cross-validation

### Key Classes and Methods

#### `BacktestEngine` (engine.py)

```python
class BacktestEngine:
    """Core backtesting engine with quote-based fills."""

    def __init__(self, config: BacktestConfig)
    def run(self, market_data: pl.DataFrame, signals: list[Signal]) -> BacktestResult
```

#### `BacktestConfig` (engine.py)

```python
@dataclass
class BacktestConfig:
    initial_capital: float
    cost_model: TransactionCostModel
    risk_checker: RiskChecker
    default_order_notional: float = 10_000.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float | None = None
    book_notional_default: float = 500_000.0
    short_term_vol_default: float = 0.001
```

#### `TransactionCostModel` (slippage.py)

```python
class TransactionCostModel:
    """Almgren-Chriss slippage model."""

    def __init__(
        self,
        fixed_cost_bps: float = 1.0,
        impact_coefficient: float = 0.1,
        volatility_multiplier: float = 1.0
    )
    def calculate_fill_price(
        self,
        side: str,
        ask_price: float,
        bid_price: float,
        order_notional: float,
        book_notional: float,
        short_term_vol: float
    ) -> float
```

#### `MetricsCalculator` (metrics.py)

```python
class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    def calculate(self, result: BacktestResult) -> dict[str, float]
```

**Returns:** `sharpe`, `sortino`, `max_drawdown`, `calmar`, `profit_factor`, `win_rate`, `num_trades`

#### `WalkForwardValidator` (walk_forward.py)

```python
class WalkForwardValidator:
    """Walk-forward validation with expanding/rolling windows."""

    def __init__(
        self,
        train_days: int,
        test_days: int,
        step_days: int,
        expanding: bool = False
    )
    def split(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]
```

#### `PurgedKFold` (purged_cv.py)

```python
class PurgedKFold:
    """Purged K-Fold cross-validation for time series."""

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01
    )
    def split(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]
```

#### `RegimeDetector` (regimes.py)

```python
class RegimeDetector:
    """Detect market regimes for regime-aware backtesting."""

    def __init__(self, lookback: int = 20, vol_threshold: float = 0.02)
    def detect(self, prices: pl.DataFrame) -> pl.DataFrame
```

### Input/Output Data Formats

**Input (Market Data):**
```
timestamp: datetime
symbol: str
bid_price: float
ask_price: float
bid_size: float (optional)
ask_size: float (optional)
vol: float (optional)
```

**Input (Signal):**
```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType  # LONG, SHORT, FLAT
    strength: float = 1.0
    target_notional: float | None = None
```

**Output (BacktestResult):**
```python
@dataclass
class BacktestResult:
    fills: list[Fill]
    positions: dict[str, Position]
    equity_curve: pl.DataFrame
    daily_returns: pl.DataFrame
    total_pnl: float
    total_trades: int
    risk_violations: list[str]
```

### Configuration Options

```python
# Backtest configuration
config = BacktestConfig(
    initial_capital=100_000.0,
    cost_model=TransactionCostModel(
        fixed_cost_bps=1.0,
        impact_coefficient=0.1,
        volatility_multiplier=1.0
    ),
    risk_checker=RiskChecker(
        max_position_pct=0.10,
        max_drawdown_pct=0.05,
        max_daily_loss_pct=0.02
    ),
    default_order_notional=10_000.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)
```

### Running Tests

```bash
cd services/backtester_py
pytest tests/ -v

# Run specific test categories
pytest tests/test_engine.py -v
pytest tests/test_metrics.py -v
```

### Example Usage

```python
from datetime import datetime
import polars as pl
from backtester_py.engine import BacktestEngine, BacktestConfig, Signal, SignalType
from backtester_py.slippage import TransactionCostModel
from backtester_py.risk import RiskChecker
from backtester_py.metrics import MetricsCalculator

# Load market data
market_data = pl.read_parquet("data/quotes/SPY.parquet")

# Create signals
signals = [
    Signal(
        timestamp=datetime(2024, 1, 15, 10, 0),
        symbol="SPY",
        signal_type=SignalType.LONG,
        strength=0.8,
        target_notional=10_000.0
    ),
    Signal(
        timestamp=datetime(2024, 1, 15, 14, 0),
        symbol="SPY",
        signal_type=SignalType.FLAT,
        strength=1.0
    )
]

# Configure backtest
config = BacktestConfig(
    initial_capital=100_000.0,
    cost_model=TransactionCostModel(fixed_cost_bps=2.0),
    risk_checker=RiskChecker(max_position_pct=0.20)
)

# Run backtest
engine = BacktestEngine(config)
result = engine.run(market_data, signals)

# Calculate metrics
metrics = MetricsCalculator().calculate(result)
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Total Trades: {result.total_trades}")
```

---

## 5. optimizer_py - Strategy Optimization

### Purpose and Responsibilities

The optimizer service handles strategy parameter optimization:

- Strategy family framework with signal generation
- Parameter sweep (grid search, random search)
- Candidate ranking with complexity penalties
- Multi-objective optimization support

### Key Classes and Methods

#### `StrategyFamily` (strategies.py)

```python
class StrategyFamily(ABC):
    """Base class for strategy families."""

    @abstractmethod
    def generate_signal(self, features: pl.DataFrame, params: dict) -> StrategySignal

    @abstractmethod
    def get_parameter_space(self) -> dict[str, ParameterRange]

    @property
    @abstractmethod
    def family_name(self) -> str
```

#### `StrategySignal` (strategies.py)

```python
@dataclass
class StrategySignal:
    timestamp: datetime
    symbol: str
    direction: SignalDirection  # LONG, SHORT, FLAT
    strength: float
    metadata: dict[str, Any]
```

#### `ParameterSweep` (sweep.py)

```python
class ParameterSweep:
    """Parameter optimization via grid and random search."""

    def __init__(
        self,
        strategy: StrategyFamily,
        backtester: BacktestEngine,
        n_jobs: int = -1
    )
    def grid_search(self, features: pl.DataFrame, market_data: pl.DataFrame) -> list[CandidateScore]
    def random_search(
        self,
        features: pl.DataFrame,
        market_data: pl.DataFrame,
        n_iter: int = 100
    ) -> list[CandidateScore]
```

#### `CandidateRanker` (ranking.py)

```python
class CandidateRanker:
    """Rank strategy candidates by performance metrics."""

    def __init__(self, config: RankingConfig | None = None)
    def rank_across_families(self, candidates: list[CandidateScore]) -> list[CandidateScore]
    def rank_within_family(self, candidates: list[CandidateScore]) -> list[CandidateScore]
    def filter_candidates(self, candidates: list[CandidateScore]) -> list[CandidateScore]
    def get_top_candidates(self, candidates: list[CandidateScore], n: int = 10) -> list[CandidateScore]
    def get_top_per_family(self, candidates: list[CandidateScore], n: int = 1) -> list[CandidateScore]
```

#### `RankingConfig` (ranking.py)

```python
@dataclass
class RankingConfig:
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

#### `CandidateScore` (ranking.py)

```python
@dataclass
class CandidateScore:
    strategy_id: str
    family: str
    params: dict[str, Any]
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    num_trades: int
    composite_score: float = 0.0
```

#### `ComplexityPenaltyConfig` (complexity.py)

```python
@dataclass
class ComplexityPenaltyConfig:
    max_parameters: int = 10
    penalty_per_excess_param: float = 0.1
    nested_penalty: float = 0.05
```

### Input/Output Data Formats

**Input (Parameter Space):**
```python
{
    "lookback": ParameterRange(min=5, max=50, step=5),
    "threshold": ParameterRange(min=0.001, max=0.01, step=0.001),
    "stop_loss": ParameterRange(min=0.01, max=0.05, step=0.01)
}
```

**Output (CandidateScore):**
```python
CandidateScore(
    strategy_id="trend_v1_123",
    family="trend",
    params={"lookback": 20, "threshold": 0.005},
    sharpe=1.5,
    sortino=2.1,
    max_drawdown=0.08,
    profit_factor=1.8,
    win_rate=0.55,
    num_trades=150,
    composite_score=1.42
)
```

### Configuration Options

```python
# Ranking configuration
ranking_config = RankingConfig(
    metric=RankingMetric.COMPOSITE,
    min_trades=30,
    max_drawdown_threshold=0.20,
    sharpe_weight=0.4,
    sortino_weight=0.3,
    profit_factor_weight=0.2,
    drawdown_penalty_weight=0.1
)

# Complexity penalty configuration
complexity_config = ComplexityPenaltyConfig(
    max_parameters=10,
    penalty_per_excess_param=0.1
)
```

### Running Tests

```bash
cd services/optimizer_py
pytest tests/ -v
```

### Example Usage

```python
from optimizer_py.strategies import TrendStrategy
from optimizer_py.sweep import ParameterSweep
from optimizer_py.ranking import CandidateRanker, RankingConfig, RankingMetric

# Create strategy
strategy = TrendStrategy()

# Create parameter sweep
sweep = ParameterSweep(
    strategy=strategy,
    backtester=engine,
    n_jobs=4
)

# Run random search
candidates = sweep.random_search(
    features=features,
    market_data=market_data,
    n_iter=100
)

# Rank candidates
config = RankingConfig(
    metric=RankingMetric.COMPOSITE,
    min_trades=50,
    max_drawdown_threshold=0.15
)
ranker = CandidateRanker(config)

# Get top candidates
filtered = ranker.filter_candidates(candidates)
ranked = ranker.rank_across_families(filtered)
top_5 = ranker.get_top_candidates(ranked, n=5)

for candidate in top_5:
    print(f"{candidate.strategy_id}: Sharpe={candidate.sharpe:.2f}, "
          f"MaxDD={candidate.max_drawdown:.2%}")
```

---

## 6. registry_api_py - Strategy Registry API

### Purpose and Responsibilities

The registry API provides a FastAPI-based REST interface for:

- Strategy CRUD operations
- Artifact storage (ONNX models, configs)
- Backtest run management
- Promotion state machine
- Gate evaluation for quality control
- Comprehensive audit logging

### Key Classes and Methods

#### FastAPI Application (app.py)

**Strategy Endpoints:**
- `POST /strategies` - Create strategy
- `GET /strategies` - List strategies with pagination
- `GET /strategies/{id}` - Get strategy by ID
- `PUT /strategies/{id}` - Update strategy
- `DELETE /strategies/{id}` - Delete strategy

**Artifact Endpoints:**
- `POST /strategies/{id}/artifacts` - Upload artifact
- `GET /strategies/{id}/artifacts` - List artifacts
- `DELETE /artifacts/{id}` - Delete artifact

**Backtest Endpoints:**
- `POST /backtest-runs` - Create backtest run
- `GET /backtest-runs` - List backtest runs
- `GET /backtest-runs/{id}` - Get backtest run

**Promotion Endpoints:**
- `POST /strategies/{id}/promote` - Promote strategy
- `POST /strategies/{id}/demote` - Demote strategy
- `POST /strategies/{id}/retire` - Retire strategy

**Gate Endpoints:**
- `POST /gates` - Create gate
- `GET /gates` - List gates
- `POST /gates/{id}/evaluate/{strategy_id}` - Evaluate gate

#### SQLAlchemy Models (models.py)

```python
class Strategy(Base):
    id: int
    name: str
    family: str
    version: str
    parameters: dict  # JSON
    state: PromotionState  # candidate, shadow, paper, retired
    created_at: datetime
    updated_at: datetime

class Artifact(Base):
    id: int
    strategy_id: int
    artifact_type: str  # "onnx", "json", "config"
    path: str
    checksum: str  # SHA-256
    created_at: datetime

class BacktestRun(Base):
    id: int
    strategy_id: int
    dataset_snapshot_id: int | None
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    num_trades: int
    started_at: datetime
    completed_at: datetime
    training_device: str | None  # "cpu" | "cuda"
    gpu_backend: str | None
    cuda_version: str | None
    seed: int | None

class Gate(Base):
    id: int
    name: str
    gate_type: str  # "backtest", "shadow", "paper"
    criteria: dict  # JSON with thresholds
    is_active: bool

class Promotion(Base):
    id: int
    strategy_id: int
    gate_id: int
    from_state: PromotionState
    to_state: PromotionState
    passed: bool
    evaluation_results: dict
    promoted_at: datetime
```

#### PromotionManager (promotions.py)

```python
class PromotionManager:
    """Manage strategy promotion lifecycle."""

    def __init__(self, requirements: PromotionRequirements, audit_log: AuditLog)
    def can_transition(self, from_state: PromotionState, to_state: PromotionState) -> bool
    def check_requirements(self, strategy_id: int, target_state: PromotionState, db: Session) -> dict[str, bool]
    def promote(self, strategy_id: int, target_state: PromotionState, db: Session) -> PromotionResult
    def demote(self, strategy_id: int, reason: str, db: Session) -> PromotionResult
    def retire(self, strategy_id: int, reason: str, db: Session) -> PromotionResult
```

**Valid State Transitions:**
```
CANDIDATE -> SHADOW, RETIRED
SHADOW -> PAPER, CANDIDATE, RETIRED
PAPER -> RETIRED, SHADOW
RETIRED -> (terminal, no transitions)
```

#### GateEvaluator (gates.py)

```python
class GateEvaluator:
    """Evaluate quality gates for promotion."""

    def evaluate(self, gate: Gate, strategy_id: int, db: Session) -> GateResult
```

**Gate Types:**
- `MIN_SHARPE` - Minimum Sharpe ratio (default: 0.5)
- `MAX_DRAWDOWN` - Maximum drawdown (default: 0.15)
- `MIN_TRADES` - Minimum trade count (default: 30)
- `MIN_WIN_RATE` - Minimum win rate (default: 0.45)
- `MIN_PROFIT_FACTOR` - Minimum profit factor (default: 1.1)

#### AuditLog (audit.py)

```python
class AuditLog:
    """Record and query audit events."""

    def __init__(self, db: Session)
    def record(
        self,
        event_type: AuditEventType,
        entity_type: str,
        entity_id: int,
        message: str,
        old_value: dict | None = None,
        new_value: dict | None = None,
        user_id: str = "system",
        metadata: dict | None = None
    ) -> AuditEntry
    def get_history(self, entity_type: str, entity_id: int) -> list[AuditEntry]
    def query(
        self,
        event_type: AuditEventType | None = None,
        entity_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100
    ) -> list[AuditEntry]
```

### Input/Output Data Formats

**Strategy Create Request:**
```json
{
    "name": "trend_v1",
    "family": "trend",
    "version": "1.0.0",
    "parameters": {
        "lookback": 20,
        "threshold": 0.005
    }
}
```

**Strategy Response:**
```json
{
    "id": 1,
    "name": "trend_v1",
    "family": "trend",
    "version": "1.0.0",
    "parameters": {"lookback": 20, "threshold": 0.005},
    "state": "candidate",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": null
}
```

**Promotion Request:**
```json
{
    "target_state": "shadow"
}
```

**Gate Evaluation Response:**
```json
{
    "gate_id": 1,
    "strategy_id": 1,
    "passed": true,
    "criteria_results": {
        "min_sharpe": true,
        "max_drawdown": true,
        "min_trades": true
    },
    "evaluated_at": "2024-01-15T10:00:00Z"
}
```

### Configuration Options

```python
# Environment variables
DATABASE_URL = "sqlite:///./registry.db"

# Promotion requirements
requirements = PromotionRequirements(
    to_shadow={
        "min_sharpe": 0.5,
        "max_drawdown": 0.15,
        "min_trades": 100
    },
    to_paper={
        "shadow_days": 5,
        "shadow_sharpe": 0.4,
        "no_alerts": True
    }
)
```

### Running Tests

```bash
cd services/registry_api_py
pytest tests/ -v

# Run with database fixture
pytest tests/test_app.py -v
```

### Example Usage

```python
import httpx

# Create a strategy
response = httpx.post(
    "http://localhost:8080/strategies",
    json={
        "name": "momentum_v1",
        "family": "momentum",
        "version": "1.0.0",
        "parameters": {"lookback": 14, "threshold": 0.01}
    }
)
strategy = response.json()
strategy_id = strategy["id"]

# Upload backtest results
httpx.post(
    "http://localhost:8080/backtest-runs",
    json={
        "strategy_id": strategy_id,
        "sharpe": 1.5,
        "sortino": 2.0,
        "max_drawdown": 0.08,
        "profit_factor": 1.8,
        "win_rate": 0.55,
        "num_trades": 150
    }
)

# Promote to shadow
httpx.post(
    f"http://localhost:8080/strategies/{strategy_id}/promote",
    json={"target_state": "shadow"}
)

# Check audit history
response = httpx.get(f"http://localhost:8080/strategies/{strategy_id}/audit")
audit_entries = response.json()
```

---

## 7. monitor_py - Performance Monitoring

### Purpose and Responsibilities

The monitor service tracks live performance metrics:

- Real-time performance metrics calculation
- Drawdown monitoring and alerts
- Slippage analysis
- Fill rate tracking
- Portfolio turnover monitoring
- Behavior comparison (live vs backtest)
- Feed health monitoring
- Automatic rollback triggers

### Key Classes and Methods

#### `PerformanceMetrics` (metrics.py)

```python
class PerformanceMetrics:
    """Calculate real-time performance metrics."""

    def __init__(self, risk_free_rate: float = 0.0)
    def calculate(self, returns: pl.DataFrame) -> dict[str, float]
    def calculate_rolling(
        self,
        returns: pl.DataFrame,
        window: int = 20
    ) -> pl.DataFrame
```

**Metrics calculated:** `sharpe`, `sortino`, `calmar`, `max_drawdown`, `volatility`, `skewness`, `kurtosis`

#### `DrawdownMonitor` (drawdown.py)

```python
class DrawdownMonitor:
    """Monitor drawdowns and trigger alerts."""

    def __init__(
        self,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        halt_threshold: float = 0.15
    )
    def update(self, equity: float) -> DrawdownAlert | None
    def get_current_drawdown(self) -> float
    def get_max_drawdown(self) -> float
```

#### `SlippageAnalyzer` (slippage.py)

```python
class SlippageAnalyzer:
    """Analyze execution slippage."""

    def __init__(self, expected_slippage_bps: float = 2.0)
    def analyze(
        self,
        fills: list[Fill],
        quotes: pl.DataFrame
    ) -> SlippageReport
    def get_rolling_slippage(self, window: int = 20) -> pl.DataFrame
```

#### `FillRateMonitor` (fill_rate.py)

```python
class FillRateMonitor:
    """Monitor order fill rates."""

    def __init__(self, target_fill_rate: float = 0.95)
    def update(self, order_submitted: bool, order_filled: bool) -> None
    def get_fill_rate(self) -> float
    def get_rolling_fill_rate(self, window: int = 20) -> float
```

#### `TurnoverMonitor` (turnover.py)

```python
class TurnoverMonitor:
    """Monitor portfolio turnover."""

    def __init__(self, max_daily_turnover: float = 0.5)
    def update(self, trade_notional: float, portfolio_value: float) -> None
    def get_daily_turnover(self) -> float
    def check_limit(self) -> bool
```

#### `BehaviorComparison` (behavior_comparison.py)

```python
class BehaviorComparison:
    """Compare live vs backtest behavior."""

    def __init__(self, tolerance_pct: float = 0.10)
    def compare(
        self,
        live_metrics: dict[str, float],
        backtest_metrics: dict[str, float]
    ) -> ComparisonResult
    def detect_drift(
        self,
        live_returns: pl.DataFrame,
        backtest_returns: pl.DataFrame
    ) -> bool
```

#### `FeedHealthMonitor` (feed_health.py)

```python
class FeedHealthMonitor:
    """Monitor data feed health."""

    def __init__(
        self,
        max_latency_ms: float = 100.0,
        max_gap_seconds: float = 5.0
    )
    def update(self, timestamp: datetime, latency_ms: float) -> None
    def check_health(self) -> FeedHealthStatus
    def get_metrics(self) -> dict[str, float]
```

#### `RollbackTrigger` (rollback.py)

```python
class RollbackTrigger:
    """Automatic rollback on performance degradation."""

    def __init__(
        self,
        max_drawdown: float = 0.10,
        max_consecutive_losses: int = 5,
        max_daily_loss: float = 0.03
    )
    def check(self, state: MonitoringState) -> RollbackDecision
    def execute_rollback(self, strategy_id: int, reason: str) -> bool
```

### Input/Output Data Formats

**Input (Returns DataFrame):**
```
timestamp: datetime
strategy_id: str
return: float
equity: float
```

**Output (DrawdownAlert):**
```python
@dataclass
class DrawdownAlert:
    level: str  # "warning", "critical", "halt"
    drawdown: float
    peak_equity: float
    current_equity: float
    timestamp: datetime
    message: str
```

**Output (SlippageReport):**
```python
@dataclass
class SlippageReport:
    mean_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    excess_slippage_bps: float  # vs expected
    fill_count: int
```

### Configuration Options

```python
# Drawdown thresholds
drawdown_config = {
    "warning_threshold": 0.05,   # 5%
    "critical_threshold": 0.10,  # 10%
    "halt_threshold": 0.15       # 15%
}

# Rollback triggers
rollback_config = {
    "max_drawdown": 0.10,
    "max_consecutive_losses": 5,
    "max_daily_loss": 0.03
}

# Feed health
feed_health_config = {
    "max_latency_ms": 100.0,
    "max_gap_seconds": 5.0
}
```

### Running Tests

```bash
cd services/monitor_py
pytest tests/ -v
```

### Example Usage

```python
from datetime import datetime
import polars as pl
from monitor_py.metrics import PerformanceMetrics
from monitor_py.drawdown import DrawdownMonitor
from monitor_py.slippage import SlippageAnalyzer
from monitor_py.rollback import RollbackTrigger

# Initialize monitors
metrics = PerformanceMetrics(risk_free_rate=0.04)
drawdown_monitor = DrawdownMonitor(
    warning_threshold=0.05,
    critical_threshold=0.10
)
rollback = RollbackTrigger(max_drawdown=0.10)

# Load live returns
returns = pl.read_parquet("data/live_returns.parquet")

# Calculate metrics
perf = metrics.calculate(returns)
print(f"Sharpe: {perf['sharpe']:.2f}")
print(f"Max Drawdown: {perf['max_drawdown']:.2%}")

# Check drawdown
for equity in returns["equity"]:
    alert = drawdown_monitor.update(equity)
    if alert and alert.level == "critical":
        print(f"CRITICAL: Drawdown at {alert.drawdown:.2%}")

# Check rollback
state = MonitoringState(
    current_drawdown=drawdown_monitor.get_current_drawdown(),
    consecutive_losses=3,
    daily_loss=0.01
)
decision = rollback.check(state)
if decision.should_rollback:
    print(f"Rollback triggered: {decision.reason}")
```

---

## 8. runner_py - Execution Runner

### Purpose and Responsibilities

The runner service executes trading signals:

- Shadow execution (simulation without real orders)
- Paper trading via Alpaca API
- Autonomous orchestration loop
- Automatic promotion management
- Signal logging to Parquet

### Key Classes and Methods

#### `AutonomousOrchestrator` (orchestrator.py)

```python
class AutonomousOrchestrator:
    """Orchestrate the autonomous trading loop."""

    def __init__(self, config: OrchestratorConfig)
    def register_task(
        self,
        task_type: TaskType,
        handler: Callable,
        frequency: ScheduleFrequency,
        time_of_day: time | None = None
    ) -> None
    def schedule_default_tasks(self) -> None
    async def start(self) -> None
    async def stop(self) -> None
    def is_market_hours(self) -> bool
    def get_task_status(self) -> dict[str, Any]
```

**Task Types:**
- `FEATURE_REFRESH` - Refresh feature data
- `SIGNAL_GENERATION` - Generate trading signals
- `SHADOW_EXECUTION` - Execute in shadow mode
- `PAPER_EXECUTION` - Execute in paper mode
- `PERFORMANCE_EVAL` - Evaluate performance
- `PROMOTION_CHECK` - Check promotion eligibility

#### `Orchestrator` (orchestrator.py)

```python
class Orchestrator:
    """Coordinate the trading loop phases."""

    def __init__(self, config: OrchestrationConfig)
    async def start(self) -> None
    async def stop(self) -> None
    def get_state(self) -> OrchestrationState
    def is_market_open(self) -> bool
```

**Phases:** `CLOSED`, `PRE_MARKET`, `TRADING`, `POST_MARKET`

#### `ShadowExecutor` (shadow.py)

```python
class ShadowExecutor:
    """Execute signals in shadow mode (no real orders)."""

    def __init__(self, config: ShadowConfig)
    def execute(self, signal: Signal, market_data: MarketData) -> ShadowExecutionResult
    def simulate_fill(self, signal: Signal, market_data: MarketData) -> ShadowFill
    def check_risk_limits(self, signal: Signal) -> tuple[bool, str | None]
    def get_position(self, symbol: str) -> float
    def update_position(self, fill: ShadowFill) -> None
    def get_pnl(self) -> dict[str, float]
    def get_total_pnl(self) -> float
    def get_trade_history(self) -> list[ShadowFill]
```

#### `PaperExecutor` (paper.py)

```python
class PaperExecutor:
    """Execute signals via Alpaca paper trading API."""

    def __init__(self, config: PaperConfig)
    async def execute(self, signal: Signal) -> ExecutionResult
    def submit_order(self, signal: Signal) -> str
    def get_order_status(self, order_id: str) -> dict[str, Any]
    def cancel_order(self, order_id: str) -> bool
    def get_positions(self) -> list[dict[str, Any]]
    def get_account(self) -> dict[str, Any]
```

**Critical:** Always uses `paper=True` to prevent live trading.

#### `AutoPromoter` (promotion.py)

```python
class AutoPromoter:
    """Automatic shadow to paper promotion."""

    def __init__(
        self,
        criteria: PromotionCriteria,
        registry_api_url: str = "http://localhost:8080"
    )
    async def evaluate_promotion(self, strategy_id: str) -> PromotionEvaluation
    async def promote_strategy(self, strategy_id: str) -> bool
    async def check_all_candidates(self) -> list[PromotionEvaluation]
```

#### `PromotionManager` (promotion.py)

```python
class PromotionManager:
    """High-level promotion lifecycle manager."""

    def __init__(self, criteria: PromotionCriteria, registry_api_url: str)
    def validate_transition(self, from_state: PromotionState, to_state: PromotionState) -> bool
    async def evaluate_promotion(self, strategy_id: str, target_state: PromotionState | None) -> PromotionDecision
    async def promote(self, strategy_id: str, target_state: PromotionState | None, auto: bool) -> PromotionDecision
    async def batch_evaluate(self, from_state: PromotionState) -> list[PromotionDecision]
    async def batch_promote(self, from_state: PromotionState, max_promotions: int | None) -> list[PromotionDecision]
```

#### `SignalLogger` (signal_log.py)

```python
class SignalLogger:
    """Log signals to Parquet files."""

    def __init__(self, config: SignalLogConfig)
    def log_signal(self, signal: Signal, execution_result: ShadowExecutionResult | None) -> None
    def log_batch(self, signals: list[tuple[Signal, ShadowExecutionResult | None]]) -> None
    def flush(self) -> None
    def read_logs(self, start_time: datetime | None, end_time: datetime | None) -> pl.DataFrame
    def get_signal_stats(self) -> dict[str, Any]
```

### Input/Output Data Formats

**Signal:**
```python
@dataclass(frozen=True, slots=True)
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType  # LONG, SHORT, FLAT
    strength: float = 1.0
    target_notional: float = 10000.0
    strategy_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

**MarketData:**
```python
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float

    @property
    def spread_bps(self) -> float
```

**ShadowExecutionResult:**
```python
@dataclass
class ShadowExecutionResult:
    mode: ExecutionMode  # SHADOW
    signal: Signal
    fill: ShadowFill | None
    success: bool
    message: str
```

**ExecutionResult (Paper):**
```python
@dataclass
class ExecutionResult:
    success: bool
    order_id: str | None
    signal: Signal | None
    error: str | None
    submitted_at: datetime | None
    filled_at: datetime | None
    filled_qty: float | None
    filled_price: float | None
```

### Configuration Options

```python
# Shadow configuration
shadow_config = ShadowConfig(
    strategy_id="trend_v1",
    symbols=["SPY", "QQQ", "IWM"],
    max_position_size=10000.0,
    slippage_model_bps=2.0,
    check_risk_limits=True
)

# Paper configuration
paper_config = PaperConfig(
    api_key="your-key",
    api_secret="your-secret",
    paper=True,  # ALWAYS True
    max_position_value=10000.0
)

# Orchestrator configuration
orchestrator_config = OrchestratorConfig(
    market_open=time(9, 30),
    market_close=time(16, 0),
    feature_refresh_interval_minutes=5,
    signal_generation_interval_minutes=1,
    daily_eval_time=time(16, 30),
    weekly_optimization_day=6,  # Sunday
    weekly_optimization_time=time(18, 0)
)

# Promotion criteria
promotion_criteria = PromotionCriteria(
    min_shadow_days=5,
    min_shadow_sharpe=0.4,
    max_shadow_drawdown=0.10,
    min_shadow_trades=20,
    no_alerts_required=True
)
```

### Running Tests

```bash
cd services/runner_py
pytest tests/ -v

# Run integration tests
pytest tests/test_integration.py -v
```

### Example Usage

```python
import asyncio
from datetime import datetime, UTC, time
from pathlib import Path
from runner_py.orchestrator import AutonomousOrchestrator, OrchestratorConfig, TaskType, ScheduleFrequency
from runner_py.shadow import ShadowExecutor, ShadowConfig
from runner_py.types import Signal, SignalDirection, MarketData
from runner_py.signal_log import SignalLogger, SignalLogConfig

# Configure shadow executor
shadow_config = ShadowConfig(
    strategy_id="momentum_v1",
    symbols=["SPY", "QQQ"],
    max_position_size=10000.0
)
shadow = ShadowExecutor(shadow_config)

# Create signal
signal = Signal(
    timestamp=datetime.now(UTC),
    symbol="SPY",
    signal_type=SignalDirection.LONG,
    strength=0.8,
    target_notional=5000.0,
    strategy_id="momentum_v1"
)

# Create market data
market_data = MarketData(
    symbol="SPY",
    timestamp=datetime.now(UTC),
    bid=450.00,
    ask=450.05,
    last=450.02
)

# Execute in shadow mode
result = shadow.execute(signal, market_data)
print(f"Success: {result.success}")
print(f"Fill Price: {result.fill.simulated_price if result.fill else 'N/A'}")
print(f"Slippage: {result.fill.simulated_slippage_bps if result.fill else 0} bps")

# Log signal
log_config = SignalLogConfig(
    log_path=Path("./logs/signals"),
    buffer_size=100
)
logger = SignalLogger(log_config)
logger.log_signal(signal, result)
logger.flush()

# Run autonomous orchestrator
async def run_orchestrator():
    config = OrchestratorConfig(
        market_open=time(9, 30),
        market_close=time(16, 0)
    )
    orchestrator = AutonomousOrchestrator(config)
    orchestrator.schedule_default_tasks()

    # Run for a period
    try:
        await asyncio.wait_for(orchestrator.start(), timeout=60)
    except asyncio.TimeoutError:
        await orchestrator.stop()

asyncio.run(run_orchestrator())
```

---

## Summary

| Service | Primary Function | Key Technologies |
|---------|-----------------|------------------|
| **ingestor_py** | Market data ingestion | Alpaca API, Parquet, WebSocket |
| **feature_builder_py** | Feature construction | Polars, Multi-timeframe |
| **labeler_py** | Target labeling | Forward returns, Quote pricing |
| **backtester_py** | Backtest simulation | Quote-based fills, Slippage modeling |
| **optimizer_py** | Strategy optimization | Grid/Random search, Ranking |
| **registry_api_py** | Strategy registry | FastAPI, SQLAlchemy, State machine |
| **monitor_py** | Performance monitoring | Drawdown, Slippage, Rollback |
| **runner_py** | Execution runner | Shadow/Paper trading, Orchestration |

All services use:
- **Polars** for DataFrame operations
- **pytest** for testing
- **Type hints** for code clarity
- **Dataclasses** for structured data
- **Structured logging** via structlog
