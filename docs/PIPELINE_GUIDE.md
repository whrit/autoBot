# Phase 1 Trading Pipeline Guide

This document provides a comprehensive technical guide to the data flow through the Phase 1 autonomous equities taker trading engine.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Ingestion Flow](#2-data-ingestion-flow)
3. [Feature Construction Pipeline](#3-feature-construction-pipeline)
4. [Labeling Process](#4-labeling-process)
5. [Backtesting Pipeline](#5-backtesting-pipeline)
6. [Optimization Loop](#6-optimization-loop)
7. [Execution Flow](#7-execution-flow)
8. [Data Schemas Reference](#8-data-schemas-reference)

---

## 1. Pipeline Overview

The Phase 1 pipeline implements a complete autonomous trading loop:

```
+------------------+     +--------------------+     +----------------+
|                  |     |                    |     |                |
|  DATA INGESTION  | --> |  FEATURE BUILDING  | --> |    LABELING    |
|                  |     |                    |     |                |
+------------------+     +--------------------+     +----------------+
         |                        |                        |
         v                        v                        v
+------------------+     +--------------------+     +----------------+
|                  |     |                    |     |                |
|   Parquet Lake   |     |   Decision Frame   |     |  Forward Rets  |
|   (Immutable)    |     |   Feature Matrix   |     |  + Direction   |
|                  |     |                    |     |                |
+------------------+     +--------------------+     +----------------+
                                  |
                                  v
              +-------------------------------------------+
              |                                           |
              |            BACKTESTING ENGINE             |
              |   (Quote-based fills + Slippage Model)    |
              |                                           |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |                                           |
              |           OPTIMIZER / RANKER              |
              |   (Walk-forward + Complexity Penalties)   |
              |                                           |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |                                           |
              |              EXECUTION FLOW               |
              |   Shadow --> Paper --> [Live Phase 2+]    |
              |                                           |
              +-------------------------------------------+
```

### Key Properties

- **Immutable Data Lake**: All raw data is stored as Parquet files, never modified
- **As-Of Joins**: All feature construction uses point-in-time joins (zero lookahead)
- **Quote-Based Fills**: Taker execution model using bid/ask, not mid-price
- **Microstructure-Aware**: Sub-minute features (5s, 15s, 30s) capture market dynamics
- **Walk-Forward Validation**: Rolling train/test splits prevent overfitting
- **Automatic Promotion**: Shadow -> Paper promotion based on live performance

---

## 2. Data Ingestion Flow

### 2.1 Architecture

```
                     +---------------------------+
                     |      ALPACA DATA API      |
                     |  (REST + WebSocket Feeds) |
                     +---------------------------+
                              |        |
              +---------------+        +---------------+
              |                                        |
              v                                        v
    +------------------+                    +------------------+
    |    HISTORICAL    |                    |    REAL-TIME     |
    |    BACKFILL      |                    |    STREAMING     |
    |                  |                    |                  |
    | BackfillOrch.    |                    | RealtimeStreamer |
    +------------------+                    +------------------+
              |                                        |
              +---------------+        +---------------+
                              |        |
                              v        v
                     +---------------------------+
                     |     PARQUET WRITER        |
                     |  (Partitioned by dt/sym)  |
                     +---------------------------+
                              |
                              v
                     +---------------------------+
                     |       DATA LAKE           |
                     |     lake/raw/{type}/      |
                     |  dt=YYYY-MM-DD/symbol=X/  |
                     +---------------------------+
```

### 2.2 Historical Backfill

The `BackfillOrchestrator` coordinates downloading historical data:

**Location**: `/home/bw/Projects/autoBot/services/ingestor_py/ingestor_py/backfill.py`

```python
# Initialize backfill orchestrator
orchestrator = BackfillOrchestrator(
    api_key=api_key,
    api_secret=api_secret,
    lake_path="/path/to/lake",
    feed="iex"  # or "sip" for paid feed
)

# Backfill last 30 days
result = orchestrator.backfill_date_range(
    symbols=["SPY", "QQQ", "AAPL"],
    days=30,
    data_types=["trades", "quotes", "bars"]
)
```

**Data Types Fetched**:
- **Trades**: Individual trade executions (price, size, exchange, conditions)
- **Quotes**: NBBO quotes (bid/ask price and size)
- **Bars**: Provider 1-minute bars (validation only)

### 2.3 Real-Time WebSocket Streaming

The `RealtimeStreamer` handles live data ingestion:

**Location**: `/home/bw/Projects/autoBot/services/ingestor_py/ingestor_py/streaming.py`

**Features**:
- Dual timestamp capture: `ts_event` (exchange) + `ts_recv` (local receipt)
- Batch writing (1000 records default) for efficiency
- Exponential backoff reconnection (max 10 attempts)
- Automatic buffer flushing on disconnect

```python
streamer = RealtimeStreamer(
    api_key=api_key,
    api_secret=api_secret,
    lake_path="/path/to/lake",
    symbols=["SPY", "QQQ"],
    feed="iex"
)

# Start async streaming
await streamer.start()
```

### 2.4 Parquet Storage Structure

**Location**: `/home/bw/Projects/autoBot/services/ingestor_py/ingestor_py/writer.py`

```
lake/
+-- raw/
    +-- trades/
    |   +-- dt=2024-01-15/
    |   |   +-- symbol=SPY/
    |   |   |   +-- data.parquet
    |   |   +-- symbol=QQQ/
    |   |       +-- data.parquet
    |   +-- dt=2024-01-16/
    |       +-- ...
    +-- quotes/
    |   +-- dt=YYYY-MM-DD/
    |       +-- symbol=XXX/
    |           +-- data.parquet
    +-- bars_provider/
        +-- dt=YYYY-MM-DD/
            +-- symbol=XXX/
                +-- data.parquet
```

**Partitioning Scheme**: `dt=YYYY-MM-DD/symbol=XXX/`

This enables:
- Efficient date-range queries (partition pruning)
- Symbol-level parallelism
- Easy incremental updates

---

## 3. Feature Construction Pipeline

### 3.1 Architecture

```
+------------------+     +------------------+     +------------------+
|   RAW TRADES     |     |   RAW QUOTES     |     |   RAW TRADES     |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        |
+------------------+     +------------------+              |
| MICROSTRUCTURE   |     | MICROSTRUCTURE   |              |
|   BAR BUILDER    |     |   BAR BUILDER    |              |
|    (5s,15s,30s)  |     |                  |              |
+------------------+     +------------------+              |
         |                        |                        |
         |    +-------------------+                        |
         |    |                                            |
         v    v                                            v
+------------------+                            +------------------+
| MICRO BARS       |                            | STANDARD BARS    |
| - spread         |                            |   (1m, 5m, 15m)  |
| - microprice     |                            | - OHLCV          |
| - quote_imbalance|                            | - returns        |
| - trade_imbalance|                            | - ATR            |
| - realized_vol   |                            | - realized_vol   |
+------------------+                            +------------------+
         |                                               |
         +-------------------+       +-------------------+
                             |       |
                             v       v
                    +---------------------------+
                    |      AS-OF JOINER         |
                    |  (Point-in-Time Correct)  |
                    +---------------------------+
                             |
                             v
                    +---------------------------+
                    |    DECISION FRAME         |
                    |    FEATURE MATRIX         |
                    +---------------------------+
```

### 3.2 Microstructure Bars (5s, 15s, 30s)

**Location**: `/home/bw/Projects/autoBot/services/feature_builder_py/feature_builder_py/micro_bars.py`

The `MicrostructureBarBuilder` computes high-frequency features:

| Feature | Formula | Description |
|---------|---------|-------------|
| `vwap` | sum(price * size) / sum(size) | Volume-weighted average price |
| `midprice` | (last_bid + last_ask) / 2 | Simple midpoint |
| `microprice` | (bid * ask_size + ask * bid_size) / (bid_size + ask_size) | Size-weighted fair value |
| `spread` | avg(ask - bid) | Average bid-ask spread |
| `quote_imbalance` | (bid_size - ask_size) / (bid_size + ask_size) | Order book imbalance [-1, 1] |
| `trade_imbalance` | (buy_vol - sell_vol) / (buy_vol + sell_vol) | Trade flow imbalance (tick rule) |
| `realized_vol` | std(intra-bar returns) | High-frequency volatility |

**Tick Rule for Trade Classification**:
```
if price > prev_price:  direction = BUY (+1)
if price < prev_price:  direction = SELL (-1)
if price == prev_price: direction = SAME (0)
```

### 3.3 Standard Bars (1m, 5m, 15m)

**Location**: `/home/bw/Projects/autoBot/services/feature_builder_py/feature_builder_py/bars.py`

The `StandardBarBuilder` computes OHLCV bars with derived features:

| Feature | Formula | Description |
|---------|---------|-------------|
| `open` | first trade price | Opening price |
| `high` | max(prices) | High price |
| `low` | min(prices) | Low price |
| `close` | last trade price | Closing price |
| `volume` | sum(sizes) | Total volume |
| `returns` | close / prev_close - 1 | Close-to-close return |
| `atr` | rolling_mean(true_range, 14) | Average True Range |
| `realized_vol` | rolling_std(returns, 14) | Return volatility |

**True Range Calculation**:
```
true_range = max(
    high - low,
    abs(high - prev_close),
    abs(low - prev_close)
)
```

### 3.4 As-Of Joins (Zero Leakage)

**Location**: `/home/bw/Projects/autoBot/services/feature_builder_py/feature_builder_py/joins.py`

The `AsOfJoiner` ensures point-in-time correctness:

```
Decision Time (t)
      |
      v
+-----+-----------------------------------------------------+
|     |                                                     |
|  PAST DATA ONLY                    FUTURE DATA (HIDDEN)   |
|                                                           |
|  Micro bars ending <= t            Micro bars ending > t  |
|  1m bars ending <= t               1m bars ending > t     |
|  5m bars ending <= t               5m bars ending > t     |
|  15m bars ending <= t              15m bars ending > t    |
|                                                           |
+-----------------------------------------------------------+
```

**Join Strategy**: `backward` (find most recent record where `bar_end <= decision_ts`)

```python
joiner = AsOfJoiner()
result = joiner.join(
    left=decision_times_df,      # decision_ts column
    right=feature_bars_df,       # bar_end column
    on="decision_ts",
    by="bar_end"
)
```

### 3.5 Decision Frame Builder

**Location**: `/home/bw/Projects/autoBot/services/feature_builder_py/feature_builder_py/decision_frame.py`

The `DecisionFrameBuilder` combines all timeframes:

```python
builder = DecisionFrameBuilder(schema_version="1.0.0")

decision_frame = builder.build(
    decision_times=decision_times_df,  # 1-minute clock
    micro_bars_30s=micro_30s_df,
    bars_1m=bars_1m_df,
    bars_5m=bars_5m_df,
    bars_15m=bars_15m_df,
    symbol="SPY"
)
```

**Output Schema**:
```
+------------------+-------------------------------------------+
| Column           | Description                               |
+------------------+-------------------------------------------+
| symbol           | Trading symbol                            |
| decision_ts      | Decision timestamp                        |
| spread_30s       | 30-second spread                          |
| microprice_30s   | 30-second microprice                      |
| quote_imbalance_30s | 30-second order book imbalance         |
| vol_30s          | 30-second realized volatility             |
| ret_1m           | 1-minute return                           |
| vol_1m           | 1-minute realized volatility              |
| atr_1m           | 1-minute ATR                              |
| ret_5m           | 5-minute return                           |
| trend_5m         | 5-minute trend (close - open)             |
| vol_5m           | 5-minute realized volatility              |
| ret_15m          | 15-minute return                          |
| trend_15m        | 15-minute trend                           |
| vol_15m          | 15-minute realized volatility             |
| schema_version   | Feature schema version                    |
+------------------+-------------------------------------------+
```

---

## 4. Labeling Process

### 4.1 Architecture

```
+------------------+     +------------------+     +------------------+
| DECISION FRAME   |     |   RAW QUOTES     |     |   HORIZONS       |
| (decision_ts)    |     | (bid/ask prices) |     | [60, 300, 900]s  |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                    +---------------------------+
                    |  FORWARD RETURNS CALC     |
                    |  (Midprice or Microprice) |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   NET-OF-SPREAD CALC      |
                    | return_net = return_mid   |
                    |  - half_spread_cost       |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   DIRECTION LABELER       |
                    |  +1 if ret > threshold    |
                    |   0 if |ret| < threshold  |
                    |  -1 if ret < -threshold   |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |       LABELS OUTPUT       |
                    |  fwd_return_mid           |
                    |  fwd_return_net           |
                    |  direction (-1, 0, +1)    |
                    +---------------------------+
```

### 4.2 Forward Returns Calculation

**Location**: `/home/bw/Projects/autoBot/services/labeler_py/labeler_py/returns.py`

The `ForwardReturnsCalculator` computes future returns using quote prices:

```python
calculator = ForwardReturnsCalculator(horizons=[60, 300, 900])  # 1m, 5m, 15m

returns_df = calculator.calculate(
    quotes_df=quotes,
    decision_times=decision_timestamps,
    symbol="SPY",
    use_microprice=False  # True for size-weighted price
)
```

**Critical Design Decision**: Uses midprice/microprice from quotes, NOT last trade price. This provides a more accurate estimate of fair value for taker execution.

**Midprice**:
```
midprice = (bid_price + ask_price) / 2
```

**Microprice** (size-weighted):
```
microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
```

### 4.3 Net-of-Spread Returns

**Location**: `/home/bw/Projects/autoBot/services/labeler_py/labeler_py/net_returns.py`

The `NetOfSpreadCalculator` adjusts returns for transaction costs:

```python
calculator = NetOfSpreadCalculator(half_spread_factor=1.0)

net_returns_df = calculator.calculate_net_returns(returns_df)
```

**Formula**:
```
half_spread = (ask - bid) / 2
midprice = (ask + bid) / 2
spread_cost = (half_spread / midprice) * half_spread_factor

fwd_return_net = fwd_return_mid - spread_cost
```

### 4.4 Direction Labels

**Location**: `/home/bw/Projects/autoBot/services/labeler_py/labeler_py/labels.py`

The `DirectionLabeler` converts returns to trading signals:

```python
labeler = DirectionLabeler(no_trade_threshold=0.0005)  # 5 basis points

labeled_df = labeler.label_returns(returns_df)
```

**Direction Logic**:
```
if fwd_return > +threshold:  direction = +1 (LONG)
if fwd_return < -threshold:  direction = -1 (SHORT)
otherwise:                   direction =  0 (NO TRADE)
```

The no-trade band filters out small returns that would be unprofitable after costs.

### 4.5 Label Output Schema

```
+------------------+-------------------------------------------+
| Column           | Description                               |
+------------------+-------------------------------------------+
| symbol           | Trading symbol                            |
| decision_ts      | Decision timestamp                        |
| horizon          | Forward horizon in seconds                |
| fwd_return_mid   | Forward return (gross)                    |
| fwd_return_net   | Forward return (net of spread)            |
| direction        | -1 (short), 0 (flat), +1 (long)           |
+------------------+-------------------------------------------+
```

---

## 5. Backtesting Pipeline

### 5.1 Architecture

```
+------------------+     +------------------+     +------------------+
| DECISION FRAME   |     |    STRATEGY      |     |   QUOTE DATA     |
| (Features)       |     |   (Signals)      |     | (Bid/Ask/Size)   |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                    +---------------------------+
                    |      FILL SIMULATOR       |
                    |  Buy @ ASK + slippage     |
                    |  Sell @ BID - slippage    |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |    SLIPPAGE INTEGRATOR    |
                    |  slip = a*spread +        |
                    |         b*(order/book) +  |
                    |         c*vol             |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   WALK-FORWARD ENGINE     |
                    |  Rolling train/test       |
                    |  No lookahead bias        |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   METRICS CALCULATOR      |
                    |  Sharpe, Sortino, MDD,    |
                    |  Profit Factor, Win Rate  |
                    +---------------------------+
```

### 5.2 Quote-Based Fill Simulation

**Location**: `/home/bw/Projects/autoBot/services/backtester_py/backtester_py/fills.py`

The `FillSimulator` implements realistic taker execution:

**Fill Price Logic**:
```
BUY ORDER:   fill_price = ask_price + slippage
SELL ORDER:  fill_price = bid_price - slippage
```

**Rejection Conditions**:
- Stale quote (age > max_quote_age_seconds)
- Crossed market (bid > ask)
- No liquidity on relevant side

```python
simulator = FillSimulator(cost_model, max_quote_age_seconds=5.0)

result = simulator.simulate_fill(
    side="buy",
    bid_price=150.00,
    ask_price=150.02,
    bid_size=1000,
    ask_size=900,
    order_notional=10000,
    short_term_vol=0.001,
    order_time=order_ts,
    quote_time=quote_ts
)
```

### 5.3 Slippage Model

**Location**: `/home/bw/Projects/autoBot/services/backtester_py/backtester_py/slippage.py`

The slippage model uses three factors:

```
slippage_bps = a * spread_bps
             + b * (order_notional / book_notional) * 100
             + c * short_term_vol * 100
```

| Coefficient | Factor | Description |
|-------------|--------|-------------|
| a | spread_bps | Spread impact |
| b | order/book ratio | Market impact |
| c | volatility | Timing risk |

**Slippage Application**:
```python
integrator = SlippageIntegrator(cost_model)

fill_price, slippage_bps = integrator.calculate_fill_price(
    side="buy",
    bid_price=150.00,
    ask_price=150.02,
    order_notional=10000,
    book_notional=150000,
    short_term_vol=0.001
)
```

### 5.4 Walk-Forward Evaluation

**Location**: `/home/bw/Projects/autoBot/services/backtester_py/backtester_py/walk_forward.py`

The `WalkForwardOptimizer` creates rolling train/test splits:

```
Time ------>

|<-- Train Window -->|<-- Test -->|
                     |<-- Train Window -->|<-- Test -->|
                                          |<-- Train Window -->|<-- Test -->|

Step Size: How far to advance between windows
```

```python
optimizer = WalkForwardOptimizer(
    train_size=252 * 5,   # 5 years training
    test_size=252,        # 1 year test
    step_size=63          # Advance 3 months
)

for train_df, test_df in optimizer.split(data):
    # Train on train_df
    # Evaluate on test_df
    pass
```

### 5.5 Metrics Computation

**Location**: `/home/bw/Projects/autoBot/services/backtester_py/backtester_py/metrics.py`

The `MetricsCalculator` computes comprehensive performance metrics:

```python
calculator = MetricsCalculator(
    risk_free_rate=0.0,
    periods_per_year=252  # Daily bars
)

metrics = calculator.calculate(equity_curve, trades)
```

**Metrics Computed**:

| Metric | Formula | Description |
|--------|---------|-------------|
| `sharpe_ratio` | (mean_ret - rf) / std_ret * sqrt(252) | Risk-adjusted return |
| `sortino_ratio` | (mean_ret - rf) / downside_std * sqrt(252) | Downside risk-adjusted |
| `max_drawdown` | max(1 - equity / running_max) | Peak-to-trough loss |
| `profit_factor` | gross_profit / gross_loss | Win/loss ratio |
| `win_rate` | winning_trades / total_trades | Trade success rate |
| `cagr` | (final/initial)^(1/years) - 1 | Compound annual growth |
| `calmar_ratio` | cagr / max_drawdown | Return per unit drawdown |

---

## 6. Optimization Loop

### 6.1 Architecture

```
+------------------+
|  STRATEGY FAMILY |
|  - Trend         |
|  - Mean-Reversion|
|  - Volatility    |
|  - ML            |
+------------------+
         |
         v
+------------------+
|  PARAMETER SWEEP |
|  Grid search     |
|  Random search   |
+------------------+
         |
         v
+------------------+
|  WALK-FORWARD    |
|  BACKTEST        |
+------------------+
         |
         v
+------------------+     +------------------+
|  CANDIDATE       |     |  COMPLEXITY      |
|  RANKING         | <-- |  PENALTY         |
+------------------+     +------------------+
         |
         v
+------------------+
|  TOP CANDIDATES  |
|  Export to ONNX  |
+------------------+
```

### 6.2 Strategy Families

**Location**: `/home/bw/Projects/autoBot/services/optimizer_py/optimizer_py/strategies.py`

The `StrategyFamily` abstract base class defines the strategy interface:

```python
class StrategyFamily(ABC):
    family_name: ClassVar[str] = "base"

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> dict[str, Any]:
        """Get default parameters."""
        ...

    @abstractmethod
    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """Generate trading signals."""
        ...
```

**Strategy Family Types**:

| Family | Description | Key Parameters |
|--------|-------------|----------------|
| Trend | Follow momentum | lookback, threshold |
| Mean-Reversion | Trade against extremes | z_score_threshold, holding_period |
| Volatility | Breakout on vol expansion | vol_multiplier, atr_period |
| ML | XGBoost/Neural predictors | n_estimators, max_depth, learning_rate |

### 6.3 Parameter Sweep

The optimizer searches the parameter space:

```python
# Example parameter grid
param_grid = {
    "lookback": [10, 20, 50, 100],
    "threshold": [0.001, 0.002, 0.005],
    "holding_period": [1, 5, 10]
}
```

### 6.4 Candidate Ranking

**Location**: `/home/bw/Projects/autoBot/services/optimizer_py/optimizer_py/ranking.py`

The `CandidateRanker` scores and ranks strategy candidates:

```python
config = RankingConfig(
    metric=RankingMetric.COMPOSITE,
    min_trades=30,
    max_drawdown_threshold=0.20,
    sharpe_weight=0.4,
    sortino_weight=0.3,
    profit_factor_weight=0.2,
    drawdown_penalty_weight=0.1
)

ranker = CandidateRanker(config)
ranked = ranker.rank_across_families(candidates)
top_3 = ranker.get_top_candidates(candidates, n=3)
```

**Composite Score Formula**:
```
score = sharpe_weight * sharpe
      + sortino_weight * sortino
      + profit_factor_weight * profit_factor
      - drawdown_penalty_weight * max_drawdown
      - complexity_penalty
```

### 6.5 Complexity Penalties

**Location**: `/home/bw/Projects/autoBot/services/optimizer_py/optimizer_py/complexity.py`

The complexity penalty discourages overly complex strategies:

```python
complexity_config = ComplexityPenaltyConfig(
    max_parameters=10,
    param_penalty=0.01,
    depth_penalty=0.02
)

penalty = compute_complexity_penalty(strategy_params, complexity_config)
```

### 6.6 ONNX Export

**Location**: `/home/bw/Projects/autoBot/services/optimizer_py/optimizer_py/onnx_export.py`

The `ONNXExporter` exports ML models for production:

```python
exporter = ONNXExporter()

config = ONNXExportConfig(
    model_path=Path("artifacts/model.onnx"),
    input_features=[
        "spread_30s", "microprice_30s", "quote_imbalance_30s", "vol_30s",
        "ret_1m", "vol_1m", "atr_1m",
        "ret_5m", "trend_5m", "vol_5m",
        "ret_15m", "trend_15m", "vol_15m"
    ],
    opset_version=15
)

onnx_bytes = exporter.export_xgboost(model, config)
```

**ONNX Model Contract**:

**Inputs** (13 features, exact order):
```
[spread_30s, microprice_30s, quote_imbalance_30s, vol_30s,
 ret_1m, vol_1m, atr_1m,
 ret_5m, trend_5m, vol_5m,
 ret_15m, trend_15m, vol_15m]
```

**Outputs**:
```
expected_return: float
confidence_score: float
```

---

## 7. Execution Flow

### 7.1 Architecture

```
+------------------+
|  PROMOTED        |
|  STRATEGY        |
+------------------+
         |
         v
+------------------+     +------------------+
|  SHADOW MODE     |     |  Signal Logging  |
|  (No Orders)     | --> |  Performance     |
|                  |     |  Tracking        |
+------------------+     +------------------+
         |
         | (Auto-promotion if criteria met)
         v
+------------------+     +------------------+
|  PAPER MODE      |     |  Alpaca Paper    |
|  (Paper Account) | --> |  API Execution   |
|                  |     |  Real Fills      |
+------------------+     +------------------+
         |
         | (Phase 2+)
         v
+------------------+
|  LIVE MODE       |
|  (Real Capital)  |
+------------------+
```

### 7.2 Shadow Mode (Signal Logging)

**Location**: `/home/bw/Projects/autoBot/services/runner_py/runner_py/shadow.py`

The `ShadowExecutor` simulates execution without real orders:

```python
config = ShadowConfig(
    strategy_id="trend_5m_v3",
    symbols=["SPY", "QQQ"],
    max_position_size=10000.0,
    slippage_model_bps=2.0,
    check_risk_limits=True
)

executor = ShadowExecutor(config)

result = executor.execute(signal, market_data)
```

**Shadow Execution Process**:
1. Risk limit check (symbol allowed, position size)
2. Fill simulation (apply slippage to quote)
3. Position tracking (P&L, quantity)
4. Metrics collection

**Shadow Fill Simulation**:
```
BUY:  simulated_price = ask + (ask * slippage_bps / 10000)
SELL: simulated_price = bid - (bid * slippage_bps / 10000)
```

### 7.3 Paper Mode (Alpaca Paper API)

**Location**: `/home/bw/Projects/autoBot/services/runner_py/runner_py/paper.py`

The `PaperExecutor` sends real orders to Alpaca's paper trading:

```python
config = PaperConfig(
    api_key="...",
    api_secret="...",
    paper=True,  # CRITICAL: Always True
    max_position_value=10000.0
)

executor = PaperExecutor(config)

result = await executor.execute(signal)
```

**Paper Execution Features**:
- Real order submission via Alpaca API
- Market orders with notional sizing
- Position tracking via API
- Order status monitoring
- Fill confirmation

### 7.4 Promotion Workflow

**Location**: `/home/bw/Projects/autoBot/services/runner_py/runner_py/promotion.py`

The `AutoPromoter` manages automatic promotion based on criteria:

```python
criteria = PromotionCriteria(
    min_shadow_days=5,
    min_shadow_sharpe=0.4,
    max_shadow_drawdown=0.10,
    min_shadow_trades=20,
    no_alerts_required=True
)

promoter = AutoPromoter(criteria)

# Evaluate promotion eligibility
evaluation = await promoter.evaluate_promotion(strategy_id)

if evaluation.eligible:
    success = await promoter.promote_strategy(strategy_id)
```

**Promotion State Machine**:

```
           +------------+
           | CANDIDATE  |
           +------------+
                 |
                 v
           +------------+
           |   SHADOW   | <---+
           +------------+     |
                 |            |
                 v            |
           +------------+     |
           |   PAPER    | ----+  (Rollback)
           +------------+
                 |
                 v
           +------------+
           |    LIVE    |  (Phase 2+)
           +------------+
                 |
                 v
           +------------+
           |  RETIRED   |
           +------------+
```

**Valid State Transitions**:
- CANDIDATE -> SHADOW, RETIRED
- SHADOW -> PAPER, RETIRED
- PAPER -> LIVE, SHADOW (rollback), RETIRED
- LIVE -> PAPER (rollback), RETIRED
- RETIRED -> (terminal)

### 7.5 Promotion Criteria

| Criterion | Default | Description |
|-----------|---------|-------------|
| min_shadow_days | 5 | Minimum shadow trading days |
| min_shadow_sharpe | 0.4 | Minimum Sharpe ratio |
| max_shadow_drawdown | 0.10 | Maximum drawdown (10%) |
| min_shadow_trades | 20 | Minimum trade count |
| no_alerts_required | True | No active alerts |

---

## 8. Data Schemas Reference

### 8.1 Raw Trades Schema

**Path**: `lake/raw/trades/dt=YYYY-MM-DD/symbol=XXX/`

```
+------------+------------------+-----------------------------------+
| Field      | Type             | Description                       |
+------------+------------------+-----------------------------------+
| ts_event   | timestamp[us,UTC]| Exchange timestamp                |
| ts_recv    | timestamp[us,UTC]| Local receipt timestamp           |
| price      | float64          | Trade price                       |
| size       | float64          | Trade size                        |
| exchange   | string           | Exchange code                     |
| conditions | string           | Trade conditions (space-separated)|
+------------+------------------+-----------------------------------+
```

### 8.2 Raw Quotes Schema

**Path**: `lake/raw/quotes/dt=YYYY-MM-DD/symbol=XXX/`

```
+-------------+------------------+-----------------------------------+
| Field       | Type             | Description                       |
+-------------+------------------+-----------------------------------+
| ts_event    | timestamp[us,UTC]| Exchange timestamp                |
| ts_recv     | timestamp[us,UTC]| Local receipt timestamp           |
| bid_price   | float64          | Best bid price                    |
| bid_size    | float64          | Size at best bid                  |
| ask_price   | float64          | Best ask price                    |
| ask_size    | float64          | Size at best ask                  |
| bid_exchange| string           | Bid exchange code                 |
| ask_exchange| string           | Ask exchange code                 |
| conditions  | string           | Quote conditions                  |
+-------------+------------------+-----------------------------------+
```

### 8.3 Microstructure Bars Schema

**Path**: `lake/features/micro_bars/{granularity}/`

```
+----------------+------------------+-----------------------------------+
| Field          | Type             | Description                       |
+----------------+------------------+-----------------------------------+
| symbol         | string           | Trading symbol                    |
| bar_start      | timestamp[us,UTC]| Bar start timestamp               |
| bar_end        | timestamp[us,UTC]| Bar end timestamp                 |
| vwap           | float64          | Volume-weighted average price     |
| midprice       | float64          | (bid + ask) / 2                   |
| microprice     | float64          | Size-weighted fair value          |
| spread         | float64          | Average bid-ask spread            |
| bid_size       | float64          | Average bid size                  |
| ask_size       | float64          | Average ask size                  |
| quote_imbalance| float64          | (bid_size - ask_size) / total     |
| trade_imbalance| float64          | (buy_vol - sell_vol) / total      |
| trade_volume   | float64          | Total trade volume                |
| realized_vol   | float64          | Intra-bar return volatility       |
+----------------+------------------+-----------------------------------+
```

### 8.4 Standard Bars Schema

**Path**: `lake/features/bars/{granularity}/`

```
+-------------+------------------+-----------------------------------+
| Field       | Type             | Description                       |
+-------------+------------------+-----------------------------------+
| symbol      | string           | Trading symbol                    |
| bar_start   | timestamp[us,UTC]| Bar start timestamp               |
| bar_end     | timestamp[us,UTC]| Bar end timestamp                 |
| open        | float64          | Opening price                     |
| high        | float64          | High price                        |
| low         | float64          | Low price                         |
| close       | float64          | Closing price                     |
| volume      | float64          | Total volume                      |
| returns     | float64          | Close-to-close return             |
| atr         | float64          | Average True Range                |
| realized_vol| float64          | Rolling return volatility         |
+-------------+------------------+-----------------------------------+
```

### 8.5 Decision Frame Schema

**Path**: `lake/features/decision_frame/`

```
+-------------------+------------------+-------------------------------+
| Field             | Type             | Description                   |
+-------------------+------------------+-------------------------------+
| symbol            | string           | Trading symbol                |
| decision_ts       | timestamp[us,UTC]| Decision timestamp            |
| spread_30s        | float64          | 30s spread                    |
| microprice_30s    | float64          | 30s microprice                |
| quote_imbalance_30s| float64         | 30s order book imbalance      |
| vol_30s           | float64          | 30s realized volatility       |
| ret_1m            | float64          | 1m return                     |
| vol_1m            | float64          | 1m realized volatility        |
| atr_1m            | float64          | 1m ATR                        |
| ret_5m            | float64          | 5m return                     |
| trend_5m          | float64          | 5m trend (close - open)       |
| vol_5m            | float64          | 5m realized volatility        |
| ret_15m           | float64          | 15m return                    |
| trend_15m         | float64          | 15m trend                     |
| vol_15m           | float64          | 15m realized volatility       |
| schema_version    | string           | Feature schema version        |
+-------------------+------------------+-------------------------------+
```

### 8.6 Labels Schema

**Path**: `lake/labels/`

```
+---------------+------------------+-----------------------------------+
| Field         | Type             | Description                       |
+---------------+------------------+-----------------------------------+
| symbol        | string           | Trading symbol                    |
| decision_ts   | timestamp[us,UTC]| Decision timestamp                |
| horizon       | int32            | Forward horizon (seconds)         |
| fwd_return_mid| float64          | Forward return (midprice)         |
| fwd_return_net| float64          | Forward return (net of spread)    |
| direction     | int8             | -1 (short), 0 (flat), +1 (long)   |
+---------------+------------------+-----------------------------------+
```

---

## Summary

The Phase 1 pipeline implements a complete, production-grade autonomous trading system with:

1. **Robust Data Ingestion**: Historical backfill and real-time streaming with dual timestamps
2. **Microstructure-Aware Features**: High-frequency (5s-30s) features capturing market dynamics
3. **Zero-Leakage Design**: As-of joins ensure point-in-time correctness
4. **Realistic Execution Modeling**: Quote-based fills with configurable slippage
5. **Walk-Forward Validation**: Rolling train/test splits prevent overfitting
6. **Complexity-Penalized Ranking**: Discourages overfit strategies
7. **Automatic Promotion**: Shadow -> Paper based on live performance metrics

The system is designed for autonomous operation while maintaining full auditability and the ability to roll back to previous artifacts at any stage.
