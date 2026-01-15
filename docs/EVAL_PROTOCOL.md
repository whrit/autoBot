# EVAL_PROTOCOL.md — Evaluation & Promotion Protocol

This document defines **exact metrics, thresholds, and gates** for:
- candidate selection (offline)
- shadow validation
- paper trading validation
- canary and full live promotion (Phase 2+)

All thresholds are **defaults**; they should be parameterized per strategy family and liquidity bucket.

---

## 1) Definitions

### Time segments
- **Train**: historical window for fitting params/models
- **Validate**: used for early stopping / tuning
- **Test**: held-out walk-forward segment
- **Shadow**: live signals only, no orders
- **Paper**: paper account execution
- **Canary**: small notional live deployment
- **Full**: production deployment

### Return conventions
- Net returns are **after** modeled or realized costs.
- For bars-based backtests: fills are **quote-based** (crossing bid/ask) plus modeled slippage.

---

## 2) Core Metrics (computed per strategy, per symbol, and portfolio)

### Profitability & quality
- **Sharpe (annualized)**: mean / std of daily net returns
- **Sortino (annualized)**: mean / std of downside daily net returns
- **Profit factor**: gross profits / gross losses
- **Hit rate**: fraction of trades with positive net PnL
- **Expectancy**: avg(net pnl per trade)

### Risk & tail
- **Max drawdown (MDD)**: peak-to-trough net equity
- **CVaR(95%)**: conditional tail loss
- **Worst day**: min daily net return
- **Volatility (ann.)**: std of daily net returns

### Trading frictions & realism
- **Turnover**: total traded notional / avg equity
- **Avg spread paid**: avg(|fill - mid| / mid)
- **Slippage (bps)**: realized vs expected execution price, per side
- **Fill rate**: filled qty / submitted qty

### Stability & robustness
- **Parameter sensitivity**: performance under ±10% perturbations
- **Regime stability**: performance across regime buckets
- **Cost sensitivity**: performance under cost multipliers (1.0x, 1.5x, 2.0x)

---

## 3) Offline Acceptance Criteria (Candidate → Eligible for Shadow)

A candidate is eligible for shadow if all are true on the **walk-forward test**:

### Portfolio-level (minimums)
- Sharpe (net) ≥ **0.8**
- Max drawdown ≤ **12%**
- Profit factor ≥ **1.15**
- Worst day ≥ **-4.0%**
- Turnover ≤ **4.0x / day** (or family-specific cap)

### Robustness requirements
- Under **1.5x costs**, Sharpe ≥ **0.4**
- Under **2.0x costs**, MDD ≤ **18%**
- Parameter sensitivity: Sharpe drop ≤ **35%** under ±10% params
- Must be positive net PnL in ≥ **60%** of rolling 1-month windows in test

### Incumbent comparison (if exists)
- Candidate net Sharpe ≥ incumbent net Sharpe + **0.2**
  OR
- Candidate MDD ≤ incumbent MDD - **2%** with similar Sharpe (within 0.1)

---

## 4) Shadow Gate (Shadow → Paper)

Shadow runs for **N_shdw = 5 trading days** minimum.

Pass if:
- Signal rate within expected band: ± **25%** vs backtest median
- Feature drift (PSI): each critical feature PSI ≤ **0.2**
- No feed integrity issues:
  - missing quotes/trades < **0.5%**
  - max data gap ≤ **5s** during market hours
- Estimated slippage (from quotes) ≤ backtest p75 + **5 bps**

Fail-fast conditions:
- Any day with projected MDD > **6%**
- Any feature PSI ≥ **0.35** on critical features

---

## 5) Paper Gate (Paper → Canary / Live-Eligible)

Paper runs for **N_paper = 10 trading days** minimum.

Pass if:
- Realized slippage (median) ≤ **backtest median + 7 bps**
- Fill rate ≥ **95%** (taker orders; adjustable)
- Daily loss limit never breached
- Rolling 5-day net return ≥ **-1.5%**
- Paper Sharpe (annualized, short sample) ≥ **0.0** (non-negative)

Fail-fast:
- Any single-day loss ≤ **-3.0%**
- Reject rate ≥ **2%** of orders
- Persistent quote lag > **250ms** (if measured) or systematic stale quotes

---

## 6) Canary Gate (Phase 2+) (Canary → Full)

Canary uses **5–10%** of intended max notional for **N_canary = 10 trading days**.

Pass if:
- Live slippage median ≤ paper median + **5 bps**
- Live MDD ≤ **5%**
- Live Sharpe ≥ **0.3** (informational but required to be positive expectancy)
- No operational incidents triggered (feed, order routing, risk)

Rollback triggers:
- Live MDD ≥ **6%**
- Slippage p90 ≥ paper p90 + **10 bps** for 2 consecutive days
- Reject rate ≥ **3%** any day
- Unexplained drift PSI ≥ **0.35** on critical features

---

## 7) Allocator & Library Evaluation (Phase 3)

Allocator is evaluated on:
- Portfolio Sharpe vs best single strategy: must be ≥ **90%** of best Sharpe
- Portfolio MDD vs best single strategy: must be ≤ **best MDD + 2%**
- Turnover: must be ≤ **1.25x** average turnover of constituents
- Weight stability: avg daily L1 weight change ≤ **0.25**

---

## 8) Reporting Requirements

Every run must emit:
- `metrics.json` (full metric set)
- `equity_curve.csv`
- `trades.csv` (fills + mid + spread paid)
- `params.json` or ONNX metadata
- `data_snapshot_manifest.json`

All reports are stored with immutable IDs and linked in the registry.
