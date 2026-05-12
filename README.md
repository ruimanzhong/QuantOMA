# hybrid_market_quant

Quant research and live-signal tooling for AU9999 gold, optional news/Polymarket overlays, and a side-car gold options execution plan. The current core workflow is:

```text
market/news/Polymarket data
-> gold feature dataset
-> calibrated 5-trading-day probability model
-> 0-1 AU9999 target exposure
-> optional options overlay
```

## Project Layout

- `src/market_quant/`: reusable library code for data, features, models, gold logic, LLM, options, diagnostics, and workflows.
- `scripts/`: thin command-line entrypoints grouped by task. Scripts should parse args, read config, call `src`, and print/write outputs.
- `config/`: model, data-source, feature-set, and LLM prompt/config files.
- `data/`: local raw data, features, predictions, reports, and backtests.
- `tests/`: unit and workflow tests.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional providers:

```bash
python -m pip install efinance baostock pyqlib pyarrow
```

## Gold Data And Training

Refresh gold market data:

```bash
python scripts/data/fetch_gold_data.py
```

Refresh news and Polymarket overlays:

```bash
python scripts/data/fetch_gold_news.py
python scripts/data/fetch_gold_polymarket.py
python scripts/features/process_gold_news_events.py --limit 25 --min-relevance 1 --recent-first
```

Build AU9999 Alpha158 provider and model features:

```bash
python scripts/features/build_gold_qlib_provider.py
python scripts/features/build_gold_features.py
```

Train and backtest the gold model:

```bash
python scripts/models/train_gold_model.py
python scripts/backtests/backtest_gold_model.py
python scripts/backtests/run_gold_strategy_sensitivity.py
python scripts/diagnostics/run_gold_regime_diagnostics.py
python scripts/strategies/run_gold_layered_strategy.py
```

Key outputs:

- `data/features/gold/gold_model_dataset.csv`
- `data/features/gold/gold_model_features.txt`
- `data/predictions/gold_model_walk_forward_predictions.csv`
- `data/reports/gold_model_summary.json`
- `data/reports/gold_strategy_sensitivity.csv`
- `data/backtest/gold_layered_strategy_backtest.csv`

## Gold Live Prediction

The live workflow uses the latest local dataset and trained model settings to produce:

- `probability`: calibrated probability that AU9999 has positive 5-trading-day return.
- `final_position`: target AU9999 exposure from 0 to 1.
- `execution`: suggested exposure change after rebalance band and max-change controls.
- `options`: optional side-car options plan.

Run a live or near-close prediction:

```bash
python scripts/models/predict_gold_live_signal.py \
  --signal-date 2026-05-12 \
  --as-of-date 2026-05-12 \
  --manual-primary-price 1034 \
  --current-position 0.60
```

Useful arguments:

- `--signal-date`: decision date for the new signal.
- `--as-of-date`: latest date allowed for news, Polymarket, and overlays.
- `--manual-primary-price`: current/near-close AU9999 price override.
- `--current-position`: current gold exposure from 0 to 1.
- `--rebalance-band`: skip trading when target-current exposure is too small.
- `--max-position-change`: cap one-signal exposure change.

Outputs:

- `data/reports/gold_live_signal.json`
- `data/reports/gold_live_selected_features.csv`

The model horizon is 5 trading days. This does not mean every execution should be held for exactly 5 days; the live signal should be re-evaluated daily.

## News And LLM Events

Raw news and LLM event extraction are separated:

```text
data/raw/gold/news.csv
-> scripts/features/process_gold_news_events.py
-> data/raw/gold/news_events.csv
```

The event processor sends only title/summary style inputs to the configured LLM and writes structured event fields such as `event_type`, `gold_direction`, `confidence`, `surprise`, `llm_summary`, and `rationale`. It does not store raw article body text in `news_events.csv`.

LLM config and prompt text live in:

- `config/llm_messages.yaml`
- `src/market_quant/llm/client/`
- `src/market_quant/workflows/gold_news.py`

## Gold Options Overlay

The options module is optional and lives under `src/market_quant/options`. It does not replace the AU9999 model; it attaches an execution plan under `payload["options"]` when enabled.

Timing model:

```text
Alpha signal horizon: 5 trading days
Option expiry bucket: strategy-dependent DTE
Option holding period: dynamic 1-5 trading days with daily re-evaluation
```

Default config in `config/gold_model.yaml`:

```yaml
options:
  enabled: false
  signal_horizon_days: 5
  chain_path: null
  default_multiplier: 1000.0

  expiry_selection:
    trend_min_dte: 25
    trend_max_dte: 60
    event_min_dte: 10
    event_max_dte: 30
    hedge_min_dte: 20
    hedge_max_dte: 45

  holding:
    max_holding_days: 5
    reevaluate_daily: true
    close_on_signal_flip: true
    close_on_edge_decay: true

  exits:
    long_option_take_profit_pct: 0.40
    long_option_stop_loss_pct: 0.35
    spread_take_profit_of_max_profit: 0.50
    spread_stop_loss_of_max_loss: 0.35
```

To enable options, set:

```yaml
options:
  enabled: true
  chain_path: "data/raw/gold/au9999_options_chain.csv"
  account_equity: 100000
  default_multiplier: 1000
```

Minimum option-chain columns after aliases:

- `date`
- `symbol`
- `option_type`
- `expiry`
- `strike`
- `underlying_price`

Recommended additional columns:

- `bid`
- `ask`
- `settlement`
- `volume`
- `open_interest`
- `iv`
- `multiplier`

Strategy mapping:

- `P_up >= 0.60`: prefer `bull_call_spread` using the trend DTE bucket.
- `P_up >= 0.65` and IV is not high: allow `long_call`.
- Neutral probability plus event risk or low IV: consider `long_straddle` or `long_strangle` using the event DTE bucket.
- `P_up <= 0.40`: consider bearish or hedge structures using the hedge DTE bucket.
- Credit spreads and naked short options are disabled by default.

Risk controls include delta-equivalent leverage, optional gross notional cap, premium risk, single-trade max loss, and naked-short blocking.

## A-share And ETF Utilities

The repo still includes A-share/ETF data, rule baselines, and Alpha158 diagnostics. These are secondary utilities now, but remain useful for validation and experiments.

```bash
python scripts/data/fetch_a_share_data.py
python scripts/diagnostics/check_data_coverage.py
python scripts/features/build_qlib_alpha158_features.py
python scripts/strategies/run_rule_baselines.py
python scripts/strategies/evaluate_rule_baselines.py
python scripts/strategies/compare_benchmarks.py
```

## Diagnostics

Useful diagnostics:

```bash
python scripts/diagnostics/audit_gold_pipeline.py
python scripts/diagnostics/audit_gold_news_events.py
python scripts/diagnostics/audit_gold_long_history_readiness.py
python scripts/diagnostics/diagnose_alpha158_features.py
python scripts/diagnostics/diagnose_sample_attrition.py
```

China data proxy is configured under `gold.china_gold_data.network` in `config/data_sources.yaml`. The proxy scope is limited to China ETF/SGE fetches and should not affect Tiingo, FRED, news, or Polymarket requests.

## Tests

```bash
python -B -m pytest -p no:cacheprovider
```

Current expected status is all tests passing, with at most dependency or pandas deprecation warnings.

## Common Issues

- `efinance` missing: install with `python -m pip install efinance`.
- `baostock` missing: install with `python -m pip install baostock`.
- `pyqlib` missing: install with `python -m pip install pyqlib`.
- Parquet support missing: install with `python -m pip install pyarrow`.
- LLM key missing: check `.env` or `../gold_llm_quant/.env`, and `config/llm_messages.yaml`.
- Option plan disabled: set `options.enabled: true` and configure `options.chain_path`.
