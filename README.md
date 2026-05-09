# hybrid_market_quant

`hybrid_market_quant` is a new standalone quant research project. Phase 1 focuses only on A-share/ETF daily data, Microsoft Qlib Alpha158 feature access, rule-based baselines, benchmark/backtest scaffolding, and diagnostics.

## Phase 1 Scope

Phase 1 validates:

- A-share/ETF daily data through efinance, with BaoStock kept for fallback and stock data.
- Qlib Alpha158 as the standard price/volume feature provider.
- Rule-based baseline strategies.
- Daily long-only backtest and benchmark metrics.
- Interfaces that leave room for future data-source migration.

This project does not copy Qlib Alpha158 source code. Qlib is used as a feature provider through `market_quant.features.qlib_alpha158_adapter`; it is not the only framework for the whole system.

`alpha_style_fallback` is only a lightweight fallback for non-Qlib data or environments where Qlib is unavailable. It is not a full Alpha158 implementation.

## Not In Phase 1

Phase 1 does not migrate old `gold_llm_quant` data sources, connect News, connect Polymarket, train iTransformer/Mamba/deep models, train complex ML, perform feature fusion, or implement a production trading system.

Future Phase 2 work can migrate the old `gold_llm_quant` gold, macro, News, Polymarket, and regime module data sources, then build feature-fusion workflows on top of the Phase 1 interfaces.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional providers:

```bash
python -m pip install efinance
python -m pip install baostock
python -m pip install pyqlib
python -m pip install pyarrow
```

## A-share Data Providers

Phase 1 uses `efinance` as the default A-share ETF daily provider. BaoStock remains useful for stock history, but the current BaoStock ETF path has shown limited historical coverage for the Phase 1 ETF universe: a 2015-01-01 to 2026-04-30 request can return only 76 rows from 2026-01-05 to 2026-04-29 for the configured ETFs.

BaoStock is kept as `fallback_provider` and can still be selected explicitly in `config/data_sources.yaml`. TuShare `fund_daily` is a candidate future enhancement source, but it requires the appropriate account permissions or points.

The efinance price adjustment and Eastmoney data口径 should be independently verified before production research. All provider outputs must pass the coverage check:

```bash
python scripts/check_data_coverage.py
```

Do not use the 76-row BaoStock ETF smoke-test output for strategy conclusions.

## Run

Fetch A-share ETF daily data:

```bash
python scripts/fetch_a_share_data.py
python scripts/check_data_coverage.py
```

Build Qlib Alpha158 features:

```bash
python scripts/build_qlib_alpha158_features.py
```

Run rule baselines:

```bash
python scripts/run_rule_baselines.py
```

Evaluate rule baselines against buy-and-hold:

```bash
python scripts/evaluate_rule_baselines.py
python scripts/compare_benchmarks.py
```

Outputs:

- `data/raw/a_share/a_share_etf_daily.csv`
- `data/reports/a_share_data_coverage.csv`
- `data/features/qlib_alpha158_features.parquet` or `.csv` when parquet support is unavailable
- `data/backtest/rule_baselines_daily.csv`
- `data/reports/rule_baseline_comparison.csv`
- `data/reports/rule_vs_buy_hold_comparison.csv`
- `data/reports/best_strategy_by_asset.csv`
- `data/reports/strategy_stability_summary.csv`
- `data/reports/phase1_benchmark_comparison.csv`

## Rule Baseline Evaluation

Phase 1 rule strategies must be compared with `buy_and_hold`, not judged by standalone cumulative return. A rule that makes money can still be worse than passive exposure, and a rule that reduces drawdown may only be lowering exposure without improving return or Sharpe.

The evaluation reports compare cumulative return, Sharpe, max drawdown, active ratio, and turnover. Sharpe helps normalize return by volatility, max drawdown captures path risk, active ratio shows how often capital is exposed, and turnover makes trading intensity visible.

Run:

```bash
python scripts/run_rule_baselines.py
python scripts/evaluate_rule_baselines.py
python scripts/compare_benchmarks.py
```

Outputs:

- `data/reports/rule_vs_buy_hold_comparison.csv`: per asset and strategy comparison with buy-and-hold.
- `data/reports/best_strategy_by_asset.csv`: best strategy by return, Sharpe, and drawdown for each ETF.
- `data/reports/strategy_stability_summary.csv`: cross-ETF stability summary and simple stability score.
- `data/reports/phase1_benchmark_comparison.csv`: sorted Phase 1 benchmark table for quick inspection.

This remains Phase 1. Do not jump from these reports directly to News, Polymarket, or deep models. Phase 2 should only be considered after rule baselines and data quality are stable.

## Rule Robustness Evaluation

Phase 1 should not rely on full-sample return alone. A rule can look strong because one ETF, one year range, one parameter, or one low-cost assumption happened to fit the sample.

Robustness diagnostics split results by calendar year and fixed subperiods, then stress parameters and transaction costs. Annual and subperiod checks show whether a rule survives different market windows. Parameter sensitivity helps identify single-parameter luck. Cost sensitivity shows whether turnover quietly consumes the edge.

Run:

```bash
python scripts/diagnose_rule_annual_performance.py
python scripts/diagnose_rule_subperiod_performance.py
python scripts/run_rule_parameter_sensitivity.py
python scripts/run_transaction_cost_sensitivity.py
python scripts/summarize_rule_robustness.py
```

Outputs:

- `data/reports/rule_annual_performance.csv`
- `data/reports/rule_subperiod_performance.csv`
- `data/reports/rule_parameter_sensitivity.csv`
- `data/reports/transaction_cost_sensitivity.csv`
- `data/reports/rule_robustness_summary.csv`

This is still Phase 1. Do not use a single full-sample result as a reason to jump into ML, News, Polymarket, or deep models.

## Qlib Alpha158 Validation

Qlib `cn_data` is a Qlib-format Chinese market provider. It is not the same object as the efinance ETF CSV used by the rule baselines. The Qlib Alpha158 adapter builds a feature matrix from the configured Qlib provider, while the rule baseline layer evaluates explicit trading rules on the ETF price CSV.

Phase 1 does not train LightGBM, transformers, or any deep model. The goal here is only to confirm that Alpha158 can be built and diagnosed, and to check whether the Qlib universe overlaps the efinance ETF universe.

Run:

```bash
python scripts/inspect_qlib_provider.py
python scripts/build_qlib_alpha158_features.py
python scripts/diagnose_alpha158_features.py
```

Outputs:

- `data/reports/qlib_provider_summary.json`
- `data/reports/qlib_instruments_sample.csv`
- `data/features/qlib_alpha158_features.parquet` or `data/features/qlib_alpha158_features.csv`
- `data/reports/alpha158_feature_summary.csv`
- `data/reports/alpha158_asset_coverage.csv`
- `data/reports/alpha158_price_universe_alignment.csv`

Common outcomes:

- Qlib `cn_data` often covers stock universes such as `csi300` and `csi500`, but may not cover the ETF symbols used by the efinance ETF baseline.
- If Alpha158 does not cover the efinance ETF universe, that is not automatically a bug. It means this provider/universe combination is not aligned with the ETF baseline universe.
- Later Phase 1 options include using Alpha158 for stock/index-component research, converting ETF data into a Qlib provider, or using `alpha_style_fallback` for lightweight ETF price/volume features.
- For local extension beyond 2020, you can build a Qlib provider from the ETF daily universe and then point Alpha158 at that provider.

This remains Phase 1 validation only. No LightGBM, no deep model training, no News, and no Polymarket are introduced here.

## Phase 1.1 Local ETF Alpha158

Phase 1.1 extends Phase 1 validation by converting the verified efinance ETF daily CSV into a local Qlib provider, then using Qlib's maintained Alpha158 handler against that local ETF provider.

This is still Phase 1 work because it only validates the ETF price/volume feature pipeline. It does not migrate `gold_llm_quant`, does not add macro/gold external data, does not connect News or Polymarket, and does not train ML or deep models.

This path is useful when an installed Qlib `cn_data` provider ends early, for example at 2020-09-25. Qlib cannot calculate post-2020 Alpha158 features from a provider that has no post-2020 prices, but it can calculate Alpha158 from a local provider built from the ETF data that already extends to later dates.

Run the Phase 1.1 Alpha158 gate:

```bash
python scripts/run_phase11_alpha158_verification.py
```

Or run the steps manually:

```bash
python scripts/build_etf_qlib_provider.py
python scripts/inspect_qlib_provider.py --feature-set alpha158_etf
python scripts/build_qlib_alpha158_features.py --feature-set alpha158_etf
python scripts/diagnose_alpha158_features.py --feature-set alpha158_etf
```

Outputs:

- `data/qlib_data/a_share_etf_cn_data/`
- `data/features/qlib_alpha158_etf_features.parquet`
- `data/reports/qlib_provider_summary_alpha158_etf.json`
- `data/reports/qlib_instruments_sample_alpha158_etf.csv`
- `data/reports/alpha158_feature_summary_alpha158_etf.csv`
- `data/reports/alpha158_asset_coverage_alpha158_etf.csv`
- `data/reports/alpha158_price_universe_alignment_alpha158_etf.csv`

## Phase 2 Gold Bootstrap

Phase 2 starts by treating gold as an independent research pipeline, separate from the A-share ETF baseline. The first step loads already available local gold, market, FX, and macro CSVs from the old `gold_llm_quant` workspace into the unified daily series schema:

- `date`
- `series_id`
- `value`
- `source`

This bootstrap does not connect News or Polymarket, does not train LightGBM or deep models, does not build regime overlays, and does not perform feature fusion. It only makes the independent gold daily data auditable before later feature work.

Run:

```bash
python scripts/fetch_gold_local_series.py
```

Outputs:

- `data/raw/gold/gold_daily_series.csv`
- `data/reports/gold_daily_series_coverage.csv`

The default gold config reads local files from `../gold_llm_quant`:

- `data/raw/market/market_data.csv`
- `data/raw/market/china_gold_etf.csv`
- `data/raw/market/sge_gold.csv`
- `data/raw/macro/fred_macro.csv`

Build and evaluate the first independent gold model baseline:

```bash
python scripts/process_gold_news_events.py
python scripts/build_gold_features.py
python scripts/train_gold_model.py
python scripts/train_gold_position_model.py
python scripts/backtest_gold_model.py
python scripts/backtest_gold_position_model.py
python scripts/evaluate_gold_model_vs_buy_hold.py
python scripts/run_gold_strategy_sensitivity.py
python scripts/run_gold_regime_diagnostics.py
python scripts/run_gold_layered_strategy.py
python scripts/audit_gold_long_history_readiness.py
```

This model uses purged walk-forward validation and a tabular classifier, preferring LightGBM when installed and falling back through the modeling layer only when needed. News and Polymarket inputs are supported as optional local files under `data/raw/gold/`; if those files are absent, the model runs without those feature groups and reports that they were skipped. `process_gold_news_events.py` reads raw article metadata from `data/raw/gold/news.csv`, sends only title and short text to the configured LLM client, and writes structured events to `data/raw/gold/news_events.csv` without retaining raw article text in the event file. For sparse RSS items, use `--enrich-url-text` to fetch a transient URL excerpt for LLM input only; the event output still stores only `llm_summary`, rationale, and structured fields. Regime diagnostics split the same OOS predictions by calendar subperiod and by a prior-close price regime, so the overlay can be evaluated as a regime-aware exposure controller rather than only as an all-weather return enhancer. The layered strategy combines regime score, model probability, execution overlay, and risk limits into one long-only exposure framework.

Outputs:

- `data/features/gold/gold_model_dataset.csv`
- `data/raw/gold/news_events.csv`
- `data/features/gold/gold_model_features.txt`
- `data/predictions/gold_model_walk_forward_predictions.csv`
- `data/predictions/gold_position_model_walk_forward_predictions.csv`
- `data/reports/gold_model_walk_forward_fold_metrics.csv`
- `data/reports/gold_model_summary.json`
- `data/backtest/gold_model_backtest.csv`
- `data/backtest/gold_position_model_backtest.csv`
- `data/reports/gold_model_backtest_metrics.json`
- `data/reports/gold_model_vs_buy_hold.csv`
- `data/reports/gold_alpha158_selection_stability.csv`
- `data/reports/gold_strategy_sensitivity.csv`
- `data/reports/gold_regime_subperiod_performance.csv`
- `data/reports/gold_price_regime_performance.csv`
- `data/reports/gold_external_feature_status.csv`
- `data/backtest/gold_layered_strategy_backtest.csv`
- `data/reports/gold_layered_strategy_metrics.json`
- `data/reports/gold_long_history_source_audit.csv`
- `data/reports/gold_long_history_series_audit.csv`

For 2008-start research, run the long-history audit first. The current local `../gold_llm_quant` inputs start in 2018 or later, so a true 2008 study requires adding a longer proxy history such as XAUUSD/GLD, macro series, and an explicit target/execution mapping before retraining.

## Tests

```bash
pytest
```

## Phase 1 Verification Checklist

Recommended one-command Phase 1 verification:

```bash
python scripts/run_phase1_verification.py
```

If local A-share ETF data already exists, skip the data fetch step:

```bash
python scripts/run_phase1_verification.py --skip-fetch
```

Qlib Alpha158 validation remains a separate manual step because `provider_uri` depends on local Qlib data preparation.

Run unit tests:

```bash
python -B -m pytest -p no:cacheprovider
```

Fetch A-share/ETF data and verify coverage:

```bash
python scripts/fetch_a_share_data.py
python scripts/check_data_coverage.py
```

Run rule baselines:

```bash
python scripts/run_rule_baselines.py
python scripts/evaluate_rule_baselines.py
```

Diagnose sample attrition:

```bash
python scripts/diagnose_sample_attrition.py
```

Build Qlib Alpha158 features:

```bash
python scripts/build_qlib_alpha158_features.py
```

Common errors:

- efinance is not installed: install it with `python -m pip install efinance`.
- BaoStock is not installed: install it with `python -m pip install baostock`.
- Qlib is not installed: install it with `python -m pip install pyqlib`.
- Qlib `provider_uri` does not exist: prepare Qlib data first or update `config/feature_sets.yaml`.

## Plan
Phase 1:
新项目 hybrid_market_quant
A 股 ETF / 指数
Qlib Alpha158 adapter
rule-based baseline
daily backtest engine

Phase 1.1:
本地 ETF Qlib provider
ETF universe Alpha158 计算与诊断
provider/date coverage verification
不训练模型
不接 News/Polymarket/macro

Phase 2:
迁移旧 gold_llm_quant 数据源
黄金 ETF / XAUUSD / macro / DXY / real yield / VIX
统一 schema 和 feature availability
黄金独立处理，先做本地 CSV/schema/coverage bootstrap
不和 A 股 ETF 混合训练
不接 News/Polymarket
不训练模型

Phase 3:
语义 + 概率特征接口
News daily score
Polymarket snapshot / probability features
不直接训练深度模型

Phase 4:
regime overlay / strategy selection
trend_state
risk_state
macro_tailwind_state
volatility scaling
rule-core ensemble

Phase 5:
ML meta-filter
不直接预测涨跌
预测 rule signal 是否值得执行

Phase 6:
iTransformer / Mamba-2 slow-line analyst
只有当数据历史足够、baseline 稳定后再做
