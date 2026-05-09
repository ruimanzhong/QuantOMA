#!/usr/bin/env python
"""Evaluate Phase 1 rule baselines against buy-and-hold."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.rule_baseline_evaluation import (
    best_strategy_by_asset,
    evaluate_rule_vs_buy_hold,
    strategy_stability_summary,
)


def _ensure_coverage_ok() -> None:
    coverage_path = PROJECT_ROOT / "data" / "reports" / "a_share_data_coverage.csv"
    if not coverage_path.exists():
        return
    coverage = pd.read_csv(coverage_path)
    if "coverage_ok" in coverage.columns and not coverage["coverage_ok"].astype(bool).all():
        failed = ", ".join(coverage.loc[~coverage["coverage_ok"].astype(bool), "asset_id"].astype(str))
        raise SystemExit(f"data coverage check failed for: {failed}. Run python scripts/check_data_coverage.py first.")


def main() -> None:
    comparison_path = PROJECT_ROOT / "data" / "reports" / "rule_baseline_comparison.csv"
    if not comparison_path.exists():
        raise SystemExit("Run python scripts/run_rule_baselines.py first.")
    _ensure_coverage_ok()

    comparison_df = pd.read_csv(comparison_path)
    rule_vs_buy_hold = evaluate_rule_vs_buy_hold(comparison_df)
    best_by_asset = best_strategy_by_asset(comparison_df)
    stability = strategy_stability_summary(comparison_df)

    report_dir = PROJECT_ROOT / "data" / "reports"
    out_vs = report_dir / "rule_vs_buy_hold_comparison.csv"
    out_best = report_dir / "best_strategy_by_asset.csv"
    out_stability = report_dir / "strategy_stability_summary.csv"
    rule_vs_buy_hold.to_csv(out_vs, index=False)
    best_by_asset.to_csv(out_best, index=False)
    stability.to_csv(out_stability, index=False)

    print("Best by Sharpe:")
    print(best_by_asset[["asset_id", "best_by_sharpe", "best_by_sharpe_value"]].to_string(index=False))
    print("\nBest by Return:")
    print(best_by_asset[["asset_id", "best_by_return", "best_by_return_value"]].to_string(index=False))
    print("\nStability score top 5:")
    print(stability.head(5).to_string(index=False))

    majority = stability["n_assets"] / 2
    lowers_drawdown = stability[stability["n_assets_reducing_drawdown"] > majority]
    print("\nStrategies reducing drawdown in most assets:")
    if lowers_drawdown.empty:
        print("none")
    else:
        print(lowers_drawdown[["strategy_name", "n_assets_reducing_drawdown", "n_assets"]].to_string(index=False))

    exposure_only = rule_vs_buy_hold[
        (rule_vs_buy_hold["active_ratio"] < 1.0)
        & ~rule_vs_buy_hold["beats_buy_hold_return"]
        & ~rule_vs_buy_hold["beats_buy_hold_sharpe"]
    ]
    exposure_only_summary = (
        exposure_only.groupby("strategy_name")
        .agg(n_assets=("asset_id", "nunique"), avg_active_ratio=("active_ratio", "mean"))
        .reset_index()
        .sort_values(["n_assets", "avg_active_ratio"], ascending=[False, True])
    )
    print("\nStrategies lowering exposure without beating buy-and-hold:")
    if exposure_only_summary.empty:
        print("none")
    else:
        print(exposure_only_summary.to_string(index=False))

    print(f"\nsaved rule vs buy-hold comparison to {out_vs}")
    print(f"saved best strategy by asset to {out_best}")
    print(f"saved strategy stability summary to {out_stability}")


if __name__ == "__main__":
    main()
