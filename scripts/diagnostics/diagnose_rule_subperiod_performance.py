#!/usr/bin/env python
"""Diagnose subperiod Phase 1 rule baseline performance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.rule_robustness import subperiod_strategy_performance


FOCUS = ["momentum_20d", "donchian_breakout_20d", "buy_and_hold"]


def main() -> None:
    path = PROJECT_ROOT / "data" / "backtest" / "rule_baselines_daily.csv"
    if not path.exists():
        raise SystemExit("Run python scripts/strategies/run_rule_baselines.py first.")
    report = subperiod_strategy_performance(pd.read_csv(path))
    out = PROJECT_ROOT / "data" / "reports" / "rule_subperiod_performance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    focus = report[report["strategy_name"].isin(FOCUS)]
    pivot = focus.groupby(["strategy_name", "subperiod"]).agg(
        avg_return=("cumulative_return", "mean"),
        avg_sharpe=("sharpe", "mean"),
        positive_assets=("cumulative_return", lambda s: int((s > 0).sum())),
        n_assets=("asset_id", "nunique"),
    )
    print("Focused subperiod performance:")
    print(pivot.reset_index().to_string(index=False))

    summary = report.groupby("strategy_name").agg(
        n_subperiods=("subperiod", "count"),
        positive_subperiods=("cumulative_return", lambda s: int((s > 0).sum())),
        avg_sharpe=("sharpe", "mean"),
    ).reset_index()
    summary["positive_ratio"] = summary["positive_subperiods"] / summary["n_subperiods"]
    stable = summary.sort_values(["positive_ratio", "avg_sharpe"], ascending=[False, False])
    single_period = summary[summary["positive_subperiods"] <= 1]

    print("\nMost stable across subperiods:")
    print(stable.head(10).to_string(index=False))
    print("\nStrategies only effective in one or zero subperiod observations:")
    print(single_period.to_string(index=False) if not single_period.empty else "none")
    print(f"\nsaved subperiod rule performance to {out}")


if __name__ == "__main__":
    main()
