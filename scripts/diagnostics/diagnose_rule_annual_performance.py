#!/usr/bin/env python
"""Diagnose annual Phase 1 rule baseline performance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.rule_robustness import annual_strategy_performance


def main() -> None:
    path = PROJECT_ROOT / "data" / "backtest" / "rule_baselines_daily.csv"
    if not path.exists():
        raise SystemExit("Run python scripts/strategies/run_rule_baselines.py first.")
    report = annual_strategy_performance(pd.read_csv(path))
    out = PROJECT_ROOT / "data" / "reports" / "rule_annual_performance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    summary = (
        report.groupby("strategy_name")
        .agg(
            n_years=("year", "count"),
            n_positive_years=("cumulative_return", lambda s: int((s > 0).sum())),
            avg_annual_sharpe=("sharpe", "mean"),
        )
        .reset_index()
        .sort_values(["n_positive_years", "avg_annual_sharpe"], ascending=[False, False])
    )
    print(summary.to_string(index=False))
    majority = summary[summary["n_positive_years"] > summary["n_years"] / 2]
    print("\nStrategies positive in most asset-years:")
    print(majority[["strategy_name", "n_positive_years", "n_years"]].to_string(index=False))
    print(f"\nsaved annual rule performance to {out}")


if __name__ == "__main__":
    main()
