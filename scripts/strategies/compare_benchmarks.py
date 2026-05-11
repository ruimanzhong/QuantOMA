#!/usr/bin/env python
"""Assemble the Phase 1 benchmark comparison report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_required(path: Path, command: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run {command} first.")
    return pd.read_csv(path)


def main() -> None:
    report_dir = PROJECT_ROOT / "data" / "reports"
    comparison = _read_required(
        report_dir / "rule_baseline_comparison.csv",
        "python scripts/strategies/run_rule_baselines.py",
    )
    vs_buy_hold = _read_required(
        report_dir / "rule_vs_buy_hold_comparison.csv",
        "python scripts/strategies/evaluate_rule_baselines.py",
    )
    best = _read_required(
        report_dir / "best_strategy_by_asset.csv",
        "python scripts/strategies/evaluate_rule_baselines.py",
    )
    stability = _read_required(
        report_dir / "strategy_stability_summary.csv",
        "python scripts/strategies/evaluate_rule_baselines.py",
    )

    ranked = comparison.sort_values(["asset_id", "sharpe"], ascending=[True, False])
    output = report_dir / "phase1_benchmark_comparison.csv"
    ranked.to_csv(output, index=False)

    print("Best strategy by asset:")
    print(best.to_string(index=False))
    print("\nTop 10 strategies by cross-asset stability:")
    print(stability.head(10).to_string(index=False))
    print("\nStrategies beating buy-and-hold Sharpe by asset:")
    print(
        vs_buy_hold[vs_buy_hold["beats_buy_hold_sharpe"]]
        .sort_values(["asset_id", "sharpe_diff_vs_buy_hold"], ascending=[True, False])
        [["asset_id", "strategy_name", "sharpe_diff_vs_buy_hold", "excess_return_vs_buy_hold"]]
        .to_string(index=False)
    )
    print(f"\nsaved benchmark comparison to {output}")


if __name__ == "__main__":
    main()
