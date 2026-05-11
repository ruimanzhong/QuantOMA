#!/usr/bin/env python
"""Compare the gold model backtest with buy-and-hold on the same OOS window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.backtest.metrics import summarize_backtest


def main() -> None:
    pred_path = PROJECT_ROOT / "data" / "predictions" / "gold_model_walk_forward_predictions.csv"
    model_metrics_path = PROJECT_ROOT / "data" / "reports" / "gold_model_backtest_metrics.json"
    selected_path = PROJECT_ROOT / "data" / "reports" / "gold_alpha158_selected_features.csv"
    if not pred_path.exists() or not model_metrics_path.exists():
        raise SystemExit("Run python scripts/models/train_gold_model.py and python scripts/backtests/backtest_gold_model.py first.")
    pred = pd.read_csv(pred_path, parse_dates=["date"]).sort_values("date")
    buy_hold_daily = pred.copy()
    buy_hold_daily["asset_return"] = buy_hold_daily["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    buy_hold_daily["position"] = 1.0
    buy_hold_daily["executed_position"] = 1.0
    buy_hold_daily["strategy_return"] = buy_hold_daily["asset_return"]
    buy_hold_metrics = summarize_backtest(buy_hold_daily.assign(asset_id="buy_and_hold"))
    model_metrics = json.loads(model_metrics_path.read_text())

    rows = [
        {"strategy_name": "gold_model", **model_metrics},
        {"strategy_name": "buy_and_hold", **buy_hold_metrics},
    ]
    comparison = pd.DataFrame(rows)
    comparison_path = PROJECT_ROOT / "data" / "reports" / "gold_model_vs_buy_hold.csv"
    comparison.to_csv(comparison_path, index=False)

    if selected_path.exists() and selected_path.stat().st_size > 0:
        try:
            selected = pd.read_csv(selected_path)
        except pd.errors.EmptyDataError:
            selected = pd.DataFrame()
        if not selected.empty and {"selected", "feature", "fold_id", "abs_ic", "ic"}.issubset(selected.columns):
            stable = (
                selected[selected["selected"]]
                .groupby("feature")
                .agg(n_folds=("fold_id", "nunique"), avg_abs_ic=("abs_ic", "mean"), avg_ic=("ic", "mean"))
                .sort_values(["n_folds", "avg_abs_ic"], ascending=[False, False])
                .reset_index()
            )
            stable_path = PROJECT_ROOT / "data" / "reports" / "gold_alpha158_selection_stability.csv"
            stable.to_csv(stable_path, index=False)
            print("Top selected Alpha158 indicators:")
            print(stable.head(20).to_string(index=False))
            print(f"saved selection stability to {stable_path}")

    model_return = float(model_metrics["cumulative_return"])
    buy_hold_return = float(buy_hold_metrics["cumulative_return"])
    model_sharpe = float(model_metrics["sharpe"])
    buy_hold_sharpe = float(buy_hold_metrics["sharpe"])
    print("\nGold model vs buy-and-hold:")
    print(comparison[["strategy_name", "cumulative_return", "sharpe", "max_drawdown", "active_ratio", "turnover"]].to_string(index=False))
    print(f"\nbeats buy-and-hold return: {model_return > buy_hold_return}")
    print(f"beats buy-and-hold sharpe: {model_sharpe > buy_hold_sharpe}")
    print(f"saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
