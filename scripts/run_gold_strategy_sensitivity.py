#!/usr/bin/env python
"""Evaluate gold execution strategy variants from the same OOS probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.backtest.metrics import summarize_backtest
from market_quant.gold.backtest import run_gold_long_flat_backtest


def _buy_hold_metrics(predictions: pd.DataFrame) -> dict:
    daily = predictions.copy().sort_values("date")
    daily["asset_return"] = daily["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    daily["position"] = 1.0
    daily["executed_position"] = 1.0
    daily["strategy_return"] = daily["asset_return"]
    metrics = summarize_backtest(daily.assign(asset_id="buy_and_hold"))
    metrics["strategy_name"] = "buy_and_hold"
    return metrics


def main() -> None:
    pred_path = PROJECT_ROOT / "data" / "predictions" / "gold_model_walk_forward_predictions.csv"
    if not pred_path.exists():
        raise SystemExit("Run python scripts/train_gold_model.py first.")
    predictions = pd.read_csv(pred_path, parse_dates=["date"])
    variants = []

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
        variants.append({"strategy_name": f"threshold_{threshold:.2f}", "strategy_type": "threshold", "threshold": threshold})
    for threshold in [0.55, 0.60]:
        for smoothing_window in [3, 5]:
            variants.append(
                {
                    "strategy_name": f"threshold_{threshold:.2f}_smooth_{smoothing_window}d",
                    "strategy_type": "threshold",
                    "threshold": threshold,
                    "smoothing_window": smoothing_window,
                }
            )
    for threshold in [0.55, 0.60]:
        for min_holding_days in [3, 5, 10]:
            variants.append(
                {
                    "strategy_name": f"threshold_{threshold:.2f}_hold_{min_holding_days}d",
                    "strategy_type": "threshold",
                    "threshold": threshold,
                    "min_holding_days": min_holding_days,
                }
            )
    for lower, upper in [(0.40, 0.60), (0.45, 0.60), (0.45, 0.65)]:
        variants.append(
            {
                "strategy_name": f"prob_scaled_{lower:.2f}_{upper:.2f}",
                "strategy_type": "probability_scaled",
                "lower_threshold": lower,
                "threshold": upper,
            }
        )
    for lower in [0.35, 0.40, 0.45, 0.50]:
        for defensive_position in [0.2, 0.3, 0.5]:
            variants.append(
                {
                    "strategy_name": f"overlay_low_{lower:.2f}_pos_{defensive_position:.1f}",
                    "strategy_type": "defensive_overlay",
                    "lower_threshold": lower,
                    "defensive_position": defensive_position,
                }
            )

    rows = [_buy_hold_metrics(predictions)]
    for variant in variants:
        _, metrics = run_gold_long_flat_backtest(
            predictions,
            threshold=float(variant.get("threshold", 0.55)),
            lower_threshold=float(variant.get("lower_threshold", 0.45)),
            defensive_position=float(variant.get("defensive_position", 0.3)),
            smoothing_window=int(variant.get("smoothing_window", 1)),
            min_holding_days=int(variant.get("min_holding_days", 1)),
            strategy_type=variant["strategy_type"],
            transaction_cost_bps=2.0,
        )
        rows.append({"strategy_name": variant["strategy_name"], **metrics})

    report = pd.DataFrame(rows)
    buy_hold = report[report["strategy_name"] == "buy_and_hold"].iloc[0]
    report["excess_return_vs_buy_hold"] = report["cumulative_return"] - buy_hold["cumulative_return"]
    report["sharpe_diff_vs_buy_hold"] = report["sharpe"] - buy_hold["sharpe"]
    report["drawdown_reduction_vs_buy_hold"] = report["max_drawdown"] - buy_hold["max_drawdown"]
    output = PROJECT_ROOT / "data" / "reports" / "gold_strategy_sensitivity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    print("Top strategies by Sharpe:")
    print(
        report.sort_values("sharpe", ascending=False)
        [["strategy_name", "cumulative_return", "sharpe", "max_drawdown", "active_ratio", "turnover", "drawdown_reduction_vs_buy_hold"]]
        .head(15)
        .to_string(index=False)
    )
    print("\nTop strategies by drawdown reduction:")
    print(
        report.sort_values("drawdown_reduction_vs_buy_hold", ascending=False)
        [["strategy_name", "cumulative_return", "sharpe", "max_drawdown", "active_ratio", "turnover", "drawdown_reduction_vs_buy_hold"]]
        .head(15)
        .to_string(index=False)
    )
    print(f"\nsaved strategy sensitivity to {output}")


if __name__ == "__main__":
    main()
