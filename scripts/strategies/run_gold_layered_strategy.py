#!/usr/bin/env python
"""Run the layered gold regime/signal/strategy/risk exposure framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.layered import run_layered_gold_backtest


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    pred_path = PROJECT_ROOT / config["output"]["predictions"]
    feature_path = PROJECT_ROOT / config["output"]["dataset"]
    if not pred_path.exists() or not feature_path.exists():
        raise SystemExit("Run python scripts/features/build_gold_features.py and python scripts/models/train_gold_model.py first.")

    predictions = pd.read_csv(pred_path, parse_dates=["date"])
    features = pd.read_csv(feature_path, parse_dates=["date"])
    daily, metrics = run_layered_gold_backtest(
        predictions,
        features,
        transaction_cost_bps=float(config["backtest"].get("transaction_cost_bps", 2.0)),
    )

    backtest_path = PROJECT_ROOT / "data" / "backtest" / "gold_layered_strategy_backtest.csv"
    metrics_path = PROJECT_ROOT / "data" / "reports" / "gold_layered_strategy_metrics.json"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(backtest_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))

    print("Layered gold strategy metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"\nsaved layered backtest to {backtest_path}")
    print(f"saved layered metrics to {metrics_path}")


if __name__ == "__main__":
    main()
