#!/usr/bin/env python
"""Backtest direct 0-1 gold position model outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.backtest import run_gold_position_backtest


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    pred_path = PROJECT_ROOT / "data" / "predictions" / "gold_position_model_walk_forward_predictions.csv"
    if not pred_path.exists():
        raise SystemExit("Run python scripts/models/train_gold_position_model.py first.")
    predictions = pd.read_csv(pred_path)
    daily, metrics = run_gold_position_backtest(
        predictions,
        transaction_cost_bps=float(config["backtest"].get("transaction_cost_bps", 2.0)),
    )
    backtest_path = PROJECT_ROOT / "data" / "backtest" / "gold_position_model_backtest.csv"
    metrics_path = PROJECT_ROOT / "data" / "reports" / "gold_position_model_backtest_metrics.json"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(backtest_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"saved position model backtest to {backtest_path}")
    print(f"saved position model metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
