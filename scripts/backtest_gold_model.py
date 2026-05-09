#!/usr/bin/env python
"""Backtest the independent gold model signal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.backtest import run_gold_long_flat_backtest


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    pred_path = PROJECT_ROOT / config["output"]["predictions"]
    if not pred_path.exists():
        raise SystemExit("Run python scripts/train_gold_model.py first.")
    predictions = pd.read_csv(pred_path)
    daily, metrics = run_gold_long_flat_backtest(
        predictions,
        threshold=float(config["backtest"].get("signal_threshold", 0.55)),
        transaction_cost_bps=float(config["backtest"].get("transaction_cost_bps", 2.0)),
    )
    backtest_path = PROJECT_ROOT / config["output"]["backtest"]
    metrics_path = PROJECT_ROOT / config["output"]["backtest_metrics"]
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(backtest_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"saved gold model backtest to {backtest_path}")
    print(f"saved gold model metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
