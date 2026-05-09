#!/usr/bin/env python
"""Train a gold model that directly predicts 0-1 target exposure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.models.walk_forward import run_walk_forward_regressor


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    dataset_path = PROJECT_ROOT / config["output"]["dataset"]
    feature_path = PROJECT_ROOT / config["output"]["feature_columns"]
    if not dataset_path.exists() or not feature_path.exists():
        raise SystemExit("Run python scripts/build_gold_features.py first.")

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])
    df = df.set_index("date").sort_index()
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    result = run_walk_forward_regressor(
        df,
        feature_columns=features,
        target_column="target_position",
        model_name=config["model"].get("name", "lightgbm"),
        model_parameters=config["model"].get("parameters", {}),
        fold_config=config.get("walk_forward", {}),
    )

    pred_path = PROJECT_ROOT / "data" / "predictions" / "gold_position_model_walk_forward_predictions.csv"
    metrics_path = PROJECT_ROOT / "data" / "reports" / "gold_position_model_walk_forward_fold_metrics.csv"
    summary_path = PROJECT_ROOT / "data" / "reports" / "gold_position_model_summary.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(pred_path, index=False)
    result.fold_metrics.to_csv(metrics_path, index=False)
    summary = {
        "requested_model": config["model"].get("name", "lightgbm"),
        "model_name_used": result.model_name_used,
        "target_column": "target_position",
        "target_price_time": config.get("features", {}).get("target_price_time", "daily_close_proxy"),
        "n_predictions": int(len(result.predictions)),
        "n_folds": int(result.fold_metrics["fold_id"].nunique()),
        "mean_mae": float(result.fold_metrics["mae"].mean()),
        "mean_rmse": float(result.fold_metrics["rmse"].mean()),
        "mean_predicted_position": float(result.fold_metrics["predicted_mean_position"].mean()),
        "mean_target_position": float(result.fold_metrics["target_mean_position"].mean()),
        "mean_n_features": float(result.fold_metrics["n_features"].mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved position predictions to {pred_path}")
    print(f"saved fold metrics to {metrics_path}")
    print(f"saved model summary to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
