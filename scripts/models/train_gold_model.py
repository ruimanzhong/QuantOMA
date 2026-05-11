#!/usr/bin/env python
"""Train the independent gold model with purged walk-forward validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.models.walk_forward import run_walk_forward_classifier


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    dataset_path = PROJECT_ROOT / config["output"]["dataset"]
    feature_path = PROJECT_ROOT / config["output"]["feature_columns"]
    if not dataset_path.exists() or not feature_path.exists():
        raise SystemExit("Run python scripts/features/build_gold_features.py first.")

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])
    df = df.set_index("date").sort_index()
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    result = run_walk_forward_classifier(
        df,
        feature_columns=features,
        target_column="target_direction",
        model_name=config["model"].get("name", "lightgbm"),
        model_parameters=config["model"].get("parameters", {}),
        fold_config=config.get("walk_forward", {}),
        feature_selection_config=config.get("feature_selection", {}),
    )

    pred_path = PROJECT_ROOT / config["output"]["predictions"]
    metrics_path = PROJECT_ROOT / config["output"]["fold_metrics"]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(pred_path, index=False)
    result.fold_metrics.to_csv(metrics_path, index=False)
    selected_path = PROJECT_ROOT / config["output"].get("selected_features", "data/reports/gold_alpha158_selected_features.csv")
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    result.selected_features.to_csv(selected_path, index=False)

    summary = {
        "requested_model": config["model"].get("name", "lightgbm"),
        "model_name_used": result.model_name_used,
        "n_predictions": int(len(result.predictions)),
        "n_folds": int(result.fold_metrics["fold_id"].nunique()),
        "mean_accuracy": float(result.fold_metrics["accuracy"].mean()),
        "mean_brier_score": float(result.fold_metrics["brier_score"].mean()) if "brier_score" in result.fold_metrics else None,
        "mean_log_loss": float(result.fold_metrics["log_loss"].mean()) if "log_loss" in result.fold_metrics else None,
        "mean_raw_brier_score": float(result.fold_metrics["raw_brier_score"].mean()) if "raw_brier_score" in result.fold_metrics else None,
        "mean_raw_log_loss": float(result.fold_metrics["raw_log_loss"].mean()) if "raw_log_loss" in result.fold_metrics else None,
        "mean_positive_rate": float(result.fold_metrics["positive_rate"].mean()),
        "mean_n_features": float(result.fold_metrics["n_features"].mean()),
        "mean_n_alpha158_features": float(result.fold_metrics["n_alpha158_features"].mean()),
        "calibration_method": config.get("walk_forward", {}).get("calibration", {}).get("method", "none"),
    }
    summary_path = PROJECT_ROOT / config["output"]["model_summary"]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"saved predictions to {pred_path}")
    print(f"saved fold metrics to {metrics_path}")
    print(f"saved selected feature report to {selected_path}")
    print(f"saved model summary to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
