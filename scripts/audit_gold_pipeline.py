#!/usr/bin/env python
"""Audit gold model data alignment, trading-day target logic, and leakage boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.gold_pipeline_audit import audit_gold_dataset, audit_walk_forward_folds


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    dataset_path = PROJECT_ROOT / config["output"]["dataset"]
    feature_path = PROJECT_ROOT / config["output"]["feature_columns"]
    fold_path = PROJECT_ROOT / config["output"]["fold_metrics"]
    if not dataset_path.exists() or not feature_path.exists():
        raise SystemExit("Run python scripts/build_gold_features.py first.")

    dataset = pd.read_csv(dataset_path)
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    audit = audit_gold_dataset(dataset, int(config["features"]["horizon_days"]), features)

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_audit_path = report_dir / "gold_pipeline_dataset_audit.csv"
    audit.to_csv(dataset_audit_path, index=False)
    print("Gold dataset audit:")
    print(audit.to_string(index=False))
    print(f"saved dataset audit to {dataset_audit_path}")

    if fold_path.exists():
        fold_audit = audit_walk_forward_folds(dataset, pd.read_csv(fold_path))
        fold_audit_path = report_dir / "gold_pipeline_fold_audit.csv"
        fold_audit.to_csv(fold_audit_path, index=False)
        print("\nGold fold audit:")
        print(fold_audit.to_string(index=False))
        print(f"saved fold audit to {fold_audit_path}")
        if not fold_audit["passed"].all():
            raise SystemExit("Gold fold audit failed.")

    if not audit["passed"].all():
        raise SystemExit("Gold dataset audit failed.")


if __name__ == "__main__":
    main()
