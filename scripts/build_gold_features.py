#!/usr/bin/env python
"""Build the independent gold model dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.features import build_gold_dataset, load_alpha158_features


def _read_optional(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    target = PROJECT_ROOT / path
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    series_path = PROJECT_ROOT / config["input"]["daily_series"]
    if not series_path.exists():
        raise SystemExit("Run python scripts/fetch_gold_local_series.py first.")
    series = pd.read_csv(series_path)
    news = _read_optional(config.get("input", {}).get("news_events"))
    polymarket = _read_optional(config.get("input", {}).get("polymarket"))
    feature_config = config.get("features", {})
    alpha_path = config.get("input", {}).get("alpha158")
    alpha = pd.DataFrame()
    if alpha_path:
        try:
            alpha = load_alpha158_features(PROJECT_ROOT / alpha_path, feature_config.get("primary_alpha158_asset_id", "518880"))
        except ImportError as exc:
            print(f"Alpha158 features: skipped because parquet support is unavailable: {exc}")
    bundle = build_gold_dataset(
        series,
        feature_config,
        news_events=news,
        polymarket_data=polymarket,
        alpha158_features=alpha,
    )

    output = PROJECT_ROOT / config["output"]["dataset"]
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.data.reset_index(names="date").to_csv(output, index=False)

    feature_path = PROJECT_ROOT / config["output"]["feature_columns"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text("\n".join(bundle.feature_columns) + "\n")

    print(f"saved gold dataset to {output}")
    print(f"saved feature columns to {feature_path}")
    print(f"n rows: {len(bundle.data)}")
    print(f"n features: {len(bundle.feature_columns)}")
    print(f"date range: {bundle.data.index.min()} to {bundle.data.index.max()}")
    print(f"target positive rate: {bundle.data[bundle.target_column].mean():.4f}")
    if news.empty:
        print("news features: no local news event file found; skipped")
    if polymarket.empty:
        print("polymarket features: no local polymarket file found; skipped")
    if alpha.empty:
        print("Alpha158 features: no local gold Alpha158 file found; skipped")
    else:
        print(f"Alpha158 features: included {alpha.shape[1]} columns for {feature_config.get('primary_alpha158_asset_id', '518880')}")


if __name__ == "__main__":
    main()
