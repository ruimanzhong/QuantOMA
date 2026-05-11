#!/usr/bin/env python
"""Diagnose generated Qlib Alpha158 feature coverage."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.alpha158_diagnostics import (
    alpha158_asset_coverage,
    compare_alpha158_with_price_universe,
    summarize_alpha158_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose generated Qlib Alpha158 feature coverage.")
    parser.add_argument("--feature-set", default="alpha158", help="feature_sets.yaml key to diagnose")
    return parser.parse_args()


def _configured_output_path(feature_set: str) -> Path:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())[feature_set]
    output = Path(config["output"])
    return output if output.is_absolute() else PROJECT_ROOT / output


def _load_alpha_features(feature_set: str) -> pd.DataFrame:
    configured_path = _configured_output_path(feature_set)
    parquet_path = configured_path if configured_path.suffix == ".parquet" else configured_path.with_suffix(".parquet")
    csv_path = configured_path if configured_path.suffix == ".csv" else configured_path.with_suffix(".csv")
    fallback_csv_path = configured_path.with_name("qlib_alpha158_features.csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            if not csv_path.exists():
                raise RuntimeError(f"failed to read {parquet_path}: {exc}") from exc
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if fallback_csv_path.exists():
        return pd.read_csv(fallback_csv_path)
    raise SystemExit(f"Run python scripts/features/build_qlib_alpha158_features.py --feature-set {feature_set} first.")


def _load_price_universe(feature_set: str) -> pd.DataFrame | None:
    if feature_set == "alpha158_gold":
        path = PROJECT_ROOT / "../gold_llm_quant/data/raw/market/china_gold_etf.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df.rename(columns={"symbol": "asset_id"})[["date", "asset_id", "close"]]
    path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main() -> None:
    args = parse_args()
    alpha = _load_alpha_features(args.feature_set)
    summary = summarize_alpha158_features(alpha)
    coverage = alpha158_asset_coverage(alpha)

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.feature_set == "alpha158" else f"_{args.feature_set}"
    summary_path = report_dir / f"alpha158_feature_summary{suffix}.csv"
    coverage_path = report_dir / f"alpha158_asset_coverage{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print("Alpha158 summary:")
    print(summary.to_string(index=False))
    print("\nAlpha158 asset coverage sample:")
    print(coverage.head(20).to_string(index=False))

    price = _load_price_universe(args.feature_set)
    if price is not None:
        alignment = compare_alpha158_with_price_universe(alpha, price)
        alignment_path = report_dir / f"alpha158_price_universe_alignment{suffix}.csv"
        alignment.to_csv(alignment_path, index=False)
        print("\nAlpha158 vs price universe alignment:")
        price_rows = alignment[alignment["in_price_data"]].copy()
        alpha_only_sample = alignment[~alignment["in_price_data"] & alignment["in_alpha158"]].head(20)
        print(price_rows.to_string(index=False))
        print("\nAlpha-only instrument sample:")
        print(alpha_only_sample.to_string(index=False))
        print(f"saved alignment report to {alignment_path}")

    print(f"saved feature summary to {summary_path}")
    print(f"saved asset coverage to {coverage_path}")


if __name__ == "__main__":
    main()
