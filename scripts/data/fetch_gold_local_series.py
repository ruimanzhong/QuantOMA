#!/usr/bin/env python
"""Load independent gold market and macro CSVs into the unified daily series schema."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.phase2_local_data import load_gold_local_series
from market_quant.diagnostics.series_coverage import summarize_daily_series_coverage


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    if config.get("provider") != "local_csv":
        raise SystemExit("Gold bootstrap currently supports provider='local_csv' only.")

    series = load_gold_local_series(config, PROJECT_ROOT)
    output = PROJECT_ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output, index=False)

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage = summarize_daily_series_coverage(series)
    coverage_path = report_dir / "gold_daily_series_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    print(f"saved gold daily series to {output}")
    print(f"saved gold coverage report to {coverage_path}")
    print(f"n rows: {len(series)}")
    print(f"n series: {series['series_id'].nunique()}")
    print(f"date range: {series['date'].min()} to {series['date'].max()}")
    print("\nCoverage sample:")
    print(coverage.sort_values(["coverage_ok", "series_id"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
