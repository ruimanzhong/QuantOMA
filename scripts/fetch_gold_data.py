#!/usr/bin/env python
"""Fetch and assemble gold research data inside the current project."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.env import load_env_files
from market_quant.data.gold_china_data import fetch_china_gold_etfs, fetch_sge_gold
from market_quant.data.gold_macro_data import fetch_gold_macro_data
from market_quant.data.gold_market_data import fetch_gold_market_data
from market_quant.data.phase2_local_data import load_gold_local_series
from market_quant.diagnostics.series_coverage import summarize_daily_series_coverage


def parse_args() -> object:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-china", action="store_true")
    parser.add_argument("--skip-sge", action="store_true")
    return parser.parse_args()


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {len(df)} rows to {path}")
    if "date" in df and not df.empty:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"date range: {dates.min()} to {dates.max()}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    load_env_files(PROJECT_ROOT)
    raw_outputs = config.get("raw_outputs", {})

    if not args.skip_market:
        market = fetch_gold_market_data(config["market_data"])
        _write(market, PROJECT_ROOT / raw_outputs.get("market_data", "data/raw/gold/market_data.csv"))

    if not args.skip_macro:
        macro = fetch_gold_macro_data(config["macro_data"])
        _write(macro, PROJECT_ROOT / raw_outputs.get("macro", "data/raw/gold/fred_macro.csv"))

    if not args.skip_china:
        try:
            etf = fetch_china_gold_etfs(config["china_gold_data"])
            _write(etf, PROJECT_ROOT / raw_outputs.get("china_gold_etf", "data/raw/gold/china_gold_etf.csv"))
        except Exception as exc:
            print(f"warning: China gold ETF fetch failed: {exc}")

    if not args.skip_sge:
        try:
            sge = fetch_sge_gold(config["china_gold_data"])
            _write(sge, PROJECT_ROOT / raw_outputs.get("sge_gold", "data/raw/gold/sge_gold.csv"))
        except Exception as exc:
            print(f"warning: SGE gold fetch failed: {exc}")

    series = load_gold_local_series({**config, "source_root": "."}, PROJECT_ROOT)
    output = PROJECT_ROOT / config.get("output", "data/raw/gold/gold_daily_series.csv")
    _write(series, output)

    coverage = summarize_daily_series_coverage(series)
    coverage_path = PROJECT_ROOT / "data" / "reports" / "gold_daily_series_coverage.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)
    print(f"saved coverage report to {coverage_path}")


if __name__ == "__main__":
    main()
