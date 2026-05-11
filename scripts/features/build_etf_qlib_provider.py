#!/usr/bin/env python
"""Build a local Qlib provider from the ETF daily research universe."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.qlib_provider_builder import build_etf_qlib_provider


def main() -> None:
    data_config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["a_share"]
    feature_config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())["alpha158_etf"]
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    if not price_path.exists():
        raise SystemExit("Run python scripts/data/fetch_a_share_data.py first.")

    df = pd.read_csv(price_path)
    summary = build_etf_qlib_provider(df, PROJECT_ROOT / feature_config["provider_uri"])

    print(f"provider_uri: {summary['provider_uri']}")
    print(f"n_assets: {summary['n_assets']}")
    print(f"n_calendar_days: {summary['n_calendar_days']}")
    print(f"first_date: {summary['first_date']}")
    print(f"last_date: {summary['last_date']}")
    print(f"fields: {', '.join(summary['fields'])}")
    print(f"saved provider summary to {summary['provider_uri']}/provider_summary.json")
    print(f"ETF universe source: {data_config['symbols']}")


if __name__ == "__main__":
    main()
