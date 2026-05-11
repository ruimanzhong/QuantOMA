#!/usr/bin/env python
"""Check A-share ETF data date coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.schema import check_date_coverage


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["a_share"]
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    df = pd.read_csv(price_path)
    report = check_date_coverage(df, config["start_date"], config["end_date"])

    out = PROJECT_ROOT / "data" / "reports" / "a_share_data_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"saved A-share data coverage report to {out}")

    if not report["coverage_ok"].all():
        failed = ", ".join(report.loc[~report["coverage_ok"], "asset_id"].astype(str))
        raise SystemExit(f"data coverage check failed for: {failed}")


if __name__ == "__main__":
    main()
