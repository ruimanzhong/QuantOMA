#!/usr/bin/env python
"""Fetch Phase 1 A-share/ETF daily data from BaoStock."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.baostock_data import fetch_baostock_daily
from market_quant.data.efinance_data import fetch_efinance_daily
from market_quant.data.schema import check_date_coverage


def _fetch_with_provider(provider: str, config: dict):
    if provider == "efinance":
        if config.get("adjust"):
            print("warning: provider=efinance ignores adjust; efinance data口径需后续核验")
        return fetch_efinance_daily(
            symbols=config["symbols"],
            start_date=config["start_date"],
            end_date=config["end_date"],
        )
    if provider == "baostock":
        return fetch_baostock_daily(
            symbols=config["symbols"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            adjust=config.get("adjust", "qfq"),
        )
    raise ValueError(f"unsupported A-share data provider: {provider}")


def fetch_configured_data(config: dict):
    provider = config.get("provider", "baostock")
    fallback_provider = config.get("fallback_provider")
    try:
        return _fetch_with_provider(provider, config)
    except Exception as exc:
        if not fallback_provider:
            raise
        print(f"warning: primary provider {provider!r} failed: {exc}; trying fallback {fallback_provider!r}")
        return _fetch_with_provider(fallback_provider, config)


def print_coverage_summary(df, config: dict, coverage=None) -> None:
    coverage = coverage if coverage is not None else check_date_coverage(df, config["start_date"], config["end_date"])
    for row in coverage.itertuples(index=False):
        print(
            "asset_id={asset_id} min_date={actual_start_date:%Y-%m-%d} "
            "max_date={actual_end_date:%Y-%m-%d} row_count={n_rows} "
            "expected_start_date={requested_start_date:%Y-%m-%d} "
            "actual_start_gap_days={start_gap_days} coverage_ok={coverage_ok}".format(
                asset_id=row.asset_id,
                actual_start_date=row.actual_start_date,
                actual_end_date=row.actual_end_date,
                n_rows=row.n_rows,
                requested_start_date=row.requested_start_date,
                start_gap_days=row.start_gap_days,
                coverage_ok=row.coverage_ok,
            )
        )


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    config = yaml.safe_load(config_path.read_text())["a_share"]
    df = fetch_configured_data(config)
    coverage = check_date_coverage(df, config["start_date"], config["end_date"])
    print_coverage_summary(df, config, coverage)
    if not coverage["coverage_ok"].all():
        failed = ", ".join(coverage.loc[~coverage["coverage_ok"], "asset_id"].astype(str))
        raise SystemExit(f"refusing to write data with failed coverage for: {failed}")

    output = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
