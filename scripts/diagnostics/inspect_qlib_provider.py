#!/usr/bin/env python
"""Inspect configured Qlib provider and instrument universe."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.qlib_provider_diagnostics import inspect_qlib_provider, list_qlib_instruments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a configured Qlib provider and instrument universe.")
    parser.add_argument("--feature-set", default="alpha158", help="feature_sets.yaml key to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())[args.feature_set]
    provider_uri = config["provider_uri"]
    summary = inspect_qlib_provider(provider_uri)
    instruments = list_qlib_instruments(
        provider_uri=provider_uri,
        instruments=config.get("instruments", "all"),
        start_time=config["start_time"],
        end_time=config["end_time"],
    )

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.feature_set == "alpha158" else f"_{args.feature_set}"
    summary_path = report_dir / f"qlib_provider_summary{suffix}.json"
    sample_path = report_dir / f"qlib_instruments_sample{suffix}.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    instruments.head(200).to_csv(sample_path, index=False)

    print(f"provider_uri: {summary['provider_uri']}")
    print(f"provider_uri exists: {summary['provider_uri_exists']}")
    print(f"calendar range: {summary.get('first_calendar_date')} to {summary.get('last_calendar_date')}")
    print("common universe instrument counts:")
    for name, info in summary["universes"].items():
        print(f"  {name}: count={info.get('count')} error={info.get('error')}")
    print("sample instruments:")
    print(instruments.head(20).to_string(index=False))
    print(f"saved provider summary to {summary_path}")
    print(f"saved instrument sample to {sample_path}")


if __name__ == "__main__":
    main()
