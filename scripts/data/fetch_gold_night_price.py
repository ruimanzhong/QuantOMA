#!/usr/bin/env python
"""Fetch AU9999 night-session quotes and store the 23:00 decision price."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.env import load_env_files
from market_quant.data.gold_night_data import fetch_and_store_night_decision_price


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-time", default=None, help="Override configured local night-session decision time, e.g. 23:00.")
    parser.add_argument("--tolerance-minutes", type=int, default=None, help="Override configured maximum distance from decision time.")
    parser.add_argument("--intraday-output", default=None)
    parser.add_argument("--decision-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_files(PROJECT_ROOT)
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    china_config = config.get("china_gold_data", {})
    night_config = {**config.get("night_session", {})}
    night_config["network"] = china_config.get("network")
    decision_time = args.decision_time or night_config.get("decision_time", "23:00")
    tolerance_minutes = args.tolerance_minutes if args.tolerance_minutes is not None else int(night_config.get("tolerance_minutes", 10))
    timezone = night_config.get("timezone", "Asia/Shanghai")
    intraday_output = args.intraday_output or night_config.get("intraday_output", "data/raw/gold/sge_au9999_intraday.csv")
    decision_output = args.decision_output or night_config.get("decision_output", "data/raw/gold/au9999_night_decision_prices.csv")

    intraday_path = PROJECT_ROOT / intraday_output
    decision_path = PROJECT_ROOT / decision_output
    quotes, row, decision_prices = fetch_and_store_night_decision_price(
        night_config,
        intraday_path,
        decision_path,
        decision_time=decision_time,
        tolerance_minutes=tolerance_minutes,
    )
    print(f"saved {len(quotes)} rows to {intraday_path}")
    print(f"saved {len(decision_prices)} rows to {decision_path}")
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
