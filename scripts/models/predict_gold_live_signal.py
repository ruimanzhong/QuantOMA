#!/usr/bin/env python
"""Produce a live gold ETF probability and 0-1 exposure signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.live_pipeline import run_live_prediction


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", required=True, help="Decision date, e.g. 2026-05-06.")
    parser.add_argument("--as-of-date", default=None, help="Latest data date allowed for live overlays, e.g. 2026-05-05.")
    parser.add_argument("--preopen", action="store_true", help="Add a synthetic pre-open primary ETF row if the decision date has no China close.")
    parser.add_argument("--manual-primary-price", type=float, default=None, help="Override/add the primary ETF price for signal-date for near-close live decisions.")
    parser.add_argument("--current-position", type=float, default=None, help="Current portfolio gold exposure from 0 to 1.")
    parser.add_argument("--rebalance-band", type=float, default=0.05, help="Do not trade if target-current absolute delta is below this band.")
    parser.add_argument("--max-position-change", type=float, default=0.35, help="Maximum one-signal position change.")
    parser.add_argument("--output", default="data/reports/gold_live_signal.json")
    parser.add_argument("--selected-output", default="data/reports/gold_live_selected_features.csv")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    payload, selected_report = run_live_prediction(
        PROJECT_ROOT,
        config,
        signal_date=args.signal_date,
        as_of_date=args.as_of_date,
        manual_primary_price=args.manual_primary_price,
        preopen=args.preopen,
        current_position=args.current_position,
        rebalance_band=args.rebalance_band,
        max_position_change=args.max_position_change,
    )
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))

    selected_output = PROJECT_ROOT / args.selected_output
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_report.to_csv(selected_output, index=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    print(f"saved live signal to {output}")
    print(f"saved selected feature report to {selected_output}")


if __name__ == "__main__":
    main()
