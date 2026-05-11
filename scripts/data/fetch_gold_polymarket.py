#!/usr/bin/env python
"""Fetch Polymarket probability snapshots for gold macro features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.polymarket_data import POLYMARKET_COLUMNS, append_polymarket_cache, fetch_gold_polymarket_snapshots


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())
    gold_cfg = config.get("gold", {})
    poly_cfg = gold_cfg.get("polymarket", {})
    output = PROJECT_ROOT / poly_cfg.get("output", "data/raw/gold/polymarket_markets.csv")
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=POLYMARKET_COLUMNS)
    snapshot = fetch_gold_polymarket_snapshots(gold_cfg)
    out = append_polymarket_cache(existing, snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(f"saved Polymarket snapshots to {output}")
    print(f"new rows: {len(snapshot)}")
    print(f"total rows: {len(out)}")
    if not snapshot.empty:
        summary = snapshot.groupby("market_group")["probability"].agg(["count", "mean", "max"]).reset_index()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
