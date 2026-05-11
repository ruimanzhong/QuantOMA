#!/usr/bin/env python
"""Build a local Qlib provider from AU9999 SGE OHLC data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.qlib_provider_builder import build_etf_qlib_provider


def _au9999_sge_to_price_frame(path: Path, asset_id: str = "AU9999") -> pd.DataFrame:
    if not path.exists():
        raise SystemExit("Missing SGE gold CSV. Run python scripts/data/fetch_gold_data.py first.")
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SGE gold CSV missing columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = asset_id
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # SGE spot endpoint does not expose volume consistently. Keep it missing so
    # volume-derived Alpha158 columns are filtered rather than fabricated.
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce") if "volume" in out.columns else np.nan
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out.columns else np.nan
    out["source"] = "sge_akshare"
    return out[["date", "asset_id", "open", "high", "low", "close", "volume", "amount", "source"]]


def main() -> None:
    feature_config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())["alpha158_gold"]
    source_path = PROJECT_ROOT / "data" / "raw" / "gold" / "sge_gold.csv"
    provider_uri = PROJECT_ROOT / feature_config["provider_uri"]
    instruments = feature_config.get("instruments", ["AU9999"])
    asset_id = instruments[0] if isinstance(instruments, list) else str(instruments)
    price_df = _au9999_sge_to_price_frame(source_path.resolve(), asset_id=asset_id)
    summary = build_etf_qlib_provider(price_df, provider_uri)
    print(f"provider_uri: {summary['provider_uri']}")
    print(f"n_assets: {summary['n_assets']}")
    print(f"n_calendar_days: {summary['n_calendar_days']}")
    print(f"first_date: {summary['first_date']}")
    print(f"last_date: {summary['last_date']}")
    print(f"fields: {', '.join(summary['fields'])}")


if __name__ == "__main__":
    main()
