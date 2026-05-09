#!/usr/bin/env python
"""Build a local Qlib provider from independent gold ETF OHLCV data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.qlib_provider_builder import build_etf_qlib_provider


def _gold_ohlcv_to_price_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit("Missing gold ETF CSV. Run python scripts/fetch_gold_local_series.py first.")
    df = pd.read_csv(path)
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"gold ETF CSV missing columns: {sorted(missing)}")
    out = df.rename(columns={"symbol": "asset_id"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = out["asset_id"].astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" not in out.columns:
        out["amount"] = out["close"] * out["volume"]
    out["source"] = "gold_local_csv"
    return out[["date", "asset_id", "open", "high", "low", "close", "volume", "amount", "source"]]


def main() -> None:
    feature_config = yaml.safe_load((PROJECT_ROOT / "config" / "feature_sets.yaml").read_text())["alpha158_gold"]
    source_path = PROJECT_ROOT / "../gold_llm_quant/data/raw/market/china_gold_etf.csv"
    provider_uri = PROJECT_ROOT / feature_config["provider_uri"]
    price_df = _gold_ohlcv_to_price_frame(source_path.resolve())
    summary = build_etf_qlib_provider(price_df, provider_uri)
    print(f"provider_uri: {summary['provider_uri']}")
    print(f"n_assets: {summary['n_assets']}")
    print(f"n_calendar_days: {summary['n_calendar_days']}")
    print(f"first_date: {summary['first_date']}")
    print(f"last_date: {summary['last_date']}")
    print(f"fields: {', '.join(summary['fields'])}")


if __name__ == "__main__":
    main()
