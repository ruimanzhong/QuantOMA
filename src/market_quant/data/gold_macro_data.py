"""FRED macro data adapter for gold research."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def fetch_gold_macro_data(config: dict[str, Any]) -> pd.DataFrame:
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError("fredapi is required for FRED macro data.") from exc
    api_key_env = config.get("api_key_env", "FRED_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing FRED API key env var: {api_key_env}")

    fred = Fred(api_key=api_key)
    start = config["start_date"]
    end = config.get("end_date")
    frames = []
    for alias, series_id in config.get("symbols", {}).items():
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        out = pd.Series(series, name=alias, dtype="float64")
        out.index = pd.to_datetime(out.index).normalize()
        frames.append(out.sort_index())
    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1).sort_index()
    full_index = pd.date_range(macro.index.min(), macro.index.max(), freq="D")
    macro = macro.reindex(full_index)
    if config.get("fill_method", "ffill") == "ffill":
        macro = macro.ffill()
    macro.index.name = "date"
    return macro.reset_index()
