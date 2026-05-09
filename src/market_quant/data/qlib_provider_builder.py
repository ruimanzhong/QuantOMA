"""Build a local Qlib provider from Phase 1 ETF daily data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from market_quant.data.schema import validate_price_frame


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_feature_bin(path: Path, values: np.ndarray, start_index: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.hstack(([start_index], values.astype(np.float32, copy=False))).astype("<f")
    with path.open("wb") as fp:
        arr.tofile(fp)


def _build_series(group: pd.DataFrame, calendar: pd.Index, field: str) -> np.ndarray:
    series = pd.Series(np.nan, index=calendar, dtype="float32")
    values = group.set_index("date")[field].astype("float32")
    series.loc[values.index] = values.values
    return series.to_numpy(dtype="float32", copy=False)


def build_etf_qlib_provider(price_df: pd.DataFrame, provider_uri: str) -> dict:
    """Write a Qlib-compatible provider from ETF daily price data."""
    df = validate_price_frame(price_df)
    provider = Path(provider_uri).expanduser().resolve()
    if provider.exists():
        shutil.rmtree(provider)
    provider.mkdir(parents=True, exist_ok=True)

    calendar = pd.Index(pd.to_datetime(sorted(df["date"].unique())))
    calendar_lines = [_format_date(date) for date in calendar]
    _write_text_lines(provider / "calendars" / "day.txt", calendar_lines)

    instrument_lines: list[str] = []
    for asset_id, group in df.groupby("asset_id", sort=True):
        instrument_lines.append(
            f"{asset_id}\t{_format_date(group['date'].min())}\t{_format_date(group['date'].max())}"
        )
    _write_text_lines(provider / "instruments" / "all.txt", instrument_lines)

    fields = {
        "open": lambda group: _build_series(group, calendar, "open"),
        "high": lambda group: _build_series(group, calendar, "high"),
        "low": lambda group: _build_series(group, calendar, "low"),
        "close": lambda group: _build_series(group, calendar, "close"),
        "volume": lambda group: _build_series(group, calendar, "volume"),
        "amount": lambda group: _build_series(group, calendar, "amount"),
        "money": lambda group: _build_series(group, calendar, "amount"),
        "vwap": lambda group: _build_series(
            group.assign(vwap=np.where(group["volume"].to_numpy() > 0, group["amount"] / group["volume"], np.nan)),
            calendar,
            "vwap",
        ),
        "factor": lambda group: np.where(np.isfinite(_build_series(group, calendar, "close")), 1.0, np.nan),
        "paused": lambda group: np.where(np.isfinite(_build_series(group, calendar, "close")), 0.0, np.nan),
    }

    for asset_id, group in df.groupby("asset_id", sort=True):
        inst_dir = provider / "features" / asset_id.lower()
        for field, builder in fields.items():
            _write_feature_bin(inst_dir / f"{field}.day.bin", builder(group))

    summary = {
        "provider_uri": str(provider),
        "n_assets": int(df["asset_id"].nunique()),
        "n_calendar_days": int(len(calendar)),
        "first_date": _format_date(calendar.min()),
        "last_date": _format_date(calendar.max()),
        "fields": sorted(fields.keys()),
    }
    (provider / "provider_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
