"""Canonical daily series schema for future scalar daily inputs."""

from __future__ import annotations

import pandas as pd


REQUIRED_SERIES_COLUMNS = {
    "date",
    "series_id",
    "value",
    "source",
}


def validate_daily_series_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a tidy daily series frame used by scalar daily data sources."""
    missing = REQUIRED_SERIES_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"daily series frame missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["series_id"] = out["series_id"].astype(str)
    out["source"] = out["source"].astype(str)

    duplicate_mask = out.duplicated(["series_id", "date"], keep=False)
    if duplicate_mask.any():
        duplicates = out.loc[duplicate_mask, ["series_id", "date"]].head(10)
        raise ValueError(f"duplicate series_id/date rows found: {duplicates.to_dict('records')}")

    if out["value"].isna().any():
        bad_rows = out.loc[out["value"].isna(), ["series_id", "date"]].head(10)
        raise ValueError(f"value contains missing values: {bad_rows.to_dict('records')}")

    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def normalize_daily_series_frame(df: pd.DataFrame, value_name: str = "value") -> pd.DataFrame:
    """Normalize common wide or scalar daily data into the canonical series layout."""
    out = df.copy()
    if "series_id" not in out.columns and "asset_id" in out.columns:
        out = out.rename(columns={"asset_id": "series_id"})
    if value_name != "value" and value_name in out.columns:
        out = out.rename(columns={value_name: "value"})
    return validate_daily_series_frame(out)
