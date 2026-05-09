"""Coverage diagnostics for future scalar daily series inputs."""

from __future__ import annotations

import pandas as pd

from market_quant.data.series_schema import validate_daily_series_frame


def summarize_daily_series_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-series coverage for scalar daily inputs."""
    series_df = validate_daily_series_frame(df)
    rows = []
    for series_id, group in series_df.groupby("series_id", sort=True):
        rows.append(
            {
                "series_id": series_id,
                "source": group["source"].iloc[0],
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "n_rows": len(group),
                "missing_rate": float(group["value"].isna().mean()),
                "duplicate_series_date_count": int(group.duplicated(["series_id", "date"]).sum()),
                "coverage_ok": len(group) >= 100 and float(group["value"].isna().mean()) < 0.5,
            }
        )
    return pd.DataFrame(rows)
