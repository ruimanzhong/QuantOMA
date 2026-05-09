"""Sample attrition diagnostics."""

from __future__ import annotations

import pandas as pd


def summarize_sample_attrition(df: pd.DataFrame, group_col: str = "asset_id") -> pd.DataFrame:
    out = []
    for key, group in df.groupby(group_col):
        out.append(
            {
                group_col: key,
                "start_date": pd.to_datetime(group["date"]).min(),
                "end_date": pd.to_datetime(group["date"]).max(),
                "n_rows": len(group),
                "n_missing_close": int(group["close"].isna().sum()) if "close" in group else 0,
            }
        )
    return pd.DataFrame(out)
