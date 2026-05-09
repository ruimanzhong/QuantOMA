"""Feature availability timing utilities."""

from __future__ import annotations

import pandas as pd


def apply_feature_availability_lag(
    df: pd.DataFrame,
    lag_rows: int = 1,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Lag feature columns by asset to model next-bar availability.

    Identifier columns are never shifted. Additional columns can be excluded
    when they are metadata or already point-in-time safe.
    """
    if lag_rows < 0:
        raise ValueError("lag_rows must be non-negative")
    required = {"date", "asset_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"feature frame missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["asset_id", "date"]).reset_index(drop=True)

    if lag_rows == 0:
        return out

    excluded = {"date", "asset_id", *(exclude_columns or [])}
    lag_columns = [col for col in out.columns if col not in excluded]
    out[lag_columns] = out.groupby("asset_id", group_keys=False)[lag_columns].shift(lag_rows)
    return out
