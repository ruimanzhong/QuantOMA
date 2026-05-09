"""Lightweight alpha-style price/volume fallback features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0, np.nan)


def build_alpha_style_fallback_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Build a compact set of price/volume features by asset.

    This is a fallback for non-Qlib data and is not a full Alpha158 clone.
    """
    windows = windows or [5, 10, 20, 60]
    required = {"date", "asset_id", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"fallback feature input missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["asset_id", "date"]).reset_index(drop=True)
    grouped = out.groupby("asset_id", group_keys=False)

    for window in windows:
        close = grouped["close"]
        out[f"ret_{window}d"] = close.pct_change(window, fill_method=None)
        ma = close.transform(lambda s: s.rolling(window, min_periods=window).mean())
        out[f"ma_dist_{window}d"] = out["close"] / ma - 1.0
        out[f"volatility_{window}d"] = close.transform(
            lambda s: s.pct_change(fill_method=None).rolling(window, min_periods=window).std()
        )
        rolling_high = close.transform(lambda s: s.rolling(window, min_periods=window).max())
        rolling_low = close.transform(lambda s: s.rolling(window, min_periods=window).min())
        out[f"rolling_high_{window}d"] = rolling_high
        out[f"rolling_low_{window}d"] = rolling_low
        out[f"drawdown_{window}d"] = out["close"] / rolling_high - 1.0
        if "volume" in out.columns:
            out[f"volume_zscore_{window}d"] = grouped["volume"].transform(
                lambda s: _volume_zscore(s.astype(float), window)
            )

    return out
