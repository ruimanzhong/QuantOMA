"""Simple exposure diagnostics for rule signals."""

from __future__ import annotations

import pandas as pd


def summarize_signal_exposure(signal_df: pd.DataFrame) -> pd.DataFrame:
    return (
        signal_df.groupby(["asset_id", "strategy_name"], as_index=False)["position"]
        .agg(active_ratio=lambda s: float((s.fillna(0.0).abs() > 0).mean()), avg_position="mean")
    )
