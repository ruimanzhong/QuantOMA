"""Support pullback baseline."""

from __future__ import annotations

import pandas as pd

from market_quant.strategies.rule_core import support_pullback_trend_signal


def support_pullback_trend(
    df: pd.DataFrame,
    ma_window: int = 20,
    trend_window: int = 60,
    pullback_threshold: float = 0.015,
) -> pd.DataFrame:
    return support_pullback_trend_signal(df, ma_window, trend_window, pullback_threshold)
