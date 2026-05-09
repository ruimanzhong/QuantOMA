"""Volume-confirmed breakout baseline."""

from __future__ import annotations

import pandas as pd

from market_quant.strategies.rule_core import volume_confirmed_breakout_signal


def volume_confirmed_breakout(
    df: pd.DataFrame,
    breakout_window: int = 20,
    volume_zscore_threshold: float = 1.0,
) -> pd.DataFrame:
    return volume_confirmed_breakout_signal(df, breakout_window, volume_zscore_threshold)
