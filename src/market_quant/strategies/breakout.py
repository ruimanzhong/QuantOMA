"""Breakout rule baselines."""

from __future__ import annotations

import pandas as pd

from market_quant.strategies.rule_core import donchian_breakout_signal


def donchian_breakout_20d(df: pd.DataFrame) -> pd.DataFrame:
    return donchian_breakout_signal(df, 20, "donchian_breakout_20d")


def donchian_breakout_60d(df: pd.DataFrame) -> pd.DataFrame:
    return donchian_breakout_signal(df, 60, "donchian_breakout_60d")
