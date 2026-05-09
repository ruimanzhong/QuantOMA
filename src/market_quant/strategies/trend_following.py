"""Trend-following rule baselines."""

from __future__ import annotations

import pandas as pd

from market_quant.strategies.rule_core import dual_momentum_signal, ma_trend_signal, momentum_signal


def momentum_20d(df: pd.DataFrame) -> pd.DataFrame:
    return momentum_signal(df, 20, "momentum_20d")


def momentum_60d(df: pd.DataFrame) -> pd.DataFrame:
    return momentum_signal(df, 60, "momentum_60d")


def dual_momentum_20_60(df: pd.DataFrame) -> pd.DataFrame:
    return dual_momentum_signal(df, 20, 60)


def ma_trend(df: pd.DataFrame) -> pd.DataFrame:
    return ma_trend_signal(df)
