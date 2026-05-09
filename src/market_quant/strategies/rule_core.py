"""Shared rule-strategy signal helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd


SignalFunc = Callable[[pd.DataFrame], pd.DataFrame]


def _prepare_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _format_signal(df: pd.DataFrame, strategy_name: str, raw_signal: pd.Series) -> pd.DataFrame:
    position = raw_signal.clip(lower=0, upper=1).fillna(0.0)
    return pd.DataFrame(
        {
            "date": df["date"],
            "asset_id": df["asset_id"],
            "strategy_name": strategy_name,
            "raw_signal": raw_signal.fillna(0.0).astype(float),
            "position": position.astype(float),
        }
    )


def momentum_signal(df: pd.DataFrame, window: int, strategy_name: str | None = None) -> pd.DataFrame:
    price = _prepare_price(df)
    ret = price.groupby("asset_id")["close"].pct_change(window, fill_method=None)
    raw = (ret > 0).astype(float)
    return _format_signal(price, strategy_name or f"momentum_{window}d", raw)


def dual_momentum_signal(df: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
    price = _prepare_price(df)
    grouped = price.groupby("asset_id")["close"]
    short_ret = grouped.pct_change(short_window, fill_method=None)
    long_ret = grouped.pct_change(long_window, fill_method=None)
    raw = ((short_ret > 0) & (long_ret > 0) & (short_ret > long_ret)).astype(float)
    return _format_signal(price, f"dual_momentum_{short_window}_{long_window}", raw)


def ma_trend_signal(df: pd.DataFrame, ma_window: int = 20, trend_window: int = 60) -> pd.DataFrame:
    price = _prepare_price(df)
    grouped = price.groupby("asset_id")["close"]
    fast_ma = grouped.transform(lambda s: s.rolling(ma_window, min_periods=ma_window).mean())
    slow_ma = grouped.transform(lambda s: s.rolling(trend_window, min_periods=trend_window).mean())
    raw = ((price["close"] > fast_ma) & (fast_ma > slow_ma)).astype(float)
    return _format_signal(price, "ma_trend", raw)


def donchian_breakout_signal(df: pd.DataFrame, window: int, strategy_name: str | None = None) -> pd.DataFrame:
    price = _prepare_price(df)
    prior_high = price.groupby("asset_id")["high"].transform(
        lambda s: s.rolling(window, min_periods=window).max().shift(1)
    )
    raw = (price["close"] > prior_high).astype(float)
    return _format_signal(price, strategy_name or f"donchian_breakout_{window}d", raw)


def support_pullback_trend_signal(
    df: pd.DataFrame,
    ma_window: int = 20,
    trend_window: int = 60,
    pullback_threshold: float = 0.015,
) -> pd.DataFrame:
    price = _prepare_price(df)
    grouped = price.groupby("asset_id")["close"]
    ma = grouped.transform(lambda s: s.rolling(ma_window, min_periods=ma_window).mean())
    trend_ma = grouped.transform(lambda s: s.rolling(trend_window, min_periods=trend_window).mean())
    uptrend = (price["close"] > trend_ma) & (ma > trend_ma)
    near_support = (price["close"] >= ma * (1.0 - pullback_threshold)) & (
        price["close"] <= ma * (1.0 + pullback_threshold)
    )
    raw = (uptrend & near_support).astype(float)
    return _format_signal(price, "support_pullback_trend", raw)


def volume_confirmed_breakout_signal(
    df: pd.DataFrame,
    breakout_window: int = 20,
    volume_zscore_threshold: float = 1.0,
) -> pd.DataFrame:
    price = _prepare_price(df)
    if "volume" not in price.columns:
        warnings.warn("volume_confirmed_breakout requires volume; returning flat signal", RuntimeWarning)
        return _format_signal(price, "volume_confirmed_breakout", pd.Series(0.0, index=price.index))

    prior_high = price.groupby("asset_id")["high"].transform(
        lambda s: s.rolling(breakout_window, min_periods=breakout_window).max().shift(1)
    )

    def zscore(s: pd.Series) -> pd.Series:
        mean = s.rolling(breakout_window, min_periods=breakout_window).mean()
        std = s.rolling(breakout_window, min_periods=breakout_window).std()
        return (s - mean) / std.replace(0, np.nan)

    vol_z = price.groupby("asset_id")["volume"].transform(lambda s: zscore(s.astype(float)))
    raw = ((price["close"] > prior_high) & (vol_z >= volume_zscore_threshold)).astype(float)
    return _format_signal(price, "volume_confirmed_breakout", raw)


def rule_core_vote_signal(
    df: pd.DataFrame,
    vote_threshold: float = 0.5,
    strategy_funcs: list[SignalFunc] | None = None,
) -> pd.DataFrame:
    price = _prepare_price(df)
    strategy_funcs = strategy_funcs or [
        lambda frame: momentum_signal(frame, 20),
        lambda frame: momentum_signal(frame, 60),
        dual_momentum_signal,
        ma_trend_signal,
        lambda frame: donchian_breakout_signal(frame, 20),
        lambda frame: donchian_breakout_signal(frame, 60),
        support_pullback_trend_signal,
        volume_confirmed_breakout_signal,
    ]
    signals = [func(price)["position"].reset_index(drop=True) for func in strategy_funcs]
    if not signals:
        raw = pd.Series(0.0, index=price.index)
    else:
        vote = pd.concat(signals, axis=1).mean(axis=1)
        raw = (vote >= vote_threshold).astype(float)
    return _format_signal(price, "rule_core_vote", raw)
