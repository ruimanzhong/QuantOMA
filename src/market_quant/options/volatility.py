"""Volatility-state helpers for the options overlay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def realized_volatility(price: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    """Annualized realized volatility from close-to-close returns."""
    px = pd.to_numeric(price, errors="coerce").dropna()
    returns = px.pct_change()
    return returns.rolling(int(window)).std() * np.sqrt(float(annualization))


def _infer_price_column(feature_frame: pd.DataFrame, config: dict[str, Any]) -> str | None:
    candidates = [
        config.get("price_column"),
        "primary_etf",
        "primary_price",
        "au9999_close",
        "AU9999_close",
        "close",
    ]
    for col in candidates:
        if col and col in feature_frame.columns:
            return str(col)
    numeric = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    return numeric[0] if numeric else None


def _atm_row(chain: pd.DataFrame) -> pd.Series | None:
    if chain.empty:
        return None
    underlying = float(pd.to_numeric(chain["underlying_price"], errors="coerce").dropna().median())
    if not np.isfinite(underlying) or underlying <= 0:
        return None
    idx = (pd.to_numeric(chain["strike"], errors="coerce") - underlying).abs().idxmin()
    return chain.loc[idx]


def build_option_vol_state(
    chain: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_date: str | pd.Timestamp | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight IV/RV/VRP state for live option selection.

    Phase 1 uses realized-vol fallback for forecast RV.  A future ML RV model can
    replace ``forecast_rv_5d`` without changing downstream interfaces.
    """
    cfg = config or {}
    vol_cfg = cfg.get("volatility", cfg)
    rv_window = int(vol_cfg.get("rv_window", 20))
    fallback_rv = float(vol_cfg.get("fallback_rv", 0.20))
    low_iv_threshold = float(vol_cfg.get("low_iv_threshold", 0.16))
    high_iv_threshold = float(vol_cfg.get("high_iv_threshold", 0.28))

    iv_series = pd.to_numeric(chain.get("iv", pd.Series(dtype=float)), errors="coerce").dropna()
    atm = _atm_row(chain)
    iv_atm = None
    if atm is not None and "iv" in atm and pd.notna(atm["iv"]):
        iv_atm = float(atm["iv"])
    elif not iv_series.empty:
        iv_atm = float(iv_series.median())

    price_col = _infer_price_column(feature_frame, vol_cfg) if not feature_frame.empty else None
    rv_20d = fallback_rv
    if price_col:
        history = feature_frame.copy()
        if isinstance(history.index, pd.DatetimeIndex) and feature_date is not None:
            history = history.loc[history.index <= pd.Timestamp(feature_date)]
        rv = realized_volatility(history[price_col], window=rv_window).dropna()
        if not rv.empty and np.isfinite(rv.iloc[-1]):
            rv_20d = float(rv.iloc[-1])
    forecast_rv_5d = float(vol_cfg.get("forecast_rv_5d", rv_20d))

    # If no IV is available, use RV as a conservative fallback but mark it.
    iv_source = "chain_iv"
    if iv_atm is None or not np.isfinite(iv_atm) or iv_atm <= 0:
        iv_atm = forecast_rv_5d
        iv_source = "rv_fallback"

    if iv_atm <= low_iv_threshold:
        state = "low_iv"
    elif iv_atm >= high_iv_threshold:
        state = "high_iv"
    else:
        state = "neutral_iv"

    iv_rank = None
    if not iv_series.empty:
        iv_rank = float((iv_series <= iv_atm).mean())

    return {
        "iv_atm": float(iv_atm),
        "iv_rank": iv_rank,
        "rv_20d": float(rv_20d),
        "forecast_rv_5d": float(forecast_rv_5d),
        "vrp": float(iv_atm - forecast_rv_5d),
        "volatility_state": state,
        "iv_source": iv_source,
        "price_column": price_col,
    }
