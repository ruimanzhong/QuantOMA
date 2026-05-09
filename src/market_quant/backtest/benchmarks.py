"""Benchmark helpers for buy-and-hold comparisons."""

from __future__ import annotations

import pandas as pd

from market_quant.backtest.engine import run_daily_long_only_backtest


def build_buy_and_hold_signal(price_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df[["date", "asset_id"]].copy()
    out["strategy_name"] = "buy_and_hold"
    out["raw_signal"] = 1.0
    out["position"] = 1.0
    return out


def run_buy_and_hold_benchmark(price_df: pd.DataFrame, transaction_cost_bps: float = 0.0) -> pd.DataFrame:
    return run_daily_long_only_backtest(
        price_df,
        build_buy_and_hold_signal(price_df),
        transaction_cost_bps=transaction_cost_bps,
    )
