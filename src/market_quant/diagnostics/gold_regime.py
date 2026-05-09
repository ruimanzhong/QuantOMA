"""Regime diagnostics for the independent gold strategy pipeline."""

from __future__ import annotations

import pandas as pd

from market_quant.backtest.metrics import summarize_backtest


DEFAULT_GOLD_SUBPERIODS = [
    {"name": "2022_oos_start_to_2023", "start_date": "2022-05-26", "end_date": "2023-12-31"},
    {"name": "2024", "start_date": "2024-01-01", "end_date": "2024-12-31"},
    {"name": "2025_to_2026_oos_end", "start_date": "2025-01-01", "end_date": "2026-12-31"},
]


def prepare_gold_daily_strategy_frame(daily: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Normalize one daily gold backtest frame for grouped diagnostics."""
    required = {"date", "strategy_return", "executed_position", "primary_etf"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily strategy frame missing columns: {sorted(missing)}")
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = "gold_518880"
    out["strategy_name"] = strategy_name
    return out.sort_values("date").reset_index(drop=True)


def build_buy_hold_daily(predictions: pd.DataFrame, strategy_name: str = "buy_and_hold") -> pd.DataFrame:
    """Build buy-and-hold daily returns on the same prediction window."""
    required = {"date", "primary_etf"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["asset_return"] = out["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    out["position"] = 1.0
    out["executed_position"] = 1.0
    out["strategy_return"] = out["asset_return"]
    return prepare_gold_daily_strategy_frame(out, strategy_name)


def summarize_by_subperiod(
    daily_backtests: pd.DataFrame,
    periods: list[dict] | None = None,
) -> pd.DataFrame:
    """Summarize strategy performance for fixed calendar subperiods."""
    required = {"date", "asset_id", "strategy_name", "strategy_return", "executed_position"}
    missing = required.difference(daily_backtests.columns)
    if missing:
        raise ValueError(f"daily_backtests missing columns: {sorted(missing)}")
    df = daily_backtests.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for period in periods or DEFAULT_GOLD_SUBPERIODS:
        name = str(period.get("name") or period.get("subperiod"))
        start_date = pd.Timestamp(period["start_date"])
        end_date = pd.Timestamp(period["end_date"])
        part = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if part.empty:
            continue
        for (_, strategy_name), group in part.groupby(["asset_id", "strategy_name"], sort=True):
            row = {"subperiod": name, "strategy_name": strategy_name}
            row.update(summarize_backtest(group))
            rows.append(row)
    return pd.DataFrame(rows)


def add_price_regime_labels(
    daily_backtests: pd.DataFrame,
    short_window: int = 60,
    long_window: int = 120,
    momentum_window: int = 60,
) -> pd.DataFrame:
    """Attach simple price-only regime labels observable at prior close."""
    required = {"date", "primary_etf"}
    missing = required.difference(daily_backtests.columns)
    if missing:
        raise ValueError(f"daily_backtests missing columns: {sorted(missing)}")
    df = daily_backtests.copy()
    df["date"] = pd.to_datetime(df["date"])
    prices = df[["date", "primary_etf"]].drop_duplicates("date").sort_values("date").set_index("date")["primary_etf"]
    short_ma = prices.rolling(short_window, min_periods=max(5, short_window // 3)).mean()
    long_ma = prices.rolling(long_window, min_periods=max(10, long_window // 3)).mean()
    momentum = prices.pct_change(momentum_window, fill_method=None)
    observable = pd.DataFrame(
        {
            "date": prices.index,
            "short_ma": short_ma.shift(1).to_numpy(),
            "long_ma": long_ma.shift(1).to_numpy(),
            "momentum": momentum.shift(1).to_numpy(),
        }
    )
    out = df.merge(observable, on="date", how="left")
    out["price_regime"] = "insufficient_history"
    trend_up = (out["short_ma"] > out["long_ma"]) & (out["momentum"] > 0)
    trend_down = (out["short_ma"] < out["long_ma"]) & (out["momentum"] < 0)
    range_bound = out["short_ma"].notna() & out["long_ma"].notna() & out["momentum"].notna() & ~(trend_up | trend_down)
    out.loc[trend_up, "price_regime"] = "trend_up"
    out.loc[trend_down, "price_regime"] = "trend_down"
    out.loc[range_bound, "price_regime"] = "range_bound"
    return out.drop(columns=["short_ma", "long_ma", "momentum"])


def summarize_by_price_regime(daily_backtests: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy performance under each price-only regime."""
    required = {"price_regime", "asset_id", "strategy_name", "strategy_return", "executed_position"}
    missing = required.difference(daily_backtests.columns)
    if missing:
        raise ValueError(f"daily_backtests missing columns: {sorted(missing)}")
    rows = []
    for (regime, strategy_name), group in daily_backtests.groupby(["price_regime", "strategy_name"], sort=True):
        row = {"price_regime": regime, "strategy_name": strategy_name}
        row.update(summarize_backtest(group))
        rows.append(row)
    return pd.DataFrame(rows)
