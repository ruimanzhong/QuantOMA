"""Robustness diagnostics for Phase 1 rule baselines."""

from __future__ import annotations

import pandas as pd

from market_quant.backtest.metrics import (
    active_ratio,
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    n_trading_days,
    sharpe,
    turnover,
)


DEFAULT_SUBPERIODS = [
    {"name": "2015-2018", "start_date": "2015-01-01", "end_date": "2018-12-31"},
    {"name": "2019-2021", "start_date": "2019-01-01", "end_date": "2021-12-31"},
    {"name": "2022-2024", "start_date": "2022-01-01", "end_date": "2024-12-31"},
    {"name": "2025-2026", "start_date": "2025-01-01", "end_date": "2026-12-31"},
]


def _validate_daily_backtest(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset_id", "strategy_name", "strategy_return", "executed_position"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"daily_backtest_df missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["asset_id", "strategy_name", "date"]).reset_index(drop=True)


def _summarize_group(group: pd.DataFrame, include_annualized: bool = True) -> dict:
    returns = group["strategy_return"]
    positions = group["executed_position"]
    row = {
        "n_trading_days": n_trading_days(returns),
        "cumulative_return": cumulative_return(returns),
        "sharpe": sharpe(returns) if len(returns.dropna()) >= 2 else float("nan"),
        "max_drawdown": max_drawdown(returns),
        "active_ratio": active_ratio(positions),
        "turnover": turnover(positions),
    }
    if include_annualized:
        row["annualized_return"] = annualized_return(returns) if len(returns.dropna()) >= 2 else float("nan")
        row["annualized_volatility"] = annualized_volatility(returns) if len(returns.dropna()) >= 2 else float("nan")
    return row


def annual_strategy_performance(daily_backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy performance independently for each calendar year."""
    df = _validate_daily_backtest(daily_backtest_df)
    df["year"] = df["date"].dt.year
    rows = []
    for (asset_id, strategy_name, year), group in df.groupby(["asset_id", "strategy_name", "year"], sort=True):
        row = {"asset_id": asset_id, "strategy_name": strategy_name, "year": int(year)}
        row.update(_summarize_group(group, include_annualized=True))
        rows.append(row)
    columns = [
        "asset_id",
        "strategy_name",
        "year",
        "n_trading_days",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "active_ratio",
        "turnover",
    ]
    return pd.DataFrame(rows)[columns]


def subperiod_strategy_performance(
    daily_backtest_df: pd.DataFrame,
    periods: list[dict] | None = None,
) -> pd.DataFrame:
    """Summarize strategy performance within fixed user-supplied subperiods."""
    df = _validate_daily_backtest(daily_backtest_df)
    periods = periods or DEFAULT_SUBPERIODS
    rows = []
    for period in periods:
        name = period.get("name") or period.get("subperiod")
        start_date = pd.Timestamp(period["start_date"])
        end_date = pd.Timestamp(period["end_date"])
        period_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if period_df.empty:
            continue
        for (asset_id, strategy_name), group in period_df.groupby(["asset_id", "strategy_name"], sort=True):
            row = {
                "asset_id": asset_id,
                "strategy_name": strategy_name,
                "subperiod": name,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
            }
            row.update(_summarize_group(group, include_annualized=False))
            rows.append(row)
    columns = [
        "asset_id",
        "strategy_name",
        "subperiod",
        "start_date",
        "end_date",
        "n_trading_days",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "active_ratio",
        "turnover",
    ]
    return pd.DataFrame(rows)[columns]
