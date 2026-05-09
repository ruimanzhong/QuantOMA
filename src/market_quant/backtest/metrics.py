"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def cumulative_return(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    return float((1.0 + clean).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.fillna(0.0)
    if clean.empty:
        return 0.0
    total = (1.0 + clean).prod()
    return float(total ** (periods_per_year / len(clean)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    return annualized_return(returns, periods_per_year) / vol


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def hit_rate(returns: pd.Series) -> float:
    active = returns.dropna()
    if active.empty:
        return 0.0
    return float((active > 0).mean())


def turnover(positions: pd.Series) -> float:
    clean = positions.fillna(0.0)
    if clean.empty:
        return 0.0
    return float(clean.diff().abs().fillna(clean.abs()).mean())


def active_ratio(positions: pd.Series) -> float:
    clean = positions.fillna(0.0)
    if clean.empty:
        return 0.0
    return float((clean.abs() > 0).mean())


def n_trading_days(returns: pd.Series) -> int:
    return int(returns.shape[0])


def n_active_days(positions: pd.Series) -> int:
    return int((positions.fillna(0.0).abs() > 0).sum())


def summarize_backtest(group: pd.DataFrame) -> dict[str, float | int | str]:
    returns = group["strategy_return"]
    positions = group["executed_position"] if "executed_position" in group.columns else group["position"]
    return {
        "start_date": group["date"].min(),
        "end_date": group["date"].max(),
        "n_trading_days": n_trading_days(returns),
        "n_active_days": n_active_days(positions),
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "active_ratio": active_ratio(positions),
        "turnover": turnover(positions),
    }
