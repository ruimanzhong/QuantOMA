#!/usr/bin/env python
"""Run Phase 1 transaction cost sensitivity diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.backtest.engine import run_daily_long_only_backtest
from market_quant.backtest.metrics import summarize_backtest
from market_quant.data.schema import validate_price_frame
from market_quant.strategies.breakout import donchian_breakout_20d
from market_quant.strategies.rule_core import rule_core_vote_signal
from market_quant.strategies.trend_following import momentum_20d, momentum_60d
from market_quant.strategies.volume_confirmed import volume_confirmed_breakout


def buy_and_hold_signal(price_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": price_df["date"],
            "asset_id": price_df["asset_id"],
            "strategy_name": "buy_and_hold",
            "raw_signal": 1.0,
            "position": 1.0,
        }
    )


def run_cost_sensitivity(price_df: pd.DataFrame, costs: list[float] | None = None) -> pd.DataFrame:
    price_df = validate_price_frame(price_df)
    costs = costs or [0, 2, 5, 10]
    strategies = [
        ("buy_and_hold", buy_and_hold_signal),
        ("momentum_20d", momentum_20d),
        ("momentum_60d", momentum_60d),
        ("donchian_breakout_20d", donchian_breakout_20d),
        ("rule_core_vote", rule_core_vote_signal),
        ("volume_confirmed_breakout", volume_confirmed_breakout),
    ]
    rows = []
    for asset_id, asset_price in price_df.groupby("asset_id", sort=True):
        for cost in costs:
            for strategy_name, func in strategies:
                signals = func(asset_price)
                bt = run_daily_long_only_backtest(asset_price, signals, transaction_cost_bps=cost)
                metrics = summarize_backtest(bt)
                rows.append(
                    {
                        "asset_id": asset_id,
                        "strategy_name": strategy_name,
                        "transaction_cost_bps": cost,
                        "cumulative_return": metrics["cumulative_return"],
                        "sharpe": metrics["sharpe"],
                        "max_drawdown": metrics["max_drawdown"],
                        "turnover": metrics["turnover"],
                        "active_ratio": metrics["active_ratio"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    if not price_path.exists():
        raise SystemExit("Run python scripts/fetch_a_share_data.py first.")
    report = run_cost_sensitivity(pd.read_csv(price_path))
    out = PROJECT_ROOT / "data" / "reports" / "transaction_cost_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    pivot = report.pivot_table(
        index=["asset_id", "strategy_name"],
        columns="transaction_cost_bps",
        values=["cumulative_return", "sharpe", "turnover"],
    )
    degradation = []
    for (asset_id, strategy_name), row in pivot.iterrows():
        if ("cumulative_return", 0) in row.index and ("cumulative_return", 10) in row.index:
            degradation.append(
                {
                    "asset_id": asset_id,
                    "strategy_name": strategy_name,
                    "turnover_0bps": row.get(("turnover", 0), 0.0),
                    "return_drop_0_to_10bps": row[("cumulative_return", 0)] - row[("cumulative_return", 10)],
                    "sharpe_drop_0_to_10bps": row[("sharpe", 0)] - row[("sharpe", 10)],
                }
            )
    degradation_df = pd.DataFrame(degradation).sort_values("return_drop_0_to_10bps", ascending=False)
    print("Largest 0bps to 10bps degradation:")
    print(degradation_df.head(12).to_string(index=False))
    print(f"\nsaved transaction cost sensitivity to {out}")


if __name__ == "__main__":
    main()
