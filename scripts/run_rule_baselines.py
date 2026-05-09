#!/usr/bin/env python
"""Run Phase 1 rule strategy baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.backtest.engine import run_daily_long_only_backtest
from market_quant.backtest.metrics import summarize_backtest
from market_quant.data.schema import validate_price_frame
from market_quant.strategies.breakout import donchian_breakout_20d, donchian_breakout_60d
from market_quant.strategies.rule_core import rule_core_vote_signal
from market_quant.strategies.support_pullback import support_pullback_trend
from market_quant.strategies.trend_following import dual_momentum_20_60, ma_trend, momentum_20d, momentum_60d
from market_quant.strategies.volume_confirmed import volume_confirmed_breakout


def buy_and_hold_signal(price_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": price_df["date"],
            "asset_id": price_df["asset_id"],
            "raw_signal": 1.0,
            "position": 1.0,
            "strategy_name": "buy_and_hold",
        }
    )


def run_buy_and_hold_backtest(asset_price: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    signals = buy_and_hold_signal(asset_price)
    bt = run_daily_long_only_backtest(asset_price, signals, transaction_cost_bps=cost_bps)
    bt["executed_position"] = 1.0
    bt["strategy_return"] = bt["asset_return"].fillna(0.0)
    return bt


def main() -> None:
    strategy_config = yaml.safe_load((PROJECT_ROOT / "config" / "strategies.yaml").read_text())
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    price_df = validate_price_frame(pd.read_csv(price_path))
    cost_bps = float(strategy_config.get("transaction_cost_bps", 2.0))

    strategy_funcs = [
        momentum_20d,
        momentum_60d,
        dual_momentum_20_60,
        ma_trend,
        donchian_breakout_20d,
        donchian_breakout_60d,
        lambda df: support_pullback_trend(df, **strategy_config.get("support_pullback", {})),
        lambda df: volume_confirmed_breakout(df, **strategy_config.get("volume_confirmed_breakout", {})),
        lambda df: rule_core_vote_signal(df, **strategy_config.get("rule_core", {})),
    ]

    backtests = []
    summaries = []
    for asset_id, asset_price in price_df.groupby("asset_id", sort=True):
        buy_hold_bt = run_buy_and_hold_backtest(asset_price, cost_bps)
        backtests.append(buy_hold_bt)
        row = {"asset_id": asset_id, "strategy_name": "buy_and_hold"}
        row.update(summarize_backtest(buy_hold_bt))
        row["active_ratio"] = 1.0
        row["n_active_days"] = row["n_trading_days"]
        summaries.append(row)

        for func in strategy_funcs:
            signals = func(asset_price)
            bt = run_daily_long_only_backtest(asset_price, signals, transaction_cost_bps=cost_bps)
            backtests.append(bt)
            strategy_name = signals["strategy_name"].iloc[0]
            row = {"asset_id": asset_id, "strategy_name": strategy_name}
            row.update(summarize_backtest(bt))
            summaries.append(row)

    backtest_df = pd.concat(backtests, ignore_index=True)
    comparison_df = pd.DataFrame(summaries)
    out_bt = PROJECT_ROOT / "data" / "backtest" / "rule_baselines_daily.csv"
    out_cmp = PROJECT_ROOT / "data" / "reports" / "rule_baseline_comparison.csv"
    out_bt.parent.mkdir(parents=True, exist_ok=True)
    out_cmp.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(out_bt, index=False)
    comparison_df.to_csv(out_cmp, index=False)
    print(f"saved daily backtests to {out_bt}")
    print(f"saved comparison report to {out_cmp}")


if __name__ == "__main__":
    main()
