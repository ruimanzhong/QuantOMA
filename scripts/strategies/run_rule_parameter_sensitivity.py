#!/usr/bin/env python
"""Run Phase 1 rule parameter sensitivity diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.backtest.engine import run_daily_long_only_backtest
from market_quant.backtest.metrics import summarize_backtest
from market_quant.data.schema import validate_price_frame
from market_quant.diagnostics.rule_baseline_evaluation import evaluate_rule_vs_buy_hold
from market_quant.strategies.rule_core import (
    donchian_breakout_signal,
    ma_trend_signal,
    momentum_signal,
    support_pullback_trend_signal,
)


def _buy_and_hold_summary(asset_price: pd.DataFrame) -> dict:
    returns = asset_price.sort_values("date")["close"].pct_change(fill_method=None).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * (252**0.5)) if returns.std(ddof=1) else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def run_parameter_sensitivity(price_df: pd.DataFrame, transaction_cost_bps: float = 2.0) -> pd.DataFrame:
    price_df = validate_price_frame(price_df)
    rows = []
    for asset_id, asset_price in price_df.groupby("asset_id", sort=True):
        buy_hold = _buy_and_hold_summary(asset_price)
        variants = []
        for window in [10, 20, 40, 60]:
            variants.append(("momentum", f"momentum_{window}d", f"window={window}", lambda df, w=window: momentum_signal(df, w, f"momentum_{w}d")))
        for window in [10, 20, 40, 60]:
            variants.append(("donchian_breakout", f"donchian_breakout_{window}d", f"window={window}", lambda df, w=window: donchian_breakout_signal(df, w, f"donchian_breakout_{w}d")))
        for fast, slow in [(5, 20), (10, 40), (20, 60)]:
            variants.append(("ma_trend", f"ma_trend_{fast}_{slow}", f"fast={fast};slow={slow}", lambda df, f=fast, s=slow: ma_trend_signal(df, f, s).assign(strategy_name=f"ma_trend_{f}_{s}")))
        for threshold in [0.01, 0.015, 0.02, 0.03]:
            variants.append(("support_pullback", f"support_pullback_{threshold:g}", f"threshold={threshold:g}", lambda df, t=threshold: support_pullback_trend_signal(df, pullback_threshold=t).assign(strategy_name=f"support_pullback_{t:g}")))

        for family, variant, parameter, func in variants:
            bt = run_daily_long_only_backtest(asset_price, func(asset_price), transaction_cost_bps=transaction_cost_bps)
            metrics = summarize_backtest(bt)
            rows.append(
                {
                    "asset_id": asset_id,
                    "strategy_family": family,
                    "strategy_variant": variant,
                    "parameter": parameter,
                    "cumulative_return": metrics["cumulative_return"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "active_ratio": metrics["active_ratio"],
                    "turnover": metrics["turnover"],
                    "beats_buy_hold_sharpe": metrics["sharpe"] > buy_hold["sharpe"],
                    "beats_buy_hold_return": metrics["cumulative_return"] > buy_hold["cumulative_return"],
                }
            )
    return pd.DataFrame(rows)


def _smoothness_note(group: pd.DataFrame) -> str:
    ranked = group.groupby("strategy_variant")["sharpe"].mean().sort_values(ascending=False)
    if len(ranked) < 2:
        return "insufficient_variants"
    gap = ranked.iloc[0] - ranked.iloc[1]
    return "single_parameter_spike" if gap > max(0.25, abs(ranked.iloc[1]) * 0.75) else "reasonably_smooth"


def main() -> None:
    price_path = PROJECT_ROOT / "data" / "raw" / "a_share" / "a_share_etf_daily.csv"
    if not price_path.exists():
        raise SystemExit("Run python scripts/data/fetch_a_share_data.py first.")
    report = run_parameter_sensitivity(pd.read_csv(price_path))
    out = PROJECT_ROOT / "data" / "reports" / "rule_parameter_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    best = (
        report.groupby(["strategy_family", "strategy_variant", "parameter"])
        .agg(avg_sharpe=("sharpe", "mean"), avg_return=("cumulative_return", "mean"))
        .reset_index()
        .sort_values(["strategy_family", "avg_sharpe"], ascending=[True, False])
        .groupby("strategy_family")
        .head(1)
    )
    notes = report.groupby("strategy_family").apply(_smoothness_note, include_groups=False).reset_index(name="parameter_shape")
    print("Best parameter by average Sharpe:")
    print(best.to_string(index=False))
    print("\nParameter smoothness:")
    print(notes.to_string(index=False))
    print(f"\nsaved parameter sensitivity to {out}")


if __name__ == "__main__":
    main()
