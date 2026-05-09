#!/usr/bin/env python
"""Diagnose gold strategy performance by calendar subperiod and price regime."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.diagnostics.gold_regime import (
    add_price_regime_labels,
    build_buy_hold_daily,
    prepare_gold_daily_strategy_frame,
    summarize_by_price_regime,
    summarize_by_subperiod,
)
from market_quant.gold.backtest import run_gold_long_flat_backtest
from market_quant.gold.layered import run_layered_gold_backtest


def _feature_source_status(feature_path: Path) -> pd.DataFrame:
    if not feature_path.exists():
        return pd.DataFrame(
            [
                {"source": "news", "available_in_model_features": False, "n_features": 0},
                {"source": "polymarket", "available_in_model_features": False, "n_features": 0},
            ]
        )
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    rows = []
    for source, prefixes in {
        "news": ("news_",),
        "polymarket": ("polymarket_",),
    }.items():
        if source == "polymarket":
            matched = [feature for feature in features if feature.startswith(prefixes) or "_probability_" in feature]
        else:
            matched = [feature for feature in features if feature.startswith(prefixes)]
        rows.append(
            {
                "source": source,
                "available_in_model_features": bool(matched),
                "n_features": len(matched),
                "features": ",".join(matched),
            }
        )
    return pd.DataFrame(rows)


def build_gold_regime_diagnostics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = PROJECT_ROOT / "data" / "features" / "gold" / "gold_model_dataset.csv"
    strategy_frames = [build_buy_hold_daily(predictions)]

    default_daily, _ = run_gold_long_flat_backtest(
        predictions,
        threshold=0.55,
        transaction_cost_bps=2.0,
        strategy_type="threshold",
    )
    strategy_frames.append(prepare_gold_daily_strategy_frame(default_daily, "threshold_0.55"))

    overlay_daily, _ = run_gold_long_flat_backtest(
        predictions,
        lower_threshold=0.50,
        defensive_position=0.3,
        transaction_cost_bps=2.0,
        strategy_type="defensive_overlay",
    )
    strategy_frames.append(prepare_gold_daily_strategy_frame(overlay_daily, "overlay_low_0.50_pos_0.3"))

    if feature_path.exists():
        features = pd.read_csv(feature_path, parse_dates=["date"])
        layered_daily, _ = run_layered_gold_backtest(predictions, features, transaction_cost_bps=2.0)
        strategy_frames.append(prepare_gold_daily_strategy_frame(layered_daily, "layered_regime_exposure"))

    daily = pd.concat(strategy_frames, ignore_index=True)
    daily_with_regime = add_price_regime_labels(daily)
    return summarize_by_subperiod(daily), summarize_by_price_regime(daily_with_regime)


def main() -> None:
    pred_path = PROJECT_ROOT / "data" / "predictions" / "gold_model_walk_forward_predictions.csv"
    feature_path = PROJECT_ROOT / "data" / "features" / "gold" / "gold_model_features.txt"
    if not pred_path.exists():
        raise SystemExit("Run python scripts/train_gold_model.py first.")

    predictions = pd.read_csv(pred_path, parse_dates=["date"]).sort_values("date")
    subperiod, regime = build_gold_regime_diagnostics(predictions)
    source_status = _feature_source_status(feature_path)

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    subperiod_path = report_dir / "gold_regime_subperiod_performance.csv"
    regime_path = report_dir / "gold_price_regime_performance.csv"
    source_path = report_dir / "gold_external_feature_status.csv"
    subperiod.to_csv(subperiod_path, index=False)
    regime.to_csv(regime_path, index=False)
    source_status.to_csv(source_path, index=False)

    print("Gold OOS prediction window:")
    print(f"{predictions['date'].min().date()} to {predictions['date'].max().date()}, rows={len(predictions)}")
    print("\nSubperiod performance:")
    print(
        subperiod[
            [
                "subperiod",
                "strategy_name",
                "n_trading_days",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
                "active_ratio",
                "turnover",
            ]
        ].to_string(index=False)
    )
    print("\nPrice-regime performance:")
    print(
        regime[
            [
                "price_regime",
                "strategy_name",
                "n_trading_days",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
                "active_ratio",
                "turnover",
            ]
        ].to_string(index=False)
    )
    print("\nExternal feature status:")
    print(source_status[["source", "available_in_model_features", "n_features"]].to_string(index=False))
    print(f"\nsaved subperiod diagnostics to {subperiod_path}")
    print(f"saved price-regime diagnostics to {regime_path}")
    print(f"saved external feature status to {source_path}")


if __name__ == "__main__":
    main()
