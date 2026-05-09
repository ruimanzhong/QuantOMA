"""Daily long-only backtest engine."""

from __future__ import annotations

import pandas as pd


def run_daily_long_only_backtest(
    price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    price_col: str = "close",
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Run a daily long-only backtest using prior-day signal execution."""
    required_price = {"date", "asset_id", price_col}
    required_signal = {"date", "asset_id", "raw_signal", "position"}
    missing_price = required_price.difference(price_df.columns)
    missing_signal = required_signal.difference(signal_df.columns)
    if missing_price:
        raise ValueError(f"price_df missing columns: {sorted(missing_price)}")
    if missing_signal:
        raise ValueError(f"signal_df missing columns: {sorted(missing_signal)}")

    price = price_df.copy()
    signals = signal_df.copy()
    price["date"] = pd.to_datetime(price["date"])
    signals["date"] = pd.to_datetime(signals["date"])

    if price[price_col].isna().any():
        bad_rows = price.loc[price[price_col].isna(), ["asset_id", "date"]].head(10)
        raise ValueError(f"price contains NaN values; no forward fill is allowed: {bad_rows.to_dict('records')}")

    price = price.sort_values(["asset_id", "date"]).reset_index(drop=True)
    signal_cols = ["date", "asset_id", "raw_signal", "position"]
    if "strategy_name" in signals.columns:
        signal_cols.append("strategy_name")
    signals = signals[signal_cols].sort_values(["asset_id", "date"]).reset_index(drop=True)

    merged = price[["date", "asset_id", price_col]].merge(
        signals,
        on=["date", "asset_id"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values(["asset_id", "date"]).reset_index(drop=True)
    grouped = merged.groupby("asset_id", group_keys=False)
    merged["asset_return"] = grouped[price_col].pct_change(fill_method=None)
    merged["executed_position"] = grouped["position"].shift(1).fillna(0.0)
    position_change = grouped["executed_position"].diff().abs().fillna(merged["executed_position"].abs())
    transaction_cost = position_change * (transaction_cost_bps / 10000.0)
    merged["strategy_return"] = merged["executed_position"] * merged["asset_return"].fillna(0.0) - transaction_cost
    merged = merged.rename(columns={price_col: "close"})
    output_cols = [
        "date",
        "asset_id",
        "close",
        "raw_signal",
        "position",
        "executed_position",
        "asset_return",
        "strategy_return",
    ]
    if "strategy_name" in merged.columns:
        output_cols.insert(3, "strategy_name")
    return merged[output_cols]
