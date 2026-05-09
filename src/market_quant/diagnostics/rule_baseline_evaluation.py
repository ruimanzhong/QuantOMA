"""Diagnostics for Phase 1 rule baseline evaluation."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "asset_id",
    "strategy_name",
    "cumulative_return",
    "sharpe",
    "max_drawdown",
    "active_ratio",
    "turnover",
}


def _validate_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df missing required columns: {sorted(missing)}")

    df = comparison_df.copy()
    missing_buy_hold = []
    for asset_id, group in df.groupby("asset_id", sort=True):
        if "buy_and_hold" not in set(group["strategy_name"]):
            missing_buy_hold.append(str(asset_id))
    if missing_buy_hold:
        raise ValueError(f"missing buy_and_hold benchmark for asset_id: {', '.join(missing_buy_hold)}")
    return df


def evaluate_rule_vs_buy_hold(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Compare each strategy row with the asset's buy-and-hold benchmark."""
    df = _validate_comparison(comparison_df)
    buy_hold = (
        df[df["strategy_name"] == "buy_and_hold"]
        .set_index("asset_id")[["cumulative_return", "sharpe", "max_drawdown"]]
        .rename(
            columns={
                "cumulative_return": "buy_hold_cumulative_return",
                "sharpe": "buy_hold_sharpe",
                "max_drawdown": "buy_hold_max_drawdown",
            }
        )
    )
    out = df.merge(buy_hold, left_on="asset_id", right_index=True, how="left", validate="many_to_one")
    out = out.rename(
        columns={
            "cumulative_return": "strategy_cumulative_return",
            "sharpe": "strategy_sharpe",
            "max_drawdown": "strategy_max_drawdown",
        }
    )
    out["excess_return_vs_buy_hold"] = out["strategy_cumulative_return"] - out["buy_hold_cumulative_return"]
    out["sharpe_diff_vs_buy_hold"] = out["strategy_sharpe"] - out["buy_hold_sharpe"]
    out["drawdown_reduction"] = out["strategy_max_drawdown"] - out["buy_hold_max_drawdown"]
    out["beats_buy_hold_return"] = out["excess_return_vs_buy_hold"] > 0
    out["beats_buy_hold_sharpe"] = out["sharpe_diff_vs_buy_hold"] > 0
    out["reduces_drawdown"] = out["drawdown_reduction"] > 0

    columns = [
        "asset_id",
        "strategy_name",
        "strategy_cumulative_return",
        "buy_hold_cumulative_return",
        "excess_return_vs_buy_hold",
        "strategy_sharpe",
        "buy_hold_sharpe",
        "sharpe_diff_vs_buy_hold",
        "strategy_max_drawdown",
        "buy_hold_max_drawdown",
        "drawdown_reduction",
        "active_ratio",
        "turnover",
        "beats_buy_hold_return",
        "beats_buy_hold_sharpe",
        "reduces_drawdown",
    ]
    return out[columns].sort_values(["asset_id", "strategy_name"]).reset_index(drop=True)


def _rank_position(group: pd.DataFrame, strategy_name: str, column: str, ascending: bool) -> int:
    ranked = group.sort_values(column, ascending=ascending, kind="mergesort").reset_index(drop=True)
    matches = ranked.index[ranked["strategy_name"] == strategy_name].tolist()
    return int(matches[0] + 1)


def best_strategy_by_asset(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Select the leading strategies by return, Sharpe, and drawdown for each asset."""
    df = _validate_comparison(comparison_df)
    rows = []
    for asset_id, group in df.groupby("asset_id", sort=True):
        best_return = group.sort_values("cumulative_return", ascending=False).iloc[0]
        best_sharpe = group.sort_values("sharpe", ascending=False).iloc[0]
        lowest_drawdown = group.sort_values("max_drawdown", ascending=False).iloc[0]
        buy_hold = group[group["strategy_name"] == "buy_and_hold"].iloc[0]
        notes = []
        if best_return["strategy_name"] == "buy_and_hold":
            notes.append("buy_and_hold_best_return")
        if best_sharpe["strategy_name"] == "buy_and_hold":
            notes.append("buy_and_hold_best_sharpe")
        if lowest_drawdown["strategy_name"] != "buy_and_hold":
            notes.append("rule_reduces_drawdown")
        rows.append(
            {
                "asset_id": asset_id,
                "best_by_return": best_return["strategy_name"],
                "best_by_return_value": best_return["cumulative_return"],
                "best_by_sharpe": best_sharpe["strategy_name"],
                "best_by_sharpe_value": best_sharpe["sharpe"],
                "lowest_drawdown_strategy": lowest_drawdown["strategy_name"],
                "lowest_drawdown_value": lowest_drawdown["max_drawdown"],
                "buy_hold_rank_by_sharpe": _rank_position(group, "buy_and_hold", "sharpe", ascending=False),
                "buy_hold_rank_by_return": _rank_position(group, "buy_and_hold", "cumulative_return", ascending=False),
                "notes": ";".join(notes),
            }
        )
        _ = buy_hold
    return pd.DataFrame(rows)


def strategy_stability_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-asset stability for each strategy."""
    evaluated = evaluate_rule_vs_buy_hold(comparison_df)
    rows = []
    for strategy_name, group in evaluated.groupby("strategy_name", sort=True):
        n_assets = int(group["asset_id"].nunique())
        n_positive = int((group["strategy_cumulative_return"] > 0).sum())
        n_negative = n_assets - n_positive
        n_beating_return = int(group["beats_buy_hold_return"].sum())
        n_beating_sharpe = int(group["beats_buy_hold_sharpe"].sum())
        n_reducing_drawdown = int(group["reduces_drawdown"].sum())
        rows.append(
            {
                "strategy_name": strategy_name,
                "n_assets": n_assets,
                "avg_sharpe": group["strategy_sharpe"].mean(),
                "median_sharpe": group["strategy_sharpe"].median(),
                "avg_cumulative_return": group["strategy_cumulative_return"].mean(),
                "median_cumulative_return": group["strategy_cumulative_return"].median(),
                "avg_max_drawdown": group["strategy_max_drawdown"].mean(),
                "n_assets_positive_return": n_positive,
                "n_assets_beating_buy_hold_return": n_beating_return,
                "n_assets_beating_buy_hold_sharpe": n_beating_sharpe,
                "n_assets_reducing_drawdown": n_reducing_drawdown,
                "stability_score": n_beating_sharpe + n_reducing_drawdown + n_positive - n_negative,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stability_score", "avg_sharpe"], ascending=[False, False]
    ).reset_index(drop=True)
