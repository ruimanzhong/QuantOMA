import pandas as pd
import pytest

from market_quant.diagnostics.rule_baseline_evaluation import (
    best_strategy_by_asset,
    evaluate_rule_vs_buy_hold,
    strategy_stability_summary,
)


def comparison_frame():
    return pd.DataFrame(
        [
            {
                "asset_id": "510300.SH",
                "strategy_name": "buy_and_hold",
                "cumulative_return": 1.0,
                "sharpe": 0.5,
                "max_drawdown": -0.5,
                "active_ratio": 1.0,
                "turnover": 0.0,
            },
            {
                "asset_id": "510300.SH",
                "strategy_name": "rule_a",
                "cumulative_return": 1.2,
                "sharpe": 0.7,
                "max_drawdown": -0.3,
                "active_ratio": 0.6,
                "turnover": 0.1,
            },
            {
                "asset_id": "510300.SH",
                "strategy_name": "rule_b",
                "cumulative_return": 0.8,
                "sharpe": 0.4,
                "max_drawdown": -0.2,
                "active_ratio": 0.2,
                "turnover": 0.05,
            },
            {
                "asset_id": "510500.SH",
                "strategy_name": "buy_and_hold",
                "cumulative_return": 0.5,
                "sharpe": 0.3,
                "max_drawdown": -0.4,
                "active_ratio": 1.0,
                "turnover": 0.0,
            },
            {
                "asset_id": "510500.SH",
                "strategy_name": "rule_a",
                "cumulative_return": -0.1,
                "sharpe": 0.1,
                "max_drawdown": -0.2,
                "active_ratio": 0.5,
                "turnover": 0.2,
            },
            {
                "asset_id": "510500.SH",
                "strategy_name": "rule_b",
                "cumulative_return": 0.7,
                "sharpe": 0.6,
                "max_drawdown": -0.6,
                "active_ratio": 0.8,
                "turnover": 0.15,
            },
        ]
    )


def test_evaluate_rule_vs_buy_hold_calculates_diffs_and_flags():
    out = evaluate_rule_vs_buy_hold(comparison_frame())
    row = out[(out["asset_id"] == "510300.SH") & (out["strategy_name"] == "rule_a")].iloc[0]

    assert row["excess_return_vs_buy_hold"] == pytest.approx(0.2)
    assert row["sharpe_diff_vs_buy_hold"] == pytest.approx(0.2)
    assert row["drawdown_reduction"] == pytest.approx(0.2)
    assert row["beats_buy_hold_return"] == True
    assert row["beats_buy_hold_sharpe"] == True
    assert row["reduces_drawdown"] == True


def test_best_strategy_by_asset_selects_expected_strategies():
    out = best_strategy_by_asset(comparison_frame())
    row_300 = out[out["asset_id"] == "510300.SH"].iloc[0]
    row_500 = out[out["asset_id"] == "510500.SH"].iloc[0]

    assert row_300["best_by_return"] == "rule_a"
    assert row_300["best_by_sharpe"] == "rule_a"
    assert row_300["lowest_drawdown_strategy"] == "rule_b"
    assert row_500["best_by_return"] == "rule_b"
    assert row_500["best_by_sharpe"] == "rule_b"
    assert row_500["lowest_drawdown_strategy"] == "rule_a"


def test_strategy_stability_summary_counts_and_scores():
    out = strategy_stability_summary(comparison_frame())
    row_a = out[out["strategy_name"] == "rule_a"].iloc[0]
    row_b = out[out["strategy_name"] == "rule_b"].iloc[0]

    assert row_a["n_assets"] == 2
    assert row_a["avg_sharpe"] == pytest.approx(0.4)
    assert row_a["n_assets_beating_buy_hold_return"] == 1
    assert row_a["n_assets_beating_buy_hold_sharpe"] == 1
    assert row_a["n_assets_reducing_drawdown"] == 2
    assert row_a["stability_score"] == 3
    assert row_b["n_assets_beating_buy_hold_return"] == 1
    assert row_b["n_assets_beating_buy_hold_sharpe"] == 1
    assert row_b["n_assets_reducing_drawdown"] == 1
    assert row_b["stability_score"] == 4


def test_missing_buy_and_hold_raises_clear_error():
    df = comparison_frame()
    df = df[~((df["asset_id"] == "510500.SH") & (df["strategy_name"] == "buy_and_hold"))]

    with pytest.raises(ValueError, match="missing buy_and_hold benchmark for asset_id: 510500.SH"):
        evaluate_rule_vs_buy_hold(df)


def test_evaluation_output_fields_complete():
    out = evaluate_rule_vs_buy_hold(comparison_frame())
    assert list(out.columns) == [
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
