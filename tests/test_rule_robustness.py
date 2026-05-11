import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from market_quant.diagnostics.rule_robustness import annual_strategy_performance, subperiod_strategy_performance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    script_dirs = {
        "run_rule_parameter_sensitivity.py": "strategies",
        "run_transaction_cost_sensitivity.py": "backtests",
        "summarize_rule_robustness.py": "diagnostics",
    }
    path = PROJECT_ROOT / "scripts" / script_dirs[name] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_test_module"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_backtest_frame():
    dates = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"])
    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": ["510300.SH"] * 4,
            "strategy_name": ["momentum_20d"] * 4,
            "strategy_return": [0.01, -0.005, 0.02, 0.01],
            "executed_position": [1.0, 1.0, 0.0, 1.0],
        }
    )


def price_frame(n=90):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": ["510300.SH"] * n,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000] * n,
            "amount": [10000] * n,
            "source": ["test"] * n,
        }
    )


def test_annual_strategy_performance_splits_years():
    out = annual_strategy_performance(daily_backtest_frame())

    assert out["year"].tolist() == [2024, 2025]
    assert out["n_trading_days"].tolist() == [2, 2]
    assert out.loc[out["year"] == 2024, "cumulative_return"].iloc[0] == pytest.approx(1.01 * 0.995 - 1)


def test_subperiod_strategy_performance_splits_periods():
    periods = [
        {"name": "late_2024", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        {"name": "early_2025", "start_date": "2025-01-01", "end_date": "2025-12-31"},
    ]
    out = subperiod_strategy_performance(daily_backtest_frame(), periods)

    assert out["subperiod"].tolist() == ["late_2024", "early_2025"]
    assert out["n_trading_days"].tolist() == [2, 2]


def test_short_sample_does_not_crash():
    df = daily_backtest_frame().head(1)
    out = annual_strategy_performance(df)

    assert len(out) == 1
    assert pd.isna(out["sharpe"].iloc[0])


def test_parameter_sensitivity_output_fields_complete():
    module = load_script("run_rule_parameter_sensitivity.py")
    out = module.run_parameter_sensitivity(price_frame(), transaction_cost_bps=2.0)

    assert {
        "asset_id",
        "strategy_family",
        "strategy_variant",
        "parameter",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "active_ratio",
        "turnover",
        "beats_buy_hold_sharpe",
        "beats_buy_hold_return",
    }.issubset(out.columns)
    assert set(out["strategy_family"]) == {"momentum", "donchian_breakout", "ma_trend", "support_pullback"}


def test_transaction_cost_increase_does_not_improve_high_turnover_return():
    module = load_script("run_transaction_cost_sensitivity.py")
    out = module.run_cost_sensitivity(price_frame(), costs=[0, 10])
    rule_core = out[out["strategy_name"] == "rule_core_vote"]
    pivot = rule_core.pivot(index="asset_id", columns="transaction_cost_bps", values="cumulative_return")

    assert (pivot[10] <= pivot[0]).all()


def test_robustness_summary_missing_file_clear_error(tmp_path):
    module = load_script("summarize_rule_robustness.py")
    paths = {name: tmp_path / f"{name}.csv" for name in module.REQUIRED_REPORTS}
    for name, path in paths.items():
        if name != "annual":
            path.write_text("x\n1\n")

    with pytest.raises(FileNotFoundError, match="required robustness input missing"):
        module.build_robustness_summary(paths)
