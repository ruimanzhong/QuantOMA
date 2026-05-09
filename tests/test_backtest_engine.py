import pandas as pd
import pytest
import inspect

from market_quant.backtest.engine import run_daily_long_only_backtest
import market_quant.backtest.engine as engine_module


def make_price(asset="A"):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "asset_id": [asset] * 4,
            "close": [100.0, 110.0, 121.0, 121.0],
        }
    )


def make_signal(asset="A"):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "asset_id": [asset] * 4,
            "strategy_name": ["s"] * 4,
            "raw_signal": [1.0, 1.0, 0.0, 0.0],
            "position": [1.0, 1.0, 0.0, 0.0],
        }
    )


def test_strategy_return_uses_position_shift():
    out = run_daily_long_only_backtest(make_price(), make_signal(), transaction_cost_bps=0)
    assert out.loc[0, "executed_position"] == 0.0
    assert out.loc[1, "executed_position"] == 1.0
    assert round(out.loc[1, "strategy_return"], 6) == 0.1
    assert out.loc[3, "executed_position"] == 0.0


def test_pct_change_fill_method_none_and_price_nan_raises():
    price = make_price()
    price.loc[1, "close"] = None
    with pytest.raises(ValueError, match="NaN"):
        run_daily_long_only_backtest(price, make_signal())


def test_transaction_cost_correct():
    out = run_daily_long_only_backtest(make_price(), make_signal(), transaction_cost_bps=10)
    assert out.loc[0, "strategy_return"] == 0.0
    assert round(out.loc[1, "strategy_return"], 6) == 0.099
    assert round(out.loc[3, "strategy_return"], 6) == -0.001


def test_multi_asset_grouping():
    price = pd.concat([make_price("A"), make_price("B")], ignore_index=True)
    signal = pd.concat([make_signal("A"), make_signal("B")], ignore_index=True)
    out = run_daily_long_only_backtest(price, signal, transaction_cost_bps=0)
    assert out.groupby("asset_id")["executed_position"].first().tolist() == [0.0, 0.0]
    assert out.groupby("asset_id")["asset_return"].nth(1).round(6).tolist() == [0.1, 0.1]


def test_multi_asset_position_shift_does_not_bleed_between_assets():
    price = pd.concat([make_price("A"), make_price("B")], ignore_index=True)
    signal_a = make_signal("A")
    signal_b = make_signal("B")
    signal_b["position"] = [0.0, 0.0, 1.0, 1.0]
    signal = pd.concat([signal_a, signal_b], ignore_index=True)
    out = run_daily_long_only_backtest(price, signal, transaction_cost_bps=0)
    b = out[out["asset_id"] == "B"].reset_index(drop=True)
    assert b["executed_position"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_price_middle_nan_raises_value_error():
    price = make_price()
    price.loc[2, "close"] = None
    with pytest.raises(ValueError, match="no forward fill"):
        run_daily_long_only_backtest(price, make_signal())


def test_engine_explicitly_disables_pct_change_forward_fill():
    source = inspect.getsource(engine_module.run_daily_long_only_backtest)
    assert "pct_change(fill_method=None)" in source


def test_transaction_cost_charged_on_position_changes_only():
    price = make_price()
    signal = make_signal()
    signal["position"] = [1.0, 0.0, 1.0, 1.0]
    out = run_daily_long_only_backtest(price, signal, transaction_cost_bps=10)
    assert out.loc[0, "strategy_return"] == 0.0
    assert round(out.loc[1, "strategy_return"], 6) == 0.099
    assert round(out.loc[2, "strategy_return"], 6) == -0.001
    assert round(out.loc[3, "strategy_return"], 6) == -0.001
