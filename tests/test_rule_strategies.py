import pandas as pd
import pytest

from market_quant.strategies.breakout import donchian_breakout_20d
from market_quant.strategies.rule_core import dual_momentum_signal
from market_quant.strategies.support_pullback import support_pullback_trend
from market_quant.strategies.trend_following import momentum_20d
from market_quant.strategies.volume_confirmed import volume_confirmed_breakout


def make_price(n=80, close=None, volume=True):
    close = close or list(range(1, n + 1))
    data = {
        "date": pd.date_range("2024-01-01", periods=n),
        "asset_id": ["A"] * n,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "amount": [1000] * n,
        "source": ["test"] * n,
    }
    if volume:
        data["volume"] = [100] * n
    return pd.DataFrame(data)


def test_momentum_correct():
    signal = momentum_20d(make_price(25))
    assert signal.loc[19, "position"] == 0.0
    assert signal.loc[20, "position"] == 1.0


def test_dual_momentum_correct():
    signal = dual_momentum_signal(make_price(70), 20, 60)
    assert signal.loc[59, "position"] == 0.0
    assert signal.loc[60, "position"] == 0.0
    close = [100.0] * 70
    close[40] = 80.0
    close[60] = 120.0
    accelerating = make_price(70, close=close)
    signal = dual_momentum_signal(accelerating, 20, 60)
    assert signal.loc[60, "position"] == 1.0


def test_donchian_breakout_uses_shifted_rolling_high():
    close = [10.0] * 20 + [10.0, 10.1]
    df = make_price(22, close=close)
    signal = donchian_breakout_20d(df)
    assert signal.loc[20, "position"] == 0.0
    assert signal.loc[21, "position"] == 1.0


def test_support_pullback_logic():
    close = [100.0] * 60 + [101.0, 101.5, 101.0]
    df = make_price(63, close=close)
    signal = support_pullback_trend(df, ma_window=3, trend_window=60, pullback_threshold=0.02)
    assert signal.loc[62, "position"] == 1.0


def test_volume_confirmed_without_volume_warns_and_is_flat():
    df = make_price(25, volume=False)
    with pytest.warns(RuntimeWarning, match="requires volume"):
        signal = volume_confirmed_breakout(df)
    assert signal["position"].sum() == 0.0
