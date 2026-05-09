import pandas as pd

from market_quant.features.alpha_style_fallback import build_alpha_style_fallback_features


def test_fallback_returns_are_correct():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "asset_id": ["A"] * 4,
            "close": [100.0, 110.0, 121.0, 120.0],
            "volume": [10, 20, 30, 40],
        }
    )
    out = build_alpha_style_fallback_features(df, windows=[1, 2])
    assert round(out.loc[1, "ret_1d"], 6) == 0.1
    assert round(out.loc[2, "ret_2d"], 6) == 0.21


def test_rolling_does_not_cross_asset_id():
    df = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=3)) * 2,
            "asset_id": ["A"] * 3 + ["B"] * 3,
            "close": [1.0, 2.0, 3.0, 100.0, 100.0, 100.0],
        }
    )
    out = build_alpha_style_fallback_features(df, windows=[2])
    first_b = out[out["asset_id"] == "B"].iloc[0]
    assert pd.isna(first_b["ret_2d"])
    assert pd.isna(first_b["rolling_high_2d"])


def test_volume_missing_does_not_fail():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "asset_id": ["A"] * 3, "close": [1, 2, 3]})
    out = build_alpha_style_fallback_features(df, windows=[2])
    assert "volume_zscore_2d" not in out.columns
