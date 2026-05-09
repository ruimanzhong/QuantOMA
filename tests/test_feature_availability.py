import pandas as pd

from market_quant.features.availability import apply_feature_availability_lag


def make_features():
    return pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=3)) * 2,
            "asset_id": ["A"] * 3 + ["B"] * 3,
            "alpha": [1, 2, 3, 10, 20, 30],
            "beta": [4, 5, 6, 40, 50, 60],
            "metadata": ["x", "x", "x", "y", "y", "y"],
        }
    )


def test_apply_feature_availability_lag_shifts_by_asset():
    out = apply_feature_availability_lag(make_features(), lag_rows=1)
    assert pd.isna(out.loc[0, "alpha"])
    assert out.loc[1, "alpha"] == 1
    assert pd.isna(out.loc[3, "alpha"])
    assert out.loc[4, "alpha"] == 10


def test_date_and_asset_id_do_not_shift():
    out = apply_feature_availability_lag(make_features(), lag_rows=1)
    assert out["date"].tolist() == make_features()["date"].tolist()
    assert out["asset_id"].tolist() == make_features()["asset_id"].tolist()


def test_exclude_columns_do_not_shift():
    out = apply_feature_availability_lag(make_features(), lag_rows=1, exclude_columns=["metadata"])
    assert out["metadata"].tolist() == ["x", "x", "x", "y", "y", "y"]


def test_lag_rows_zero_leaves_values_unchanged():
    df = make_features()
    out = apply_feature_availability_lag(df, lag_rows=0)
    pd.testing.assert_frame_equal(out, df)


def test_different_asset_ids_do_not_bleed_into_each_other():
    out = apply_feature_availability_lag(make_features(), lag_rows=1)
    first_b = out[out["asset_id"] == "B"].iloc[0]
    assert pd.isna(first_b["alpha"])
    assert pd.isna(first_b["beta"])
