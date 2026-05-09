import pandas as pd
import pytest

from market_quant.diagnostics.alpha158_diagnostics import (
    alpha158_asset_coverage,
    compare_alpha158_with_price_universe,
    summarize_alpha158_features,
)


def alpha_frame():
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
            "asset_id": ["SH600000", "SH600000", "SH600000", "SZ000001"],
            "f1": [1.0, None, None, 4.0],
            "f2": [2.0, 3.0, 3.0, None],
        }
    )


def test_summarize_alpha158_features_fields_and_duplicates():
    out = summarize_alpha158_features(alpha_frame())

    assert {
        "n_rows",
        "n_assets",
        "first_date",
        "last_date",
        "n_feature_columns",
        "missing_rate_avg",
        "missing_rate_max",
        "duplicate_asset_date_count",
    }.issubset(out.columns)
    assert out.loc[0, "duplicate_asset_date_count"] == 1


def test_alpha158_asset_coverage_counts_missing_rate():
    dates = pd.bdate_range("2024-01-01", periods=101)
    df = pd.DataFrame({"date": dates, "asset_id": ["A"] * 101, "f1": [1.0] * 101, "f2": [None] + [1.0] * 100})
    out = alpha158_asset_coverage(df)

    assert out.loc[0, "n_rows"] == 101
    assert out.loc[0, "n_missing_any_feature"] == 1
    assert out.loc[0, "missing_rate_avg"] == pytest.approx(1 / 202)
    assert out.loc[0, "coverage_ok"] == True


def test_compare_alpha158_with_price_universe_alignment():
    alpha = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "asset_id": ["A", "A", "C"],
            "f1": [1.0, 2.0, 3.0],
        }
    )
    price = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["A", "A", "B"],
        }
    )
    out = compare_alpha158_with_price_universe(alpha, price).set_index("asset_id")

    assert out.loc["A", "in_price_data"] == True
    assert out.loc["A", "in_alpha158"] == True
    assert out.loc["A", "overlapping_rows"] == 2
    assert out.loc["A", "alpha_coverage_ratio_on_price_dates"] == 1.0
    assert out.loc["B", "in_price_data"] == True
    assert out.loc["B", "in_alpha158"] == False
    assert out.loc["C", "in_price_data"] == False
    assert out.loc["C", "in_alpha158"] == True


def test_empty_alpha_dataframe_raises_clear_value_error():
    with pytest.raises(ValueError, match="Alpha158 feature DataFrame is empty"):
        summarize_alpha158_features(pd.DataFrame())
