import pandas as pd
import pytest

from market_quant.data.series_schema import normalize_daily_series_frame, validate_daily_series_frame
from market_quant.diagnostics.series_coverage import summarize_daily_series_coverage


def test_validate_daily_series_frame_sorts_and_checks():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "series_id": ["xauusd", "xauusd"],
            "value": [2.0, 1.0],
            "source": ["test", "test"],
        }
    )
    out = validate_daily_series_frame(df)
    assert out["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def test_validate_daily_series_frame_duplicate_raises():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "series_id": ["vix", "vix"],
            "value": [1.0, 1.1],
            "source": ["test", "test"],
        }
    )
    with pytest.raises(ValueError, match="duplicate series_id/date rows found"):
        validate_daily_series_frame(df)


def test_normalize_daily_series_frame_from_asset_id():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "asset_id": ["xauusd"],
            "close": [1.0],
            "source": ["test"],
        }
    )
    out = normalize_daily_series_frame(df, value_name="close")
    assert list(out.columns[:4]) == ["date", "series_id", "value", "source"]


def test_series_coverage_summary_fields():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120),
            "series_id": ["dxy"] * 120,
            "value": [1.0] * 120,
            "source": ["test"] * 120,
        }
    )
    out = summarize_daily_series_coverage(df)
    assert list(out.columns) == [
        "series_id",
        "source",
        "first_date",
        "last_date",
        "n_rows",
        "missing_rate",
        "duplicate_series_date_count",
        "coverage_ok",
    ]
    assert out.loc[0, "coverage_ok"] == True
