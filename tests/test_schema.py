import pandas as pd
import pytest

from market_quant.data.schema import check_date_coverage, validate_price_frame


def price_frame():
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "asset_id": ["510300.SH", "510300.SH"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.1, 1.0],
            "volume": [100, 100],
            "amount": [1000, 1000],
            "source": ["test", "test"],
        }
    )


def test_duplicate_asset_date_raises():
    df = price_frame()
    df.loc[1, "date"] = "2024-01-02"
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_frame(df)


def test_close_missing_raises():
    df = price_frame()
    df.loc[0, "close"] = None
    with pytest.raises(ValueError, match="close contains missing"):
        validate_price_frame(df)


def test_date_sorting():
    out = validate_price_frame(price_frame())
    assert out["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def long_history_frame():
    dates = pd.bdate_range("2015-01-01", "2026-04-29")
    return pd.DataFrame(
        {
            "date": dates,
            "asset_id": ["510300.SH"] * len(dates),
            "open": [1.0] * len(dates),
            "high": [1.1] * len(dates),
            "low": [0.9] * len(dates),
            "close": [1.0] * len(dates),
            "volume": [100] * len(dates),
            "amount": [1000] * len(dates),
            "source": ["test"] * len(dates),
        }
    )


def test_check_date_coverage_flags_late_start():
    df = long_history_frame()
    df = df[df["date"] >= "2026-01-05"].head(76)

    report = check_date_coverage(df, "2015-01-01", "2026-04-30")

    assert report.loc[0, "start_gap_days"] > 365
    assert report.loc[0, "coverage_ok"] == False


def test_check_date_coverage_flags_76_rows_from_2026_for_2015_request():
    dates = pd.bdate_range("2026-01-05", periods=76)
    df = pd.DataFrame(
        {
            "date": dates,
            "asset_id": ["510300.SH"] * len(dates),
            "open": [1.0] * len(dates),
            "high": [1.1] * len(dates),
            "low": [0.9] * len(dates),
            "close": [1.0] * len(dates),
            "volume": [100] * len(dates),
            "amount": [1000] * len(dates),
            "source": ["baostock"] * len(dates),
        }
    )

    report = check_date_coverage(df, "2015-01-01", "2026-04-30")

    assert report.loc[0, "n_rows"] == 76
    assert report.loc[0, "coverage_ok"] == False


def test_check_date_coverage_accepts_normal_long_history():
    report = check_date_coverage(long_history_frame(), "2015-01-01", "2026-04-30")

    assert report.loc[0, "coverage_ok"] == True


def test_check_date_coverage_allows_later_listing_with_dense_history():
    dates = pd.bdate_range("2020-11-16", "2026-04-29")
    df = pd.DataFrame(
        {
            "date": dates,
            "asset_id": ["588000.SH"] * len(dates),
            "open": [1.0] * len(dates),
            "high": [1.1] * len(dates),
            "low": [0.9] * len(dates),
            "close": [1.0] * len(dates),
            "volume": [100] * len(dates),
            "amount": [1000] * len(dates),
            "source": ["efinance"] * len(dates),
        }
    )

    report = check_date_coverage(df, "2015-01-01", "2026-04-30")

    assert report.loc[0, "start_gap_days"] > 365
    assert report.loc[0, "coverage_ok"] == True
