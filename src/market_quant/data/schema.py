"""Canonical price schema utilities."""

from __future__ import annotations

import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "date",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
}


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a daily OHLCV price frame.

    This function intentionally does not forward-fill missing prices.
    """
    missing = REQUIRED_PRICE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"price frame missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = out["asset_id"].astype(str)

    duplicate_mask = out.duplicated(["asset_id", "date"], keep=False)
    if duplicate_mask.any():
        duplicates = out.loc[duplicate_mask, ["asset_id", "date"]].head(10)
        raise ValueError(f"duplicate asset_id/date rows found: {duplicates.to_dict('records')}")

    if out["close"].isna().any():
        bad_rows = out.loc[out["close"].isna(), ["asset_id", "date"]].head(10)
        raise ValueError(f"close contains missing values: {bad_rows.to_dict('records')}")

    return out.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _estimated_trading_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    if end_date < start_date:
        return 0
    return len(pd.bdate_range(start_date, end_date))


def check_date_coverage(
    df: pd.DataFrame,
    requested_start_date: str,
    requested_end_date: str,
    min_coverage_ratio: float = 0.7,
    max_start_gap_days: int = 365,
) -> pd.DataFrame:
    """Summarize whether each asset has plausible date coverage for a requested window."""
    price_df = validate_price_frame(df)
    requested_start = pd.Timestamp(requested_start_date)
    requested_end = pd.Timestamp(requested_end_date)
    expected_days = _estimated_trading_days(requested_start, requested_end)

    rows = []
    for asset_id, group in price_df.groupby("asset_id", sort=True):
        actual_start = group["date"].min()
        actual_end = group["date"].max()
        n_rows = int(len(group))
        start_gap_days = int((actual_start - requested_start).days)
        requested_row_ratio = n_rows / expected_days if expected_days else 0.0
        actual_expected_days = _estimated_trading_days(actual_start, min(actual_end, requested_end))
        actual_row_ratio = n_rows / actual_expected_days if actual_expected_days else 0.0
        actual_span_days = int((actual_end - actual_start).days)
        full_window_ok = start_gap_days <= max_start_gap_days and requested_row_ratio >= min_coverage_ratio
        listed_later_ok = (
            start_gap_days > max_start_gap_days
            and actual_span_days >= max_start_gap_days
            and actual_row_ratio >= min_coverage_ratio
        )
        coverage_ok = full_window_ok or listed_later_ok
        rows.append(
            {
                "asset_id": asset_id,
                "requested_start_date": requested_start,
                "requested_end_date": requested_end,
                "actual_start_date": actual_start,
                "actual_end_date": actual_end,
                "n_rows": n_rows,
                "start_gap_days": start_gap_days,
                "coverage_ok": bool(coverage_ok),
            }
        )

    return pd.DataFrame(rows)
