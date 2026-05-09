"""Audits for gold model alignment, trading-day targets, and leakage boundaries."""

from __future__ import annotations

import pandas as pd


def audit_gold_dataset(
    dataset: pd.DataFrame,
    horizon_days: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return pass/fail checks for the gold model dataset."""
    if "date" in dataset.columns:
        df = dataset.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        df = dataset.copy().sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("gold dataset must have a DatetimeIndex or date column")

    rows = []
    rows.append(
        {
            "check_name": "unique_trading_dates",
            "passed": not df.index.has_duplicates,
            "details": f"duplicate_dates={int(df.index.duplicated().sum())}",
        }
    )
    required = {"primary_etf", "target_direction", "label_end_date"}
    missing = required.difference(df.columns)
    rows.append({"check_name": "required_columns", "passed": not missing, "details": f"missing={sorted(missing)}"})
    if missing:
        return pd.DataFrame(rows)

    label_end = pd.to_datetime(df["label_end_date"])
    label_positions = pd.Series(range(len(df)), index=df.index)
    mapped = label_end.map(label_positions)
    valid = mapped.notna()
    expected = pd.Series(range(len(df)), index=df.index) + horizon_days
    rows.append(
        {
            "check_name": "label_end_is_dataset_trading_date",
            "passed": bool((~valid).sum() <= horizon_days),
            "details": f"missing_label_end_in_index={int((~valid).sum())}",
        }
    )
    rows.append(
        {
            "check_name": "label_end_uses_row_horizon",
            "passed": bool((mapped[valid].astype(int) == expected[valid]).all()),
            "details": f"horizon_days={horizon_days}",
        }
    )
    feature_missing = [col for col in feature_columns if col not in df.columns]
    rows.append(
        {
            "check_name": "feature_columns_exist",
            "passed": not feature_missing,
            "details": f"missing_features={feature_missing[:10]}",
        }
    )
    suspicious = [col for col in feature_columns if "forward_return" in col or col in {"target_direction", "label_end_date"}]
    rows.append(
        {
            "check_name": "no_obvious_target_leakage_features",
            "passed": not suspicious,
            "details": f"suspicious_features={suspicious[:10]}",
        }
    )
    return pd.DataFrame(rows)


def audit_walk_forward_folds(dataset: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Audit fold train/test ordering and label purge boundaries from saved fold metrics."""
    df = dataset.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    label_end = pd.to_datetime(df["label_end_date"])
    rows = []
    for _, fold in fold_metrics.iterrows():
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        train_index = df.index[df.index <= train_end]
        max_label_end = label_end.loc[train_index].max()
        rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "train_end": train_end,
                "test_start": test_start,
                "max_train_label_end": max_label_end,
                "passed": bool(max_label_end < test_start),
            }
        )
    return pd.DataFrame(rows)
