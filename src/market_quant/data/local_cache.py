"""Small local cache helpers for CSV/parquet research artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame to parquet when possible, falling back to CSV."""
    target = ensure_parent(path)
    if target.suffix.lower() == ".parquet":
        try:
            df.to_parquet(target, index=False)
            return target
        except ImportError:
            csv_target = target.with_suffix(".csv")
            df.to_csv(csv_target, index=False)
            return csv_target
    df.to_csv(target, index=False)
    return target
