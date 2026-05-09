"""Benchmark report assembly helpers."""

from __future__ import annotations

import pandas as pd


def compare_metric_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)
