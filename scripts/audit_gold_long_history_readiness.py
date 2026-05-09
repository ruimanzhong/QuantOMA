#!/usr/bin/env python
"""Audit whether gold research inputs are ready for a 2008-start study."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.phase2_local_data import load_gold_local_series


TARGET_START_DATE = pd.Timestamp("2008-01-01")


def _resolve_path(source_root: Path, relative_path: str) -> Path:
    return source_root / relative_path


def _source_root(config: dict) -> Path:
    root = Path(config.get("source_root", ".")).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def audit_configured_gold_sources(config: dict) -> pd.DataFrame:
    source_root = _source_root(config)
    rows = []
    for source_name, source_cfg in config.get("inputs", {}).items():
        path = _resolve_path(source_root, source_cfg["path"])
        row = {
            "source": source_name,
            "path": str(path),
            "exists": path.exists(),
            "min_date": pd.NaT,
            "max_date": pd.NaT,
            "n_rows": 0,
            "ready_for_2008": False,
            "blocking_issue": "",
        }
        if not path.exists():
            row["blocking_issue"] = "missing_file"
            rows.append(row)
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns:
            row["blocking_issue"] = "missing_date_column"
            rows.append(row)
            continue
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        row["n_rows"] = len(df)
        if dates.empty:
            row["blocking_issue"] = "no_valid_dates"
        else:
            row["min_date"] = dates.min()
            row["max_date"] = dates.max()
            row["ready_for_2008"] = bool(row["min_date"] <= TARGET_START_DATE)
            if not row["ready_for_2008"]:
                row["blocking_issue"] = f"starts_after_{TARGET_START_DATE.date()}"
        rows.append(row)
    return pd.DataFrame(rows)


def audit_unified_gold_series(config: dict) -> pd.DataFrame:
    unrestricted = dict(config)
    unrestricted["start_date"] = None
    series = load_gold_local_series(unrestricted, PROJECT_ROOT)
    grouped = (
        series.groupby("series_id")
        .agg(min_date=("date", "min"), max_date=("date", "max"), n_rows=("date", "size"))
        .reset_index()
        .sort_values(["min_date", "series_id"])
    )
    grouped["ready_for_2008"] = grouped["min_date"] <= TARGET_START_DATE
    return grouped


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    source_audit = audit_configured_gold_sources(config)
    series_audit = audit_unified_gold_series(config)

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    source_path = report_dir / "gold_long_history_source_audit.csv"
    series_path = report_dir / "gold_long_history_series_audit.csv"
    source_audit.to_csv(source_path, index=False)
    series_audit.to_csv(series_path, index=False)

    print("Configured gold source readiness for 2008-start study:")
    print(source_audit[["source", "exists", "min_date", "max_date", "n_rows", "ready_for_2008", "blocking_issue"]].to_string(index=False))
    print("\nEarliest unified series:")
    print(series_audit.head(20).to_string(index=False))
    print(f"\nsaved source audit to {source_path}")
    print(f"saved series audit to {series_path}")


if __name__ == "__main__":
    main()
