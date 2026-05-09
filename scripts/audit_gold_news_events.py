#!/usr/bin/env python
"""Audit raw news, LLM events, and model feature readiness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_dates(path: Path, date_col: str) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_csv(path)
    if date_col not in df:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[date_col], errors="coerce").dropna()


def _status_row(name: str, path: Path, date_col: str, reference_dates: pd.Series) -> dict:
    dates = _read_dates(path, date_col)
    row = {
        "dataset": name,
        "path": str(path),
        "exists": path.exists(),
        "n_rows": int(len(dates)),
        "min_date": dates.min() if not dates.empty else pd.NaT,
        "max_date": dates.max() if not dates.empty else pd.NaT,
        "overlap_with_reference_days": 0,
    }
    if not dates.empty and not reference_dates.empty:
        row["overlap_with_reference_days"] = int(pd.Series(dates.dt.normalize().unique()).isin(set(reference_dates.dt.normalize())).sum())
    return row


def main() -> None:
    data_cfg = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    llm_cfg = yaml.safe_load((PROJECT_ROOT / "config" / "llm_messages.yaml").read_text())
    raw_news_path = PROJECT_ROOT / data_cfg.get("news", {}).get("output", "data/raw/gold/news.csv")
    events_path = PROJECT_ROOT / llm_cfg.get("input", {}).get("output_events", "data/raw/gold/news_events.csv")
    features_path = PROJECT_ROOT / "data" / "features" / "gold" / "gold_model_dataset.csv"
    predictions_path = PROJECT_ROOT / "data" / "predictions" / "gold_model_walk_forward_predictions.csv"

    reference_dates = _read_dates(features_path, "date")
    rows = [
        _status_row("raw_news", raw_news_path, "published_at", reference_dates),
        _status_row("news_events", events_path, "published_at", reference_dates),
        _status_row("gold_model_dataset", features_path, "date", reference_dates),
        _status_row("gold_oos_predictions", predictions_path, "date", reference_dates),
    ]
    status = pd.DataFrame(rows)

    event_summary = pd.DataFrame()
    if events_path.exists():
        events = pd.read_csv(events_path)
        if not events.empty:
            event_summary = (
                events.groupby(["event_type", "gold_direction"])
                .size()
                .reset_index(name="n_events")
                .sort_values("n_events", ascending=False)
            )

    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "gold_news_data_status.csv"
    event_summary_path = report_dir / "gold_news_event_summary.csv"
    status.to_csv(status_path, index=False)
    event_summary.to_csv(event_summary_path, index=False)

    print("Gold news data status:")
    print(status.to_string(index=False))
    if not event_summary.empty:
        print("\nEvent summary:")
        print(event_summary.to_string(index=False))
    print(f"\nsaved news status to {status_path}")
    print(f"saved event summary to {event_summary_path}")


if __name__ == "__main__":
    main()
