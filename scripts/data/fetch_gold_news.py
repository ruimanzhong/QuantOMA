#!/usr/bin/env python
"""Fetch configured raw news metadata for gold event extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.workflows.gold_news import fetch_and_store_gold_news


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    result = fetch_and_store_gold_news(PROJECT_ROOT, config)
    print(f"saved gold raw news to {result.output_path}")
    print(f"new rows fetched: {result.fetched_rows}")
    print(f"total rows: {result.total_rows}")
    if not result.news.empty:
        print(f"date range: {result.news['date'].min()} to {result.news['date'].max()}")
        print("source counts:")
        print(result.news["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
