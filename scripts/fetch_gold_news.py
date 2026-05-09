#!/usr/bin/env python
"""Fetch configured raw news metadata for gold event extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.data.news_data import fetch_gold_news


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "data_sources.yaml").read_text())["gold"]
    news_config = config.get("news", {})
    output = PROJECT_ROOT / news_config.get("output", "data/raw/gold/news.csv")
    news = fetch_gold_news(news_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output, index=False)
    print(f"saved gold raw news to {output}")
    print(f"n rows: {len(news)}")
    if not news.empty:
        print(f"date range: {news['date'].min()} to {news['date'].max()}")
        print("source counts:")
        print(news["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
