#!/usr/bin/env python
"""Extract structured gold news events with an LLM client."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.llm.client import LLMClient
from market_quant.llm.news_events import extract_news_events
from market_quant.data.news_data import fetch_url_excerpt


def parse_args() -> object:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N raw news rows.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Override LLM request concurrency.")
    parser.add_argument("--mock", action="store_true", help="Force mock extraction regardless of config.")
    parser.add_argument("--min-relevance", type=int, default=None, help="Process only rows with relevance_score >= N.")
    parser.add_argument("--recent-first", action="store_true", help="Process newest raw news rows first.")
    parser.add_argument("--enrich-url-text", action="store_true", help="Fetch transient URL excerpts for LLM input without storing article text in event output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "llm_messages.yaml").read_text())
    if args.mock:
        config.setdefault("llm", {})["mock_mode"] = True
    if args.max_concurrency is not None:
        config.setdefault("llm", {})["max_concurrency"] = args.max_concurrency
    input_path = PROJECT_ROOT / config.get("input", {}).get("raw_news", "data/raw/gold/news.csv")
    output_path = PROJECT_ROOT / config.get("input", {}).get("output_events", "data/raw/gold/news_events.csv")
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}. Place raw news there before extracting events.")

    news = pd.read_csv(input_path)
    if args.min_relevance is not None and "relevance_score" in news:
        news = news[pd.to_numeric(news["relevance_score"], errors="coerce").fillna(0) >= args.min_relevance].copy()
    if args.recent_first and "published_at" in news:
        news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce")
        news = news.sort_values("published_at", ascending=False).copy()
    if args.limit is not None:
        news = news.head(args.limit).copy()
    if args.enrich_url_text and "url" in news:
        enriched = []
        for _, row in news.iterrows():
            excerpt = ""
            try:
                excerpt = fetch_url_excerpt(str(row.get("url", "")))
            except Exception as exc:  # noqa: BLE001 - URL enrichment should be best effort.
                excerpt = f"URL excerpt unavailable: {type(exc).__name__}: {exc}"
            base_summary = "" if pd.isna(row.get("summary", "")) else str(row.get("summary", ""))
            enriched.append(f"{base_summary}\n\nURL excerpt:\n{excerpt}" if excerpt else base_summary)
        news = news.copy()
        news["summary"] = enriched
    client = LLMClient(config.get("llm", {}), project_root=PROJECT_ROOT)
    events = extract_news_events(news, config, client)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False)

    print(f"saved gold news events to {output_path}")
    print(f"n raw news rows: {len(news)}")
    print(f"n extracted events: {len(events)}")
    print(f"mock mode: {client.mock_mode}")
    print(f"max concurrency: {config.get('llm', {}).get('max_concurrency', 1)}")
    if "extraction_error" in events and events["extraction_error"].astype(str).str.len().gt(0).any():
        print("extraction errors:")
        print(events.loc[events["extraction_error"].astype(str).str.len().gt(0), "extraction_error"].value_counts().head(10).to_string())
    if not events.empty:
        print("event type distribution:")
        print(events["event_type"].value_counts().to_string())
        print("gold direction distribution:")
        print(events["gold_direction"].value_counts().to_string())


if __name__ == "__main__":
    main()
