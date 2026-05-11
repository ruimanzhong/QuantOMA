#!/usr/bin/env python
"""Extract structured gold news events with an LLM client."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.workflows.gold_news import process_gold_news_events


def parse_args() -> object:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N raw news rows.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Override LLM request concurrency.")
    parser.add_argument("--mock", action="store_true", help="Force mock extraction regardless of config.")
    parser.add_argument("--min-relevance", type=int, default=None, help="Process only rows with relevance_score >= N.")
    parser.add_argument("--recent-first", action="store_true", help="Process newest raw news rows first.")
    parser.add_argument("--all", action="store_true", help="Process all raw news rows instead of only unprocessed rows.")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and print summary without writing output_events.")
    parser.add_argument("--enrich-url-text", action="store_true", help="Fetch transient URL excerpts for LLM input without storing article text in event output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "llm_messages.yaml").read_text())
    result = process_gold_news_events(
        PROJECT_ROOT,
        config,
        limit=args.limit,
        max_concurrency=args.max_concurrency,
        mock=args.mock,
        min_relevance=args.min_relevance,
        recent_first=args.recent_first,
        process_all=args.all,
        dry_run=args.dry_run,
        enrich_url_text=args.enrich_url_text,
    )

    action = "would save" if args.dry_run else "saved"
    events = result.events
    print(f"{action} gold news events to {result.output_path}")
    print(f"n raw news rows: {result.raw_rows}")
    print(f"n newly extracted events: {result.new_event_rows}")
    print(f"n total events: {result.total_event_rows}")
    print(f"mock mode: {result.mock_mode}")
    print(f"max concurrency: {result.max_concurrency}")
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
