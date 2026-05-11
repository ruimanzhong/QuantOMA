"""Gold news fetch and LLM event processing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_quant.data.news_data import NEWS_COLUMNS, fetch_gold_news, fetch_url_excerpt, merge_news_cache
from market_quant.llm.client import LLMClient
from market_quant.llm.news_events import EVENT_OUTPUT_COLUMNS, extract_news_events, merge_news_events, select_unprocessed_news


@dataclass(frozen=True)
class RawNewsFetchResult:
    output_path: Path
    fetched_rows: int
    total_rows: int
    news: pd.DataFrame


@dataclass(frozen=True)
class NewsEventProcessingResult:
    output_path: Path
    raw_rows: int
    new_event_rows: int
    total_event_rows: int
    mock_mode: bool
    max_concurrency: int
    events: pd.DataFrame


def fetch_and_store_gold_news(project_root: str | Path, gold_config: dict[str, Any]) -> RawNewsFetchResult:
    root = Path(project_root)
    news_config = gold_config.get("news", {})
    output_path = root / news_config.get("output", "data/raw/gold/news.csv")
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(columns=NEWS_COLUMNS)
    fetched = fetch_gold_news(news_config)
    news = merge_news_cache(existing, fetched)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output_path, index=False)
    return RawNewsFetchResult(output_path=output_path, fetched_rows=len(fetched), total_rows=len(news), news=news)


def process_gold_news_events(
    project_root: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    max_concurrency: int | None = None,
    mock: bool = False,
    min_relevance: int | None = None,
    recent_first: bool = True,
    process_all: bool = False,
    dry_run: bool = False,
    enrich_url_text: bool = False,
) -> NewsEventProcessingResult:
    root = Path(project_root)
    run_config = _prepare_llm_config(config, max_concurrency=max_concurrency, mock=mock)
    input_path = root / run_config.get("input", {}).get("raw_news", "data/raw/gold/news.csv")
    output_path = root / run_config.get("input", {}).get("output_events", "data/raw/gold/news_events.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Place raw news there before extracting events.")

    news = pd.read_csv(input_path)
    existing_events = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    news = _select_news_for_processing(
        news,
        existing_events,
        min_relevance=min_relevance,
        recent_first=recent_first,
        process_all=process_all,
        limit=limit,
    )
    if enrich_url_text:
        news = _with_transient_url_excerpts(news)

    client = LLMClient(run_config.get("llm", {}), project_root=root)
    new_events = extract_news_events(news, run_config, client)
    events = merge_news_events(existing_events, new_events)
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(output_path, index=False)
    return NewsEventProcessingResult(
        output_path=output_path,
        raw_rows=len(news),
        new_event_rows=len(new_events),
        total_event_rows=len(events),
        mock_mode=client.mock_mode,
        max_concurrency=int(run_config.get("llm", {}).get("max_concurrency", 1)),
        events=events,
    )


def _prepare_llm_config(config: dict[str, Any], *, max_concurrency: int | None, mock: bool) -> dict[str, Any]:
    run_config = dict(config)
    run_config["llm"] = dict(config.get("llm", {}))
    if mock:
        run_config["llm"]["mock_mode"] = True
    if max_concurrency is not None:
        run_config["llm"]["max_concurrency"] = max_concurrency
    return run_config


def _select_news_for_processing(
    news: pd.DataFrame,
    existing_events: pd.DataFrame,
    *,
    min_relevance: int | None,
    recent_first: bool,
    process_all: bool,
    limit: int | None,
) -> pd.DataFrame:
    out = news.copy()
    if min_relevance is not None and "relevance_score" in out:
        out = out[pd.to_numeric(out["relevance_score"], errors="coerce").fillna(0) >= min_relevance].copy()
    if not process_all:
        out = select_unprocessed_news(out, existing_events, recent_first=recent_first)
    elif recent_first and "published_at" in out:
        out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
        out = out.sort_values("published_at", ascending=False).copy()
    if limit is not None:
        out = out.head(limit).copy()
    return out.reset_index(drop=True)


def _with_transient_url_excerpts(news: pd.DataFrame) -> pd.DataFrame:
    if "url" not in news:
        return news
    enriched = []
    for _, row in news.iterrows():
        excerpt = ""
        try:
            excerpt = fetch_url_excerpt(str(row.get("url", "")))
        except Exception as exc:  # noqa: BLE001 - URL enrichment should be best effort.
            excerpt = f"URL excerpt unavailable: {type(exc).__name__}: {exc}"
        base_summary = "" if pd.isna(row.get("summary", "")) else str(row.get("summary", ""))
        enriched.append(f"{base_summary}\n\nURL excerpt:\n{excerpt}" if excerpt else base_summary)
    out = news.copy()
    out["summary"] = enriched
    return out
