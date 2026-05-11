"""Convert raw news rows into structured gold event rows."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from market_quant.llm.client import LLMClient
from market_quant.llm.event_schema import GoldNewsEvent
from market_quant.llm.prompts import build_event_messages


RAW_NEWS_COLUMNS = ["published_at", "source", "title", "summary", "url"]
EVENT_OUTPUT_COLUMNS = [
    "news_id",
    "published_at",
    "date",
    "source",
    "event_type",
    "gold_direction",
    "confidence",
    "surprise",
    "time_horizon",
    "affected_channels",
    "llm_summary",
    "rationale",
    "extraction_error",
]


def mock_extract_event(title: str, summary: str, allowed_event_types: set[str] | None = None) -> GoldNewsEvent:
    text = f"{title} {summary}".lower()
    if any(word in text for word in ["fed", "rate cut", "dovish", "real yield"]):
        data = {
            "event_type": "fed",
            "gold_direction": "bullish",
            "confidence": 0.70,
            "surprise": 0.40,
            "time_horizon": "1w",
            "affected_channels": ["real_yield", "usd"],
            "llm_summary": "Dovish policy language points to lower real-yield pressure.",
            "rationale": "Lower real yields and a softer USD are generally supportive for gold.",
        }
    elif any(word in text for word in ["war", "iran", "israel", "russia", "ukraine", "geopolitical"]):
        data = {
            "event_type": "geopolitics",
            "gold_direction": "bullish",
            "confidence": 0.68,
            "surprise": 0.50,
            "time_horizon": "1w",
            "affected_channels": ["safe_haven"],
            "llm_summary": "Geopolitical risk headlines may increase safe-haven demand.",
            "rationale": "Safe-haven demand can support gold during geopolitical stress.",
        }
    elif any(word in text for word in ["dollar", "usd", "dxy"]):
        data = {
            "event_type": "usd",
            "gold_direction": "bearish",
            "confidence": 0.60,
            "surprise": 0.30,
            "time_horizon": "1d",
            "affected_channels": ["usd"],
            "llm_summary": "Dollar strength creates a near-term headwind for gold.",
            "rationale": "A stronger USD often pressures USD-denominated gold.",
        }
    else:
        data = {
            "event_type": "other",
            "gold_direction": "neutral",
            "confidence": 0.50,
            "surprise": 0.10,
            "time_horizon": "1d",
            "affected_channels": [],
            "llm_summary": "No clear gold-specific channel was identified.",
            "rationale": "The item does not map cleanly to a major gold driver.",
        }
    return GoldNewsEvent.from_mapping(data, allowed_event_types)


def extract_event(title: str, summary: str, config: dict[str, Any], client: LLMClient) -> GoldNewsEvent:
    allowed = set(config.get("event_types", [])) or None
    if client.mock_mode:
        return mock_extract_event(title, summary, allowed)
    data = client.chat_json(build_event_messages(config, title, summary))
    return GoldNewsEvent.from_mapping(data, allowed)


async def extract_event_async(title: str, summary: str, config: dict[str, Any], client: LLMClient) -> GoldNewsEvent:
    allowed = set(config.get("event_types", [])) or None
    if client.mock_mode:
        return mock_extract_event(title, summary, allowed)
    data = await client.chat_json_async(build_event_messages(config, title, summary))
    return GoldNewsEvent.from_mapping(data, allowed)


def extract_news_events(news: pd.DataFrame, config: dict[str, Any], client: LLMClient) -> pd.DataFrame:
    return asyncio.run(extract_news_events_async(news, config, client))


async def extract_news_events_async(news: pd.DataFrame, config: dict[str, Any], client: LLMClient) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    missing = {"published_at", "title"}.difference(news.columns)
    if missing:
        raise ValueError(f"raw news missing columns: {sorted(missing)}")

    records = news.reset_index(drop=True).to_dict("records")
    max_workers = max(1, int(config.get("llm", {}).get("max_concurrency", 1)))
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [_extract_one(index, article, config, client, semaphore) for index, article in enumerate(records)]
    rows = await asyncio.gather(*tasks)
    rows = [row for row in rows if row]
    out = pd.DataFrame(rows, columns=EVENT_OUTPUT_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    out["affected_channels"] = out["affected_channels"].apply(lambda value: ",".join(value) if isinstance(value, list) else str(value))
    return out.sort_values(["published_at", "source"]).reset_index(drop=True)


def select_unprocessed_news(
    raw_news: pd.DataFrame,
    existing_events: pd.DataFrame,
    recent_first: bool = True,
) -> pd.DataFrame:
    """Return raw news rows whose news_id has not been converted into events."""
    if raw_news.empty:
        return raw_news.copy()
    out = raw_news.copy()
    if "news_id" in out and existing_events is not None and not existing_events.empty and "news_id" in existing_events:
        processed = set(existing_events["news_id"].dropna().astype(str))
        out = out[~out["news_id"].astype(str).isin(processed)].copy()
    if recent_first and "published_at" in out:
        out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
        out = out.sort_values("published_at", ascending=False).copy()
    return out.reset_index(drop=True)


def merge_news_events(existing_events: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:
    """Append newly extracted events while preserving existing event history."""
    frames = []
    if existing_events is not None and not existing_events.empty:
        frames.append(existing_events)
    if new_events is not None and not new_events.empty:
        frames.append(new_events)
    if not frames:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for column in EVENT_OUTPUT_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.drop_duplicates(subset=["news_id"], keep="last")
    return out[EVENT_OUTPUT_COLUMNS].sort_values(["published_at", "source"]).reset_index(drop=True)


async def _extract_one(
    index: int,
    article: dict[str, Any],
    config: dict[str, Any],
    client: LLMClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    published_at = pd.to_datetime(article.get("published_at"), utc=True, errors="coerce")
    if pd.isna(published_at):
        return None
    base = {
        "news_id": article.get("news_id", ""),
        "published_at": published_at.tz_convert(None),
        "date": published_at.tz_convert(None).normalize(),
        "source": article.get("source", ""),
        "extraction_error": "",
    }
    try:
        async with semaphore:
            event = await _extract_with_retries_async(str(article.get("title", "")), str(article.get("summary", "")), config, client)
        row = event.to_dict()
        row.update(base)
        return row
    except Exception as exc:  # noqa: BLE001 - batch extraction should keep partial successes.
        row = GoldNewsEvent.from_mapping(
            {
                "event_type": "other",
                "gold_direction": "neutral",
                "confidence": 0.0,
                "surprise": 0.0,
                "time_horizon": "1d",
                "affected_channels": [],
                "llm_summary": "Extraction failed.",
                "rationale": "No structured event was produced.",
            },
            set(config.get("event_types", [])) or None,
        ).to_dict()
        row.update(base)
        row["extraction_error"] = f"{type(exc).__name__}: {exc}"
        return row


def _extract_with_retries(title: str, summary: str, config: dict[str, Any], client: LLMClient) -> GoldNewsEvent:
    attempts = max(1, int(config.get("llm", {}).get("max_retries", 1)))
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return extract_event(title, summary, config, client)
        except Exception as exc:  # noqa: BLE001 - caller records final failure.
            last_exc = exc
    assert last_exc is not None
    raise last_exc


async def _extract_with_retries_async(title: str, summary: str, config: dict[str, Any], client: LLMClient) -> GoldNewsEvent:
    attempts = max(1, int(config.get("llm", {}).get("max_retries", 1)))
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return await extract_event_async(title, summary, config, client)
        except Exception as exc:  # noqa: BLE001 - caller records final failure.
            last_exc = exc
    assert last_exc is not None
    raise last_exc
