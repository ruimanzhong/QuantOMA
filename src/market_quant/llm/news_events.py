"""Convert raw news rows into structured gold event rows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def extract_news_events(news: pd.DataFrame, config: dict[str, Any], client: LLMClient) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    missing = {"published_at", "title"}.difference(news.columns)
    if missing:
        raise ValueError(f"raw news missing columns: {sorted(missing)}")

    records = news.reset_index(drop=True).to_dict("records")
    max_workers = max(1, int(config.get("llm", {}).get("max_concurrency", 1)))
    if client.mock_mode or max_workers == 1:
        rows = [_extract_one(index, article, config, client) for index, article in enumerate(records)]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_one, index, article, config, client): index for index, article in enumerate(records)}
            for future in as_completed(futures):
                rows.append(future.result())
    rows = [row for row in rows if row]
    out = pd.DataFrame(rows, columns=EVENT_OUTPUT_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=EVENT_OUTPUT_COLUMNS)
    out["affected_channels"] = out["affected_channels"].apply(lambda value: ",".join(value) if isinstance(value, list) else str(value))
    return out.sort_values(["published_at", "source"]).reset_index(drop=True)


def _extract_one(index: int, article: dict[str, Any], config: dict[str, Any], client: LLMClient) -> dict[str, Any] | None:
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
        event = _extract_with_retries(str(article.get("title", "")), str(article.get("summary", "")), config, client)
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
