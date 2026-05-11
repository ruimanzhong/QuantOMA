import asyncio

import pandas as pd

from market_quant.llm.client import LLMClient
from market_quant.llm.news_events import extract_news_events, extract_news_events_async, merge_news_events, select_unprocessed_news


def test_extract_news_events_drops_raw_text_fields():
    news = pd.DataFrame(
        {
            "published_at": ["2024-01-01T12:00:00Z"],
            "source": ["test"],
            "title": ["Fed signals rate cut path"],
            "summary": ["Officials sounded dovish as inflation cooled."],
            "url": ["https://example.com/article"],
        }
    )
    config = {"event_types": ["fed", "other"], "llm": {"mock_mode": True}}
    client = LLMClient(config["llm"])

    out = extract_news_events(news, config, client)

    assert len(out) == 1
    assert "title" not in out.columns
    assert "summary" not in out.columns
    assert "url" not in out.columns
    assert out.loc[0, "event_type"] == "fed"
    assert out.loc[0, "llm_summary"]


def test_extract_news_events_keeps_error_rows(monkeypatch):
    news = pd.DataFrame(
        {
            "published_at": ["2024-01-01T12:00:00Z"],
            "source": ["test"],
            "title": ["Fed signals rate cut path"],
            "summary": ["Officials sounded dovish as inflation cooled."],
        }
    )
    config = {
        "event_types": ["fed", "other"],
        "llm": {"mock_mode": False, "max_concurrency": 2},
        "system_prompt": "system",
        "event_extraction_prompt": "Title: {title}\nSummary: {summary}",
    }
    client = LLMClient(config["llm"])

    def fail(_messages):
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "chat_json", fail)
    out = extract_news_events(news, config, client)

    assert len(out) == 1
    assert out.loc[0, "event_type"] == "other"
    assert "RuntimeError: boom" in out.loc[0, "extraction_error"]


def test_extract_news_events_async_respects_concurrency_limit():
    news = pd.DataFrame(
        {
            "published_at": ["2024-01-01T12:00:00Z", "2024-01-01T12:01:00Z", "2024-01-01T12:02:00Z"],
            "source": ["test", "test", "test"],
            "title": ["Fed rate cut", "Dollar strengthens", "War risk rises"],
            "summary": ["", "", ""],
        }
    )
    config = {
        "event_types": ["fed", "usd", "geopolitics", "other"],
        "llm": {"mock_mode": False, "max_concurrency": 2, "max_retries": 1},
        "system_prompt": "system",
        "event_extraction_prompt": "Title: {title}\nSummary: {summary}",
    }
    client = LLMClient(config["llm"])
    state = {"active": 0, "max_active": 0}

    async def fake_chat_json_async(_messages):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        try:
            await asyncio.sleep(0.01)
            return {
                "event_type": "other",
                "gold_direction": "neutral",
                "confidence": 0.5,
                "surprise": 0.1,
                "time_horizon": "1d",
                "affected_channels": [],
                "llm_summary": "ok",
                "rationale": "ok",
            }
        finally:
            state["active"] -= 1

    client.chat_json_async = fake_chat_json_async

    out = asyncio.run(extract_news_events_async(news, config, client))

    assert len(out) == 3
    assert state["max_active"] == 2


def test_select_unprocessed_news_filters_existing_news_ids():
    raw = pd.DataFrame(
        {
            "news_id": ["a", "b"],
            "published_at": ["2024-01-01", "2024-01-02"],
            "title": ["old", "new"],
        }
    )
    existing = pd.DataFrame({"news_id": ["a"]})

    out = select_unprocessed_news(raw, existing)

    assert out["news_id"].tolist() == ["b"]


def test_merge_news_events_deduplicates_by_news_id():
    existing = pd.DataFrame(
        {
            "news_id": ["a"],
            "published_at": ["2024-01-01"],
            "date": ["2024-01-01"],
            "source": ["test"],
            "event_type": ["other"],
            "gold_direction": ["neutral"],
            "confidence": [0.1],
            "surprise": [0.1],
            "time_horizon": ["1d"],
            "affected_channels": [""],
            "llm_summary": ["old"],
            "rationale": ["old"],
            "extraction_error": [""],
        }
    )
    new = existing.copy()
    new["llm_summary"] = ["new"]

    out = merge_news_events(existing, new)

    assert len(out) == 1
    assert out["llm_summary"].tolist() == ["new"]
