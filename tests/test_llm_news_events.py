import pandas as pd

from market_quant.llm.client import LLMClient
from market_quant.llm.news_events import extract_news_events


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
