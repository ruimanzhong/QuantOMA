import pandas as pd

from market_quant.data.news_data import NEWS_COLUMNS, fetch_gold_news, merge_news_cache


def test_fetch_gold_news_disabled_returns_schema():
    out = fetch_gold_news({"enabled": False})

    assert out.columns.tolist() == NEWS_COLUMNS
    assert out.empty


def test_merge_news_cache_keeps_existing_when_fetch_empty():
    existing = pd.DataFrame(
        {
            "news_id": ["a"],
            "date": ["2024-01-01"],
            "source": ["test"],
            "title": ["Old"],
            "summary": [""],
            "url": ["https://example.com"],
            "published_at": ["2024-01-01 00:00:00"],
            "relevance_score": [1],
        }
    )

    out = merge_news_cache(existing, pd.DataFrame(columns=NEWS_COLUMNS))

    assert len(out) == 1
    assert out["title"].tolist() == ["Old"]
