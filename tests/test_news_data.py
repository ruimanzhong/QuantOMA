from market_quant.data.news_data import NEWS_COLUMNS, fetch_gold_news


def test_fetch_gold_news_disabled_returns_schema():
    out = fetch_gold_news({"enabled": False})

    assert out.columns.tolist() == NEWS_COLUMNS
    assert out.empty
