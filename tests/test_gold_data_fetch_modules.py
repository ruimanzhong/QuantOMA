import pandas as pd
import pytest

from market_quant.data.gold_china_data import fetch_china_gold_etfs
from market_quant.data.gold_night_data import append_night_decision_price
from market_quant.data.gold_night_data import build_night_decision_row
from market_quant.data.gold_night_data import normalize_sge_intraday_quotes
from market_quant.data.gold_night_data import select_decision_price
from market_quant.data.network_proxy import scoped_proxy_environment
from market_quant.data.gold_market_data import normalize_ohlcv


def test_normalize_ohlcv_accepts_date_index():
    df = pd.DataFrame({"Close": [1.0, 1.1], "Open": [0.9, 1.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

    out = normalize_ohlcv(df, "xauusd")

    assert out["symbol"].tolist() == ["xauusd", "xauusd"]
    assert out["close"].tolist() == [1.0, 1.1]


def test_fetch_china_gold_etfs_uses_efinance_adapter(monkeypatch):
    def fake_fetch(symbols, start_date, end_date):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "asset_id": [symbols[0]],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [100],
                "amount": [1000],
                "source": ["test"],
            }
        )

    monkeypatch.setattr("market_quant.data.gold_china_data.fetch_efinance_daily", fake_fetch)
    out = fetch_china_gold_etfs({"etf_symbols": ["518880.SH"], "start_date": "2024-01-01", "end_date": "2024-01-02"})

    assert out["symbol"].tolist() == ["518880.SH"]
    assert out["close"].tolist() == [1.0]


def test_scoped_proxy_environment_restores_existing_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://global-proxy:7890")
    monkeypatch.setenv("CHINA_DATA_PROXY", "socks5://china-proxy:1080")

    with scoped_proxy_environment({"proxy": {"enabled": True, "proxy_env": "CHINA_DATA_PROXY"}}, label="test"):
        assert "china-proxy" in __import__("os").environ["HTTPS_PROXY"]

    assert __import__("os").environ["HTTPS_PROXY"] == "http://global-proxy:7890"


def test_fetch_china_gold_etfs_applies_scoped_proxy(monkeypatch):
    observed = {}
    monkeypatch.setenv("CHINA_DATA_PROXY", "socks5://china-proxy:1080")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)

    def fake_fetch(symbols, start_date, end_date):
        observed["https_proxy"] = __import__("os").environ.get("HTTPS_PROXY")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "asset_id": [symbols[0]],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [100],
                "amount": [1000],
                "source": ["test"],
            }
        )

    monkeypatch.setattr("market_quant.data.gold_china_data.fetch_efinance_daily", fake_fetch)
    out = fetch_china_gold_etfs(
        {
            "etf_symbols": ["518880.SH"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "network": {"proxy": {"enabled": True, "proxy_env": "CHINA_DATA_PROXY"}},
        }
    )

    assert out["symbol"].tolist() == ["518880.SH"]
    assert observed["https_proxy"] == "socks5://china-proxy:1080"
    assert __import__("os").environ.get("HTTPS_PROXY") is None


def test_normalize_sge_intraday_quotes_builds_timestamps():
    raw = pd.DataFrame(
        {
            "品种": ["Au99.99", "Au99.99"],
            "时间": ["22:59:00", "23:00:00"],
            "现价": [1029.5, 1030.0],
            "更新时间": ["2026年05月12日 23:00:05", "2026年05月12日 23:00:05"],
        }
    )

    out = normalize_sge_intraday_quotes(raw)

    assert out["price"].tolist() == [1029.5, 1030.0]
    assert out["timestamp_utc"].iloc[-1] == pd.Timestamp("2026-05-12 15:00:00", tz="UTC")


def test_select_decision_price_uses_nearest_quote_within_tolerance():
    quotes = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-05-12", "2026-05-12"]),
            "time": ["22:58:00", "23:04:00"],
            "timestamp_utc": pd.to_datetime(["2026-05-12 14:58:00+00:00", "2026-05-12 15:04:00+00:00"]),
            "symbol": ["Au99.99", "Au99.99"],
            "price": [1029.0, 1031.0],
            "updated_at_utc": pd.to_datetime(["2026-05-12 15:04:10+00:00", "2026-05-12 15:04:10+00:00"]),
            "source": ["test", "test"],
        }
    )

    row = select_decision_price(quotes, "23:00", tolerance_minutes=5)

    assert row["price"] == 1029.0


def test_select_decision_price_rejects_stale_quote():
    quotes = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-05-12"]),
            "time": ["22:40:00"],
            "timestamp_utc": pd.to_datetime(["2026-05-12 14:40:00+00:00"]),
            "symbol": ["Au99.99"],
            "price": [1029.0],
            "updated_at_utc": pd.to_datetime(["2026-05-12 14:40:10+00:00"]),
            "source": ["test"],
        }
    )

    with pytest.raises(ValueError, match="nearest quote"):
        select_decision_price(quotes, "23:00", tolerance_minutes=5)


def test_append_night_decision_price_replaces_same_decision_time():
    first = build_night_decision_row(
        pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2026-05-12 15:00:00", tz="UTC"),
                "symbol": "Au99.99",
                "price": 1030.0,
                "source": "test",
                "updated_at_utc": pd.Timestamp("2026-05-12 15:00:05", tz="UTC"),
            }
        )
    )
    second = build_night_decision_row(
        pd.Series(
            {
                "timestamp_utc": pd.Timestamp("2026-05-12 15:01:00", tz="UTC"),
                "symbol": "Au99.99",
                "price": 1031.0,
                "source": "test",
                "updated_at_utc": pd.Timestamp("2026-05-12 15:01:05", tz="UTC"),
            }
        )
    )

    out = append_night_decision_price(first, second)

    assert len(out) == 1
    assert out["price"].iloc[0] == 1031.0
