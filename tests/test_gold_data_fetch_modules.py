import pandas as pd

from market_quant.data.gold_china_data import fetch_china_gold_etfs
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
