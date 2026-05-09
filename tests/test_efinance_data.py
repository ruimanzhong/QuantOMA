import builtins
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from market_quant.data.efinance_data import fetch_efinance_daily, to_efinance_symbol


def quote_frame():
    return pd.DataFrame(
        {
            "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "股票代码": ["510300", "510300", "510300"],
            "开盘": [1.0, 1.1, 1.2],
            "收盘": [1.1, 1.2, 1.3],
            "最高": [1.2, 1.3, 1.4],
            "最低": [0.9, 1.0, 1.1],
            "成交量": [100, 200, 300],
            "成交额": [1000, 2000, 3000],
        }
    )


def install_fake_efinance(monkeypatch, frame=None, calls=None):
    calls = calls if calls is not None else []

    def get_quote_history(code, **kwargs):
        calls.append((code, kwargs))
        return quote_frame() if frame is None else frame

    fake = SimpleNamespace(stock=SimpleNamespace(get_quote_history=get_quote_history))
    monkeypatch.setitem(sys.modules, "efinance", fake)
    return calls


def test_efinance_missing_clear_import_error(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "efinance":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "efinance", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="efinance is required for ETF historical data"):
        fetch_efinance_daily(["510300.SH"], "2024-01-01", "2024-01-03")


def test_symbol_mapping():
    assert to_efinance_symbol("510300.SH") == "510300"
    assert to_efinance_symbol("159915.SZ") == "159915"


def test_chinese_field_mapping_and_schema(monkeypatch):
    calls = install_fake_efinance(monkeypatch)
    out = fetch_efinance_daily(["510300.SH"], "2024-01-01", "2024-01-03")

    assert calls[0][0] == "510300"
    assert calls[0][1]["beg"] == "20240101"
    assert calls[0][1]["end"] == "20240103"
    assert out["asset_id"].tolist() == ["510300.SH", "510300.SH", "510300.SH"]
    assert out["source"].unique().tolist() == ["efinance"]
    assert out["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert out["close"].tolist() == [1.1, 1.2, 1.3]


def test_start_end_filtering(monkeypatch):
    install_fake_efinance(monkeypatch)
    out = fetch_efinance_daily(["510300.SH"], "2024-01-02", "2024-01-02")

    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_empty_return_clear_error(monkeypatch):
    install_fake_efinance(monkeypatch, frame=pd.DataFrame())

    with pytest.raises(RuntimeError, match="efinance returned no data for symbols: 510300.SH"):
        fetch_efinance_daily(["510300.SH"], "2024-01-01", "2024-01-03")
