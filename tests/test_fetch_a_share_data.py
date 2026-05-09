import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_a_share_data.py"


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_a_share_data_test_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def price_frame(source="test"):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "asset_id": ["510300.SH", "510300.SH"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 200],
            "amount": [1000, 2000],
            "source": [source, source],
        }
    )


def base_config(provider):
    return {
        "provider": provider,
        "symbols": ["510300.SH"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "adjust": "qfq",
    }


def test_provider_efinance_calls_adapter(monkeypatch):
    module = load_fetch_module()
    calls = []

    def fake_efinance(**kwargs):
        calls.append(kwargs)
        return price_frame("efinance")

    monkeypatch.setattr(module, "fetch_efinance_daily", fake_efinance)
    out = module.fetch_configured_data(base_config("efinance"))

    assert calls == [{"symbols": ["510300.SH"], "start_date": "2024-01-01", "end_date": "2024-01-02"}]
    assert out["source"].unique().tolist() == ["efinance"]


def test_provider_baostock_calls_adapter(monkeypatch):
    module = load_fetch_module()
    calls = []

    def fake_baostock(**kwargs):
        calls.append(kwargs)
        return price_frame("baostock")

    monkeypatch.setattr(module, "fetch_baostock_daily", fake_baostock)
    out = module.fetch_configured_data(base_config("baostock"))

    assert calls == [
        {
            "symbols": ["510300.SH"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "adjust": "qfq",
        }
    ]
    assert out["source"].unique().tolist() == ["baostock"]


def test_primary_failure_uses_fallback(monkeypatch, capsys):
    module = load_fetch_module()
    calls = []
    config = base_config("efinance")
    config["fallback_provider"] = "baostock"

    def fail_efinance(**kwargs):
        raise RuntimeError("primary failed")

    def fake_baostock(**kwargs):
        calls.append(kwargs)
        return price_frame("baostock")

    monkeypatch.setattr(module, "fetch_efinance_daily", fail_efinance)
    monkeypatch.setattr(module, "fetch_baostock_daily", fake_baostock)
    out = module.fetch_configured_data(config)

    captured = capsys.readouterr()
    assert "warning: primary provider 'efinance' failed" in captured.out
    assert calls
    assert out["source"].unique().tolist() == ["baostock"]


def test_summary_contains_row_count_and_date_range(capsys):
    module = load_fetch_module()
    module.print_coverage_summary(price_frame(), base_config("efinance"))

    captured = capsys.readouterr()
    assert "asset_id=510300.SH" in captured.out
    assert "min_date=2024-01-01" in captured.out
    assert "max_date=2024-01-02" in captured.out
    assert "row_count=2" in captured.out
