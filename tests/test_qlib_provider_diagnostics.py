import sys
import types

import pandas as pd
import pytest

from market_quant.diagnostics.qlib_provider_diagnostics import inspect_qlib_provider, list_qlib_instruments


def install_fake_qlib(monkeypatch):
    qlib = types.ModuleType("qlib")
    calls = {}

    def init(provider_uri, **kwargs):
        calls["init"] = {"provider_uri": provider_uri, **kwargs}

    qlib.init = init
    config_mod = types.ModuleType("qlib.config")
    config_mod.REG_CN = "cn"

    data_mod = types.ModuleType("qlib.data")

    class FakeD:
        @staticmethod
        def calendar(freq="day"):
            return pd.to_datetime(["2024-01-01", "2024-01-02"])

        @staticmethod
        def instruments(market):
            if market == "csi100":
                raise ValueError("universe missing")
            return market

        @staticmethod
        def list_instruments(instruments, start_time=None, end_time=None, as_list=True):
            if instruments == "csi100":
                raise ValueError("universe missing")
            if isinstance(instruments, list):
                return instruments
            return ["SH600000", "SZ000001"]

    data_mod.D = FakeD
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setitem(sys.modules, "qlib.config", config_mod)
    monkeypatch.setitem(sys.modules, "qlib.data", data_mod)
    return calls


def test_provider_uri_missing_raises_clear_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Qlib provider_uri does not exist"):
        inspect_qlib_provider(str(tmp_path / "missing"))


def test_provider_subdir_check_and_universe_error(monkeypatch, tmp_path):
    install_fake_qlib(monkeypatch)
    for name in ["calendars", "features"]:
        (tmp_path / name).mkdir()

    out = inspect_qlib_provider(str(tmp_path))

    assert out["subdirs"] == {"calendars": True, "features": True, "instruments": False}
    assert out["calendar_length"] == 2
    assert out["universes"]["all"]["count"] == 2
    assert out["universes"]["csi100"]["error"] == "universe missing"


def test_list_qlib_instruments_fields_and_no_etf_mapping(monkeypatch, tmp_path):
    install_fake_qlib(monkeypatch)
    out = list_qlib_instruments(str(tmp_path), ["SH510300", "SZ159915"], "2024-01-01", "2024-01-02")

    assert list(out.columns) == ["universe_name", "asset_id", "start_time", "end_time"]
    assert out["asset_id"].tolist() == ["SH510300", "SZ159915"]
    assert "510300.SH" not in out["asset_id"].tolist()
