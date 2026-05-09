import sys
import types
import builtins

import pandas as pd
import pytest

from market_quant.features import qlib_alpha158_adapter as adapter


def clear_fake_qlib(monkeypatch):
    for name in list(sys.modules):
        if name == "qlib" or name.startswith("qlib."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def install_fake_qlib(monkeypatch, frame):
    qlib = types.ModuleType("qlib")
    calls = {"init": False, "fit": False}

    def init(provider_uri, **kwargs):
        calls["init"] = {"provider_uri": provider_uri, **kwargs}

    qlib.init = init
    config_mod = types.ModuleType("qlib.config")
    config_mod.REG_CN = "cn"

    handler_mod = types.ModuleType("qlib.contrib.data.handler")

    class FakeAlpha158:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fetch(self, *args, **kwargs):
            return frame

        def fit(self):
            calls["fit"] = True
            raise AssertionError("training should not be called")

    handler_mod.Alpha158 = FakeAlpha158
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setitem(sys.modules, "qlib.config", config_mod)
    monkeypatch.setitem(sys.modules, "qlib.contrib", types.ModuleType("qlib.contrib"))
    monkeypatch.setitem(sys.modules, "qlib.contrib.data", types.ModuleType("qlib.contrib.data"))
    monkeypatch.setitem(sys.modules, "qlib.contrib.data.handler", handler_mod)
    return calls


def test_qlib_missing_raises_clear_import_error(monkeypatch):
    clear_fake_qlib(monkeypatch)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "qlib" or name.startswith("qlib."):
            raise ImportError("missing qlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ImportError, match="python -m pip install pyqlib"):
        adapter.build_qlib_alpha158_features("/tmp/qlib", "all", "2024-01-01", "2024-01-02")


def test_provider_uri_missing_raises_clear_file_not_found(monkeypatch, tmp_path):
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-01"), "A")], names=["date", "asset_id"])
    install_fake_qlib(monkeypatch, pd.DataFrame({"f": [1]}, index=index))
    missing = tmp_path / "missing_provider"

    with pytest.raises(FileNotFoundError, match="Qlib provider_uri does not exist"):
        adapter.build_qlib_alpha158_features(str(missing), "all", "2024-01-01", "2024-01-02")


def test_multiindex_flatten_contains_date_asset_id(monkeypatch, tmp_path):
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-01")], ["sh.510300"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"RESI5": [1.0]}, index=index)
    calls = install_fake_qlib(monkeypatch, frame)
    out = adapter.build_qlib_alpha158_features(str(tmp_path), "all", "2024-01-01", "2024-01-02")
    assert calls["init"]["provider_uri"] == str(tmp_path)
    assert calls["init"]["region"] == "cn"
    assert out.columns[:2].tolist() == ["date", "asset_id"]
    assert out.loc[0, "asset_id"] == "sh.510300"


def test_does_not_call_training_logic(monkeypatch, tmp_path):
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-01"), "A")], names=["date", "asset_id"])
    calls = install_fake_qlib(monkeypatch, pd.DataFrame({"f": [1]}, index=index))
    adapter.build_qlib_alpha158_features(str(tmp_path), "all", "2024-01-01", "2024-01-02")
    assert calls["fit"] is False


def test_output_path_save_falls_back_to_csv(monkeypatch, tmp_path):
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-01"), "A")], names=["date", "asset_id"])
    install_fake_qlib(monkeypatch, pd.DataFrame({"f": [1]}, index=index))

    def raise_import_error(self, *args, **kwargs):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raise_import_error)
    output = tmp_path / "features.parquet"
    adapter.build_qlib_alpha158_features(str(tmp_path), "all", "2024-01-01", "2024-01-02", output_path=str(output))
    assert not output.exists()
    assert (tmp_path / "qlib_alpha158_features.csv").exists()
