from pathlib import Path

import pandas as pd
import pytest

from market_quant.data.phase2_local_data import (
    load_gold_local_series,
    load_macro_wide_csv,
    load_ohlcv_long_csv,
    load_series_long_csv,
)


def test_load_ohlcv_long_csv_maps_symbol_and_field(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["xauusd", "xauusd"],
            "close": [2000.0, 2010.0],
            "volume": [0, 0],
        }
    ).to_csv(path, index=False)

    out = load_ohlcv_long_csv(path, ["close"], "market")

    assert out["series_id"].tolist() == ["xauusd_close", "xauusd_close"]
    assert out["source"].unique().tolist() == ["market"]


def test_load_macro_wide_csv_drops_missing_without_forward_fill(tmp_path):
    path = tmp_path / "macro.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "dxy": [None, 100.0],
            "vix": [12.0, 13.0],
        }
    ).to_csv(path, index=False)

    out = load_macro_wide_csv(path, ["dxy", "vix"], "macro")

    assert out[out["series_id"] == "dxy"]["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert out[out["series_id"] == "vix"]["value"].tolist() == [12.0, 13.0]


def test_load_series_long_csv_maps_symbol_column(tmp_path):
    path = tmp_path / "sge.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "sge_symbol": ["Au99.99"],
            "sge_gold_cny_per_g": [480.0],
        }
    ).to_csv(path, index=False)

    out = load_series_long_csv(path, "sge_symbol", ["sge_gold_cny_per_g"], "sge_gold")

    assert out.loc[0, "series_id"] == "au9999_sge_gold_cny_per_g"


def test_load_gold_local_series_combines_inputs(tmp_path):
    source_root = tmp_path / "old"
    (source_root / "market").mkdir(parents=True)
    (source_root / "macro").mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-01"], "symbol": ["xauusd"], "close": [2000.0]}).to_csv(
        source_root / "market" / "market_data.csv",
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "dxy": [100.0]}).to_csv(
        source_root / "macro" / "fred_macro.csv",
        index=False,
    )
    config = {
        "source_root": str(source_root),
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "inputs": {
            "market_data": {
                "path": "market/market_data.csv",
                "type": "ohlcv_long",
                "value_columns": ["close"],
            },
            "macro": {
                "path": "macro/fred_macro.csv",
                "type": "macro_wide",
                "value_columns": ["dxy"],
            },
        },
    }

    out = load_gold_local_series(config, Path.cwd())

    assert set(out["series_id"]) == {"xauusd_close", "dxy"}


def test_load_gold_local_series_missing_file_clear_error(tmp_path):
    config = {
        "source_root": str(tmp_path),
        "inputs": {
            "market_data": {
                "path": "missing.csv",
                "type": "ohlcv_long",
                "value_columns": ["close"],
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="Gold local CSV does not exist"):
        load_gold_local_series(config, Path.cwd())
