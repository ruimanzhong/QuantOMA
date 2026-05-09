import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_test_module"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_configured_gold_sources_flags_late_start(tmp_path):
    module = load_script("audit_gold_long_history_readiness.py")
    source_root = tmp_path / "gold"
    source_root.mkdir()
    pd.DataFrame({"date": ["2018-01-02"], "symbol": ["xauusd"], "close": [1300.0]}).to_csv(
        source_root / "market.csv",
        index=False,
    )
    config = {
        "source_root": str(source_root),
        "inputs": {
            "market": {
                "path": "market.csv",
                "type": "ohlcv_long",
                "value_columns": ["close"],
            }
        },
    }

    out = module.audit_configured_gold_sources(config)

    assert out.loc[0, "exists"]
    assert not out.loc[0, "ready_for_2008"]
    assert "starts_after_2008-01-01" in out.loc[0, "blocking_issue"]
