"""Diagnostics for local Qlib providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _require_qlib():
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
    except ImportError as exc:
        raise ImportError(
            "Qlib is required to inspect Qlib provider data. Install it with: python -m pip install pyqlib"
        ) from exc
    return qlib, REG_CN, D


def _check_provider_uri(provider_uri: str) -> Path:
    path = Path(provider_uri).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Qlib provider_uri does not exist: {path}. Prepare Qlib data first or update config/feature_sets.yaml."
        )
    return path


def _init_qlib(provider_uri: Path, region: str):
    qlib, REG_CN, D = _require_qlib()
    qlib_region = REG_CN if region.lower() == "cn" else region
    qlib.init(provider_uri=str(provider_uri), region=qlib_region)
    return D


def _instrument_count(D: Any, universe: str) -> dict:
    try:
        instruments = D.instruments(universe)
        listed = D.list_instruments(instruments=instruments, as_list=True)
        return {"count": len(listed), "error": None, "sample": list(listed[:20])}
    except Exception as exc:
        return {"count": None, "error": str(exc), "sample": []}


def inspect_qlib_provider(provider_uri: str, region: str = "cn") -> dict:
    """Inspect provider directory shape, calendar range, and common universes."""
    path = _check_provider_uri(provider_uri)
    subdirs = {name: (path / name).exists() for name in ["calendars", "features", "instruments"]}
    D = _init_qlib(path, region)

    summary = {
        "provider_uri": str(path),
        "provider_uri_exists": True,
        "subdirs": subdirs,
        "calendar_length": None,
        "first_calendar_date": None,
        "last_calendar_date": None,
        "universes": {},
    }
    try:
        calendar = pd.to_datetime(D.calendar(freq="day"))
        summary["calendar_length"] = int(len(calendar))
        if len(calendar) > 0:
            summary["first_calendar_date"] = str(calendar.min().date())
            summary["last_calendar_date"] = str(calendar.max().date())
    except Exception as exc:
        summary["calendar_error"] = str(exc)

    for universe in ["all", "csi300", "csi500", "csi100"]:
        summary["universes"][universe] = _instrument_count(D, universe)
    return summary


def list_qlib_instruments(
    provider_uri: str,
    instruments,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """List Qlib instruments for a universe without remapping symbols."""
    path = _check_provider_uri(provider_uri)
    D = _init_qlib(path, "cn")
    universe_name = ",".join(instruments) if isinstance(instruments, list) else str(instruments)
    if isinstance(instruments, list):
        listed = instruments
    else:
        qlib_instruments = D.instruments(instruments)
        listed = D.list_instruments(
            instruments=qlib_instruments,
            start_time=start_time,
            end_time=end_time,
            as_list=True,
        )
    return pd.DataFrame(
        {
            "universe_name": [universe_name] * len(listed),
            "asset_id": [str(item) for item in listed],
            "start_time": [start_time] * len(listed),
            "end_time": [end_time] * len(listed),
        }
    )
