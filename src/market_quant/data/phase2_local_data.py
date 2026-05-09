"""Phase 2 local CSV adapters for the independent gold data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_quant.data.series_schema import validate_daily_series_frame


def _resolve_path(project_root: Path, source_root: str, relative_path: str) -> Path:
    root = Path(source_root).expanduser()
    if not root.is_absolute():
        root = (project_root / root).resolve()
    return root / relative_path


def _filter_dates(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out


def _series_id(*parts: object) -> str:
    clean_parts = []
    for part in parts:
        text = str(part).strip().lower().replace(" ", "_")
        if "." in text and text.split(".")[-1] in {"sh", "sz"}:
            text = text.split(".")[0]
        else:
            text = text.replace(".", "")
        if text:
            clean_parts.append(text)
    return "_".join(clean_parts)


def load_ohlcv_long_csv(
    path: str | Path,
    value_columns: list[str],
    source_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load long OHLCV CSV rows into the canonical daily series schema."""
    df = pd.read_csv(path)
    missing = {"date", "symbol"}.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} CSV missing required columns: {sorted(missing)}")
    df = _filter_dates(df, start_date, end_date)

    frames = []
    for value_col in value_columns:
        if value_col not in df.columns:
            raise ValueError(f"{source_name} CSV missing value column: {value_col}")
        part = df[["date", "symbol", value_col]].rename(columns={value_col: "value"}).copy()
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        part = part.dropna(subset=["value"])
        part["series_id"] = [_series_id(symbol, value_col) for symbol in part["symbol"]]
        part["source"] = source_name
        part["source_path"] = str(path)
        part["field"] = value_col
        frames.append(part[["date", "series_id", "value", "source", "source_path", "field"]])
    return validate_daily_series_frame(pd.concat(frames, ignore_index=True))


def load_series_long_csv(
    path: str | Path,
    symbol_column: str,
    value_columns: list[str],
    source_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load long scalar series CSV rows into the canonical daily series schema."""
    df = pd.read_csv(path)
    missing = {"date", symbol_column}.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} CSV missing required columns: {sorted(missing)}")
    df = _filter_dates(df, start_date, end_date)

    frames = []
    for value_col in value_columns:
        if value_col not in df.columns:
            raise ValueError(f"{source_name} CSV missing value column: {value_col}")
        part = df[["date", symbol_column, value_col]].rename(columns={value_col: "value"}).copy()
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        part = part.dropna(subset=["value"])
        part["series_id"] = [_series_id(symbol, value_col) for symbol in part[symbol_column]]
        part["source"] = source_name
        part["source_path"] = str(path)
        part["field"] = value_col
        frames.append(part[["date", "series_id", "value", "source", "source_path", "field"]])
    return validate_daily_series_frame(pd.concat(frames, ignore_index=True))


def load_macro_wide_csv(
    path: str | Path,
    value_columns: list[str],
    source_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load wide macro CSV rows into the canonical daily series schema."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{source_name} CSV missing required column: date")
    df = _filter_dates(df, start_date, end_date)

    frames = []
    for value_col in value_columns:
        if value_col not in df.columns:
            raise ValueError(f"{source_name} CSV missing value column: {value_col}")
        part = df[["date", value_col]].rename(columns={value_col: "value"}).copy()
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        part = part.dropna(subset=["value"])
        part["series_id"] = value_col
        part["source"] = source_name
        part["source_path"] = str(path)
        part["field"] = value_col
        frames.append(part[["date", "series_id", "value", "source", "source_path", "field"]])
    return validate_daily_series_frame(pd.concat(frames, ignore_index=True))


def load_gold_local_series(config: dict[str, Any], project_root: str | Path) -> pd.DataFrame:
    """Load configured gold local CSV inputs into one daily series frame."""
    root = Path(project_root)
    source_root = config.get("source_root", ".")
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    frames = []

    for source_name, source_cfg in config.get("inputs", {}).items():
        path = _resolve_path(root, source_root, source_cfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"Gold local CSV does not exist: {path}")
        value_columns = list(source_cfg.get("value_columns", []))
        input_type = source_cfg["type"]
        if input_type == "ohlcv_long":
            frames.append(load_ohlcv_long_csv(path, value_columns, source_name, start_date, end_date))
        elif input_type == "series_long":
            frames.append(
                load_series_long_csv(
                    path,
                    source_cfg.get("symbol_column", "symbol"),
                    value_columns,
                    source_name,
                    start_date,
                    end_date,
                )
            )
        elif input_type == "macro_wide":
            frames.append(load_macro_wide_csv(path, value_columns, source_name, start_date, end_date))
        else:
            raise ValueError(f"Unsupported gold local input type for {source_name}: {input_type}")

    if not frames:
        raise ValueError("No gold local CSV inputs configured")
    return validate_daily_series_frame(pd.concat(frames, ignore_index=True))


def load_phase2_local_series(config: dict[str, Any], project_root: str | Path) -> pd.DataFrame:
    """Backward-compatible alias for the gold Phase 2 local series loader."""
    return load_gold_local_series(config, project_root)
