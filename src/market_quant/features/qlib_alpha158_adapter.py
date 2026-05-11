"""Qlib Alpha158 feature provider adapter.

The adapter calls Qlib's maintained Alpha158 implementation and converts the
result into a plain pandas DataFrame. It does not copy Alpha158 source code,
train models, or run backtests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _require_qlib() -> tuple[Any, Any]:
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.contrib.data.handler import Alpha158
    except ImportError as exc:
        raise ImportError(
            "Qlib is required to build Alpha158 features. Install it with: "
            "python -m pip install pyqlib"
        ) from exc
    return qlib, REG_CN, Alpha158


def _flatten_qlib_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(level) for level in col if str(level) not in {"", "None"})
            for col in out.columns.to_flat_index()
        ]

    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        out = out.reset_index()
        rename_map = {}
        for col in out.columns[: len(names)]:
            lower = str(col).lower()
            if lower in {"datetime", "date", "time"}:
                rename_map[col] = "date"
            elif lower in {"instrument", "asset_id", "symbol", "code"}:
                rename_map[col] = "asset_id"
        out = out.rename(columns=rename_map)
    else:
        out = out.reset_index()
        first_col = out.columns[0]
        if str(first_col).lower() in {"index", "datetime", "date", "time"}:
            out = out.rename(columns={first_col: "date"})

    if "date" not in out.columns:
        for candidate in ["datetime", "time"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "date"})
                break
    if "asset_id" not in out.columns:
        for candidate in ["instrument", "symbol", "code"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "asset_id"})
                break

    if "date" not in out.columns or "asset_id" not in out.columns:
        raise ValueError("Qlib Alpha158 output must contain date and asset_id/instrument levels")

    out["date"] = pd.to_datetime(out["date"])
    out["asset_id"] = out["asset_id"].astype(str)
    leading = ["date", "asset_id"]
    rest = [col for col in out.columns if col not in leading]
    return out[leading + rest]


def _save_features(df: pd.DataFrame, output_path: str | None) -> str | None:
    if output_path is None:
        return None
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        try:
            df.to_parquet(target, index=False)
            return str(target)
        except Exception as exc:
            csv_target = target.with_name("qlib_alpha158_features.csv")
            df.to_csv(csv_target, index=False)
            print(f"warning: failed to save parquet to {target}: {exc}; saved csv to {csv_target}")
            return str(csv_target)
    df.to_csv(target, index=False)
    return str(target)


def _handler_to_frame(handler: Any) -> pd.DataFrame:
    if hasattr(handler, "fetch"):
        try:
            fetched = handler.fetch(col_set="feature")
        except TypeError:
            fetched = handler.fetch()
        if isinstance(fetched, pd.DataFrame):
            return fetched

    if hasattr(handler, "get_cols"):
        cols = handler.get_cols(col_set="feature")
        fetched = handler.fetch(cols=cols)
        if isinstance(fetched, pd.DataFrame):
            return fetched

    raise TypeError("Alpha158 handler did not return a pandas DataFrame")


def build_qlib_alpha158_features(
    provider_uri: str,
    instruments,
    start_time: str,
    end_time: str,
    fit_start_time: str | None = None,
    fit_end_time: str | None = None,
    freq: str = "day",
    normalize: bool = False,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build Qlib Alpha158 features as a plain pandas DataFrame."""
    qlib, REG_CN, Alpha158 = _require_qlib()
    resolved_provider_uri = Path(provider_uri).expanduser()
    if not resolved_provider_uri.exists():
        raise FileNotFoundError(
            f"Qlib provider_uri does not exist: {resolved_provider_uri}. "
            "Prepare Qlib data first or update config/feature_sets.yaml."
        )
    qlib.init(provider_uri=str(resolved_provider_uri), region=REG_CN)
    infer_processors = []
    learn_processors = []
    if normalize:
        infer_processors = [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        learn_processors = infer_processors

    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time or start_time,
        fit_end_time=fit_end_time or end_time,
        freq=freq,
        infer_processors=infer_processors,
        learn_processors=learn_processors,
    )
    out = _flatten_qlib_frame(_handler_to_frame(handler))
    _save_features(out, output_path)
    return out
