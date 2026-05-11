"""Reusable live gold prediction pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_quant.gold.features import (
    append_manual_primary_price_row,
    append_preopen_primary_row,
    build_gold_feature_frame,
    load_alpha158_features,
)
from market_quant.gold.live_signal import build_live_gold_signal

try:
    from market_quant.options.chain import load_option_chain
    from market_quant.options.live import build_live_option_plan
except ImportError:  # pragma: no cover - options overlay is optional
    load_option_chain = None
    build_live_option_plan = None


@dataclass(frozen=True)
class LivePipelineInputs:
    feature_frame: pd.DataFrame
    training: pd.DataFrame
    features: list[str]
    overlay_as_of_date: pd.Timestamp | None
    manual_price_used: bool


def read_optional_csv(project_root: Path, path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    target = project_root / path
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def filter_as_of(df: pd.DataFrame, timestamp_columns: list[str], as_of_date: str | None) -> pd.DataFrame:
    if df.empty or not as_of_date:
        return df
    cutoff = pd.Timestamp(as_of_date).normalize() + pd.Timedelta(days=1)
    out = df.copy()
    for column in timestamp_columns:
        if column in out.columns:
            ts = pd.to_datetime(out[column], errors="coerce")
            return out[ts < cutoff].copy()
    return out


def latest_available_date(df: pd.DataFrame, timestamp_columns: list[str]) -> pd.Timestamp | None:
    if df.empty:
        return None
    for column in timestamp_columns:
        if column in df.columns:
            ts = pd.to_datetime(df[column], errors="coerce").dropna()
            if not ts.empty:
                return ts.max().normalize()
    return None


def infer_overlay_as_of_date(
    news: pd.DataFrame,
    polymarket: pd.DataFrame,
    explicit_as_of_date: str | None,
) -> pd.Timestamp | None:
    if explicit_as_of_date:
        return pd.Timestamp(explicit_as_of_date).normalize()
    candidates = [
        latest_available_date(news, ["published_at", "date"]),
        latest_available_date(polymarket, ["timestamp"]),
    ]
    candidates = [date for date in candidates if date is not None]
    if not candidates:
        return None
    return max(candidates)


def load_live_pipeline_inputs(
    project_root: Path,
    config: dict[str, Any],
    signal_date: str,
    as_of_date: str | None = None,
    manual_primary_price: float | None = None,
    preopen: bool = False,
) -> LivePipelineInputs:
    feature_config = config.get("features", {})
    primary_col = feature_config.get("primary_etf_series", "518880_close")

    series_path = project_root / config["input"]["daily_series"]
    dataset_path = project_root / config["output"]["dataset"]
    feature_path = project_root / config["output"]["feature_columns"]
    if not series_path.exists():
        raise FileNotFoundError("Run python scripts/data/fetch_gold_data.py first.")
    if not dataset_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Run python scripts/features/build_gold_features.py first.")

    series = pd.read_csv(series_path)
    manual_price_used = manual_primary_price is not None
    if manual_price_used:
        series = append_manual_primary_price_row(series, primary_col, signal_date, float(manual_primary_price))
    if preopen:
        series = append_preopen_primary_row(series, primary_col, signal_date)

    raw_news = read_optional_csv(project_root, config.get("input", {}).get("news_events"))
    raw_polymarket = read_optional_csv(project_root, config.get("input", {}).get("polymarket"))
    overlay_as_of_date = infer_overlay_as_of_date(raw_news, raw_polymarket, as_of_date)
    overlay_as_of_text = overlay_as_of_date.strftime("%Y-%m-%d") if overlay_as_of_date is not None else None
    news = filter_as_of(raw_news, ["published_at", "date"], overlay_as_of_text)
    polymarket = filter_as_of(raw_polymarket, ["timestamp"], overlay_as_of_text)

    alpha = pd.DataFrame()
    alpha_path = config.get("input", {}).get("alpha158")
    if alpha_path and feature_config.get("include_alpha158", False):
        alpha_asset_id = feature_config.get("primary_alpha158_asset_id")
        if alpha_asset_id:
            alpha = load_alpha158_features(project_root / alpha_path, alpha_asset_id)
        else:
            print("Alpha158 features: skipped because primary_alpha158_asset_id is not configured")

    feature_bundle = build_gold_feature_frame(
        series,
        feature_config,
        news_events=news,
        polymarket_data=polymarket,
        alpha158_features=alpha,
    )

    training = pd.read_csv(dataset_path)
    training["date"] = pd.to_datetime(training["date"])
    training["label_end_date"] = pd.to_datetime(training["label_end_date"])
    training = training.set_index("date").sort_index()
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]

    return LivePipelineInputs(
        feature_frame=feature_bundle.data,
        training=training,
        features=features,
        overlay_as_of_date=overlay_as_of_date,
        manual_price_used=manual_price_used,
    )


def build_execution_plan(
    target_position: float,
    current_position: float | None,
    rebalance_band: float,
    max_position_change: float,
) -> dict[str, Any]:
    if current_position is None:
        return {
            "current_position": None,
            "target_position": float(target_position),
            "recommended_position": float(target_position),
            "trade_delta": None,
            "action": "set_target_when_current_position_is_supplied",
            "rebalance_band": float(rebalance_band),
            "max_position_change": float(max_position_change),
        }

    current = max(0.0, min(1.0, float(current_position)))
    target = max(0.0, min(1.0, float(target_position)))
    raw_delta = target - current
    if abs(raw_delta) < rebalance_band:
        trade_delta = 0.0
        action = "hold"
    else:
        trade_delta = max(-max_position_change, min(max_position_change, raw_delta))
        action = "increase" if trade_delta > 0 else "decrease"
    recommended = max(0.0, min(1.0, current + trade_delta))
    return {
        "current_position": current,
        "target_position": target,
        "recommended_position": recommended,
        "trade_delta": trade_delta,
        "action": action,
        "rebalance_band": float(rebalance_band),
        "max_position_change": float(max_position_change),
    }


def run_live_prediction(
    project_root: Path,
    config: dict[str, Any],
    signal_date: str,
    as_of_date: str | None = None,
    manual_primary_price: float | None = None,
    preopen: bool = False,
    current_position: float | None = None,
    rebalance_band: float = 0.05,
    max_position_change: float = 0.35,
):
    inputs = load_live_pipeline_inputs(
        project_root,
        config,
        signal_date=signal_date,
        as_of_date=as_of_date,
        manual_primary_price=manual_primary_price,
        preopen=preopen,
    )
    signal, selected_report = build_live_gold_signal(
        inputs.training,
        inputs.feature_frame,
        inputs.features,
        signal_date=signal_date,
        target_column="target_direction",
        model_name=config["model"].get("name", "lightgbm"),
        model_parameters=config["model"].get("parameters", {}),
        feature_selection_config=config.get("feature_selection", {}),
        calibration_config=config.get("walk_forward", {}).get("calibration", {}),
        sizing_config=config.get("sizing", {}),
        regime_config=config.get("regime", {}),
        live_overlay_config=config.get("live_overlay", {}),
    )
    execution = build_execution_plan(
        signal.final_position,
        current_position,
        rebalance_band=rebalance_band,
        max_position_change=max_position_change,
    )
    payload = {
        "signal_date": signal.signal_date,
        "as_of_date": inputs.overlay_as_of_date,
        "feature_date": signal.feature_date,
        "manual_primary_price": float(manual_primary_price) if inputs.manual_price_used else None,
        "raw_probability": signal.raw_probability,
        "probability": signal.probability,
        "overlay_position": signal.overlay_position,
        "sigmoid_position": signal.sigmoid_position,
        "volatility_adjusted_position": signal.volatility_adjusted_position,
        "layered_position": signal.layered_position,
        "final_position": signal.final_position,
        "regime_score": signal.regime_score,
        "regime_label": signal.regime_label,
        "execution": execution,
        "diagnostics": signal.diagnostics,
    }

    option_cfg = config.get("options", {})
    if option_cfg.get("enabled", False):
        if load_option_chain is None or build_live_option_plan is None:
            payload["options"] = {"enabled": False, "reason": "options_module_unavailable"}
        else:
            option_chain_path = option_cfg.get("chain_path")
            if not option_chain_path:
                payload["options"] = {"enabled": False, "reason": "option_chain_path_not_configured"}
            else:
                chain_path = project_root / option_chain_path
                option_chain = load_option_chain(
                    chain_path,
                    as_of_date=signal.feature_date,
                    default_multiplier=float(option_cfg.get("default_multiplier", 1000.0)),
                )
                payload["options"] = build_live_option_plan(
                    gold_signal=signal,
                    feature_frame=inputs.feature_frame,
                    option_chain=option_chain,
                    config=option_cfg,
                )
    else:
        payload["options"] = {"enabled": False, "reason": "disabled"}

    return payload, selected_report
