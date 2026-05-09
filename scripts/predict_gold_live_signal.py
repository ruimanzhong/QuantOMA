#!/usr/bin/env python
"""Produce a live gold ETF probability and 0-1 exposure signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_quant.gold.features import (
    append_preopen_primary_row,
    build_gold_feature_frame,
    load_alpha158_features,
)
from market_quant.gold.live_signal import build_live_gold_signal


def _read_optional(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    target = PROJECT_ROOT / path
    if not target.exists():
        return pd.DataFrame()
    return pd.read_csv(target)


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _filter_as_of(df: pd.DataFrame, timestamp_columns: list[str], as_of_date: str | None) -> pd.DataFrame:
    if df.empty or not as_of_date:
        return df
    cutoff = pd.Timestamp(as_of_date).normalize() + pd.Timedelta(days=1)
    out = df.copy()
    for column in timestamp_columns:
        if column in out.columns:
            ts = pd.to_datetime(out[column], errors="coerce")
            return out[ts < cutoff].copy()
    return out


def _latest_available_date(df: pd.DataFrame, timestamp_columns: list[str]) -> pd.Timestamp | None:
    if df.empty:
        return None
    for column in timestamp_columns:
        if column in df.columns:
            ts = pd.to_datetime(df[column], errors="coerce").dropna()
            if not ts.empty:
                return ts.max().normalize()
    return None


def _infer_overlay_as_of_date(news: pd.DataFrame, polymarket: pd.DataFrame, explicit_as_of_date: str | None) -> pd.Timestamp | None:
    if explicit_as_of_date:
        return pd.Timestamp(explicit_as_of_date).normalize()
    candidates = [
        _latest_available_date(news, ["published_at", "date"]),
        _latest_available_date(polymarket, ["timestamp"]),
    ]
    candidates = [date for date in candidates if date is not None]
    if not candidates:
        return None
    return max(candidates)


def _build_execution_plan(
    target_position: float,
    current_position: float | None,
    rebalance_band: float,
    max_position_change: float,
) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", required=True, help="Decision date, e.g. 2026-05-06.")
    parser.add_argument("--as-of-date", default=None, help="Latest data date allowed for live overlays, e.g. 2026-05-05.")
    parser.add_argument("--preopen", action="store_true", help="Add a synthetic pre-open primary ETF row if the decision date has no China close.")
    parser.add_argument("--current-position", type=float, default=None, help="Current portfolio gold exposure from 0 to 1.")
    parser.add_argument("--rebalance-band", type=float, default=0.05, help="Do not trade if target-current absolute delta is below this band.")
    parser.add_argument("--max-position-change", type=float, default=0.35, help="Maximum one-signal position change.")
    parser.add_argument("--output", default="data/reports/gold_live_signal.json")
    parser.add_argument("--selected-output", default="data/reports/gold_live_selected_features.csv")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "gold_model.yaml").read_text())
    feature_config = config.get("features", {})
    primary_col = feature_config.get("primary_etf_series", "518880_close")

    series_path = PROJECT_ROOT / config["input"]["daily_series"]
    dataset_path = PROJECT_ROOT / config["output"]["dataset"]
    feature_path = PROJECT_ROOT / config["output"]["feature_columns"]
    if not series_path.exists():
        raise SystemExit("Run python scripts/fetch_gold_data.py first.")
    if not dataset_path.exists() or not feature_path.exists():
        raise SystemExit("Run python scripts/build_gold_features.py first.")

    series = pd.read_csv(series_path)
    if args.preopen:
        series = append_preopen_primary_row(series, primary_col, args.signal_date)

    raw_news = _read_optional(config.get("input", {}).get("news_events"))
    raw_polymarket = _read_optional(config.get("input", {}).get("polymarket"))
    overlay_as_of_date = _infer_overlay_as_of_date(raw_news, raw_polymarket, args.as_of_date)
    overlay_as_of_text = overlay_as_of_date.strftime("%Y-%m-%d") if overlay_as_of_date is not None else None
    news = _filter_as_of(raw_news, ["published_at", "date"], overlay_as_of_text)
    polymarket = _filter_as_of(raw_polymarket, ["timestamp"], overlay_as_of_text)
    alpha = pd.DataFrame()
    alpha_path = config.get("input", {}).get("alpha158")
    if alpha_path:
        alpha = load_alpha158_features(PROJECT_ROOT / alpha_path, feature_config.get("primary_alpha158_asset_id", "518880"))

    feature_frame = build_gold_feature_frame(
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

    signal, selected_report = build_live_gold_signal(
        training,
        feature_frame.data,
        features,
        signal_date=args.signal_date,
        target_column="target_direction",
        model_name=config["model"].get("name", "lightgbm"),
        model_parameters=config["model"].get("parameters", {}),
        feature_selection_config=config.get("feature_selection", {}),
        calibration_config=config.get("walk_forward", {}).get("calibration", {}),
        sizing_config=config.get("sizing", {}),
        regime_config=config.get("regime", {}),
        live_overlay_config=config.get("live_overlay", {}),
    )

    execution = _build_execution_plan(
        signal.final_position,
        args.current_position,
        rebalance_band=args.rebalance_band,
        max_position_change=args.max_position_change,
    )
    payload = {
        "signal_date": signal.signal_date,
        "as_of_date": overlay_as_of_date,
        "feature_date": signal.feature_date,
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
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))

    selected_output = PROJECT_ROOT / args.selected_output
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_report.to_csv(selected_output, index=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    print(f"saved live signal to {output}")
    print(f"saved selected feature report to {selected_output}")


if __name__ == "__main__":
    main()
