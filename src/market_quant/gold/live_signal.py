"""Live gold signal utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from market_quant.gold.backtest import apply_volatility_targeting, build_gold_positions
from market_quant.gold.layered import build_gold_regime_scores, build_hmm_style_regime_prior, build_layered_gold_positions
from market_quant.gold.live_overlay import build_live_overlay_result
from market_quant.models.walk_forward import (
    apply_probability_calibrator,
    fit_probability_calibrator,
    make_classifier,
    orthogonalize_training_frames,
    predict_classifier_probability,
    select_features_for_training,
    split_train_calibration_index,
)


@dataclass(frozen=True)
class LiveGoldSignal:
    signal_date: pd.Timestamp
    feature_date: pd.Timestamp
    probability: float
    raw_probability: float
    overlay_position: float
    sigmoid_position: float
    volatility_adjusted_position: float
    layered_position: float
    final_position: float
    regime_score: float
    regime_label: str
    selected_features: list[str]
    diagnostics: dict[str, Any]


def select_live_features(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    feature_selection_config: dict[str, Any] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    cfg = feature_selection_config or {}
    if not cfg.get("enabled", False) and not cfg.get("correlation_clustering", {}).get("enabled", False):
        return feature_columns, pd.DataFrame()
    return select_features_for_training(training_data, feature_columns, target_column, cfg)


def build_live_gold_signal(
    training_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    signal_date: str | pd.Timestamp,
    target_column: str = "target_direction",
    model_name: str = "lightgbm",
    model_parameters: dict[str, Any] | None = None,
    feature_selection_config: dict[str, Any] | None = None,
    overlay_lower_threshold: float = 0.50,
    overlay_defensive_position: float = 0.30,
    calibration_config: dict[str, Any] | None = None,
    sizing_config: dict[str, Any] | None = None,
    regime_config: dict[str, Any] | None = None,
    live_overlay_config: dict[str, Any] | None = None,
) -> tuple[LiveGoldSignal, pd.DataFrame]:
    """Train on all labeled history and convert latest probability to exposure."""
    if not isinstance(training_data.index, pd.DatetimeIndex) or not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise TypeError("training_data and feature_frame must use DatetimeIndex")
    if target_column not in training_data:
        raise ValueError(f"training data missing target column: {target_column}")

    signal_ts = pd.Timestamp(signal_date).normalize()
    available = feature_frame.loc[feature_frame.index <= signal_ts].sort_index()
    if available.empty:
        raise ValueError(f"no feature row available on or before signal_date={signal_ts.date()}")
    live_row = available.tail(1)
    feature_date = live_row.index[0]

    calibration_cfg = calibration_config or {}
    calibration_method = str(calibration_cfg.get("method", "none"))
    calibration_size = int(calibration_cfg.get("size", 0))
    min_fit_samples = int(calibration_cfg.get("min_fit_samples", 300))
    fit_index, calibration_index = split_train_calibration_index(training_data.index, calibration_size, min_fit_samples)
    feature_train = training_data.loc[fit_index]
    calibration = training_data.loc[calibration_index] if len(calibration_index) else pd.DataFrame()

    orthogonalization_cfg = (feature_selection_config or {}).get("orthogonalization", {})
    feature_train_model, calibration_model, live_model, model_feature_columns, orth_report = orthogonalize_training_frames(
        feature_train,
        calibration,
        live_row,
        feature_columns,
        orthogonalization_cfg,
    )
    selected_features, feature_score = select_live_features(feature_train_model, model_feature_columns, target_column, feature_selection_config)
    if not orth_report.empty:
        feature_score = pd.concat([orth_report, feature_score], ignore_index=True, sort=False) if not feature_score.empty else orth_report
    selected_features = [col for col in selected_features if col in feature_train_model and col in live_model]
    if not selected_features:
        raise ValueError("no selected features available for live inference")

    train = feature_train_model.dropna(subset=[target_column]).copy()
    x_train = train[selected_features].astype(float)
    y_train = train[target_column].astype(int)
    model = make_classifier(model_name, model_parameters)
    model.fit(x_train, y_train)
    raw_probability = float(predict_classifier_probability(model, live_model[selected_features].astype(float))[0])
    probability = raw_probability
    calibration_used = "none"
    if calibration_method not in {"none", "off", ""} and not calibration.empty and calibration[target_column].nunique() > 1:
        calibration_raw = predict_classifier_probability(model, calibration_model[selected_features].astype(float))
        calibrator = fit_probability_calibrator(calibration_raw, calibration_model[target_column].astype(int), calibration_method)
        probability = float(apply_probability_calibrator(calibrator, [raw_probability])[0])
        calibration_used = calibration_method

    prediction = pd.DataFrame(
        {
            "date": [feature_date],
            "probability": [probability],
            "primary_etf": [float(live_row["primary_etf"].iloc[0])],
        }
    )
    overlay_position = float(
        build_gold_positions(
            prediction["probability"],
            strategy_type="defensive_overlay",
            lower_threshold=overlay_lower_threshold,
            defensive_position=overlay_defensive_position,
        ).iloc[0]
    )
    sizing_cfg = sizing_config or {}
    sigmoid_position = float(
        build_gold_positions(
            prediction["probability"],
            strategy_type=sizing_cfg.get("strategy_type", "sigmoid_bet_sizing"),
            base_position=float(sizing_cfg.get("base_position", 0.10)),
            sigmoid_threshold=float(sizing_cfg.get("sigmoid_threshold", 0.55)),
            sigmoid_sigma=float(sizing_cfg.get("sigmoid_sigma", 0.08)),
        ).iloc[0]
    )
    if sizing_cfg.get("volatility_targeting", True):
        history_price = feature_frame.loc[feature_frame.index <= feature_date, "primary_etf"].tail(int(sizing_cfg.get("volatility_window", 20)) + 5)
        position_series = pd.Series([sigmoid_position], index=[feature_date])
        price_series = pd.concat([history_price.iloc[:-1], pd.Series([float(live_row["primary_etf"].iloc[0])], index=[feature_date])])
        volatility_adjusted_position = float(
            apply_volatility_targeting(
                position_series,
                price_series,
                target_volatility=float(sizing_cfg.get("target_volatility", 0.18)),
                volatility_window=int(sizing_cfg.get("volatility_window", 20)),
                floor_multiplier=float(sizing_cfg.get("floor_multiplier", 0.25)),
            ).iloc[-1]
        )
    else:
        volatility_adjusted_position = sigmoid_position

    regime_cfg = regime_config or {}
    regime_history = feature_frame.loc[feature_frame.index <= feature_date].reset_index(names="date")
    regime = build_gold_regime_scores(live_row.reset_index(names="date"))
    if regime_cfg.get("hmm_prior", {}).get("enabled", True):
        hmm_cfg = regime_cfg.get("hmm_prior", {})
        hmm_prior = build_hmm_style_regime_prior(
            regime_history,
            transition_stickiness=float(hmm_cfg.get("transition_stickiness", 0.94)),
            crisis_base_position=float(hmm_cfg.get("base_position_floor", 0.10)),
            crisis_position_scale=float(hmm_cfg.get("crisis_position_scale", 0.50)),
        ).tail(1)
        regime = regime.merge(hmm_prior, on="date", how="left")
    layered = build_layered_gold_positions(prediction, regime)
    layered_position = float(layered["position"].iloc[0])
    regime_score = float(layered["regime_score"].iloc[0])
    regime_label = str(layered["regime_label"].iloc[0])

    live_overlay = build_live_overlay_result(live_row, live_overlay_config)
    final_position = min(layered_position, volatility_adjusted_position) * live_overlay.multiplier
    if probability < overlay_lower_threshold:
        final_position = min(final_position, overlay_position)

    diagnostics = {
        "model_name_used": type(model).__name__,
        "calibration_method": calibration_used,
        "n_calibration_rows": int(len(calibration)),
        "n_training_rows": int(len(train)),
        "n_selected_features": int(len(selected_features)),
        "n_alpha158_selected_features": int(sum(col.startswith("alpha158_") for col in selected_features)),
        "primary_etf_last_close": float(live_row["primary_etf"].iloc[0]),
        "external_gap_xauusd_return": _safe_float(live_row, "external_gap_xauusd_return"),
        "external_gap_implied_gold_cny_per_g_return": _safe_float(live_row, "external_gap_implied_gold_cny_per_g_return"),
        "primary_trading_gap_calendar_days": _safe_float(live_row, "primary_trading_gap_calendar_days"),
        "external_gap_observation_count": _safe_float(live_row, "external_gap_observation_count"),
        "crisis_probability": _safe_float(layered, "crisis_probability"),
        "adaptive_base_position": _safe_float(layered, "adaptive_base_position"),
        "hmm_regime_label": str(layered["hmm_regime_label"].iloc[0]) if "hmm_regime_label" in layered else None,
        "live_overlay_multiplier": live_overlay.multiplier,
        "live_overlay_reasons": live_overlay.reasons,
        **live_overlay.diagnostics,
    }
    signal = LiveGoldSignal(
        signal_date=signal_ts,
        feature_date=feature_date,
        probability=probability,
        raw_probability=raw_probability,
        overlay_position=overlay_position,
        sigmoid_position=sigmoid_position,
        volatility_adjusted_position=volatility_adjusted_position,
        layered_position=layered_position,
        final_position=float(max(0.0, min(1.0, final_position))),
        regime_score=regime_score,
        regime_label=regime_label,
        selected_features=selected_features,
        diagnostics=diagnostics,
    )
    return signal, feature_score


def _safe_float(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    value = pd.to_numeric(df[column], errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)
