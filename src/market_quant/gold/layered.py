"""Layered gold exposure framework: regime, signal, strategy, and risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_quant.backtest.metrics import summarize_backtest
from market_quant.gold.backtest import build_gold_positions


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_gold_regime_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Build observable daily regime scores from model features."""
    required = {"date"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {sorted(missing)}")
    out = features[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"])

    trend_score = (
        (_numeric(features, "xauusd_momentum_20d") > 0).astype(float)
        + (_numeric(features, "xauusd_momentum_60d") > 0).astype(float)
        + (_numeric(features, "xauusd_ma_distance_60d") > 0).astype(float)
    ) / 3.0
    macro_score = (
        (_numeric(features, "dxy_trend_20d") < 0).astype(float)
        + (_numeric(features, "real_yield_trend_20d") < 0).astype(float)
        + (_numeric(features, "vix_change_5d") > 0).astype(float)
    ) / 3.0

    news_cols = [col for col in features.columns if col.startswith("news_bullish_score")]
    news_bearish_cols = [col for col in features.columns if col.startswith("news_bearish_score")]
    if news_cols or news_bearish_cols:
        bullish = features[news_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if news_cols else 0.0
        bearish = features[news_bearish_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if news_bearish_cols else 0.0
        news_score = ((bullish - bearish) > 0).astype(float)
    else:
        news_score = pd.Series(0.5, index=features.index)

    poly_cols = [col for col in features.columns if col.endswith("_probability_change_3d") or col.endswith("_probability_change_5d")]
    if poly_cols:
        poly_change = features[poly_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        polymarket_score = (poly_change > 0).astype(float)
    else:
        polymarket_score = pd.Series(0.5, index=features.index)

    out["trend_score"] = trend_score.astype(float)
    out["macro_score"] = macro_score.astype(float)
    out["news_score"] = news_score.astype(float)
    out["polymarket_score"] = polymarket_score.astype(float)
    out["regime_score"] = (
        0.45 * out["trend_score"]
        + 0.35 * out["macro_score"]
        + 0.10 * out["news_score"]
        + 0.10 * out["polymarket_score"]
    )
    out["regime_label"] = "neutral"
    out.loc[out["regime_score"] >= 0.60, "regime_label"] = "supportive"
    out.loc[out["regime_score"] <= 0.40, "regime_label"] = "hostile"
    return out


def build_hmm_style_regime_prior(
    features: pd.DataFrame,
    transition_stickiness: float = 0.94,
    crisis_base_position: float = 0.10,
    crisis_position_scale: float = 0.50,
) -> pd.DataFrame:
    """Estimate a lightweight two-state macro crisis prior.

    This is a dependency-light HMM-style filter: observations are clustered into
    normal/crisis Gaussian emissions, then smoothed with a sticky transition
    matrix. It is deliberately modest, but gives us a probabilistic macro prior
    before adding a heavier HMM dependency.
    """
    required = {"date"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {sorted(missing)}")
    out = features[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    emission = _build_macro_emission_matrix(features)
    if emission.shape[0] < 50 or emission.shape[1] == 0:
        out["crisis_probability"] = 0.5
        out["adaptive_base_position"] = crisis_base_position + crisis_position_scale * out["crisis_probability"]
        out["hmm_regime_label"] = "unknown"
        return out

    x = emission.to_numpy(dtype=float)
    try:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=2, covariance_type="full", random_state=42, reg_covar=1e-4)
        raw_state_probability = model.fit_predict(x)
        posterior = model.predict_proba(x)
        means = pd.DataFrame(model.means_, columns=emission.columns)
    except ImportError:
        raw_state_probability, posterior, means = _fallback_two_state_emissions(x, emission.columns)

    crisis_state = _identify_crisis_state(means)
    crisis_emission_probability = posterior[:, crisis_state]
    out["crisis_probability"] = _sticky_two_state_filter(crisis_emission_probability, transition_stickiness)
    out["adaptive_base_position"] = (crisis_base_position + crisis_position_scale * out["crisis_probability"]).clip(0.0, 1.0)
    out["hmm_regime_label"] = "normal"
    out.loc[out["crisis_probability"] >= 0.60, "hmm_regime_label"] = "crisis"
    out.loc[out["crisis_probability"].between(0.40, 0.60, inclusive="left"), "hmm_regime_label"] = "transition"
    return out


def _build_macro_emission_matrix(features: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "vix_level",
        "vix_change_5d",
        "vix_trend_20d",
        "sp500_return_20d",
        "sp500_trend_20d",
        "dxy_return_20d",
        "dxy_trend_20d",
        "yield_10y_change",
        "real_yield_10y_change",
        "real_yield_trend_20d",
        "xauusd_volatility_20d",
    ]
    cols = [col for col in candidates if col in features]
    if not cols:
        return pd.DataFrame(index=features.index)
    out = features[cols].apply(pd.to_numeric, errors="coerce")
    out = out.ffill().bfill()
    out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
    std = out.std(ddof=0).replace(0.0, np.nan)
    out = (out - out.mean()) / std
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=-6.0, upper=6.0)


def _identify_crisis_state(means: pd.DataFrame) -> int:
    score = pd.Series(0.0, index=means.index)
    for col in ["vix_level", "vix_change_5d", "vix_trend_20d", "xauusd_volatility_20d"]:
        if col in means:
            score = score + means[col]
    for col in ["sp500_return_20d", "sp500_trend_20d"]:
        if col in means:
            score = score - means[col]
    for col in ["yield_10y_change", "real_yield_10y_change"]:
        if col in means:
            score = score + means[col].abs()
    return int(score.idxmax())


def _sticky_two_state_filter(crisis_emission_probability: np.ndarray, transition_stickiness: float) -> np.ndarray:
    stay = min(max(float(transition_stickiness), 0.50), 0.995)
    switch = 1.0 - stay
    p = 0.5
    filtered = []
    for emission_p in np.clip(crisis_emission_probability, 1e-6, 1 - 1e-6):
        prior = stay * p + switch * (1.0 - p)
        numerator = emission_p * prior
        denominator = numerator + (1.0 - emission_p) * (1.0 - prior)
        p = numerator / denominator if denominator else prior
        filtered.append(p)
    return np.asarray(filtered, dtype=float)


def _fallback_two_state_emissions(x: np.ndarray, columns: pd.Index) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    stress = x.mean(axis=1)
    threshold = np.nanmedian(stress)
    state = (stress > threshold).astype(int)
    posterior = np.column_stack([1.0 - state, state]).astype(float)
    means = pd.DataFrame([x[state == 0].mean(axis=0), x[state == 1].mean(axis=0)], columns=columns)
    return state, posterior, means


def build_layered_gold_positions(
    predictions: pd.DataFrame,
    regime_scores: pd.DataFrame,
    base_position: float = 0.7,
    min_position: float = 0.2,
    max_position: float = 1.0,
    defensive_position: float = 0.3,
    lower_threshold: float = 0.50,
    probability_lower: float = 0.45,
    probability_upper: float = 0.65,
    max_daily_position_change: float = 0.35,
) -> pd.DataFrame:
    """Combine regime and model probability into final long-only exposure."""
    required = {"date", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if {"date", "regime_score", "regime_label"}.difference(regime_scores.columns):
        raise ValueError("regime_scores must include date, regime_score, and regime_label")

    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    regime = regime_scores.copy()
    regime["date"] = pd.to_datetime(regime["date"])
    out = out.merge(regime, on="date", how="left")
    out["regime_score"] = out["regime_score"].fillna(0.5)
    out["regime_label"] = out["regime_label"].fillna("neutral")
    if "adaptive_base_position" in out:
        adaptive_base = pd.to_numeric(out["adaptive_base_position"], errors="coerce").fillna(base_position).clip(min_position, max_position)
    else:
        adaptive_base = pd.Series(base_position, index=out.index, dtype=float)

    overlay = build_gold_positions(
        out["probability"],
        strategy_type="defensive_overlay",
        lower_threshold=lower_threshold,
        defensive_position=defensive_position,
    )
    scaled = build_gold_positions(
        out["probability"],
        strategy_type="probability_scaled",
        lower_threshold=probability_lower,
        threshold=probability_upper,
    )
    out["overlay_position"] = overlay.to_numpy()
    out["scaled_signal_position"] = scaled.to_numpy()
    out["regime_position"] = (min_position + out["regime_score"] * (max_position - min_position)).clip(min_position, max_position)
    out["raw_position"] = (
        0.45 * out["regime_position"]
        + 0.35 * out["overlay_position"]
        + 0.20 * out["scaled_signal_position"]
    ).clip(min_position, max_position)
    out["raw_position"] = out["raw_position"].where(out["raw_position"] >= adaptive_base, adaptive_base)
    hostile = out["regime_label"] == "hostile"
    out.loc[hostile, "raw_position"] = out.loc[hostile, "raw_position"].clip(upper=adaptive_base.loc[hostile].clip(lower=base_position))
    out.loc[out["regime_label"] == "supportive", "raw_position"] = out.loc[out["regime_label"] == "supportive", "raw_position"].clip(lower=adaptive_base)

    values = out["raw_position"].to_numpy(copy=True)
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        if abs(delta) > max_daily_position_change:
            values[idx] = values[idx - 1] + max_daily_position_change * (1 if delta > 0 else -1)
    out["position"] = pd.Series(values, index=out.index).clip(min_position, max_position)
    return out


def run_layered_gold_backtest(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    transaction_cost_bps: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """Backtest the layered regime-aware gold exposure framework."""
    if "primary_etf" not in predictions:
        raise ValueError("predictions missing columns: ['primary_etf']")
    regime = build_gold_regime_scores(features)
    out = build_layered_gold_positions(predictions, regime)
    out = out.sort_values("date").reset_index(drop=True)
    out["asset_return"] = out["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    out["executed_position"] = out["position"].shift(1).fillna(0.0)
    trade = out["executed_position"].diff().abs().fillna(out["executed_position"].abs())
    out["transaction_cost"] = trade * transaction_cost_bps / 10000.0
    out["strategy_return"] = out["executed_position"] * out["asset_return"] - out["transaction_cost"]
    metrics = summarize_backtest(out.assign(asset_id="gold_layered"))
    metrics["strategy_type"] = "layered_regime_exposure"
    metrics["avg_regime_score"] = float(out["regime_score"].mean())
    metrics["supportive_ratio"] = float((out["regime_label"] == "supportive").mean())
    metrics["hostile_ratio"] = float((out["regime_label"] == "hostile").mean())
    return out, metrics
