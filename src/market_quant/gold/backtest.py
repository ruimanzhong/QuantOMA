"""Gold model signal backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from market_quant.backtest.metrics import summarize_backtest


def _apply_min_holding(position: pd.Series, min_holding_days: int) -> pd.Series:
    if min_holding_days <= 1 or position.empty:
        return position.astype(float)
    values = position.astype(float).to_numpy(copy=True)
    current = values[0]
    held = 1
    for idx in range(1, len(values)):
        desired = values[idx]
        if desired != current and held < min_holding_days:
            values[idx] = current
            held += 1
        elif desired != current:
            current = desired
            values[idx] = current
            held = 1
        else:
            held += 1
    return pd.Series(values, index=position.index)


def build_gold_positions(
    probability: pd.Series,
    strategy_type: str = "threshold",
    threshold: float = 0.55,
    lower_threshold: float = 0.45,
    defensive_position: float = 0.3,
    base_position: float = 0.1,
    sigmoid_threshold: float = 0.55,
    sigmoid_sigma: float = 0.08,
    smoothing_window: int = 1,
    min_holding_days: int = 1,
) -> pd.Series:
    """Convert model probabilities into long-only gold exposure."""
    prob = probability.astype(float).copy()
    if smoothing_window > 1:
        prob = prob.rolling(smoothing_window, min_periods=1).mean()

    if strategy_type == "threshold":
        position = (prob >= threshold).astype(float)
    elif strategy_type == "probability_scaled":
        position = ((prob - lower_threshold) / max(threshold - lower_threshold, 1e-9)).clip(0.0, 1.0)
    elif strategy_type == "defensive_overlay":
        position = pd.Series(1.0, index=prob.index)
        position[prob < lower_threshold] = defensive_position
    elif strategy_type == "sigmoid_bet_sizing":
        scaled = (prob - sigmoid_threshold) / max(sigmoid_sigma, 1e-9)
        sigmoid = 1.0 / (1.0 + np.exp(-scaled))
        position = pd.Series(np.maximum(base_position, sigmoid), index=prob.index).clip(0.0, 1.0)
    elif strategy_type == "de_prado_sigmoid":
        denom = np.sqrt((prob * (1.0 - prob)).clip(lower=1e-9))
        z_score = (prob - 0.5) / denom
        normal_cdf = 0.5 * (1.0 + pd.Series([math.erf(value / math.sqrt(2.0)) for value in z_score], index=prob.index))
        conviction = (2.0 * normal_cdf - 1.0).clip(0.0, 1.0)
        position = pd.Series(np.maximum(base_position, conviction), index=prob.index).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported gold strategy_type: {strategy_type}")
    return _apply_min_holding(position, min_holding_days)


def apply_volatility_targeting(
    position: pd.Series,
    price: pd.Series,
    target_volatility: float = 0.18,
    volatility_window: int = 20,
    floor_multiplier: float = 0.25,
) -> pd.Series:
    """Scale exposure down when realized annualized volatility is above target."""
    pos = position.astype(float).copy()
    if target_volatility <= 0:
        return pos.clip(0.0, 1.0)
    returns = price.astype(float).pct_change(fill_method=None)
    actual_vol = returns.rolling(volatility_window, min_periods=max(2, volatility_window // 2)).std(ddof=0) * np.sqrt(252.0)
    multiplier = (target_volatility / actual_vol.replace(0.0, np.nan)).clip(lower=floor_multiplier, upper=1.0).fillna(1.0)
    return (pos * multiplier).clip(0.0, 1.0)


def run_gold_long_flat_backtest(
    predictions: pd.DataFrame,
    threshold: float = 0.55,
    transaction_cost_bps: float = 2.0,
    strategy_type: str = "threshold",
    lower_threshold: float = 0.45,
    defensive_position: float = 0.3,
    base_position: float = 0.1,
    sigmoid_threshold: float = 0.55,
    sigmoid_sigma: float = 0.08,
    smoothing_window: int = 1,
    min_holding_days: int = 1,
    volatility_targeting: bool = False,
    target_volatility: float = 0.18,
    volatility_window: int = 20,
) -> tuple[pd.DataFrame, dict]:
    required = {"date", "probability", "primary_etf"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["position"] = build_gold_positions(
        out["probability"],
        strategy_type=strategy_type,
        threshold=threshold,
        lower_threshold=lower_threshold,
        defensive_position=defensive_position,
        base_position=base_position,
        sigmoid_threshold=sigmoid_threshold,
        sigmoid_sigma=sigmoid_sigma,
        smoothing_window=smoothing_window,
        min_holding_days=min_holding_days,
    ).to_numpy()
    if volatility_targeting:
        out["position"] = apply_volatility_targeting(
            out["position"],
            out["primary_etf"],
            target_volatility=target_volatility,
            volatility_window=volatility_window,
        ).to_numpy()
    out["asset_return"] = out["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    out["executed_position"] = out["position"].shift(1).fillna(0.0)
    trade = out["executed_position"].diff().abs().fillna(out["executed_position"].abs())
    out["transaction_cost"] = trade * transaction_cost_bps / 10000.0
    out["strategy_return"] = out["executed_position"] * out["asset_return"] - out["transaction_cost"]
    metrics = summarize_backtest(out.assign(asset_id="gold_model"))
    metrics["strategy_type"] = strategy_type
    metrics["threshold"] = threshold
    metrics["lower_threshold"] = lower_threshold
    metrics["defensive_position"] = defensive_position
    metrics["base_position"] = base_position
    metrics["sigmoid_threshold"] = sigmoid_threshold
    metrics["sigmoid_sigma"] = sigmoid_sigma
    metrics["smoothing_window"] = smoothing_window
    metrics["min_holding_days"] = min_holding_days
    metrics["volatility_targeting"] = volatility_targeting
    metrics["target_volatility"] = target_volatility
    metrics["volatility_window"] = volatility_window
    return out, metrics


def run_gold_position_backtest(
    predictions: pd.DataFrame,
    transaction_cost_bps: float = 2.0,
    position_column: str = "position",
) -> tuple[pd.DataFrame, dict]:
    """Backtest direct 0-1 model position outputs."""
    required = {"date", position_column, "primary_etf"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["position"] = pd.to_numeric(out[position_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["asset_return"] = out["primary_etf"].pct_change(fill_method=None).fillna(0.0)
    out["executed_position"] = out["position"].shift(1).fillna(0.0)
    trade = out["executed_position"].diff().abs().fillna(out["executed_position"].abs())
    out["transaction_cost"] = trade * transaction_cost_bps / 10000.0
    out["strategy_return"] = out["executed_position"] * out["asset_return"] - out["transaction_cost"]
    metrics = summarize_backtest(out.assign(asset_id="gold_position_model"))
    metrics["strategy_type"] = "direct_position"
    metrics["position_column"] = position_column
    return out, metrics
