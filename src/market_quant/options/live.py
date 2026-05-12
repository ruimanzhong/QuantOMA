"""Live side-car integration for gold option execution plans."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_quant.options.chain import filter_tradeable_chain
from market_quant.options.greeks import attach_greeks
from market_quant.options.selector import select_best_option_strategy
from market_quant.options.strategies import generate_option_candidates
from market_quant.options.volatility import build_option_vol_state

DEFAULT_OPTIONS_CONFIG: dict[str, Any] = {
    "enabled": False,
    "signal_horizon_days": 5,
    "account_equity": 100000.0,
    "risk_free_rate": 0.02,
    "risk": {
        "max_account_leverage": 2.0,
        "max_delta_notional_multiplier": 2.0,
        # Gross notional can be much larger than 2x for one SHFE gold option
        # contract. Leave unset by default and use delta-equivalent leverage.
        "max_gross_notional_multiplier": None,
        "max_premium_risk_pct": 0.08,
        "max_single_trade_loss_pct": 0.03,
        "allow_naked_short_options": False,
        "allow_credit_spreads": False,
    },
    "chain_filter": {
        "min_dte": 10,
        "max_dte": 90,
        "min_volume": 0,
        "min_open_interest": 0,
        "max_bid_ask_spread_pct": 0.20,
    },
    "expiry_selection": {
        "trend_min_dte": 25,
        "trend_max_dte": 60,
        "event_min_dte": 10,
        "event_max_dte": 30,
        "hedge_min_dte": 20,
        "hedge_max_dte": 45,
    },
    "holding": {
        "max_holding_days": 5,
        "reevaluate_daily": True,
        "close_on_signal_flip": True,
        "close_on_edge_decay": True,
    },
    "exits": {
        "long_option_take_profit_pct": 0.40,
        "long_option_stop_loss_pct": 0.35,
        "spread_take_profit_of_max_profit": 0.50,
        "spread_stop_loss_of_max_loss": 0.35,
        "signal_flip_probability": 0.52,
        "theta_to_premium_exit_threshold": 0.08,
    },
    "strategy": {
        "bullish_threshold": 0.60,
        "strong_bullish_threshold": 0.65,
        "bearish_threshold": 0.40,
        "neutral_lower": 0.45,
        "neutral_upper": 0.55,
        "prefer_spreads": True,
        # Enable convex call+put combinations such as long straddle and
        # long strangle. These remain defined-risk because both legs are long.
        "allow_call_put_combinations": True,
        "call_put_combo_modes": ["straddle", "strangle"],
        "combo_probability_lower": 0.45,
        "combo_probability_upper": 0.55,
        "combo_requires_low_iv_or_event": True,
        "strangle_call_target_delta": 0.25,
        "strangle_put_target_delta": -0.25,
    },
    "selector": {
        "min_edge_score": 0.02,
    },
    "volatility": {
        "rv_window": 20,
        "fallback_rv": 0.20,
        "low_iv_threshold": 0.16,
        "high_iv_threshold": 0.28,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _get_signal_value(gold_signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(gold_signal, dict):
        return gold_signal.get(name, default)
    return getattr(gold_signal, name, default)


def _infer_event_risk(gold_signal: Any, config: dict[str, Any]) -> bool:
    if "event_risk" in config:
        return bool(config["event_risk"])
    diagnostics = _get_signal_value(gold_signal, "diagnostics", {}) or {}
    reasons = diagnostics.get("live_overlay_reasons") or diagnostics.get("overlay_reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    event_keywords = {"event", "war", "fed", "fomc", "geopolitical", "holiday", "gap", "polymarket", "news", "crisis"}
    return any(any(keyword in str(reason).lower() for keyword in event_keywords) for reason in reasons)


def _serialize_plan(plan, config: dict[str, Any]) -> dict[str, Any]:
    def serialize_leg(leg) -> dict[str, Any]:
        c = leg.contract
        return {
            "symbol": c.symbol,
            "underlying_symbol": c.underlying_symbol,
            "side": leg.side,
            "quantity": int(leg.quantity),
            "option_type": c.option_type,
            "strike": float(c.strike),
            "expiry": str(pd.Timestamp(c.expiry).date()) if c.expiry is not None else None,
            "dte": int(c.dte),
            "underlying_price": float(c.underlying_price),
            "mid": float(c.mid),
            "bid": c.bid,
            "ask": c.ask,
            "iv": c.iv,
            "delta": c.delta,
            "gamma": c.gamma,
            "vega": c.vega,
            "theta": c.theta,
            "multiplier": float(c.multiplier),
            "diagnostics": c.diagnostics,
        }

    return {
        "enabled": bool(plan.enabled),
        "strategy_type": plan.strategy_type,
        "reason": plan.reason,
        "signal_horizon_days": int(config.get("signal_horizon_days", 5)),
        "holding": config.get("holding", {}),
        "exits": config.get("exits", {}),
        "account_equity": float(config.get("account_equity", 100000.0)),
        "max_account_leverage": float(config.get("risk", {}).get("max_account_leverage", 2.0)),
        "premium": float(plan.premium),
        "max_loss": None if plan.max_loss is None else float(plan.max_loss),
        "max_profit": None if plan.max_profit is None else float(plan.max_profit),
        "gross_notional": float(plan.gross_notional),
        "delta_notional": float(plan.delta_notional),
        "leverage": float(plan.leverage),
        "expected_5d_pnl": None if plan.expected_5d_pnl is None else float(plan.expected_5d_pnl),
        "cvar_95": None if plan.cvar_95 is None else float(plan.cvar_95),
        "score": None if plan.score is None else float(plan.score),
        "legs": [serialize_leg(leg) for leg in plan.legs],
        "diagnostics": plan.diagnostics,
    }


def build_live_option_plan(
    gold_signal: Any,
    feature_frame: pd.DataFrame,
    option_chain: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an optional live options plan from the existing gold signal.

    This function does not mutate the existing gold signal or linear execution
    output.  It returns a serializable dictionary that can be attached under
    payload["options"].
    """
    cfg = _deep_merge(DEFAULT_OPTIONS_CONFIG, config or {})
    if not cfg.get("enabled", False):
        return {"enabled": False, "reason": "disabled"}
    if option_chain is None or option_chain.empty:
        return {"enabled": False, "reason": "option_chain_not_supplied"}

    chain_filter = cfg.get("chain_filter", {})
    chain = filter_tradeable_chain(
        option_chain,
        min_dte=int(chain_filter.get("min_dte", 10)),
        max_dte=int(chain_filter.get("max_dte", 90)),
        min_volume=float(chain_filter.get("min_volume", 0)),
        min_open_interest=float(chain_filter.get("min_open_interest", 0)),
        max_bid_ask_spread_pct=chain_filter.get("max_bid_ask_spread_pct", 0.20),
    )
    if chain.empty:
        return {"enabled": False, "reason": "no_tradeable_option_contracts"}

    vol_state = build_option_vol_state(
        chain,
        feature_frame=feature_frame,
        feature_date=_get_signal_value(gold_signal, "feature_date"),
        config=cfg,
    )
    chain = attach_greeks(
        chain,
        risk_free_rate=float(cfg.get("risk_free_rate", 0.02)),
        fallback_iv=float(vol_state.get("iv_atm", cfg.get("volatility", {}).get("fallback_rv", 0.20))),
    )
    probability = float(_get_signal_value(gold_signal, "probability", 0.5))
    event_risk = _infer_event_risk(gold_signal, cfg)
    strategy_cfg = _deep_merge(
        cfg.get("strategy", {}),
        {
            "expiry_selection": cfg.get("expiry_selection", {}),
            "signal_horizon_days": cfg.get("signal_horizon_days", 5),
        },
    )
    candidates = generate_option_candidates(
        chain,
        p_up=probability,
        vol_state=vol_state,
        event_risk=event_risk,
        config=strategy_cfg,
    )
    if not candidates:
        return {
            "enabled": False,
            "strategy_type": "no_trade",
            "reason": "no_strategy_candidate_for_current_signal_state",
            "signal_horizon_days": int(cfg.get("signal_horizon_days", 5)),
            "holding": cfg.get("holding", {}),
            "exits": cfg.get("exits", {}),
            "diagnostics": {
                "probability": probability,
                "event_risk": event_risk,
                "n_tradeable_contracts": int(len(chain)),
                **vol_state,
            },
        }

    selector_cfg = _deep_merge(cfg.get("selector", {}), {"risk": cfg.get("risk", {}), "account_equity": cfg.get("account_equity")})
    plan = select_best_option_strategy(
        candidates,
        p_up=probability,
        vol_state=vol_state,
        account_equity=float(cfg.get("account_equity", 100000.0)),
        config=selector_cfg,
    )
    serialized = _serialize_plan(plan, cfg)
    if not serialized.get("enabled", False):
        serialized.setdefault("signal_horizon_days", int(cfg.get("signal_horizon_days", 5)))
        serialized.setdefault("holding", cfg.get("holding", {}))
        serialized.setdefault("exits", cfg.get("exits", {}))
    serialized.setdefault("diagnostics", {})
    serialized["diagnostics"].update(
        {
            "probability": probability,
            "linear_final_position": _get_signal_value(gold_signal, "final_position"),
            "regime_score": _get_signal_value(gold_signal, "regime_score"),
            "regime_label": _get_signal_value(gold_signal, "regime_label"),
            "event_risk": event_risk,
            "n_tradeable_contracts": int(len(chain)),
            **vol_state,
        }
    )
    return serialized
