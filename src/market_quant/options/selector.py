"""Score and select option strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from market_quant.options.contracts import OptionLeg, OptionStrategyCandidate, OptionStrategyPlan
from market_quant.options.pricing import black76_price
from market_quant.options.risk import build_plan_from_candidate, enforce_option_risk_limits


def _reprice_leg_5d(leg: OptionLeg, scenario_return: float, iv_shift: float, days_forward: int = 5) -> float:
    c = leg.contract
    new_underlying = max(0.01, float(c.underlying_price) * (1.0 + scenario_return))
    new_dte = max(int(c.dte) - int(days_forward), 1)
    t = new_dte / 252.0
    iv = c.iv if c.iv is not None and c.iv > 0 else 0.20
    new_iv = max(0.01, float(iv) + float(iv_shift))
    new_price = black76_price(
        forward=new_underlying,
        strike=float(c.strike),
        rate=0.02,
        volatility=new_iv,
        time_to_expiry=t,
        option_type=c.option_type,
    )
    return float(new_price) * float(c.multiplier) * int(leg.quantity)


def strategy_mark_value_5d(legs: tuple[OptionLeg, ...], scenario_return: float, iv_shift: float) -> float:
    value = 0.0
    for leg in legs:
        leg_value = _reprice_leg_5d(leg, scenario_return=scenario_return, iv_shift=iv_shift)
        value += leg_value if leg.side == "buy" else -leg_value
    return float(value)


def scenario_pnl_distribution(
    candidate: OptionStrategyCandidate,
    p_up: float,
    vol_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    """Three-scenario 5-day PnL distribution.

    This is intentionally simple for Phase 1.  It estimates mark-to-market PnL
    after 5 trading days under down/base/up price scenarios and IV shifts.
    """
    cfg = config or {}
    forecast_rv = float(vol_state.get("forecast_rv_5d", cfg.get("fallback_rv", 0.20)))
    forecast_move = float(cfg.get("forecast_5d_move", forecast_rv * np.sqrt(5.0 / 252.0)))
    vrp = float(vol_state.get("vrp", 0.0))
    # If IV is rich versus RV, assume modest IV mean reversion; if cheap, modest IV lift.
    iv_shift_base = -0.25 * vrp
    premium = sum((leg.contract.mid * leg.contract.multiplier * leg.quantity) * (1 if leg.side == "buy" else -1) for leg in candidate.legs)
    current_value = premium

    base_weight = float(cfg.get("base_scenario_weight", 0.20))
    directional_weight = max(0.0, 1.0 - base_weight)
    up_weight = directional_weight * float(p_up)
    down_weight = directional_weight * (1.0 - float(p_up))
    scenarios = [
        {"name": "down", "ret": -forecast_move, "iv_shift": iv_shift_base + 0.02, "weight": down_weight},
        {"name": "base", "ret": 0.0, "iv_shift": iv_shift_base, "weight": base_weight},
        {"name": "up", "ret": forecast_move, "iv_shift": iv_shift_base - 0.01, "weight": up_weight},
    ]
    total_weight = sum(s["weight"] for s in scenarios) or 1.0
    pnls = []
    for s in scenarios:
        mark = strategy_mark_value_5d(candidate.legs, scenario_return=s["ret"], iv_shift=s["iv_shift"])
        pnl = mark - current_value
        s["pnl"] = float(pnl)
        s["weight"] = float(s["weight"] / total_weight)
        pnls.append(float(pnl))
    expected = float(sum(s["weight"] * s["pnl"] for s in scenarios))
    cvar_95 = float(min(pnls))
    return expected, cvar_95, scenarios


def score_option_strategy(
    candidate: OptionStrategyCandidate,
    p_up: float,
    vol_state: dict[str, Any],
    account_equity: float,
    config: dict[str, Any] | None = None,
) -> tuple[OptionStrategyPlan | None, str]:
    cfg = config or {}
    ok, risk_reason, risk_diag = enforce_option_risk_limits(candidate, account_equity=account_equity, config=cfg)
    if not ok:
        return None, risk_reason
    expected, cvar_95, scenarios = scenario_pnl_distribution(candidate, p_up=p_up, vol_state=vol_state, config=cfg.get("scenario", {}))
    max_loss = float(risk_diag.get("max_loss", 0.0) or 0.0)
    account = float(risk_diag.get("account_equity", account_equity))
    liquidity_penalty = 0.0
    for leg in candidate.legs:
        source = leg.contract.diagnostics.get("pricing_source")
        if source == "settlement_fallback":
            liquidity_penalty += float(cfg.get("settlement_fallback_penalty", 0.05))
        if leg.contract.bid and leg.contract.ask and leg.contract.mid > 0:
            liquidity_penalty += max(0.0, (leg.contract.ask - leg.contract.bid) / leg.contract.mid) * 0.05
    loss_denom = max(max_loss, 1.0)
    score = expected / loss_denom
    score -= float(cfg.get("cvar_penalty", 0.25)) * max(0.0, -cvar_95) / max(account, 1.0)
    score -= liquidity_penalty
    score -= float(cfg.get("leverage_penalty", 0.02)) * float(risk_diag.get("delta_leverage", 0.0))
    plan = build_plan_from_candidate(
        candidate,
        reason=candidate.reason,
        risk_diagnostics=risk_diag,
        expected_5d_pnl=expected,
        cvar_95=cvar_95,
        score=float(score),
        extra_diagnostics={"scenario_pnls": scenarios, "risk_reason": risk_reason},
    )
    return plan, "scored"


def select_best_option_strategy(
    candidates: list[OptionStrategyCandidate],
    p_up: float,
    vol_state: dict[str, Any],
    account_equity: float,
    config: dict[str, Any] | None = None,
) -> OptionStrategyPlan:
    cfg = config or {}
    min_edge_score = float(cfg.get("min_edge_score", 0.02))
    rejected: list[dict[str, Any]] = []
    plans: list[OptionStrategyPlan] = []
    for candidate in candidates:
        plan, reason = score_option_strategy(candidate, p_up=p_up, vol_state=vol_state, account_equity=account_equity, config=cfg)
        if plan is None:
            rejected.append({"strategy_type": candidate.strategy_type, "reason": reason})
        else:
            plans.append(plan)
    if not plans:
        return OptionStrategyPlan(
            enabled=False,
            strategy_type="no_trade",
            reason="no_candidate_passed_risk_checks",
            diagnostics={"rejected_candidates": rejected, **vol_state},
        )
    best = max(plans, key=lambda p: p.score if p.score is not None else -1e9)
    if best.score is None or best.score <= min_edge_score:
        return OptionStrategyPlan(
            enabled=False,
            strategy_type="no_trade",
            reason="option_edge_below_threshold",
            diagnostics={
                "best_rejected_score": best.score,
                "best_rejected_strategy": best.strategy_type,
                "rejected_candidates": rejected,
                **vol_state,
            },
        )
    return best
