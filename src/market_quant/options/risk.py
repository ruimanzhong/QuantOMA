"""Risk, leverage, and payoff utilities for option strategies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from market_quant.options.contracts import OptionLeg, OptionStrategyCandidate, OptionStrategyPlan


def leg_cash_price(leg: OptionLeg) -> float:
    return float(leg.contract.mid) * float(leg.contract.multiplier) * int(leg.quantity)


def compute_strategy_premium(legs: tuple[OptionLeg, ...] | list[OptionLeg]) -> float:
    """Net premium paid by the strategy.

    Positive means cash outflow/debit.  Negative means net credit.
    """
    total = 0.0
    for leg in legs:
        cash = leg_cash_price(leg)
        total += cash if leg.side == "buy" else -cash
    return float(total)


def compute_gross_notional(legs: tuple[OptionLeg, ...] | list[OptionLeg]) -> float:
    return float(
        sum(
            abs(int(leg.quantity))
            * float(leg.contract.underlying_price)
            * float(leg.contract.multiplier)
            for leg in legs
        )
    )


def compute_delta_notional(legs: tuple[OptionLeg, ...] | list[OptionLeg]) -> float:
    total = 0.0
    for leg in legs:
        delta = leg.contract.delta if leg.contract.delta is not None else 0.0
        total += (
            leg.signed_quantity
            * float(delta)
            * float(leg.contract.underlying_price)
            * float(leg.contract.multiplier)
        )
    return float(total)


def _terminal_payoff_one(leg: OptionLeg, underlying: float) -> float:
    c = leg.contract
    if c.option_type == "call":
        intrinsic = max(0.0, underlying - c.strike)
    else:
        intrinsic = max(0.0, c.strike - underlying)
    return leg.signed_quantity * intrinsic * c.multiplier


def terminal_strategy_profit(legs: tuple[OptionLeg, ...] | list[OptionLeg], underlying: float) -> float:
    payoff = sum(_terminal_payoff_one(leg, underlying) for leg in legs)
    return float(payoff - compute_strategy_premium(legs))


def estimate_payoff_bounds(legs: tuple[OptionLeg, ...] | list[OptionLeg]) -> tuple[float | None, float | None]:
    """Estimate max_loss/max_profit for simple defined-risk structures.

    Returns (max_loss, max_profit).  Max profit is None when materially
    unbounded on the tested range (for example long calls).
    """
    if not legs:
        return 0.0, 0.0
    s0 = float(np.median([leg.contract.underlying_price for leg in legs]))
    strikes = sorted({float(leg.contract.strike) for leg in legs})
    grid = [0.01, s0 * 0.25, s0 * 0.5, s0 * 0.75, s0, s0 * 1.25, s0 * 1.5, s0 * 2.0]
    grid.extend(strikes)
    for k in strikes:
        grid.extend([max(0.01, k * 0.99), k * 1.01])
    profits = [terminal_strategy_profit(legs, float(s)) for s in sorted(set(grid))]
    min_profit = min(profits)
    max_profit = max(profits)
    max_loss = max(0.0, -min_profit)

    # Long call style: upside can be unbounded.  Detect a positive terminal slope.
    high = max(grid)
    higher = high * 1.5
    high_profit = terminal_strategy_profit(legs, high)
    higher_profit = terminal_strategy_profit(legs, higher)
    if higher_profit > high_profit + 1e-6:
        return float(max_loss), None
    return float(max_loss), float(max_profit)


def has_naked_short_options(legs: tuple[OptionLeg, ...] | list[OptionLeg]) -> bool:
    """Conservative naked-short detector.

    A short call is considered covered only by a same-expiry long call with an
    equal/lower strike and at least as much quantity.  A short put is covered only
    by a same-expiry long put with an equal/higher strike and at least as much
    quantity.
    """
    for short in [leg for leg in legs if leg.side == "sell"]:
        c = short.contract
        covering_qty = 0
        for long_leg in [leg for leg in legs if leg.side == "buy"]:
            lc = long_leg.contract
            if lc.option_type != c.option_type or str(lc.expiry) != str(c.expiry):
                continue
            if c.option_type == "call" and lc.strike <= c.strike:
                covering_qty += int(long_leg.quantity)
            if c.option_type == "put" and lc.strike >= c.strike:
                covering_qty += int(long_leg.quantity)
        if covering_qty < int(short.quantity):
            return True
    return False


def enforce_option_risk_limits(
    candidate: OptionStrategyCandidate,
    account_equity: float,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Check leverage, loss, and naked-short limits.

    By default the 2x leverage limit is applied to delta-equivalent notional.
    Gross notional is reported and can also be limited by setting
    ``max_gross_notional_multiplier`` to a finite number.
    """
    cfg = config or {}
    risk_cfg = cfg.get("risk", cfg)
    account_equity = float(risk_cfg.get("account_equity", account_equity))
    max_delta_mult = float(risk_cfg.get("max_delta_notional_multiplier", risk_cfg.get("max_account_leverage", 2.0)))
    max_gross_mult_value = risk_cfg.get("max_gross_notional_multiplier")
    max_gross_mult = None if max_gross_mult_value in {None, "", "off"} else float(max_gross_mult_value)
    max_premium_pct = float(risk_cfg.get("max_premium_risk_pct", 0.08))
    max_trade_loss_pct = float(risk_cfg.get("max_single_trade_loss_pct", 0.03))
    allow_naked = bool(risk_cfg.get("allow_naked_short_options", False))
    allow_credit = bool(risk_cfg.get("allow_credit_spreads", False))

    legs = candidate.legs
    premium = compute_strategy_premium(legs)
    max_loss, max_profit = estimate_payoff_bounds(legs)
    gross_notional = compute_gross_notional(legs)
    delta_notional = compute_delta_notional(legs)
    delta_leverage = abs(delta_notional) / account_equity if account_equity > 0 else math.inf
    gross_leverage = gross_notional / account_equity if account_equity > 0 else math.inf
    diagnostics = {
        "premium": premium,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "gross_notional": gross_notional,
        "delta_notional": delta_notional,
        "delta_leverage": delta_leverage,
        "gross_leverage": gross_leverage,
        "account_equity": account_equity,
        "max_delta_notional_multiplier": max_delta_mult,
        "max_gross_notional_multiplier": max_gross_mult,
    }

    if account_equity <= 0:
        return False, "invalid_account_equity", diagnostics
    if has_naked_short_options(legs) and not allow_naked:
        return False, "naked_short_options_disabled", diagnostics
    if premium < 0 and not allow_credit:
        return False, "credit_spreads_disabled", diagnostics
    if max_loss is None or not np.isfinite(max_loss):
        return False, "max_loss_unavailable", diagnostics
    if max_loss > max_trade_loss_pct * account_equity:
        return False, "single_trade_max_loss_exceeded", diagnostics
    premium_at_risk = max(0.0, premium)
    if premium_at_risk > max_premium_pct * account_equity:
        return False, "premium_risk_exceeded", diagnostics
    if abs(delta_notional) > max_delta_mult * account_equity:
        return False, "delta_leverage_exceeded", diagnostics
    if max_gross_mult is not None and gross_notional > max_gross_mult * account_equity:
        return False, "gross_leverage_exceeded", diagnostics
    return True, "risk_checks_passed", diagnostics


def build_plan_from_candidate(
    candidate: OptionStrategyCandidate,
    reason: str,
    risk_diagnostics: dict[str, Any],
    expected_5d_pnl: float | None = None,
    cvar_95: float | None = None,
    score: float | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
) -> OptionStrategyPlan:
    diagnostics = {**candidate.diagnostics, **risk_diagnostics}
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    account_equity = float(risk_diagnostics.get("account_equity", 0.0))
    delta_notional = float(risk_diagnostics.get("delta_notional", 0.0))
    return OptionStrategyPlan(
        enabled=True,
        strategy_type=candidate.strategy_type,
        reason=reason,
        legs=candidate.legs,
        premium=float(risk_diagnostics.get("premium", 0.0)),
        max_loss=risk_diagnostics.get("max_loss"),
        max_profit=risk_diagnostics.get("max_profit"),
        gross_notional=float(risk_diagnostics.get("gross_notional", 0.0)),
        delta_notional=delta_notional,
        leverage=abs(delta_notional) / account_equity if account_equity > 0 else 0.0,
        expected_5d_pnl=expected_5d_pnl,
        cvar_95=cvar_95,
        score=score,
        diagnostics=diagnostics,
    )
