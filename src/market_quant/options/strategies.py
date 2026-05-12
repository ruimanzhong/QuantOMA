"""Candidate option strategy generation.

The generator deliberately keeps strategy construction separate from strategy
selection.  It may emit both directional strategies and convex call+put
combinations; the selector/risk layer decides whether any candidate is worth
trading under the account-level leverage and loss budgets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_quant.options.chain import row_to_contract
from market_quant.options.contracts import OptionLeg, OptionStrategyCandidate


def _nearest_expiry_slice(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return chain
    # Prefer the nearest tradable expiry after filters.
    expiry = chain.sort_values(["dte", "expiry"])["expiry"].iloc[0]
    return chain.loc[chain["expiry"] == expiry].copy()


def _expiry_bucket_slice(
    chain: pd.DataFrame,
    min_dte: int | float | None,
    max_dte: int | float | None,
    fallback_to_nearest: bool = True,
) -> pd.DataFrame:
    if chain.empty:
        return chain
    out = chain.copy()
    dte = pd.to_numeric(out["dte"], errors="coerce")
    if min_dte is not None:
        out = out[dte >= float(min_dte)]
        dte = pd.to_numeric(out["dte"], errors="coerce")
    if max_dte is not None:
        out = out[dte <= float(max_dte)]
    if out.empty and fallback_to_nearest:
        return _nearest_expiry_slice(chain)
    return _nearest_expiry_slice(out)


def _row_by_delta_or_moneyness(
    rows: pd.DataFrame,
    target_delta: float | None,
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> pd.Series | None:
    if rows.empty:
        return None
    out = rows.copy()
    if min_strike is not None:
        out = out[out["strike"] >= min_strike]
    if max_strike is not None:
        out = out[out["strike"] <= max_strike]
    if out.empty:
        return None
    if target_delta is not None and "delta" in out.columns:
        delta = pd.to_numeric(out["delta"], errors="coerce")
        # For puts target_delta is usually negative; this also works with
        # chains that omit greeks because attach_greeks fills them before this
        # generator is called in live mode.
        idx = (delta - target_delta).abs().idxmin()
    else:
        underlying = float(pd.to_numeric(out["underlying_price"], errors="coerce").median())
        idx = (pd.to_numeric(out["strike"], errors="coerce") - underlying).abs().idxmin()
    return out.loc[idx]


def _candidate(
    strategy_type: str,
    legs: list[OptionLeg],
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> OptionStrategyCandidate:
    return OptionStrategyCandidate(
        strategy_type=strategy_type,
        legs=tuple(legs),
        reason=reason,
        diagnostics=diagnostics or {},
    )  # type: ignore[arg-type]


def generate_directional_candidates(
    chain: pd.DataFrame,
    p_up: float,
    config: dict[str, Any] | None = None,
) -> list[OptionStrategyCandidate]:
    cfg = config or {}
    expiry_cfg = cfg.get("expiry_selection", {})
    trend_chain = _expiry_bucket_slice(
        chain,
        min_dte=expiry_cfg.get("trend_min_dte", 25),
        max_dte=expiry_cfg.get("trend_max_dte", 60),
    )
    hedge_chain = _expiry_bucket_slice(
        chain,
        min_dte=expiry_cfg.get("hedge_min_dte", 20),
        max_dte=expiry_cfg.get("hedge_max_dte", 45),
    )
    calls = trend_chain[trend_chain["option_type"] == "call"].sort_values("strike")
    puts = hedge_chain[hedge_chain["option_type"] == "put"].sort_values("strike")
    candidates: list[OptionStrategyCandidate] = []

    bullish_threshold = float(cfg.get("bullish_threshold", 0.60))
    strong_bullish_threshold = float(cfg.get("strong_bullish_threshold", 0.65))
    bearish_threshold = float(cfg.get("bearish_threshold", 0.40))
    prefer_spreads = bool(cfg.get("prefer_spreads", True))
    iv_is_high = cfg.get("volatility_state") == "high_iv"

    if p_up >= bullish_threshold and not calls.empty:
        buy_row = _row_by_delta_or_moneyness(calls, target_delta=float(cfg.get("call_buy_target_delta", 0.45)))
        if buy_row is not None:
            buy_leg = OptionLeg(row_to_contract(buy_row), "buy", 1)
            diagnostics = {
                "expiry_bucket": "trend",
                "signal_horizon_days": cfg.get("signal_horizon_days", 5),
                "selected_dte": int(buy_row["dte"]),
            }
            if not prefer_spreads and p_up >= strong_bullish_threshold and not iv_is_high:
                candidates.append(_candidate("long_call", [buy_leg], "strong_bullish_probability_long_call", diagnostics))
            sell_row = _row_by_delta_or_moneyness(
                calls,
                target_delta=float(cfg.get("call_sell_target_delta", 0.25)),
                min_strike=float(buy_row["strike"]) + 1e-9,
            )
            if sell_row is not None:
                candidates.append(
                    _candidate(
                        "bull_call_spread",
                        [buy_leg, OptionLeg(row_to_contract(sell_row), "sell", 1)],
                        "bullish_probability_defined_risk_call_spread",
                        {
                            **diagnostics,
                            "sell_dte": int(sell_row["dte"]),
                            "preferred_first_version_strategy": True,
                        },
                    )
                )
            if p_up >= strong_bullish_threshold and not iv_is_high:
                candidates.append(_candidate("long_call", [buy_leg], "strong_bullish_probability_long_call_fallback", diagnostics))

    if p_up <= bearish_threshold and not puts.empty:
        buy_row = _row_by_delta_or_moneyness(puts, target_delta=float(cfg.get("put_buy_target_delta", -0.45)))
        if buy_row is not None:
            buy_leg = OptionLeg(row_to_contract(buy_row), "buy", 1)
            diagnostics = {
                "expiry_bucket": "hedge",
                "signal_horizon_days": cfg.get("signal_horizon_days", 5),
                "selected_dte": int(buy_row["dte"]),
            }
            sell_row = _row_by_delta_or_moneyness(
                puts,
                target_delta=float(cfg.get("put_sell_target_delta", -0.25)),
                max_strike=float(buy_row["strike"]) - 1e-9,
            )
            if sell_row is not None:
                candidates.append(
                    _candidate(
                        "bear_put_spread",
                        [buy_leg, OptionLeg(row_to_contract(sell_row), "sell", 1)],
                        "bearish_probability_defined_risk_put_spread",
                        diagnostics,
                    )
                )
            candidates.append(_candidate("long_put", [buy_leg], "bearish_probability_long_put_fallback", diagnostics))

    return candidates


def _should_generate_call_put_combo(
    p_up: float,
    vol_state: dict[str, Any],
    event_risk: bool,
    cfg: dict[str, Any],
) -> bool:
    if not bool(cfg.get("allow_call_put_combinations", True)):
        return False

    lower = float(cfg.get("combo_probability_lower", cfg.get("neutral_lower", 0.45)))
    upper = float(cfg.get("combo_probability_upper", cfg.get("neutral_upper", 0.55)))
    if not (lower <= p_up <= upper):
        return False

    # A long call+put combination is long gamma/vega and pays theta, so it is
    # normally only sensible when IV is cheap or there is external event risk.
    require_low_iv_or_event = bool(cfg.get("combo_requires_low_iv_or_event", True))
    if require_low_iv_or_event:
        is_low_iv = vol_state.get("volatility_state") == "low_iv"
        if not (is_low_iv or bool(event_risk)):
            return False
    return True


def _atm_call_put_rows(calls: pd.DataFrame, puts: pd.DataFrame, underlying: float) -> tuple[pd.Series | None, pd.Series | None]:
    common_strikes = sorted(set(pd.to_numeric(calls["strike"], errors="coerce")).intersection(set(pd.to_numeric(puts["strike"], errors="coerce"))))
    common_strikes = [float(k) for k in common_strikes if np.isfinite(k)]
    if common_strikes:
        atm_strike = min(common_strikes, key=lambda k: abs(k - underlying))
        return calls.loc[calls["strike"] == atm_strike].iloc[0], puts.loc[puts["strike"] == atm_strike].iloc[0]
    return _row_by_delta_or_moneyness(calls, target_delta=None), _row_by_delta_or_moneyness(puts, target_delta=None)


def generate_call_put_combo_candidates(
    chain: pd.DataFrame,
    p_up: float,
    vol_state: dict[str, Any],
    event_risk: bool,
    config: dict[str, Any] | None = None,
) -> list[OptionStrategyCandidate]:
    """Generate long call + long put combinations.

    Supported combinations:
    - long_straddle: buy ATM call and ATM put with the same expiry/strike.
    - long_strangle: buy OTM call and OTM put with the same expiry but
      different strikes.  This is usually cheaper than a straddle and fits the
      user's request to allow call+put option combinations without introducing
      naked short risk.
    """
    cfg = config or {}
    if not _should_generate_call_put_combo(p_up, vol_state, event_risk, cfg):
        return []

    expiry_cfg = cfg.get("expiry_selection", {})
    expiry_chain = _expiry_bucket_slice(
        chain,
        min_dte=expiry_cfg.get("event_min_dte", 10),
        max_dte=expiry_cfg.get("event_max_dte", 30),
    )
    calls = expiry_chain[expiry_chain["option_type"] == "call"].sort_values("strike")
    puts = expiry_chain[expiry_chain["option_type"] == "put"].sort_values("strike")
    if calls.empty or puts.empty:
        return []

    underlying = float(pd.to_numeric(expiry_chain["underlying_price"], errors="coerce").median())
    modes = cfg.get("call_put_combo_modes", ["straddle", "strangle"])
    if isinstance(modes, str):
        modes = [modes]
    modes = {str(m).lower() for m in modes}

    candidates: list[OptionStrategyCandidate] = []

    if "straddle" in modes:
        call_row, put_row = _atm_call_put_rows(calls, puts, underlying)
        if call_row is not None and put_row is not None:
            candidates.append(
                _candidate(
                    "long_straddle",
                    [OptionLeg(row_to_contract(call_row), "buy", 1), OptionLeg(row_to_contract(put_row), "buy", 1)],
                    "call_put_combo_long_straddle",
                    {
                        "event_risk": bool(event_risk),
                        "combo_type": "straddle",
                        "expiry_bucket": "event",
                        "signal_horizon_days": cfg.get("signal_horizon_days", 5),
                        "selected_dte": int(call_row["dte"]),
                    },
                )
            )

    if "strangle" in modes:
        call_row = _row_by_delta_or_moneyness(
            calls,
            target_delta=float(cfg.get("strangle_call_target_delta", 0.25)),
            min_strike=underlying + 1e-9,
        )
        put_row = _row_by_delta_or_moneyness(
            puts,
            target_delta=float(cfg.get("strangle_put_target_delta", -0.25)),
            max_strike=underlying - 1e-9,
        )
        if call_row is not None and put_row is not None:
            candidates.append(
                _candidate(
                    "long_strangle",
                    [OptionLeg(row_to_contract(call_row), "buy", 1), OptionLeg(row_to_contract(put_row), "buy", 1)],
                    "call_put_combo_long_strangle",
                    {
                        "event_risk": bool(event_risk),
                        "combo_type": "strangle",
                        "expiry_bucket": "event",
                        "signal_horizon_days": cfg.get("signal_horizon_days", 5),
                        "selected_dte": int(call_row["dte"]),
                    },
                )
            )

    return candidates


def generate_straddle_candidates(
    chain: pd.DataFrame,
    p_up: float,
    vol_state: dict[str, Any],
    event_risk: bool,
    config: dict[str, Any] | None = None,
) -> list[OptionStrategyCandidate]:
    """Backward-compatible alias for long call+put combo generation."""
    cfg = dict(config or {})
    cfg.setdefault("call_put_combo_modes", ["straddle"])
    return generate_call_put_combo_candidates(
        chain=chain,
        p_up=p_up,
        vol_state=vol_state,
        event_risk=event_risk,
        config=cfg,
    )


def generate_option_candidates(
    chain: pd.DataFrame,
    p_up: float,
    vol_state: dict[str, Any],
    event_risk: bool = False,
    config: dict[str, Any] | None = None,
) -> list[OptionStrategyCandidate]:
    if chain.empty:
        return []
    cfg = config or {}
    candidates: list[OptionStrategyCandidate] = []
    directional_cfg = {**cfg, "volatility_state": vol_state.get("volatility_state")}
    candidates.extend(generate_directional_candidates(chain, p_up, directional_cfg))
    candidates.extend(generate_call_put_combo_candidates(chain, p_up, vol_state, event_risk, cfg))
    return candidates
