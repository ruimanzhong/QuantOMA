"""Black-76 Greeks for normalized option-chain rows."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from market_quant.options.pricing import normal_cdf, normal_pdf


def black76_greeks(
    forward: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
    option_type: str,
) -> dict[str, float]:
    """Return Black-76 delta, gamma, vega, theta per unit quote.

    Theta is returned as annual theta in quote units; downstream code uses it
    only as a diagnostic.  Scenario re-pricing is used for PnL estimation.
    """
    if forward <= 0 or strike <= 0 or volatility <= 0 or time_to_expiry <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount = math.exp(-rate * time_to_expiry)
    pdf = normal_pdf(d1)
    kind = option_type.lower()
    if kind == "call":
        delta = discount * normal_cdf(d1)
        theta = (
            -discount * forward * pdf * volatility / (2.0 * sqrt_t)
            + rate * discount * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
        )
    elif kind == "put":
        delta = -discount * normal_cdf(-d1)
        theta = (
            -discount * forward * pdf * volatility / (2.0 * sqrt_t)
            + rate * discount * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))
        )
    else:
        raise ValueError(f"unsupported option_type: {option_type!r}")
    gamma = discount * pdf / (forward * vol_sqrt_t)
    vega = discount * forward * pdf * sqrt_t
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


def attach_greeks(chain: pd.DataFrame, risk_free_rate: float = 0.02, fallback_iv: float = 0.20) -> pd.DataFrame:
    """Attach Black-76 Greeks to a normalized option chain."""
    if chain.empty:
        return chain
    out = chain.copy()
    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        iv = row.get("iv")
        if pd.isna(iv) or float(iv) <= 0:
            iv = fallback_iv
        t = max(float(row.get("dte", 0)) / 252.0, 1.0 / 252.0)
        rows.append(
            black76_greeks(
                forward=float(row["underlying_price"]),
                strike=float(row["strike"]),
                rate=float(risk_free_rate),
                volatility=float(iv),
                time_to_expiry=t,
                option_type=str(row["option_type"]),
            )
        )
    greek_df = pd.DataFrame(rows, index=out.index)
    for col in ["delta", "gamma", "vega", "theta"]:
        out[col] = pd.to_numeric(greek_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
