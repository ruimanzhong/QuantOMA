"""Black-76 pricing utilities for futures-style gold options."""

from __future__ import annotations

import math

SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate_inputs(forward: float, strike: float, volatility: float, time_to_expiry: float) -> bool:
    return forward > 0 and strike > 0 and volatility > 0 and time_to_expiry > 0


def black76_price(
    forward: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
    option_type: str,
) -> float:
    """Return Black-76 option price per unit of underlying quote.

    The output is in the same unit as ``forward`` and ``strike``; for SHFE gold
    options this is typically RMB/gram.
    """
    if not _validate_inputs(forward, strike, volatility, time_to_expiry):
        return max(0.0, forward - strike) if option_type == "call" else max(0.0, strike - forward)
    vol_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount = math.exp(-rate * time_to_expiry)
    kind = option_type.lower()
    if kind == "call":
        return discount * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    if kind == "put":
        return discount * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))
    raise ValueError(f"unsupported option_type: {option_type!r}")


def implied_volatility_black76(
    market_price: float,
    forward: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: str,
    min_vol: float = 1e-4,
    max_vol: float = 5.0,
    max_iter: int = 80,
    tolerance: float = 1e-6,
) -> float | None:
    """Infer Black-76 implied volatility by bisection.

    Returns None when the market price is outside no-arbitrage bounds or inputs
    are invalid.
    """
    if market_price <= 0 or forward <= 0 or strike <= 0 or time_to_expiry <= 0:
        return None
    intrinsic = black76_price(forward, strike, rate, min_vol, time_to_expiry, option_type)
    high_price = black76_price(forward, strike, rate, max_vol, time_to_expiry, option_type)
    if market_price < intrinsic - tolerance or market_price > high_price + tolerance:
        return None
    lo, hi = min_vol, max_vol
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = black76_price(forward, strike, rate, mid, time_to_expiry, option_type)
        if abs(price - market_price) <= tolerance:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
