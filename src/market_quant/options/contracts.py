"""Typed contracts for the gold options overlay module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OptionType = Literal["call", "put"]
OptionSide = Literal["buy", "sell"]
StrategyType = Literal[
    "no_trade",
    "long_call",
    "long_put",
    "bull_call_spread",
    "bear_put_spread",
    "long_straddle",
    "long_strangle",
    "call_put_combo",
]


@dataclass(frozen=True)
class OptionContract:
    """A normalized option contract row.

    Prices are assumed to be quoted per gram for SHFE-style gold options.
    ``multiplier`` converts per-gram option premium into cash value.
    """

    date: Any
    symbol: str
    underlying_symbol: str
    option_type: OptionType
    expiry: Any
    strike: float
    dte: int
    underlying_price: float
    mid: float
    bid: float | None = None
    ask: float | None = None
    settlement: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    iv: float | None = None
    multiplier: float = 1000.0
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionLeg:
    contract: OptionContract
    side: OptionSide
    quantity: int = 1

    @property
    def signed_quantity(self) -> int:
        return int(self.quantity) if self.side == "buy" else -int(self.quantity)


@dataclass(frozen=True)
class OptionStrategyCandidate:
    strategy_type: StrategyType
    legs: tuple[OptionLeg, ...]
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionStrategyPlan:
    enabled: bool
    strategy_type: str
    reason: str
    legs: tuple[OptionLeg, ...] = ()
    premium: float = 0.0
    max_loss: float | None = None
    max_profit: float | None = None
    gross_notional: float = 0.0
    delta_notional: float = 0.0
    leverage: float = 0.0
    expected_5d_pnl: float | None = None
    cvar_95: float | None = None
    score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
