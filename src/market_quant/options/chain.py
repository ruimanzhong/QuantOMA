"""Option chain loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_quant.options.contracts import OptionContract

_COLUMN_ALIASES = {
    "trade_date": "date",
    "datetime": "date",
    "ts_code": "symbol",
    "code": "symbol",
    "option_code": "symbol",
    "underlying": "underlying_symbol",
    "underlying_code": "underlying_symbol",
    "cp": "option_type",
    "call_put": "option_type",
    "type": "option_type",
    "maturity": "expiry",
    "expire_date": "expiry",
    "expiration": "expiry",
    "exercise_price": "strike",
    "exerciseprice": "strike",
    "underlying_close": "underlying_price",
    "futures_price": "underlying_price",
    "close": "settlement",
    "last": "settlement",
    "settle": "settlement",
    "settle_price": "settlement",
    "oi": "open_interest",
    "openint": "open_interest",
    "implied_vol": "iv",
    "implied_volatility": "iv",
    "contract_multiplier": "multiplier",
}

_REQUIRED_COLUMNS = {"date", "symbol", "option_type", "expiry", "strike", "underlying_price"}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename: dict[str, str] = {}
    lower_to_original = {str(c).lower(): c for c in out.columns}
    for lower, original in lower_to_original.items():
        if lower in _COLUMN_ALIASES:
            rename[original] = _COLUMN_ALIASES[lower]
    return out.rename(columns=rename)


def _normalize_option_type(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"c", "call", "认购", "购", "看涨"}:
        return "call"
    if text in {"p", "put", "认沽", "沽", "看跌"}:
        return "put"
    raise ValueError(f"unsupported option_type value: {value!r}")


def normalize_option_chain(df: pd.DataFrame, default_multiplier: float = 1000.0) -> pd.DataFrame:
    """Normalize raw option-chain data into the schema expected by the overlay.

    Supported input columns include common aliases.  Canonical columns are:
    date, symbol, underlying_symbol, option_type, expiry, strike,
    underlying_price, bid, ask, settlement, volume, open_interest, iv,
    multiplier.
    """
    if df.empty:
        return pd.DataFrame()

    out = _canonicalize_columns(df)
    if "underlying_symbol" not in out.columns:
        out["underlying_symbol"] = "UNKNOWN"
    missing = sorted(_REQUIRED_COLUMNS - set(out.columns))
    if missing:
        raise ValueError(f"option chain missing required columns: {missing}")

    out = out.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "expiry"])
    out["option_type"] = out["option_type"].map(_normalize_option_type)

    numeric_cols = [
        "strike",
        "underlying_price",
        "bid",
        "ask",
        "settlement",
        "volume",
        "open_interest",
        "iv",
        "multiplier",
        "dte",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "multiplier" not in out.columns:
        out["multiplier"] = float(default_multiplier)
    out["multiplier"] = out["multiplier"].fillna(float(default_multiplier))

    if "dte" not in out.columns:
        out["dte"] = (out["expiry"] - out["date"]).dt.days
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce").astype("Int64")

    bid = pd.to_numeric(out["bid"], errors="coerce") if "bid" in out else pd.Series(np.nan, index=out.index)
    ask = pd.to_numeric(out["ask"], errors="coerce") if "ask" in out else pd.Series(np.nan, index=out.index)
    settlement = pd.to_numeric(out["settlement"], errors="coerce") if "settlement" in out else pd.Series(np.nan, index=out.index)
    mid_from_bid_ask = (bid + ask) / 2.0
    out["mid"] = mid_from_bid_ask.where(bid.notna() & ask.notna() & (bid > 0) & (ask > 0), settlement)
    out["pricing_source"] = np.where(
        bid.notna() & ask.notna() & (bid > 0) & (ask > 0),
        "bid_ask_mid",
        "settlement_fallback",
    )
    out = out.dropna(subset=["strike", "underlying_price", "mid", "dte"])
    out = out[(out["strike"] > 0) & (out["underlying_price"] > 0) & (out["mid"] > 0) & (out["dte"] > 0)]
    return out.sort_values(["date", "expiry", "option_type", "strike", "symbol"]).reset_index(drop=True)


def load_option_chain(path: str | Path, as_of_date: str | pd.Timestamp | None = None, default_multiplier: float = 1000.0) -> pd.DataFrame:
    """Load and normalize an option chain CSV.

    If ``as_of_date`` is provided, the latest chain date on or before that date
    is returned.  This avoids look-ahead in live runs.
    """
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    raw = pd.read_csv(target)
    chain = normalize_option_chain(raw, default_multiplier=default_multiplier)
    if chain.empty or as_of_date is None:
        return chain
    cutoff = pd.Timestamp(as_of_date).normalize()
    chain = chain.loc[chain["date"] <= cutoff].copy()
    if chain.empty:
        return chain
    latest = chain["date"].max()
    return chain.loc[chain["date"] == latest].reset_index(drop=True)


def filter_tradeable_chain(
    chain: pd.DataFrame,
    min_dte: int = 10,
    max_dte: int = 90,
    min_volume: float = 0,
    min_open_interest: float = 0,
    max_bid_ask_spread_pct: float | None = 0.20,
) -> pd.DataFrame:
    """Apply liquidity and maturity filters to a normalized option chain."""
    if chain.empty:
        return chain
    out = chain.copy()
    out = out[(out["dte"].astype(float) >= min_dte) & (out["dte"].astype(float) <= max_dte)]
    if "volume" in out.columns and min_volume > 0:
        out = out[out["volume"].fillna(0) >= min_volume]
    if "open_interest" in out.columns and min_open_interest > 0:
        out = out[out["open_interest"].fillna(0) >= min_open_interest]
    if max_bid_ask_spread_pct is not None and "bid" in out.columns and "ask" in out.columns:
        bid = pd.to_numeric(out["bid"], errors="coerce")
        ask = pd.to_numeric(out["ask"], errors="coerce")
        mid = pd.to_numeric(out["mid"], errors="coerce")
        spread_pct = (ask - bid) / mid.replace(0, np.nan)
        # Keep settlement-only rows; they will be penalized later.
        out = out[(spread_pct.isna()) | (spread_pct <= float(max_bid_ask_spread_pct))]
    return out.reset_index(drop=True)


def row_to_contract(row: pd.Series) -> OptionContract:
    """Convert a normalized chain row into an OptionContract dataclass."""
    def optional_float(name: str) -> float | None:
        if name not in row or pd.isna(row[name]):
            return None
        return float(row[name])

    return OptionContract(
        date=row.get("date"),
        symbol=str(row.get("symbol")),
        underlying_symbol=str(row.get("underlying_symbol", "UNKNOWN")),
        option_type=str(row.get("option_type")),  # type: ignore[arg-type]
        expiry=row.get("expiry"),
        strike=float(row.get("strike")),
        dte=int(row.get("dte")),
        underlying_price=float(row.get("underlying_price")),
        bid=optional_float("bid"),
        ask=optional_float("ask"),
        settlement=optional_float("settlement"),
        mid=float(row.get("mid")),
        volume=optional_float("volume"),
        open_interest=optional_float("open_interest"),
        iv=optional_float("iv"),
        multiplier=float(row.get("multiplier", 1000.0)),
        delta=optional_float("delta"),
        gamma=optional_float("gamma"),
        vega=optional_float("vega"),
        theta=optional_float("theta"),
        diagnostics={"pricing_source": str(row.get("pricing_source", "unknown"))},
    )
