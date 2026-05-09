"""efinance ETF daily data adapter."""

from __future__ import annotations

import time

import pandas as pd

from market_quant.data.schema import validate_price_frame


FIELD_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "股票代码": "asset_id_raw",
}


def to_efinance_symbol(symbol: str) -> str:
    """Convert project symbol format like 510300.SH to efinance code 510300."""
    return symbol.split(".")[0]


def _date_arg(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _get_quote_history(ef, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return ef.stock.get_quote_history(
                code,
                beg=_date_arg(start_date),
                end=_date_arg(end_date),
                klt=101,
            )
        except TypeError:
            return ef.stock.get_quote_history(code, beg=_date_arg(start_date), end=_date_arg(end_date))
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(f"efinance query failed for {code}: {last_exc}") from last_exc


def fetch_efinance_daily(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ETF OHLCV daily data from efinance."""
    try:
        import efinance as ef
    except ImportError as exc:
        raise ImportError(
            "efinance is required for ETF historical data. Install with: python -m pip install efinance"
        ) from exc

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    rows: list[pd.DataFrame] = []
    empty_symbols: list[str] = []

    for symbol in symbols:
        raw = _get_quote_history(ef, to_efinance_symbol(symbol), start_date, end_date)
        if raw is None or raw.empty:
            empty_symbols.append(symbol)
            continue

        part = raw.rename(columns=FIELD_MAPPING).copy()
        missing = {"date", "open", "high", "low", "close", "volume", "amount"}.difference(part.columns)
        if missing:
            raise ValueError(f"efinance response for {symbol} missing required fields: {sorted(missing)}")

        part["date"] = pd.to_datetime(part["date"])
        part = part[(part["date"] >= start_ts) & (part["date"] <= end_ts)].copy()
        if part.empty:
            empty_symbols.append(symbol)
            continue

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            part[col] = pd.to_numeric(part[col], errors="coerce")

        part["asset_id"] = symbol
        part["source"] = "efinance"
        rows.append(part[["date", "asset_id", "open", "high", "low", "close", "volume", "amount", "source"]])

    if empty_symbols:
        raise RuntimeError(f"efinance returned no data for symbols: {', '.join(empty_symbols)}")
    if not rows:
        raise RuntimeError("efinance returned no data for all requested symbols")

    return validate_price_frame(pd.concat(rows, ignore_index=True))
