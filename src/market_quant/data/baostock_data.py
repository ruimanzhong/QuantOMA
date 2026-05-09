"""BaoStock A-share/ETF daily data adapter."""

from __future__ import annotations

import pandas as pd

from market_quant.data.schema import validate_price_frame


DEFAULT_A_SHARE_ETF_SYMBOLS = [
    "510300.SH",
    "510050.SH",
    "510500.SH",
    "159915.SZ",
    "588000.SH",
    "518880.SH",
]


def to_baostock_symbol(symbol: str) -> str:
    """Convert project symbol format like 510300.SH to BaoStock sh.510300."""
    code, suffix = symbol.split(".")
    return f"{suffix.lower()}.{code}"


def _adjust_flag(adjust: str) -> str:
    mapping = {"": "", "none": "", "qfq": "2", "hfq": "1", "post": "1", "pre": "2"}
    key = adjust.lower()
    if key not in mapping:
        raise ValueError(f"unsupported adjust value {adjust!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def fetch_baostock_daily(
    symbols: list[str] | None = None,
    start_date: str = "2015-01-01",
    end_date: str = "2026-04-30",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch daily A-share/ETF OHLCV data from BaoStock."""
    try:
        import baostock as bs
    except ImportError as exc:
        raise ImportError(
            "baostock is required to fetch A-share data. Install with: "
            "python -m pip install baostock"
        ) from exc

    symbols = symbols or DEFAULT_A_SHARE_ETF_SYMBOLS
    fields = "date,code,open,high,low,close,volume,amount"
    rows: list[pd.DataFrame] = []

    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")

    try:
        for symbol in symbols:
            rs = bs.query_history_k_data_plus(
                to_baostock_symbol(symbol),
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=_adjust_flag(adjust),
            )
            if getattr(rs, "error_code", "0") != "0":
                raise RuntimeError(f"BaoStock query failed for {symbol}: {rs.error_msg}")

            records = []
            while rs.next():
                records.append(rs.get_row_data())
            if not records:
                continue

            part = pd.DataFrame(records, columns=rs.fields)
            part = part.rename(columns={"code": "baostock_code"})
            part["asset_id"] = symbol
            part["source"] = "baostock"
            rows.append(part)
    finally:
        bs.logout()

    if not rows:
        columns = ["date", "asset_id", "open", "high", "low", "close", "volume", "amount", "source"]
        return pd.DataFrame(columns=columns)

    out = pd.concat(rows, ignore_index=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[["date", "asset_id", "open", "high", "low", "close", "volume", "amount", "source"]]
    return validate_price_frame(out)
