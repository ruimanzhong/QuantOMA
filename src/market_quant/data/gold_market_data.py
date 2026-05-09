"""International gold and FX market data adapters."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


STANDARD_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename = {col: str(col).lower().replace(" ", "_") for col in df.columns}
    out = df.rename(columns=rename).copy()
    if "adj_close" in out.columns and "close" not in out.columns:
        out["close"] = out["adj_close"]
    if "date" not in out.columns:
        out = out.reset_index().rename(columns={"index": "date"})
    out = out.rename(columns={col: "date" for col in out.columns if str(col).lower() in {"date", "datetime"}})
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    out["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[STANDARD_COLUMNS].dropna(subset=["date", "close"]).sort_values(["symbol", "date"])


def _date_chunks(start: str, end: str | None) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    chunks = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(pd.Timestamp(chunk_start.year, 12, 31), end_ts)
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return chunks


def _fetch_tiingo_json(url: str, params: dict[str, Any], api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        url,
        params=params,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Tiingo response type: {type(payload).__name__}")
    return payload


def _fetch_tiingo_chunked(url: str, params: dict[str, Any], api_key: str, start: str, end: str | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _date_chunks(start, end):
        try:
            payload.extend(_fetch_tiingo_json(url, {**params, "startDate": chunk_start, "endDate": chunk_end}, api_key))
        except requests.RequestException as exc:
            print(f"warning: Tiingo chunk failed for {chunk_start} to {chunk_end}: {exc}")
    return payload


def fetch_tiingo_symbol(alias: str, ticker: str, config: dict[str, Any], start: str, end: str | None) -> pd.DataFrame:
    api_key_env = config.get("api_key_env", "TIINGO_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing Tiingo API key env var: {api_key_env}")
    base_url = config["base_url"].rstrip("/")
    if alias.startswith("usd_"):
        endpoint = config["fx_endpoint"].format(ticker=ticker)
        payload = _fetch_tiingo_chunked(f"{base_url}{endpoint}", {"resampleFreq": "1day"}, api_key, start, end)
        frame = pd.DataFrame(payload)
        if "close" not in frame.columns and "rate" in frame.columns:
            frame["close"] = frame["rate"]
    else:
        endpoint = config["price_endpoint"].format(ticker=ticker)
        frame = pd.DataFrame(_fetch_tiingo_chunked(f"{base_url}{endpoint}", {}, api_key, start, end))
    if frame.empty:
        raise ValueError(f"No Tiingo data returned for {ticker}")
    return normalize_ohlcv(frame, alias)


def fetch_yfinance_symbol(alias: str, ticker: str, start: str, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for yfinance fallback.") from exc
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No yfinance data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return normalize_ohlcv(data, alias)


def fetch_gold_market_data(config: dict[str, Any]) -> pd.DataFrame:
    provider = config.get("provider", "tiingo")
    fallback_provider = config.get("fallback_provider", "yfinance")
    start = config["start_date"]
    end = config.get("end_date")
    rows = []
    for alias, symbol_cfg in config.get("symbols", {}).items():
        try:
            if provider == "tiingo":
                rows.append(fetch_tiingo_symbol(alias, symbol_cfg["tiingo"], config["tiingo"], start, end))
            elif provider == "yfinance":
                rows.append(fetch_yfinance_symbol(alias, symbol_cfg["yfinance"], start, end))
        except Exception as exc:
            if fallback_provider != "yfinance" or not symbol_cfg.get("yfinance"):
                print(f"warning: market fetch failed for {alias}: {exc}")
                continue
            print(f"warning: primary market fetch failed for {alias}: {exc}; falling back to yfinance")
            try:
                rows.append(fetch_yfinance_symbol(alias, symbol_cfg["yfinance"], start, end))
            except Exception as fallback_exc:
                print(f"warning: fallback market fetch failed for {alias}: {fallback_exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=STANDARD_COLUMNS)
