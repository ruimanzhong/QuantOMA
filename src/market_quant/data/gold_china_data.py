"""China gold ETF and SGE data adapters."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_quant.data.efinance_data import fetch_efinance_daily


CHINA_OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
SGE_COLUMNS = ["date", "sge_symbol", "sge_gold_cny_per_g"]


def fetch_china_gold_etfs(config: dict[str, Any]) -> pd.DataFrame:
    symbols = list(config.get("etf_symbols", []))
    if not symbols:
        return pd.DataFrame(columns=CHINA_OHLCV_COLUMNS)
    start = config["start_date"]
    end = config.get("end_date") or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        df = fetch_efinance_daily(symbols, start, end)
        out = df.rename(columns={"asset_id": "symbol"}).copy()
        return out[CHINA_OHLCV_COLUMNS].sort_values(["symbol", "date"])
    except Exception as exc:
        print(f"warning: efinance China gold ETF fetch failed: {exc}; trying AkShare")
    return fetch_china_gold_etfs_akshare(symbols, start, end)


def fetch_china_gold_etfs_akshare(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is required for China gold ETF fallback.") from exc
    frames = []
    for symbol in symbols:
        code = symbol.split(".")[0]
        try:
            raw = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
                end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
                adjust="qfq",
            )
            if raw is None or raw.empty:
                raise ValueError("empty AkShare ETF response")
            out = _normalize_etf_columns(raw)
            out["symbol"] = symbol
            frames.append(out[CHINA_OHLCV_COLUMNS])
        except Exception as exc:
            print(f"warning: AkShare ETF fetch failed for {symbol}: {exc}")
    if not frames:
        raise RuntimeError("No China gold ETF data fetched")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def _normalize_etf_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    out = df.rename(columns={col: rename_map.get(str(col), str(col)) for col in df.columns}).copy()
    if "date" not in out.columns:
        raise ValueError(f"ETF response missing date column. Columns: {list(df.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date")


def _normalize_akshare_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "交易日期": "date",
        "时间": "date",
        "收盘": "close",
        "收盘价": "close",
        "最新价": "close",
        "加权平均价": "close",
        "均价": "close",
        "参考价": "close",
    }
    out = df.rename(columns={col: rename_map.get(str(col), str(col)) for col in df.columns}).copy()
    if "date" not in out.columns:
        raise ValueError(f"AkShare response missing date column. Columns: {list(df.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def fetch_sge_gold(config: dict[str, Any]) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is required for SGE gold data.") from exc
    symbol = config.get("sge_symbol", "Au99.99")
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config.get("end_date") or pd.Timestamp.today().strftime("%Y-%m-%d"))
    candidates = [
        ("spot_hist_sge", {"symbol": symbol}),
        ("macro_china_shgold", {}),
    ]
    errors = []
    for func_name, kwargs in candidates:
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            raw = func(**kwargs)
            if raw is None or raw.empty:
                errors.append(f"{func_name}: empty")
                continue
            out = _normalize_akshare_columns(raw)
            if "品种" in out.columns:
                out = out[out["品种"].astype(str).str.contains(symbol, case=False, regex=False)]
            if "close" not in out.columns:
                numeric_cols = [col for col in out.columns if col != "date" and pd.to_numeric(out[col], errors="coerce").notna().any()]
                if not numeric_cols:
                    errors.append(f"{func_name}: no numeric price column")
                    continue
                out["close"] = pd.to_numeric(out[numeric_cols[0]], errors="coerce")
            result = pd.DataFrame(
                {
                    "date": out["date"],
                    "sge_symbol": symbol,
                    "sge_gold_cny_per_g": pd.to_numeric(out["close"], errors="coerce"),
                }
            )
            result = result[(result["date"] >= start) & (result["date"] <= end)]
            result = result.dropna(subset=["sge_gold_cny_per_g"]).drop_duplicates("date")
            if not result.empty:
                return result[SGE_COLUMNS].sort_values("date")
        except Exception as exc:
            errors.append(f"{func_name}: {exc}")
    raise RuntimeError(f"No supported AkShare SGE function succeeded. Tried: {errors}")
