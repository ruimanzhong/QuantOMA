"""AU9999 night-session quote helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_quant.data.network_proxy import scoped_proxy_environment


INTRADAY_COLUMNS = ["trade_date", "time", "timestamp_utc", "symbol", "price", "updated_at_utc", "source"]
NIGHT_DECISION_COLUMNS = [
    "decision_date",
    "decision_time",
    "timezone",
    "timestamp_utc",
    "symbol",
    "price",
    "source",
    "updated_at_utc",
]


def _parse_sge_updated_at(value: object) -> pd.Timestamp:
    text = str(value)
    if not text or text.lower() in {"nat", "nan", "none"}:
        return pd.NaT
    normalized = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .strip()
    )
    return pd.to_datetime(normalized, errors="coerce")


def normalize_sge_intraday_quotes(raw: pd.DataFrame, symbol: str = "Au99.99", timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Normalize AkShare SGE quote rows into timestamped intraday prices."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    rename_map = {
        "品种": "symbol",
        "时间": "time",
        "现价": "price",
        "更新时间": "updated_at",
    }
    out = raw.rename(columns={col: rename_map.get(str(col), str(col)) for col in raw.columns}).copy()
    missing = {"time", "price"}.difference(out.columns)
    if missing:
        raise ValueError(f"SGE intraday quote response missing columns: {sorted(missing)}")
    if "symbol" not in out:
        out["symbol"] = symbol
    if "updated_at" not in out:
        out["updated_at"] = pd.NaT

    out["symbol"] = out["symbol"].astype(str)
    out = out[out["symbol"].str.contains(symbol, case=False, regex=False)].copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["updated_at"] = out["updated_at"].map(_parse_sge_updated_at)
    updated_dates = out["updated_at"].dropna()
    trade_date = updated_dates.max().normalize() if not updated_dates.empty else pd.Timestamp.today().normalize()
    out["trade_date"] = trade_date
    out["time"] = out["time"].astype(str)
    timestamp_local = pd.to_datetime(
        out["trade_date"].dt.strftime("%Y-%m-%d") + " " + out["time"],
        errors="coerce",
    )
    out["timestamp_utc"] = timestamp_local.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    out["updated_at_utc"] = out["updated_at"].dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    out = out.dropna(subset=["timestamp_utc", "price"]).drop_duplicates(["symbol", "timestamp_utc"], keep="last")
    out["source"] = "akshare_spot_quotations_sge"
    return out[INTRADAY_COLUMNS].sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def fetch_sge_intraday_quotes(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fetch current-day SGE Au99.99 minute quotes from AkShare."""
    cfg = config or {}
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is required for SGE intraday quote data.") from exc
    symbol = cfg.get("sge_symbol", "Au99.99")
    timezone = cfg.get("timezone", "Asia/Shanghai")
    with scoped_proxy_environment(cfg.get("network"), label="sge_intraday"):
        raw = ak.spot_quotations_sge(symbol=symbol)
    return normalize_sge_intraday_quotes(raw, symbol=symbol, timezone=timezone)


def select_decision_price(
    quotes: pd.DataFrame,
    decision_time: str = "23:00",
    tolerance_minutes: int = 10,
    timezone: str = "Asia/Shanghai",
) -> pd.Series:
    """Select the quote nearest to decision_time within a tolerance window."""
    if quotes.empty:
        raise ValueError("no SGE intraday quotes available")
    df = quotes.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc", "price"]).sort_values("timestamp_utc")
    if df.empty:
        raise ValueError("no valid SGE intraday quotes available")
    trade_date = pd.to_datetime(df["trade_date"], errors="coerce").dropna().max().normalize()
    target_local = pd.Timestamp(f"{trade_date.date()} {decision_time}", tz=timezone)
    target_utc = target_local.tz_convert("UTC")
    delta = (df["timestamp_utc"] - target_utc).abs()
    idx = delta.idxmin()
    if delta.loc[idx] > pd.Timedelta(minutes=tolerance_minutes):
        raise ValueError(
            f"nearest quote is {delta.loc[idx]} away from decision_time={decision_time}; "
            f"increase tolerance or run closer to the decision time"
        )
    return df.loc[idx]


def build_night_decision_row(
    quote: pd.Series,
    decision_time: str = "23:00",
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Convert a selected quote into the persistent night decision schema."""
    timestamp_utc = pd.Timestamp(quote["timestamp_utc"])
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.tz_localize("UTC")
    timestamp_utc = timestamp_utc.tz_convert("UTC")
    decision_date = timestamp_utc.tz_convert(timezone).normalize().tz_localize(None)
    row = {
        "decision_date": decision_date,
        "decision_time": decision_time,
        "timezone": timezone,
        "timestamp_utc": timestamp_utc,
        "symbol": quote.get("symbol", "Au99.99"),
        "price": float(quote["price"]),
        "source": quote.get("source", "akshare_spot_quotations_sge"),
        "updated_at_utc": quote.get("updated_at_utc", pd.NaT),
    }
    return pd.DataFrame([row], columns=NIGHT_DECISION_COLUMNS)


def append_night_decision_price(existing: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """Append or replace one decision-date/time row."""
    if existing is None or existing.empty:
        combined = row.copy()
    else:
        combined = pd.concat([existing, row], ignore_index=True)
    combined["decision_date"] = pd.to_datetime(combined["decision_date"], errors="coerce").dt.normalize()
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], errors="coerce", utc=True)
    if "updated_at_utc" in combined:
        combined["updated_at_utc"] = pd.to_datetime(combined["updated_at_utc"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["decision_date", "decision_time", "price"])
    return (
        combined.sort_values(["decision_date", "decision_time", "timestamp_utc"])
        .drop_duplicates(["decision_date", "decision_time", "symbol"], keep="last")
        .reset_index(drop=True)
    )


def write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def read_existing_frame(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(target)


def fetch_and_store_night_decision_price(
    config: dict[str, Any],
    intraday_output: str | Path,
    decision_output: str | Path,
    decision_time: str = "23:00",
    tolerance_minutes: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch current AU9999 quotes and persist the configured night decision price."""
    timezone = config.get("timezone", "Asia/Shanghai")
    quotes = fetch_sge_intraday_quotes(config)
    write_frame(quotes, intraday_output)
    quote = select_decision_price(
        quotes,
        decision_time=decision_time,
        tolerance_minutes=tolerance_minutes,
        timezone=timezone,
    )
    row = build_night_decision_row(quote, decision_time=decision_time, timezone=timezone)
    existing = read_existing_frame(decision_output, NIGHT_DECISION_COLUMNS)
    decision_prices = append_night_decision_price(existing, row)
    write_frame(decision_prices, decision_output)
    return quotes, row, decision_prices
