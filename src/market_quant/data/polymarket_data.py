"""Polymarket public data adapters for gold macro features."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import parse, request

import pandas as pd


POLYMARKET_COLUMNS = [
    "timestamp",
    "market_group",
    "market_id",
    "slug",
    "question",
    "probability",
    "volume",
    "liquidity",
    "end_date",
    "source",
]


@dataclass(frozen=True)
class PolymarketQuery:
    market_group: str
    query: str
    max_markets: int = 5
    min_volume: float = 0.0
    probability_side: str = "Yes"


def _get_json(base_url: str, path: str, params: dict[str, Any], timeout: int = 20) -> Any:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}?{parse.urlencode(params, doseq=True)}"
    req = request.Request(url, headers={"User-Agent": "hybrid-market-quant/0.1"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def extract_outcome_probability(market: dict[str, Any], side: str = "Yes") -> float | None:
    outcomes = [str(item).lower() for item in _as_list(market.get("outcomes"))]
    prices = _as_list(market.get("outcomePrices"))
    if not outcomes or not prices:
        return None
    side_lower = side.lower()
    try:
        idx = outcomes.index(side_lower)
    except ValueError:
        idx = 0
    try:
        return float(prices[idx])
    except (TypeError, ValueError, IndexError):
        return None


def normalize_polymarket_market(market: dict[str, Any], market_group: str, probability_side: str = "Yes", timestamp: pd.Timestamp | None = None) -> dict[str, Any] | None:
    probability = extract_outcome_probability(market, probability_side)
    if probability is None:
        return None
    return {
        "timestamp": timestamp or pd.Timestamp.utcnow().tz_convert(None),
        "market_group": market_group,
        "market_id": str(market.get("id", "")),
        "slug": str(market.get("slug", "")),
        "question": str(market.get("question") or market.get("title") or ""),
        "probability": probability,
        "volume": _safe_float(market.get("volume") or market.get("volumeNum") or market.get("volume24hr")),
        "liquidity": _safe_float(market.get("liquidity") or market.get("liquidityNum")),
        "end_date": market.get("endDate") or market.get("end_date"),
        "source": "polymarket_gamma_public_search",
    }


def search_polymarket_markets(query: PolymarketQuery, base_url: str = "https://gamma-api.polymarket.com", timeout: int = 20) -> pd.DataFrame:
    payload = _get_json(
        base_url,
        "/public-search",
        {
            "q": query.query,
            "limit_per_type": query.max_markets,
            "events_status": "active",
            "keep_closed_markets": 0,
        },
        timeout=timeout,
    )
    markets = list(payload.get("markets") or [])
    for event in payload.get("events") or []:
        markets.extend(event.get("markets") or [])
    timestamp = pd.Timestamp.utcnow().tz_convert(None)
    rows = []
    for market in markets:
        row = normalize_polymarket_market(market, query.market_group, query.probability_side, timestamp)
        if row is None:
            continue
        if row["volume"] is not None and row["volume"] < query.min_volume:
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS)
    out = pd.DataFrame(rows).drop_duplicates(subset=["market_group", "market_id", "slug"])
    out = out.sort_values(["market_group", "volume", "liquidity"], ascending=[True, False, False])
    return out.groupby("market_group", group_keys=False).head(query.max_markets)[POLYMARKET_COLUMNS].reset_index(drop=True)


def fetch_gold_polymarket_snapshots(config: dict[str, Any]) -> pd.DataFrame:
    poly_cfg = config.get("polymarket", {})
    if not poly_cfg.get("enabled", False):
        return pd.DataFrame(columns=POLYMARKET_COLUMNS)
    base_url = poly_cfg.get("base_url", "https://gamma-api.polymarket.com")
    queries = [
        PolymarketQuery(
            market_group=item["market_group"],
            query=item["query"],
            max_markets=int(item.get("max_markets", poly_cfg.get("max_markets_per_group", 5))),
            min_volume=float(item.get("min_volume", poly_cfg.get("min_volume", 0.0))),
            probability_side=item.get("probability_side", "Yes"),
        )
        for item in poly_cfg.get("queries", [])
    ]
    frames = []
    for query in queries:
        try:
            frames.append(search_polymarket_markets(query, base_url=base_url, timeout=int(poly_cfg.get("timeout", 20))))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: failed to fetch Polymarket query {query.market_group}: {exc}")
    if not frames:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp", "market_group", "market_id", "slug"])


def append_polymarket_cache(existing: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [existing, snapshot] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.drop_duplicates(subset=["timestamp", "market_group", "market_id", "slug"]).sort_values(["timestamp", "market_group", "market_id"]).reset_index(drop=True)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
