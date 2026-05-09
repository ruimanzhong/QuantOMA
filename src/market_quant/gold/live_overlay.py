"""Live-only overlays for short-history external signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LiveOverlayResult:
    multiplier: float
    reasons: list[str]
    diagnostics: dict[str, Any]


def build_live_overlay_result(row: pd.DataFrame, config: dict[str, Any] | None = None) -> LiveOverlayResult:
    """Convert short-history News/Polymarket states into a risk multiplier."""
    cfg = config or {}
    if not cfg.get("enabled", True):
        return LiveOverlayResult(1.0, [], {})

    multiplier = 1.0
    reasons: list[str] = []
    diagnostics: dict[str, Any] = {}

    polymarket_cfg = cfg.get("polymarket", {})
    if polymarket_cfg.get("enabled", True):
        risk_level = _max_value(row, [
            "geopolitical_risk_probability_level",
            "tariff_trade_risk_probability_level",
            "us_recession_probability_level",
        ])
        bullish_level = _max_value(row, ["gold_upside_probability_level"])
        fed_no_cut = _value(row, "fed_rate_cut_probability_level")
        diagnostics.update(
            {
                "polymarket_risk_probability": risk_level,
                "polymarket_gold_upside_probability": bullish_level,
                "polymarket_fed_rate_cut_probability": fed_no_cut,
            }
        )
        high_risk_threshold = float(polymarket_cfg.get("high_risk_threshold", 0.60))
        extreme_risk_threshold = float(polymarket_cfg.get("extreme_risk_threshold", 0.75))
        if risk_level is not None and risk_level >= extreme_risk_threshold:
            multiplier *= float(polymarket_cfg.get("extreme_risk_multiplier", 0.70))
            reasons.append(f"polymarket_extreme_risk={risk_level:.3f}")
        elif risk_level is not None and risk_level >= high_risk_threshold:
            multiplier *= float(polymarket_cfg.get("high_risk_multiplier", 0.85))
            reasons.append(f"polymarket_high_risk={risk_level:.3f}")
        if bullish_level is not None and bullish_level >= float(polymarket_cfg.get("gold_upside_support_threshold", 0.70)):
            multiplier = max(multiplier, float(polymarket_cfg.get("gold_upside_floor_multiplier", 0.90)))
            reasons.append(f"polymarket_gold_upside_support={bullish_level:.3f}")

    news_cfg = cfg.get("news", {})
    if news_cfg.get("enabled", True):
        bullish = _max_value(row, ["news_bullish_score", "news_bullish_score_sum_3d", "fed_dovish_score", "fed_dovish_score_sum_3d"])
        bearish = _max_value(row, ["news_bearish_score", "news_bearish_score_sum_3d", "usd_pressure_score", "usd_pressure_score_sum_3d"])
        diagnostics.update({"news_bullish_pressure": bullish, "news_bearish_pressure": bearish})
        if bearish is not None and bullish is not None and bearish - bullish >= float(news_cfg.get("bearish_spread_threshold", 0.30)):
            multiplier *= float(news_cfg.get("bearish_multiplier", 0.85))
            reasons.append(f"news_bearish_spread={bearish - bullish:.3f}")

    multiplier = max(float(cfg.get("min_multiplier", 0.50)), min(1.0, multiplier))
    diagnostics["live_overlay_multiplier"] = multiplier
    return LiveOverlayResult(multiplier, reasons, diagnostics)


def _value(row: pd.DataFrame, column: str) -> float | None:
    if column not in row:
        return None
    value = pd.to_numeric(row[column], errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _max_value(row: pd.DataFrame, columns: list[str]) -> float | None:
    values = [_value(row, col) for col in columns]
    values = [value for value in values if value is not None]
    return max(values) if values else None
