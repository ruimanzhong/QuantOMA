"""Structured LLM event schema for gold news."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
VALID_HORIZONS = {"intraday", "1d", "1w", "1m"}


@dataclass(frozen=True)
class GoldNewsEvent:
    event_type: str
    gold_direction: str
    confidence: float
    surprise: float
    time_horizon: str
    affected_channels: list[str]
    llm_summary: str
    rationale: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any], allowed_event_types: set[str] | None = None) -> "GoldNewsEvent":
        event_type = str(data.get("event_type", "other")).strip().lower() or "other"
        if allowed_event_types and event_type not in allowed_event_types:
            event_type = "other"
        direction = str(data.get("gold_direction", "neutral")).strip().lower()
        if direction not in VALID_DIRECTIONS:
            direction = "neutral"
        horizon = str(data.get("time_horizon", "1d")).strip().lower()
        if horizon not in VALID_HORIZONS:
            horizon = "1d"
        channels = data.get("affected_channels", [])
        if isinstance(channels, str):
            channels = [part.strip() for part in channels.split(",") if part.strip()]
        if not isinstance(channels, list):
            channels = []
        return cls(
            event_type=event_type,
            gold_direction=direction,
            confidence=_clip_float(data.get("confidence", 0.0)),
            surprise=_clip_float(data.get("surprise", 0.0)),
            time_horizon=horizon,
            affected_channels=[str(channel).strip().lower() for channel in channels if str(channel).strip()],
            llm_summary=str(data.get("llm_summary", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "gold_direction": self.gold_direction,
            "confidence": self.confidence,
            "surprise": self.surprise,
            "time_horizon": self.time_horizon,
            "affected_channels": self.affected_channels,
            "llm_summary": self.llm_summary,
            "rationale": self.rationale,
        }


def _clip_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return max(0.0, min(1.0, out))
