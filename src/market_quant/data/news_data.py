"""News source adapters for gold research."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import hashlib
from html.parser import HTMLParser
from typing import Any
from urllib import request
from xml.etree import ElementTree

import pandas as pd


NEWS_COLUMNS = ["news_id", "date", "source", "title", "summary", "url", "published_at", "relevance_score"]


GOLD_RELEVANCE_KEYWORDS = {
    "monetary": 3,
    "policy": 3,
    "fomc": 3,
    "rate": 3,
    "inflation": 3,
    "economic outlook": 2,
    "labor market": 2,
    "balance sheet": 2,
    "financial stability": 2,
    "energy": 1,
    "transitory": 2,
    "tariff": 2,
    "dollar": 2,
    "liquidity": 2,
}

BOILERPLATE_PATTERNS = [
    "official website",
    "secure .gov",
    "share sensitive information",
    "the federal reserve, the central bank",
    "reporting forms",
    "legal developments",
    "payment policies",
    "working papers",
]


def _text(node: ElementTree.Element, child: str) -> str:
    found = node.find(child)
    return "" if found is None or found.text is None else found.text.strip()


def _parse_datetime(value: str) -> pd.Timestamp:
    if not value:
        return pd.NaT
    try:
        parsed = parsedate_to_datetime(value)
        return pd.Timestamp(parsed).tz_convert(None) if parsed.tzinfo else pd.Timestamp(parsed)
    except (TypeError, ValueError):
        return pd.to_datetime(value, utc=True, errors="coerce").tz_convert(None)


def score_news_relevance(title: str, summary: str) -> int:
    text = f"{title} {summary}".lower()
    return int(sum(weight for keyword, weight in GOLD_RELEVANCE_KEYWORDS.items() if keyword in text))


def build_news_id(source: str, title: str, published_at: object) -> str:
    raw = f"{source}|{title}|{published_at}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def fetch_rss_source(source: dict[str, Any]) -> pd.DataFrame:
    """Fetch a simple RSS feed without requiring feedparser."""
    req = request.Request(source["url"], headers={"User-Agent": "hybrid-market-quant/0.1"})
    with request.urlopen(req, timeout=20) as response:
        payload = response.read()
    root = ElementTree.fromstring(payload)
    rows = []
    for item in root.findall(".//item"):
        published_at = _parse_datetime(_text(item, "pubDate") or _text(item, "updated"))
        rows.append(
            {
                "source": source["name"],
                "title": _text(item, "title"),
                "summary": _text(item, "description"),
                "url": _text(item, "link"),
                "published_at": published_at,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    out = out.dropna(subset=["published_at"]).copy()
    out["date"] = pd.to_datetime(out["published_at"]).dt.normalize()
    out["relevance_score"] = [score_news_relevance(title, summary) for title, summary in zip(out["title"], out["summary"], strict=False)]
    out["news_id"] = [build_news_id(source, title, published_at) for source, title, published_at in zip(out["source"], out["title"], out["published_at"], strict=False)]
    return out[NEWS_COLUMNS]


def fetch_gold_news(news_config: dict[str, Any]) -> pd.DataFrame:
    """Fetch configured gold news sources into a raw article metadata frame."""
    if not news_config.get("enabled", False):
        return pd.DataFrame(columns=NEWS_COLUMNS)
    frames = []
    for source in news_config.get("sources", []):
        if source.get("type") != "rss":
            continue
        try:
            frames.append(fetch_rss_source(source))
        except (OSError, ElementTree.ParseError, ValueError) as exc:
            print(f"warning: failed to fetch news source {source.get('name')}: {exc}")
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source", "title", "published_at"])
    min_relevance = int(news_config.get("min_relevance_score", 0))
    if min_relevance > 0:
        out = out[out["relevance_score"] >= min_relevance].copy()
    out = out.sort_values(["date", "published_at", "source"])
    return out.groupby("date", group_keys=False).head(int(news_config.get("max_articles_per_day", 80))).reset_index(drop=True)


def merge_news_cache(existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    """Merge newly fetched news with cache without letting empty fetches erase history."""
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    if fetched is not None and not fetched.empty:
        frames.append(fetched)
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for column in NEWS_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.drop_duplicates(subset=["source", "title", "published_at"], keep="last")
    return out[NEWS_COLUMNS].sort_values(["date", "published_at", "source"]).reset_index(drop=True)


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_paragraph = False
        self._parts: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "p":
            self._in_paragraph = True
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p" and self._in_paragraph:
            text = " ".join("".join(self._current).split())
            if text:
                self._parts.append(text)
            self._in_paragraph = False

    def handle_data(self, data: str) -> None:
        if self._in_paragraph:
            self._current.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def fetch_url_excerpt(url: str, max_chars: int = 1600) -> str:
    """Fetch a short transient page excerpt for LLM input."""
    if not url:
        return ""
    req = request.Request(url, headers={"User-Agent": "hybrid-market-quant/0.1"})
    with request.urlopen(req, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
    parser = _ParagraphExtractor()
    parser.feed(html)
    paragraphs = [part.strip() for part in parser._parts if part.strip()]
    filtered = [
        part
        for part in paragraphs
        if len(part) >= 40 and not any(pattern in part.lower() for pattern in BOILERPLATE_PATTERNS)
    ]
    scored = sorted(
        filtered,
        key=lambda text: (score_news_relevance(text, ""), len(text)),
        reverse=True,
    )
    selected = scored[:8] if scored else filtered[:8]
    return " ".join(selected)[:max_chars].strip()
