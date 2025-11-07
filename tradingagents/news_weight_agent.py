from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tradingagents.dataflows.interface import route_to_vendor

_POSITIVE_TERMS = {
    "beat",
    "beats",
    "growth",
    "surge",
    "surges",
    "record",
    "bullish",
    "expansion",
    "strong",
    "outperform",
    "outperformance",
    "upgrade",
    "upgrades",
    "partnership",
    "approval",
    "profit",
    "profits",
    "profitability",
    "guidance raise",
    "exceeds",
    "strategic",
    "momentum",
}

_NEGATIVE_TERMS = {
    "lawsuit",
    "lawsuits",
    "probe",
    "investigation",
    "downgrade",
    "downgrades",
    "decline",
    "declines",
    "miss",
    "misses",
    "loss",
    "losses",
    "recall",
    "cut",
    "cuts",
    "layoff",
    "layoffs",
    "slowdown",
    "headwind",
    "headwinds",
    "concern",
    "concerns",
    "slump",
    "slumps",
}


@dataclass
class NewsArticle:
    headline: str
    published_at: Optional[str]
    summary: Optional[str]
    source: Optional[str]
    url: Optional[str]
    sentiment: str
    sentiment_score: int


@dataclass
class NewsWeightReport:
    ticker: str
    weight: float
    as_of: str
    lookback_days: int
    judgement: str
    points: List[str]
    articles: List[NewsArticle]

    def to_markdown(self, include_articles: bool = True) -> str:
        header = (
            f"# News-Based Weight Review: {self.ticker}\n\n"
            f"- **As of:** {self.as_of}\n"
            f"- **Assigned Weight:** {self.weight:.2%}\n"
            f"- **News Lookback:** {self.lookback_days} day(s)\n\n"
        )

        bullet_lines = "\n".join(f"- {point}" for point in self.points)
        sections = [header, "## Coverage Assessment\n", bullet_lines, "\n"]

        if include_articles and self.articles:
            sections.extend(["## Notable Headlines\n", _format_articles_table(self.articles), "\n"])

        return "".join(sections)


class NewsWeightReviewAgent:
    """Reviews an assigned portfolio weight against recent news flow."""

    def __init__(self, *, default_as_of: Optional[date] = None):
        self._default_as_of = default_as_of or date.today()

    def generate_report(
        self,
        ticker: str,
        weight: float,
        *,
        as_of: Optional[str] = None,
        lookback_days: int = 7,
        max_articles: int = 8,
    ) -> NewsWeightReport:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise ValueError("Ticker symbol cannot be empty")
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0 inclusive")
        if lookback_days <= 0:
            raise ValueError("Lookback window must be positive")
        if max_articles <= 0:
            raise ValueError("max_articles must be positive")

        as_of_date = self._resolve_date(as_of)
        start_date = as_of_date - timedelta(days=lookback_days)

        raw_news = self._fetch_news(clean_ticker, start_date, as_of_date)
        articles = self._parse_news_payload(raw_news)
        articles = self._score_articles(articles)
        articles = articles[:max_articles]

        judgement, supporting_points = self._build_opinion(weight, articles)
        points = [judgement] + supporting_points
        points = points[:4]

        return NewsWeightReport(
            ticker=clean_ticker,
            weight=weight,
            as_of=as_of_date.isoformat(),
            lookback_days=lookback_days,
            judgement=judgement,
            points=points,
            articles=articles,
        )

    def _resolve_date(self, as_of: Optional[str]) -> date:
        if not as_of:
            return self._default_as_of
        try:
            return datetime.strptime(as_of, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("as_of must be in YYYY-MM-DD format") from exc

    def _fetch_news(self, ticker: str, start_date: date, end_date: date) -> Any:
        return route_to_vendor(
            "get_news",
            ticker,
            start_date.isoformat(),
            end_date.isoformat(),
        )

    def _parse_news_payload(self, payload: Any) -> List[NewsArticle]:
        if payload is None:
            return []

        if isinstance(payload, str):
            trimmed = payload.strip()
            if not trimmed:
                return []
            if trimmed.startswith("{") or trimmed.startswith("["):
                try:
                    parsed = json.loads(trimmed)
                    return self._parse_news_payload(parsed)
                except json.JSONDecodeError:
                    pass
            return self._parse_unstructured_text(trimmed)

        if isinstance(payload, list):
            articles: List[NewsArticle] = []
            for item in payload:
                articles.extend(self._parse_news_payload(item))
            return _deduplicate_articles(articles)

        if isinstance(payload, dict):
            for key in ("feed", "items", "data", "articles", "news", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    raw_articles = []
                    for entry in value:
                        raw_articles.extend(self._parse_news_payload(entry))
                    return _deduplicate_articles(raw_articles)

            headline = _first_present(payload, ["headline", "title", "news_title", "name"])
            summary = _first_present(payload, ["summary", "description", "content", "snippet"])
            if headline:
                published_at = _normalise_datetime(
                    _first_present(payload, ["datetime", "published_at", "timestamp", "date", "time"])
                )
                source = _first_present(payload, ["source", "publisher", "site", "author"])
                url = _first_present(payload, ["url", "link", "article_url"])
                return [
                    NewsArticle(
                        headline=headline.strip(),
                        published_at=published_at,
                        summary=summary.strip() if summary else None,
                        source=source.strip() if source else None,
                        url=url.strip() if url else None,
                        sentiment="neutral",
                        sentiment_score=0,
                    )
                ]

        return []

    def _parse_unstructured_text(self, text: str) -> List[NewsArticle]:
        blocks = re.split(r"\n\s*\n", text)
        articles: List[NewsArticle] = []
        for block in blocks:
            snippet = block.strip()
            if not snippet:
                continue
            lines = [line.strip() for line in snippet.splitlines() if line.strip()]
            if not lines:
                continue
            headline = lines[0][:200]
            summary = " ".join(lines[1:]) or None
            articles.append(
                NewsArticle(
                    headline=headline,
                    published_at=None,
                    summary=summary,
                    source=None,
                    url=None,
                    sentiment="neutral",
                    sentiment_score=0,
                )
            )
        return _deduplicate_articles(articles)

    def _score_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        scored: List[NewsArticle] = []
        for article in articles:
            text = " ".join(filter(None, [article.headline, article.summary]))
            label, score = _score_text(text)
            scored.append(
                NewsArticle(
                    headline=article.headline,
                    published_at=article.published_at,
                    summary=article.summary,
                    source=article.source,
                    url=article.url,
                    sentiment=label,
                    sentiment_score=score,
                )
            )
        scored.sort(key=lambda a: (a.sentiment_score, a.headline.lower()), reverse=True)
        return scored

    def _build_opinion(
        self, weight: float, articles: List[NewsArticle]
    ) -> Tuple[str, List[str]]:
        if not articles:
            judgement = (
                "No recent vendor news was available; maintain the current allocation until coverage improves."
            )
            return judgement, ["Absence of fresh headlines keeps the allocation decision data-light."]

        positives = [a for a in articles if a.sentiment_score > 0]
        negatives = [a for a in articles if a.sentiment_score < 0]
        net_score = sum(a.sentiment_score for a in articles)

        judgement = _compose_weight_statement(weight, net_score, len(positives), len(negatives))

        supporting: List[str] = []
        coverage_summary = _coverage_summary(len(positives), len(negatives), len(articles))
        supporting.append(coverage_summary)

        for article in _top_articles(positives, negatives):
            tone = "supportive" if article.sentiment_score > 0 else "cautionary"
            source = article.source or "vendor"
            date_str = article.published_at or "recent"
            supporting.append(
                f"{tone.title()} headline from {source} ({date_str}): {article.headline}"
            )
            if len(supporting) >= 3:
                break

        if len(supporting) < 3 and not negatives and positives:
            supporting.append("Coverage skews upbeat with no material red flags flagged by vendors.")
        if len(supporting) < 3 and not positives and negatives:
            supporting.append("Flow is dominated by risk-oriented stories; watch for escalation.")

        return judgement, supporting


def _top_articles(positives: List[NewsArticle], negatives: List[NewsArticle]) -> Iterable[NewsArticle]:
    ordered = sorted(positives, key=lambda a: -a.sentiment_score) + sorted(
        negatives, key=lambda a: a.sentiment_score
    )
    return ordered


def _coverage_summary(pos_count: int, neg_count: int, total: int) -> str:
    if total == 0:
        return "Coverage volume was negligible over the review window."
    neutral_count = total - pos_count - neg_count
    return (
        f"News tone snapshot: {pos_count} positive, {neg_count} negative, {neutral_count} neutral items in the sample."
    )


def _compose_weight_statement(
    weight: float, net_score: int, pos_count: int, neg_count: int
) -> str:
    weight_pct = weight * 100.0
    if net_score >= 2 and weight_pct < 8:
        return (
            f"Coverage skews constructive while the position sits at {weight_pct:.1f}% — consider whether the weight is too light relative to sentiment."
        )
    if net_score >= 2:
        return (
            f"Positive news flow backs the current {weight_pct:.1f}% allocation; staying the course is consistent with headlines."
        )
    if net_score <= -2 and weight_pct > 12:
        return (
            f"Risk-heavy coverage clashes with a {weight_pct:.1f}% stake — trim or hedge until the narrative stabilizes."
        )
    if net_score <= -2:
        return (
            f"Bearish news cadence argues for keeping exposure restrained at {weight_pct:.1f}% or lower."
        )
    if pos_count and neg_count:
        return (
            f"Mixed headlines (bulls vs bears split) suggest the {weight_pct:.1f}% weight is acceptable but needs monitoring."
        )
    return (
        f"Muted or neutral coverage leaves the {weight_pct:.1f}% allocation as a discretionary call pending clearer catalysts."
    )


def _score_text(text: str) -> Tuple[str, int]:
    lowered = text.lower()
    pos_hits = sum(1 for term in _POSITIVE_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
    neg_hits = sum(1 for term in _NEGATIVE_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
    score = pos_hits - neg_hits
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", 0


def _first_present(payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalise_datetime(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        # Handle integer timestamps passed as strings
        if raw.isdigit():
            ts = int(raw)
            # Many vendors return seconds since epoch
            return datetime.utcfromtimestamp(ts).isoformat()
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.isoformat()
    except ValueError:
        return raw


def _deduplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    seen = set()
    unique: List[NewsArticle] = []
    for article in articles:
        key = article.headline.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _format_articles_table(articles: List[NewsArticle]) -> str:
    header = "| Date | Source | Tone | Headline |\n| --- | --- | --- | --- |"
    rows = []
    for article in articles:
        date_str = article.published_at.split("T")[0] if article.published_at else "--"
        source = article.source or "--"
        tone = article.sentiment
        headline = article.headline.replace("|", "/")
        rows.append(f"| {date_str} | {source} | {tone} | {headline} |")
    return "\n".join([header] + rows)
