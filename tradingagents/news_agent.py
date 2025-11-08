from __future__ import annotations

import re
import contextlib
import html
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

import xml.etree.ElementTree as ET

import yfinance as yf

from tradingagents import llm_client

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
    generated_via_llm: bool = False

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
        use_llm: bool = False,
        llm_model: Optional[str] = None,
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

        articles = self._fetch_news(clean_ticker, start_date, as_of_date)
        articles = self._score_articles(articles)
        articles = articles[:max_articles]

        judgement, supporting_points = self._build_opinion(weight, articles)
        points = [judgement] + supporting_points
        points = points[:4]
        generated_via_llm = False

        if use_llm:
            article_summaries = _articles_prompt_digest(articles)
            net_sentiment = sum(article.sentiment_score for article in articles)
            llm_points = llm_client.summarise_news(
                ticker=clean_ticker,
                weight=weight,
                as_of=as_of_date.isoformat(),
                lookback_days=lookback_days,
                article_summaries=article_summaries,
                net_sentiment=net_sentiment,
                max_points=4,
                model=llm_model,
            )
            if llm_points:
                points = llm_points[:4]
                if points:
                    judgement = points[0]
                generated_via_llm = True

        return NewsWeightReport(
            ticker=clean_ticker,
            weight=weight,
            as_of=as_of_date.isoformat(),
            lookback_days=lookback_days,
            judgement=judgement,
            points=points,
            articles=articles,
            generated_via_llm=generated_via_llm,
        )

    def _resolve_date(self, as_of: Optional[str]) -> date:
        if not as_of:
            return self._default_as_of
        try:
            return datetime.strptime(as_of, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("as_of must be in YYYY-MM-DD format") from exc

    def _fetch_news(self, ticker: str, start_date: date, end_date: date) -> List[NewsArticle]:
        primary = self._fetch_google_news(ticker, start_date, end_date)
        if primary:
            return primary
        return self._fetch_yfinance_news(ticker, start_date, end_date)

    def _fetch_google_news(self, ticker: str, start_date: date, end_date: date) -> List[NewsArticle]:
        query = quote_plus(f"{ticker} stock")
        url = (
            "https://news.google.com/rss/search?q="
            f"{query}&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            with contextlib.closing(urlopen(url, timeout=10)) as response:
                payload = response.read()
        except URLError:
            return []
        except TimeoutError:
            return []

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return []

        articles: List[NewsArticle] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue

            pub_date_raw = item.findtext("pubDate")
            publish_dt: Optional[datetime] = None
            if pub_date_raw:
                try:
                    publish_dt = parsedate_to_datetime(pub_date_raw)
                    if publish_dt.tzinfo is None:
                        publish_dt = publish_dt.replace(tzinfo=timezone.utc)
                    else:
                        publish_dt = publish_dt.astimezone(timezone.utc)
                except (TypeError, ValueError):
                    publish_dt = None
            if publish_dt is None:
                continue
            if publish_dt.date() < start_date or publish_dt.date() > end_date:
                continue

            summary_raw = item.findtext("description") or ""
            summary = _strip_html(summary_raw).strip() or None
            source_elem = item.find("{http://news.google.com/newssources}news-source")
            source = source_elem.text.strip() if source_elem is not None and source_elem.text else None
            link = (item.findtext("link") or "").strip() or None

            articles.append(
                NewsArticle(
                    headline=html.unescape(title),
                    published_at=publish_dt.isoformat(),
                    summary=html.unescape(summary) if summary else None,
                    source=source,
                    url=link,
                    sentiment="neutral",
                    sentiment_score=0,
                )
            )

        articles.sort(key=lambda article: article.published_at or "", reverse=True)
        return _deduplicate_articles(articles)

    def _fetch_yfinance_news(self, ticker: str, start_date: date, end_date: date) -> List[NewsArticle]:
        try:
            payload = yf.Ticker(ticker).news or []
        except Exception:
            payload = []

        articles: List[NewsArticle] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            published = _extract_publish_datetime(item)
            if published is None:
                continue
            if published.date() < start_date or published.date() > end_date:
                continue
            headline = (item.get("title") or item.get("headline") or "").strip()
            if not headline:
                continue
            summary = (item.get("summary") or item.get("content") or "").strip() or None
            source = (item.get("publisher") or item.get("source") or "").strip() or None
            url = (item.get("link") or item.get("url") or "").strip() or None
            articles.append(
                NewsArticle(
                    headline=headline,
                    published_at=published.isoformat(),
                    summary=summary,
                    source=source,
                    url=url,
                    sentiment="neutral",
                    sentiment_score=0,
                )
            )

        articles.sort(key=lambda article: article.published_at or "", reverse=True)
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


def _articles_prompt_digest(articles: List[NewsArticle]) -> str:
    lines: List[str] = []
    for article in articles:
        tone = article.sentiment
        date_str = article.published_at or "recent"
        source = article.source or "vendor"
        summary = article.summary or "(no summary provided)"
        lines.append(
            f"- {tone.title()} | {source} | {date_str}: {article.headline} — {summary}"
        )
    return "\n".join(lines)


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


def _extract_publish_datetime(item: dict) -> Optional[datetime]:
    raw = item.get("providerPublishTime")
    if raw is not None:
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            pass

    for key in ("pubDate", "publishedAt", "date", "time"):
        raw_value = item.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                continue
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                continue
            if candidate.isdigit():
                try:
                    return datetime.fromtimestamp(int(candidate), tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    continue
            try:
                return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _deduplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    seen: set[str] = set()
    deduped: List[NewsArticle] = []
    for article in articles:
        key = article.headline.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


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
