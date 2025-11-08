from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from tradingagents.fundamental_agent import FundamentalWeightAgent, WeightReport
from tradingagents.news_agent import (
	NewsArticle,
	NewsWeightReport,
	NewsWeightReviewAgent,
)
from tradingagents.llm_client import summarise_weight_points


@dataclass
class WeightSynthesisReport:
	ticker: str
	weight: float
	as_of: str
	lookback_days: int
	summary_points: List[str]
	fundamental_report: WeightReport
	news_report: NewsWeightReport
	generated_via_llm: bool = False

	def to_markdown(
		self,
		*,
		include_components: bool = False,
		include_metrics: bool = True,
		include_articles: bool = True,
	) -> str:
		header = (
			f"# Combined Weight Review: {self.ticker}\n\n"
			f"- **As of:** {self.as_of}\n"
			f"- **Assigned Weight:** {self.weight:.2%}\n"
			f"- **News Lookback:** {self.lookback_days} day(s)\n\n"
		)

		summary_section = "\n".join(f"- {point}" for point in self.summary_points)

		sections = [header, "## Unified Summary\n", summary_section, "\n"]

		if include_components:
			fund_markdown = _strip_top_heading(
				self.fundamental_report.to_markdown(include_metrics=include_metrics)
			)
			news_markdown = _strip_top_heading(
				self.news_report.to_markdown(include_articles=include_articles)
			)
			sections.extend(
				[
					"## Fundamental Agent Detail\n",
					fund_markdown + "\n\n",
					"## News Agent Detail\n",
					news_markdown + "\n",
				]
			)

		return "".join(sections)


class WeightSynthesisAgent:
	"""Coordinates fundamental and news agents to deliver a unified view."""

	def __init__(self):
		self._fundamental_agent = FundamentalWeightAgent()
		self._news_agent = NewsWeightReviewAgent()

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
	) -> WeightSynthesisReport:
		fund_report = self._fundamental_agent.generate_report(ticker, weight, as_of=as_of)
		news_report = self._news_agent.generate_report(
			ticker,
			weight,
			as_of=as_of,
			lookback_days=lookback_days,
			max_articles=max_articles,
		)

		summary_points = _synthesise_summary(fund_report, news_report)
		used_llm = False
		if use_llm:
			fund_markdown = fund_report.to_markdown(include_metrics=True)
			news_markdown = news_report.to_markdown(include_articles=True)
			llm_points = summarise_weight_points(
				ticker=fund_report.ticker,
				weight=weight,
				as_of=fund_report.as_of,
				fundamental_points=fund_report.rationale_points,
				news_points=news_report.points,
				metrics_table=fund_markdown,
				news_table=news_markdown,
				max_points=len(summary_points) or 6,
				model=llm_model,
			)
			if llm_points:
				summary_points = llm_points
				used_llm = True

		return WeightSynthesisReport(
			ticker=fund_report.ticker,
			weight=weight,
			as_of=fund_report.as_of,
			lookback_days=lookback_days,
			summary_points=summary_points,
			fundamental_report=fund_report,
			news_report=news_report,
			generated_via_llm=used_llm,
		)


def _synthesise_summary(
	fund_report: WeightReport,
	news_report: NewsWeightReport,
	*,
	min_points: int = 5,
	max_points: int = 6,
) -> List[str]:
	summary: List[str] = []
	seen = set()

	def add(point: Optional[str]) -> None:
		if not point:
			return
		normalised = " ".join(point.lower().split())
		if normalised in seen:
			return
		seen.add(normalised)
		summary.append(point)

	fund_points = fund_report.rationale_points or []
	news_points = news_report.points or []

	if fund_points:
		add(fund_points[0])
	if news_points:
		add(news_points[0])

	for point in fund_points[1:]:
		if len(summary) >= max_points:
			break
		add(point)

	for point in news_points[1:]:
		if len(summary) >= max_points:
			break
		add(point)

	if len(summary) < min_points:
		metrics_glance = _metrics_snapshot(fund_report.metrics)
		add(metrics_glance)

	if len(summary) < min_points:
		coverage_glance = _news_snapshot(news_report.articles)
		add(coverage_glance)

	while len(summary) < min_points:
		add("Vendor data remains sparse; maintain close monitoring before resizing.")

	return summary[:max_points]


def _metrics_snapshot(metrics: dict) -> Optional[str]:
	key_map = {
		"pe_ratio": ("P/E", lambda v: f"{v:.1f}×"),
		"roe": ("ROE", lambda v: f"{v:.1f}%"),
		"profit_margin": ("margin", lambda v: f"{v:.1f}%"),
		"revenue_growth": ("growth", lambda v: f"{v:.1f}%"),
		"debt_to_equity": ("D/E", lambda v: f"{v:.2f}×"),
	}
	parts = []
	for key in ["pe_ratio", "roe", "profit_margin", "revenue_growth", "debt_to_equity"]:
		value = metrics.get(key)
		if value is None:
			continue
		label, formatter = key_map[key]
		parts.append(f"{label} {formatter(float(value))}")
	if not parts:
		return None
	return "Fundamentals check-in: " + ", ".join(parts)


def _news_snapshot(articles: List[NewsArticle]) -> Optional[str]:
	total = len(articles)
	if total == 0:
		return None
	positives = sum(1 for article in articles if article.sentiment_score > 0)
	negatives = sum(1 for article in articles if article.sentiment_score < 0)
	neutrals = total - positives - negatives
	return (
		f"News cadence: {positives} positive / {negatives} negative / {neutrals} neutral headlines in scope."
	)


def _strip_top_heading(markdown: str) -> str:
	lines = markdown.strip().splitlines()
	if not lines:
		return markdown
	if lines[0].startswith("#"):
		lines = lines[1:]
	return "\n".join(lines).strip()