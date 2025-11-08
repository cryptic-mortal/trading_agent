from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf  # type: ignore[import]

from tradingagents import llm_client

_METRIC_FIELDS = [
    ("revenue", "Total Revenue", "currency"),
    ("net_income", "Net Income", "currency"),
    ("operating_income", "Operating Income", "currency"),
    ("operating_cash_flow", "Operating Cash Flow", "currency"),
    ("gross_profit", "Gross Profit", "currency"),
    ("equity", "Stockholder Equity", "currency"),
    ("liabilities", "Total Liabilities", "currency"),
    ("profit_margin", "Profit Margin", "percent"),
    ("roe", "Return on Equity", "percent"),
    ("revenue_growth", "Revenue Growth", "percent"),
    ("pe_ratio", "Price/Earnings", "multiple"),
    ("debt_to_equity", "Debt/Equity", "multiple"),
    ("dividend_yield", "Dividend Yield", "percent"),
]


@dataclass
class WeightReport:
    """Structured presentation of a fundamentals-based weight decision."""

    ticker: str
    weight: float
    as_of: str
    rationale_points: List[str]
    metrics: Dict[str, Optional[float]]
    generated_via_llm: bool = False

    def to_markdown(self, include_metrics: bool = True) -> str:
        header = (
            f"# Portfolio Weight Rationale: {self.ticker}\n\n"
            f"- **As of:** {self.as_of}\n"
            f"- **Assigned Weight:** {self.weight:.2%}\n\n"
        )

        rationale_body = "\n".join(f"- {point}" for point in self.rationale_points)
        if not rationale_body:
            rationale_body = (
                "- Unable to derive a data-backed rationale; please review the fundamentals manually."
            )

        sections = [header, "## Why This Weight\n", rationale_body, "\n"]

        if include_metrics:
            sections.extend(["## Key Fundamental Metrics\n", _format_metrics_table(self.metrics), "\n"])

        return "".join(sections)


class FundamentalWeightAgent:
    """Generates weight rationales using Yahoo Finance fundamentals."""

    def __init__(self, *, default_as_of: Optional[date] = None):
        self._default_as_of = default_as_of or date.today()

    def generate_report(
        self,
        ticker: str,
        weight: float,
        *,
        as_of: Optional[str] = None,
        use_llm: bool = False,
        llm_model: Optional[str] = None,
    ) -> WeightReport:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise ValueError("Ticker symbol cannot be empty")
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0 inclusive")

        as_of_str = as_of or self._default_as_of.isoformat()
        info, financials, balance_sheet, cashflow = self._fetch_fundamentals(clean_ticker)
        metrics = _calculate_metrics(info, financials, balance_sheet, cashflow)
        rationale = _build_rationale(clean_ticker, weight, metrics)
        generated_via_llm = False

        if use_llm:
            metrics_table = _format_metrics_table(metrics)
            metrics_summary = _metrics_prompt_summary(metrics)
            llm_points = llm_client.summarise_fundamentals(
                ticker=clean_ticker,
                weight=weight,
                as_of=as_of_str,
                metrics_table=metrics_table,
                metrics_summary=metrics_summary,
                max_points=4,
                model=llm_model,
            )
            if llm_points:
                rationale = llm_points
                generated_via_llm = True

        return WeightReport(
            ticker=clean_ticker,
            weight=weight,
            as_of=as_of_str,
            rationale_points=rationale,
            metrics=metrics,
            generated_via_llm=generated_via_llm,
        )

    def _fetch_fundamentals(
        self, ticker: str
    ) -> Tuple[Dict[str, Any], Any, Any, Any]:
        ticker_obj = yf.Ticker(ticker)

        info: Dict[str, Any] = {}
        try:
            info = ticker_obj.get_info()
        except Exception:
            try:
                info = getattr(ticker_obj, "info", {}) or {}
            except Exception:
                info = {}

        financials = None
        balance_sheet = None
        cashflow = None

        try:
            financials = ticker_obj.get_financials()
        except Exception:
            financials = None

        try:
            balance_sheet = ticker_obj.get_balance_sheet()
        except Exception:
            balance_sheet = None

        try:
            cashflow = ticker_obj.get_cashflow()
        except Exception:
            cashflow = None

        return info or {}, financials, balance_sheet, cashflow


def _calculate_metrics(
    info: Dict[str, Any],
    financials,
    balance_sheet,
    cashflow,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "pe_ratio": _first_not_none(
            info.get("trailingPE"), info.get("forwardPE")
        ),
        "profit_margin": _maybe_percent(info.get("profitMargins")),
        "roe": _maybe_percent(info.get("returnOnEquity")),
        "dividend_yield": _maybe_percent(info.get("dividendYield")),
        "revenue_growth": _maybe_percent(info.get("revenueGrowth")),
    }

    revenue = _latest_financial_value(financials, "Total Revenue")
    net_income = _latest_financial_value(financials, "Net Income")
    operating_income = _latest_financial_value(financials, "Operating Income")
    gross_profit = _latest_financial_value(financials, "Gross Profit")

    equity = _latest_financial_value(balance_sheet, "Total Stockholder Equity")
    liabilities = _latest_financial_value(balance_sheet, "Total Liab")

    operating_cash_flow = _latest_financial_value(cashflow, "Operating Cash Flow")

    metrics.update(
        {
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "operating_cash_flow": operating_cash_flow,
            "gross_profit": gross_profit,
            "equity": equity,
            "liabilities": liabilities,
        }
    )

    if (
        metrics["profit_margin"] is None
        and revenue is not None
        and net_income is not None
        and revenue not in (0.0, 0)
    ):
        metrics["profit_margin"] = _safe_percent(net_income / revenue)

    if (
        metrics["roe"] is None
        and equity is not None
        and net_income is not None
        and equity not in (0.0, 0)
    ):
        metrics["roe"] = _safe_percent(net_income / equity)

    if metrics.get("revenue_growth") is None:
        metrics["revenue_growth"] = _compute_growth(financials, "Total Revenue")

    if liabilities is not None and equity is not None and equity not in (0.0, 0):
        metrics["debt_to_equity"] = float(liabilities) / float(equity)
    else:
        metrics["debt_to_equity"] = None

    dividend = metrics.get("dividend_yield")
    if dividend is not None:
        dividend = float(dividend)
        if abs(dividend) <= 3:
            dividend *= 100.0
        metrics["dividend_yield"] = dividend if abs(dividend) <= 100 else None

    for key in ("profit_margin", "roe", "revenue_growth"):
        value = metrics.get(key)
        if value is None:
            continue
        value = float(value)
        if abs(value) <= 5:
            value *= 100.0
        metrics[key] = value if abs(value) <= 500 else None

    return metrics


def _first_not_none(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _maybe_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _safe_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) * 100.0


def _latest_financial_value(frame, label: str) -> Optional[float]:
    if frame is None or getattr(frame, "empty", True):
        return None
    if label not in frame.index:
        return None
    series = frame.loc[label].dropna()
    if series.empty:
        return None
    try:
        return float(series.iloc[0])
    except (TypeError, ValueError):
        return None


def _compute_growth(frame, label: str) -> Optional[float]:
    if frame is None or getattr(frame, "empty", True):
        return None
    if label not in frame.index:
        return None
    series = frame.loc[label].dropna()
    if len(series) < 2:
        return None
    latest = float(series.iloc[0])
    prior = float(series.iloc[1])
    if prior in (0.0, 0):
        return None
    return ((latest - prior) / prior) * 100.0


def _build_rationale(ticker: str, weight: float, metrics: Dict[str, Optional[float]]) -> List[str]:
    positives: List[str] = []
    negatives: List[str] = []
    score = 0.0

    pe = metrics.get("pe_ratio")
    if pe is not None:
        if pe <= 18:
            score += 1.5
            positives.append(f"Valuation at {pe:.1f}× earnings leaves room for multiple expansion.")
        elif pe >= 40:
            score -= 1.5
            negatives.append(f"Rich {pe:.1f}× P/E implies the weight leans on premium expectations.")

    margin = metrics.get("profit_margin")
    if margin is not None:
        if margin >= 15:
            score += 1.0
            positives.append(f"Net margin of {margin:.1f}% underscores durable profitability.")
        elif margin <= 5:
            score -= 1.0
            negatives.append(f"Margin at {margin:.1f}% compresses the earnings buffer behind the position.")

    roe = metrics.get("roe")
    if roe is not None:
        if roe >= 18:
            score += 1.0
            positives.append(f"ROE of {roe:.1f}% signals efficient capital deployment.")
        elif roe <= 8:
            score -= 0.5
            negatives.append(f"ROE of {roe:.1f}% trails quality benchmarks, moderating conviction.")

    growth = metrics.get("revenue_growth")
    if growth is not None:
        if growth >= 20:
            score += 1.0
            positives.append(f"Revenue growth running {growth:.1f}% keeps the topline momentum supportive.")
        elif growth <= 0:
            score -= 1.0
            negatives.append(f"Revenue growth at {growth:.1f}% points to a deceleration risk for sizing up.")

    dte = metrics.get("debt_to_equity")
    if dte is not None:
        if dte <= 0.7:
            score += 0.5
            positives.append(f"Balance sheet leverage at {dte:.2f}× keeps downside risk manageable.")
        elif dte >= 2.0:
            score -= 1.0
            negatives.append(f"Leverage of {dte:.2f}× nudges position sizing toward caution.")

    dividend = metrics.get("dividend_yield")
    if dividend is not None and dividend > 0:
        if dividend >= 3.0:
            positives.append(f"Dividend yield of {dividend:.1f}% provides a steady carry component.")
        elif dividend < 1.0:
            negatives.append(f"Token yield of {dividend:.1f}% keeps the case focused on growth delivery.")

    rationale: List[str] = []
    alignment = _alignment_statement(ticker, weight, score)
    rationale.append(alignment)

    ordered = positives + negatives if score >= 0 else negatives + positives
    for point in ordered:
        if len(rationale) >= 4:
            break
        rationale.append(point)

    if len(rationale) < 4:
        summary = _metric_summary(metrics)
        if summary:
            rationale.append(f"Key fundamentals in view: {summary}.")

    if len(rationale) < 3:
        rationale.append(
            "Limited fundamentals were available via Yahoo Finance; hold sizing until more disclosures land."
        )

    return rationale[:4]


def _alignment_statement(ticker: str, weight: float, score: float) -> str:
    if score >= 2.5 and weight < 0.15:
        return (
            f"Fundamental scorecard tilts bullish, indicating capacity to scale beyond the current {weight:.2%} weight."
        )
    if score <= -1.5 and weight > 0.1:
        return (
            f"Fundamental mix lands light, so the {weight:.2%} allocation should stay capped pending stronger data."
        )
    return f"Assigned weight of {weight:.2%} aligns with the latest fundamental balance for {ticker}."


def _metric_summary(metrics: Dict[str, Optional[float]]) -> str:
    pieces: List[str] = []
    pe = metrics.get("pe_ratio")
    if pe is not None:
        pieces.append(f"P/E {pe:.1f}×")
    roe = metrics.get("roe")
    if roe is not None:
        pieces.append(f"ROE {roe:.1f}%")
    margin = metrics.get("profit_margin")
    if margin is not None:
        pieces.append(f"Margin {margin:.1f}%")
    growth = metrics.get("revenue_growth")
    if growth is not None:
        pieces.append(f"Growth {growth:.1f}%")
    dte = metrics.get("debt_to_equity")
    if dte is not None:
        pieces.append(f"D/E {dte:.2f}×")
    return ", ".join(pieces)


def _metrics_prompt_summary(metrics: Dict[str, Optional[float]]) -> str:
    lines: List[str] = []
    for key, label, value_type in _METRIC_FIELDS:
        value = metrics.get(key)
        if value is None:
            continue
        if value_type == "currency":
            formatted = _format_currency(value)
        elif value_type == "percent":
            formatted = f"{float(value):.2f}%"
        else:
            formatted = f"{float(value):.2f}"
        lines.append(f"- {label}: {formatted}")
    return "\n".join(lines)


def _format_metrics_table(metrics: Dict[str, Optional[float]]) -> str:
    rows: List[str] = []
    for key, label, value_type in _METRIC_FIELDS:
        value = metrics.get(key)
        if value is None:
            continue
        if value_type == "currency":
            rows.append(f"| {label} | {_format_currency(value)} |")
        elif value_type == "percent":
            rows.append(f"| {label} | {value:.2f}% |")
        else:
            rows.append(f"| {label} | {value:.2f} |")

    if not rows:
        return "No fundamentals were returned by Yahoo Finance for this ticker."

    header = "| Metric | Value |\n| --- | --- |"
    return "\n".join([header] + rows)


def _format_currency(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"
