from __future__ import annotations

import os
from typing import Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    genai = None  # type: ignore


_DEFAULT_MODEL = os.getenv("TRADINGAGENTS_LLM_MODEL", "gemini-2.0-flash")
LAST_LLM_ERROR: Optional[str] = None
LAST_LLM_ERROR: Optional[str] = None


def summarise_weight_points(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    fundamental_points: Iterable[str],
    news_points: Iterable[str],
    metrics_table: str,
    news_table: str,
    max_points: int = 6,
    model: Optional[str] = None,
) -> Optional[List[str]]:
    """Generate summary bullets using an LLM when available."""

    global LAST_LLM_ERROR
    LAST_LLM_ERROR = None


    fundamental_text = "\n".join(f"- {point}" for point in fundamental_points)
    news_text = "\n".join(f"- {point}" for point in news_points)

    prompt = f"""
You are assisting a portfolio manager. Produce up to {max_points} succinct bullet points
that justify the current weight for {ticker} as of {as_of}. Blend fundamentals and news insights. You can either support or challenge the weight based on the data provided. Try to provide unique points that do not overlap with each other.

Current portfolio weight: {weight:.2%}

Fundamental signals:
{fundamental_text or "(none)"}

News signals:
{news_text or "(none)"}

Fundamental metrics table (Markdown):
{metrics_table or "(none)"}

News headlines table (Markdown):
{news_table or "(none)"}

Output format: one bullet per line, concise, informative, no numbering, no preamble or postscript.
""".strip()


    return generate_bullets(prompt, max_points=max_points, model=model)


def summarise_fundamentals(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    metrics_table: str,
    metrics_summary: str,
    max_points: int = 4,
    model: Optional[str] = None,
) -> Optional[List[str]]:
    """Summarise key fundamental data into actionable bullets."""

    prompt = f"""
You are the fundamentals analyst on a portfolio desk. Review the metrics and craft up to {max_points} bullet points that explain whether the current allocation for {ticker} at {weight:.2%} is justified as of {as_of}.

Key metrics overview:
{metrics_summary or "(no key metrics)"}

Detailed metrics (Markdown table):
{metrics_table or "(none)"}

Output requirements:
- First bullet must state a clear action (maintain, increase, reduce, accumulate, watch) tied to fundamentals.
- Each bullet should reference specific metrics or trends.
- Keep bullets concise and avoid repeating facts.
""".strip()

    return generate_bullets(prompt, max_points=max_points, model=model)


def summarise_news(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    lookback_days: int,
    article_summaries: str,
    net_sentiment: int,
    max_points: int = 4,
    model: Optional[str] = None,
) -> Optional[List[str]]:
    """Summarise headline flow into guidance bullets."""

    prompt = f"""
You are the news-flow specialist on a portfolio team. Recent vendor headlines for {ticker} over the last {lookback_days} day(s) carry a net sentiment score of {net_sentiment} (positives minus negatives).

The portfolio holds a {weight:.2%} weight as of {as_of}. Produce up to {max_points} bullet points advising how to manage this weight given the news.

Headlines digest:
{article_summaries or "(no headlines in scope)"}

Output requirements:
- First bullet should be the recommendation (maintain, add, trim, hedge) referencing sentiment.
- Remaining bullets should cite specific headlines or themes and their expected impact.
- Keep bullets concise and avoid duplicating points.
""".strip()

    return generate_bullets(prompt, max_points=max_points, model=model)


def generate_bullets(
    prompt: str,
    *,
    max_points: int = 6,
    model: Optional[str] = None,
) -> Optional[List[str]]:
    """Shared helper that routes to the configured LLM provider."""

    global LAST_LLM_ERROR
    LAST_LLM_ERROR = None

    chosen_model = (model or _DEFAULT_MODEL).strip()
    if not chosen_model:
        LAST_LLM_ERROR = "No model provided"
        return None

    if _looks_like_gemini(chosen_model):
        return _invoke_gemini(prompt, max_points, chosen_model)

    return _invoke_openai(prompt, max_points, chosen_model)


def _invoke_openai(prompt: str, max_points: int, model: str) -> Optional[List[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        _set_error("OpenAI client unavailable or OPENAI_API_KEY missing")
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
    except Exception as exc:  # noqa: BLE001
        _set_error(f"OpenAI request failed: {exc}")
        return None

    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", "") != "message":
            continue
        message = getattr(output, "message", None)
        if not message:
            continue
        content = getattr(message, "content", []) or []
        for item in content:
            if getattr(item, "type", "") == "text":
                text = getattr(item, "text", "")
                return _normalise_output(text, max_points)
    _set_error("OpenAI response contained no text output")
    return None


def _invoke_gemini(prompt: str, max_points: int, model: str) -> Optional[List[str]]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        _set_error("Gemini client unavailable or GOOGLE_API_KEY/GEMINI_API_KEY missing")
        return None

    try:
        genai.configure(api_key=api_key)
        generation_model = genai.GenerativeModel(model)
        response = generation_model.generate_content(prompt)
    except Exception as exc:  # noqa: BLE001
        _set_error(f"Gemini request failed: {exc}")
        return None

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return _normalise_output(text, max_points)

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return _normalise_output(part_text, max_points)

    _set_error("Gemini response contained no text output")
    return None


def _looks_like_gemini(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith("gemini") or lowered.startswith("flash-")


def _normalise_output(raw_text: str, max_points: int) -> List[str]:
    lines: List[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip().lstrip("-•*").strip()
        if not cleaned:
            continue
        lines.append(cleaned)
        if len(lines) >= max_points:
            break
    return lines


def _set_error(message: str) -> None:
    global LAST_LLM_ERROR
    LAST_LLM_ERROR = message
