"""
analysis_logger.py — Comprehensive stock analysis logging for agent learning.

Captures for every analysis:
  • Price & technicals, fundamentals, technical pattern classification
  • Fundamental quality scoring, market context (S&P / VIX / TA-35)
  • News sentiment (score + top headlines)
  • Full reasoning chain extracted from markdown
  • Comparison to previous analysis of the same ticker
  • Outcome tracking placeholders (filled by outcome_tracker.py)
  • Full analysis text for NLP / fine-tuning
"""

import json
import os
import re
from datetime import datetime
from typing import Optional

ANALYSES_FILE = os.path.join(os.path.dirname(__file__), "stock_analyses.json")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load() -> list:
    if not os.path.exists(ANALYSES_FILE):
        return []
    with open(ANALYSES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []


def _save(analyses: list) -> None:
    with open(ANALYSES_FILE, "w", encoding="utf-8") as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, section_name: str) -> str:
    """Extract content of a markdown section by header name."""
    pattern = rf"###?\s+[^\n]*{re.escape(section_name)}[^\n]*\n(.*?)(?=\n###?|\Z)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_recommendation(text: str) -> str:
    """Extract BUY/SELL/HOLD — searches TL;DR / Summary first, then full text."""
    RECS = r"\b(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|ACCUMULATE|REDUCE|AVOID)\b"
    for sec in ["TL;DR", "Summary", "Outlook", "המלצה"]:
        section = _extract_section(text, sec)
        m = re.search(RECS, section, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(RECS, text, re.IGNORECASE)
    return m.group(1).upper() if m else "N/A"


def _extract_confidence(text: str) -> int:
    """
    Infer confidence level 1–5 from language cues.
    5 = very high conviction, 1 = very uncertain / contradictory signals.
    """
    t = text.lower()
    high = len(re.findall(
        r"\b(strong|clearly|definitively|high confidence|compelling|conviction|"
        r"excellent|robust|outstanding|undeniable)\b", t))
    low = len(re.findall(
        r"\b(uncertain|caution|cautious|risk|volatile|wait|unclear|mixed|monitor|"
        r"concerns?|headwinds?|challenging|difficult|avoid)\b", t))
    return max(1, min(5, 3 + min(high, 2) - min(low, 2)))


def _extract_bullets(text: str, section_names: list) -> list[str]:
    """Extract bullet-point content from first matching section."""
    for name in section_names:
        section = _extract_section(text, name)
        if section:
            bullets = re.findall(r"[-•*]\s+(.+)", section)
            if bullets:
                return [b.strip() for b in bullets[:8]]
    return []


def _extract_tldr(text: str) -> str:
    return _extract_section(text, "TL;DR") or _extract_section(text, "Summary") or ""


# ---------------------------------------------------------------------------
# Technical pattern classifier
# ---------------------------------------------------------------------------

def _classify_technical(stock_data: dict) -> dict:
    """
    Classify the technical setup into named patterns and generate signal list.
    Returns: {patterns: [...], signals: [...], trend: "bullish|bearish|neutral"}
    """
    patterns, signals = [], []

    rsi    = stock_data.get("rsi_14")
    price  = stock_data.get("current_price")
    sma50  = stock_data.get("sma_50")
    sma200 = stock_data.get("sma_200")
    w52h   = stock_data.get("week_52_high")
    w52l   = stock_data.get("week_52_low")
    vol    = stock_data.get("volume_ratio")
    vs50   = stock_data.get("price_vs_sma50_pct")
    vs200  = stock_data.get("price_vs_sma200_pct")

    bullish_count = 0
    bearish_count = 0

    # RSI
    if rsi is not None:
        if rsi >= 70:
            patterns.append("OVERBOUGHT")
            signals.append(f"RSI {rsi:.1f} — overbought territory")
            bearish_count += 1
        elif rsi <= 30:
            patterns.append("OVERSOLD")
            signals.append(f"RSI {rsi:.1f} — oversold, potential reversal")
            bullish_count += 1
        elif rsi >= 50:
            signals.append(f"RSI {rsi:.1f} — bullish momentum")
            bullish_count += 1
        else:
            signals.append(f"RSI {rsi:.1f} — bearish momentum")
            bearish_count += 1

    # MA alignment
    if price and sma50 and sma200:
        if price > sma50 > sma200:
            patterns.append("UPTREND")
            signals.append("Price > SMA-50 > SMA-200: bullish alignment")
            bullish_count += 2
        elif price < sma50 < sma200:
            patterns.append("DOWNTREND")
            signals.append("Price < SMA-50 < SMA-200: bearish alignment")
            bearish_count += 2
        elif sma50 > sma200 and price < sma50:
            patterns.append("PULLBACK_IN_UPTREND")
            signals.append("Pulling back toward SMA-50 in overall uptrend")
            bullish_count += 1

    # Golden / Death Cross
    if sma50 and sma200:
        ratio = sma50 / sma200
        if 0.999 <= ratio <= 1.001:
            patterns.append("MA_CROSSOVER_ZONE")
            signals.append("SMA-50 / SMA-200 crossover imminent")
        elif ratio > 1.0:
            bullish_count += 1
        else:
            bearish_count += 1

    # Distance from 52-week extremes
    if price and w52h:
        pct_from_high = (w52h - price) / w52h * 100
        if pct_from_high < 3:
            patterns.append("AT_52W_HIGH")
            signals.append(f"Only {pct_from_high:.1f}% below 52-week high")
            bullish_count += 1
        elif pct_from_high > 30:
            patterns.append("DEEP_CORRECTION")
            signals.append(f"{pct_from_high:.1f}% below 52-week high — deep correction")
            bearish_count += 1

    if price and w52l:
        pct_from_low = (price - w52l) / w52l * 100
        if pct_from_low < 5:
            patterns.append("AT_52W_LOW")
            signals.append(f"Only {pct_from_low:.1f}% above 52-week low — potential base")

    # Volume
    if vol:
        if vol >= 1.5:
            patterns.append("HIGH_VOLUME")
            signals.append(f"Volume ratio {vol:.1f}x — unusual activity")
        elif vol <= 0.5:
            patterns.append("LOW_VOLUME")
            signals.append(f"Volume ratio {vol:.1f}x — low conviction move")

    trend = "bullish" if bullish_count > bearish_count else \
            "bearish" if bearish_count > bullish_count else "neutral"

    return {
        "patterns":       patterns,
        "signals":        signals,
        "trend":          trend,
        "bullish_count":  bullish_count,
        "bearish_count":  bearish_count,
    }


# ---------------------------------------------------------------------------
# Fundamental quality scorer
# ---------------------------------------------------------------------------

def _classify_fundamentals(stock_data: dict) -> dict:
    """
    Score fundamental quality and generate a labelled signal list.
    Returns: {signals: [...], quality: "strong|weak|mixed", score: int}
    """
    signals = []
    score   = 0

    pe  = stock_data.get("pe_ratio")
    fpe = stock_data.get("forward_pe")
    peg = stock_data.get("peg_ratio")
    roe = stock_data.get("roe")
    pm  = stock_data.get("profit_margin")
    de  = stock_data.get("debt_to_equity")
    eg  = stock_data.get("earnings_growth")
    rg  = stock_data.get("revenue_growth")

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    pe, fpe, peg = _num(pe), _num(fpe), _num(peg)
    roe, pm, de  = _num(roe), _num(pm), _num(de)
    eg, rg       = _num(eg), _num(rg)

    # Valuation
    if pe is not None:
        if pe < 0:
            signals.append(f"P/E negative — currently unprofitable")
            score -= 2
        elif pe < 12:
            signals.append(f"P/E {pe:.1f} — deep value")
            score += 2
        elif pe < 20:
            signals.append(f"P/E {pe:.1f} — reasonable valuation")
            score += 1
        elif pe > 50:
            signals.append(f"P/E {pe:.1f} — expensive, priced for perfection")
            score -= 1

    if fpe is not None and pe is not None and fpe > 0:
        if fpe < pe * 0.85:
            signals.append(f"Forward P/E {fpe:.1f} < Trailing {pe:.1f} — earnings growth expected")
            score += 1

    if peg is not None and peg > 0:
        if peg < 1.0:
            signals.append(f"PEG {peg:.2f} — undervalued relative to growth")
            score += 2
        elif peg > 3.0:
            signals.append(f"PEG {peg:.2f} — expensive relative to growth")
            score -= 1

    # Profitability
    if pm is not None:
        if pm > 0.25:
            signals.append(f"Net margin {pm*100:.1f}% — high-quality business")
            score += 2
        elif pm > 0.10:
            signals.append(f"Net margin {pm*100:.1f}% — healthy margins")
            score += 1
        elif pm < 0:
            signals.append(f"Net margin {pm*100:.1f}% — company losing money")
            score -= 2

    if roe is not None:
        if roe > 0.25:
            signals.append(f"ROE {roe*100:.1f}% — excellent capital efficiency")
            score += 2
        elif roe > 0.15:
            signals.append(f"ROE {roe*100:.1f}% — solid returns on equity")
            score += 1
        elif roe < 0:
            signals.append(f"ROE {roe*100:.1f}% — negative returns on equity")
            score -= 1

    # Leverage
    if de is not None:
        if de > 3.0:
            signals.append(f"D/E {de:.1f} — high leverage, financial risk")
            score -= 2
        elif de > 1.5:
            signals.append(f"D/E {de:.1f} — elevated debt")
            score -= 1
        elif de < 0.3:
            signals.append(f"D/E {de:.1f} — strong balance sheet")
            score += 1

    # Growth
    if eg is not None:
        if eg > 0.20:
            signals.append(f"EPS growth {eg*100:.1f}% YoY — strong growth")
            score += 2
        elif eg > 0:
            signals.append(f"EPS growth {eg*100:.1f}% YoY — positive growth")
            score += 1
        elif eg < -0.10:
            signals.append(f"EPS growth {eg*100:.1f}% YoY — declining earnings")
            score -= 2

    if rg is not None and rg > 0.15:
        signals.append(f"Revenue growth {rg*100:.1f}% YoY — top-line momentum")
        score += 1

    quality = "strong" if score >= 3 else "weak" if score <= -2 else "mixed"
    return {"signals": signals, "quality": quality, "score": score}


# ---------------------------------------------------------------------------
# News sentiment analyser
# ---------------------------------------------------------------------------

def _analyse_news_sentiment(news_results: list) -> dict:
    """
    Compute a [-1, +1] sentiment score from news tool results.
    Returns: {score, label, headline_count, top_headlines, source_breakdown}
    """
    POSITIVE = ["surge", "soar", "beat", "record", "strong", "growth", "gain",
                "rally", "profit", "rise", "upgrade", "buy", "bullish", "upbeat",
                "exceed", "outperform", "breakthrough", "partnership", "expansion"]
    NEGATIVE = ["fall", "drop", "miss", "weak", "loss", "decline", "concern",
                "risk", "volatile", "cut", "warning", "bearish", "crash", "slump",
                "downgrade", "sell", "lawsuit", "investigation", "default", "layoff",
                "recession", "inflation", "geopolit", "conflict", "sanction"]

    all_articles, source_counts = [], {}
    for result in (news_results or []):
        if isinstance(result, dict):
            articles = result.get("articles", [])
            source   = result.get("query", "unknown")
            source_counts[source] = len(articles)
            all_articles.extend(articles)

    if not all_articles:
        return {"score": 0.0, "label": "neutral", "headline_count": 0,
                "top_headlines": [], "source_breakdown": {}}

    score_sum, scored = 0.0, 0
    top_headlines = []

    for article in all_articles[:20]:
        title = (article.get("title") or "") if isinstance(article, dict) else str(article)
        if not title:
            continue
        top_headlines.append(title)
        tl = title.lower()
        pos = sum(1 for w in POSITIVE if w in tl)
        neg = sum(1 for w in NEGATIVE if w in tl)
        score_sum += (pos - neg) * 0.1
        scored += 1

    score = max(-1.0, min(1.0, score_sum / max(scored, 1) * 5))
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"

    return {
        "score":            round(score, 3),
        "label":            label,
        "headline_count":   len(all_articles),
        "top_headlines":    top_headlines[:5],
        "source_breakdown": source_counts,
    }


# ---------------------------------------------------------------------------
# Market context extractor
# ---------------------------------------------------------------------------

def _extract_market_context(market_data: Optional[dict]) -> dict:
    """Pull key index levels + VIX from a get_market_indices result."""
    if not market_data or "indices" not in market_data:
        return {}

    idx = market_data["indices"]

    def _idx(name):
        d = idx.get(name, {})
        price = d.get("price")
        chg   = d.get("change_pct")
        return {"price": price, "change_pct": chg} if price else {}

    vix_price = idx.get("VIX", {}).get("price")
    vix_regime = (
        "extreme_fear" if vix_price and vix_price > 35 else
        "fear"         if vix_price and vix_price > 25 else
        "neutral"      if vix_price and vix_price > 15 else
        "complacency"  if vix_price else None
    )

    return {
        "sp500":      _idx("S&P 500"),
        "nasdaq":     _idx("Nasdaq"),
        "dow":        _idx("Dow Jones"),
        "vix":        {**_idx("VIX"), "regime": vix_regime},
        "ta35":       _idx("TA-35"),
        "ta125":      _idx("TA-125"),
        "timestamp":  market_data.get("timestamp"),
    }


# ---------------------------------------------------------------------------
# Previous-analysis comparison
# ---------------------------------------------------------------------------

def _get_previous(ticker: str, analyses: list) -> Optional[dict]:
    for entry in reversed(analyses):
        if entry.get("ticker") == ticker.upper():
            return entry
    return None


def _build_comparison(prev: Optional[dict], new_rec: str, stock_data: dict) -> dict:
    if not prev:
        return {"is_first_analysis": True, "prior_analyses_total": 0}

    all_prior = [e for e in _load() if e.get("ticker") == stock_data.get("ticker", "")]
    prev_price = (prev.get("outcomes") or {}).get("entry_price") or \
                 (prev.get("price_data") or {}).get("price")
    curr_price = stock_data.get("current_price")
    price_chg = None
    if prev_price and curr_price:
        price_chg = round((curr_price - prev_price) / prev_price * 100, 2)

    prev_outcome_7d = (prev.get("outcomes") or {}).get("7d", {})

    return {
        "is_first_analysis":           False,
        "prior_analyses_total":        len(all_prior),
        "previous_analysis_timestamp": prev.get("timestamp"),
        "previous_recommendation":     prev.get("recommendation"),
        "direction_changed":           prev.get("recommendation") != new_rec,
        "price_change_since_last_pct": price_chg,
        "previous_confidence":         prev.get("confidence_score"),
        "previous_rsi":                (prev.get("price_data") or {}).get("rsi_14"),
        "previous_7d_outcome":         prev_outcome_7d.get("correct"),   # True/False/None
        "previous_7d_return_pct":      prev_outcome_7d.get("return_pct"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_stock_analysis(
    ticker: str,
    stock_data: dict,
    analysis_text: str,
    news_results: Optional[list] = None,
    market_data: Optional[dict] = None,
    tools_used: Optional[list] = None,
    analysis_duration_sec: Optional[float] = None,
    model_used: Optional[str] = None,
) -> dict:
    """
    Append a fully enriched stock analysis entry to stock_analyses.json.

    Args:
        ticker               : Ticker symbol (e.g. "AAPL").
        stock_data           : Dict from get_stock_data().
        analysis_text        : Full markdown response from the agent.
        news_results         : List of raw get_financial_news() result dicts.
        market_data          : Dict from get_market_indices().
        tools_used           : List of tool names called during the run.
        analysis_duration_sec: Wall-clock seconds from first API call to response.
        model_used           : Claude model ID used.

    Returns:
        The entry dict that was appended.
    """
    analyses   = _load()
    prev       = _get_previous(ticker, analyses)
    rec        = _extract_recommendation(analysis_text)
    confidence = _extract_confidence(analysis_text)
    tech_class = _classify_technical(stock_data)
    fund_class = _classify_fundamentals(stock_data)
    news_sent  = _analyse_news_sentiment(news_results)

    entry = {
        # ── Identity ──────────────────────────────────────────────────────
        "id":           f"{ticker.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp":    datetime.now().isoformat(),
        "ticker":       ticker.upper(),
        "company_name": stock_data.get("company_name", "N/A"),
        "sector":       stock_data.get("sector",       "N/A"),
        "industry":     stock_data.get("industry",     "N/A"),
        "currency":     stock_data.get("currency",     "USD"),

        # ── Recommendation & Confidence ───────────────────────────────────
        "recommendation":  rec,
        "confidence_score": confidence,       # 1–5
        "tldr":            _extract_tldr(analysis_text),

        # ── Price & Technicals ────────────────────────────────────────────
        "price_data": {k: v for k, v in {
            "price":               stock_data.get("current_price"),
            "change_pct":          stock_data.get("change_pct"),
            "week_52_high":        stock_data.get("week_52_high"),
            "week_52_low":         stock_data.get("week_52_low"),
            "sma_50":              stock_data.get("sma_50"),
            "sma_200":             stock_data.get("sma_200"),
            "rsi_14":              stock_data.get("rsi_14"),
            "price_vs_sma50_pct":  stock_data.get("price_vs_sma50_pct"),
            "price_vs_sma200_pct": stock_data.get("price_vs_sma200_pct"),
            "volume_ratio":        stock_data.get("volume_ratio"),
        }.items() if v is not None},

        # ── Fundamentals ──────────────────────────────────────────────────
        "fundamental_data": {k: v for k, v in {
            "market_cap":       stock_data.get("market_cap"),
            "market_cap_human": stock_data.get("market_cap_human"),
            "pe_ratio":         stock_data.get("pe_ratio"),
            "forward_pe":       stock_data.get("forward_pe"),
            "peg_ratio":        stock_data.get("peg_ratio"),
            "eps":              stock_data.get("eps"),
            "profit_margin":    stock_data.get("profit_margin"),
            "roe":              stock_data.get("roe"),
            "debt_to_equity":   stock_data.get("debt_to_equity"),
            "revenue_growth":   stock_data.get("revenue_growth"),
            "earnings_growth":  stock_data.get("earnings_growth"),
            "next_earnings":    stock_data.get("next_earnings_date"),
        }.items() if v is not None},

        # ── Derived Classifications ───────────────────────────────────────
        "technical_classification":   tech_class,
        "fundamental_classification": fund_class,

        # ── Market Context (snapshot at time of analysis) ─────────────────
        "market_context": _extract_market_context(market_data),

        # ── News Sentiment ────────────────────────────────────────────────
        "news_sentiment": news_sent,

        # ── Reasoning Chain (extracted from markdown) ─────────────────────
        "reasoning_chain": {
            "technical_signals":   _extract_bullets(analysis_text, ["Technical", "Technicals", "Technical Overview"]),
            "fundamental_signals": _extract_bullets(analysis_text, ["Fundamental", "Fundamentals", "Fundamental Snapshot"]),
            "risk_factors":        _extract_bullets(analysis_text, ["Risk", "Risks", "Risk Factors"]),
            "macro_signals":       _extract_bullets(analysis_text, ["Macro", "Geo-political", "Geo", "Context"]),
            "news_highlights":     _extract_bullets(analysis_text, ["News", "Sentiment", "Recent News"]),
        },

        # ── Comparison to Previous Analysis of Same Ticker ────────────────
        "vs_previous": _build_comparison(prev, rec, stock_data),

        # ── Meta ──────────────────────────────────────────────────────────
        "meta": {
            "tools_used":            sorted(set(tools_used or [])),
            "analysis_duration_sec": round(analysis_duration_sec, 2) if analysis_duration_sec else None,
            "model_used":            model_used,
            "logger_version":        "3.0",
        },

        # ── Full Analysis Text (for NLP / future fine-tuning) ─────────────
        "full_analysis": analysis_text,

        # ── Outcome Tracking (filled later by outcome_tracker.py) ─────────
        "outcomes": {
            "entry_price": stock_data.get("current_price"),
            "7d":  {"price": None, "return_pct": None, "correct": None, "checked_at": None},
            "30d": {"price": None, "return_pct": None, "correct": None, "checked_at": None},
            "90d": {"price": None, "return_pct": None, "correct": None, "checked_at": None},
        },
    }

    analyses.append(entry)
    _save(analyses)
    return entry
