"""
tools.py — Core data-fetching functions for FinanceGPT.
Handles stock data (yfinance), news (RSS feeds), and market indices.
"""

import json
import os
import re
from datetime import datetime, date, timedelta
from typing import Optional

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# RSS feed sources (no API key required)
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance":    "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":      "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Top News":    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Investing.com":    "https://www.investing.com/rss/news.rss",
    "Seeking Alpha":    "https://seekingalpha.com/feed.xml",
}

# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.iloc[:period].mean()
    avg_loss = loss.iloc[:period].mean()

    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _fmt(value, decimals: int = 2):
    """Safely round a numeric value, return None if missing."""
    try:
        return round(float(value), decimals) if value is not None else None
    except (TypeError, ValueError):
        return None


def _human_cap(value) -> Optional[str]:
    """Convert raw market-cap integer to human-readable string (e.g. 2.85T)."""
    if value is None:
        return None
    try:
        v = float(value)
        if v >= 1e12:
            return f"${v / 1e12:.2f}T"
        if v >= 1e9:
            return f"${v / 1e9:.2f}B"
        if v >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tool 1: get_stock_data
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str) -> dict:
    """
    Fetch comprehensive stock data for a given ticker symbol.

    Returns fundamentals (P/E, EPS, market cap, earnings date),
    technicals (price, SMA-50, SMA-200, RSI-14, 52-week range),
    and company profile (sector, industry, description).
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # --- Validate ticker: info must have at least a symbol or shortName ----
        if not info.get("symbol") and not info.get("shortName") and not info.get("longName"):
            return {"error": f"Ticker '{ticker}' not found or no data available."}

        # --- Historical price data (1 year for SMA-200) -----------------------
        hist = stock.history(period="1y", auto_adjust=True)
        if hist.empty:
            return {"error": f"No price history found for '{ticker}'."}

        close = hist["Close"]
        sma50  = _fmt(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        sma200 = _fmt(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        rsi14  = _calculate_rsi(close)

        current_price = _fmt(close.iloc[-1])
        week52_high   = _fmt(hist["High"].max())
        week52_low    = _fmt(hist["Low"].min())

        # Price relative to moving averages
        price_vs_sma50  = None
        price_vs_sma200 = None
        if current_price and sma50:
            price_vs_sma50  = round(((current_price - sma50)  / sma50)  * 100, 2)
        if current_price and sma200:
            price_vs_sma200 = round(((current_price - sma200) / sma200) * 100, 2)

        # --- Earnings date ----------------------------------------------------
        next_earnings = None
        try:
            cal = stock.calendar
            if cal is not None and not (isinstance(cal, dict) and len(cal) == 0):
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_earnings = str(ed[0]) if hasattr(ed, "__iter__") and not isinstance(ed, str) else str(ed)
                elif hasattr(cal, "iloc"):
                    # Older yfinance returns a DataFrame
                    if "Earnings Date" in cal.columns:
                        next_earnings = str(cal["Earnings Date"].iloc[0])
        except Exception:
            pass

        # --- Volume stats -----------------------------------------------------
        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        last_volume = int(hist["Volume"].iloc[-1]) if not hist["Volume"].empty else None
        volume_ratio = None
        if last_volume and avg_volume and avg_volume > 0:
            volume_ratio = round(last_volume / avg_volume, 2)

        return {
            "ticker":            ticker,
            "company_name":      info.get("longName") or info.get("shortName", "N/A"),
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
            "exchange":          info.get("exchange", "N/A"),
            "currency":          info.get("currency", "USD"),
            # Price / technicals
            "current_price":     current_price,
            "week_52_high":      week52_high,
            "week_52_low":       week52_low,
            "sma_50":            sma50,
            "sma_200":           sma200,
            "rsi_14":            rsi14,
            "price_vs_sma50_pct":  price_vs_sma50,
            "price_vs_sma200_pct": price_vs_sma200,
            "last_volume":       last_volume,
            "avg_volume":        avg_volume,
            "volume_ratio":      volume_ratio,
            "beta":              _fmt(info.get("beta"), 3),
            # Fundamentals
            "market_cap":        info.get("marketCap"),
            "market_cap_human":  _human_cap(info.get("marketCap")),
            "pe_ratio":          _fmt(info.get("trailingPE")),
            "forward_pe":        _fmt(info.get("forwardPE")),
            "peg_ratio":         _fmt(info.get("pegRatio")),
            "eps_ttm":           _fmt(info.get("trailingEps")),
            "eps_forward":       _fmt(info.get("forwardEps")),
            "revenue_growth_yoy":_fmt(info.get("revenueGrowth")),
            "earnings_growth_yoy":_fmt(info.get("earningsGrowth")),
            "profit_margin":     _fmt(info.get("profitMargins")),
            "roe":               _fmt(info.get("returnOnEquity")),
            "debt_to_equity":    _fmt(info.get("debtToEquity")),
            "dividend_yield":    _fmt(info.get("dividendYield"), 4),
            "next_earnings_date":next_earnings,
            # Analyst consensus (Wall Street)
            "analyst_target_price":     _fmt(info.get("targetMeanPrice")),
            "analyst_target_low":       _fmt(info.get("targetLowPrice")),
            "analyst_target_high":      _fmt(info.get("targetHighPrice")),
            "analyst_recommendation":   info.get("recommendationKey"),   # e.g. "buy", "hold", "sell"
            "analyst_count":            info.get("numberOfAnalystOpinions"),
            "analyst_upside_pct":       round((info.get("targetMeanPrice", 0) - current_price) / current_price * 100, 2)
                                        if info.get("targetMeanPrice") and current_price else None,
            # Ownership & short interest
            "short_percent_float":  _fmt(info.get("shortPercentOfFloat"), 4),
            "inst_ownership":       _fmt(info.get("heldPercentInstitutions"), 4),
            "insider_ownership":    _fmt(info.get("heldPercentInsiders"), 4),
            # Company info
            "description":       (info.get("longBusinessSummary") or "")[:600] or None,
            "website":           info.get("website"),
            "employees":         info.get("fullTimeEmployees"),
        }

    except Exception as exc:
        return {"error": f"Failed to fetch data for '{ticker}': {exc}"}


# ---------------------------------------------------------------------------
# Sector → ETF mapping (used by get_quant_metrics)
# ---------------------------------------------------------------------------
_SECTOR_ETF: dict[str, str] = {
    # Industry-level (checked first)
    "Semiconductors":                  "SMH",
    "Semiconductor Equipment":         "SMH",
    "Software—Application":            "IGV",
    "Software—Infrastructure":         "IGV",
    "Biotechnology":                   "XBI",
    "Drug Manufacturers—General":      "XPH",
    "Drug Manufacturers—Specialty":    "XPH",
    "Oil & Gas E&P":                   "XOP",
    "Oil & Gas Integrated":            "XLE",
    "Oil & Gas Midstream":             "XLE",
    "Banks—Diversified":               "XLF",
    "Banks—Regional":                  "KRE",
    "Aerospace & Defense":             "ITA",
    "Auto Manufacturers":              "CARZ",
    # Sector-level (fallback)
    "Technology":                      "XLK",
    "Financial Services":              "XLF",
    "Healthcare":                      "XLV",
    "Energy":                          "XLE",
    "Consumer Cyclical":               "XLY",
    "Consumer Defensive":              "XLP",
    "Industrials":                     "XLI",
    "Basic Materials":                 "XLB",
    "Real Estate":                     "XLRE",
    "Communication Services":          "XLC",
    "Utilities":                       "XLU",
}


# ---------------------------------------------------------------------------
# Quant metrics: RS, Beta, Sharpe, POC, Sector ETF
# ---------------------------------------------------------------------------

def get_quant_metrics(ticker: str) -> dict:
    """
    Calculate institutional-grade quant metrics for a stock:
      - Relative Strength (RS) vs SPY: excess return over 50d / 200d
      - RS Rating (1–99): IBD-style percentile approximation
      - Beta (60-day rolling vs SPY)
      - Sharpe Ratio (annualized, trailing 252 trading days)
      - Point of Control (POC): price level with highest traded volume
      - Sector ETF: mapped ETF symbol + 50-day performance vs SPY

    Returns a dict; on failure returns {"error": str}.
    """
    ticker = ticker.strip().upper()
    try:
        hist_stock = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        hist_spy   = yf.Ticker("SPY").history(period="1y", auto_adjust=True)

        if hist_stock.empty or len(hist_stock) < 60:
            return {"error": "Insufficient price history for quant metrics."}

        close_s = hist_stock["Close"]
        # Align SPY to stock trading dates
        close_m = hist_spy["Close"].reindex(close_s.index, method="ffill").dropna()
        close_s = close_s.reindex(close_m.index)

        # ── Relative Strength ─────────────────────────────────────────────────
        def _rs(days: int):
            if len(close_s) < days or len(close_m) < days:
                return None
            return round(
                float(close_s.iloc[-1] / close_s.iloc[-days] - 1) * 100
                - float(close_m.iloc[-1] / close_m.iloc[-days] - 1) * 100,
                2,
            )

        rs_50d  = _rs(50)
        rs_200d = _rs(200) if len(close_s) >= 200 else None

        # RS Rating 1–99: scaled excess 12-month return vs SPY
        stock_12m = float(close_s.iloc[-1] / close_s.iloc[0] - 1) * 100
        spy_12m   = float(close_m.iloc[-1] / close_m.iloc[0] - 1) * 100
        excess_12m = stock_12m - spy_12m
        rs_rating  = int(max(1, min(99, 50 + excess_12m / max(abs(spy_12m), 5) * 25)))

        # ── Beta (60-day rolling) ─────────────────────────────────────────────
        ret_s = close_s.pct_change().dropna()
        ret_m = close_m.pct_change().dropna()
        aligned = pd.DataFrame({"s": ret_s, "m": ret_m}).dropna().tail(60)
        if len(aligned) >= 30:
            cov    = float(np.cov(aligned["s"], aligned["m"])[0][1])
            var_m  = float(np.var(aligned["m"]))
            beta_60d = round(cov / var_m, 3) if var_m > 0 else None
        else:
            beta_60d = None

        # ── Sharpe Ratio (annualised, trailing year) ──────────────────────────
        rf_daily = 0.05 / 252   # 5% annualised risk-free proxy
        excess_d = ret_s - rf_daily
        sharpe = (
            round(float(excess_d.mean() / excess_d.std() * np.sqrt(252)), 3)
            if len(excess_d) >= 30 and excess_d.std() > 0
            else None
        )

        # ── Point of Control (POC) — volume-profile ───────────────────────────
        poc = None
        try:
            prices  = hist_stock["Close"].values.astype(float)
            volumes = hist_stock["Volume"].values.astype(float)
            p_min, p_max = prices.min(), prices.max()
            if p_max > p_min:
                n_bins = 60
                edges  = np.linspace(p_min, p_max, n_bins + 1)
                midpts = (edges[:-1] + edges[1:]) / 2
                vols   = np.zeros(n_bins)
                for p, v in zip(prices, volumes):
                    idx = min(int((p - p_min) / (p_max - p_min) * n_bins), n_bins - 1)
                    vols[idx] += v
                poc = round(float(midpts[int(np.argmax(vols))]), 2)
        except Exception:
            pass

        # ── Sector ETF comparison ─────────────────────────────────────────────
        sector_etf = sector_etf_50d = sector_vs_spy_50d = None
        try:
            info     = yf.Ticker(ticker).info or {}
            etf_sym  = (
                _SECTOR_ETF.get(info.get("industry", ""))
                or _SECTOR_ETF.get(info.get("sector", ""))
            )
            if etf_sym:
                sector_etf = etf_sym
                etf_close  = yf.Ticker(etf_sym).history(period="1y", auto_adjust=True)["Close"]
                etf_close  = etf_close.reindex(close_m.index, method="ffill").dropna()
                if len(etf_close) >= 50:
                    etf_50d  = round(float(etf_close.iloc[-1] / etf_close.iloc[-50] - 1) * 100, 2)
                    spy_50d_ = round(float(close_m.iloc[-1]   / close_m.iloc[-50]   - 1) * 100, 2)
                    sector_etf_50d    = etf_50d
                    sector_vs_spy_50d = round(etf_50d - spy_50d_, 2)
        except Exception:
            pass

        return {
            "ticker":             ticker,
            "rs_50d":             rs_50d,
            "rs_200d":            rs_200d,
            "rs_rating":          rs_rating,
            "beta_60d":           beta_60d,
            "sharpe_ratio":       sharpe,
            "point_of_control":   poc,
            "sector_etf":         sector_etf,
            "sector_etf_50d":     sector_etf_50d,
            "sector_vs_spy_50d":  sector_vs_spy_50d,
        }

    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 2: get_financial_news
# ---------------------------------------------------------------------------

def get_financial_news(query: str, max_results: int = 8) -> dict:
    """
    Search RSS feeds from major financial outlets for news matching `query`.

    Returns a list of articles with title, summary, source, publish date, and URL.
    Falls back to top headlines if no query-specific results are found.
    """
    query_lower = query.lower().strip()
    articles: list[dict] = []
    seen_titles: set[str] = set()

    for source, url in RSS_FEEDS.items():
        if len(articles) >= max_results:
            break
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for entry in feed.entries:
                if len(articles) >= max_results:
                    break
                title   = (entry.get("title")   or "").strip()
                summary = (entry.get("summary")  or entry.get("description") or "").strip()
                link    = entry.get("link",    "")
                pub     = entry.get("published", "")

                if not title or title in seen_titles:
                    continue

                # Relevance check (broad match or generic market queries)
                generic = query_lower in {"market", "general", "news", "headlines", "macro"}
                relevant = (
                    generic
                    or query_lower in title.lower()
                    or query_lower in summary.lower()
                )
                if not relevant:
                    continue

                seen_titles.add(title)
                # Strip HTML tags from summary
                clean_summary = re.sub(r"<[^>]+>", "", summary)[:250]
                articles.append({
                    "source":    source,
                    "title":     title,
                    "summary":   clean_summary,
                    "published": pub,
                    "link":      link,
                })
        except Exception:
            continue

    # Fallback: if nothing matched, return top headlines from first available feed
    if not articles:
        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
                for entry in feed.entries[:max_results]:
                    title = (entry.get("title") or "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    clean_summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:250]
                    articles.append({
                        "source":    source,
                        "title":     title,
                        "summary":   clean_summary,
                        "published": entry.get("published", ""),
                        "link":      entry.get("link", ""),
                    })
                if articles:
                    break
            except Exception:
                continue

    return {
        "query":    query,
        "count":    len(articles),
        "articles": articles[:max_results],
    }


# ---------------------------------------------------------------------------
# TASE sector indices — supplemental scraper
# ---------------------------------------------------------------------------

_TASE_SECTOR_TARGETS = {
    # display name → Hebrew search strings (checked via 'in')
    "TA-125":       ["ת\"א-125", "תל אביב 125", "ta-125", "ta 125"],
    "מדד בנקים-5":  ["בנקים 5", "בנקים-5", "bank", "bk35"],
    "ביטוח":        ["ביטוח", "insur", "tins"],
}

def _fetch_tase_extra_indices() -> dict:
    """
    Fetch TA-125, Banks-5, and Insurance index data from the TASE official REST API.
    Returns a dict keyed by display name, compatible with get_market_indices() output.
    Returns an empty dict (silently) on any failure.
    """
    _TASE_API = "https://api.tase.co.il/api/index/indicesdata"
    _HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Origin": "https://www.tase.co.il",
        "Referer": "https://www.tase.co.il/",
    }

    def _g(obj: dict, *keys, default=None):
        """Try multiple key spellings, return first match."""
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default

    try:
        resp = requests.get(_TASE_API, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}

    # The TASE API may wrap the list; try common field names
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get(
            "indicesData",
            payload.get("IndicesData",
            payload.get("data",
            payload.get("Data", []))),
        )
    else:
        return {}

    results: dict[str, dict] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        # Collect all string values from this item for matching
        name_fields = [
            str(_g(item, "indexName", "IndexName", "name", "Name", default="")).lower(),
            str(_g(item, "indexNameEn", "IndexNameEn", "nameEn", "NameEn", default="")).lower(),
            str(_g(item, "indexId", "IndexId", "id", "Id", default="")).lower(),
        ]
        combined = " ".join(name_fields)

        for display_name, keywords in _TASE_SECTOR_TARGETS.items():
            if display_name in results:
                continue  # already matched
            if any(kw.lower() in combined for kw in keywords):
                raw_price  = _g(item, "closingIndexRate", "ClosingIndexRate",
                                "lastPrice", "LastPrice", "price", "Price")
                raw_change = _g(item, "change", "Change", "indexChange", "IndexChange")
                raw_pct    = _g(item, "changePercentage", "ChangePercentage",
                                "changePct", "ChangePct", "pct", "Pct")

                try:
                    price  = float(raw_price)  if raw_price  is not None else None
                    change = float(raw_change) if raw_change is not None else None
                    pct    = float(raw_pct)    if raw_pct    is not None else None
                except (TypeError, ValueError):
                    price = change = pct = None

                results[display_name] = {
                    "symbol":     display_name,
                    "category":   "israel",
                    "price":      price,
                    "prev_close": round(price - change, 4) if (price and change) else None,
                    "change":     change,
                    "change_pct": pct,
                    "volume":     None,
                    "sparkline":  [],
                    "direction":  (
                        "up"   if pct and pct > 0 else
                        "down" if pct and pct < 0 else
                        "flat"
                    ),
                }
                break  # found this display_name, move to next item

    return results


# ---------------------------------------------------------------------------
# Tool 3: get_market_indices
# ---------------------------------------------------------------------------

def get_market_indices() -> dict:
    """
    Fetch price, daily change, volume, and 7-day sparkline for major indices and crypto.

    US:     S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX
    Israel: TA-35, TA-125
    Crypto: Bitcoin (BTC-USD), Ethereum (ETH-USD)
    """
    # (symbol, category) — only symbols confirmed working on Yahoo Finance
    INDEX_MAP: dict[str, tuple[str, str]] = {
        "S&P 500":      ("^GSPC",   "us"),
        "Nasdaq":       ("^IXIC",   "us"),
        "Dow Jones":    ("^DJI",    "us"),
        "Russell 2000": ("^RUT",    "us"),
        "VIX":          ("^VIX",    "us"),
        "TA-35":        ("TA35.TA", "israel"),
        "TA-90":        ("TA90.TA", "israel"),
        "Bitcoin":      ("BTC-USD", "crypto"),
        "Ethereum":     ("ETH-USD", "crypto"),
    }

    results: dict[str, dict] = {}

    for name, (symbol, category) in INDEX_MAP.items():
        try:
            ticker = yf.Ticker(symbol)
            # 10 trading days → enough for a 7-day sparkline + daily change
            hist   = ticker.history(period="10d", auto_adjust=True)

            if hist.empty:
                results[name] = {"symbol": symbol, "category": category,
                                 "error": "No data returned"}
                continue

            current = _fmt(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev       = _fmt(hist["Close"].iloc[-2])
                change     = _fmt(current - prev) if (current and prev) else None
                change_pct = _fmt(((current - prev) / prev) * 100) if (current and prev and prev != 0) else None
            else:
                prev = change = change_pct = None

            last_vol = hist["Volume"].iloc[-1]
            volume   = int(last_vol) if (last_vol is not None and not np.isnan(last_vol) and last_vol > 0) else None

            # 7-day sparkline (last 7 available closes)
            sparkline = [round(float(v), 4) for v in hist["Close"].tail(7).tolist()]

            results[name] = {
                "symbol":      symbol,
                "category":    category,
                "price":       current,
                "prev_close":  prev,
                "change":      change,
                "change_pct":  change_pct,
                "volume":      volume,
                "sparkline":   sparkline,
                "direction":   ("up" if change_pct and change_pct > 0
                                else "down" if change_pct and change_pct < 0
                                else "flat"),
            }
        except Exception as exc:
            results[name] = {"symbol": symbol, "category": category, "error": str(exc)}

    # Supplement with TA-125 / Banks-5 / Insurance from TASE official API
    tase_extra = _fetch_tase_extra_indices()
    results.update(tase_extra)

    return {
        "indices":   results,
        "timestamp": datetime.now().isoformat(),
        "date":      datetime.now().strftime("%A, %B %d, %Y"),
    }


# ---------------------------------------------------------------------------
# Helper (not a Claude tool): used by Live Monitor
# ---------------------------------------------------------------------------

def get_all_rss_headlines(max_per_feed: int = 10) -> list[dict]:
    """
    Pull the latest headlines from all RSS feeds for live-monitor screening.
    Returns a flat list of {source, title, published, link}.
    """
    headlines: list[dict] = []
    seen: set[str] = set()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for entry in feed.entries[:max_per_feed]:
                title = (entry.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                headlines.append({
                    "source":    source,
                    "title":     title,
                    "published": entry.get("published", ""),
                    "link":      entry.get("link", ""),
                })
        except Exception:
            continue

    return headlines


# ---------------------------------------------------------------------------
# Fibonacci Retracement & Extension + ML Forecasting
# ---------------------------------------------------------------------------

FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT    = [1.272, 1.618, 2.0, 2.618]   # Extension levels


def calculate_fibonacci_levels(ticker: str, period: str = "1y") -> dict:
    """
    Calculate Fibonacci retracement and extension levels from the
    significant High and Low over `period`.

    Returns:
        {
          "high": float, "low": float, "range": float,
          "direction": "uptrend"|"downtrend",
          "levels": {
              "0.0": price, "23.6": price, "38.2": price,
              "50.0": price, "61.8": price, "78.6": price, "100.0": price
          },
          "extensions": {"127.2": price, "161.8": price, "200.0": price, "261.8": price},
          "current_price": float,
          "nearest_level": str,          # e.g. "61.8%"
          "nearest_price": float,
          "distance_pct": float,         # % distance to nearest level
          "at_golden_pocket": bool,      # price between 50% and 61.8%
          "confluence_sma": str | None,  # which SMA is near a fib level
        }
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        hist  = stock.history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < 20:
            return {"error": "Insufficient price history for Fibonacci analysis."}

        swing_high = float(hist["High"].max())
        swing_low  = float(hist["Low"].min())
        price_rng  = swing_high - swing_low
        current    = float(hist["Close"].iloc[-1])

        # Determine trend direction: if price is in upper half → uptrend (retrace down)
        direction  = "uptrend" if current > (swing_low + price_rng * 0.5) else "downtrend"

        # Retracement levels (from high to low for uptrend)
        def _ret(ratio):
            if direction == "uptrend":
                return round(swing_high - ratio * price_rng, 4)
            else:
                return round(swing_low  + ratio * price_rng, 4)

        levels = {f"{r*100:.1f}": _ret(r) for r in FIB_RATIOS}

        # Extension levels
        def _ext(ratio):
            if direction == "uptrend":
                return round(swing_high + (ratio - 1.0) * price_rng, 4)
            else:
                return round(swing_low  - (ratio - 1.0) * price_rng, 4)

        extensions = {f"{r*100:.1f}": _ext(r) for r in FIB_EXT}

        # Nearest retracement level
        all_levels = {**levels}
        nearest_key = min(all_levels, key=lambda k: abs(all_levels[k] - current))
        nearest_price = all_levels[nearest_key]
        distance_pct  = round(abs(current - nearest_price) / current * 100, 2)

        # Golden Pocket (50% – 61.8%)
        gp_lo = min(levels["50.0"], levels["61.8"])
        gp_hi = max(levels["50.0"], levels["61.8"])
        at_golden_pocket = gp_lo <= current <= gp_hi

        # SMA confluence check
        sma50  = float(hist["Close"].rolling(50).mean().iloc[-1])  if len(hist) >= 50  else None
        sma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else None
        confluence_sma = None
        for sma_val, sma_name in [(sma50, "SMA-50"), (sma200, "SMA-200")]:
            if sma_val is None:
                continue
            for k, fib_price in levels.items():
                if abs(sma_val - fib_price) / max(fib_price, 1) < 0.02:   # within 2%
                    confluence_sma = f"{sma_name} ≈ {k}% (${fib_price:,.2f})"
                    break

        return {
            "ticker":           ticker.upper(),
            "high":             round(swing_high, 4),
            "low":              round(swing_low,  4),
            "range":            round(price_rng,  4),
            "direction":        direction,
            "levels":           levels,
            "extensions":       extensions,
            "current_price":    round(current, 4),
            "nearest_level":    f"{nearest_key}%",
            "nearest_price":    round(nearest_price, 4),
            "distance_pct":     distance_pct,
            "at_golden_pocket": at_golden_pocket,
            "confluence_sma":   confluence_sma,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_forecasting_models(ticker: str, forecast_days: int = 30) -> dict:
    """
    Run three forecasting models on historical close prices:
      1. Linear Regression  — simple trend extrapolation
      2. Random Forest      — feature-engineered tree ensemble
      3. Monte Carlo        — 1000-path GBM simulation

    Returns targets at `forecast_days` from today, with Fibonacci alignment info.
    """
    try:
        import yfinance as yf
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        stock = yf.Ticker(ticker.upper())
        hist  = stock.history(period="2y", auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return {"error": "Insufficient history for forecasting."}

        closes = hist["Close"].values.astype(float)
        n      = len(closes)

        # ── 1. Linear Regression ────────────────────────────────────────────
        X_lr = np.arange(n).reshape(-1, 1)
        lr   = LinearRegression().fit(X_lr, closes)
        lr_target = float(lr.predict([[n + forecast_days]])[0])

        # ── 2. Random Forest ────────────────────────────────────────────────
        # Features: lag-1, lag-5, lag-10, rolling mean 10, rolling std 10
        def make_features(arr):
            rows = []
            for i in range(10, len(arr)):
                rows.append([
                    arr[i - 1],
                    arr[i - 5],
                    arr[i - 10],
                    arr[i - 10:i].mean(),
                    arr[i - 10:i].std(),
                ])
            return np.array(rows)

        Xrf = make_features(closes)
        yrf = closes[10:]
        rf  = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(Xrf, yrf)

        # Walk-forward to forecast_days
        buf = list(closes)
        for _ in range(forecast_days):
            arr  = np.array(buf)
            feat = np.array([[arr[-1], arr[-5], arr[-10],
                              arr[-10:].mean(), arr[-10:].std()]])
            nxt  = float(rf.predict(feat)[0])
            buf.append(nxt)
        rf_target = round(buf[-1], 4)

        # ── 3. Monte Carlo (GBM, 1000 paths) ───────────────────────────────
        log_returns = np.diff(np.log(closes))
        mu          = log_returns.mean()
        sigma       = log_returns.std()
        S0          = closes[-1]
        np.random.seed(42)
        paths       = np.exp(
            np.cumsum(
                mu + sigma * np.random.randn(1000, forecast_days),
                axis=1,
            )
        ) * S0
        mc_median = round(float(np.median(paths[:, -1])), 4)
        mc_p10    = round(float(np.percentile(paths[:, -1], 10)), 4)
        mc_p90    = round(float(np.percentile(paths[:, -1], 90)), 4)

        # ── Fibonacci alignment check ───────────────────────────────────────
        fib_data   = calculate_fibonacci_levels(ticker)
        fib_align  = {}
        if "extensions" in fib_data:
            all_fib = {**fib_data.get("levels", {}), **fib_data.get("extensions", {})}
            for model_name, target in [
                ("linear_regression", lr_target),
                ("random_forest",     rf_target),
                ("monte_carlo_p50",   mc_median),
            ]:
                for fib_key, fib_price in all_fib.items():
                    if fib_price > 0 and abs(target - fib_price) / fib_price < 0.03:
                        fib_align[model_name] = {
                            "fib_level": f"{fib_key}%",
                            "fib_price": round(fib_price, 2),
                            "delta_pct": round((target - fib_price) / fib_price * 100, 2),
                        }
                        break

        return {
            "ticker":         ticker.upper(),
            "forecast_days":  forecast_days,
            "current_price":  round(float(closes[-1]), 4),
            "linear_regression": {
                "target": round(lr_target, 4),
                "change_pct": round((lr_target - closes[-1]) / closes[-1] * 100, 2),
            },
            "random_forest": {
                "target": rf_target,
                "change_pct": round((rf_target - closes[-1]) / closes[-1] * 100, 2),
            },
            "monte_carlo": {
                "median":     mc_median,
                "p10_bear":   mc_p10,
                "p90_bull":   mc_p90,
                "change_pct": round((mc_median - closes[-1]) / closes[-1] * 100, 2),
            },
            "fibonacci_alignment": fib_align,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# YouTube Transcript Extractor
# ---------------------------------------------------------------------------

def _extract_video_id(url: str) -> str | None:
    """Extract 11-char YouTube video ID from any common URL format."""
    patterns = [
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript(url: str) -> dict:
    """
    Extract the spoken transcript from a YouTube video URL.

    Tries Hebrew (iw/he) first, then English, then any available language.

    Returns:
        Success: {"video_id": str, "transcript": str, "language": str, "word_count": int}
        Failure: {"error": str}
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return {"error": "youtube-transcript-api לא מותקן. הרץ: pip install youtube-transcript-api"}

    video_id = _extract_video_id(url)
    if not video_id:
        return {"error": "לא ניתן לחלץ ID מהקישור — ודא שהקישור תקין."}

    api = YouTubeTranscriptApi()

    # Try preferred languages first
    for langs, label in [
        (["iw", "he"],              "עברית"),
        (["en", "en-US", "en-GB"], "אנגלית"),
    ]:
        try:
            fetched = api.fetch(video_id, languages=langs)
            text = " ".join(item.text for item in fetched)
            return {
                "video_id":   video_id,
                "transcript": text,
                "language":   label,
                "word_count": len(text.split()),
            }
        except Exception:
            continue

    # Fallback: grab the first available transcript
    try:
        transcript_list = api.list(video_id)
        for t in transcript_list:
            try:
                fetched = t.fetch()
                text = " ".join(item.text for item in fetched)
                return {
                    "video_id":   video_id,
                    "transcript": text,
                    "language":   getattr(t, "language", "לא ידוע"),
                    "word_count": len(text.split()),
                }
            except Exception:
                continue
    except Exception:
        pass

    return {"error": f"לא נמצאו כתוביות לסרטון זה (ID: {video_id}). ייתכן שהכתוביות מושבתות."}


# ---------------------------------------------------------------------------
# Quarterly EPS history (last 8 quarters)
# ---------------------------------------------------------------------------

def get_quarterly_earnings(ticker: str) -> dict:
    """
    Fetch the last 8 quarters of EPS data (actual vs estimate).

    Returns:
        {"ticker": str, "quarters": [{"quarter", "actual", "estimate", "surprise_pct"}, ...]}
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)

        # earnings_history: columns epsActual, epsEstimate, epsDifference, surprisePercent
        hist = stock.earnings_history
        if hist is not None and not hist.empty:
            hist = hist.tail(8).reset_index()
            records = []
            for _, row in hist.iterrows():
                label = str(row.get("quarter", row.iloc[0]))
                records.append({
                    "quarter":      label,
                    "actual":       _fmt(row.get("epsActual"), 2),
                    "estimate":     _fmt(row.get("epsEstimate"), 2),
                    "surprise_pct": _fmt(row.get("surprisePercent"), 1),
                })
            if records:
                return {"ticker": ticker, "quarters": records}

        # Fallback: quarterly_earnings DataFrame
        qe = stock.quarterly_earnings
        if qe is not None and not qe.empty:
            qe = qe.tail(8).reset_index()
            records = []
            for _, row in qe.iterrows():
                label = str(row.iloc[0])
                records.append({
                    "quarter":      label,
                    "actual":       _fmt(row.get("Earnings"), 2),
                    "estimate":     None,
                    "surprise_pct": None,
                })
            return {"ticker": ticker, "quarters": records}

        return {"error": "No quarterly earnings data available."}

    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# NASDAQ 100 Smart Screener
# ---------------------------------------------------------------------------

NASDAQ100_TICKERS: list[str] = [
    # Mega-cap tech / AI
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    # Semiconductors
    "AMD", "QCOM", "AMAT", "MU", "LRCX", "KLAC", "ADI", "NXPI", "MCHP",
    "MPWR", "ON", "MRVL", "INTC", "ASML", "ARM", "SMCI",
    # Software / Cloud / Cybersecurity
    "ADBE", "INTU", "CDNS", "SNPS", "WDAY", "TEAM", "CRWD", "PANW", "FTNT",
    "ZS", "DDOG", "APP", "TTD", "ANSS", "FICO", "ROP",
    # Internet / E-commerce / Media
    "NFLX", "MELI", "DASH", "PYPL", "ABNB", "WBD", "CMCSA", "CHTR",
    # Healthcare / Biotech
    "REGN", "GILD", "VRTX", "BIIB", "IDXX", "DXCM", "GEHC", "AZN", "ILMN", "AMGN",
    # Consumer / Retail
    "COST", "BKNG", "SBUX", "LULU", "ORLY", "ROST", "DLTR", "MNST", "MDLZ",
    # Industrials / Transport
    "CSX", "PCAR", "ODFL", "FAST", "CTAS", "HON", "CPRT", "VRSK",
    # Fintech / Services
    "PAYX", "CDW",
    # Utilities / Energy
    "AEP", "EXC", "CEG", "FANG",
    # Telecom / Other
    "TMUS", "KDP", "PEP", "TXN", "CSCO", "SIRI", "TTWO", "ZM", "ENPH",
]


def run_nasdaq_screener(
    min_rs: int   = 70,
    max_beta: float = 2.0,
    min_sharpe: float = 0.3,
    above_sma50: bool = True,
    min_rsi: float = 20,
    max_rsi: float = 80,
) -> list[dict]:
    """
    Batch-download 1-year price data for all NASDAQ 100 tickers + SPY in a
    single network call, then compute RS Rating, Beta (60d), Sharpe (1y),
    RSI-14, and SMA-50/200 for each ticker.  Apply the requested filters
    and return results sorted by RS Rating descending.

    Returns:
        list of dicts: {ticker, price, change_pct, rs_rating, rs_50d,
                        beta, sharpe, rsi, above_sma50, above_sma200}
    """
    all_tickers = NASDAQ100_TICKERS + ["SPY"]

    try:
        raw = yf.download(
            all_tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yfinance returns MultiIndex columns (Metric, Ticker) for multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw  # single ticker fallback (shouldn't happen here)
    except Exception as exc:
        return [{"error": str(exc)}]

    if "SPY" not in closes.columns:
        return [{"error": "SPY data unavailable — cannot compute RS Rating"}]

    spy_full = closes["SPY"].dropna()
    results: list[dict] = []

    for ticker in NASDAQ100_TICKERS:
        if ticker not in closes.columns:
            continue

        stock_full = closes[ticker].dropna()
        if len(stock_full) < 60:
            continue

        # Align to common trading days
        common  = stock_full.index.intersection(spy_full.index)
        if len(common) < 60:
            continue
        stock = stock_full.reindex(common)
        spy   = spy_full.reindex(common)

        price   = float(stock.iloc[-1])
        prev    = float(stock.iloc[-2]) if len(stock) >= 2 else price
        chg_pct = round((price - prev) / prev * 100, 2) if prev else None

        # ── RS Rating (12-month excess return, scaled 1-99) ─────────────────
        s12 = float(stock.iloc[-1] / stock.iloc[0]  - 1) * 100
        m12 = float(spy.iloc[-1]   / spy.iloc[0]    - 1) * 100
        excess12 = s12 - m12
        rs_rating = int(max(1, min(99, 50 + excess12 / max(abs(m12), 5) * 25)))

        # ── RS 50d excess return ─────────────────────────────────────────────
        rs_50d = None
        if len(stock) >= 50:
            s50 = float(stock.iloc[-1] / stock.iloc[-50] - 1) * 100
            m50 = float(spy.iloc[-1]   / spy.iloc[-50]   - 1) * 100
            rs_50d = round(s50 - m50, 2)

        # ── Beta 60d ─────────────────────────────────────────────────────────
        beta = None
        ret_s = stock.pct_change().dropna().tail(60)
        ret_m = spy.pct_change().dropna().tail(60)
        aligned = pd.DataFrame({"s": ret_s, "m": ret_m}).dropna()
        if len(aligned) >= 30:
            cov   = float(np.cov(aligned["s"], aligned["m"])[0][1])
            var_m = float(np.var(aligned["m"]))
            beta  = round(cov / var_m, 2) if var_m > 0 else None

        # ── Sharpe Ratio (annualised, trailing 1y) ───────────────────────────
        sharpe = None
        ret_all = stock.pct_change().dropna()
        rf_d    = 0.05 / 252
        exc_d   = ret_all - rf_d
        if len(exc_d) >= 30 and exc_d.std() > 0:
            sharpe = round(float(exc_d.mean() / exc_d.std() * np.sqrt(252)), 2)

        # ── RSI-14 ───────────────────────────────────────────────────────────
        rsi = _calculate_rsi(stock)

        # ── SMAs ─────────────────────────────────────────────────────────────
        sma50_val  = float(stock.rolling(50).mean().iloc[-1])  if len(stock) >= 50  else None
        sma200_val = float(stock.rolling(200).mean().iloc[-1]) if len(stock) >= 200 else None
        ok_sma50   = (sma50_val  is not None and price > sma50_val)
        ok_sma200  = (sma200_val is not None and price > sma200_val)

        # ── Apply filters ─────────────────────────────────────────────────────
        if rs_rating < min_rs:
            continue
        if beta is not None and beta > max_beta:
            continue
        if sharpe is not None and sharpe < min_sharpe:
            continue
        if above_sma50 and not ok_sma50:
            continue
        if rsi is not None and (rsi < min_rsi or rsi > max_rsi):
            continue

        results.append({
            "ticker":      ticker,
            "price":       round(price, 2),
            "change_pct":  chg_pct,
            "rs_rating":   rs_rating,
            "rs_50d":      rs_50d,
            "beta":        beta,
            "sharpe":      sharpe,
            "rsi":         round(rsi, 1) if rsi else None,
            "above_sma50":  ok_sma50,
            "above_sma200": ok_sma200,
        })

    return sorted(results, key=lambda x: x["rs_rating"], reverse=True)


# ---------------------------------------------------------------------------
# Portfolio Snapshot — user's core holdings
# ---------------------------------------------------------------------------

PORTFOLIO_TICKERS = ["NVDA", "SOFI", "IBIT", "ETHA", "MAGS", "MSTU"]

_PORTFOLIO_LABELS: dict[str, str] = {
    "NVDA": "NVIDIA",
    "SOFI": "SoFi",
    "IBIT": "iShares BTC",
    "ETHA": "iShares ETH",
    "MAGS": "Magnificent 7",
    "MSTU": "MicroStrategy 2x",
}


def get_portfolio_snapshot() -> dict:
    """
    Fetch current price and daily change for the user's 6 core holdings.

    Returns:
        {ticker: {"label", "price", "change", "change_pct", "direction"}}
    """
    results: dict[str, dict] = {}
    for ticker in PORTFOLIO_TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if hist.empty:
                results[ticker] = {"label": _PORTFOLIO_LABELS.get(ticker, ticker), "error": "no data"}
                continue
            current = _fmt(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev       = _fmt(hist["Close"].iloc[-2])
                change     = _fmt(current - prev) if (current and prev) else None
                change_pct = _fmt(((current - prev) / prev) * 100) if (current and prev and prev != 0) else None
            else:
                change = change_pct = None
            results[ticker] = {
                "label":      _PORTFOLIO_LABELS.get(ticker, ticker),
                "price":      current,
                "change":     change,
                "change_pct": change_pct,
                "direction":  ("up" if change_pct and change_pct > 0
                               else "down" if change_pct and change_pct < 0
                               else "flat"),
            }
        except Exception as exc:
            results[ticker] = {"label": _PORTFOLIO_LABELS.get(ticker, ticker), "error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# Macro Dashboard: key market/economic indicators
# ---------------------------------------------------------------------------

_MACRO_MAP: dict[str, str] = {
    "VIX":       "^VIX",
    "10Y Yield": "^TNX",
    "DXY":       "DX-Y.NYB",
    "Gold":      "GC=F",
    "Oil (WTI)": "CL=F",
    "SPY":       "SPY",
    "QQQ":       "QQQ",
}


def get_macro_data() -> dict:
    """
    Fetch current price and 1-day change for key macro indicators:
      VIX, 10-Year Treasury Yield, DXY (Dollar Index),
      Gold Futures, WTI Crude, SPY, QQQ.

    Returns {"macro": {name: {"symbol", "price", "change_pct"}}, "timestamp": str}
    """
    results: dict[str, dict] = {}

    for name, symbol in _MACRO_MAP.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
            if hist.empty:
                results[name] = {"symbol": symbol, "price": None, "change_pct": None}
                continue

            current = _fmt(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev = _fmt(hist["Close"].iloc[-2])
                change_pct = (
                    _fmt(((current - prev) / prev) * 100)
                    if current is not None and prev and prev != 0
                    else None
                )
            else:
                change_pct = None

            results[name] = {
                "symbol":     symbol,
                "price":      current,
                "change_pct": change_pct,
            }
        except Exception:
            results[name] = {"symbol": symbol, "price": None, "change_pct": None}

    return {"macro": results, "timestamp": datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# Watchlist persistence & lightweight quote
# ---------------------------------------------------------------------------

_FAVORITES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favorites.json")
_DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "TSLA", "SOFI"]


def _load_favorites_raw() -> dict:
    """Load the raw favorites file, migrating the old list format if needed."""
    try:
        with open(_FAVORITES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old format: ["NVDA", "AAPL", ...] → new dict format
        if isinstance(data, list):
            return {"tickers": [str(t).upper().strip() for t in data if t], "verdicts": {}}
        if isinstance(data, dict):
            return {
                "tickers":  [str(t).upper().strip() for t in data.get("tickers", []) if t],
                "verdicts": data.get("verdicts", {}),
            }
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"tickers": list(_DEFAULT_WATCHLIST), "verdicts": {}}


def _save_favorites_raw(data: dict) -> None:
    try:
        with open(_FAVORITES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_watchlist() -> list[str]:
    """Return the ordered list of favorite tickers."""
    return _load_favorites_raw()["tickers"]


def load_watchlist_verdicts() -> dict:
    """Return the verdict metadata dict: {ticker: {rec, confidence}}."""
    return _load_favorites_raw().get("verdicts", {})


def save_watchlist(tickers: list[str]) -> None:
    """Persist ticker order; preserves existing verdict metadata."""
    raw = _load_favorites_raw()
    raw["tickers"] = [t.upper().strip() for t in tickers]
    # Drop verdicts for tickers no longer in the list
    raw["verdicts"] = {k: v for k, v in raw.get("verdicts", {}).items()
                       if k in raw["tickers"]}
    _save_favorites_raw(raw)


def update_watchlist_verdict(ticker: str, rec: str, confidence: int) -> None:
    """Attach or refresh the AI verdict for a watchlist ticker."""
    ticker = ticker.upper().strip()
    raw = _load_favorites_raw()
    if ticker not in raw["tickers"]:
        return   # only store verdicts for saved tickers
    raw.setdefault("verdicts", {})[ticker] = {
        "rec":        rec.upper(),
        "confidence": int(confidence),
    }
    _save_favorites_raw(raw)


# ---------------------------------------------------------------------------
# Trending Stocks — Heat Score engine
# ---------------------------------------------------------------------------

# Curated pool: liquid, large-cap (>$2B by definition), sector-diverse
TRENDING_CANDIDATES: list[str] = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","NFLX",
    "AMD","AVGO","CRM","QCOM","MU","AMAT","SMCI","ARM","INTC",
    "JPM","BAC","GS","V","MA","PYPL","COIN",
    "LLY","PFE","ABBV","MRNA","JNJ",
    "XOM","CVX",
    "HD","NKE","SBUX",
    "PLTR","SOFI","MSTR","RKLB","IONQ",
]

# Static sector ETF per candidate — avoids slow info() calls
_CANDIDATE_ETF: dict[str, str] = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","AVGO":"XLK",
    "CRM":"XLK","QCOM":"XLK","MU":"XLK","AMAT":"XLK","SMCI":"XLK",
    "ARM":"XLK","INTC":"XLK","PLTR":"XLK","MSTR":"XLK","IONQ":"XLK",
    "GOOGL":"XLC","META":"XLC","NFLX":"XLC",
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","NKE":"XLY","SBUX":"XLY",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","V":"XLF","MA":"XLF",
    "PYPL":"XLF","COIN":"XLF","SOFI":"XLF",
    "LLY":"XLV","PFE":"XLV","ABBV":"XLV","MRNA":"XLV","JNJ":"XLV",
    "XOM":"XLE","CVX":"XLE",
    "RKLB":"XLI",
}

_ETF_NAME_HE: dict[str, str] = {
    "XLK":"טכנולוגיה","XLC":"תקשורת","XLY":"צריכה שיקולית",
    "XLF":"פיננסים","XLV":"בריאות","XLI":"תעשייה","XLE":"אנרגיה",
}


def _detect_ma_pattern(price: float, sma44: Optional[float], sma150: Optional[float]) -> str:
    """
    Classify a stock's position relative to SMA_44 and SMA_150.

    Returns one of:
      "bullish_stack"    — Price > SMA_44 > SMA_150 (strong uptrend)
      "mean_reversion"   — Price > 5% below SMA_150 (oversold/reversion candidate)
      "support_test_150" — Price within ±2% of SMA_150 (testing 150-day support)
      "support_test_44"  — Price within ±2% of SMA_44  (testing 44-day support)
      "above_150"        — Price above SMA_150 but not a full bullish stack
      "below_44"         — Price below SMA_44
      "neutral"          — No specific pattern
      "N/A"              — Insufficient SMA data
    """
    if sma44 is None and sma150 is None:
        return "N/A"

    near_44  = sma44  is not None and abs(price - sma44)  / sma44  <= 0.02
    near_150 = sma150 is not None and abs(price - sma150) / sma150 <= 0.02

    if sma44 is not None and sma150 is not None and price > sma44 > sma150:
        return "bullish_stack"

    if sma150 is not None and price < sma150 * 0.95:
        return "mean_reversion"

    if near_150:
        return "support_test_150"

    if near_44:
        return "support_test_44"

    if sma150 is not None and price > sma150:
        return "above_150"

    if sma44 is not None and price < sma44:
        return "below_44"

    return "neutral"


def get_trending_stocks(top_n: int = 5) -> list[dict]:
    """
    Scan TRENDING_CANDIDATES and return the top_n stocks by Heat Score (0–100).

    Score components:
      +30  Relative Volume ≥ 2× 10-day average
      +20  Gap Up ≥ 2% above previous close
      +20  Sector outperformance ≥ 2% vs sector ETF
      +15  Volatility spike ≥ 1.5× ATR(14)
      +15  RSI(14) just crossed above 70

    Each result dict also includes: sma_44, sma_150, dist_to_44_pct,
    dist_to_150_pct, technical_state.
    """
    # ── 1. Batch-download 1 year of data (needed for SMA_44 and SMA_150) ─────
    all_fetch = list(set(TRENDING_CANDIDATES + list(set(_CANDIDATE_ETF.values()))))
    try:
        raw = yf.download(
            all_fetch, period="1y", auto_adjust=True,
            group_by="ticker", progress=False, threads=True,
        )
    except Exception:
        return []

    def _ticker_df(sym: str) -> pd.DataFrame | None:
        try:
            df = raw[sym] if len(all_fetch) > 1 else raw
            df = df.dropna(how="all")
            return df if len(df) >= 15 else None
        except Exception:
            return None

    # ── 2. Pre-compute sector ETF daily returns ──────────────────────────────
    etf_rets: dict[str, float] = {}
    for etf in set(_CANDIDATE_ETF.values()):
        df = _ticker_df(etf)
        if df is not None and len(df) >= 2:
            etf_rets[etf] = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100

    # ── 3. Score each candidate ──────────────────────────────────────────────
    results: list[dict] = []
    for ticker in TRENDING_CANDIDATES:
        df = _ticker_df(ticker)
        if df is None:
            continue
        try:
            score    = 0
            triggers: list[str] = []
            today     = df.iloc[-1]
            yesterday = df.iloc[-2]

            # Relative Volume (+30)
            avg_vol   = df["Volume"].iloc[-11:-1].mean()
            vol_ratio = float(today["Volume"]) / avg_vol if avg_vol > 0 else 0
            if vol_ratio >= 2.0:
                score += 30
                triggers.append(f"מחזור פי {vol_ratio:.1f}x מהממוצע")

            # Gap Up (+20)
            gap_pct = (float(today["Open"]) - float(yesterday["Close"])) \
                      / float(yesterday["Close"]) * 100
            if gap_pct >= 2.0:
                score += 20
                triggers.append(f"פתיחת פער חיובי של {gap_pct:.1f}%")

            # Sector Outperformance (+20)
            stock_ret   = (float(today["Close"]) / float(yesterday["Close"]) - 1) * 100
            etf_sym     = _CANDIDATE_ETF.get(ticker, "")
            etf_ret     = etf_rets.get(etf_sym, 0.0)
            outperf     = stock_ret - etf_ret
            if outperf >= 2.0:
                score += 20
                sector_he = _ETF_NAME_HE.get(etf_sym, "הסקטור")
                triggers.append(f"עוצמה יחסית +{outperf:.1f}% על {sector_he}")

            # Volatility Spike — ATR(14) (+15)
            tr_vals: list[float] = []
            for i in range(1, 15):
                if i + 1 >= len(df):
                    break
                h  = float(df["High"].iloc[-i])
                lo = float(df["Low"].iloc[-i])
                pc = float(df["Close"].iloc[-i - 1])
                tr_vals.append(max(h - lo, abs(h - pc), abs(lo - pc)))
            if len(tr_vals) >= 10:
                atr14       = sum(tr_vals) / len(tr_vals)
                today_range = float(today["High"]) - float(today["Low"])
                if atr14 > 0 and today_range >= 1.5 * atr14:
                    score += 15
                    triggers.append(f"תנודתיות יומית פי {today_range / atr14:.1f}x ATR")

            # RSI Momentum — crossed above 70 (+15)
            rsi_now  = _calculate_rsi(df["Close"], 14)
            rsi_prev = _calculate_rsi(df["Close"].iloc[:-1], 14)
            if rsi_now and rsi_prev and rsi_now >= 70 and rsi_prev < 70:
                score += 15
                triggers.append(f"RSI חצה מעל 70 ({rsi_now:.0f})")

            if score < 15:
                continue

            # ── SMA_44 and SMA_150 calculations ─────────────────────────────
            close_series  = df["Close"]
            current_price = float(today["Close"])

            sma44  = float(close_series.rolling(44).mean().iloc[-1])  if len(close_series) >= 44  else None
            sma150 = float(close_series.rolling(150).mean().iloc[-1]) if len(close_series) >= 150 else None

            dist_to_44_pct  = round((current_price - sma44)  / sma44  * 100, 2) if sma44  else None
            dist_to_150_pct = round((current_price - sma150) / sma150 * 100, 2) if sma150 else None

            technical_state = _detect_ma_pattern(current_price, sma44, sma150)

            badge = (
                "🔥 חם מאוד" if score >= 70 else
                "⚡ בתנופה"  if score >= 50 else
                "📈 עולה"    if score >= 30 else
                "👀 מעקב"
            )
            reason_he = " · ".join(triggers[:2]) + "." if triggers else "פעילות חריגה."

            results.append({
                "ticker":          ticker,
                "score":           score,
                "badge":           badge,
                "reason_he":       reason_he,
                "price":           _fmt(today["Close"]),
                "change_pct":      round(stock_ret, 2),
                "sma_44":          round(sma44,  2) if sma44  else None,
                "sma_150":         round(sma150, 2) if sma150 else None,
                "dist_to_44_pct":  dist_to_44_pct,
                "dist_to_150_pct": dist_to_150_pct,
                "technical_state": technical_state,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Economic Calendar — US & Israel high-impact events
# ---------------------------------------------------------------------------

# FOMC statement dates (official Fed calendar — 8 meetings/year)
_FOMC_DATES: list[str] = [
    "2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-11-05","2025-12-17",
    "2026-01-29","2026-03-18","2026-04-29","2026-06-17",
    "2026-07-29","2026-09-16","2026-10-28","2026-12-09",
]

# Bank of Israel MPC decision dates
_BOI_DATES: list[str] = [
    "2025-02-24","2025-04-07","2025-05-26","2025-07-07",
    "2025-08-25","2025-10-06","2025-11-24",
    "2026-01-12","2026-02-24","2026-04-07","2026-05-25",
    "2026-07-06","2026-08-24","2026-10-05","2026-11-23",
]

# Israeli CPI release dates (typically mid-month)
_IL_CPI_DATES: list[str] = [
    "2025-03-15","2025-04-15","2025-05-15","2025-06-15",
    "2025-07-15","2025-08-15","2025-09-15","2025-10-15",
    "2025-11-15","2025-12-15",
    "2026-01-15","2026-02-15","2026-03-15","2026-04-15",
    "2026-05-15","2026-06-15","2026-07-15","2026-08-15",
]


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of a given month."""
    from calendar import monthrange
    d = date(year, month, 1)
    while d.weekday() != 4:   # 4 = Friday
        d += timedelta(days=1)
    return d


def _release_day(year: int, month: int, day: int) -> date:
    """Return date; if it falls on weekend push to next Monday."""
    d = date(year, month, min(day, 28))
    if d.weekday() == 5: d += timedelta(days=2)
    if d.weekday() == 6: d += timedelta(days=1)
    return d


def get_economic_calendar(days_ahead: int = 90) -> dict:
    """
    Return upcoming high-impact US and Israeli economic events.

    Builds the calendar from published official schedules (Fed, BLS, BOI, BEA).
    Each event: {date, time, event, event_he, previous, forecast, impact, country}
    impact: "critical" | "high"
    """
    today  = date.today()
    cutoff = today + timedelta(days=days_ahead)

    us_events:     list[dict] = []
    israel_events: list[dict] = []

    # ── US Events ─────────────────────────────────────────────────────────────

    # FOMC Rate Decision
    for ds in _FOMC_DATES:
        d = date.fromisoformat(ds)
        if today <= d <= cutoff:
            us_events.append({
                "date": ds, "time": "14:00 ET",
                "event": "FOMC Interest Rate Decision",
                "event_he": "החלטת ריבית — Fed",
                "previous": "4.50%", "forecast": "4.50%",
                "impact": "critical", "country": "us",
                "category": "monetary_policy",
            })

    # NFP — first Friday every month
    for month_offset in range(4):
        ref = today + timedelta(days=month_offset * 30)
        ff  = _first_friday(ref.year, ref.month)
        if today <= ff <= cutoff:
            us_events.append({
                "date": ff.isoformat(), "time": "08:30 ET",
                "event": "Non-Farm Payrolls",
                "event_he": "דוח שוק העבודה — NFP",
                "previous": "151K", "forecast": "155K",
                "impact": "critical", "country": "us",
                "category": "employment",
            })

    # CPI — around 12th each month
    for month_offset in range(4):
        ref = today + timedelta(days=month_offset * 30)
        d   = _release_day(ref.year, ref.month, 12)
        if today <= d <= cutoff:
            us_events.append({
                "date": d.isoformat(), "time": "08:30 ET",
                "event": "CPI (Consumer Price Index)",
                "event_he": "מדד המחירים לצרכן — CPI",
                "previous": "2.8% YoY", "forecast": "2.9% YoY",
                "impact": "critical", "country": "us",
                "category": "inflation",
            })

    # PPI — around 14th each month
    for month_offset in range(4):
        ref = today + timedelta(days=month_offset * 30)
        d   = _release_day(ref.year, ref.month, 14)
        if today <= d <= cutoff:
            us_events.append({
                "date": d.isoformat(), "time": "08:30 ET",
                "event": "PPI (Producer Price Index)",
                "event_he": "מדד מחירי היצרן — PPI",
                "previous": "3.2% YoY", "forecast": "3.1% YoY",
                "impact": "high", "country": "us",
                "category": "inflation",
            })

    # GDP — quarterly, around end of month
    for q_month in [1, 4, 7, 10]:
        d = _release_day(today.year, q_month, 30) if q_month != 2 else \
            _release_day(today.year, q_month, 28)
        if today <= d <= cutoff:
            us_events.append({
                "date": d.isoformat(), "time": "08:30 ET",
                "event": "GDP Growth Rate (Advance)",
                "event_he": "צמיחת התמ\"ג — GDP",
                "previous": "2.3%", "forecast": "2.5%",
                "impact": "critical", "country": "us",
                "category": "growth",
            })

    # PCE — around 28th each month
    for month_offset in range(3):
        ref = today + timedelta(days=month_offset * 30)
        d   = _release_day(ref.year, ref.month, 28)
        if today <= d <= cutoff:
            us_events.append({
                "date": d.isoformat(), "time": "08:30 ET",
                "event": "PCE Price Index",
                "event_he": "מדד PCE — האינפלציה המועדפת על ה-Fed",
                "previous": "2.5% YoY", "forecast": "2.5% YoY",
                "impact": "high", "country": "us",
                "category": "inflation",
            })

    # ── Israel Events ─────────────────────────────────────────────────────────

    # Bank of Israel rate decisions
    for ds in _BOI_DATES:
        d = date.fromisoformat(ds)
        if today <= d <= cutoff:
            israel_events.append({
                "date": ds, "time": "16:00 IST",
                "event": "Bank of Israel Rate Decision",
                "event_he": "החלטת ריבית — בנק ישראל",
                "previous": "4.50%", "forecast": "4.25%",
                "impact": "critical", "country": "il",
                "category": "monetary_policy",
            })

    # Israeli CPI
    for ds in _IL_CPI_DATES:
        d = date.fromisoformat(ds)
        if today <= d <= cutoff:
            israel_events.append({
                "date": ds, "time": "18:30 IST",
                "event": "Israeli CPI",
                "event_he": "מדד המחירים לצרכן — ישראל",
                "previous": "3.4% YoY", "forecast": "3.3% YoY",
                "impact": "high", "country": "il",
                "category": "inflation",
            })

    # Sort both lists
    us_events.sort(key=lambda x: x["date"])
    israel_events.sort(key=lambda x: x["date"])

    # Next event overall
    all_sorted = sorted(us_events + israel_events, key=lambda x: x["date"])
    next_event = all_sorted[0] if all_sorted else None

    return {
        "us_events":     us_events,
        "israel_events": israel_events,
        "next_event":    next_event,
        "timestamp":     datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Chart Pattern Recognition — peaks/valleys, Cup&Handle, Double Top/Bottom
# ---------------------------------------------------------------------------

def get_chart_patterns(ticker: str) -> dict:
    """
    Fetch 1 year of daily OHLCV data and perform chart-based pattern recognition.

    Returns:
      - Full chart series (dates, closes, highs, lows, sma44/150/200 lists) for Plotly.
      - Detected patterns: Cup & Handle, Double Top/Bottom, MA Bounce/Breakout,
        Bullish Stack — each with a Hebrew description and price level.
      - Current MA distances (dist_to_44_pct, dist_to_150_pct, dist_to_200_pct).
      - Recent significant peaks and valleys (last 6 of each).
      - A compact pattern_summary list (Hebrew strings) for the agent prompt.
    """
    ticker = ticker.strip().upper()
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 44:
            return {"error": "Insufficient price history for chart analysis."}

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]
        dates  = [d.strftime("%Y-%m-%d") for d in hist.index]

        # ── Compute SMA series ────────────────────────────────────────────────
        sma44_s  = close.rolling(44).mean()
        sma150_s = close.rolling(150).mean()
        sma200_s = close.rolling(200).mean()

        current_price = float(close.iloc[-1])
        sma44  = float(sma44_s.iloc[-1])  if len(close) >= 44  else None
        sma150 = float(sma150_s.iloc[-1]) if len(close) >= 150 else None
        sma200 = float(sma200_s.iloc[-1]) if len(close) >= 200 else None

        dist_to_44  = round((current_price - sma44)  / sma44  * 100, 2) if sma44  else None
        dist_to_150 = round((current_price - sma150) / sma150 * 100, 2) if sma150 else None
        dist_to_200 = round((current_price - sma200) / sma200 * 100, 2) if sma200 else None

        # ── Local extrema detection (rolling ±8 day window) ──────────────────
        WIN = 8
        peaks: list[dict]   = []
        valleys: list[dict] = []

        for i in range(WIN, len(close) - WIN):
            h_slice = high.iloc[i - WIN: i + WIN + 1]
            l_slice = low.iloc[i  - WIN: i + WIN + 1]
            if float(high.iloc[i]) >= float(h_slice.max()):
                peaks.append({
                    "index": i,
                    "price": round(float(high.iloc[i]), 2),
                    "date":  dates[i],
                })
            if float(low.iloc[i]) <= float(l_slice.min()):
                valleys.append({
                    "index": i,
                    "price": round(float(low.iloc[i]), 2),
                    "date":  dates[i],
                })

        # Deduplicate: keep only peaks/valleys that are local maxima/minima
        # (remove consecutive duplicates within 3 days of each other)
        def _dedup(pts: list[dict]) -> list[dict]:
            out: list[dict] = []
            for pt in pts:
                if not out or pt["index"] - out[-1]["index"] > 3:
                    out.append(pt)
                elif pt["price"] > out[-1]["price"] if pt in peaks else pt["price"] < out[-1]["price"]:
                    out[-1] = pt
            return out

        peaks   = _dedup(peaks)[-8:]
        valleys = _dedup(valleys)[-8:]

        # ── Pattern Detection ─────────────────────────────────────────────────
        detected: list[dict] = []

        # 1. Bullish Stack: Price > SMA_44 > SMA_150
        if sma44 and sma150 and current_price > sma44 > sma150:
            detected.append({
                "name":           "Bullish Stack",
                "name_he":        "מחסנית שורית",
                "description_he": (
                    f"מחיר (${current_price:.2f}) > ממוצע 44 (${sma44:.2f}) > "
                    f"ממוצע 150 (${sma150:.2f}) — מבנה מגמה שורי חזק"
                ),
                "type":           "bullish",
                "level":          round(sma44, 2),
                "annotations":    [],
            })

        # 2. Double Top (two recent peaks within 3% of each other)
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(p1["price"] - p2["price"]) / max(p1["price"], 1) < 0.03:
                lvl = round((p1["price"] + p2["price"]) / 2, 2)
                detected.append({
                    "name":           "Double Top",
                    "name_he":        "תקרה כפולה",
                    "description_he": (
                        f"שני שיאים ב-${p1['price']:,.2f} ({p1['date']}) ו-${p2['price']:,.2f} "
                        f"({p2['date']}) — התנגדות מרכזית ב-${lvl:,.2f}"
                    ),
                    "type":           "bearish",
                    "level":          lvl,
                    "annotations": [
                        {"date": p1["date"], "price": p1["price"], "label": "T1"},
                        {"date": p2["date"], "price": p2["price"], "label": "T2"},
                    ],
                })

        # 3. Double Bottom (two recent valleys within 3% of each other)
        if len(valleys) >= 2:
            v1, v2 = valleys[-2], valleys[-1]
            if abs(v1["price"] - v2["price"]) / max(v1["price"], 1) < 0.03:
                lvl = round((v1["price"] + v2["price"]) / 2, 2)
                detected.append({
                    "name":           "Double Bottom",
                    "name_he":        "תחתית כפולה",
                    "description_he": (
                        f"שני שפלים ב-${v1['price']:,.2f} ({v1['date']}) ו-${v2['price']:,.2f} "
                        f"({v2['date']}) — תמיכה מרכזית ב-${lvl:,.2f}"
                    ),
                    "type":           "bullish",
                    "level":          lvl,
                    "annotations": [
                        {"date": v1["date"], "price": v1["price"], "label": "B1"},
                        {"date": v2["date"], "price": v2["price"], "label": "B2"},
                    ],
                })

        # 4. Cup & Handle: 6-month U-shape + 10-day consolidation handle
        if len(hist) >= 130:
            cup   = close.iloc[-130:-10]
            handle = close.iloc[-10:]

            cup_start = float(cup.iloc[0])
            cup_floor = float(cup.min())
            cup_end   = float(cup.iloc[-1])
            depth     = (cup_start - cup_floor) / cup_start if cup_start > 0 else 0
            recovery  = (cup_end - cup_floor) / (cup_start - cup_floor) if (cup_start - cup_floor) > 0 else 0

            handle_tight = float(handle.std()) < float(cup.std()) * 0.65

            if 0.08 <= depth <= 0.45 and recovery >= 0.80 and abs(cup_end - cup_start) / cup_start < 0.06 and handle_tight:
                detected.append({
                    "name":           "Cup and Handle",
                    "name_he":        "ספל עם ידית",
                    "description_he": (
                        f"ספל עמוק {depth*100:.0f}% עם שחזור של {recovery*100:.0f}% — "
                        f"ידית קונסולידציה ב-10 הימים האחרונים. דפוס שורי קלאסי."
                    ),
                    "type":           "bullish",
                    "level":          round(cup_start, 2),
                    "annotations":    [],
                })

        # 5. MA Bounce (price crossed above SMA_44 in last 5 days)
        if sma44 and len(close) >= 49:
            c5, s5 = close.iloc[-5:].values, sma44_s.iloc[-5:].values
            for i in range(1, 5):
                if (not pd.isna(s5[i]) and not pd.isna(s5[i-1]) and
                        float(c5[i]) > float(s5[i]) and float(c5[i-1]) <= float(s5[i-1])):
                    detected.append({
                        "name":           "MA-44 Bounce",
                        "name_he":        "קפיצה מממוצע 44",
                        "description_he": (
                            f"המחיר חצה מעל ממוצע 44 יום (${sma44:.2f}) ב-{dates[len(dates)-5+i]}"
                            f" — אות כניסה פוטנציאלי"
                        ),
                        "type":           "bullish",
                        "level":          round(sma44, 2),
                        "annotations":    [],
                    })
                    break

        # 6. MA-150 Breakout (price crossed above SMA_150 in last 5 days)
        if sma150 and len(close) >= 155:
            c5, s5 = close.iloc[-5:].values, sma150_s.iloc[-5:].values
            for i in range(1, 5):
                if (not pd.isna(s5[i]) and not pd.isna(s5[i-1]) and
                        float(c5[i]) > float(s5[i]) and float(c5[i-1]) <= float(s5[i-1])):
                    detected.append({
                        "name":           "MA-150 Breakout",
                        "name_he":        "פריצת ממוצע 150",
                        "description_he": (
                            f"המחיר פרץ מעל ממוצע 150 יום (${sma150:.2f}) ב-{dates[len(dates)-5+i]}"
                            f" — אות מגמה שורית חזקה"
                        ),
                        "type":           "bullish",
                        "level":          round(sma150, 2),
                        "annotations":    [],
                    })
                    break

        # ── Serialize series for Plotly (NaN → None) ──────────────────────────
        def _lst(s: pd.Series) -> list:
            return [round(float(v), 4) if not pd.isna(v) else None for v in s]

        # ── Build compact agent summary ────────────────────────────────────────
        pattern_summary = [
            f"{p['name_he']}: {p['description_he']}"
            for p in detected
        ] or ["אין דפוסים ברורים שזוהו."]

        return {
            "ticker":            ticker,
            # Chart series
            "dates":             dates,
            "closes":            _lst(close),
            "highs":             _lst(high),
            "lows":              _lst(low),
            "volumes":           [int(v) if not pd.isna(v) else None for v in volume],
            "sma44_series":      _lst(sma44_s),
            "sma150_series":     _lst(sma150_s),
            "sma200_series":     _lst(sma200_s),
            # Current MA snapshot
            "current_price":     round(current_price, 2),
            "sma_44":            round(sma44,  2) if sma44  else None,
            "sma_150":           round(sma150, 2) if sma150 else None,
            "sma_200":           round(sma200, 2) if sma200 else None,
            "dist_to_44_pct":    dist_to_44,
            "dist_to_150_pct":   dist_to_150,
            "dist_to_200_pct":   dist_to_200,
            # Extrema
            "peaks":             peaks[-6:],
            "valleys":           valleys[-6:],
            # Patterns
            "detected_patterns": detected,
            "pattern_count":     len(detected),
            "pattern_summary":   pattern_summary,
        }

    except Exception as exc:
        return {"error": str(exc)}


def get_watchlist_quote(ticker: str) -> dict:
    """
    Lightweight quote for watchlist display: price, daily change %, company name, 5-day sparkline.
    Uses a single history() call + info for company name.
    """
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", auto_adjust=True)
        if hist.empty:
            return {"ticker": ticker, "error": "no data"}

        price = _fmt(hist["Close"].iloc[-1])
        change_pct = None
        if len(hist) >= 2:
            prev = _fmt(hist["Close"].iloc[-2])
            if price is not None and prev and prev != 0:
                change_pct = _fmt((price - prev) / prev * 100)

        sparkline = [round(float(v), 4) for v in hist["Close"].tolist()]

        try:
            info = t.info or {}
            company_name = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            company_name = ticker

        return {
            "ticker":       ticker,
            "company_name": company_name,
            "price":        price,
            "change_pct":   change_pct,
            "sparkline":    sparkline,
        }
    except Exception:
        return {"ticker": ticker, "error": "fetch failed"}


# ════════════════════════════════════════════════════════════════════════════
# HOT SECTORS — sector universe + dynamic scanner
# ════════════════════════════════════════════════════════════════════════════

# Curated stock universe per sector (8-10 liquid names each)
SECTOR_UNIVERSE: dict[str, dict] = {
    "🤖 Tech & AI":        {"etf": "XLK",  "stocks": ["NVDA","MSFT","AAPL","META","GOOGL","AVGO","ORCL","CRM","AMZN","AMD"]},
    "🔬 Semiconductors":   {"etf": "SOXX", "stocks": ["NVDA","AMD","AVGO","QCOM","AMAT","KLAC","LRCX","MU","TXN","ASML"]},
    "🛡️ Defense & Space":  {"etf": "ITA",  "stocks": ["LMT","RTX","NOC","GD","BA","RKLB","ASTS","HII","LDOS","SAIC"]},
    "💊 Healthcare":       {"etf": "XLV",  "stocks": ["LLY","UNH","ABBV","TMO","MRK","JNJ","ISRG","ABT","DHR","SYK"]},
    "🧬 Biotech":          {"etf": "XBI",  "stocks": ["MRNA","REGN","GILD","VRTX","BIIB","ALNY","BMRN","INCY","SRPT","EXAS"]},
    "⚡ Energy":           {"etf": "XLE",  "stocks": ["XOM","CVX","COP","SLB","EOG","OXY","MPC","PSX","VLO","HAL"]},
    "🏦 Financials":       {"etf": "XLF",  "stocks": ["JPM","GS","BAC","MS","WFC","BLK","V","MA","AXP","C"]},
    "🛒 Consumer":         {"etf": "XLY",  "stocks": ["AMZN","TSLA","HD","MCD","COST","NKE","SBUX","LOW","TGT","CMG"]},
    "☁️ Cloud & SaaS":    {"etf": "WCLD", "stocks": ["CRM","NOW","SNOW","DDOG","CRWD","NET","ZS","MDB","BILL","HUBS"]},
    "📡 Communication":    {"etf": "XLC",  "stocks": ["META","GOOGL","NFLX","DIS","SPOT","TMUS","CMCSA","T","VZ","CHTR"]},
    "🏭 Industrials":      {"etf": "XLI",  "stocks": ["GE","CAT","HON","DE","UNP","FDX","UPS","EMR","ITW","MMM"]},
    "🔋 Clean Energy":     {"etf": "ICLN", "stocks": ["ENPH","FSLR","NEE","PLUG","BE","RUN","SEDG","ARRY","CWEN","AES"]},
}

# Hardcoded short names to avoid slow .info calls per-ticker
_TICKER_NAMES: dict[str, str] = {
    "NVDA":"NVIDIA","MSFT":"Microsoft","AAPL":"Apple","META":"Meta","GOOGL":"Alphabet",
    "AVGO":"Broadcom","ORCL":"Oracle","CRM":"Salesforce","AMZN":"Amazon","AMD":"AMD",
    "QCOM":"Qualcomm","AMAT":"Applied Materials","KLAC":"KLA Corp","LRCX":"Lam Research",
    "MU":"Micron","TXN":"Texas Instruments","ASML":"ASML",
    "LMT":"Lockheed Martin","RTX":"RTX Corp","NOC":"Northrop Grumman","GD":"General Dynamics",
    "BA":"Boeing","RKLB":"Rocket Lab","ASTS":"AST SpaceMobile","HII":"HII","LDOS":"Leidos","SAIC":"SAIC",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","ABBV":"AbbVie","TMO":"Thermo Fisher",
    "MRK":"Merck","JNJ":"J&J","ISRG":"Intuitive Surgical","ABT":"Abbott","DHR":"Danaher","SYK":"Stryker",
    "MRNA":"Moderna","REGN":"Regeneron","GILD":"Gilead","VRTX":"Vertex","BIIB":"Biogen",
    "ALNY":"Alnylam","BMRN":"BioMarin","INCY":"Incyte","SRPT":"Sarepta","EXAS":"Exact Sciences",
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","SLB":"SLB","EOG":"EOG Resources",
    "OXY":"Occidental","MPC":"Marathon Petroleum","PSX":"Phillips 66","VLO":"Valero","HAL":"Halliburton",
    "JPM":"JPMorgan","GS":"Goldman Sachs","BAC":"Bank of America","MS":"Morgan Stanley",
    "WFC":"Wells Fargo","BLK":"BlackRock","V":"Visa","MA":"Mastercard","AXP":"Amex","C":"Citigroup",
    "TSLA":"Tesla","HD":"Home Depot","MCD":"McDonald's","COST":"Costco","NKE":"Nike",
    "SBUX":"Starbucks","LOW":"Lowe's","TGT":"Target","CMG":"Chipotle",
    "NOW":"ServiceNow","SNOW":"Snowflake","DDOG":"Datadog","CRWD":"CrowdStrike",
    "NET":"Cloudflare","ZS":"Zscaler","MDB":"MongoDB","BILL":"Bill.com","HUBS":"HubSpot",
    "NFLX":"Netflix","DIS":"Disney","SPOT":"Spotify","TMUS":"T-Mobile",
    "CMCSA":"Comcast","T":"AT&T","VZ":"Verizon","CHTR":"Charter",
    "GE":"GE Aerospace","CAT":"Caterpillar","HON":"Honeywell","DE":"John Deere",
    "UNP":"Union Pacific","FDX":"FedEx","UPS":"UPS","EMR":"Emerson","ITW":"Illinois Tool","MMM":"3M",
    "ENPH":"Enphase","FSLR":"First Solar","NEE":"NextEra","PLUG":"Plug Power",
    "BE":"Bloom Energy","RUN":"Sunrun","SEDG":"SolarEdge","ARRY":"Array Technologies",
    "CWEN":"Clearway Energy","AES":"AES Corp",
}


# ---------------------------------------------------------------------------
# Tool: get_technical_indicators
# ---------------------------------------------------------------------------

def get_technical_indicators(ticker: str) -> dict:
    """
    Advanced technical indicators beyond basic RSI/SMA:
      - MACD (line, signal, histogram) — trend momentum
      - SMA-20 — short-term trend
      - Volume analysis: up-day vs down-day volume ratio (accumulation/distribution)
      - Weinstein Stage (1/2/3/4) based on SMA slopes and price position
      - RSI divergence detection (bullish/bearish)
    """
    ticker = ticker.strip().upper()
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return {"error": "Insufficient price history for technical indicators."}

        close  = hist["Close"]
        volume = hist["Volume"]

        # ── SMA-20 ────────────────────────────────────────────────────────────
        sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None

        # ── MACD (12/26/9) ───────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line

        macd_val    = round(float(macd_line.iloc[-1]),   4)
        signal_val  = round(float(signal_line.iloc[-1]), 4)
        hist_val    = round(float(histogram.iloc[-1]),   4)
        hist_prev   = round(float(histogram.iloc[-2]),   4) if len(histogram) >= 2 else None

        macd_cross = "none"
        if len(macd_line) >= 2:
            if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                macd_cross = "bullish_cross"
            elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                macd_cross = "bearish_cross"
            elif macd_val > signal_val:
                macd_cross = "bullish"
            else:
                macd_cross = "bearish"

        histogram_expanding = None
        if hist_prev is not None:
            histogram_expanding = abs(hist_val) > abs(hist_prev)

        # ── Volume Analysis — accumulation vs distribution ────────────────────
        daily_ret = close.pct_change().dropna()
        vol_aligned = volume.reindex(daily_ret.index)

        up_days   = daily_ret[daily_ret > 0]
        down_days = daily_ret[daily_ret < 0]
        up_vol    = vol_aligned[up_days.index].mean()
        down_vol  = vol_aligned[down_days.index].mean()

        up_vol_avg   = round(float(up_vol),   0)   if not np.isnan(up_vol)   else None
        down_vol_avg = round(float(down_vol), 0)   if not np.isnan(down_vol) else None
        vol_ratio_ud = round(up_vol / down_vol, 2) if (up_vol and down_vol and down_vol > 0) else None

        if vol_ratio_ud:
            if vol_ratio_ud > 1.2:
                vol_signal = "accumulation"     # more vol on up days → institutions buying
            elif vol_ratio_ud < 0.8:
                vol_signal = "distribution"     # more vol on down days → institutions selling
            else:
                vol_signal = "neutral"
        else:
            vol_signal = "unknown"

        # ── Weinstein Stage ───────────────────────────────────────────────────
        stage = None
        stage_desc = None
        try:
            sma150 = float(close.rolling(150).mean().iloc[-1]) if len(close) >= 150 else None
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            price  = float(close.iloc[-1])

            # Slope: compare SMA-150 now vs 4 weeks ago
            sma150_series = close.rolling(150).mean()
            sma200_series = close.rolling(200).mean()
            sma150_slope = None
            sma200_slope = None
            if len(sma150_series.dropna()) >= 20:
                sma150_slope = float(sma150_series.iloc[-1]) - float(sma150_series.iloc[-20])
            if len(sma200_series.dropna()) >= 20:
                sma200_slope = float(sma200_series.iloc[-1]) - float(sma200_series.iloc[-20])

            if sma150 and sma200:
                if price > sma150 > sma200 and sma150_slope and sma150_slope > 0:
                    stage = 2
                    stage_desc = "Stage 2 Uptrend — strong bull trend, price above rising SMAs"
                elif price < sma150 and price < sma200:
                    if sma150_slope and sma150_slope < 0:
                        stage = 4
                        stage_desc = "Stage 4 Decline — bearish trend, price below falling SMAs"
                    else:
                        stage = 1
                        stage_desc = "Stage 1 Base — consolidation, SMAs flattening out"
                elif price > sma150 and (sma150_slope and sma150_slope < 0):
                    stage = 3
                    stage_desc = "Stage 3 Top — price still high but SMAs starting to roll over"
                else:
                    stage = 1
                    stage_desc = "Stage 1 Base — no clear trend direction"
        except Exception:
            pass

        # ── RSI Divergence ────────────────────────────────────────────────────
        divergence = "none"
        try:
            # Use last 20 bars for divergence detection
            window = 20
            if len(close) >= window + 14:
                rsi_series = []
                for i in range(len(close) - window, len(close)):
                    rsi_series.append(_calculate_rsi(close.iloc[max(0, i-14):i+1]))
                rsi_series = [r for r in rsi_series if r is not None]

                if len(rsi_series) >= window:
                    prices_w = close.iloc[-window:].values
                    rsi_w    = rsi_series[-window:]

                    price_hl = prices_w[-1] > prices_w[0]   # price making higher high
                    rsi_hl   = rsi_w[-1]    > rsi_w[0]      # RSI making higher high

                    if price_hl and not rsi_hl:
                        divergence = "bearish"   # price up, RSI failing → weakness
                    elif not price_hl and rsi_hl:
                        divergence = "bullish"   # price down, RSI holding → strength
        except Exception:
            pass

        current_price = round(float(close.iloc[-1]), 2)

        return {
            "ticker":               ticker,
            "current_price":        current_price,
            "sma_20":               sma20,
            "price_vs_sma20_pct":   round((current_price - sma20) / sma20 * 100, 2) if sma20 else None,
            "macd": {
                "macd_line":          macd_val,
                "signal_line":        signal_val,
                "histogram":          hist_val,
                "histogram_prev":     hist_prev,
                "signal":             macd_cross,
                "histogram_expanding": histogram_expanding,
            },
            "volume_analysis": {
                "avg_up_day_volume":   up_vol_avg,
                "avg_down_day_volume": down_vol_avg,
                "up_down_vol_ratio":   vol_ratio_ud,
                "signal":              vol_signal,
            },
            "weinstein_stage": {
                "stage":       stage,
                "description": stage_desc,
            },
            "rsi_divergence": divergence,
        }

    except Exception as exc:
        return {"error": f"get_technical_indicators failed for '{ticker}': {exc}"}


# ---------------------------------------------------------------------------
# Tool: get_options_flow
# ---------------------------------------------------------------------------

def get_options_flow(ticker: str) -> dict:
    """
    Options market snapshot from yfinance:
      - Put/Call ratio (open interest-based)
      - Average implied volatility (near-the-money options)
      - Max Pain level (price at which option buyers lose the most)
      - Nearest expiration summary
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": "No options data available for this ticker."}

        # Use the nearest expiration with meaningful data
        chain = None
        used_exp = None
        for exp in expirations[:4]:
            try:
                c = stock.option_chain(exp)
                if len(c.calls) > 5 and len(c.puts) > 5:
                    chain = c
                    used_exp = exp
                    break
            except Exception:
                continue

        if chain is None:
            return {"error": "Could not retrieve a valid options chain."}

        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # ── Put/Call ratio ────────────────────────────────────────────────────
        total_call_oi = calls["openInterest"].fillna(0).sum()
        total_put_oi  = puts["openInterest"].fillna(0).sum()
        put_call_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

        if put_call_ratio:
            if put_call_ratio > 1.0:
                pc_signal = "bearish_sentiment"
            elif put_call_ratio < 0.7:
                pc_signal = "bullish_sentiment"
            else:
                pc_signal = "neutral"
        else:
            pc_signal = "unknown"

        # ── Implied Volatility (near-the-money) ───────────────────────────────
        current_price = float(stock.history(period="2d")["Close"].iloc[-1])
        atm_range = current_price * 0.05  # within 5% of current price

        atm_calls = calls[abs(calls["strike"] - current_price) <= atm_range]
        atm_puts  = puts[abs(puts["strike"]  - current_price) <= atm_range]
        atm_all   = pd.concat([atm_calls, atm_puts])

        avg_iv = None
        if not atm_all.empty and "impliedVolatility" in atm_all.columns:
            iv_vals = atm_all["impliedVolatility"].dropna()
            if len(iv_vals) > 0:
                avg_iv = round(float(iv_vals.mean()) * 100, 2)  # as %

        # ── Max Pain ──────────────────────────────────────────────────────────
        max_pain = None
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            pain_at = {}
            for s in strikes:
                call_pain = calls[calls["strike"] < s].apply(
                    lambda row: (s - row["strike"]) * row["openInterest"], axis=1
                ).sum()
                put_pain = puts[puts["strike"] > s].apply(
                    lambda row: (row["strike"] - s) * row["openInterest"], axis=1
                ).sum()
                pain_at[s] = call_pain + put_pain
            if pain_at:
                max_pain = min(pain_at, key=pain_at.get)
        except Exception:
            pass

        # ── Top OI strikes ────────────────────────────────────────────────────
        top_call_strikes = (
            calls.nlargest(3, "openInterest")[["strike", "openInterest"]]
            .to_dict("records") if not calls.empty else []
        )
        top_put_strikes = (
            puts.nlargest(3, "openInterest")[["strike", "openInterest"]]
            .to_dict("records") if not puts.empty else []
        )

        return {
            "ticker":            ticker,
            "expiration_used":   used_exp,
            "current_price":     round(current_price, 2),
            "put_call_ratio":    put_call_ratio,
            "pc_signal":         pc_signal,
            "avg_implied_vol_pct": avg_iv,
            "max_pain":          round(float(max_pain), 2) if max_pain else None,
            "max_pain_vs_price_pct": round((float(max_pain) - current_price) / current_price * 100, 2)
                                     if max_pain else None,
            "top_call_oi_strikes": top_call_strikes,
            "top_put_oi_strikes":  top_put_strikes,
            "total_call_oi":     int(total_call_oi),
            "total_put_oi":      int(total_put_oi),
        }

    except Exception as exc:
        return {"error": f"get_options_flow failed for '{ticker}': {exc}"}


# ---------------------------------------------------------------------------
# Tool: get_insider_transactions
# ---------------------------------------------------------------------------

def get_insider_transactions(ticker: str) -> dict:
    """
    Fetch recent insider transactions (SEC Form 4) via yfinance:
      - Last 10 transactions: insider name, title, transaction type, shares, value
      - 6-month summary: net shares bought/sold, unique insiders, dominant action
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        trans = stock.insider_transactions

        if trans is None or (hasattr(trans, "empty") and trans.empty):
            return {"error": "No insider transaction data available."}

        df = trans.copy()

        # Normalise column names (yfinance changes them occasionally)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Keep last 6 months
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=180)
        if "start_date" in df.columns:
            date_col = "start_date"
        elif "date" in df.columns:
            date_col = "date"
        else:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        recent = df[df[date_col] >= cutoff].copy()

        if recent.empty:
            recent = df.head(10)  # fallback: just return most recent regardless of date

        # Shares column
        shares_col = next((c for c in df.columns if "share" in c), None)
        value_col  = next((c for c in df.columns if "value" in c), None)
        text_col   = next((c for c in df.columns if "text" in c or "transaction" in c), None)
        insider_col = next((c for c in df.columns if "insider" in c or "name" in c), None)
        title_col   = next((c for c in df.columns if "title" in c or "position" in c), None)

        # Build last 10 transactions list
        transactions = []
        for _, row in recent.head(10).iterrows():
            tx = {
                "date":    str(row.get(date_col, ""))[:10],
                "insider": str(row.get(insider_col, "N/A")) if insider_col else "N/A",
                "title":   str(row.get(title_col,  "N/A")) if title_col  else "N/A",
                "type":    str(row.get(text_col,   "N/A")) if text_col   else "N/A",
                "shares":  int(row.get(shares_col, 0))     if shares_col else None,
                "value":   round(float(row.get(value_col, 0)), 0) if value_col else None,
            }
            transactions.append(tx)

        # 6-month summary
        buys  = recent[recent[text_col].str.contains("Purchase|Buy|Acquire", case=False, na=False)] if text_col else pd.DataFrame()
        sells = recent[recent[text_col].str.contains("Sale|Sell|Dispose",   case=False, na=False)] if text_col else pd.DataFrame()

        net_shares_bought = int(buys[shares_col].fillna(0).sum())  if (not buys.empty  and shares_col) else 0
        net_shares_sold   = int(sells[shares_col].fillna(0).sum()) if (not sells.empty and shares_col) else 0
        net_shares        = net_shares_bought - net_shares_sold

        dominant = "net_buying" if net_shares > 0 else ("net_selling" if net_shares < 0 else "neutral")

        unique_insiders = len(recent[insider_col].unique()) if insider_col else None

        return {
            "ticker":             ticker,
            "period":             "last_6_months",
            "transactions":       transactions,
            "summary": {
                "net_shares_bought": net_shares_bought,
                "net_shares_sold":   net_shares_sold,
                "net_shares":        net_shares,
                "dominant_action":   dominant,
                "unique_insiders":   unique_insiders,
                "buy_transactions":  len(buys),
                "sell_transactions": len(sells),
            },
        }

    except Exception as exc:
        return {"error": f"get_insider_transactions failed for '{ticker}': {exc}"}


# ---------------------------------------------------------------------------
# Tool: get_balance_sheet_deep
# ---------------------------------------------------------------------------

def get_balance_sheet_deep(ticker: str) -> dict:
    """
    Deep balance sheet and cash flow analysis:
      - EBITDA, Free Cash Flow (OCF - CapEx)
      - Cash & equivalents, total debt, net debt
      - Cash runway (quarters at current burn rate)
      - Interest coverage ratio (EBIT / interest expense)
      - EV/EBITDA, P/S ratio
      - Gross margin, operating margin trends (last 4 quarters)
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # ── Annual financials ─────────────────────────────────────────────────
        try:
            income_annual  = stock.income_stmt
            cashflow_annual = stock.cashflow
            balance_annual  = stock.balance_sheet
        except Exception:
            income_annual = cashflow_annual = balance_annual = None

        def _safe_val(df, row_key, col_idx=0):
            """Safely extract a value from a financial DataFrame."""
            if df is None or df.empty:
                return None
            matching = [r for r in df.index if row_key.lower() in str(r).lower()]
            if not matching:
                return None
            try:
                val = df.loc[matching[0]].iloc[col_idx]
                return None if pd.isna(val) else float(val)
            except Exception:
                return None

        # ── EBITDA ────────────────────────────────────────────────────────────
        ebitda = info.get("ebitda") or _safe_val(income_annual, "EBITDA")

        # Manual EBITDA: EBIT + D&A
        if not ebitda:
            ebit  = _safe_val(income_annual, "EBIT") or _safe_val(income_annual, "Operating Income")
            da    = _safe_val(cashflow_annual, "Depreciation")
            ebitda = (ebit + da) if (ebit and da) else None

        # ── Free Cash Flow ────────────────────────────────────────────────────
        fcf = info.get("freeCashflow")
        if not fcf:
            ocf   = _safe_val(cashflow_annual, "Operating Cash Flow") or _safe_val(cashflow_annual, "Total Cash From Operating Activities")
            capex = _safe_val(cashflow_annual, "Capital Expenditure") or _safe_val(cashflow_annual, "Capital Expenditures")
            if capex and capex > 0:
                capex = -capex  # capex is usually negative in yfinance
            fcf = (ocf + capex) if (ocf is not None and capex is not None) else None

        # ── Balance Sheet items ───────────────────────────────────────────────
        cash_raw       = info.get("totalCash") or _safe_val(balance_annual, "Cash And Cash Equivalents")
        total_debt_raw = info.get("totalDebt") or _safe_val(balance_annual, "Total Debt")
        net_debt       = (total_debt_raw - cash_raw) if (total_debt_raw and cash_raw) else None

        # ── Cash runway ───────────────────────────────────────────────────────
        runway_quarters = None
        if cash_raw and fcf and fcf < 0:
            quarterly_burn = abs(fcf) / 4
            if quarterly_burn > 0:
                runway_quarters = round(cash_raw / quarterly_burn, 1)

        # ── Interest Coverage ─────────────────────────────────────────────────
        interest_coverage = None
        ebit = _safe_val(income_annual, "EBIT") or _safe_val(income_annual, "Operating Income")
        interest_exp = _safe_val(income_annual, "Interest Expense")
        if ebit and interest_exp and interest_exp != 0:
            interest_coverage = round(abs(ebit / interest_exp), 2)

        # ── Enterprise Value & Ratios ─────────────────────────────────────────
        market_cap = info.get("marketCap")
        ev         = info.get("enterpriseValue")
        if not ev and market_cap and total_debt_raw and cash_raw:
            ev = market_cap + total_debt_raw - cash_raw

        ev_ebitda = None
        if ev and ebitda and ebitda > 0:
            ev_ebitda = round(ev / ebitda, 2)

        revenue = info.get("totalRevenue") or _safe_val(income_annual, "Total Revenue")
        ps_ratio = None
        if market_cap and revenue and revenue > 0:
            ps_ratio = round(market_cap / revenue, 2)

        # ── Quarterly margin trends ───────────────────────────────────────────
        margin_trend = []
        try:
            q_income = stock.quarterly_income_stmt
            if q_income is not None and not q_income.empty:
                rev_rows  = [r for r in q_income.index if "revenue" in str(r).lower() and "total" in str(r).lower()]
                gp_rows   = [r for r in q_income.index if "gross" in str(r).lower()]
                op_rows   = [r for r in q_income.index if "operating" in str(r).lower() and "income" in str(r).lower()]
                ni_rows   = [r for r in q_income.index if "net income" in str(r).lower()]

                for col in q_income.columns[:4]:
                    rev  = float(q_income.loc[rev_rows[0], col])  if rev_rows  else None
                    gp   = float(q_income.loc[gp_rows[0],  col])  if gp_rows   else None
                    op   = float(q_income.loc[op_rows[0],  col])  if op_rows   else None
                    ni   = float(q_income.loc[ni_rows[0],  col])  if ni_rows   else None
                    entry = {
                        "quarter":        str(col)[:10],
                        "gross_margin":   round(gp / rev * 100, 1) if (gp and rev and rev > 0) else None,
                        "op_margin":      round(op / rev * 100, 1) if (op and rev and rev > 0) else None,
                        "net_margin":     round(ni / rev * 100, 1) if (ni and rev and rev > 0) else None,
                    }
                    margin_trend.append(entry)
        except Exception:
            pass

        def _fmt_b(val):
            """Format large numbers as $XB or $XM."""
            if val is None:
                return None
            if abs(val) >= 1e9:
                return f"${val/1e9:.2f}B"
            if abs(val) >= 1e6:
                return f"${val/1e6:.1f}M"
            return f"${val:,.0f}"

        return {
            "ticker":           ticker,
            "ebitda":           _fmt_b(ebitda),
            "ebitda_raw":       ebitda,
            "free_cash_flow":   _fmt_b(fcf),
            "fcf_raw":          fcf,
            "cash":             _fmt_b(cash_raw),
            "total_debt":       _fmt_b(total_debt_raw),
            "net_debt":         _fmt_b(net_debt),
            "runway_quarters":  runway_quarters,
            "interest_coverage": interest_coverage,
            "ev_ebitda":        ev_ebitda,
            "ps_ratio":         ps_ratio,
            "revenue":          _fmt_b(revenue),
            "market_cap":       _fmt_b(market_cap),
            "is_self_funding":  fcf > 0 if fcf is not None else None,
            "margin_trend":     margin_trend,
        }

    except Exception as exc:
        return {"error": f"get_balance_sheet_deep failed for '{ticker}': {exc}"}


def get_hot_sectors(top_n_sectors: int = 3, stocks_per_sector: int = 3) -> list[dict]:
    """
    Identify top-performing sectors vs SPY and return their best stocks.

    Steps:
      1. Batch-download 3-month closes for all ETFs + stocks in one call.
      2. Rank sector ETFs by 50-day return vs SPY.
      3. Within each top sector, rank stocks by 50-day return vs their sector ETF.
      4. Return top_n_sectors × stocks_per_sector entries with price/RSI/levels.

    Returns list of sector dicts:
      { sector, etf, etf_return_50d, vs_spy_50d, top_stocks: [...] }
    """
    # ── Collect all tickers ──────────────────────────────────────────────────
    all_etfs    = ["SPY"] + [v["etf"] for v in SECTOR_UNIVERSE.values()]
    all_stocks  = list({s for v in SECTOR_UNIVERSE.values() for s in v["stocks"]})
    all_tickers = list(set(all_etfs + all_stocks))

    # ── Batch download ───────────────────────────────────────────────────────
    try:
        raw = yf.download(all_tickers, period="3mo", auto_adjust=True,
                          progress=False, group_by="ticker")
    except Exception:
        return []

    # Normalise to a flat DataFrame of closing prices keyed by ticker
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            closes = pd.DataFrame({
                t: raw[t]["Close"] for t in all_tickers
                if t in raw and "Close" in raw[t].columns
            })
        else:
            closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
    except Exception:
        return []

    if closes.empty:
        return []

    # ── Helper: 50-day return ────────────────────────────────────────────────
    def _ret50(ticker: str) -> Optional[float]:
        if ticker not in closes.columns:
            return None
        s = closes[ticker].dropna()
        if len(s) < 52:
            return None
        return float((s.iloc[-1] - s.iloc[-52]) / s.iloc[-52] * 100)

    spy_ret = _ret50("SPY")
    if spy_ret is None:
        return []

    # ── Rank sectors ────────────────────────────────────────────────────────
    sector_ranks: list[dict] = []
    for sector_name, sector_data in SECTOR_UNIVERSE.items():
        etf_ret = _ret50(sector_data["etf"])
        if etf_ret is None:
            continue
        sector_ranks.append({
            "sector":         sector_name,
            "etf":            sector_data["etf"],
            "etf_return_50d": round(etf_ret, 2),
            "vs_spy_50d":     round(etf_ret - spy_ret, 2),
            "stocks":         sector_data["stocks"],
        })

    top_sectors = sorted(sector_ranks, key=lambda x: x["vs_spy_50d"], reverse=True)[:top_n_sectors]

    # ── For each top sector, rank stocks ─────────────────────────────────────
    result: list[dict] = []
    for sector in top_sectors:
        etf_ret = _ret50(sector["etf"]) or 0.0
        scored: list[dict] = []

        for ticker in sector["stocks"]:
            if ticker not in closes.columns:
                continue
            s = closes[ticker].dropna()
            if len(s) < 22:
                continue

            stock_ret = _ret50(ticker)
            if stock_ret is None:
                continue

            price      = round(float(s.iloc[-1]), 2)
            change_pct = round(float((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100), 2)
            rsi        = _calculate_rsi(s)
            recent20   = s.iloc[-21:]
            support    = round(float(recent20.min()), 2)
            resistance = round(float(recent20.max()), 2)

            scored.append({
                "ticker":       ticker,
                "company_name": _TICKER_NAMES.get(ticker, ticker),
                "price":        price,
                "change_pct":   change_pct,
                "rsi_14":       round(rsi, 1) if rsi else None,
                "rs_vs_etf":    round(stock_ret - etf_ret, 2),
                "return_50d":   round(stock_ret, 2),
                "support":      support,
                "resistance":   resistance,
            })

        scored.sort(key=lambda x: x["rs_vs_etf"], reverse=True)
        result.append({
            "sector":         sector["sector"],
            "etf":            sector["etf"],
            "etf_return_50d": sector["etf_return_50d"],
            "vs_spy_50d":     sector["vs_spy_50d"],
            "top_stocks":     scored[:stocks_per_sector],
        })

    return result
