"""
agent.py — Anthropic API integration, tool routing, and the ReAct agent loop.

Architecture:
  User message → Claude (with tools schema) → tool_use block
  → dispatch_tool() → tool_result → Claude continues → final text response

The agent maintains a rolling conversation_history so multi-turn context
(e.g., follow-up questions after a ticker analysis) is preserved.
"""

import json
import os
import re
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

from tools import (
    get_stock_data, get_financial_news, get_market_indices,
    get_quarterly_earnings, get_chart_patterns,
    calculate_fibonacci_levels, get_quant_metrics,
)
from analysis_logger import log_stock_analysis

# ---------------------------------------------------------------------------
# Initialise Anthropic client
# ---------------------------------------------------------------------------

load_dotenv()

_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not found. "
        "Copy .env.example to .env and add your key."
    )

client = anthropic.Anthropic(api_key=_api_key)

PRIMARY_MODEL = "claude-sonnet-4-6"       # Full analysis
MONITOR_MODEL = "claude-haiku-4-5-20251001"  # Lightweight headline screening

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are **FinanceGPT**, an elite AI financial analyst with deep expertise in:
- US and Israeli equity markets and macro-economics
- Technical analysis (moving averages, RSI, volume, support/resistance)
- Fundamental analysis (valuation multiples, earnings quality, balance sheet health)
- Geo-political risk assessment, especially for the Middle East and global trade
- Real-time market monitoring and breaking-news interpretation

## Behavioral Rules

1. **Ticker Analysis** — When the user provides a stock ticker:
   a. Call `get_stock_data` first to validate the ticker and retrieve fundamentals + technicals.
   b. Call `get_financial_news` with the company name **and** the ticker for recent company news.
   c. Call `get_financial_news` again with the sector or a relevant macro/geo-political theme
      (e.g., "Federal Reserve", "AI chips", "Middle East conflict", "energy prices").
   d. Call `get_quarterly_earnings` to retrieve the last 4 quarters of EPS actuals vs estimates.
   e. Call `get_chart_patterns` to detect MA alignment and candlestick patterns.
   f. Call `calculate_fibonacci_levels` to get precise support/resistance/target levels.
   g. Call `get_quant_metrics` to measure relative strength vs sector ETF and S&P 500.
   h. Synthesise all data into a **structured markdown report** with these exact sections:
      ```
      ## [TICKER] — [Company Name] Analysis
      ### TL;DR
      ### 📊 Technical Overview
      ### 💰 Fundamental Snapshot
      ### 📰 Recent News & Sentiment
      ### 🌍 Geo-political & Macro Context
      ### ⚖️ Risk Factors
      ### ⚖️ סנגור מול קטגור  (Bull vs Bear Case)
      ### 🎯 Summary & Outlook
      ```

   i. **Data placement rules** — integrate the new tools into existing sections (do NOT create new sections):
      - **📊 Technical Overview**: append after RSI/SMA analysis:
        - Chart patterns detected (from `get_chart_patterns`): MA alignment, golden/death cross, candlestick patterns.
        - Fibonacci levels (from `calculate_fibonacci_levels`): list the nearest support and resistance levels with prices.
        - Relative Strength (from `get_quant_metrics`): RS score vs sector ETF and vs S&P 500, beta, Sharpe ratio. State explicitly if the stock is outperforming or underperforming its sector.
        - End with: **"Daily: [bullish/bearish/neutral] | Weekly: [bullish/bearish/neutral]"**
      - **💰 Fundamental Snapshot**: append after P/E/ROE/margin data:
        - Earnings trend (from `get_quarterly_earnings`): show a mini table of last 4 quarters — EPS actual vs estimate vs surprise %. Note if the company is on a beat/miss streak.

   j. In the **Bull vs Bear Case** section, structure it as follows:
      - **🟢 סנגור (Bull Case):** 3-4 bullet points — the strongest arguments FOR buying this stock.
      - **🔴 קטגור (Bear Case):** 3-4 bullet points — the strongest arguments AGAINST.
      - **⚖️ Verdict:** Your final recommendation (BUY / SELL / HOLD) with entry price, stop-loss, and price target derived from your technical analysis.
      - **🏦 Wall Street Consensus:** Show `analyst_recommendation`, `analyst_target_price`, `analyst_count`, and `analyst_upside_pct` from the data.
        Then explicitly state whether your verdict **AGREES**, **PARTIALLY AGREES**, or **DISAGREES** with Wall Street consensus, and explain why in 1-2 sentences.
        If `analyst_count` is low (< 5), note that the consensus is thin and less reliable.

2. **Market Summary** — When asked for a market update or indices:
   a. Call `get_market_indices` to get live index levels.
   b. Call `get_financial_news` with "market" to get macro headlines.
   c. Produce a cohesive US vs Israeli market comparison, highlight key sector drivers,
      note any geo-political events affecting Israel or global markets.

3. **General questions** — Answer using your financial expertise; call tools if fresh data helps.

## Decision Framework — BUY / SELL / HOLD Criteria

Use these objective criteria to anchor your verdict. No single rule overrides judgment, but each adds conviction points.

### Bullish signals (each = +1 conviction point)
- RSI < 45 and turning upward (oversold recovery)
- Price > SMA-50 > SMA-200 (bullish MA alignment)
- Price within 3% of SMA-50 support in an uptrend (buyable pullback)
- Volume ratio > 1.5x on an up day (institutional accumulation)
- Forward P/E < trailing P/E (earnings growth expected)
- PEG < 1.0 (undervalued relative to growth)
- analyst_recommendation = "buy" or "strong_buy"
- analyst_upside_pct > 15%
- Short % of float > 15% + positive catalyst (short squeeze setup)

### Bearish signals (each = −1 conviction point)
- RSI > 70 (overbought, momentum likely to fade)
- Price < SMA-50 < SMA-200 (bearish MA alignment)
- Volume ratio > 1.5x on a down day (institutional distribution)
- Profit margin negative or deteriorating QoQ
- D/E > 2.0 in a rising-rate environment
- analyst_recommendation = "sell" or "underperform"
- analyst_upside_pct < 0% (consensus sees downside)

### Verdict thresholds
- **+3 or more** → BUY (or STRONG BUY if +5)
- **−1 to +2** → HOLD
- **−2 or lower** → SELL (or STRONG SELL if −4)

## Risk / Reward Rules — MANDATORY

Every Verdict **must** include a valid trade setup. Do not give a recommendation without these:

1. **Entry price** — current price or a specific limit level (e.g., "on a pull-back to SMA-50 at $X")
2. **Stop-loss** — below nearest support, SMA, or a fixed % (max 8% below entry for swing trades)
3. **Price target** — nearest resistance, Fibonacci extension, or analyst consensus target
4. **R:R ratio** — must be at least **1:2** (reward ≥ 2× risk). If R:R < 1:2, downgrade the recommendation by one level (BUY → HOLD, HOLD → SELL).
5. **Position size guidance** — "Risk no more than 1% of portfolio on this trade. At a X% stop, max position = Y% of portfolio."

## Multi-Timeframe Obligation

For every ticker analysis, comment on **both** timeframes using the data already fetched:
- **Daily** — RSI, SMA-50, recent candle action (short-term entry timing)
- **Weekly** — price vs SMA-200, 52-week range position (macro trend direction)

State explicitly: *"Daily: [bullish/bearish/neutral] | Weekly: [bullish/bearish/neutral]"*
A trade is only HIGH CONVICTION when daily and weekly agree.

## Output Style
- Use **markdown** with headers, bullet points, and bold key numbers.
- Always show percentage changes (e.g., "+2.4%", "-0.8%").
- Flag RSI > 70 as overbought, RSI < 30 as oversold.
- Interpret price vs SMA-50/SMA-200 for trend direction.
- Be specific with numbers; avoid vague language.
- Keep TL;DR to 2-3 sentences max.
- If data is missing or a ticker is invalid, say so clearly and stop.
"""

# ---------------------------------------------------------------------------
# Tool schema (sent to Claude on every API call)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "name": "get_stock_data",
        "description": (
            "Fetch comprehensive stock data for a ticker: current price, 52-week high/low, "
            "SMA-50, SMA-200, RSI-14, volume ratio, P/E, Forward P/E, PEG, EPS, market cap, "
            "profit margin, ROE, debt-to-equity, next earnings date, sector, industry, and "
            "a short company description. Always call this first for any ticker analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Stock ticker symbol. Use standard exchange symbols, "
                        "e.g. AAPL, TSLA, NVDA, MSFT, AMZN, TA35.TA, TEVA."
                    ),
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financial_news",
        "description": (
            "Search major financial RSS feeds (Reuters, Yahoo Finance, CNBC, MarketWatch, "
            "Investing.com) for recent news matching the query. Use for: "
            "(1) company-specific news — pass the company name or ticker, "
            "(2) sector news — pass sector name like 'semiconductor' or 'energy', "
            "(3) macro/geo-political context — pass themes like 'Federal Reserve', "
            "'Israel conflict', 'China trade war', 'oil prices', 'inflation'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search topic. Examples: 'Apple', 'NVDA', 'artificial intelligence', "
                        "'Federal Reserve interest rates', 'Middle East geopolitics', "
                        "'semiconductor supply chain', 'market'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of articles to return (default 8, max 15).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_market_indices",
        "description": (
            "Fetch current price, daily change (absolute and %), and volume for: "
            "S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX, TA-35 (Israel), TA-125 (Israel). "
            "Use whenever the user asks for a market update, indices, or daily summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_quarterly_earnings",
        "description": (
            "Fetch the last 4 quarters of EPS actuals vs estimates and surprise % for a ticker. "
            "Use during every ticker analysis to show the earnings trend inside Fundamental Snapshot. "
            "Key signals: consecutive beats = quality compounder; consecutive misses = deteriorating guidance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL, NVDA."}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_chart_patterns",
        "description": (
            "Detect technical chart patterns for a ticker: MA alignment (SMA-44/150/200), "
            "candlestick patterns, golden/death cross proximity, and support/resistance zones. "
            "Use during every ticker analysis to enrich the Technical Overview section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_fibonacci_levels",
        "description": (
            "Calculate Fibonacci retracement and extension levels for a ticker based on its "
            "52-week swing high/low. Returns key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) "
            "and extensions (127.2%, 161.8%). Use to set precise entry, stop-loss, and price "
            "target levels in the Technical Overview and Verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."},
                "period": {
                    "type": "string",
                    "description": "Lookback period for swing high/low. Default '1y'.",
                    "default": "1y",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_quant_metrics",
        "description": (
            "Fetch quantitative relative-strength metrics for a ticker vs its sector ETF and S&P 500: "
            "RS score (0-100), 50-day and 200-day relative performance, beta, Sharpe ratio, RSI. "
            "Use during every ticker analysis to show sector outperformance/underperformance "
            "inside Technical Overview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol."}
            },
            "required": ["ticker"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, inputs: dict) -> str:
    """Execute the named tool with given inputs, return JSON string result."""
    try:
        if name == "get_stock_data":
            result = get_stock_data(inputs["ticker"])
        elif name == "get_financial_news":
            result = get_financial_news(
                inputs["query"],
                inputs.get("max_results", 8),
            )
        elif name == "get_market_indices":
            result = get_market_indices()
        elif name == "get_quarterly_earnings":
            result = get_quarterly_earnings(inputs["ticker"])
        elif name == "get_chart_patterns":
            result = get_chart_patterns(inputs["ticker"])
        elif name == "calculate_fibonacci_levels":
            result = calculate_fibonacci_levels(
                inputs["ticker"],
                inputs.get("period", "1y"),
            )
        elif name == "get_quant_metrics":
            result = get_quant_metrics(inputs["ticker"])
        else:
            result = {"error": f"Unknown tool: '{name}'"}
    except Exception as exc:
        result = {"error": f"Tool '{name}' raised an exception: {exc}"}

    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Main ReAct agent loop
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    conversation_history: list,
    on_tool_call=None,
) -> tuple[str, list]:
    """
    Run the Anthropic ReAct loop until a final text response is produced.

    Args:
        user_message:          The user's latest input (already enriched by main.py).
        conversation_history:  Running list of {role, content} dicts (mutated in place).
        on_tool_call:          Optional callback(tool_name, tool_input) for UI feedback.

    Returns:
        (final_text, updated_conversation_history)
    """
    # Append the user turn
    conversation_history.append({"role": "user", "content": user_message})

    # Working copy of messages sent to the API
    messages = list(conversation_history)

    # ── Logging accumulators ──────────────────────────────────────────────
    _start_time:       float            = time.time()
    _stock_data_calls: dict[str, dict]  = {}   # ticker  → get_stock_data result
    _news_results:     list             = []   # raw get_financial_news results
    _market_data:      dict | None      = None # get_market_indices result
    _tools_used:       list[str]        = []   # ordered list of tool names called

    max_iterations = 10  # Safety cap to prevent runaway loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SCHEMA,
            messages=messages,
        )

        # ---- Final text response ------------------------------------------
        if response.stop_reason == "end_turn":
            final_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            conversation_history.append(
                {"role": "assistant", "content": response.content}
            )
            # ── Save every stock analysis that occurred in this run ────────
            duration = time.time() - _start_time
            for ticker, stock_data in _stock_data_calls.items():
                try:
                    log_stock_analysis(
                        ticker               = ticker,
                        stock_data           = stock_data,
                        analysis_text        = final_text,
                        news_results         = _news_results,
                        market_data          = _market_data,
                        tools_used           = _tools_used,
                        analysis_duration_sec= duration,
                        model_used           = PRIMARY_MODEL,
                    )
                except Exception:
                    pass  # Never let logging break the agent
            return final_text, conversation_history

        # ---- Tool use -------------------------------------------------------
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if on_tool_call:
                    on_tool_call(block.name, block.input)

                result_json = dispatch_tool(block.name, block.input)
                _tools_used.append(block.name)

                # ── Capture tool results for the logger ────────────────────
                try:
                    parsed = json.loads(result_json)
                    if "error" not in parsed:
                        if block.name == "get_stock_data":
                            _stock_data_calls[block.input["ticker"].upper()] = parsed
                        elif block.name == "get_financial_news":
                            _news_results.append({
                                "query":    block.input.get("query", ""),
                                "articles": parsed.get("articles", []),
                            })
                        elif block.name == "get_market_indices":
                            _market_data = parsed
                except Exception:
                    pass

                tool_results.append(
                    {
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_json,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — break and return whatever text exists
        break

    fallback = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return fallback or "An unexpected error occurred in the agent loop.", conversation_history


# ---------------------------------------------------------------------------
# News importance evaluator (used by Live Monitor — runs Haiku for speed)
# ---------------------------------------------------------------------------

def evaluate_news_importance(headline: str) -> dict:
    """
    Lightweight Claude Haiku call to classify a headline as market-critical or not.

    Returns: {"is_critical": bool, "reason": str, "impact": "HIGH"|"MEDIUM"|"LOW"}
    """
    prompt = (
        f"Classify this financial news headline as CRITICAL (market-moving) or not.\n\n"
        f"CRITICAL criteria (any one qualifies):\n"
        f"• Central bank surprise rate decision (Fed, ECB, BOI)\n"
        f"• Major geopolitical escalation (war, sanctions, energy embargo)\n"
        f"• Mega-cap earnings miss or beat >10% vs consensus\n"
        f"• Systemic financial risk (bank failure, sovereign default)\n"
        f"• Massive regulatory action (antitrust break-up, emergency ban)\n"
        f"• Sudden commodity shock (>5% oil/gold move)\n\n"
        f'Headline: "{headline}"\n\n'
        f'Reply with ONLY valid JSON on one line:\n'
        f'{{"is_critical": true/false, "reason": "short reason", "impact": "HIGH|MEDIUM|LOW"}}'
    )

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {"is_critical": False, "reason": "evaluation failed", "impact": "LOW"}


# ---------------------------------------------------------------------------
# Market Dashboard News Summarizer — Haiku call → Hebrew bullet points
# ---------------------------------------------------------------------------

def summarize_market_news(headlines: list) -> str:
    """
    Take a list of financial headline strings and return 3–4 Hebrew bullet
    points summarising current market sentiment, geopolitical risks, and
    macro themes.  Uses Claude Haiku for speed and cost efficiency.
    """
    if not headlines:
        return "• אין כותרות חדשות זמינות כרגע."

    headlines_text = "\n".join(f"• {h}" for h in headlines[:15])
    prompt = (
        "אתה אנליסט פיננסי בכיר. קרא את כותרות החדשות הכלכליות הבאות "
        "וסכם את הסנטימנט הגלובלי בשוק ב-3–4 נקודות תמציתיות בעברית.\n"
        "התמקד ב: השפעה על השווקים, סיכונים גיאופוליטיים, ומגמות מאקרו-כלכליות.\n\n"
        f"כותרות:\n{headlines_text}\n\n"
        "ענה בדיוק בפורמט (נקודות בועות בלבד, ללא כותרות נוספות):\n"
        "• [תובנה 1]\n• [תובנה 2]\n• [תובנה 3]\n• [תובנה 4 אם רלוונטי]"
    )

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        return f"• שגיאה בטעינת סיכום: {exc}"


# ---------------------------------------------------------------------------
# Dashboard analysis — single focused Claude call for the Streamlit UI
# ---------------------------------------------------------------------------

_DASHBOARD_PROMPT = """\
You are FinanceGPT — a financial courtroom. Think in English for all internal logic \
to maximise precision, then output every text field in professional Hebrew (RTL).
Return ONLY a valid JSON object — no triple-quotes, no extra text.

== 5-STEP ANALYSIS FRAMEWORK (execute in this order internally) ==

STEP 1 — THE BIG PICTURE (Macro & Sector):
  • Is the sector trending up relative to SPY? (use sector_vs_spy_50d)
  • Is the stock outperforming its sector ETF and the S&P 500?
  → Feed findings into geopolitical_analysis and bull/bear thesis.

STEP 2 — FUNDAMENTAL SNAPSHOT:
  • Key metrics: P/E Ratio, Revenue Growth (YoY), EPS surprises, forward P/E.
  • Is the stock overvalued or undervalued vs. historical average?
  → Feed findings into fundamental_health and position_sizing.

STEP 3 — TECHNICAL DEEP DIVE:
  • Moving Averages: distance from SMA-44 (Momentum) and SMA-150 (Institutional Trend).
    Dist_44 = (Price − SMA_44) / SMA_44 × 100
    Dist_150 = (Price − SMA_150) / SMA_150 × 100
    Express with LaTeX: $\\text{{Dist}}_{{150}} = \\frac{{\\text{{Price}} - SMA_{{150}}}}{{SMA_{{150}}}} \\times 100$
  • MA state: "Bullish Stack" (Price > SMA_44 > SMA_150) | "Support Test" (±2% of MA) | "Mean Reversion" (>5% below SMA_150)
  • Pattern Recognition: identify active pattern — Cup & Handle, Bull Flag, Double Bottom, VCP, Double Top.
  → Feed findings into technical_commentary, technical_pattern, quant_audit.

STEP 4 — CATALYSTS & SENTIMENT:
  • Upcoming earnings date, Fed/CPI events that will impact this stock.
  • Summarize current "vibe" from financial news and analyst reports.
  → Feed findings into bull_thesis, bear_thesis.

STEP 5 — THE FINAL VERDICT:
  • Risk/Reward: risk_reward_ratio = (resistance - price) / (price - support), format "1:X.X".
  • Final rating: BUY/SELL/HOLD/WATCH with Confidence Score 1-10.
    Baseline: Fib + SMA + pattern confluence → +2. sector_vs_spy_50d < -3% → -1. beta > 1.5 → -1.
  • Position Sizing: beta > 1.5 → cap 2%; rs_rating < 40 → cap 3%; rs_rating > 70 AND sharpe > 1.0 → allow 5-10%.
  • trade_plan: entry = pullback to support/Fib, target = Fib extension, stop = below last support.
  → Feed findings into judge_verdict, trade_plan, position_sizing, technical_map.

== ADDITIONAL MANDATORY RULES ==
• bull_thesis MUST cite: RS rating, price vs POC, Sharpe ratio, and a specific technical pattern.
• bear_thesis MUST cite: beta, sector_vs_spy performance, POC breach risk, statistical-bubble if price > 2 SD above LR.
• judge_verdict MUST explicitly reference quant metrics (RS, Beta, Sharpe) in its reasoning.
• If ML forecast aligns with a Fib level (±3%) → cite as high-probability target.
• Use sma_50 as proxy for SMA-44 and sma_200 as proxy for SMA-150 when exact values unavailable.

== STOCK DATA ==
{stock_json}

== QUANT METRICS (RS, Beta, Sharpe, POC, Sector ETF) ==
{quant_json}

== RECENT NEWS HEADLINES ==
{news_text}

== TRADINGVIEW TECHNICAL SUMMARY ==
{ta_json}

== FIBONACCI LEVELS & ML FORECASTS ==
{fib_forecast_json}

== MA DISTANCES (1-Year Daily Chart) ==
{chart_json}
{video_section}

Return EXACTLY this JSON structure (all text fields in Hebrew):
{{
  "technical_map": "<5-line Hebrew summary — one line per step:\n• 🌍 מאקרו: [sector trend + relative strength vs SPY]\n• 📊 פונדמנטלי: [P/E vs growth, EPS surprise, valuation verdict]\n• 📈 טכני: [MA state + exact distances + active pattern]\n• 📰 קטליסטים: [next earnings date + news vibe in 1 sentence]\n• 🏆 ורדיקט: [BUY/SELL/HOLD + Confidence X/10 + suggested position size]>",
  "bull_thesis": "<【הסנגור】3-4 משפטים: ציין RS rating ומה הוא מסמל, מחיר ביחס ל-POC (תמיכה/התנגדות), Sharpe Ratio כמדד יעילות, פריצה טכנית (Cup&Handle/VCP/Base), מגמת EPS חיובית>",
  "bear_thesis": "<【הקטגור】3-4 משפטים: ציין Beta ורמת הסיכון, ביצועי הסקטור ביחס ל-SPY, סיכון פריצת ה-POC כלפי מטה, בועה סטטיסטית אם רלוונטי, חדשות שליליות>",
  "quant_audit": "<【ביקורת קוואנט】3-4 משפטים: RS {rs_50d}d / RS Rating / Beta / Sharpe — האם האותות מסכימים? ציין מרחק מ-SMA_44 ו-SMA_150 בפורמט: 'המניה נמצאת במרחק של X% מממוצע 150, מה שמעיד על נקודת כניסה אסטרטגית.' זהה דפוס: Bullish Stack / Support Test / Mean Reversion. confluence פיבונאצ'י+SMA? אמינות Monte Carlo R²>",
  "judge_verdict": "<【פסק הדין】3-4 משפטים: סינתזה של RS+Beta+Sharpe+Fib. נמק את הכיוון. ציין גודל פוזיציה ומדוע. ציין תרחיש שישנה את הדעה>",
  "position_sizing": "<X% מהתיק — נמק לפי Beta={beta}/RS={rs_rating}/Sharpe={sharpe}>",
  "trade_plan": {{
    "entry": <number>,
    "target": <number>,
    "stop_loss": <number>
  }},
  "executive_summary": "<3-4 bullet points with specific numbers>",
  "technical_pattern": "<MA state: Bullish Stack / Support Test on SMA-44 or SMA-150 / Mean Reversion / Below MA — cite exact SMA values and distances>",
  "fundamental_health": "<EPS growth, P/E, margin, ROE with numbers>",
  "technical_commentary": "<3 sentences: state exact SMA_44 and SMA_150 distances using $\\text{Dist}_{150} = \\frac{\\text{Price} - SMA_{150}}{SMA_{150}} \\times 100$ notation, classify MA state (Bullish Stack / Support Test / Mean Reversion / Below MA), then describe RSI and MACD signal>",
  "fibonacci_commentary": "<current Fib level, Golden Pocket, SMA confluence>",
  "geopolitical_analysis": "<1 paragraph: key macro theme, geopolitical risk, and rate outlook affecting this stock>",
  "technical_conflict": <true|false>,
  "conflict_explanation": "<explain if true, else ''>",
  "verdict_hebrew": "<שורי|דובי|ניטרלי>",
  "overall_sentiment": "<BULLISH|BEARISH|NEUTRAL>",
  "recommendation": "<BUY|SELL|HOLD|WATCH>",
  "confidence_score": <1-10>,
  "support_level": <number>,
  "resistance_level": <number>,
  "risk_reward_ratio": "<'1:X.X'>",
  "price_targets": {{
    "bull": <number>,
    "base": <number>,
    "bear": <number>
  }}
}}
"""


def analyze_for_dashboard(
    stock_data: dict,
    news_articles: list,
    ta_summary: dict | None = None,
    video_context: dict | None = None,
    fib_data: dict | None = None,
    forecast_data: dict | None = None,
    quant_data: dict | None = None,
    chart_patterns: dict | None = None,
) -> dict:
    """
    Single Claude Sonnet call that returns a structured analysis dict for
    all Streamlit dashboard tabs.  No tool use — data is pre-fetched and
    injected directly into the prompt.

    Returns keys: executive_summary, technical_commentary,
                  geopolitical_analysis, overall_sentiment,
                  recommendation, key_risks, price_targets.
    """
    # Trim stock data — exclude low-value fields (exchange, currency, peg, volume_ratio)
    _keep = {
        "ticker", "company_name", "sector", "industry",
        "current_price", "week_52_high", "week_52_low", "sma_50", "sma_200",
        "rsi_14", "price_vs_sma50_pct", "price_vs_sma200_pct",
        "beta", "market_cap_human", "pe_ratio", "forward_pe",
        "eps_ttm", "eps_forward", "revenue_growth_yoy", "earnings_growth_yoy",
        "profit_margin", "roe", "debt_to_equity", "next_earnings_date",
    }
    slim_data = {k: v for k, v in stock_data.items() if k in _keep}

    # Cap at 10 headlines (titles only — summaries add tokens without analytical value)
    news_text = "\n".join(
        f"• {a.get('title', '')}" for a in news_articles[:10]
    ) or "אין כותרות חדשות זמינות."

    # Build optional video insights section
    if video_context and video_context.get("key_points"):
        vc = video_context
        points   = "\n".join(f"• {p}" for p in vc.get("key_points", []))
        targets  = "، ".join(vc.get("price_targets", [])) or "לא הוזכרו"
        cats     = "، ".join(vc.get("catalysts", []))     or "לא הוזכרו"
        risks_v  = "، ".join(vc.get("risks_mentioned", [])) or "לא הוזכרו"
        macro_v  = "، ".join(vc.get("macro_views", []))   or "לא הוזכרו"
        video_section = (
            f"\n== תובנות מסרטון יוטיוב (הוסף על ידי המשתמש) ==\n"
            f"עמדת האנליסט בסרטון: {vc.get('analyst_stance','ניטרלי')}\n"
            f"נקודות מפתח:\n{points}\n"
            f"יעדי מחיר שהוזכרו: {targets}\n"
            f"קטליסטים: {cats}\n"
            f"סיכונים שהוזכרו: {risks_v}\n"
            f"תפיסות מאקרו: {macro_v}\n\n"
            f"שלב תובנות אלו בניתוח שלך. ציין 'מבוסס על ניתוח מסרטון יוטיוב' "
            f"בהמלצה האישית ובניתוח הגיאופוליטי כשרלוונטי.\n"
        )
    else:
        video_section = ""

    # Build Fibonacci + Forecast JSON section
    fib_forecast = {}
    if fib_data and "error" not in fib_data:
        # Send only the 5 most actionable levels (nearest retracements + top 2 extensions)
        all_levels = fib_data.get("levels", {})
        current    = fib_data.get("current_price", 0) or 0
        sorted_levels = sorted(all_levels.items(), key=lambda kv: abs(kv[1] - current))
        nearest5 = dict(sorted_levels[:5])
        all_ext  = fib_data.get("extensions", {})
        top2_ext = dict(list(all_ext.items())[:2])
        fib_forecast["fibonacci"] = {
            "direction":        fib_data.get("direction"),
            "nearest_level":    fib_data.get("nearest_level"),
            "nearest_price":    fib_data.get("nearest_price"),
            "distance_pct":     fib_data.get("distance_pct"),
            "at_golden_pocket": fib_data.get("at_golden_pocket"),
            "confluence_sma":   fib_data.get("confluence_sma"),
            "levels":           nearest5,
            "extensions":       top2_ext,
        }
    if forecast_data and "error" not in forecast_data:
        fib_forecast["forecasts"] = {
            "linear_regression": forecast_data.get("linear_regression"),
            "random_forest":     forecast_data.get("random_forest"),
            "monte_carlo":       forecast_data.get("monte_carlo"),
            "fibonacci_alignment": forecast_data.get("fibonacci_alignment", {}),
        }

    # Slim quant data (only fields needed by the prompt)
    _quant_keys = {
        "rs_50d", "rs_200d", "rs_rating", "beta_60d", "sharpe_ratio",
        "point_of_control", "sector_etf", "sector_etf_50d", "sector_vs_spy_50d",
    }
    slim_quant = (
        {k: v for k, v in quant_data.items() if k in _quant_keys}
        if quant_data and "error" not in quant_data
        else {}
    )

    # MA distances only — passed pre-filtered from app.py (_slim_chart)
    slim_chart: dict | str = "לא זמין"
    if chart_patterns and "error" not in chart_patterns:
        slim_chart = {
            "current_price":  chart_patterns.get("current_price"),
            "sma_44":         chart_patterns.get("sma_44"),
            "sma_150":        chart_patterns.get("sma_150"),
            "sma_200":        chart_patterns.get("sma_200"),
            "dist_to_44_pct":  chart_patterns.get("dist_to_44_pct"),
            "dist_to_150_pct": chart_patterns.get("dist_to_150_pct"),
            "dist_to_200_pct": chart_patterns.get("dist_to_200_pct"),
        }

    # Build data strings — use explicit str.replace() instead of str.format()
    # to avoid KeyError when dynamic data (news titles, descriptions) contains
    # literal "{...}" patterns that Python's formatter misinterprets.
    _pv = {
        "{rs_50d}":    str(slim_quant.get("rs_50d",  "N/A")),
        "{rs_rating}": str(slim_quant.get("rs_rating", "N/A")),
        "{beta}":      str(slim_quant.get("beta_60d", "N/A")),
        "{sharpe}":    str(slim_quant.get("sharpe_ratio", "N/A")),
    }

    # Step 1: substitute the 4 small inline tokens (they don't contain braces)
    prompt = _DASHBOARD_PROMPT
    for token, value in _pv.items():
        prompt = prompt.replace(token, value)

    # Step 2: substitute the large data blocks using a safe delimiter approach
    # (replace the named placeholders that may contain arbitrary text/braces)
    _data = {
        "{stock_json}":       json.dumps(slim_data,      default=str, ensure_ascii=False),
        "{quant_json}":       json.dumps(slim_quant,     default=str, ensure_ascii=False) if slim_quant else "לא זמין",
        "{news_text}":        news_text,
        "{ta_json}":          json.dumps(ta_summary or {}, default=str, ensure_ascii=False),
        "{fib_forecast_json}": json.dumps(fib_forecast,  default=str, ensure_ascii=False) if fib_forecast else "לא זמין",
        "{chart_json}":        json.dumps(slim_chart,    default=str, ensure_ascii=False) if isinstance(slim_chart, dict) else slim_chart,
        "{video_section}":    video_section,
    }
    for token, value in _data.items():
        prompt = prompt.replace(token, value)

    # Step 3: unescape the {{ }} used for literal JSON braces in the template
    prompt = prompt.replace("{{", "{").replace("}}", "}")

    _FALLBACK = {
        # ── Judicial Multi-Agent Protocol fields ──
        "technical_map": "• 🌍 מאקרו: נתונים אינם זמינים\n• 📊 פונדמנטלי: נתונים אינם זמינים\n• 📈 טכני: נתונים אינם זמינים\n• 📰 קטליסטים: נתונים אינם זמינים\n• 🏆 ורדיקט: נתונים אינם זמינים",
        "bull_thesis": "הסנגור אינו זמין כרגע.",
        "bear_thesis": "הקטגור אינו זמין כרגע.",
        "quant_audit": "ביקורת הנתונים אינה זמינה כרגע.",
        "judge_verdict": "פסק הדין אינו זמין כרגע.",
        "position_sizing": "לא ניתן להמליץ על גודל פוזיציה.",
        "trade_plan": {"entry": 0, "target": 0, "stop_loss": 0},
        # ── Existing fields (used by other tabs) ──
        "executive_summary": "הניתוח אינו זמין כרגע.",
        "technical_pattern": "לא זוהה דפוס טכני.",
        "fundamental_health": "נתונים פונדמנטליים אינם זמינים.",
        "technical_commentary": "הנתונים הטכניים אינם זמינים.",
        "fibonacci_commentary": "נתוני פיבונאצ'י אינם זמינים.",
        "geopolitical_analysis": "הניתוח הגיאופוליטי אינו זמין.",
        "technical_conflict": False,
        "conflict_explanation": "",
        "verdict_hebrew": "ניטרלי",
        "overall_sentiment": "NEUTRAL",
        "recommendation": "HOLD",
        "confidence_score": 1,
        "support_level": 0,
        "resistance_level": 0,
        "risk_reward_ratio": "N/A",
        "price_targets": {"bull": 0, "base": 0, "bear": 0},
    }

    def _parse_json(text: str) -> dict | None:
        """Robustly extract and parse a JSON object from Claude's response."""
        # Strip markdown fences (``` or ```json) — both leading and trailing
        text = text.strip()
        # Remove opening fence
        text = re.sub(r"^```+(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        # Remove closing fence
        text = re.sub(r"\n?```+\s*$", "", text).strip()

        # Find outermost JSON object
        start = text.find("{")
        if start == -1:
            return None

        # Walk to find matching closing brace
        depth, in_str, escape = 0, False, False
        end = -1
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        raw = text[start: end + 1]

        def _attempt(s: str):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None

        # Attempt 1: raw
        result = _attempt(raw)
        if result:
            return result

        # Attempt 2: fix literal newlines inside JSON strings
        cleaned = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
            raw,
        )
        result = _attempt(cleaned)
        if result:
            return result

        # Attempt 3: remove trailing commas before } or ]
        no_trail = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        result = _attempt(no_trail)
        if result:
            return result

        # Attempt 4: strip any text before first { (in case Claude added a preamble)
        inner_start = cleaned.find("{")
        if inner_start > 0:
            result = _attempt(cleaned[inner_start:])
            if result:
                return result

        return None

    try:
        resp = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=6000,          # Judicial protocol adds ~2k tokens vs original
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        result = _parse_json(text)
        if result:
            for k, v in _FALLBACK.items():
                result.setdefault(k, v)
            # ── Log analysis to stock_analyses.json ───────────────────────
            try:
                analysis_text = (
                    f"## {stock_data.get('ticker')} Analysis\n"
                    f"### TL;DR\n{result.get('executive_summary', '')}\n\n"
                    f"### Technical Overview\n{result.get('technical_commentary', '')}\n\n"
                    f"### Risk Factors\n{chr(10).join(f'- {r}' for r in result.get('key_risks', []))}\n\n"
                    f"### Summary & Outlook\n"
                    f"Recommendation: {result.get('recommendation', 'N/A')}\n"
                    f"{result.get('judge_verdict', '')}"
                )
                log_stock_analysis(
                    ticker        = stock_data.get("ticker", ""),
                    stock_data    = stock_data,
                    analysis_text = analysis_text,
                    news_results  = [{"query": stock_data.get("ticker",""), "articles": news_articles}],
                    tools_used    = ["analyze_for_dashboard"],
                    model_used    = PRIMARY_MODEL,
                )
            except Exception:
                pass  # never block the dashboard
            return result
        # JSON parsing failed — log the raw response for debugging
        ticker_tag = stock_data.get("ticker", "?")
        print(f"[analyze_for_dashboard] JSON parse failed for {ticker_tag}. "
              f"First 300 chars of response: {text[:300]!r}")
    except Exception as exc:
        import traceback
        ticker_tag = stock_data.get("ticker", "?")
        print(f"[analyze_for_dashboard] Exception for {ticker_tag}: {exc}")
        traceback.print_exc()

    return _FALLBACK


# ---------------------------------------------------------------------------
# NASDAQ 100 Screener — AI commentary on results
# ---------------------------------------------------------------------------

def summarize_screener_results(matches: list, criteria: dict) -> str:
    """
    2-3 sentence Hebrew commentary on screener results: what sectors dominate,
    what the common technical theme is, and any notable outliers.
    Uses Haiku for speed.
    """
    if not matches:
        return "הסריקה לא מצאה מניות העומדות בקריטריונים הנוכחיים."

    top10 = matches[:10]
    tickers_str = ", ".join(m["ticker"] for m in top10)
    avg_rs  = round(sum(m["rs_rating"] for m in top10) / len(top10), 1)
    avg_sh  = round(sum(m["sharpe"]    for m in top10 if m.get("sharpe")) /
                    max(1, sum(1 for m in top10 if m.get("sharpe"))), 2)

    prompt = (
        f"אתה FinanceGPT. סורק NASDAQ 100 מצא {len(matches)} מניות עם "
        f"RS Rating ≥ {criteria.get('min_rs', 70)}, "
        f"Beta ≤ {criteria.get('max_beta', 2.0)}, "
        f"Sharpe ≥ {criteria.get('min_sharpe', 0.3)}.\n\n"
        f"10 המניות המובילות (לפי RS): {tickers_str}\n"
        f"ממוצע RS Rating של TOP-10: {avg_rs} | ממוצע Sharpe: {avg_sh}\n\n"
        "כתוב 2-3 משפטים בעברית מקצועית המסכמים:\n"
        "1. איזה סקטור/תמה שוק בולטת בתוצאות?\n"
        "2. מה מסמן RS Rating גבוה בסביבת שוק זו?\n"
        "3. האם יש מניה בולטת שכדאי לשים לב אליה?\n"
        "התמקד בפיננסים ושווקים בלבד. היה ספציפי עם שמות."
    )

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return f"נמצאו **{len(matches)}** מניות. המובילות לפי RS: {tickers_str}."


# ---------------------------------------------------------------------------
# YouTube Transcript Summarizer
# ---------------------------------------------------------------------------

_TRANSCRIPT_PROMPT = """\
אתה אנליסט פיננסי המסכם תמלילי סרטוני ניתוח מניות.
קרא את התמליל וחלץ את נקודות המפתח הפיננסיות בלבד — התעלם מתוכן שיווקי או אישי.
{ticker_context}

תמליל:
{transcript}

החזר JSON תקני בלבד (ללא markdown fences):
{{
  "analyst_stance": "<שורי|דובי|ניטרלי>",
  "key_points": ["<5-8 תובנות מרכזיות>"],
  "price_targets": ["<יעדי מחיר שהוזכרו — אם הוזכרו>"],
  "macro_views": ["<תפיסות מאקרו — ריבית, כלכלה, גיאופוליטיקה>"],
  "catalysts": ["<קטליסטים ספציפיים למניה/סקטור>"],
  "risks_mentioned": ["<סיכונים שהוזכרו>"]
}}
מלא רק שדות שיש להם תוכן ממשי מהתמליל.
"""


def summarize_transcript(transcript_text: str, ticker: str = "") -> dict:
    """
    Compress a YouTube transcript into structured market thesis points.
    Uses Claude Haiku for speed and cost efficiency.

    Returns a dict with: analyst_stance, key_points, price_targets,
                         macro_views, catalysts, risks_mentioned.
    """
    _FALLBACK = {
        "analyst_stance":  "ניטרלי",
        "key_points":      ["לא ניתן לסכם את הסרטון"],
        "price_targets":   [],
        "macro_views":     [],
        "catalysts":       [],
        "risks_mentioned": [],
    }

    ticker_context = f"הסרטון עוסק במניה/נושא: {ticker}" if ticker else ""
    trimmed = transcript_text[:6000]
    if len(transcript_text) > 6000:
        trimmed += "\n... [תמליל קוצץ לחיסכון בטוקנים]"

    prompt = _TRANSCRIPT_PROMPT.format(
        ticker_context=ticker_context,
        transcript=trimmed,
    )

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,   # Haiku — fast and cheap
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```+(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```+\s*$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            for k, v in _FALLBACK.items():
                result.setdefault(k, v)
            return result
    except Exception:
        pass

    return _FALLBACK


# ---------------------------------------------------------------------------
# Agent Chat — "צ'אט עם הסוכן" tab
# ---------------------------------------------------------------------------

# The user's core portfolio — referenced in every chat session
_PORTFOLIO_CONTEXT = (
    "== תיק ההשקעות של המשתמש (תמיד תעדף ניתוח של אלה) ==\n"
    "• NVDA  — NVIDIA | מוליכים למחצה / AI\n"
    "• SOFI  — SoFi Technologies | פינטק / בנקאות דיגיטלית\n"
    "• IBIT  — iShares Bitcoin Trust ETF | חשיפה ל-Bitcoin\n"
    "• ETHA  — iShares Ethereum Trust ETF | חשיפה ל-Ethereum\n"
    "• MAGS  — Magnificent 7 ETF | סל טק מגה-קאפ\n"
    "• MSTU  — MicroStrategy 2x Leveraged ETF | 2x ממונף, קורלציה גבוהה ל-Bitcoin\n"
    "זכור ובנה על כל דיון קודם על הנכסים האלה מהיסטוריית השיחה.\n"
)

# Mode-specific instructions injected at the top of the system prompt
_MODE_INSTRUCTIONS: dict[str, str] = {
    "analyst": (
        "== מצב /Analyst פעיל ==\n"
        "התמקד במדדים כמותיים: Sharpe Ratio, Beta, RS Rating, VOL, רמות תמיכה/התנגדות, "
        "דפוסי מחיר טכניים.\n"
        "השתמש בסימון LaTeX לנוסחאות פיננסיות:\n"
        "  שארפ: $S = \\frac{R_p - R_f}{\\sigma_p}$\n"
        "  פיבונאצ'י: $F_{61.8\\%} = High - 0.618 \\times (High - Low)$\n"
    ),
    "strategist": (
        "== מצב /Strategist פעיל ==\n"
        "התמקד: מגמות מאקרו לטווח ארוך, רוטציית סקטורים, "
        "הקצאת תיק אופטימלית, סביבת ריבית Fed, גיאופוליטיקה ומגמות מחזוריות.\n"
    ),
    "bear": (
        "== מצב /Bear (עורך דין השטן) פעיל ==\n"
        "הנח שהתיזה השורית שגויה. זהה כל סיכון אפשרי — מה יכול להשתבש? "
        "מבחני לחץ של ההנחות, פגיעויות שהשוורים מתעלמים מהן, תרחישי זנב שמאלי.\n"
    ),
}


def handle_chat_query(
    user_message: str,
    stock_context: dict,
    chat_history: list,
    analysis_context: dict | None = None,
    video_context: dict | None = None,
) -> str:
    """
    Handle a Hebrew conversational query about the current stock / portfolio.

    Supports three mode prefixes: /Analyst, /Strategist, /Bear.
    Automatically fetches live market grounding (RSS headlines) before responding.

    Returns: Professional Hebrew response string.
    """
    ac = analysis_context or {}

    # ── 1. Detect response mode ───────────────────────────────────────────────
    mode     = "balanced"
    msg_body = user_message.strip()
    for prefix, key in [("/Analyst", "analyst"), ("/Strategist", "strategist"), ("/Bear", "bear")]:
        if msg_body.lower().startswith(prefix.lower()):
            mode     = key
            msg_body = msg_body[len(prefix):].strip() or msg_body
            break

    mode_block = _MODE_INSTRUCTIONS.get(mode, "")

    # ── 2. Live market grounding — fetch recent headlines via RSS ─────────────
    live_lines: list[str] = []
    try:
        for q in (stock_context.get("ticker", "market"), "Federal Reserve market inflation"):
            res = get_financial_news(q, 4)
            for a in res.get("articles", []):
                t = a.get("title", "").strip()
                if t and t not in live_lines:
                    live_lines.append(t)
                    if len(live_lines) >= 6:
                        break
            if len(live_lines) >= 6:
                break
    except Exception:
        pass

    live_context = (
        "\n".join(f"• {h}" for h in live_lines)
        if live_lines
        else "נתוני חדשות בזמן אמת אינם זמינים כרגע."
    )

    # ── 3. Build numeric formatter ────────────────────────────────────────────
    def _n(val, dollar=False, pct=False, suffix="x", decimals=2):
        if val is None:
            return "—"
        try:
            v = float(val)
            if dollar:
                return f"${v:,.{decimals}f}"
            if pct:
                sign = "+" if v >= 0 else ""
                return f"{sign}{v * 100:.1f}%"
            return f"{v:.{decimals}f}{suffix}"
        except Exception:
            return str(val) if val else "—"

    # ── 4. Build video context section ────────────────────────────────────────
    vc = video_context or {}
    if vc.get("key_points"):
        pts = "\n".join(f"  • {p}" for p in vc.get("key_points", []))
        _vs = (
            f"\n== תובנות מסרטון שהמשתמש הוסיף ==\n"
            f"עמדת האנליסט: {vc.get('analyst_stance','')}\n{pts}\n"
            f"אם רלוונטי, שלב תובנות אלו בתשובה.\n"
        )
    else:
        _vs = ""

    # ── 5. Assemble system prompt using safe concatenation ───────────────────
    #     (avoids KeyError when live_context/video contains literal { } chars)
    if stock_context and stock_context.get("ticker"):
        ticker_block = (
            f"== מניה / נכס בניתוח נוכחי ==\n"
            f"טיקר: {stock_context.get('ticker','—')} | "
            f"חברה: {stock_context.get('company_name','—')} | "
            f"סקטור: {stock_context.get('sector','—')}\n\n"
            f"נתוני מפתח:\n"
            f"• מחיר: {_n(stock_context.get('current_price'), dollar=True)}"
            f"  |  שווי שוק: {stock_context.get('market_cap_human') or '—'}\n"
            f"• P/E: {_n(stock_context.get('pe_ratio'))}"
            f"  |  Forward P/E: {_n(stock_context.get('forward_pe'))}"
            f"  |  EPS קדימה: {_n(stock_context.get('eps_forward'), dollar=True)}\n"
            f"• RSI(14): {_n(stock_context.get('rsi_14'), suffix='')}"
            f"  |  SMA-50: {_n(stock_context.get('sma_50'), dollar=True)}"
            f"  |  SMA-200: {_n(stock_context.get('sma_200'), dollar=True)}\n"
            f"• צמיחת הכנסות: {_n(stock_context.get('revenue_growth_yoy'), pct=True)}"
            f"  |  שולי רווח: {_n(stock_context.get('profit_margin'), pct=True)}"
            f"  |  ROE: {_n(stock_context.get('roe'), pct=True)}\n"
            f"• ביטא: {_n(stock_context.get('beta'), suffix='')}"
            f"  |  חוב/הון: {_n(stock_context.get('debt_to_equity'), suffix='')}"
            f"  |  דוח הבא: {str(stock_context.get('next_earnings_date') or '—')[:10]}\n"
            f"• סנטימנט: {ac.get('overall_sentiment','—')}"
            f"  |  המלצה: {ac.get('recommendation','—')}"
            f"  |  ורדיקט: {ac.get('verdict_hebrew','—')}\n"
        )
    else:
        ticker_block = ""

    system = (
        "You are FinanceGPT — an elite institutional portfolio advisor and market strategist. "
        "Think exclusively in English for all internal reasoning; "
        "deliver every user-facing word in professional Hebrew (RTL).\n\n"
        + mode_block
        + "\n"
        + _PORTFOLIO_CONTEXT
        + "\n== חדשות בזמן אמת (RSS Live Feed) ==\n"
        + live_context
        + "\n\n"
        + ticker_block
        + "\n== כללי תגובה ==\n"
        "• פלט בעברית מקצועית בלבד — RTL, ממוקד בפיננסים ושווקים בלבד\n"
        "• LaTeX לנוסחאות: $\\text{Dist}_{150} = \\frac{\\text{Price} - SMA_{150}}{SMA_{150}} \\times 100$\n"
        "• **Bold** למספרים מרכזיים, ▲/▼ לכיוון, ✅/⚠️/🎯 לאיתות\n"
        "• תגובה קצרה (4-6 משפטים) אלא אם מבקשים ניתוח מעמיק\n"
        "• אם אין לך נתון — אמור זאת בכנות; אל תמציא מספרים\n"
        "\n== מסגרת ניתוח מניה (5 שלבים — השתמש בסדר הזה כשמתבקש ניתוח) ==\n"
        "כשמשתמש מבקש ניתוח מניה, עקוב תמיד בסדר הבא:\n\n"
        "**שלב 1 — התמונה הגדולה (מאקרו וסקטור)**\n"
        "• האם הסקטור (טכנולוגיה / פיננסים / וכו') במגמה עולה?\n"
        "• האם המניה מנצחת את ETF הסקטור ואת S&P 500?\n\n"
        "**שלב 2 — סנאפשוט פונדמנטלי (המספרים)**\n"
        "• P/E Ratio, Revenue Growth (YoY), הפתעות EPS\n"
        "• האם המניה מוערכת יתר על המידה ביחס לממוצע ההיסטורי?\n\n"
        "**שלב 3 — צלילה טכנית (הגרף)**\n"
        "• ממוצעים נעים: מרחק מ-SMA 44 (מומנטום) ו-SMA 150 (מגמה מוסדית)\n"
        "• זיהוי דפוס: Cup and Handle, Bull Flag, Double Bottom, VCP\n"
        "• הצג פערים בנוסחת LaTeX: $\\text{Dist}_{44} = \\frac{P - SMA_{44}}{SMA_{44}} \\times 100$\n\n"
        "**שלב 4 — קטליסטים וסנטימנט (החדשות)**\n"
        "• דוחות קרובים, אירועי Fed/CPI שישפיעו על המניה\n"
        "• תמצת את ה-'ויב' הנוכחי מחדשות פיננסיות ודוחות אנליסטים\n\n"
        "**שלב 5 — הורדיקט הסופי (השופט)**\n"
        "• Risk/Reward: $R/R = \\frac{Target - Entry}{Entry - Stop}$\n"
        "• המלצה: BUY/SELL/HOLD עם Confidence Score (1-10)\n"
        "• Position Sizing: כמה אחוז מהתיק? (לפי Beta, RS Rating)\n"
        "• Entry Zone / Target / Stop-Loss ברורים\n"
        + _vs
    )

    # ── 6. Build message history ──────────────────────────────────────────────
    history_slice = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history[-20:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    history_slice.append({"role": "user", "content": msg_body})

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=1200,
            system=system,
            messages=history_slice,
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        return f"⚠️ שגיאה בקבלת תגובה: {exc}"


# ---------------------------------------------------------------------------
# Portfolio Impact Summarizer — Daily Summary for the Market Dashboard
# ---------------------------------------------------------------------------

_PORTFOLIO_IMPACT_PROMPT = """\
You are FinanceGPT. Think in English internally. Write ONLY in professional Hebrew.

== USER'S PORTFOLIO ==
{portfolio_snapshot}

== GLOBAL MARKET CONDITIONS TODAY ==
{market_snapshot}

== TODAY'S KEY HEADLINES ==
{headlines}

Task: Write EXACTLY 3 concise Hebrew bullet points explaining how TODAY's global market \
conditions (indices, crypto, macro, news) are SPECIFICALLY impacting the user's 6 assets.

Rules:
- Mention specific asset names (NVDA, SOFI, IBIT, ETHA, MAGS, MSTU) by name
- Each bullet: 1-2 sentences, cite actual numbers from the data above
- CAUSE → EFFECT logic: which market event → which asset → how (up/down/risk)
- Focus exclusively on market and financial dynamics — no other topics
- Professional Hebrew, RTL

Format (return ONLY these 3 lines, nothing else):
• [השפעה 1 — עם מספרים]
• [השפעה 2 — עם מספרים]
• [השפעה 3 — עם מספרים]
"""


def summarize_portfolio_impact(
    portfolio_snapshot: dict,
    market_data: dict,
    news_headlines: list,
) -> str:
    """
    Generate a 3-bullet Hebrew summary of how today's global market conditions
    are impacting the user's 6 core portfolio holdings.

    Args:
        portfolio_snapshot: {ticker: {price, change_pct, direction}} for each holding
        market_data:        indices dict from get_market_indices()
        news_headlines:     list of headline strings (up to 10)
    Returns: Hebrew string with 3 bullet points.
    """
    # Format portfolio snapshot
    p_lines = []
    for ticker, data in portfolio_snapshot.items():
        if "error" in data:
            p_lines.append(f"• {ticker}: נתון לא זמין")
        else:
            chg = data.get("change_pct")
            chg_str = (f"{'▲' if chg and chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "—")
            p_lines.append(f"• {ticker}: ${data.get('price','—')}  {chg_str}")
    portfolio_str = "\n".join(p_lines) or "נתוני תיק אינם זמינים."

    # Format market snapshot (indices)
    indices = (market_data or {}).get("indices", {})
    m_lines = []
    for name, item in indices.items():
        if "error" in item:
            continue
        chg = item.get("change_pct")
        chg_str = (f"{'▲' if chg and chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "—")
        m_lines.append(f"• {name}: {item.get('price','—')}  {chg_str}")
    market_str = "\n".join(m_lines[:8]) or "נתוני שוק אינם זמינים."

    # Format headlines
    news_str = "\n".join(f"• {h}" for h in news_headlines[:8]) or "אין כותרות זמינות."

    prompt = (
        _PORTFOLIO_IMPACT_PROMPT
        .replace("{portfolio_snapshot}", portfolio_str)
        .replace("{market_snapshot}",   market_str)
        .replace("{headlines}",          news_str)
    )

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        return f"• שגיאה בטעינת סיכום תיק: {exc}"


# ---------------------------------------------------------------------------
# Economic Calendar AI Analysis
# ---------------------------------------------------------------------------

_CALENDAR_IMPACT_PROMPT = """\
You are LEVI, an elite AI financial analyst for LEVI FINANCE.

Below are upcoming economic events in the next 90 days.
The user holds these 6 core assets: NVDA, SoFi (SOFI), IBIT, ETHA, MAGS, MSTU.

Upcoming Events:
{events_text}

Write a LEVI'S INSIGHT analysis in Hebrew that:
1. Identifies the 2-3 most market-moving events from the list
2. Explains specifically how each event could impact NVDA, SOFI, IBIT/ETHA (crypto), MAGS, MSTU
3. Gives an actionable watchlist note for each key event (e.g., "שים לב ל-NVDA לפני ה-FOMC")
4. Ends with a 1-sentence strategic outlook for the next 30 days

Format as clean Hebrew bullet points. Be concise and professional.
Use bold (**text**) for ticker names and event names.
Start directly with the analysis — no preamble.
"""


def analyze_calendar_portfolio_impact(events: list[dict]) -> str:
    """
    Generate Hebrew AI analysis of upcoming economic events and their impact
    on the user's 6 core portfolio holdings (NVDA, SOFI, IBIT, ETHA, MAGS, MSTU).

    Args:
        events: list of event dicts with keys: date, event_he, impact, country, previous, forecast
    Returns: Hebrew markdown string with LEVI'S INSIGHT analysis.
    """
    if not events:
        return "• אין אירועים כלכליים מתוכננים ב-90 הימים הקרובים."

    # Format events for the prompt (max 15 most impactful)
    high_impact = [e for e in events if e.get("impact") == "high"]
    medium_impact = [e for e in events if e.get("impact") == "medium"]
    selected = (high_impact + medium_impact)[:15]

    lines = []
    for ev in selected:
        impact_icon = "🔴" if ev.get("impact") == "high" else "🟡"
        date_str = ev.get("date", "")
        event_name = ev.get("event_he") or ev.get("event", "")
        country = "🇺🇸" if ev.get("country") == "US" else "🇮🇱"
        prev = ev.get("previous", "—")
        forecast = ev.get("forecast", "—")
        lines.append(
            f"{impact_icon} {country} {date_str} | {event_name} | "
            f"קודם: {prev} | תחזית: {forecast}"
        )

    events_text = "\n".join(lines) or "אין אירועים."

    prompt = _CALENDAR_IMPACT_PROMPT.replace("{events_text}", events_text)

    try:
        resp = client.messages.create(
            model=MONITOR_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        return f"• שגיאה בניתוח לוח שנה כלכלי: {exc}"
