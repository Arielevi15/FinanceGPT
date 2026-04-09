"""
app.py — FinanceGPT לוח בקרה פיננסי חכם
ממשק משתמש בעברית | עיצוב Bloomberg/TradingView כהה

הפעלה: streamlit run app.py
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import sys
from datetime import datetime, timezone, timedelta

# ── UTF-8 on Windows ────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ── third-party ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── project ─────────────────────────────────────────────────────────────────
from tools import (
    get_stock_data, get_financial_news, get_youtube_transcript,
    calculate_fibonacci_levels, run_forecasting_models, get_quant_metrics,
    get_quarterly_earnings, get_macro_data, get_market_indices,
    get_portfolio_snapshot, PORTFOLIO_TICKERS,
    run_nasdaq_screener, NASDAQ100_TICKERS,
    load_watchlist, save_watchlist, get_watchlist_quote,
    load_watchlist_verdicts, update_watchlist_verdict,
    get_trending_stocks,
    get_economic_calendar,
    get_chart_patterns,
)
from agent import (
    analyze_for_dashboard, handle_chat_query, summarize_transcript,
    summarize_market_news, summarize_portfolio_impact, summarize_screener_results,
    analyze_calendar_portfolio_impact,
)

# ── optional TradingView TA ──────────────────────────────────────────────────
try:
    from tradingview_ta import TA_Handler, Interval
    TV_TA_OK = True
except ImportError:
    TV_TA_OK = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinanceGPT | לוח בקרה פיננסי",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# CSS — Bloomberg dark + RTL Hebrew
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── RTL + base ── */
html, body { direction: rtl; }
.stApp, .block-container,
[data-testid="stAppViewContainer"] {
    background-color: #0B0E11 !important;
    color: #C9D1D9 !important;
    direction: rtl;
}

/* ── sidebar RTL ── */
[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-left: 1px solid #21262D !important;
    border-right: none !important;
    direction: rtl;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stSidebar"] .stTextInput input { text-align: right !important; }

/* ── header ── */
header[data-testid="stHeader"] { background: #0B0E11 !important; }

/* ── metric card ── */
.fin-card {
    background: linear-gradient(135deg,#161B22,#0D1117);
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 18px 20px 14px;
    text-align: center;
    direction: rtl;
}
.fin-card .lbl { font-size:11px; font-weight:600; letter-spacing:1px;
                 color:#8B949E; margin-bottom:4px; }
.fin-card .val { font-size:26px; font-weight:700; color:#E6EDF3;
                 direction: ltr; display: inline-block; }
.fin-card .dlta { font-size:13px; font-weight:600; margin-top:4px; }
.fin-card .up  { color:#3FB950; }
.fin-card .dn  { color:#F85149; }
.fin-card .neu { color:#8B949E; }

/* ── tabs RTL ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1117;
    border-bottom: 1px solid #21262D;
    flex-direction: row-reverse;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8B949E;
    border-radius: 6px 6px 0 0;
    padding: 10px 18px; font-weight:600; font-size:13px;
    direction: rtl;
}
.stTabs [aria-selected="true"] {
    background: #161B22 !important;
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
}

/* ── verdict block (השורה התחתונה) ── */
.verdict-block {
    background: linear-gradient(145deg,#0d1117,#161b22,#0d1117);
    border: 2px solid #F0B90B;
    border-radius: 16px;
    padding: 0;
    margin-bottom: 20px;
    direction: rtl;
    text-align: right;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(240,185,11,0.15);
}
.verdict-header {
    background: linear-gradient(90deg,#1a1500,#2a2000);
    border-bottom: 1px solid #F0B90B44;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-direction: row-reverse;
}
.verdict-label {
    font-size:10px; letter-spacing:2px; font-weight:700;
    color:#F0B90B; text-transform:uppercase;
}
.verdict-stamp {
    font-size:10px; color:#8B949E; font-family:monospace;
}
.verdict-body {
    padding: 20px 26px 24px;
}
.verdict-title {
    font-size:26px; font-weight:800; margin-bottom:6px;
}
.verdict-pattern {
    font-size:12px; color:#8B949E; margin-bottom:14px;
    font-style:italic; letter-spacing:0.5px;
}
.verdict-text {
    font-size:14px; color:#C9D1D9; line-height:2.0;
    border-top: 1px solid #21262D; padding-top:14px;
}
.conflict-banner {
    background: linear-gradient(90deg,#2d1b1b,#1a0f0f);
    border: 1px solid #F85149;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 12px 0;
    font-size:13px; color:#FF9492; direction:rtl; text-align:right;
}
.confidence-bar-wrap {
    margin: 14px 0 4px;
}
.confidence-label {
    font-size:11px; color:#8B949E; margin-bottom:5px;
}
.confidence-bar-bg {
    background:#21262D; border-radius:20px; height:8px; width:100%; overflow:hidden;
}
.confidence-bar-fill {
    height:8px; border-radius:20px;
    transition: width 0.5s ease;
}
.v-bull { color:#3FB950; }
.v-bear { color:#F85149; }
.v-neut { color:#D29922; }

/* ── insight panel ── */
.ins-panel {
    background: #161B22;
    border-right: 4px solid #58A6FF;
    border-left: none;
    border-radius: 8px 0 0 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    direction: rtl; text-align: right;
}
.ins-panel.red  { border-right-color: #F85149; }
.ins-panel.ylw  { border-right-color: #D29922; }
.ins-panel.grn  { border-right-color: #3FB950; }
.ins-panel .ins-title { font-size:14px; font-weight:700;
                        color:#E6EDF3; margin-bottom:6px; }
.ins-panel .ins-body  { font-size:13px; color:#8B949E; line-height:1.7; }

/* ── tech table RTL ── */
.tech-tbl { width:100%; border-collapse:collapse; font-size:13px;
            direction:rtl; }
.tech-tbl th { background:#161B22; color:#58A6FF; padding:8px 12px;
               text-align:right; font-weight:600;
               border-bottom:1px solid #21262D; }
.tech-tbl td { padding:8px 12px; border-bottom:1px solid #21262D;
               color:#C9D1D9; text-align:right; }
.tech-tbl td:nth-child(2) { direction:ltr; text-align:left; }
.tech-tbl tr:hover td { background:#161B22; }
.bull { color:#3FB950; font-weight:700; }
.bear { color:#F85149; font-weight:700; }
.neut { color:#D29922; font-weight:700; }

/* ── badges ── */
.rec-badge {
    display:inline-block; padding:4px 16px; border-radius:20px;
    font-size:13px; font-weight:700; letter-spacing:1px;
}
.rec-BUY   { background:#1a3a2a; color:#3FB950; border:1px solid #3FB950; }
.rec-SELL  { background:#3a1a1a; color:#F85149; border:1px solid #F85149; }
.rec-HOLD  { background:#3a3010; color:#D29922; border:1px solid #D29922; }
.rec-WATCH { background:#1a2a3a; color:#58A6FF; border:1px solid #58A6FF; }
.sent-BULLISH { color:#3FB950; font-size:20px; font-weight:700; }
.sent-BEARISH { color:#F85149; font-size:20px; font-weight:700; }
.sent-NEUTRAL { color:#D29922; font-size:20px; font-weight:700; }

/* ── news expander RTL ── */
.stExpander { background:#161B22 !important; border:1px solid #21262D !important;
              border-radius:8px !important; margin-bottom:6px !important; }
.stExpander summary { color:#C9D1D9 !important; font-size:13px !important;
                      direction:rtl !important; text-align:right !important; }

/* ── sidebar button ── */
div[data-testid="stButton"] button {
    background:#161B22; border:1px solid #21262D; color:#C9D1D9;
    border-radius:6px; font-size:12px; width:100%; transition:.15s;
}
div[data-testid="stButton"] button:hover {
    border-color:#58A6FF; color:#58A6FF; background:#0D1117;
}

/* ── input RTL ── */
input[type="text"], .stTextInput input {
    background:#161B22 !important; color:#E6EDF3 !important;
    border:1px solid #21262D !important; border-radius:6px !important;
    text-align: right !important;
}
::placeholder { text-align: right; }

/* ── keep charts LTR ── */
.stPlotlyChart, iframe, .js-plotly-plot { direction: ltr !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0B0E11; }
::-webkit-scrollbar-thumb { background:#21262D; border-radius:3px; }

hr { border-color:#21262D !important; }
.stSpinner > div { border-top-color:#58A6FF !important; }

/* ── youtube section ── */
.yt-loaded {
    background: linear-gradient(135deg,#1a2e1a,#0d1117);
    border: 1px solid #3FB950;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #3FB950;
    text-align: right;
    direction: rtl;
    margin-top: 6px;
}

/* ── chat tab ── */
[data-testid="stChatMessage"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    margin-bottom: 10px;
    padding: 6px 4px;
}
[data-testid="stChatMessage"][data-kind="user"] {
    background: #1a2235;
    border-color: #58A6FF44;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div {
    direction: rtl !important;
    text-align: right !important;
    color: #C9D1D9 !important;
    font-size: 14px !important;
    line-height: 1.85 !important;
}
[data-testid="stChatMessage"] .stMarkdown {
    direction: rtl !important;
}
/* Chat input box */
[data-testid="stChatInput"] > div {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea {
    color: #E6EDF3 !important;
    direction: rtl !important;
    text-align: right !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    text-align: right;
    color: #8B949E !important;
}
/* chat welcome banner */
.chat-welcome {
    background: linear-gradient(135deg,#161B22,#0d1117);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 20px;
}

/* ── watchlist card ── */
.wl-card {
    background: linear-gradient(135deg,#161B22,#0D1117);
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 8px 10px 6px;
    margin-bottom: 2px;
    direction: rtl;
}
.wl-ticker { font-size:13px; font-weight:700; color:#58A6FF; }
.wl-name   { font-size:9px;  color:#8B949E; margin-top:1px; }
.wl-price  { font-size:12px; font-weight:700; color:#E6EDF3; direction:ltr; }
.wl-chg-up { font-size:11px; font-weight:700; color:#3FB950; }
.wl-chg-dn { font-size:11px; font-weight:700; color:#F85149; }
.wl-chg-ne { font-size:11px; color:#8B949E; }

/* ── star-toggle button override ── */
div[data-testid="stButton"] button.wl-star {
    background: transparent !important;
    border: 1px solid #F0B90B55 !important;
    color: #F0B90B !important;
    font-size:13px !important;
    padding: 4px 14px !important;
    border-radius: 20px !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
WATCHLIST = ["SOFI", "IBIT", "ETHA", "NVDA", "AAPL"]
ANALYSIS_VERSION = "v10-5step-framework"  # bump to bust analysis cache on prompt changes

_YF_TO_TV = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "NYQ": "NYSE",   "PCX": "NYSE",
    "ASE": "AMEX",   "TLV": "TASE",
    "NASDAQ": "NASDAQ", "NYSE": "NYSE",
}

# Hebrew translation maps
HE_SENTIMENT = {"BULLISH": "שורי", "BEARISH": "דובי", "NEUTRAL": "ניטרלי"}
HE_REC       = {"BUY": "קנייה", "SELL": "מכירה", "HOLD": "המתנה", "WATCH": "מעקב"}
HE_TV_REC    = {
    "STRONG_BUY":  "קנייה חזקה",
    "BUY":         "קנייה",
    "NEUTRAL":     "ניטרלי",
    "SELL":        "מכירה",
    "STRONG_SELL": "מכירה חזקה",
}
HE_DIR       = {"up": "▲", "dn": "▼", "neu": ""}

# Industry → geopolitical/macro query string (used in both Tab 1 and Tab 2)
_GEO_QUERY_MAP: dict[str, str] = {
    "Semiconductors":                 "AI chips semiconductor export controls China trade war",
    "Semiconductor Equipment":        "semiconductor supply chain China export controls ASML",
    "Software—Application":           "AI software regulation antitrust big tech",
    "Software—Infrastructure":        "cloud computing AI regulation cybersecurity",
    "Consumer Electronics":           "China tariffs supply chain Apple Samsung",
    "Oil & Gas E&P":                  "oil prices OPEC Middle East energy geopolitics",
    "Oil & Gas Integrated":           "oil prices OPEC Middle East energy geopolitics",
    "Oil & Gas Midstream":            "oil prices energy infrastructure sanctions",
    "Oil & Gas Equipment":            "energy sector geopolitics oil prices",
    "Banks—Regional":                 "Federal Reserve interest rates banking crisis",
    "Banks—Diversified":              "Federal Reserve interest rates banking regulation",
    "Drug Manufacturers":             "pharmaceutical regulation FDA drug pricing",
    "Biotechnology":                  "biotech FDA regulation clinical trials",
    "Aerospace & Defense":            "defense spending Middle East conflict geopolitics",
    "Automobiles":                    "electric vehicles China tariffs auto supply chain",
    "Auto Manufacturers":             "electric vehicles China tariffs auto supply chain",
    "Internet Content & Information": "AI regulation antitrust big tech China",
    "Telecom Services":               "5G spectrum regulation China telecom",
}


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def tv_symbol(ticker: str, yf_exch: str = "") -> str:
    if ticker.upper().endswith(".TA"):
        return "TASE:" + ticker.upper().replace(".TA", "")
    return f"{_YF_TO_TV.get(yf_exch, 'NASDAQ')}:{ticker.upper()}"


def tv_screener(ticker: str, yf_exch: str = "") -> str:
    return "israel" if ticker.upper().endswith(".TA") else "america"


def tv_exchange_str(ticker: str, yf_exch: str = "") -> str:
    return "TASE" if ticker.upper().endswith(".TA") else _YF_TO_TV.get(yf_exch, "NASDAQ")


def fmt_price(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"


def fmt_pct(v) -> str:
    try:
        f = float(v)
        return f"{'+'if f>=0 else ''}{f:.2f}%"
    except Exception:
        return "—"


def fmt_vol(v) -> str:
    try:
        v = float(v)
        if v >= 1e9: return f"{v/1e9:.2f}B"
        if v >= 1e6: return f"{v/1e6:.2f}M"
        if v >= 1e3: return f"{v/1e3:.0f}K"
        return str(int(v))
    except Exception:
        return "—"


def classify_news(headline: str) -> tuple[str, str]:
    lo = headline.lower()
    if any(w in lo for w in ["plunge","crash","collapse","crisis","miss","warn","plummet"]):
        return "🔴", "bear"
    if any(w in lo for w in ["surge","rally","record","beat","soar","jump","upgrade","boom"]):
        return "🟢", "bull"
    if any(w in lo for w in ["breaking","alert","shock","unexpected","surprise","flash"]):
        return "🔥", "bull"
    if any(w in lo for w in ["dividend","buyback","stable","maintain","hold","resilient"]):
        return "🛡️", "neut"
    if any(w in lo for w in ["risk","concern","caution","probe","investigate","sanction"]):
        return "⚠️", "bear"
    return "📰", "neut"


def metric_card(label: str, value: str, delta: str = "", delta_dir: str = "neu") -> str:
    arrow = HE_DIR.get(delta_dir, "")
    delta_html = (
        f'<div class="dlta {delta_dir}">{arrow} {delta}</div>'
        if delta else ""
    )
    return (
        f'<div class="fin-card">'
        f'<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'{delta_html}</div>'
    )


# ════════════════════════════════════════════════════════════════════════════
# CACHED DATA FETCHERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(ticker: str) -> dict:
    return get_stock_data(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(query: str, n: int = 8) -> dict:
    return get_financial_news(query, n)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ta(ticker: str, screener: str, exchange: str) -> dict | None:
    if not TV_TA_OK:
        return None
    try:
        clean = ticker.upper().replace(".TA", "")
        h = TA_Handler(symbol=clean, screener=screener,
                       exchange=exchange, interval=Interval.INTERVAL_1_DAY)
        a = h.get_analysis()
        return {"summary": a.summary, "indicators": a.indicators}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fibonacci(ticker: str) -> dict:
    return calculate_fibonacci_levels(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(ticker: str) -> dict:
    return run_forecasting_models(ticker, forecast_days=30)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quant(ticker: str) -> dict:
    return get_quant_metrics(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quarterly_earnings(ticker: str) -> dict:
    return get_quarterly_earnings(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_data() -> dict:
    return get_macro_data()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_overview() -> dict:
    return get_market_indices()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_watchlist_quote(ticker: str) -> dict:
    return get_watchlist_quote(ticker)


@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache — trending data
def fetch_trending_stocks() -> list:
    return get_trending_stocks(top_n=5)


@st.cache_data(ttl=3600, show_spinner=False)  # 1-hour cache — calendar events
def fetch_economic_calendar() -> dict:
    return get_economic_calendar(days_ahead=90)


@st.cache_data(ttl=1800, show_spinner=False)  # 30-min cache — AI calendar insight
def fetch_calendar_insight(events_json: str) -> str:
    import json as _json
    events = _json.loads(events_json)
    return analyze_calendar_portfolio_impact(events)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_dashboard_news() -> list[str]:
    """Fetch general market/geo headlines for the Market Dashboard tab."""
    headlines: list[str] = []
    for q in ("market", "geopolitics Middle East Federal Reserve"):
        result = get_financial_news(q, 6)
        for a in result.get("articles", []):
            t = a.get("title", "").strip()
            if t and t not in headlines:
                headlines.append(t)
    return headlines[:12]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_summary(headlines_json: str) -> str:
    """Cached AI Hebrew summary of market headlines (key = headline content)."""
    headlines = json.loads(headlines_json)
    return summarize_market_news(headlines)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_portfolio_snapshot() -> dict:
    """Fetch live prices for the 6 core portfolio holdings."""
    return get_portfolio_snapshot()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_portfolio_impact_summary(portfolio_json: str, market_json: str,
                                   headlines_json: str) -> str:
    """Cached AI portfolio impact analysis (keyed by live data snapshots)."""
    return summarize_portfolio_impact(
        json.loads(portfolio_json),
        json.loads(market_json),
        json.loads(headlines_json),
    )


@st.cache_data(ttl=21600, show_spinner=False)   # 6-hour cache — end-of-day data
def fetch_screener(min_rs: int, max_beta: float, min_sharpe: float,
                   above_sma50: bool, min_rsi: float, max_rsi: float) -> list:
    """Run NASDAQ 100 screener; cached per filter combination for 6 hours."""
    return run_nasdaq_screener(min_rs, max_beta, min_sharpe, above_sma50, min_rsi, max_rsi)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analysis(ticker: str, version: str,
                   _stock_json: str = "", _news_json: str = "",
                   _ta_json: str = "", _video_json: str = "",
                   _fib_json: str = "", _forecast_json: str = "",
                   _quant_json: str = "", _chart_json: str = "") -> dict:
    """Cache keyed by ticker + version only — data params excluded (prefix _).
    This prevents a new Sonnet call on every 5-min price tick."""
    return analyze_for_dashboard(
        json.loads(_stock_json)    if _stock_json    else {},
        json.loads(_news_json)     if _news_json     else [],
        json.loads(_ta_json)       if _ta_json       else None,
        json.loads(_video_json)    if _video_json    else None,
        json.loads(_fib_json)      if _fib_json      else None,
        json.loads(_forecast_json) if _forecast_json else None,
        json.loads(_quant_json)    if _quant_json    else None,
        json.loads(_chart_json)    if _chart_json    else None,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chart_patterns(ticker: str) -> dict:
    """Cached 1-year chart pattern analysis (peaks, valleys, MA series, patterns)."""
    return get_chart_patterns(ticker)


# ════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW CHART WIDGET
# ════════════════════════════════════════════════════════════════════════════

def render_tv_chart(symbol: str, height: int = 540) -> None:
    components.html(f"""
    <div id="tv_chart" style="width:100%;height:{height}px;direction:ltr"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
        "container_id":"tv_chart","autosize":true,
        "symbol":"{symbol}","interval":"D",
        "timezone":"America/New_York","theme":"dark","style":"1","locale":"en",
        "toolbar_bg":"#0D1117","withdateranges":true,
        "hide_side_toolbar":false,"allow_symbol_change":false,
        "studies":["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],
        "overrides":{{
            "paneProperties.background":"#0B0E11",
            "paneProperties.vertGridProperties.color":"#21262D",
            "paneProperties.horzGridProperties.color":"#21262D",
            "scalesProperties.textColor":"#8B949E"
        }}
    }});
    </script>
    """, height=height + 20, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════
# MA CHART — Plotly price + SMA_44 / SMA_150 / SMA_200
# ════════════════════════════════════════════════════════════════════════════

def render_ma_chart(chart_data: dict) -> None:
    """
    Render a Plotly line chart with SMA_44 (blue), SMA_150 (gold), SMA_200 (gray).
    Annotates detected patterns (Double Top/Bottom peaks, Cup rim level).
    """
    if not chart_data or "error" in chart_data:
        st.info("נתוני גרף ממוצעים נעים אינם זמינים.", icon="ℹ️")
        return

    dates      = chart_data.get("dates", [])
    closes     = chart_data.get("closes", [])
    sma44_s    = chart_data.get("sma44_series", [])
    sma150_s   = chart_data.get("sma150_series", [])
    sma200_s   = chart_data.get("sma200_series", [])
    patterns   = chart_data.get("detected_patterns", [])

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        name="מחיר",
        line=dict(color="#E6EDF3", width=1.5),
        hovertemplate="%{x}<br>${%{y:,.2f}}<extra></extra>",
    ))

    # SMA_44 — blue
    if any(v is not None for v in sma44_s):
        fig.add_trace(go.Scatter(
            x=dates, y=sma44_s,
            mode="lines",
            name="ממוצע 44",
            line=dict(color="#58A6FF", width=1.5, dash="solid"),
            hovertemplate="MA44: $%{y:,.2f}<extra></extra>",
        ))

    # SMA_150 — gold
    if any(v is not None for v in sma150_s):
        fig.add_trace(go.Scatter(
            x=dates, y=sma150_s,
            mode="lines",
            name="ממוצע 150",
            line=dict(color="#F0B90B", width=2, dash="solid"),
            hovertemplate="MA150: $%{y:,.2f}<extra></extra>",
        ))

    # SMA_200 — gray
    if any(v is not None for v in sma200_s):
        fig.add_trace(go.Scatter(
            x=dates, y=sma200_s,
            mode="lines",
            name="ממוצע 200",
            line=dict(color="#8B949E", width=1.2, dash="dot"),
            hovertemplate="MA200: $%{y:,.2f}<extra></extra>",
        ))

    # Pattern annotations
    _ann_color = {"bullish": "#3FB950", "bearish": "#F85149"}
    for p in patterns:
        level = p.get("level")
        if level:
            color = _ann_color.get(p.get("type", ""), "#D29922")
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash="dash"),
                annotation_text=p.get("name_he", ""),
                annotation_position="right",
                annotation_font=dict(color=color, size=10),
            )
        for ann in p.get("annotations", []):
            fig.add_annotation(
                x=ann["date"], y=ann["price"],
                text=ann["label"],
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=_ann_color.get(p.get("type", ""), "#D29922"),
                font=dict(color="#E6EDF3", size=11),
                bgcolor="#161B22",
                bordercolor=_ann_color.get(p.get("type", ""), "#D29922"),
                borderwidth=1,
            )

    fig.update_layout(
        height=340,
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0D1117",
        font=dict(color="#C9D1D9", size=11),
        margin=dict(l=10, r=140, t=30, b=30),
        legend=dict(
            orientation="h",
            x=0, y=1.08,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="#21262D",
            tickformat="%b '%y",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#21262D",
            tickprefix="$",
            tickformat=",.0f",
            side="right",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Pattern pills below chart
    if patterns:
        _pill_color = {"bullish": ("#3FB950", "#0d2a18"), "bearish": ("#F85149", "#2a0d0d")}
        pills_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;direction:rtl">'
        for p in patterns:
            fc, bg = _pill_color.get(p.get("type", ""), ("#D29922", "#2a1f00"))
            pills_html += (
                f'<span style="font-size:11px;background:{bg};color:{fc};'
                f'border:1px solid {fc}55;border-radius:5px;padding:3px 10px;font-weight:600">'
                f'{p["name_he"]}</span>'
            )
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ════════════════════════════════════════════════════════════════════════════

def rsi_gauge(rsi: float) -> go.Figure:
    rsi = rsi or 50.0
    color = "#F85149" if rsi > 70 else "#3FB950" if rsi < 30 else "#D29922"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rsi,
        number={"font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B949E",
                     "tickfont": {"color": "#8B949E", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#161B22", "bordercolor": "#21262D",
            "steps": [
                {"range": [0, 30],   "color": "#1a3a2a"},
                {"range": [30, 70],  "color": "#1a1f28"},
                {"range": [70, 100], "color": "#3a1a1a"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": rsi},
        },
        title={"text": "RSI (14)", "font": {"color": "#8B949E", "size": 13}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
                      margin=dict(t=30, b=10, l=20, r=20), height=220)
    return fig


def ta_rec_bar(summary: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["קנייה", "ניטרלי", "מכירה"],
        y=[summary.get("BUY", 0), summary.get("NEUTRAL", 0), summary.get("SELL", 0)],
        marker_color=["#3FB950", "#D29922", "#F85149"],
        text=[summary.get("BUY", 0), summary.get("NEUTRAL", 0), summary.get("SELL", 0)],
        textposition="outside",
        textfont={"color": "#C9D1D9", "size": 13},
    ))
    fig.update_layout(
        paper_bgcolor="#0B0E11", plot_bgcolor="#161B22",
        font={"color": "#C9D1D9"},
        xaxis={"gridcolor": "#21262D"},
        yaxis={"gridcolor": "#21262D"},
        margin=dict(t=20, b=20, l=10, r=10),
        height=220, showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 4px;direction:rtl">'
            '<span style="font-size:28px">📈</span><br>'
            '<span style="font-size:20px;font-weight:700;color:#E6EDF3">FinanceGPT</span><br>'
            '<span style="font-size:11px;color:#8B949E;letter-spacing:1px">'
            'מערכת ניתוח פיננסי חכמה</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if "ticker" not in st.session_state:
            st.session_state.ticker = ""

        # ── רשימת המעקב שלי ─────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#F0B90B;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">⭐ רשימת המעקב שלי</p>',
            unsafe_allow_html=True,
        )

        _watchlist = st.session_state.get("watchlist", [])
        if not _watchlist:
            st.markdown(
                '<p style="font-size:11px;color:#8B949E;text-align:right;'
                'margin-bottom:8px">אין מניות במועדפים.<br>'
                'לחץ ⭐ בלשונית ניתוח מניה להוספה.</p>',
                unsafe_allow_html=True,
            )
        else:
            _verdicts = st.session_state.get("watchlist_verdicts", {})
            _rec_he_map    = {"BUY": "קנייה", "SELL": "מכירה", "HOLD": "המתנה", "WATCH": "מעקב"}
            _rec_color_map = {
                "BUY":   ("#3FB950", "#0d2a18"),
                "SELL":  ("#F85149", "#2a0d0d"),
                "HOLD":  ("#D29922", "#2a1f00"),
                "WATCH": ("#58A6FF", "#0d1a2a"),
            }

            for _sym in _watchlist:
                _q = fetch_watchlist_quote(_sym)
                _chg    = _q.get("change_pct")
                _price  = _q.get("price")
                _name   = (_q.get("company_name") or _sym)[:22]
                _spk    = _q.get("sparkline", [])
                _err    = "error" in _q

                # ── verdict metadata ──────────────────────────────────────────
                _vmeta   = _verdicts.get(_sym, {})
                _vrec    = _vmeta.get("rec", "")
                _vconf   = _vmeta.get("confidence")
                _vrec_he = _rec_he_map.get(_vrec, "")
                _vc, _vbg = _rec_color_map.get(_vrec, ("#8B949E", "#1a1a1a"))

                _verdict_html = ""
                if _vrec_he:
                    _conf_str = f" · {_vconf}/10" if _vconf is not None else ""
                    _verdict_html = (
                        f'<div style="display:flex;align-items:center;gap:5px;'
                        f'margin-top:5px;justify-content:flex-end" '
                        f'title="ציון אחרון מהסוכן">'
                        f'<span style="font-size:9px;background:{_vbg};color:{_vc};'
                        f'border:1px solid {_vc}55;border-radius:4px;'
                        f'padding:1px 6px;font-weight:700;letter-spacing:.4px">'
                        f'{_vrec_he}</span>'
                        f'<span style="font-size:9px;color:#8B949E">{_conf_str}</span>'
                        f'</div>'
                    )

                if not _err and _price is not None:
                    _arrow     = "▲ " if _chg and _chg >= 0 else "▼ " if _chg else ""
                    _chg_str   = f"{_arrow}{abs(_chg):.2f}%" if _chg is not None else "—"
                    _chg_cls   = "wl-chg-up" if _chg and _chg >= 0 else "wl-chg-dn" if _chg else "wl-chg-ne"
                    _spk_color = "#3FB950" if _chg and _chg >= 0 else "#F85149"
                    _spk_svg   = _sparkline_svg(_spk, _spk_color, 70, 22) if len(_spk) >= 2 else ""
                    st.markdown(
                        f'<div class="wl-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div><div class="wl-ticker">{_sym}</div>'
                        f'<div class="wl-name">{_name}</div></div>'
                        f'<div style="text-align:left">'
                        f'<div class="wl-price">${_price:,.2f}</div>'
                        f'<div class="{_chg_cls}">{_chg_str}</div></div>'
                        f'</div>'
                        f'<div style="direction:ltr;margin-top:3px">{_spk_svg}</div>'
                        f'{_verdict_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="wl-card">'
                        f'<div class="wl-ticker">{_sym}</div>'
                        f'<div class="wl-name" style="color:#555">טוען…</div>'
                        f'{_verdict_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if st.button(f"→ נתח {_sym}", key=f"wl_{_sym}", use_container_width=True):
                    st.session_state.ticker          = _sym
                    st.session_state.ticker_selected  = True
                    st.session_state._nav_to_analysis = True
                    st.rerun()

        st.markdown("---")

        # ── טיקר מותאם אישית ────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">🔍 טיקר מותאם אישית</p>',
            unsafe_allow_html=True,
        )
        # st.form ensures Enter key AND button click both capture the text value
        with st.form("ticker_form", clear_on_submit=True):
            custom = st.text_input("", placeholder="לדוגמה: TSLA, MSFT, TA35.TA",
                                   label_visibility="collapsed")
            submitted = st.form_submit_button("→ נתח", use_container_width=True)
            if submitted and custom.strip():
                st.session_state.ticker           = custom.strip().upper()
                st.session_state.ticker_selected  = True
                st.session_state._nav_to_analysis = True

        _cur_ticker = st.session_state.ticker
        _in_wl      = _cur_ticker in st.session_state.get("watchlist", [])
        _star_lbl   = "⭐ הסר מהמועדפים" if _in_wl else "☆ הוסף למועדפים"
        if st.button(_star_lbl, key="wl_toggle", use_container_width=True):
            _wl = list(st.session_state.get("watchlist", []))
            if _in_wl:
                _wl = [t for t in _wl if t != _cur_ticker]
            else:
                _wl.append(_cur_ticker)
            st.session_state.watchlist = _wl
            save_watchlist(_wl)
            fetch_watchlist_quote.clear()
            st.rerun()

        st.markdown("---")
        ticker = st.session_state.ticker

        # ── דירוג טכני TradingView ───────────────────────────────────────────
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">📡 דירוג טכני — TradingView</p>',
            unsafe_allow_html=True,
        )
        if TV_TA_OK:
            _sd   = fetch_stock(ticker)
            _exch = _sd.get("exchange", "") if "error" not in _sd else ""
            _ta   = fetch_ta(ticker, tv_screener(ticker, _exch),
                             tv_exchange_str(ticker, _exch))
            if _ta:
                rec = _ta["summary"].get("RECOMMENDATION", "NEUTRAL")
                rec_he = HE_TV_REC.get(rec, "ניטרלי")
                rec_color = {
                    "STRONG_BUY": "#3FB950", "BUY": "#3FB950",
                    "NEUTRAL":    "#D29922",
                    "SELL":       "#F85149", "STRONG_SELL": "#F85149",
                }.get(rec, "#8B949E")
                st.markdown(
                    f'<div style="text-align:center;background:#161B22;'
                    f'border:1px solid #21262D;border-radius:8px;padding:12px;">'
                    f'<div style="font-size:22px;font-weight:700;color:{rec_color}">'
                    f'{rec_he}</div>'
                    f'<div style="font-size:11px;color:#8B949E;margin-top:4px">'
                    f'ק:{_ta["summary"].get("BUY",0)} '
                    f'נ:{_ta["summary"].get("NEUTRAL",0)} '
                    f'מ:{_ta["summary"].get("SELL",0)}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("נתוני TA אינם זמינים עבור טיקר זה")
        else:
            st.caption("התקן tradingview-ta לדירוגים")

        st.markdown("---")

        # ── נתח מחדש ────────────────────────────────────────────────────────
        if st.button("🔄 נתח מחדש", use_container_width=True):
            fetch_stock.clear()
            fetch_news.clear()
            fetch_ta.clear()
            fetch_fibonacci.clear()
            fetch_forecast.clear()
            fetch_quant.clear()
            fetch_quarterly_earnings.clear()
            fetch_macro_data.clear()
            fetch_market_overview.clear()
            fetch_dashboard_news.clear()
            fetch_news_summary.clear()
            fetch_portfolio_snapshot.clear()
            fetch_portfolio_impact_summary.clear()
            fetch_screener.clear()
            fetch_analysis.clear()
            fetch_chart_patterns.clear()
            st.rerun()

        st.markdown(
            f'<p style="font-size:10px;color:#8B949E;text-align:center;margin-top:8px">'
            f'עדכון אחרון: {datetime.now().strftime("%H:%M:%S")}</p>',
            unsafe_allow_html=True,
        )

    return st.session_state.ticker


# ════════════════════════════════════════════════════════════════════════════
# HEADER — כותרת + כרטיסי מדדים
# ════════════════════════════════════════════════════════════════════════════

def render_header(sd: dict) -> None:
    name   = sd.get("company_name", sd.get("ticker", ""))
    ticker = sd.get("ticker", "")
    sector = sd.get("sector", "")
    exch   = sd.get("exchange", "")

    st.markdown(
        f'<div style="padding:4px 0 2px;direction:rtl;text-align:right">'
        f'<span style="font-size:28px;font-weight:700;color:#E6EDF3">{ticker}</span>'
        f' <span style="font-size:18px;color:#8B949E;margin-right:8px">{name}</span>'
        f'<span style="float:left;font-size:12px;color:#8B949E;padding-top:10px">'
        f'{sector} · {exch}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    price      = fmt_price(sd.get("current_price"))
    sma50_pct  = sd.get("price_vs_sma50_pct", 0) or 0
    price_dir  = "up" if sma50_pct >= 0 else "dn"
    vol_ratio  = sd.get("volume_ratio")
    vol_dir    = "up" if vol_ratio and vol_ratio > 1 else "dn" if vol_ratio and vol_ratio < 0.7 else "neu"
    rsi        = sd.get("rsi_14")
    rsi_dir    = "dn" if rsi and rsi > 70 else "up" if rsi and rsi < 30 else "neu"
    rsi_lbl    = "קנוי יתר" if rsi and rsi > 70 else "מכור יתר" if rsi and rsi < 30 else "ניטרלי"
    ed         = str(sd.get("next_earnings_date", "—") or "—")[:10]

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "מחיר נוכחי",       price,
              f"{fmt_pct(sma50_pct)} מול SMA-50", price_dir),
        (c2, "שווי שוק",          sd.get("market_cap_human") or "—", "", "neu"),
        (c3, "RSI (14)",           f"{rsi:.1f}" if rsi else "—", rsi_lbl, rsi_dir),
        (c4, "מחזור מסחר",        fmt_vol(sd.get("last_volume")),
              f"{vol_ratio:.2f}x ממוצע" if vol_ratio else "", vol_dir),
        (c5, "דוח רווחים הבא",   ed, "", "neu"),
    ]
    for col, lbl, val, delta, ddir in cards:
        with col:
            st.markdown(metric_card(lbl, val, delta, ddir),
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS — Market Overview sparkline + card
# ════════════════════════════════════════════════════════════════════════════

def _sparkline_svg(values: list, color: str = "#3FB950",
                   width: int = 88, height: int = 32) -> str:
    """Render a filled-area sparkline as an inline SVG string."""
    if not values or len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1e-9
    pad = 3  # vertical padding so the line isn't clipped
    pts = []
    for i, v in enumerate(values):
        x = round(i / (len(values) - 1) * width, 1)
        y = round(height - pad - ((v - vmin) / rng) * (height - 2 * pad), 1)
        pts.append((x, y))
    line_pts  = " ".join(f"{x},{y}" for x, y in pts)
    area_pts  = (f"0,{height} "
                 + " ".join(f"{x},{y}" for x, y in pts)
                 + f" {width},{height}")
    fill_rgba = color + "28"   # ~16 % opacity fill
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"'
        f' style="display:block;overflow:visible">'
        f'<polygon points="{area_pts}" fill="{fill_rgba}"/>'
        f'<polyline points="{line_pts}" fill="none" stroke="{color}"'
        f' stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _market_card_html(name: str, price_str: str, change_pct,
                      sparkline: list, badge: str = "") -> str:
    """Return Bloomberg-style market card HTML with sparkline."""
    if change_pct is None:
        chg_color, chg_str, arrow, spk_color = "#8B949E", "—", "", "#8B949E"
    elif float(change_pct) >= 0:
        chg_color = "#3FB950"
        chg_str   = f"+{float(change_pct):.2f}%"
        arrow, spk_color = "▲ ", "#3FB950"
    else:
        chg_color = "#F85149"
        chg_str   = f"{float(change_pct):.2f}%"
        arrow, spk_color = "▼ ", "#F85149"

    spk = _sparkline_svg(sparkline, color=spk_color) if sparkline else ""
    badge_html = (
        f'<span style="font-size:9px;background:{chg_color}22;color:{chg_color};'
        f'border:1px solid {chg_color}55;border-radius:4px;padding:1px 5px;'
        f'margin-right:4px;letter-spacing:.5px">{badge}</span>'
        if badge else ""
    )
    return (
        f'<div style="background:linear-gradient(140deg,#161B22 60%,#0D1117);'
        f'border:1px solid #21262D;border-radius:10px;padding:13px 15px 10px;'
        f'direction:rtl;text-align:right;min-height:96px">'
        f'<div style="font-size:10px;letter-spacing:.9px;color:#8B949E;'
        f'margin-bottom:4px;display:flex;align-items:center;justify-content:flex-end;'
        f'gap:4px">{badge_html}{name}</div>'
        f'<div style="font-size:20px;font-weight:900;color:#E6EDF3;'
        f'letter-spacing:-.3px;direction:ltr;text-align:left;margin-bottom:5px">'
        f'{price_str}</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-end">'
        f'<span style="font-size:13px;font-weight:700;color:{chg_color}">'
        f'{arrow}{chg_str}</span>'
        f'<span style="direction:ltr;line-height:0">{spk}</span>'
        f'</div>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════════════════════
# HELPERS — Market Clock
# ════════════════════════════════════════════════════════════════════════════

def _market_clock_html() -> str:
    """Return HTML for NYSE and TASE open/closed status with local times."""
    now_utc = datetime.now(timezone.utc)
    month   = now_utc.month

    # US DST: second Sunday of March → first Sunday of November (rough: month 3–11)
    us_dst  = 3 <= month <= 11
    # Israel DST: last Sunday of March → last Sunday of October (rough: month 3–10)
    il_dst  = 3 <= month <= 10

    et_offset  = timedelta(hours=-4 if us_dst else -5)
    ist_offset = timedelta(hours=3  if il_dst else 2)

    now_et  = now_utc + et_offset
    now_ist = now_utc + ist_offset

    et_hour  = now_et.hour  + now_et.minute  / 60
    ist_hour = now_ist.hour + now_ist.minute / 60

    # NYSE: Mon–Fri 9:30–16:00 ET
    nyse_open = (now_et.weekday() < 5) and (9.5 <= et_hour < 16.0)
    # Pre/post-market bands
    nyse_pre  = (now_et.weekday() < 5) and (4.0 <= et_hour < 9.5)
    nyse_post = (now_et.weekday() < 5) and (16.0 <= et_hour < 20.0)

    # TASE: Sun–Thu 9:45–17:30 IST (weekday 6=Sun in Python? No, 0=Mon)
    # Python: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    tase_open = (now_ist.weekday() in (6, 0, 1, 2, 3)) and (9.75 <= ist_hour < 17.5)

    def _clock_badge(label: str, time_str: str, is_open: bool,
                     note: str = "", tz_label: str = "") -> str:
        color  = "#3FB950" if is_open else "#F85149"
        status = "פתוח" if is_open else "סגור"
        if not is_open and note:
            status = note
        dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;' \
              f'background:{color};margin-left:6px;animation:{"pulse 1.5s infinite" if is_open else "none"}"></span>'
        return (
            f'<div style="background:#161B22;border:1px solid #21262D;border-radius:10px;'
            f'padding:12px 16px;text-align:center;direction:rtl;min-width:130px">'
            f'<div style="font-size:10px;letter-spacing:1px;color:#8B949E;margin-bottom:4px">'
            f'{label}</div>'
            f'<div style="font-size:20px;font-weight:800;color:#E6EDF3;direction:ltr;'
            f'text-align:center;letter-spacing:1px">{time_str}</div>'
            f'<div style="margin-top:5px;display:flex;align-items:center;justify-content:center">'
            f'{dot}'
            f'<span style="font-size:12px;font-weight:700;color:{color}">{status}</span></div>'
            f'<div style="font-size:10px;color:#8B949E;margin-top:2px">{tz_label}</div>'
            f'</div>'
        )

    nyse_time = now_et.strftime("%H:%M")
    tase_time = now_ist.strftime("%H:%M")
    nyse_note = "פרי-מרקט" if nyse_pre else ("פוסט-מרקט" if nyse_post else "סגור")

    pulse_css = (
        '<style>@keyframes pulse{'
        '0%{opacity:1}50%{opacity:.35}100%{opacity:1}}</style>'
    )

    nyse_badge = _clock_badge("NYSE — ניו יורק", nyse_time, nyse_open,
                              nyse_note, "ET (ניו יורק)")
    tase_badge = _clock_badge("TASE — תל אביב", tase_time, tase_open,
                              "סגור", "IST (ישראל)")

    utc_str = now_utc.strftime("%d/%m %H:%M UTC")
    return (
        pulse_css
        + f'<div style="display:flex;gap:12px;align-items:stretch;'
        f'flex-wrap:wrap;margin-bottom:18px;direction:rtl">'
        f'<div style="font-size:10px;color:#8B949E;align-self:flex-end;margin-bottom:4px">'
        f'⏱ {utc_str}</div>'
        f'<div style="flex:1;min-width:130px">{nyse_badge}</div>'
        f'<div style="flex:1;min-width:130px">{tase_badge}</div>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — לוח מחוונים (Market Dashboard — landing page)
# ════════════════════════════════════════════════════════════════════════════

def _portfolio_asset_card(ticker: str, data: dict) -> str:
    """Compact asset card for the portfolio daily summary row."""
    label = data.get("label", ticker)
    if "error" in data:
        return (
            f'<div style="background:#161B22;border:1px solid #21262D;border-radius:8px;'
            f'padding:10px 12px;text-align:center;min-width:100px">'
            f'<div style="font-size:10px;color:#58A6FF;font-weight:700;margin-bottom:3px">{ticker}</div>'
            f'<div style="font-size:11px;color:#8B949E">{label}</div>'
            f'<div style="font-size:16px;color:#8B949E;margin-top:4px">—</div>'
            f'</div>'
        )
    price  = data.get("price")
    chg    = data.get("change_pct")
    direc  = data.get("direction", "flat")
    color  = "#3FB950" if direc == "up" else "#F85149" if direc == "down" else "#8B949E"
    arrow  = "▲" if direc == "up" else "▼" if direc == "down" else "—"
    p_str  = f"${float(price):,.2f}" if price is not None else "—"
    c_str  = f"{arrow} {abs(float(chg)):.2f}%" if chg is not None else "—"
    return (
        f'<div style="background:linear-gradient(140deg,#161B22,#0D1117);'
        f'border:1px solid #21262D;border-top:2px solid {color};border-radius:8px;'
        f'padding:10px 12px;text-align:center;min-width:100px">'
        f'<div style="font-size:10px;color:#58A6FF;font-weight:700;letter-spacing:.8px;'
        f'margin-bottom:2px">{ticker}</div>'
        f'<div style="font-size:9px;color:#8B949E;margin-bottom:5px">{label}</div>'
        f'<div style="font-size:17px;font-weight:800;color:#E6EDF3;direction:ltr">{p_str}</div>'
        f'<div style="font-size:12px;font-weight:700;color:{color};margin-top:3px">{c_str}</div>'
        f'</div>'
    )


def tab_market_overview(market_data: dict) -> None:
    # ════════════════════════════════════════════════════════════════════════
    # PORTFOLIO DAILY SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;direction:rtl">'
        '<span style="font-size:16px">💼</span>'
        '<span style="font-size:11px;letter-spacing:1.8px;font-weight:700;color:#58A6FF;'
        'text-transform:uppercase">סיכום יומי של התיק</span>'
        '<div style="flex:1;height:1px;background:#58A6FF33;margin-right:8px"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    portfolio_snap = fetch_portfolio_snapshot()

    # ── 6 asset mini-cards ────────────────────────────────────────────────────
    p_cols = st.columns(len(PORTFOLIO_TICKERS))
    for col, ticker in zip(p_cols, PORTFOLIO_TICKERS):
        data = portfolio_snap.get(ticker, {"error": "no data"})
        with col:
            st.markdown(_portfolio_asset_card(ticker, data), unsafe_allow_html=True)

    # ── AI impact summary ─────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    dash_headlines = fetch_dashboard_news()
    indices_data   = (market_data or {}).get("indices", {})

    with st.spinner("🤖 מנתח השפעת השוק על התיק…"):
        impact_text = fetch_portfolio_impact_summary(
            json.dumps(portfolio_snap,  default=str),
            json.dumps({"indices": indices_data}, default=str),
            json.dumps(dash_headlines[:8]),
        )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0D1626,#0B0E11);'
        f'border:1px solid #1f3a5f;border-right:3px solid #58A6FF;'
        f'border-radius:0 10px 10px 0;padding:16px 20px 14px;direction:rtl;'
        f'text-align:right;font-size:13px;color:#C9D1D9;line-height:2.1">'
        f'<div style="font-size:10px;letter-spacing:1.5px;color:#58A6FF;'
        f'font-weight:700;margin-bottom:10px">🤖 ניתוח AI — השפעת השוק על התיק שלך היום</div>'
        f'{impact_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="border-color:#21262D;margin:20px 0">', unsafe_allow_html=True)

    # ── Market Clock ──────────────────────────────────────────────────────────
    st.markdown(_market_clock_html(), unsafe_allow_html=True)

    indices = (market_data or {}).get("market_data", {}).get("indices") or \
              (market_data or {}).get("indices", {})
    if not indices:
        st.info("נתוני שוק אינם זמינים כרגע.", icon="📡")
        return

    us_idx  = {k: v for k, v in indices.items() if v.get("category") == "us"}
    il_idx  = {k: v for k, v in indices.items() if v.get("category") == "israel"}
    crypto  = {k: v for k, v in indices.items() if v.get("category") == "crypto"}

    # ── Market Mood ───────────────────────────────────────────────────────────
    mood_vals = []
    for name, item in {**us_idx, **crypto}.items():
        if "error" in item or name == "VIX":
            continue
        chg = item.get("change_pct")
        if chg is not None:
            mood_vals.append(float(chg))

    vix_item = us_idx.get("VIX", {})
    vix_val  = vix_item.get("price") if "error" not in vix_item else None

    avg_chg = sum(mood_vals) / len(mood_vals) if mood_vals else 0

    if avg_chg > 1.2:
        m_icon, m_text, m_color, m_bg = "🔥", "שוק שורי חזק", "#3FB950", "#0D1F14"
    elif avg_chg > 0.3:
        m_icon, m_text, m_color, m_bg = "📈", "נטייה שורית",   "#3FB950", "#0D1F14"
    elif avg_chg > -0.3:
        m_icon, m_text, m_color, m_bg = "⚖️", "שוק מעורב",    "#D29922", "#1A1500"
    elif avg_chg > -1.2:
        m_icon, m_text, m_color, m_bg = "📉", "נטייה דובית",   "#F85149", "#1F0D0D"
    else:
        m_icon, m_text, m_color, m_bg = "🚨", "שוק דובי חזק", "#F85149", "#1F0D0D"

    # VIX override: fear spike
    if vix_val and float(vix_val) > 30:
        m_icon, m_text, m_color, m_bg = "😱", "פאניקה — VIX גבוה", "#F85149", "#1F0D0D"

    ts = (market_data or {}).get("timestamp", "")[:16].replace("T", " ")
    vix_str = f"VIX {vix_val:.1f}" if vix_val is not None else ""

    st.markdown(
        f'<div style="background:{m_bg};border:1px solid {m_color}44;border-radius:12px;'
        f'padding:14px 22px;margin-bottom:18px;display:flex;align-items:center;'
        f'justify-content:space-between;direction:rtl">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:26px">{m_icon}</span>'
        f'<div>'
        f'<div style="font-size:11px;letter-spacing:1.5px;color:#8B949E;font-weight:600">'
        f'מצב השוק</div>'
        f'<div style="font-size:20px;font-weight:800;color:{m_color}">{m_text}</div>'
        f'</div></div>'
        f'<div style="text-align:left">'
        f'<div style="font-size:12px;color:{m_color};font-weight:700">'
        f'{"+" if avg_chg >= 0 else ""}{avg_chg:.2f}% ממוצע</div>'
        f'<div style="font-size:11px;color:#8B949E;margin-top:2px">'
        f'{vix_str}{"  ·  " if vix_str else ""}{ts}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    def _fmt_price(v, symbol=""):
        if v is None:
            return "—"
        f = float(v)
        if symbol in ("^VIX", "^TNX"):
            return f"{f:.2f}"
        if f >= 10_000:
            return f"${f:,.0f}"
        if f >= 1_000:
            return f"${f:,.2f}"
        return f"${f:.2f}"

    def _section_header(icon, title, color):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:18px 0 10px;'
            f'direction:rtl">'
            f'<span style="font-size:16px">{icon}</span>'
            f'<span style="font-size:11px;letter-spacing:1.8px;font-weight:700;'
            f'color:{color};text-transform:uppercase">{title}</span>'
            f'<div style="flex:1;height:1px;background:{color}33;margin-right:8px"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── US Indices ────────────────────────────────────────────────────────────
    _section_header("🇺🇸", "מדדי ארה״ב", "#58A6FF")
    us_names  = ["S&P 500", "Nasdaq", "Dow Jones", "Russell 2000", "VIX"]
    us_cols   = st.columns(len(us_names))
    us_badges = {"S&P 500": "SPX", "Nasdaq": "NDX", "Dow Jones": "DJIA",
                 "Russell 2000": "RUT", "VIX": "FEAR"}
    for col, name in zip(us_cols, us_names):
        item = us_idx.get(name, {})
        if "error" in item:
            with col:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:10px;padding:13px;text-align:center;color:#8B949E;'
                    f'font-size:12px;min-height:96px">{name}<br>—</div>',
                    unsafe_allow_html=True)
            continue
        price_str = _fmt_price(item.get("price"), item.get("symbol", ""))
        with col:
            st.markdown(
                _market_card_html(name, price_str, item.get("change_pct"),
                                  item.get("sparkline", []), us_badges.get(name, "")),
                unsafe_allow_html=True,
            )

    # ── Israel Indices ─────────────────────────────────────────────────────────
    _section_header("🇮🇱", "מדדי ישראל", "#4CAF50")

    # Dynamic — show all indices with category=="israel", in order
    _IL_BADGES = {
        "TA-35":         "TA35",
        "TA-90":         "TA90",
        "TA-125":        "TA125",
        "מדד בנקים-5":   "BK35",
        "ביטוח":         "INS",
    }
    il_names = [k for k in _IL_BADGES if k in il_idx]
    if not il_names:
        il_names = list(il_idx.keys())   # fallback: whatever came back

    il_cols = st.columns(max(len(il_names), 1))
    for col, name in zip(il_cols, il_names):
        item = il_idx.get(name, {})
        with col:
            if "error" in item:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:10px;padding:13px;text-align:center;'
                    f'color:#8B949E;font-size:12px;min-height:96px">'
                    f'{name}<br>—</div>',
                    unsafe_allow_html=True,
                )
            else:
                price_str = _fmt_price(item.get("price"), item.get("symbol", ""))
                st.markdown(
                    _market_card_html(name, price_str,
                                      item.get("change_pct"), item.get("sparkline", []),
                                      _IL_BADGES.get(name, "")),
                    unsafe_allow_html=True,
                )

    # ── Crypto ────────────────────────────────────────────────────────────────
    _section_header("₿", "שוק הקריפטו", "#F0B90B")

    btc_item = crypto.get("Bitcoin", {})
    eth_item = crypto.get("Ethereum", {})

    cr_col1, cr_col2, cr_sp1, cr_sp2, cr_sp3 = st.columns([1, 1, 1, 1, 1])

    for col, name, item, badge in [
        (cr_col1, "Bitcoin",  btc_item, "₿ BTC"),
        (cr_col2, "Ethereum", eth_item, "Ξ ETH"),
    ]:
        if "error" in item:
            with col:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:10px;padding:13px;text-align:center;color:#8B949E;'
                    f'font-size:12px;min-height:96px">{name}<br>—</div>',
                    unsafe_allow_html=True)
            continue
        price_str = _fmt_price(item.get("price"))
        with col:
            # Crypto cards get a golden border accent
            chg = item.get("change_pct")
            chg_color = "#3FB950" if chg and float(chg) >= 0 else "#F85149" if chg else "#8B949E"
            st.markdown(
                f'<div style="border-top:2px solid #F0B90B">'
                + _market_card_html(name, price_str, chg,
                                    item.get("sparkline", []), badge)
                + f'</div>',
                unsafe_allow_html=True,
            )

    # ── 7-day performance comparison bar ─────────────────────────────────────
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    perf_items = []
    for n in ["S&P 500", "Nasdaq", "TA-35", "TA-90", "TA-125", "מדד בנקים-5", "ביטוח", "Bitcoin", "Ethereum"]:
        item = indices.get(n, {})
        if "error" not in item and item.get("change_pct") is not None:
            perf_items.append((n, float(item["change_pct"])))

    if perf_items:
        fig = go.Figure(go.Bar(
            x=[p[0] for p in perf_items],
            y=[p[1] for p in perf_items],
            marker_color=["#3FB950" if p[1] >= 0 else "#F85149" for p in perf_items],
            text=[f"{'+' if p[1] >= 0 else ''}{p[1]:.2f}%" for p in perf_items],
            textposition="outside",
            textfont={"color": "#C9D1D9", "size": 12},
        ))
        fig.update_layout(
            paper_bgcolor="#0B0E11", plot_bgcolor="#161B22",
            font={"color": "#C9D1D9"},
            xaxis={"gridcolor": "#21262D"},
            yaxis={"gridcolor": "#21262D", "title": "שינוי יומי %",
                   "zeroline": True, "zerolinecolor": "rgba(88,166,255,0.25)"},
            margin=dict(t=30, b=20, l=50, r=20),
            height=220, showlegend=False,
            title={"text": "ביצועים יומיים — השוואה", "font": {"color": "#8B949E", "size": 12},
                   "x": 1, "xanchor": "right"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Geopolitical & Macro News Section ─────────────────────────────────────
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;direction:rtl">'
        '<span style="font-size:16px">🌍</span>'
        '<span style="font-size:11px;letter-spacing:1.8px;font-weight:700;'
        'color:#D29922;text-transform:uppercase">חדשות גיאופוליטיות ומאקרו</span>'
        '<div style="flex:1;height:1px;background:#D2992233;margin-right:8px"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    headlines = fetch_dashboard_news()
    if headlines:
        # Show headlines list
        for h in headlines[:7]:
            icon, cls = classify_news(h)
            color_map = {"bull": "#3FB950", "bear": "#F85149", "neut": "#8B949E"}
            c = color_map.get(cls, "#8B949E")
            st.markdown(
                f'<div style="background:#161B22;border-right:3px solid {c};'
                f'border-radius:0 6px 6px 0;padding:7px 12px 7px 10px;'
                f'margin-bottom:5px;direction:rtl;text-align:right;'
                f'font-size:12px;color:#C9D1D9">'
                f'{icon} {h}</div>',
                unsafe_allow_html=True,
            )

        # AI Hebrew summary
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;letter-spacing:1.5px;font-weight:700;'
            'color:#58A6FF;text-align:right;margin-bottom:8px">'
            '🤖 סיכום AI — ניתוח סנטימנט שוק (עברית)</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("מסכם חדשות עם AI…"):
            summary_text = fetch_news_summary(json.dumps(headlines[:12]))

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0D1626,#0B0E11);'
            f'border:1px solid #58A6FF33;border-right:3px solid #58A6FF;'
            f'border-radius:0 10px 10px 0;padding:16px 20px;direction:rtl;'
            f'text-align:right;font-size:13px;color:#C9D1D9;line-height:2.0">'
            f'{summary_text.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("לא נמצאו כותרות חדשות כרגע.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — גרף אינטראקטיבי
# ════════════════════════════════════════════════════════════════════════════

def tab_chart(ticker: str, sd: dict) -> None:
    symbol = tv_symbol(ticker, sd.get("exchange", ""))
    st.markdown(
        f'<p style="font-size:12px;color:#8B949E;margin-bottom:6px;'
        f'direction:rtl;text-align:right">'
        f'מוצג: <code style="color:#58A6FF">{symbol}</code> · '
        f'כולל RSI, MACD ורצועות בולינגר</p>',
        unsafe_allow_html=True,
    )
    render_tv_chart(symbol, height=560)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ניתוח AI  (with verdict block)
# ════════════════════════════════════════════════════════════════════════════

def _debate_card(icon: str, title: str, body: str, border_color: str, bg_color: str) -> str:
    """Return HTML for a single debate-panel card."""
    return (
        f'<div style="background:{bg_color};border:1px solid {border_color};'
        f'border-top:3px solid {border_color};border-radius:10px;padding:18px 20px;'
        f'direction:rtl;text-align:right;height:100%">'
        f'<div style="font-size:12px;font-weight:700;letter-spacing:1.2px;'
        f'color:{border_color};margin-bottom:10px">{icon} {title}</div>'
        f'<div style="font-size:13px;color:#C9D1D9;line-height:1.85">'
        f'{body.replace(chr(10), "<br>")}</div>'
        f'</div>'
    )


def _quant_tile(label: str, value: str, sub: str, color: str) -> str:
    """Single metric tile for the institutional Risk Dashboard."""
    return (
        f'<div style="background:#161B22;border:1px solid #21262D;border-top:3px solid {color};'
        f'border-radius:10px;padding:14px 16px;text-align:center;direction:rtl">'
        f'<div style="font-size:10px;letter-spacing:1.2px;color:#8B949E;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:22px;font-weight:900;color:{color};line-height:1.1">{value}</div>'
        f'<div style="font-size:11px;color:#8B949E;margin-top:4px">{sub}</div>'
        f'</div>'
    )


def tab_ai_insights(analysis: dict, sd: dict, quant_data: dict | None = None) -> None:
    from datetime import datetime

    # ── pull all fields ───────────────────────────────────────────────────────
    sentiment     = analysis.get("overall_sentiment", "NEUTRAL")
    rec_eng       = analysis.get("recommendation", "HOLD")
    verdict_he    = analysis.get("verdict_hebrew", HE_SENTIMENT.get(sentiment, "ניטרלי"))
    targets       = analysis.get("price_targets", {})

    summary       = analysis.get("executive_summary", "")
    tech_pattern  = analysis.get("technical_pattern", "")
    fund_health   = analysis.get("fundamental_health", "")

    confidence    = int(analysis.get("confidence_score") or 1)
    support       = analysis.get("support_level", 0) or 0
    resistance    = analysis.get("resistance_level", 0) or 0
    rr_ratio      = analysis.get("risk_reward_ratio", "N/A")
    has_conflict  = bool(analysis.get("technical_conflict", False))
    conflict_text = analysis.get("conflict_explanation", "")

    # ── Judicial Multi-Agent fields ───────────────────────────────────────────
    bull_thesis   = analysis.get("bull_thesis", "")
    bear_thesis   = analysis.get("bear_thesis", "")
    quant_audit   = analysis.get("quant_audit", "")
    judge_verdict = analysis.get("judge_verdict", "")
    pos_sizing    = analysis.get("position_sizing", "")
    trade_plan    = analysis.get("trade_plan", {}) or {}
    technical_map = analysis.get("technical_map", "")

    sent_he     = HE_SENTIMENT.get(sentiment, "ניטרלי")
    rec_he      = HE_REC.get(rec_eng, "המתנה")
    verdict_css = {"שורי": "v-bull", "דובי": "v-bear"}.get(verdict_he, "v-neut")
    sent_icon   = {"שורי": "🐂", "דובי": "🐻"}.get(verdict_he, "⚖️")
    conf_color  = "#3FB950" if confidence >= 7 else "#D29922" if confidence >= 4 else "#F85149"
    ts          = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 0 — TECHNICAL MAP (5-step summary at the top)
    # ════════════════════════════════════════════════════════════════════════
    if technical_map:
        map_lines = technical_map.strip().replace("\\n", "\n").split("\n")
        map_html_rows = "".join(
            f'<div style="padding:5px 0;border-bottom:1px solid #21262D;font-size:14px;'
            f'line-height:1.6;direction:rtl;text-align:right">{line.strip()}</div>'
            for line in map_lines if line.strip()
        )
        st.markdown(
            f'<div style="background:#0D1117;border:1px solid #30363D;border-radius:10px;'
            f'padding:16px 20px;margin-bottom:18px">'
            f'<p style="font-size:11px;letter-spacing:1.8px;font-weight:700;color:#8B949E;'
            f'text-align:right;margin:0 0 10px 0">🗺️ TECHNICAL MAP — סיכום 5 שלבים</p>'
            f'{map_html_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — THE DEBATE (2×2 grid)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.8px;font-weight:700;color:#8B949E;'
        'text-align:right;margin-bottom:10px">⚖️ פרוטוקול שיפוטי רב-סוכני</p>',
        unsafe_allow_html=True,
    )

    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.markdown(
            _debate_card("🐂", "הסנגור — טיעון שורי", bull_thesis,
                         "#3FB950", "#0D1F14"),
            unsafe_allow_html=True,
        )
    with row1_right:
        st.markdown(
            _debate_card("🐻", "הקטגור — טיעון דובי", bear_thesis,
                         "#F85149", "#1F0D0D"),
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.markdown(
            _debate_card("🔬", "ביקורת נתונים — Quant Audit", quant_audit,
                         "#58A6FF", "#0D1626"),
            unsafe_allow_html=True,
        )
    with row2_right:
        # Judge card placeholder — full verdict rendered below
        st.markdown(
            _debate_card("⚖️", "פסק הדין — השופט", judge_verdict,
                         "#D29922", "#1A1500"),
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1.5 — INSTITUTIONAL RISK DASHBOARD
    # ════════════════════════════════════════════════════════════════════════
    qd = quant_data if (quant_data and "error" not in quant_data) else {}

    def _fmt_signed(v, suffix=""):
        if v is None:
            return "—"
        return f"+{v:.1f}{suffix}" if v >= 0 else f"{v:.1f}{suffix}"

    # RS Rating color
    rs_rating = qd.get("rs_rating")
    rs_color  = (
        "#3FB950" if rs_rating and rs_rating >= 70
        else "#F85149" if rs_rating and rs_rating < 40
        else "#D29922"
    )

    # Beta color
    beta_v    = qd.get("beta_60d")
    beta_color = (
        "#F85149" if beta_v and beta_v > 1.5
        else "#D29922" if beta_v and beta_v > 1.0
        else "#3FB950"
    )

    # Sharpe color
    sharpe_v  = qd.get("sharpe_ratio")
    sharpe_color = (
        "#3FB950" if sharpe_v and sharpe_v > 1.0
        else "#F85149" if sharpe_v and sharpe_v < 0
        else "#D29922"
    )

    # RS 50d excess return color
    rs_50d_v  = qd.get("rs_50d")
    rs50_color = "#3FB950" if rs_50d_v and rs_50d_v > 0 else "#F85149"

    # Sector ETF performance
    sec_vs_spy = qd.get("sector_vs_spy_50d")
    sec_color  = "#3FB950" if sec_vs_spy and sec_vs_spy > 0 else "#F85149"
    etf_sym    = qd.get("sector_etf") or "—"

    poc_v = qd.get("point_of_control")
    price_now = sd.get("current_price", 0) or 0
    poc_sub = "—"
    poc_color = "#D29922"
    if poc_v and price_now:
        poc_diff_pct = (price_now - poc_v) / poc_v * 100
        poc_sub  = f"מחיר {'מעל' if poc_diff_pct >= 0 else 'מתחת'} POC {abs(poc_diff_pct):.1f}%"
        poc_color = "#3FB950" if poc_diff_pct >= 0 else "#F85149"

    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.8px;font-weight:700;color:#8B949E;'
        'text-align:right;margin-bottom:8px">📊 לוח בקרה מוסדי — Risk Dashboard</p>',
        unsafe_allow_html=True,
    )

    rd1, rd2, rd3, rd4, rd5, rd6 = st.columns(6)
    tiles = [
        (rd1, _quant_tile("RS RATING",    f"{rs_rating}"                if rs_rating  else "—",
                          "ביצוע יחסי (1-99)",    rs_color)),
        (rd2, _quant_tile("RS 50D",       _fmt_signed(rs_50d_v, "%"),
                          "עודף תשואה vs SPY",    rs50_color)),
        (rd3, _quant_tile("BETA (60D)",   f"{beta_v:.2f}"               if beta_v     else "—",
                          "תנודתיות vs SPY",       beta_color)),
        (rd4, _quant_tile("SHARPE",       f"{sharpe_v:.2f}"             if sharpe_v   else "—",
                          "תשואה מותאמת סיכון",    sharpe_color)),
        (rd5, _quant_tile("POC",          f"${poc_v:,.2f}"              if poc_v      else "—",
                          poc_sub,                 poc_color)),
        (rd6, _quant_tile(f"סקטור ({etf_sym})",
                          _fmt_signed(sec_vs_spy, "%")                  if sec_vs_spy else "—",
                          "ביצוע ETF vs SPY 50D",  sec_color)),
    ]
    for col, html in tiles:
        with col:
            st.markdown(html, unsafe_allow_html=True)

    # ── Row 2: Ownership & Short Interest (from stock data) ───────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    short_v   = sd.get("short_percent_float")
    inst_v    = sd.get("inst_ownership")
    insider_v = sd.get("insider_ownership")

    def _pct_display(v):
        if v is None:
            return "—"
        return f"{float(v) * 100:.1f}%"

    short_color  = "#F85149" if short_v and float(short_v) > 0.15 else "#D29922" if short_v and float(short_v) > 0.05 else "#3FB950"
    inst_color   = "#3FB950" if inst_v and float(inst_v) > 0.50 else "#D29922"
    insider_color = "#D29922"

    ow1, ow2, ow3, _sp1, _sp2, _sp3 = st.columns(6)
    ownership_tiles = [
        (ow1, _quant_tile("SHORT FLOAT",     _pct_display(short_v),
                          "ריבית שורט",       short_color)),
        (ow2, _quant_tile("INST. OWNERSHIP", _pct_display(inst_v),
                          "אחזקות מוסדיות",   inst_color)),
        (ow3, _quant_tile("INSIDER HOLD.",   _pct_display(insider_v),
                          "אחזקות פנימיות",   insider_color)),
    ]
    for col, html in ownership_tiles:
        with col:
            st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — JUDGE'S VERDICT (prominent box)
    # ════════════════════════════════════════════════════════════════════════
    conflict_html = (
        f'<div class="conflict-banner">⚡ <strong>קונפליקט:</strong> {conflict_text}</div>'
        if has_conflict and conflict_text else ""
    )
    pattern_line = (
        f'<div class="verdict-pattern">📐 {tech_pattern}</div>' if tech_pattern else ""
    )

    # Trade plan row
    entry_p   = trade_plan.get("entry", 0) or 0
    target_p  = trade_plan.get("target", 0) or 0
    sl_p      = trade_plan.get("stop_loss", 0) or 0
    trade_html = ""
    if entry_p or target_p or sl_p:
        trade_html = (
            f'<div style="display:flex;gap:16px;margin-top:14px;flex-wrap:wrap;justify-content:flex-end">'
            f'<div style="background:#0D1F14;border:1px solid #3FB950;border-radius:8px;'
            f'padding:8px 14px;text-align:center;min-width:90px">'
            f'<div style="font-size:10px;color:#8B949E;margin-bottom:2px">כניסה</div>'
            f'<div style="font-size:16px;font-weight:700;color:#3FB950">${entry_p:,.2f}</div></div>'
            f'<div style="background:#0D1626;border:1px solid #58A6FF;border-radius:8px;'
            f'padding:8px 14px;text-align:center;min-width:90px">'
            f'<div style="font-size:10px;color:#8B949E;margin-bottom:2px">יעד</div>'
            f'<div style="font-size:16px;font-weight:700;color:#58A6FF">${target_p:,.2f}</div></div>'
            f'<div style="background:#1F0D0D;border:1px solid #F85149;border-radius:8px;'
            f'padding:8px 14px;text-align:center;min-width:90px">'
            f'<div style="font-size:10px;color:#8B949E;margin-bottom:2px">סטופ לוס</div>'
            f'<div style="font-size:16px;font-weight:700;color:#F85149">${sl_p:,.2f}</div></div>'
            f'</div>'
        )

    pos_html = (
        f'<div style="margin-top:12px;padding:10px 14px;background:rgba(210,153,34,0.08);'
        f'border-radius:8px;font-size:13px;color:#D29922;direction:rtl;text-align:right">'
        f'💼 גודל פוזיציה: <strong>{pos_sizing}</strong></div>'
        if pos_sizing else ""
    )

    verdict_color = {"שורי": "#3FB950", "דובי": "#F85149"}.get(verdict_he, "#D29922")
    verdict_border = {"שורי": "#3FB950", "דובי": "#F85149"}.get(verdict_he, "#D29922")

    st.markdown(
        f'<div style="background:#161B22;border:2px solid {verdict_border};'
        f'border-radius:14px;padding:22px 24px;direction:rtl;text-align:right;'
        f'box-shadow:0 0 20px {verdict_border}33">'
        f'  <div style="display:flex;justify-content:space-between;align-items:center;'
        f'flex-wrap:wrap;gap:8px;margin-bottom:14px">'
        f'    <span style="font-size:11px;font-weight:700;letter-spacing:1.5px;'
        f'color:#8B949E">🏛️ פסק הדין הסופי — FinanceGPT · {ts}</span>'
        f'    <span style="font-size:28px;font-weight:900;color:{verdict_color}">'
        f'{sent_icon} {verdict_he}</span>'
        f'  </div>'
        f'  {pattern_line}'
        f'  <div style="display:flex;align-items:center;gap:12px;margin:10px 0">'
        f'    <div style="flex:1;background:#21262D;border-radius:4px;height:8px">'
        f'      <div style="width:{confidence*10}%;height:8px;border-radius:4px;'
        f'background:{conf_color}"></div></div>'
        f'    <span style="font-size:13px;color:{conf_color};font-weight:700;white-space:nowrap">'
        f'ביטחון {confidence}/10</span>'
        f'    <span style="font-size:12px;color:#8B949E">|</span>'
        f'    <span style="font-size:12px;color:#58A6FF;font-weight:600">'
        f'ס/ס {rr_ratio}</span>'
        f'  </div>'
        f'  {conflict_html}'
        f'  {pos_html}'
        f'  {trade_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — 4-metric strip (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1.1, 1.6])

    with sc1:
        st.markdown(
            f'<div style="text-align:center;background:#161B22;border:1px solid #21262D;'
            f'border-radius:10px;padding:16px;direction:rtl">'
            f'<div style="font-size:32px">{sent_icon}</div>'
            f'<div class="sent-{sentiment}">{sent_he}</div>'
            f'<div style="font-size:11px;color:#8B949E;margin-top:4px">סנטימנט</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with sc2:
        rec_icon = {"קנייה": "📈", "מכירה": "📉"}.get(rec_he, "📊")
        st.markdown(
            f'<div style="text-align:center;background:#161B22;border:1px solid #21262D;'
            f'border-radius:10px;padding:16px;direction:rtl">'
            f'<div style="font-size:32px">{rec_icon}</div>'
            f'<span class="rec-badge rec-{rec_eng}">{rec_he}</span>'
            f'<div style="font-size:11px;color:#8B949E;margin-top:8px">המלצה</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with sc3:
        price   = sd.get("current_price", 0) or 0
        sup_pct = f"↓{((price-support)/price*100):.1f}%" if price and support else "—"
        res_pct = f"↑{((resistance-price)/price*100):.1f}%" if price and resistance else "—"
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #21262D;border-radius:10px;'
            f'padding:16px;direction:rtl;text-align:right">'
            f'<div style="font-size:11px;color:#58A6FF;font-weight:600;margin-bottom:10px">'
            f'📏 תמיכה / התנגדות</div>'
            f'<div style="font-size:13px;margin-bottom:6px">'
            f'<span style="color:#8B949E">התנגדות: </span>'
            f'<span style="color:#F85149;font-weight:700">${resistance:,.2f}</span>'
            f'<small style="color:#8B949E"> ({res_pct})</small></div>'
            f'<div style="font-size:13px;">'
            f'<span style="color:#8B949E">תמיכה: </span>'
            f'<span style="color:#3FB950;font-weight:700">${support:,.2f}</span>'
            f'<small style="color:#8B949E"> ({sup_pct})</small></div>'
            f'<div style="margin-top:8px;font-size:11px;color:#8B949E">'
            f'יחס ס/ס: <strong style="color:#58A6FF">{rr_ratio}</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with sc4:
        bull     = targets.get("bull", 0) or 0
        base     = targets.get("base", 0) or 0
        bear_t   = targets.get("bear", 0) or 0
        upside   = f"+{((bull-price)/price*100):.1f}%"   if price and bull   else "—"
        downside = f"{((bear_t-price)/price*100):.1f}%" if price and bear_t else "—"
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #21262D;'
            f'border-radius:10px;padding:16px;direction:rtl;text-align:right">'
            f'<div style="font-size:11px;letter-spacing:1px;color:#58A6FF;'
            f'font-weight:600;margin-bottom:10px">🎯 יעדי מחיר</div>'
            f'<table style="width:100%;font-size:13px">'
            f'<tr><td style="color:#8B949E">תרחיש אופטימי</td>'
            f'<td style="color:#3FB950;font-weight:700;text-align:left">'
            f'${bull:,.2f} <small style="color:#8B949E">({upside})</small></td></tr>'
            f'<tr><td style="color:#8B949E">תרחיש בסיס</td>'
            f'<td style="color:#D29922;font-weight:700;text-align:left">${base:,.2f}</td></tr>'
            f'<tr><td style="color:#8B949E">תרחיש פסימי</td>'
            f'<td style="color:#F85149;font-weight:700;text-align:left">'
            f'${bear_t:,.2f} <small style="color:#8B949E">({downside})</small></td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Fundamental health + SMA + Summary + Risks
    # ════════════════════════════════════════════════════════════════════════
    if tech_pattern or fund_health:
        col_a, col_b = st.columns(2)
        if tech_pattern:
            with col_a:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:8px;padding:16px;direction:rtl;text-align:right;height:100%">'
                    f'<div style="font-size:11px;color:#58A6FF;font-weight:600;'
                    f'letter-spacing:1px;margin-bottom:8px">📐 דפוס טכני מזוהה</div>'
                    f'<div style="font-size:15px;color:#E6EDF3;font-weight:700">'
                    f'{tech_pattern}</div></div>',
                    unsafe_allow_html=True,
                )
        if fund_health:
            with col_b:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:8px;padding:16px;direction:rtl;text-align:right;height:100%">'
                    f'<div style="font-size:11px;color:#3FB950;font-weight:600;'
                    f'letter-spacing:1px;margin-bottom:8px">💼 בריאות פונדמנטלית</div>'
                    f'<div style="font-size:13px;color:#C9D1D9;line-height:1.8">'
                    f'{fund_health}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    if summary:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-align:right">📋 סיכום מנהלים</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #21262D;border-radius:8px;'
            f'padding:18px;font-size:14px;line-height:1.8;color:#C9D1D9;'
            f'direction:rtl;text-align:right">'
            f'{summary.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ניתוח טכני
# ════════════════════════════════════════════════════════════════════════════

def tab_technicals(sd: dict, ta_data: dict | None, analysis: dict,
                   quarterly_earnings: dict | None = None,
                   chart_data: dict | None = None) -> None:
    # ── MA Chart — price + SMA_44 / SMA_150 / SMA_200 ────────────────────────
    if chart_data and "error" not in chart_data:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#F0B90B;'
            'font-weight:600;text-align:right;margin-bottom:4px">'
            '📈 גרף מחיר עם ממוצעים נעים — SMA 44 · SMA 150 · SMA 200</p>',
            unsafe_allow_html=True,
        )
        render_ma_chart(chart_data)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1.4, 1, 1.2])

    with left:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">📊 מדדים מרכזיים</p>',
            unsafe_allow_html=True,
        )
        price  = sd.get("current_price", 0) or 0
        sma50  = sd.get("sma_50")
        sma200 = sd.get("sma_200")
        rsi    = sd.get("rsi_14")
        beta   = sd.get("beta")
        vol_r  = sd.get("volume_ratio")

        macd = macd_sig = None
        if ta_data and ta_data.get("indicators"):
            ind      = ta_data["indicators"]
            macd     = ind.get("MACD.macd")
            macd_sig = ind.get("MACD.signal")

        def signal_html(val, ref, higher_good=True) -> str:
            if val is None or ref is None:
                return '<span class="neut">—</span>'
            cls = "bull" if (val > ref) == higher_good else "bear"
            return f'<span class="{cls}">{"↑" if val > ref else "↓"}</span>'

        # SMA_44 and SMA_150 from chart_data (preferred) or fall back to sma_50/200
        sma44  = chart_data.get("sma_44")  if chart_data and "error" not in chart_data else None
        sma150 = chart_data.get("sma_150") if chart_data and "error" not in chart_data else None
        d44    = chart_data.get("dist_to_44_pct")  if chart_data else None
        d150   = chart_data.get("dist_to_150_pct") if chart_data else None

        def _dist_badge(dist):
            if dist is None:
                return ""
            color = "#3FB950" if dist > 2 else "#F85149" if dist < -2 else "#D29922"
            sign  = "+" if dist >= 0 else ""
            return f'<span style="color:{color};font-weight:600">{sign}{dist:.1f}%</span>'

        rows = [
            ("מחיר נוכחי",       fmt_price(price), ""),
            ("SMA-44",            fmt_price(sma44)  if sma44  else fmt_price(sma50),
             _dist_badge(d44) or signal_html(price, sma50)),
            ("SMA-150",           fmt_price(sma150) if sma150 else fmt_price(sma200),
             _dist_badge(d150) or signal_html(price, sma200)),
            ("SMA-50",            fmt_price(sma50),
             signal_html(price, sma50)),
            ("SMA-200",           fmt_price(sma200),
             signal_html(price, sma200)),
            ("RSI (14)",          f"{rsi:.2f}" if rsi else "—",
             f'<span class="{"bear" if rsi and rsi>70 else "bull" if rsi and rsi<30 else "neut"}">'
             f'{"קנוי יתר" if rsi and rsi>70 else "מכור יתר" if rsi and rsi<30 else "ניטרלי"}'
             f'</span>' if rsi else ""),
            ("MACD",              f"{macd:.4f}" if macd is not None else "—", ""),
            ("אות MACD",          f"{macd_sig:.4f}" if macd_sig is not None else "—",
             f'<span class="{"bull" if macd and macd_sig and macd>macd_sig else "bear"}">'
             f'{"חציית שוריות" if macd and macd_sig and macd>macd_sig else "חציית דוביות"}'
             f'</span>' if macd is not None and macd_sig is not None else ""),
            ("שיא 52 שבועות",    fmt_price(sd.get("week_52_high")), ""),
            ("שפל 52 שבועות",    fmt_price(sd.get("week_52_low")),  ""),
            ("יחס מחזור",
             f"{vol_r:.2f}x" if vol_r else "—",
             f'<span class="{"bull" if vol_r and vol_r>1.2 else "bear" if vol_r and vol_r<0.7 else "neut"}">'
             f'{"גבוה" if vol_r and vol_r>1.2 else "נמוך" if vol_r and vol_r<0.7 else "רגיל"}'
             f'</span>' if vol_r else ""),
            ("בטא",              f"{beta:.3f}" if beta else "—", ""),
        ]

        tbl = (
            '<table class="tech-tbl">'
            '<tr><th>מדד</th><th>ערך</th><th>אות</th></tr>'
        )
        for r in rows:
            tbl += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>"
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

    with mid:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">🎚️ מד RSI</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(rsi_gauge(sd.get("rsi_14", 50)),
                        use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size:11px;text-align:center;color:#8B949E">'
            '<span style="color:#3FB950">■</span> מכור יתר &lt;30 &nbsp;|&nbsp;'
            '<span style="color:#D29922">■</span> ניטרלי 30-70 &nbsp;|&nbsp;'
            '<span style="color:#F85149">■</span> קנוי יתר &gt;70'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">📡 קונצנזוס TradingView</p>',
            unsafe_allow_html=True,
        )
        if ta_data and ta_data.get("summary"):
            st.plotly_chart(ta_rec_bar(ta_data["summary"]),
                            use_container_width=True,
                            config={"displayModeBar": False})
            overall    = ta_data["summary"].get("RECOMMENDATION", "")
            overall_he = HE_TV_REC.get(overall, overall)
            rec_color  = {
                "STRONG_BUY": "#3FB950", "BUY": "#3FB950",
                "NEUTRAL":    "#D29922",
                "SELL":       "#F85149", "STRONG_SELL": "#F85149",
            }.get(overall, "#8B949E")
            st.markdown(
                f'<div style="text-align:center;margin-top:-10px">'
                f'<span style="font-size:16px;font-weight:700;color:{rec_color}">'
                f'{overall_he}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("נתוני TradingView TA אינם זמינים עבור טיקר זה.", icon="ℹ️")

    # ── פרשנות טכנית של AI ───────────────────────────────────────────────────
    st.markdown("---")
    tech_commentary = analysis.get("technical_commentary", "")
    tech_pattern    = analysis.get("technical_pattern", "")
    if tech_commentary or tech_pattern:
        sa1, sa2 = st.columns([1.4, 1])
        with sa1:
            st.markdown(
                '<p style="font-size:11px;letter-spacing:1.5px;color:#D29922;'
                'font-weight:600;text-align:right">📉 ניתוח ממוצעים נעים</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #21262D;'
                f'border-left:3px solid #D29922;border-radius:8px;padding:16px;'
                f'font-size:13px;line-height:1.9;color:#C9D1D9;direction:rtl;text-align:right">'
                f'{tech_commentary.replace(chr(10), "<br>") if tech_commentary else "לא זמין."}</div>',
                unsafe_allow_html=True,
            )
        with sa2:
            st.markdown(
                '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
                'font-weight:600;text-align:right">📐 דפוס טכני מזוהה</p>',
                unsafe_allow_html=True,
            )
            conf = int(analysis.get("confidence_score") or 1)
            conf_color = "#3FB950" if conf >= 7 else "#D29922" if conf >= 4 else "#F85149"
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #21262D;'
                f'border-left:3px solid #58A6FF;border-radius:8px;padding:16px;'
                f'direction:rtl;text-align:right">'
                f'<div style="font-size:15px;color:#E6EDF3;font-weight:700;margin-bottom:12px">'
                f'{tech_pattern or "לא זוהה דפוס"}</div>'
                f'<div style="font-size:11px;color:#8B949E;margin-bottom:6px">מדד ביטחון</div>'
                f'<div style="background:#21262D;border-radius:20px;height:8px;overflow:hidden">'
                f'<div style="width:{conf*10}%;height:8px;border-radius:20px;background:{conf_color}"></div>'
                f'</div>'
                f'<div style="font-size:12px;color:{conf_color};font-weight:700;margin-top:6px">'
                f'{conf}/10</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── EPS — 8 רבעונים אחרונים ──────────────────────────────────────────────
    qe = quarterly_earnings if (quarterly_earnings and "error" not in quarterly_earnings) else None
    if qe and qe.get("quarters"):
        st.markdown("---")
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#3FB950;'
            'font-weight:600;text-align:right">📊 EPS — 8 רבעונים אחרונים</p>',
            unsafe_allow_html=True,
        )
        quarters = qe["quarters"]
        labels   = [q.get("quarter", "") for q in quarters]
        actuals  = [q.get("actual")   for q in quarters]
        estimates= [q.get("estimate") for q in quarters]
        surprise_pcts = [q.get("surprise_pct") for q in quarters]

        fig_eps = go.Figure()

        # Estimate bars (background)
        if any(e is not None for e in estimates):
            fig_eps.add_trace(go.Bar(
                x=labels,
                y=estimates,
                name="תחזית",
                marker_color="#21262D",
                marker_line_color="#58A6FF",
                marker_line_width=1,
            ))

        # Actual bars — green if beat, red if miss
        bar_colors = []
        for a, e in zip(actuals, estimates):
            if a is None:
                bar_colors.append("#8B949E")
            elif e is not None and a >= e:
                bar_colors.append("#3FB950")
            elif e is not None:
                bar_colors.append("#F85149")
            else:
                bar_colors.append("#58A6FF")

        # Surprise % text above bars
        text_labels = []
        for sp in surprise_pcts:
            if sp is not None:
                text_labels.append(f"{'+' if sp >= 0 else ''}{sp:.0f}%")
            else:
                text_labels.append("")

        fig_eps.add_trace(go.Bar(
            x=labels,
            y=actuals,
            name="בפועל",
            marker_color=bar_colors,
            text=text_labels,
            textposition="outside",
            textfont={"color": "#C9D1D9", "size": 11},
        ))

        fig_eps.update_layout(
            paper_bgcolor="#0B0E11",
            plot_bgcolor="#161B22",
            font={"color": "#C9D1D9"},
            xaxis={"gridcolor": "#21262D", "tickangle": -30},
            yaxis={"gridcolor": "#21262D", "title": "EPS ($)"},
            margin=dict(t=30, b=60, l=50, r=20),
            height=280,
            barmode="overlay",
            legend={"orientation": "h", "x": 0, "y": 1.12,
                    "font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
            showlegend=any(e is not None for e in estimates),
        )
        st.plotly_chart(fig_eps, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ניתוח גיאופוליטי ומאקרו
# ════════════════════════════════════════════════════════════════════════════

def tab_macro(analysis: dict, sd: dict, macro_data: dict | None = None) -> None:
    geo = analysis.get("geopolitical_analysis", "")

    # ── Macro Dashboard tiles ─────────────────────────────────────────────────
    md = (macro_data or {}).get("macro", {})
    if md:
        st.markdown(
            '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
            'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
            'text-align:right">🌐 מאקרו — לוח מחוונים בזמן אמת</p>',
            unsafe_allow_html=True,
        )
        macro_names = list(md.keys())
        macro_cols  = st.columns(len(macro_names))

        for col, name in zip(macro_cols, macro_names):
            item       = md[name]
            price_v    = item.get("price")
            chg_v      = item.get("change_pct")

            # Format price
            symbol = item.get("symbol", "")
            if symbol in ("^VIX", "^TNX"):
                price_str = f"{price_v:.2f}" if price_v is not None else "—"
            elif symbol == "DX-Y.NYB":
                price_str = f"{price_v:.2f}" if price_v is not None else "—"
            else:
                price_str = f"${price_v:,.2f}" if price_v is not None else "—"

            if chg_v is not None:
                chg_color = "#3FB950" if chg_v >= 0 else "#F85149"
                chg_str   = f"{'▲' if chg_v >= 0 else '▼'} {abs(chg_v):.2f}%"
            else:
                chg_color, chg_str = "#8B949E", "—"

            with col:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:8px;padding:10px 12px;text-align:center">'
                    f'<div style="font-size:10px;letter-spacing:1px;color:#8B949E;margin-bottom:2px">'
                    f'{name}</div>'
                    f'<div style="font-size:17px;font-weight:800;color:#E6EDF3;line-height:1.2">'
                    f'{price_str}</div>'
                    f'<div style="font-size:12px;color:{chg_color};font-weight:600;margin-top:2px">'
                    f'{chg_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
        'font-weight:600;text-transform:uppercase;margin-bottom:12px;'
        'text-align:right">🌍 ניתוח גיאופוליטי ומאקרו</p>',
        unsafe_allow_html=True,
    )

    paras = [p.strip() for p in geo.split("\n") if p.strip()] or [geo]
    panel_cfgs = [
        ("🌐", "הקשר גלובלי",          ""),
        ("⚡", "גורם סיכון מרכזי",     "red"),
        ("📊", "סביבת המאקרו",          "ylw"),
        ("🔭", "תחזית",                "grn"),
        ("🌐", "הקשר גלובלי",          ""),
        ("⚡", "גורם סיכון נוסף",      "red"),
        ("📊", "נתונים מאקרו",          "ylw"),
        ("🔭", "השלכות",               "grn"),
    ]
    for i, para in enumerate(paras[:8]):
        icon, title, cls = panel_cfgs[min(i, len(panel_cfgs)-1)]
        st.markdown(
            f'<div class="ins-panel {cls}">'
            f'<div class="ins-title">{icon} {title}</div>'
            f'<div class="ins-body">{para}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── הקשר פונדמנטלי ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
        'font-weight:600;text-transform:uppercase;margin-bottom:8px;'
        'text-align:right">💰 הקשר פונדמנטלי</p>',
        unsafe_allow_html=True,
    )
    f1, f2, f3, f4 = st.columns(4)
    items = [
        (f1, "מכפיל רווח (TTM)",       sd.get("pe_ratio"),       "x"),
        (f2, "מכפיל רווח קדימה",       sd.get("forward_pe"),     "x"),
        (f3, "רווח למניה (TTM)",        sd.get("eps_ttm"),        "$"),
        (f4, "שולי רווח",               sd.get("profit_margin"),  "%"),
    ]
    for col, lbl, val, suf in items:
        with col:
            try:
                fval = float(val)
                disp = (f"${fval:.2f}" if suf == "$"
                        else f"{fval*100:.1f}%" if suf == "%"
                        else f"{fval:.2f}x")
            except Exception:
                disp = "—"
            st.markdown(metric_card(lbl, disp), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — חדשות וסנטימנט
# ════════════════════════════════════════════════════════════════════════════

def tab_news(news_co: dict, news_macro: dict, news_geo: dict | None = None) -> None:
    all_art = news_co.get("articles", []) + news_macro.get("articles", []) + (news_geo or {}).get("articles", [])
    seen: set[str] = set()
    deduped = []
    for a in all_art:
        t = a.get("title", "")
        if t and t not in seen:
            seen.add(t)
            deduped.append(a)

    if not deduped:
        st.info("לא נמצאו כתבות עבור טיקר זה.", icon="📭")
        return

    st.markdown(
        f'<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
        f'font-weight:600;text-transform:uppercase;margin-bottom:12px;'
        f'text-align:right">📰 עדכוני חדשות אחרונים — {len(deduped)} כתבות</p>',
        unsafe_allow_html=True,
    )

    he_sentiment = {"bull": "שורי", "bear": "דובי", "neut": "ניטרלי"}
    badge_color  = {"bull": "#3FB950", "bear": "#F85149", "neut": "#D29922"}

    for article in deduped[:16]:
        title   = article.get("title", "")
        summary = article.get("summary", "")
        source  = article.get("source", "")
        pub     = article.get("published", "")[:25]
        link    = article.get("link", "")

        icon, css_cls = classify_news(title)
        label_he = he_sentiment.get(css_cls, "ניטרלי")
        col_hex  = badge_color.get(css_cls, "#8B949E")

        exp_label = f'{icon}  {title[:85]}{"…" if len(title)>85 else ""}  [{source}]'
        with st.expander(exp_label, expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                if summary:
                    st.markdown(
                        f'<p style="font-size:13px;color:#C9D1D9;line-height:1.6">'
                        f'{summary}</p>',
                        unsafe_allow_html=True,
                    )
                if link:
                    st.markdown(
                        f'<a href="{link}" target="_blank" '
                        f'style="font-size:12px;color:#58A6FF">'
                        f'קרא את הכתבה המלאה ←</a>',
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<span style="color:{col_hex};font-weight:700;font-size:13px">'
                    f'{label_he}</span><br>'
                    f'<span style="font-size:11px;color:#8B949E">{pub}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — פיבונאצ'י ותחזיות ML
# ════════════════════════════════════════════════════════════════════════════

def tab_fibonacci_forecast(sd: dict, fib: dict, forecast: dict, analysis: dict) -> None:
    import plotly.graph_objects as go
    import numpy as np

    ticker  = sd.get("ticker", "")
    price   = sd.get("current_price", 0) or 0

    # ── Fibonacci Chart ───────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:11px;letter-spacing:1.5px;color:#F0B90B;font-weight:600;"
        "text-align:right'>📐 רמות פיבונאצ'י — Retracement &amp; Extensions</p>",
        unsafe_allow_html=True,
    )

    if fib and "error" not in fib:
        levels   = fib.get("levels", {})
        exts     = fib.get("extensions", {})
        fib_high = fib.get("high", 0)
        fib_low  = fib.get("low", 0)
        nearest  = fib.get("nearest_level", "")
        at_gp    = fib.get("at_golden_pocket", False)
        conf_sma = fib.get("confluence_sma", None)
        direction = fib.get("direction", "uptrend")

        fig = go.Figure()

        # Golden Pocket shading (50% - 61.8%)
        gp_lo = min(float(levels.get("50.0", 0)), float(levels.get("61.8", 0)))
        gp_hi = max(float(levels.get("50.0", 0)), float(levels.get("61.8", 0)))
        if gp_lo and gp_hi:
            fig.add_hrect(
                y0=gp_lo, y1=gp_hi,
                fillcolor="rgba(240,185,11,0.12)",
                line_width=0,
                annotation_text="🏆 Golden Pocket",
                annotation_position="right",
                annotation_font=dict(color="#F0B90B", size=11),
            )

        # Retracement level lines
        colors = {
            "0.0":   "#8B949E",
            "23.6":  "#58A6FF",
            "38.2":  "#79C0FF",
            "50.0":  "#F0B90B",
            "61.8":  "#FFD700",
            "78.6":  "#D29922",
            "100.0": "#8B949E",
        }
        for key, lvl_price in levels.items():
            lvl_price = float(lvl_price)
            color = colors.get(key, "#58A6FF")
            width = 2 if key in ("61.8", "50.0") else 1
            fig.add_hline(
                y=lvl_price,
                line=dict(color=color, width=width, dash="dash"),
                annotation_text=f"{key}% — ${lvl_price:,.2f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=10),
            )

        # Extension lines
        ext_colors = ["#3FB950", "#2ea043", "#26863a", "#1c6e30"]
        for (key, ext_price), col in zip(exts.items(), ext_colors):
            ext_price = float(ext_price)
            fig.add_hline(
                y=ext_price,
                line=dict(color=col, width=1, dash="dot"),
                annotation_text=f"Ext {key}% — ${ext_price:,.2f}",
                annotation_position="right",
                annotation_font=dict(color=col, size=10),
            )

        # Current price marker
        fig.add_hline(
            y=price,
            line=dict(color="#FFFFFF", width=2),
            annotation_text=f"  מחיר נוכחי ${price:,.2f}",
            annotation_position="right",
            annotation_font=dict(color="#FFFFFF", size=11, weight="bold"),
        )

        # Y-axis range
        all_prices = [float(v) for v in {**levels, **exts}.values() if float(v) > 0]
        y_min = min(all_prices) * 0.97
        y_max = max(all_prices) * 1.03

        fig.update_layout(
            height=440,
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            font=dict(color="#C9D1D9", size=11),
            margin=dict(l=20, r=180, t=30, b=20),
            yaxis=dict(
                title="מחיר ($)",
                gridcolor="#21262D",
                range=[y_min, y_max],
                tickformat=",.0f",
            ),
            xaxis=dict(showticklabels=False, showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Info row
        gp_badge = '<span style="background:#F0B90B22;color:#F0B90B;padding:2px 8px;border-radius:4px;font-weight:700">✅ Golden Pocket</span>' if at_gp else ""
        sma_badge = f'<span style="background:#58A6FF22;color:#58A6FF;padding:2px 8px;border-radius:4px">⚡ Confluence: {conf_sma}</span>' if conf_sma else ""
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #21262D;border-radius:8px;'
            f'padding:12px 18px;direction:rtl;text-align:right;font-size:13px;'
            f'display:flex;gap:12px;flex-wrap:wrap">'
            f'<span style="color:#8B949E">מגמה: <strong style="color:#E6EDF3">{direction}</strong></span> &nbsp;|&nbsp; '
            f'<span style="color:#8B949E">רמה קרובה: <strong style="color:#F0B90B">{nearest}</strong></span> &nbsp;|&nbsp; '
            f'<span style="color:#8B949E">שיא: <strong>${fib_high:,.2f}</strong></span> &nbsp;|&nbsp; '
            f'<span style="color:#8B949E">שפל: <strong>${fib_low:,.2f}</strong></span>'
            f'&nbsp;&nbsp;{gp_badge}&nbsp;{sma_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Fibonacci AI commentary
        fib_commentary = analysis.get("fibonacci_commentary", "")
        if fib_commentary:
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #21262D;'
                f'border-left:3px solid #F0B90B;border-radius:8px;padding:16px;'
                f'font-size:14px;line-height:1.9;color:#C9D1D9;direction:rtl;text-align:right">'
                f'🤖 <strong style="color:#F0B90B">פרשנות AI:</strong> '
                f'{fib_commentary.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("נתוני פיבונאצ'י אינם זמינים.")

    st.markdown("---")

    # ── ML Forecasting ────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;font-weight:600;'
        'text-align:right">🔮 תחזיות ML — 30 יום קדימה</p>',
        unsafe_allow_html=True,
    )

    if forecast and "error" not in forecast:
        lr  = forecast.get("linear_regression", {})
        rf  = forecast.get("random_forest", {})
        mc  = forecast.get("monte_carlo", {})
        fib_align = forecast.get("fibonacci_alignment", {})

        def _pct_color(pct):
            if pct is None: return "#8B949E", "—"
            return ("#3FB950" if pct >= 0 else "#F85149"), f"{pct:+.1f}%"

        col1, col2, col3 = st.columns(3)
        for col, name, data, icon in [
            (col1, "Linear Regression", lr, "📏"),
            (col2, "Random Forest",     rf, "🌲"),
            (col3, "Monte Carlo P50",   mc, "🎲"),
        ]:
            target = data.get("target") or data.get("median", 0)
            pct    = data.get("change_pct")
            c, pct_str = _pct_color(pct)
            model_key = name.lower().replace(" ", "_").replace("_p50", "")
            if model_key == "random": model_key = "random_forest"
            if model_key == "monte":  model_key = "monte_carlo_p50"
            fib_hit = fib_align.get(model_key) or fib_align.get(model_key.replace("_p50",""))
            fib_badge = (
                f'<div style="margin-top:6px;font-size:11px;color:#F0B90B">'
                f'⚡ Fib {fib_hit["fib_level"]} — ${fib_hit["fib_price"]:,.2f}</div>'
                if fib_hit else ""
            )
            with col:
                st.markdown(
                    f'<div style="background:#161B22;border:1px solid #21262D;'
                    f'border-radius:10px;padding:16px;text-align:center;direction:rtl">'
                    f'<div style="font-size:22px;margin-bottom:4px">{icon}</div>'
                    f'<div style="font-size:11px;color:#8B949E;letter-spacing:1px;margin-bottom:8px">'
                    f'{name}</div>'
                    f'<div style="font-size:22px;font-weight:800;color:#E6EDF3">'
                    f'${target:,.2f}</div>'
                    f'<div style="font-size:14px;color:{c};font-weight:600">{pct_str}</div>'
                    f'{fib_badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Monte Carlo range
        if mc:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            p10 = mc.get("p10_bear", 0)
            p90 = mc.get("p90_bull", 0)
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #21262D;border-radius:8px;'
                f'padding:12px 18px;direction:rtl;text-align:right;font-size:13px">'
                f'🎲 <strong>Monte Carlo — טווח סיכון (P10-P90):</strong> &nbsp;'
                f'<span style="color:#F85149">${p10:,.2f}</span> — '
                f'<span style="color:#3FB950">${p90:,.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Disclaimer
        st.markdown(
            '<p style="font-size:10px;color:#8B949E;text-align:right;margin-top:8px">'
            '⚠️ התחזיות מבוססות על נתונים היסטוריים בלבד ואינן מהוות המלצת השקעה. '
            'שוק ההון כרוך בסיכון.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("תחזיות ML אינן זמינות.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — צ'אט עם הסוכן
# ════════════════════════════════════════════════════════════════════════════

def tab_chat_agent(sd: dict | None = None, analysis: dict | None = None, video_context: dict | None = None) -> None:
    sd       = sd or {}
    analysis = analysis or {}
    ticker  = sd.get("ticker", "")
    company = sd.get("company_name", "")

    # ── Detect ticker change mid-session ─────────────────────────────────────
    if ticker and st.session_state.get("chat_ticker") != ticker and st.session_state.chat_messages:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                f"📌 **הקשר הניתוח השתנה**\n\n"
                f"עברתי לנתח את **{ticker} — {company}**. "
                f"שאל אותי כל שאלה על המניה החדשה."
            ),
        })
    if ticker:
        st.session_state.chat_ticker = ticker

    # ── Welcome screen (empty chat) ──────────────────────────────────────────
    if not st.session_state.chat_messages:
        if ticker:
            subtitle = f'אני מנתח כרגע את <strong style="color:#58A6FF">{ticker} — {company}</strong>'
            sub_hint = "שאל אותי כל שאלה על המניה — טכניקה, פונדמנטלים, סיכונים, או אסטרטגיית מסחר"
        else:
            subtitle = 'יועץ פיננסי אישי — <strong style="color:#58A6FF">NVDA · SOFI · IBIT · ETHA · MAGS · MSTU</strong>'
            sub_hint = "שאל אותי על השוק, על התיק שלך, או על כל מניה שתרצה"
        st.markdown(
            f'<div class="chat-welcome">'
            f'<div style="font-size:42px;margin-bottom:8px">🤖</div>'
            f'<div style="font-size:18px;font-weight:700;color:#E6EDF3;margin-bottom:6px">'
            f'שלום! אני FinanceGPT</div>'
            f'<div style="font-size:13px;color:#8B949E;margin-bottom:14px">{subtitle}</div>'
            f'<div style="font-size:12px;color:#8B949E">{sub_hint}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Suggested questions
        st.markdown(
            '<p style="font-size:11px;color:#58A6FF;font-weight:600;'
            'letter-spacing:1px;text-align:right;margin-bottom:8px">'
            '💡 שאלות מוצעות</p>',
            unsafe_allow_html=True,
        )
        if ticker:
            suggestions = [
                f"מה הסיכונים העיקריים של {ticker} כרגע?",
                f"האם המניה קנויה יתר על פי RSI?",
                f"מה המשמעות של ה-P/E ביחס לסקטור?",
                f"מהו אזור התמיכה הקריטי הבא?",
            ]
        else:
            suggestions = [
                "מה המצב הנוכחי בשוק?",
                "איך הפורטפוליו שלי מושפע מהריבית?",
                "מה דעתך על NVDA עכשיו?",
                "השווה בין SOFI ל-IBIT מבחינת סיכון",
            ]
        col1, col2 = st.columns(2)
        for i, q in enumerate(suggestions):
            col = col1 if i % 2 == 0 else col2
            if col.button(q, key=f"sugg_{i}", use_container_width=True):
                # Inject as if user typed it
                st.session_state.chat_messages.append({"role": "user", "content": q})
                with st.spinner("🤖 מנתח..."):
                    resp = handle_chat_query(q, sd, [], analysis, video_context=video_context)
                st.session_state.chat_messages.append({"role": "assistant", "content": resp})
                st.rerun()

    # ── Render existing messages ─────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    placeholder = (
        f"שאל על {ticker}... (לדוגמה: מה הסיכוי לפריצת שיא?)"
        if ticker
        else "שאל אותי על השוק, על התיק שלך, או על כל מניה שתרצה..."
    )
    if user_input := st.chat_input(placeholder):
        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Show user bubble immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Get and show AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("מנתח..."):
                # Pass last 10 exchanges (20 msgs) excluding the message just added
                history = st.session_state.chat_messages[:-1]
                response = handle_chat_query(user_input, sd, history, analysis, video_context=video_context)
            st.markdown(response)

        # Persist assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # ── Token counter hint ────────────────────────────────────────────────────
    msg_count = len(st.session_state.chat_messages)
    if msg_count > 0:
        st.markdown(
            f'<p style="font-size:10px;color:#8B949E;text-align:left;margin-top:4px">'
            f'💬 {msg_count} הודעות בשיחה | נשלחות ל-Claude: עד 20 אחרונות</p>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB — סורק מניות NASDAQ 100
# ════════════════════════════════════════════════════════════════════════════

def tab_screener() -> None:
    st.markdown(
        '<div style="direction:rtl;text-align:right;margin-bottom:16px">'
        '<span style="font-size:22px;font-weight:800;color:#E6EDF3">🔍 סורק מניות NASDAQ 100</span><br>'
        '<span style="font-size:12px;color:#8B949E">'
        f'יקום: {len(NASDAQ100_TICKERS)} מניות · נתוני EOD · cache 6 שעות</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Filter controls ────────────────────────────────────────────────────────
    with st.expander("⚙️ פילטרים", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_rs    = st.slider("RS Rating מינימלי",  1,   99,  70, key="scr_rs")
            above_sma = st.checkbox("מעל SMA-50", value=True, key="scr_sma")
        with fc2:
            max_beta   = st.slider("Beta מקסימלי",     0.3,  3.5, 2.0, 0.1, key="scr_beta")
            min_sharpe = st.slider("Sharpe מינימלי",  -1.0,  3.0, 0.3, 0.1, key="scr_sharpe")
        with fc3:
            rsi_range = st.slider("טווח RSI", 0, 100, (20, 80), key="scr_rsi")

    run_btn = st.button("▶ הרץ סריקה", type="primary", use_container_width=True)

    if not run_btn and "screener_results" not in st.session_state:
        st.info(
            "הגדר פילטרים ולחץ **הרץ סריקה**. הסריקה הראשונה אורכת ~45 שניות; "
            "לאחר מכן התוצאות נשמרות ב-cache ל-6 שעות.",
            icon="💡",
        )
        return

    if run_btn:
        with st.spinner(f"סורק {len(NASDAQ100_TICKERS)} מניות NASDAQ 100 — אנא המתן…"):
            results = fetch_screener(
                int(min_rs), float(max_beta), float(min_sharpe),
                bool(above_sma), float(rsi_range[0]), float(rsi_range[1]),
            )
        st.session_state["screener_results"]  = results
        st.session_state["screener_criteria"] = {
            "min_rs": min_rs, "max_beta": max_beta,
            "min_sharpe": min_sharpe, "above_sma50": above_sma,
        }

    results  = st.session_state.get("screener_results",  [])
    criteria = st.session_state.get("screener_criteria", {})

    if not results or (len(results) == 1 and "error" in results[0]):
        err = results[0].get("error", "שגיאה לא ידועה") if results else ""
        st.warning(f"לא נמצאו מניות העומדות בקריטריונים. {err}")
        return

    # ── Summary banner ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1f14,#0B0E11);'
        f'border:1px solid #3FB95044;border-right:3px solid #3FB950;'
        f'border-radius:0 10px 10px 0;padding:12px 18px;margin:12px 0;'
        f'direction:rtl;text-align:right">'
        f'<span style="font-size:13px;color:#3FB950;font-weight:700">'
        f'✅ נמצאו {len(results)} מניות מתוך {len(NASDAQ100_TICKERS)}'
        f'</span>'
        f'<span style="font-size:12px;color:#8B949E;margin-right:16px">'
        f'RS ≥ {criteria.get("min_rs","—")} | '
        f'Beta ≤ {criteria.get("max_beta","—")} | '
        f'Sharpe ≥ {criteria.get("min_sharpe","—")}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── AI Commentary ──────────────────────────────────────────────────────────
    with st.spinner("🤖 מנתח תוצאות…"):
        commentary = summarize_screener_results(results, criteria)
    st.markdown(
        f'<div style="background:#0D1626;border:1px solid #58A6FF33;'
        f'border-right:3px solid #58A6FF;border-radius:0 8px 8px 0;'
        f'padding:12px 16px;direction:rtl;text-align:right;'
        f'font-size:13px;color:#C9D1D9;line-height:1.8;margin-bottom:16px">'
        f'🤖 {commentary}</div>',
        unsafe_allow_html=True,
    )

    # ── Top-5 highlight cards ──────────────────────────────────────────────────
    top5 = results[:5]
    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.5px;color:#58A6FF;'
        'font-weight:700;text-align:right;margin-bottom:8px">🏆 TOP 5 — RS RATING</p>',
        unsafe_allow_html=True,
    )
    t5_cols = st.columns(5)
    for col, m in zip(t5_cols, top5):
        chg = m.get("change_pct")
        chg_color = "#3FB950" if chg and chg >= 0 else "#F85149" if chg else "#8B949E"
        chg_arrow = "▲" if chg and chg >= 0 else "▼"
        rs = m["rs_rating"]
        rs_color = "#3FB950" if rs >= 80 else "#D29922" if rs >= 65 else "#F85149"
        sma_badge = (
            '<span style="font-size:9px;background:#1a3a2a;color:#3FB950;'
            'border-radius:3px;padding:1px 4px;margin-right:3px">SMA✓</span>'
            if m.get("above_sma50") else ""
        )
        with col:
            # Click to analyze
            if st.button(f"📈 {m['ticker']}", key=f"scr_btn_{m['ticker']}",
                         use_container_width=True):
                st.session_state.ticker          = m["ticker"]
                st.session_state.ticker_selected = True
                st.rerun()
            st.markdown(
                f'<div style="background:#161B22;border:1px solid #21262D;'
                f'border-top:3px solid {rs_color};border-radius:0 0 8px 8px;'
                f'padding:10px;text-align:center;direction:rtl">'
                f'<div style="font-size:18px;font-weight:900;color:#E6EDF3">'
                f'${m["price"]:,.2f}</div>'
                f'<div style="font-size:12px;color:{chg_color};font-weight:700">'
                f'{chg_arrow} {abs(chg):.2f}%</div>'
                f'<div style="font-size:10px;color:#8B949E;margin-top:6px">'
                f'{sma_badge}RS: <b style="color:{rs_color}">{rs}</b></div>'
                f'<div style="font-size:10px;color:#8B949E">'
                f'β {m.get("beta","—")} | S {m.get("sharpe","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Full results table ─────────────────────────────────────────────────────
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px;letter-spacing:1.5px;color:#8B949E;'
        'font-weight:700;text-align:right;margin-bottom:8px">'
        '📋 רשימה מלאה (לחץ על טיקר לניתוח מעמיק)</p>',
        unsafe_allow_html=True,
    )

    # Build table rows
    tbl_rows = []
    for m in results:
        chg  = m.get("change_pct")
        rsi  = m.get("rsi")
        tbl_rows.append({
            "טיקר":       m["ticker"],
            "מחיר":       f'${m["price"]:,.2f}',
            "שינוי %":    f'{"+" if chg and chg>=0 else ""}{chg:.2f}%' if chg else "—",
            "RS Rating":  m["rs_rating"],
            "RS 50d":     f'+{m["rs_50d"]:.1f}%' if m.get("rs_50d") and m["rs_50d"]>=0 else (f'{m["rs_50d"]:.1f}%' if m.get("rs_50d") else "—"),
            "Beta":       m.get("beta") or "—",
            "Sharpe":     m.get("sharpe") or "—",
            "RSI":        f'{rsi:.1f}' if rsi else "—",
            "SMA-50 ✓":   "✅" if m.get("above_sma50")  else "❌",
            "SMA-200 ✓":  "✅" if m.get("above_sma200") else "❌",
        })

    import pandas as _pd_scr
    df = _pd_scr.DataFrame(tbl_rows)

    def _rs_color(val):
        try:
            v = int(val)
            if v >= 80: return "color: #3FB950; font-weight:700"
            if v >= 65: return "color: #D29922; font-weight:700"
            return "color: #F85149; font-weight:700"
        except Exception:
            return ""

    styled = df.style.map(_rs_color, subset=["RS Rating"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    # Click-to-analyze buttons below table
    st.markdown(
        '<p style="font-size:11px;color:#8B949E;text-align:right;margin-top:8px">'
        '💡 לחץ על כפתור ה-📈 בכרטיסיות TOP 5 לעיל כדי לטעון מניה לניתוח מלא</p>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB — לוח שנה כלכלי (Economic Calendar)
# ════════════════════════════════════════════════════════════════════════════

def tab_economic_calendar() -> None:
    import json as _json
    from datetime import date as _date, datetime as _dt

    cal = fetch_economic_calendar()
    us_events     = cal.get("us_events", [])
    israel_events = cal.get("israel_events", [])
    next_event    = cal.get("next_event")

    # ── Countdown banner ──────────────────────────────────────────────────
    if next_event:
        try:
            nev_date = _dt.strptime(next_event["date"], "%Y-%m-%d").date()
            days_to  = (nev_date - _date.today()).days
            impact_color = "#F85149" if next_event.get("impact") == "high" else "#F0B90B"
            days_label   = "היום!" if days_to == 0 else (f"מחר" if days_to == 1 else f"בעוד {days_to} ימים")
            ev_name_he   = next_event.get("event_he") or next_event.get("event", "")
            country_flag = "🇺🇸" if next_event.get("country") == "US" else "🇮🇱"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{impact_color}22,{impact_color}11);'
                f'border:1px solid {impact_color}55;border-radius:10px;padding:14px 20px;'
                f'margin-bottom:20px;direction:rtl;display:flex;align-items:center;gap:16px">'
                f'<span style="font-size:28px">⏰</span>'
                f'<div>'
                f'<div style="font-size:11px;color:#8B949E;letter-spacing:1px;text-transform:uppercase">האירוע הבא</div>'
                f'<div style="font-size:16px;font-weight:700;color:#E6EDF3">{country_flag} {ev_name_he}</div>'
                f'<div style="font-size:13px;color:{impact_color};font-weight:600">{days_label} · {next_event["date"]}'
                f'{"  ·  " + next_event["time"] if next_event.get("time") else ""}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # ── Section header helper ─────────────────────────────────────────────
    def _section_hdr(icon, title, color):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:24px 0 12px;direction:rtl">'
            f'<span style="font-size:18px">{icon}</span>'
            f'<span style="font-size:13px;letter-spacing:2px;font-weight:700;'
            f'color:{color};text-transform:uppercase">{title}</span>'
            f'<div style="flex:1;height:1px;background:{color}33;margin-right:8px"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Event row helper ──────────────────────────────────────────────────
    def _event_rows(events: list, limit: int = 20) -> None:
        if not events:
            st.markdown(
                '<p style="color:#8B949E;font-size:13px;text-align:right">אין אירועים מתוכננים.</p>',
                unsafe_allow_html=True,
            )
            return
        rows_html = []
        for ev in events[:limit]:
            impact  = ev.get("impact", "medium")
            dot     = "🔴" if impact == "high" else "🟡"
            ev_time = ev.get("time", "")
            prev    = ev.get("previous", "—") or "—"
            fore    = ev.get("forecast", "—") or "—"
            ev_name = ev.get("event_he") or ev.get("event", "")
            ev_date = ev.get("date", "")
            bg = "#1a0e0e" if impact == "high" else "#161B22"
            rows_html.append(
                f'<tr style="background:{bg};border-bottom:1px solid #30363d">'
                f'<td style="padding:8px 12px;color:#8B949E;font-size:12px;white-space:nowrap">{ev_date}{" · "+ev_time if ev_time else ""}</td>'
                f'<td style="padding:8px 12px;font-size:12px;text-align:right">{dot} {ev_name}</td>'
                f'<td style="padding:8px 12px;color:#8B949E;font-size:11px;text-align:center">{prev}</td>'
                f'<td style="padding:8px 12px;color:#58A6FF;font-size:11px;text-align:center">{fore}</td>'
                f'</tr>'
            )
        table_html = (
            '<div style="overflow-x:auto;direction:ltr">'
            '<table style="width:100%;border-collapse:collapse;font-family:monospace">'
            '<thead><tr style="border-bottom:2px solid #30363d">'
            '<th style="padding:8px 12px;color:#8B949E;font-size:11px;text-align:left">תאריך</th>'
            '<th style="padding:8px 12px;color:#8B949E;font-size:11px;text-align:right">אירוע</th>'
            '<th style="padding:8px 12px;color:#8B949E;font-size:11px;text-align:center">קודם</th>'
            '<th style="padding:8px 12px;color:#58A6FF;font-size:11px;text-align:center">תחזית</th>'
            '</tr></thead>'
            '<tbody>' + "".join(rows_html) + '</tbody>'
            '</table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    # ── Two-column layout: US | Israel ────────────────────────────────────
    col_us, col_il = st.columns(2)

    with col_us:
        _section_hdr("🇺🇸", "ארה\"ב", "#58A6FF")
        _event_rows(us_events)

    with col_il:
        _section_hdr("🇮🇱", "ישראל", "#3FB950")
        _event_rows(israel_events)

    # ── LEVI'S INSIGHT ────────────────────────────────────────────────────
    st.markdown('<div style="margin-top:32px"></div>', unsafe_allow_html=True)
    _section_hdr("🤖", "LEVI'S INSIGHT · השפעה על התיק", "#F0B90B")

    all_events = us_events + israel_events
    all_events.sort(key=lambda e: e.get("date", ""))

    with st.spinner("LEVI מנתח את ההשפעה על התיק שלך…"):
        insight_text = fetch_calendar_insight(_json.dumps(all_events[:20]))

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #F0B90B33;border-radius:10px;'
        f'padding:20px 24px;direction:rtl;line-height:1.8;font-size:13px;color:#E6EDF3">'
        f'{insight_text.replace(chr(10), "<br>")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;gap:20px;margin-top:16px;direction:rtl;font-size:11px;color:#8B949E">'
        '<span>🔴 השפעה גבוהה</span>'
        '<span>🟡 השפעה בינונית</span>'
        '<span style="margin-right:auto;font-size:10px">⚡ מתעדכן כל שעה</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Session state init ───────────────────────────────────────────────────
    if "chat_messages"      not in st.session_state: st.session_state.chat_messages      = []
    if "chat_ticker"        not in st.session_state: st.session_state.chat_ticker        = None
    if "ticker_selected"    not in st.session_state: st.session_state.ticker_selected    = False
    if "ticker"             not in st.session_state: st.session_state.ticker             = ""
    if "watchlist"          not in st.session_state:
        st.session_state.watchlist          = load_watchlist()
    if "watchlist_verdicts" not in st.session_state:
        st.session_state.watchlist_verdicts = load_watchlist_verdicts()

    # ── Tab navigation (watchlist click → jump to stock analysis tab) ────────
    if st.session_state.get("_nav_to_analysis"):
        st.session_state._nav_to_analysis = False
        components.html(
            "<script>setTimeout(function(){"
            "var t=window.parent.document.querySelectorAll('[data-baseweb=\"tab\"]');"
            "if(t&&t.length>1){t[1].click();}"
            "},150);</script>",
            height=0,
        )

    ticker = render_sidebar()

    # ── 3 top-level tabs — Market Dashboard is the landing page ─────────────
    t_market, t_stock, t_screener, t_calendar, t_chat = st.tabs([
        "🌐 סקירת שוק",
        "📈 ניתוח מניה",
        "🔍 סורק NASDAQ",
        "📅 לוח שנה כלכלי",
        "💬 צ'אט AI",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Market Dashboard (no ticker analysis required)
    # ════════════════════════════════════════════════════════════════════════
    with t_market:
        market_overview = fetch_market_overview()
        tab_market_overview(market_overview)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Stock Analysis (all ticker-specific content lives here)
    # ════════════════════════════════════════════════════════════════════════
    with t_stock:
        if not st.session_state.get("ticker_selected"):
            # ── כותרת ─────────────────────────────────────────────────────────
            st.markdown(
                '<div style="display:flex;align-items:center;gap:10px;'
                'margin-bottom:18px;direction:rtl">'
                '<span style="font-size:22px">🔥</span>'
                '<span style="font-size:13px;letter-spacing:2px;font-weight:700;'
                'color:#F0B90B;text-transform:uppercase">מניות חמות עכשיו</span>'
                '<div style="flex:1;height:1px;background:#F0B90B33;margin-right:8px"></div>'
                '<span style="font-size:10px;color:#8B949E">עדכון כל 15 דק׳ · לחץ לניתוח מלא</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("סורק שוק לפי Heat Score…"):
                _trending = fetch_trending_stocks()

            if not _trending:
                st.markdown(
                    '<p style="color:#8B949E;text-align:right;font-size:13px">'
                    'לא נמצאו מניות חמות כרגע. נסה שוב מאוחר יותר.</p>',
                    unsafe_allow_html=True,
                )
            else:
                _badge_color = {
                    "🔥 חם מאוד": ("#F0B90B", "#2a1f00"),
                    "⚡ בתנופה":  ("#58A6FF", "#0d1a2a"),
                    "📈 עולה":    ("#3FB950", "#0d2a18"),
                    "👀 מעקב":    ("#8B949E", "#161B22"),
                }
                for _tr in _trending:
                    _sym    = _tr["ticker"]
                    _score  = _tr["score"]
                    _badge  = _tr["badge"]
                    _reason = _tr["reason_he"]
                    _price  = _tr.get("price")
                    _chg    = _tr.get("change_pct")

                    _bc, _bbg  = _badge_color.get(_badge, ("#8B949E", "#161B22"))
                    _chg_color = "#3FB950" if _chg and _chg >= 0 else "#F85149" if _chg else "#8B949E"
                    _arrow     = "▲" if _chg and _chg >= 0 else "▼" if _chg else ""
                    _chg_str   = f"{_arrow} {abs(_chg):.2f}%" if _chg is not None else "—"
                    _price_str = f"${_price:,.2f}" if _price else "—"

                    # Technical State badge (Hebrew MA label)
                    _TS_LABEL = {
                        "bullish_stack":    ("מגמה שורית | מעל 44 ו-150", "#3FB950", "#0d2a18"),
                        "mean_reversion":   ("ריבאונד אפשרי | מתחת ממוצע 150", "#F0B90B", "#2a1f00"),
                        "support_test_150": ("נתמכת על ממוצע 150", "#58A6FF", "#0d1a2a"),
                        "support_test_44":  ("נתמכת על ממוצע 44", "#58A6FF", "#0d1a2a"),
                        "above_150":        ("מעל ממוצע 150 - מגמה שורית", "#3FB950", "#0d2a18"),
                        "below_44":         ("מתחת ממוצע 44", "#F85149", "#2a0d0d"),
                    }
                    _ts = _tr.get("technical_state", "")
                    _d150 = _tr.get("dist_to_150_pct")
                    _d44  = _tr.get("dist_to_44_pct")
                    if _ts in _TS_LABEL:
                        _ts_label, _ts_color, _ts_bg = _TS_LABEL[_ts]
                        _dist_str = ""
                        if _d150 is not None:
                            _dist_str = f" ({'+' if _d150 >= 0 else ''}{_d150:.1f}% מ-MA150)"
                        elif _d44 is not None:
                            _dist_str = f" ({'+' if _d44 >= 0 else ''}{_d44:.1f}% מ-MA44)"
                        _tech_badge = (
                            f'<span style="font-size:10px;background:{_ts_bg};color:{_ts_color};'
                            f'border:1px solid {_ts_color}55;border-radius:4px;'
                            f'padding:2px 7px;font-weight:600">'
                            f'{_ts_label}{_dist_str}</span>'
                        )
                    else:
                        _tech_badge = ""

                    # Technical state row HTML (pre-built for clean f-string embedding)
                    _tech_row = (
                        f'<div style="margin-top:4px">{_tech_badge}</div>'
                        if _tech_badge else ""
                    )

                    # Heat bar width
                    _bar_w   = _score
                    _bar_clr = "#F0B90B" if _score >= 70 else "#58A6FF" if _score >= 50 else "#3FB950"

                    st.markdown(
                        f'<div style="background:linear-gradient(140deg,#161B22,#0D1117);'
                        f'border:1px solid #21262D;border-right:3px solid {_bc};'
                        f'border-radius:10px;padding:14px 16px 10px;margin-bottom:6px;direction:rtl">'
                        # row 1 — ticker + badge + price + change
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div style="display:flex;align-items:center;gap:8px">'
                        f'<span style="font-size:16px;font-weight:800;color:#E6EDF3">{_sym}</span>'
                        f'<span style="font-size:10px;background:{_bbg};color:{_bc};'
                        f'border:1px solid {_bc}55;border-radius:4px;padding:2px 7px;'
                        f'font-weight:700">{_badge}</span>'
                        f'</div>'
                        f'<div style="text-align:left">'
                        f'<span style="font-size:15px;font-weight:700;color:#E6EDF3;'
                        f'direction:ltr">{_price_str}</span>'
                        f'<span style="font-size:12px;font-weight:700;color:{_chg_color};'
                        f'margin-right:8px"> {_chg_str}</span>'
                        f'</div>'
                        f'</div>'
                        # row 2 — reason
                        f'<div style="font-size:11px;color:#8B949E;margin-top:5px;'
                        f'text-align:right">{_reason}</div>'
                        f'{_tech_row}'
                        # row 3 — heat bar
                        f'<div style="display:flex;align-items:center;gap:8px;margin-top:7px">'
                        f'<span style="font-size:9px;color:#8B949E;white-space:nowrap">'
                        f'Heat Score</span>'
                        f'<div style="flex:1;background:#21262D;border-radius:4px;height:5px">'
                        f'<div style="width:{_bar_w}%;background:{_bar_clr};'
                        f'height:5px;border-radius:4px"></div></div>'
                        f'<span style="font-size:10px;font-weight:700;color:{_bar_clr}">'
                        f'{_score}/100</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"📊 נתח את {_sym}", key=f"trend_{_sym}",
                                 use_container_width=True):
                        st.session_state.ticker          = _sym
                        st.session_state.ticker_selected = True
                        st.rerun()
        else:
            ticker = st.session_state.ticker
            with st.spinner(f"שולף נתונים עבור **{ticker}**…"):
                sd = fetch_stock(ticker)

            if "error" in sd:
                st.error(f"❌ **{sd['error']}** — אנא הכנס סמל מניה תקני.", icon="🚫")
            else:
                render_header(sd)
                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

                # ── שליפת חדשות ─────────────────────────────────────────────
                company_q = sd.get("company_name", ticker)
                sector_q  = sd.get("sector", "market")
                industry  = sd.get("industry", "")
                geo_q = _GEO_QUERY_MAP.get(industry) or f"{sector_q} geopolitics macro Federal Reserve"

                with st.spinner("טוען פידי חדשות…"):
                    news_co    = fetch_news(f"{company_q} {ticker}", 8)
                    news_macro = fetch_news(sector_q, 6)
                    news_geo   = fetch_news(geo_q, 6)

                # ── שליפת נתוני TV-TA ─────────────────────────────────────
                exch    = sd.get("exchange", "")
                ta_data = fetch_ta(ticker, tv_screener(ticker, exch),
                                   tv_exchange_str(ticker, exch))

                # ── פיבונאצ'י, תחזיות ML, מדדי קוואנט + EPS + מאקרו + Chart ──
                with st.spinner("📐 מחשב פיבונאצ'י, תחזיות, ממוצעים ומדדי סיכון…"):
                    fib_data        = fetch_fibonacci(ticker)
                    forecast_data   = fetch_forecast(ticker)
                    quant_data      = fetch_quant(ticker)
                    quarterly_earns = fetch_quarterly_earnings(ticker)
                    macro_dashboard = fetch_macro_data()
                    chart_data      = fetch_chart_patterns(ticker)

                # ── הרצת ניתוח AI ─────────────────────────────────────────
                all_articles = (
                    news_co.get("articles", [])
                    + news_macro.get("articles", [])
                    + news_geo.get("articles", [])
                )[:10]
                ta_summary = ta_data.get("summary") if ta_data else None

                # Build slim chart summary for agent — MA distances only (no patterns/peaks)
                _slim_chart: dict | None = None
                if chart_data and "error" not in chart_data:
                    _slim_chart = {
                        k: chart_data[k] for k in (
                            "current_price", "sma_44", "sma_150", "sma_200",
                            "dist_to_44_pct", "dist_to_150_pct", "dist_to_200_pct",
                        ) if k in chart_data
                    }

                with st.spinner("🤖 מריץ ניתוח AI מוסדי…"):
                    analysis = fetch_analysis(
                        ticker,
                        ANALYSIS_VERSION,
                        _stock_json    = json.dumps(sd,            default=str),
                        _news_json     = json.dumps(all_articles,  default=str),
                        _ta_json       = json.dumps(ta_summary,    default=str) if ta_summary else "",
                        _fib_json      = json.dumps(fib_data,      default=str) if fib_data      and "error" not in fib_data      else "",
                        _forecast_json = json.dumps(forecast_data, default=str) if forecast_data and "error" not in forecast_data else "",
                        _quant_json    = json.dumps(quant_data,    default=str) if quant_data    and "error" not in quant_data    else "",
                        _chart_json    = json.dumps(_slim_chart,   default=str) if _slim_chart else "",
                    )

                # Cache for chat tab
                st.session_state["_last_sd"]       = sd
                st.session_state["_last_analysis"] = analysis

                # ── שמירת Verdict ברשימת המעקב (אם הטיקר מועדף) ────────────
                if ticker in st.session_state.get("watchlist", []):
                    _v_rec  = analysis.get("recommendation", "")
                    _v_conf = analysis.get("confidence_score")
                    if _v_rec and _v_conf is not None:
                        update_watchlist_verdict(ticker, _v_rec, int(_v_conf))
                        st.session_state.watchlist_verdicts = load_watchlist_verdicts()

                # ── Sub-tabs inside Tab 2 ──────────────────────────────────
                s1, s2, s3, s4, s5, s6 = st.tabs([
                    "📈 גרף אינטראקטיבי",
                    "🤖 ניתוח AI",
                    "📊 ניתוח טכני",
                    "🌍 גיאופוליטי ומאקרו",
                    "�ום פיבונאצ'י ותחזיות",
                    "📰 חדשות",
                ])
                with s1: tab_chart(ticker, sd)
                with s2: tab_ai_insights(analysis, sd, quant_data)
                with s3: tab_technicals(sd, ta_data, analysis, quarterly_earns, chart_data)
                with s4: tab_macro(analysis, sd, macro_dashboard)
                with s5: tab_fibonacci_forecast(sd, fib_data, forecast_data, analysis)
                with s6: tab_news(news_co, news_macro, news_geo)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — NASDAQ 100 Smart Screener
    # ════════════════════════════════════════════════════════════════════════
    with t_screener:
        tab_screener()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Economic Calendar
    # ════════════════════════════════════════════════════════════════════════
    with t_calendar:
        tab_economic_calendar()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — AI Chat (uses cached stock context from Tab 2 when available)
    # ════════════════════════════════════════════════════════════════════════
    with t_chat:
        _sd       = st.session_state.get("_last_sd")
        _analysis = st.session_state.get("_last_analysis") or {}
        tab_chat_agent(
            _sd if (_sd and "error" not in _sd) else None,
            _analysis,
            None,
        )


if __name__ == "__main__":
    main()
