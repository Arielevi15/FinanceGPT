"""
main.py — CLI entry point for FinanceGPT.

Handles:
  • Input parsing and mode detection (ticker / market summary / chat)
  • Conversation history management (multi-turn context)
  • Live Monitor background thread (10-minute news polling)
  • ANSI-coloured terminal output
"""

import re
import sys
import threading
import time
from datetime import datetime

# Force UTF-8 output on Windows (avoids Hebrew codepage / cp1255 issues)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 fallback — unlikely but safe

# Guard: print a friendly error if dependencies are missing
try:
    from agent import run_agent, evaluate_news_importance
    from tools import get_all_rss_headlines
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Run:  pip install -r requirements.txt\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
DIM     = "\033[2m"


def c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes (Windows-safe: only applied if stdout is a TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


# ---------------------------------------------------------------------------
# Monitor state (shared between main thread and monitor thread)
# ---------------------------------------------------------------------------

MONITOR_INTERVAL_SECONDS = 600   # 10 minutes

_monitor_active  = False
_monitor_thread: threading.Thread | None = None
_seen_headlines: set[str] = set()
_monitor_lock   = threading.Lock()


# ---------------------------------------------------------------------------
# Banner & help
# ---------------------------------------------------------------------------

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║           FinanceGPT — AI Financial Assistant                ║
║      US & Israeli Markets  |  Real-Time News Alerts          ║
╚══════════════════════════════════════════════════════════════╝{RESET}

{YELLOW}Quick-start commands:{RESET}
  {BOLD}AAPL{RESET}            → Deep analysis report for Apple Inc.
  {BOLD}market{RESET}          → Daily US & Israeli market summary
  {BOLD}start monitor{RESET}   → Background live news alerts (every 10 min)
  {BOLD}stop monitor{RESET}    → Stop live monitoring
  {BOLD}clear{RESET}           → Clear conversation history
  {BOLD}help{RESET}            → Show this help
  {BOLD}exit / quit{RESET}     → Exit FinanceGPT

{DIM}Powered by Anthropic Claude · yfinance · RSS feeds{RESET}
"""


# ---------------------------------------------------------------------------
# Alert printer (called from monitor thread)
# ---------------------------------------------------------------------------

def print_alert(headline: str, source: str, reason: str, link: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    sep = "=" * 62
    lines = [
        "",
        c(sep, RED, BOLD),
        c(f"  🚨  BREAKING MARKET ALERT  [{ts}]", RED, BOLD),
        c(sep, RED, BOLD),
        c(f"  {headline}", YELLOW, BOLD),
        c(f"  Source : {source}", CYAN),
        c(f"  Why    : {reason}", MAGENTA),
    ]
    if link:
        lines.append(f"  Link   : {link}")
    lines.append(c(sep, RED, BOLD))
    lines.append("")
    # Print atomically so monitor output doesn't interleave with user prompts
    with _monitor_lock:
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Tool-call progress callback (shown while agent is working)
# ---------------------------------------------------------------------------

_TOOL_LABELS = {
    "get_stock_data":    "Fetching stock data",
    "get_financial_news":"Searching news feeds",
    "get_market_indices":"Fetching market indices",
}


def on_tool_call(name: str, inputs: dict) -> None:
    label  = _TOOL_LABELS.get(name, name)
    detail = ""
    if "ticker" in inputs:
        detail = f" [{inputs['ticker'].upper()}]"
    elif "query" in inputs:
        q = inputs["query"]
        detail = f' ["{q[:40]}{"…" if len(q)>40 else ""}"]'
    print(c(f"  ⚙  {label}{detail}…", DIM), flush=True)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

# Ticker: 1-5 letters, optional dot-suffix (e.g. .TA for Tel Aviv)
_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}(\.[A-Za-z]{1,4})?$")

_MARKET_KEYWORDS = {
    "market", "summary", "market summary", "market update",
    "indices", "index", "stocks today", "daily summary",
    "how are markets", "market overview",
}

_MONITOR_START = {"start monitor", "monitor start", "begin monitor", "start monitoring"}
_MONITOR_STOP  = {"stop monitor",  "monitor stop",  "end monitor",   "stop monitoring"}


def detect_mode(raw: str) -> str:
    """Return one of: exit | clear | help | start_monitor | stop_monitor |
                       market_summary | ticker_analysis | chat"""
    low = raw.strip().lower()

    if low in {"exit", "quit", "q", "bye"}:
        return "exit"
    if low == "clear":
        return "clear"
    if low in {"help", "?"}:
        return "help"
    if low in _MONITOR_START:
        return "start_monitor"
    if low in _MONITOR_STOP:
        return "stop_monitor"
    if low in _MARKET_KEYWORDS:
        return "market_summary"
    if _TICKER_RE.match(raw.strip()):
        return "ticker_analysis"
    return "chat"


def build_agent_message(raw: str, mode: str) -> str:
    """Expand shorthand inputs into rich prompts before sending to the agent."""
    if mode == "ticker_analysis":
        ticker = raw.strip().upper()
        return (
            f"Perform a full investment analysis on **{ticker}**. "
            f"Fetch the stock data, retrieve recent company news, and then search for "
            f"relevant geo-political or macro-economic context for its sector. "
            f"Generate the complete structured markdown report.\n\n"
            f"Additional requirements for this analysis:\n"
            f"- Call get_quarterly_earnings, get_chart_patterns, calculate_fibonacci_levels, "
            f"and get_quant_metrics — integrate results into existing sections as instructed.\n"
            f"- Apply the Decision Framework conviction-point system and state the final score.\n"
            f"- Use Fibonacci levels from calculate_fibonacci_levels for the entry, stop-loss, "
            f"and price target in the Verdict. R:R must be ≥ 1:2, else downgrade recommendation.\n"
            f"- State Daily/Weekly timeframe alignment explicitly in Technical Overview.\n"
            f"- Compare your verdict to Wall Street consensus and state AGREES / PARTIALLY AGREES / DISAGREES."
        )
    if mode == "market_summary":
        return (
            "Give me a comprehensive daily market summary. "
            "Fetch current US and Israeli market index levels, retrieve the top macro "
            "financial headlines, and produce a cohesive analysis that compares both "
            "markets, highlights the key sector drivers, and flags any geo-political "
            "events affecting Israel or global risk sentiment."
        )
    return raw


# ---------------------------------------------------------------------------
# Live Monitor background thread
# ---------------------------------------------------------------------------

def _monitor_loop() -> None:
    global _monitor_active, _seen_headlines

    print(c(
        f"\n[Monitor] Started — polling every {MONITOR_INTERVAL_SECONDS//60} minutes. "
        f"Type 'stop monitor' to stop.\n",
        GREEN,
    ), flush=True)

    while _monitor_active:
        try:
            headlines = get_all_rss_headlines(max_per_feed=10)
            critical_count = 0

            for item in headlines:
                if not _monitor_active:
                    break
                title = item.get("title", "").strip()
                if not title or title in _seen_headlines:
                    continue
                _seen_headlines.add(title)

                evaluation = evaluate_news_importance(title)
                if evaluation.get("is_critical") and evaluation.get("impact") == "HIGH":
                    print_alert(
                        headline=title,
                        source=item.get("source", "Unknown"),
                        reason=evaluation.get("reason", ""),
                        link=item.get("link", ""),
                    )
                    critical_count += 1

            if critical_count == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    c(f"[Monitor {ts}] No critical alerts. "
                      f"Next check in {MONITOR_INTERVAL_SECONDS//60} min.", DIM),
                    end="\r",
                    flush=True,
                )

        except Exception as exc:
            print(c(f"\n[Monitor] Error during polling: {exc}", YELLOW), flush=True)

        # Sleep in 1-second ticks so we can react quickly to stop signal
        for _ in range(MONITOR_INTERVAL_SECONDS):
            if not _monitor_active:
                break
            time.sleep(1)

    print(c("\n[Monitor] Stopped.\n", YELLOW), flush=True)


# ---------------------------------------------------------------------------
# Main CLI loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _monitor_active, _monitor_thread, _seen_headlines

    print(BANNER)

    conversation_history: list = []

    try:
        while True:
            # ---- Prompt -------------------------------------------------------
            try:
                raw = input(c("You: ", CYAN, BOLD)).strip()
            except (EOFError, KeyboardInterrupt):
                raw = "exit"

            if not raw:
                continue

            mode = detect_mode(raw)

            # ---- Local commands (no agent call needed) -------------------------
            if mode == "exit":
                _monitor_active = False
                print(c("\nGoodbye! May your portfolio stay green.\n", GREEN))
                break

            if mode == "clear":
                conversation_history = []
                print(c("Conversation history cleared.", GREEN))
                continue

            if mode == "help":
                print(BANNER)
                continue

            if mode == "start_monitor":
                if _monitor_active:
                    print(c("Monitor is already running. Type 'stop monitor' to stop it.", YELLOW))
                else:
                    _monitor_active = True
                    _seen_headlines.clear()
                    _monitor_thread = threading.Thread(
                        target=_monitor_loop, daemon=True
                    )
                    _monitor_thread.start()
                continue

            if mode == "stop_monitor":
                if _monitor_active:
                    _monitor_active = False
                    print(c("Stopping monitor…", YELLOW))
                else:
                    print(c("Monitor is not currently running.", YELLOW))
                continue

            # ---- Agent modes --------------------------------------------------
            agent_message = build_agent_message(raw, mode)

            print(c("\nFinanceGPT: ", MAGENTA, BOLD) + c("Working…\n", DIM))

            try:
                response_text, conversation_history = run_agent(
                    agent_message,
                    conversation_history,
                    on_tool_call=on_tool_call,
                )
            except Exception as exc:
                print(c(f"\n[Error] Agent call failed: {exc}\n", RED))
                continue

            print(c("\nFinanceGPT:\n", MAGENTA, BOLD))
            print(response_text)
            print()

    except Exception as fatal:
        print(c(f"\n[Fatal] Unexpected error: {fatal}", RED))
        _monitor_active = False
        sys.exit(1)


if __name__ == "__main__":
    main()
