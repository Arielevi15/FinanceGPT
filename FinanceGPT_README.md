# FinanceGPT

An autonomous AI equity research agent that produces institutional-style stock analysis — and then grades its own recommendations against what the market actually did.

Built in Python on the Anthropic API, with a ReAct tool-calling loop, 30+ market-data tools, and a Streamlit dashboard.

![Architecture](docs/architecture.png)

---

## Why this exists

Most LLM finance projects stop the moment the model returns an answer. The harder question is whether the answer was any good.

FinanceGPT logs every recommendation it makes with the entry price, then revisits each one after 7, 30, and 90 days, pulls the actual closing price, and evaluates whether the call was correct — broken down by recommendation type and by sector. An AI that gives confident answers is easy to build. An AI that keeps a scorecard is the one worth trusting.

---

## What it does

**Ticker analysis.** Give it a symbol and the agent orchestrates a chain of tool calls — fundamentals, quarterly earnings, balance sheet depth, chart patterns, technical indicators, Fibonacci levels, quantitative metrics, options flow, insider transactions, and live news — then synthesises everything into a structured research report: technical and fundamental sections, bull vs. bear case, a six-dimension scorecard, and a concrete trade setup with entry, stop-loss, targets, and position sizing.

**A decision framework, not freeform text.** The verdict is not left to the model's discretion. A conviction-point system awards or subtracts a point per signal (RSI position, moving-average alignment, MACD cross, PEG, free cash flow, insider activity, put/call ratio, analyst consensus, and others), and thresholds map the total score to STRONG BUY through STRONG SELL. Every verdict must carry a risk/reward ratio of at least 1:2 — below that, the recommendation is downgraded a level automatically. The agent also states explicitly whether it agrees or disagrees with Wall Street consensus, and why.

**Self-evaluation.** `analysis_logger.py` captures each analysis as structured JSON: price and technical data, fundamental classification, market context, news sentiment, the extracted reasoning chain, and a comparison against the previous analysis of the same ticker. `outcome_tracker.py` then backfills the outcomes and prints an accuracy report.

**Live monitoring.** A background thread polls financial RSS feeds every 10 minutes and routes each new headline through a lightweight model that classifies market impact, so only genuinely critical news surfaces as an alert. Noise filtering, not noise generation.

**Dashboard.** A six-tab Streamlit interface covering market overview, hot sectors, per-ticker analysis, a NASDAQ-100 screener, an economic calendar, and a conversational agent tab — with TradingView charts and Plotly visualisations throughout.

---

## Architecture

```
User (CLI or Streamlit)
        ↓
   Agent Core  ──  Claude, ReAct loop, multi-turn memory
        ↓ ↑
   Tool Layer  ──  30+ market-data and analysis tools
        ↓
Decision Framework  ──  conviction scoring, R:R ≥ 1:2, consensus check
        ↓
Structured Report  ──  logged as JSON with entry price
        ↓
 Outcome Tracker  ──  7 / 30 / 90-day evaluation ──┐
        └──────────  feedback into future analyses ─┘
```

The agent loop lives in `run_agent()`: the user message goes to Claude alongside the tool schema, `tool_use` blocks are dispatched through `dispatch_tool()`, results are fed back, and the loop continues until the model produces final text. Conversation history persists across turns, so follow-up questions after an analysis keep their context.

Two models are used deliberately: the primary model handles full analysis, while a smaller, faster model screens news headlines in the monitor loop — the classification task doesn't need the larger model, and running it there would be wasteful.

---

## Project structure

| File | Lines | Role |
|:---|---:|:---|
| `app.py` | 3,745 | Streamlit dashboard — six tabs, charts, watchlist |
| `tools.py` | 2,719 | Market-data and analysis tools |
| `agent.py` | 1,760 | Anthropic integration, system prompt, ReAct loop |
| `analysis_logger.py` | 572 | Structured logging of every analysis |
| `main.py` | 347 | CLI entry point, live monitor thread |
| `outcome_tracker.py` | 278 | 7/30/90-day outcome evaluation |

### Selected tools

Beyond the core price and news tools, the layer includes a NASDAQ-100 screener that batch-downloads a year of price data in a single call and computes RS Rating, Beta, Sharpe, and RSI across the index; a forecasting module running linear regression, a random-forest ensemble, and a 1,000-path Monte Carlo simulation; options flow with put/call ratio, implied volatility and max pain; Form 4 insider transactions; balance-sheet analysis with EBITDA, free cash flow and cash runway; an economic calendar; and a YouTube transcript summariser for earnings calls.

---

## Setup

```bash
git clone https://github.com/Arielevi15/FinanceGPT.git
cd FinanceGPT
pip install -r requirements.txt

cp .env.example .env        # then add your Anthropic API key
```

Run the dashboard:

```bash
streamlit run app.py
```

Or the CLI:

```bash
python main.py
```

CLI commands: a bare ticker (`AAPL`) triggers a full analysis, `market` gives a daily summary, `start monitor` / `stop monitor` control live alerts, `clear` resets conversation history.

Check prediction accuracy at any time:

```bash
python outcome_tracker.py --summary
```

---

## Stack

Python · Anthropic API · Streamlit · Plotly · yfinance · tradingview-ta · pandas · NumPy · scikit-learn · feedparser

---

## Limitations

- The outcome tracker requires elapsed real time before results become meaningful — accuracy figures are only as good as the number of analyses that have aged past their evaluation windows.
- Market data comes from free sources (yfinance, RSS), which are occasionally incomplete or delayed.
- Support for Israeli tickers (`.TA` suffix) is present but less thoroughly tested than US equities.
- The dashboard interface is in Hebrew; the agent's analytical output is in English.

---

**This is a research and learning project. Nothing it produces is investment advice.**
