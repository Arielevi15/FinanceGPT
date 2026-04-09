"""
outcome_tracker.py — Back-fills 7d / 30d / 90d price outcomes in stock_analyses.json.

Run manually or schedule daily:
    python outcome_tracker.py             # update all pending outcomes
    python outcome_tracker.py --summary   # update + print accuracy report
    python outcome_tracker.py --quiet     # update silently (for cron/scheduler)

How it works:
    For every analysis entry that has a null outcome slot whose due date has passed,
    it fetches the closing price from yfinance, computes the return %, and evaluates
    whether the recommendation was correct.

    Recommendation correctness rules:
        BUY / STRONG BUY / ACCUMULATE  →  correct if return > 0 %
        SELL / STRONG SELL / REDUCE / AVOID  →  correct if return < 0 %
        HOLD  →  correct if |return| < 5 %
        N/A   →  not evaluated
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

ANALYSES_FILE = os.path.join(os.path.dirname(__file__), "stock_analyses.json")

BUY_SIGNALS  = {"BUY", "STRONG BUY", "ACCUMULATE"}
SELL_SIGNALS = {"SELL", "STRONG SELL", "REDUCE", "AVOID"}
HOLD_SIGNALS = {"HOLD"}

PERIODS = {"7d": 7, "30d": 30, "90d": 90}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load() -> list:
    with open(ANALYSES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: list) -> None:
    with open(ANALYSES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Price fetcher
# ---------------------------------------------------------------------------

def _fetch_price_on_date(ticker: str, target_dt: datetime) -> Optional[float]:
    """
    Return the closing price nearest to target_dt.
    Searches ±4 calendar days to handle weekends / holidays.
    """
    start = (target_dt - timedelta(days=4)).strftime("%Y-%m-%d")
    end   = (target_dt + timedelta(days=4)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None
        # Normalise timezone
        if hist.index.tzinfo is not None:
            hist.index = hist.index.tz_localize(None)
        # Pick the row whose date is closest to target_dt
        deltas = abs(hist.index.to_series() - target_dt)
        closest_idx = deltas.idxmin()
        return round(float(hist.loc[closest_idx, "Close"]), 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Correctness evaluator
# ---------------------------------------------------------------------------

def _evaluate_correctness(recommendation: str, return_pct: float) -> Optional[bool]:
    rec = recommendation.upper()
    if rec in BUY_SIGNALS:
        return return_pct > 0
    if rec in SELL_SIGNALS:
        return return_pct < 0
    if rec in HOLD_SIGNALS:
        return abs(return_pct) < 5.0
    return None   # N/A — cannot evaluate


# ---------------------------------------------------------------------------
# Core updater
# ---------------------------------------------------------------------------

def update_outcomes(analyses: list, verbose: bool = True) -> int:
    """
    Scan every entry for pending outcome slots and fill them if due.
    Returns the number of individual outcome slots updated.
    """
    updated = 0
    now     = datetime.now()

    for entry in analyses:
        ticker     = entry.get("ticker")
        rec        = entry.get("recommendation", "N/A")
        ts_str     = entry.get("timestamp")
        outcomes   = entry.setdefault("outcomes", {})
        entry_price = outcomes.get("entry_price")

        if not ticker or not ts_str or entry_price is None:
            continue

        try:
            analysis_dt = datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        for label, days in PERIODS.items():
            slot = outcomes.setdefault(label, {
                "price": None, "return_pct": None, "correct": None, "checked_at": None
            })

            if slot.get("price") is not None:
                continue   # already filled

            due_dt = analysis_dt + timedelta(days=days)
            if now < due_dt:
                continue   # not yet due

            price = _fetch_price_on_date(ticker, due_dt)
            if price is None:
                if verbose:
                    print(f"  [?] {ticker} {label}: could not fetch price for {due_dt.date()}")
                continue

            ret_pct = round((price - entry_price) / entry_price * 100, 2)
            correct = _evaluate_correctness(rec, ret_pct)

            slot["price"]      = price
            slot["return_pct"] = ret_pct
            slot["correct"]    = correct
            slot["checked_at"] = now.isoformat()

            updated += 1
            if verbose:
                sign  = "✓" if correct else "✗" if correct is False else "~"
                arrow = "↑" if ret_pct > 0 else "↓"
                print(
                    f"  [{sign}] {ticker:<8} {label:<4} | rec={rec:<12} | "
                    f"entry={entry_price:.2f}  →  {price:.2f}  ({arrow}{abs(ret_pct):.1f}%)"
                )

    return updated


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------

def print_accuracy_report(analyses: list) -> None:
    """Print a detailed accuracy breakdown by recommendation type and period."""

    # ── per-recommendation, per-period stats ─────────────────────────────
    stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: {p: {"correct": 0, "incorrect": 0, "total": 0, "returns": []}
                 for p in PERIODS}
    )

    for entry in analyses:
        rec = entry.get("recommendation", "N/A")
        for label in PERIODS:
            slot = (entry.get("outcomes") or {}).get(label, {})
            ret  = slot.get("return_pct")
            ok   = slot.get("correct")
            if ok is None:
                continue
            bucket = stats[rec][label]
            bucket["total"] += 1
            if ok:
                bucket["correct"] += 1
            else:
                bucket["incorrect"] += 1
            if ret is not None:
                bucket["returns"].append(ret)

    # ── overall accuracy ──────────────────────────────────────────────────
    total_slots = correct_slots = 0
    for rec_data in stats.values():
        for period_data in rec_data.values():
            total_slots   += period_data["total"]
            correct_slots += period_data["correct"]

    print("\n" + "=" * 72)
    print("  PREDICTION ACCURACY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total analyses: {len(analyses)}   |   Evaluated outcome slots: {total_slots}")
    if total_slots:
        print(f"  Overall accuracy: {correct_slots / total_slots * 100:.1f}%")
    print("=" * 72)

    if not stats:
        print("  No evaluated outcomes yet.\n")
        return

    header = f"  {'Recommendation':<18} {'Period':<6} {'Correct':>7} {'Total':>7} {'Acc%':>7} {'Avg Ret%':>9}"
    print(header)
    print("  " + "-" * 62)

    for rec in sorted(stats):
        for label in PERIODS:
            d   = stats[rec][label]
            if d["total"] == 0:
                continue
            acc = d["correct"] / d["total"] * 100
            avg_ret = (sum(d["returns"]) / len(d["returns"])) if d["returns"] else 0
            sign = "+" if avg_ret >= 0 else ""
            print(
                f"  {rec:<18} {label:<6} {d['correct']:>7} {d['total']:>7} "
                f"{acc:>6.1f}% {sign}{avg_ret:>7.2f}%"
            )

    # ── sector breakdown ──────────────────────────────────────────────────
    sector_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in analyses:
        sector = entry.get("sector", "Unknown")
        for label in PERIODS:
            slot = (entry.get("outcomes") or {}).get(label, {})
            ok   = slot.get("correct")
            if ok is None:
                continue
            sector_stats[sector]["total"] += 1
            if ok:
                sector_stats[sector]["correct"] += 1

    if sector_stats:
        print("\n  Accuracy by Sector")
        print("  " + "-" * 40)
        for sector, d in sorted(sector_stats.items(), key=lambda x: -x[1]["total"]):
            if d["total"] == 0:
                continue
            acc = d["correct"] / d["total"] * 100
            print(f"  {sector:<28} {acc:>6.1f}%  ({d['total']} slots)")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update stock analysis outcomes and print accuracy report."
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print full accuracy report after updating")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress per-entry output")
    parser.add_argument("--report-only", action="store_true",
                        help="Print report without updating outcomes")
    args = parser.parse_args()

    data = _load()
    print(f"Loaded {len(data)} analyses from {os.path.basename(ANALYSES_FILE)}\n")

    if not args.report_only:
        n = update_outcomes(data, verbose=not args.quiet)
        if n > 0:
            _save(data)
            print(f"\nSaved {n} updated outcome slot(s).")
        else:
            print("No new outcomes to update.")

    if args.summary or args.report_only:
        print_accuracy_report(data)
