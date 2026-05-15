"""Interactive CLI for the multi-agent trading intelligence system.

Modes:
  - Interactive menu (default):  python main.py
  - One-shot:                    python main.py NVDA
  - One-shot + reflection:       python main.py NVDA --reflect
  - One-shot for a past date:    python main.py NVDA --date 2026-04-20
  - Reflection only:             python main.py --reflect-only
"""
from __future__ import annotations
import os
import sys
import argparse
import textwrap
from datetime import datetime, date

from graphs.trading_graph import build_graph
from graphs import progress as progress_mod
from memory.memory_db import init_db, get_recent_decisions, get_pending_evaluations
from memory.reflection import run_reflection
from reporting.writer import write_reports, REPORTS_ROOT


# ----- Curated demo universe -------------------------------------------------
# (ticker, company, sector) — the same list the Streamlit UI uses, kept here so
# the CLI menu can show pretty names too.
DEMO_UNIVERSE = [
    ("AAPL",  "Apple Inc.",              "Consumer Tech"),
    ("MSFT",  "Microsoft Corp.",         "Software / Cloud"),
    ("GOOGL", "Alphabet Inc. (Class A)", "Internet / Ads"),
    ("META",  "Meta Platforms, Inc.",    "Internet / Social"),
    ("NVDA",  "NVIDIA Corp.",            "Semiconductors / AI"),
]
TICKER_TO_NAME = {t: n for t, n, _ in DEMO_UNIVERSE}


# ----- ANSI color helpers (degrade gracefully on non-TTY terminals) ----------
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


_CLR = _supports_color()
def c(s, code):  return f"\033[{code}m{s}\033[0m" if _CLR else s
def bold(s):     return c(s, "1")
def dim(s):      return c(s, "2")
def red(s):      return c(s, "31")
def green(s):    return c(s, "32")
def yellow(s):   return c(s, "33")
def blue(s):     return c(s, "34")
def magenta(s):  return c(s, "35")
def cyan(s):     return c(s, "36")


# ----- date parsing ----------------------------------------------------------
def _parse_date(raw: str | None) -> str | None:
    """Accepts None | 'today' | 'YYYY-MM-DD'. Returns 'YYYY-MM-DD' or None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    if raw in {"today", "now"}:
        return date.today().isoformat()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise SystemExit(red(f"[error] bad date '{raw}'. Use YYYY-MM-DD."))


def _clamp_rounds(n: int) -> int:
    return max(1, min(3, int(n)))


# ----- core flows ------------------------------------------------------------
def analyze(ticker: str, debate_rounds: int = 2, as_of_date: str | None = None,
            write_files: bool = True, verbose: bool = True,
            show_progress: bool = True) -> dict:
    rounds = _clamp_rounds(debate_rounds)
    date_str = as_of_date or date.today().isoformat()
    company = TICKER_TO_NAME.get(ticker.upper(), "—")

    if verbose:
        print(bold(cyan(
            f"\n>>> Multi-agent pipeline | {ticker.upper()} ({company}) "
            f"| as-of {date_str} | debate rounds = {rounds}"
        )))
        print(dim("    Agents will be announced as they run.\n"))

    sink = progress_mod.console_sink(use_color=_CLR) if show_progress else None
    if sink:
        progress_mod.register(sink)
    try:
        graph = build_graph()
        result = graph.invoke({
            "ticker": ticker.upper(),
            "as_of_date": date_str,
            "debate_rounds": rounds,
        })
    finally:
        if sink:
            progress_mod.unregister(sink)

    if write_files:
        files = write_reports(result)
        if verbose:
            print(green(f"\n[reports] written to {files['report.md']}"))

    if verbose:
        _print_run_summary(result, company)

    return result


def reflect() -> None:
    print(blue("\n[reflection] checking T+5 windows..."))
    processed = run_reflection()
    if not processed:
        print(dim("[reflection] nothing to evaluate."))
        return
    for p in processed:
        color = green if p["outcome_label"] == "win" else (red if p["outcome_label"] == "loss" else yellow)
        print(color(
            f"[reflection] {p['ticker']} (id={p['id']}): "
            f"{p['outcome_label']} ({p['outcome_return']:+.2%}) — post-mortem stored."
        ))


# ----- pretty print ---------------------------------------------------------
def _decision_color(decision: str):
    return {
        "BUY":          green,
        "OVERWEIGHT":   green,
        "HOLD":         yellow,
        "UNDERWEIGHT":  red,
        "SELL":         red,
    }.get(decision.upper(), lambda x: x)


def _print_run_summary(result: dict, company: str = "—") -> None:
    final = result.get("final_decision", "HOLD")
    color = _decision_color(final)
    conv = result.get("trader_conviction", 3)
    rerun = result.get("critic_triggered_rerun", False)
    rounds = result.get("debate_rounds_completed", "?")

    print()
    print(bold(cyan("=" * 72)))
    print(bold(cyan(f"   ANALYSIS COMPLETE — {result['ticker']}  ({company})")))
    print(bold(cyan(f"   as-of: {result.get('as_of_date','today')}")))
    print(bold(cyan("=" * 72)))
    print(f"  Final Decision:    {bold(color(final))}")
    print(f"  Trader Conviction: {bold(str(conv))}/5")
    print(f"  Debate rounds:     {rounds}")
    print(f"  Critic strength:   {result.get('critic_strength','WEAK')}"
          f"   {'(triggered re-debate)' if rerun else ''}")
    print()
    print(f"  Risk panel:")
    print(f"    Tail-Risk:  {result.get('tail_risk_verdict','NORMAL')}")
    print(f"    Macro:      {result.get('macro_verdict','MIXED')}")
    print(f"    Liquidity:  {result.get('liquidity_verdict','NORMAL')}")
    print()
    last_lines = (result.get("decision_summary") or "").splitlines()
    if last_lines:
        print(dim(f"  {last_lines[-1]}"))
    print(bold(cyan("=" * 72)))


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.fill(line, width) if line else "" for line in (text or "").splitlines())


# ----- menu actions ---------------------------------------------------------
def _list_recent(ticker: str, limit: int = 10):
    rows = get_recent_decisions(ticker, limit=limit)
    if not rows:
        print(dim(f"No prior decisions for {ticker.upper()}."))
        return
    print(bold(f"\nRecent decisions for {ticker.upper()} ({TICKER_TO_NAME.get(ticker.upper(),'')}):"))
    print(dim(f"{'date':<12}{'decision':<14}{'conv':<6}{'bias':<10}{'conf':<8}outcome"))
    for r in rows:
        outcome = (
            f"{r['outcome_label']} ({r['outcome_return']:+.2%})"
            if r.get("evaluated") else "pending"
        )
        color = _decision_color(r["final_decision"])
        print(f"{r['decision_date']:<12}"
              f"{color(r['final_decision']):<24}"
              f"{str(r.get('conviction','?')):<6}"
              f"{(r.get('bias') or '?'):<10}"
              f"{(r.get('confidence') or '?'):<8}"
              f"{outcome}")


def _list_pending():
    rows = get_pending_evaluations()
    if not rows:
        print(dim("No pending T+5 evaluations."))
        return
    print(bold("\nDecisions awaiting reflection:"))
    for r in rows:
        print(f"  id={r['id']:<5} {r['ticker']:<6} "
              f"decided {r['decision_date']} -> evaluate after {r['evaluate_after']} "
              f"({r['final_decision']})")


def _show_report(ticker: str, dt: str | None = None):
    base = os.path.join(REPORTS_ROOT, ticker.upper())
    if not os.path.isdir(base):
        print(dim(f"No reports for {ticker.upper()}."))
        return
    if not dt:
        dates = sorted(os.listdir(base), reverse=True)
        if not dates:
            print(dim(f"No reports for {ticker.upper()}."))
            return
        dt = dates[0]
    path = os.path.join(base, dt, "report.md")
    if not os.path.isfile(path):
        print(dim(f"No report at {path}"))
        return
    print(bold(cyan(f"\n----- {path} -----\n")))
    with open(path, "r", encoding="utf-8") as f:
        print(f.read())


def _ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    raw = input(f"{prompt} [{default}, range {lo}-{hi}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return max(lo, min(hi, v))
    except ValueError:
        return default


def _ask_str(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or (default or "")


def _ask_ticker_from_universe() -> str:
    """Show a numbered universe + free-text fallback."""
    print(bold("\nPick a company:"))
    for i, (tk, nm, sect) in enumerate(DEMO_UNIVERSE, 1):
        print(f"  {i}) {tk:<6} — {nm:<28} {dim(sect)}")
    print(f"  6) custom ticker")
    raw = input("Choose [1-6, default 1]: ").strip() or "1"
    if raw == "6":
        return _ask_str("Custom ticker", "AAPL").upper()
    try:
        idx = max(1, min(5, int(raw))) - 1
        return DEMO_UNIVERSE[idx][0]
    except ValueError:
        return DEMO_UNIVERSE[0][0]


def _ask_date() -> str | None:
    raw = input("Analysis date (YYYY-MM-DD or 'today') [today]: ").strip()
    if not raw or raw.lower() == "today":
        return date.today().isoformat()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
    except ValueError:
        print(red("  bad date, falling back to today"))
        return date.today().isoformat()


# ----- main menu ------------------------------------------------------------
MENU = """
{title}
  1) Analyze a ticker
  2) Run reflection on matured decisions
  3) List recent decisions for a ticker
  4) List pending T+5 evaluations
  5) View latest report for a ticker
  6) Quit
"""


def interactive_menu():
    print(bold(magenta("\n  Multi-Agent Trading Intelligence  ")))
    print(dim(f"  reports dir = {REPORTS_ROOT}    "
              f"db = {os.getenv('TRADINGAGENTS_DB_PATH', 'tradingagents_memory.db')}"))
    while True:
        print(MENU.format(title=bold(cyan("Menu"))))
        choice = input("Choose: ").strip()
        if choice == "1":
            ticker = _ask_ticker_from_universe()
            as_of = _ask_date()
            rounds = _ask_int("Debate rounds", 2, 1, 3)
            try:
                analyze(ticker, debate_rounds=rounds, as_of_date=as_of)
            except Exception as e:
                print(red(f"[error] {e}"))
        elif choice == "2":
            try:
                reflect()
            except Exception as e:
                print(red(f"[error] {e}"))
        elif choice == "3":
            ticker = _ask_str("Ticker").upper()
            if ticker:
                _list_recent(ticker)
        elif choice == "4":
            _list_pending()
        elif choice == "5":
            ticker = _ask_str("Ticker").upper()
            if ticker:
                dt = _ask_str("Date (YYYY-MM-DD, blank = latest)")
                _show_report(ticker, dt or None)
        elif choice in {"6", "q", "quit", "exit"}:
            print(dim("bye."))
            return
        else:
            print(dim("?"))


# ----- entrypoint -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-agent trading intelligence")
    parser.add_argument("ticker", nargs="?", default=None, help="Ticker symbol (one-shot mode)")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds (1-3)")
    parser.add_argument("--date", type=str, default=None,
                        help="Analysis date YYYY-MM-DD (default: today)")
    parser.add_argument("--reflect", action="store_true", help="Run reflection after analysis")
    parser.add_argument("--reflect-only", action="store_true", help="Only run reflection")
    parser.add_argument("--no-reports", action="store_true", help="Skip writing markdown reports")
    parser.add_argument("--quiet", action="store_true", help="Suppress live agent progress")
    parser.add_argument("--menu", action="store_true",
                        help="Force interactive menu even if a ticker is given")
    args = parser.parse_args()

    init_db()

    if args.reflect_only:
        reflect()
        return

    if args.menu or args.ticker is None:
        interactive_menu()
        return

    rounds = _clamp_rounds(args.rounds)
    as_of = _parse_date(args.date)
    analyze(
        args.ticker,
        debate_rounds=rounds,
        as_of_date=as_of,
        write_files=not args.no_reports,
        show_progress=not args.quiet,
    )
    if args.reflect:
        reflect()


if __name__ == "__main__":
    main()
