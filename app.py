"""Streamlit UI for the Multi-Agent Trading Intelligence system.

Run with:
    streamlit run app.py

Features
--------
- Dropdown of curated demo companies (ticker + company name + sector)
- Date picker (any past trading day) and debate-rounds slider (1-3)
- Live "agent panel" that lights up green as each stage completes
- Final-decision card with conviction, risk verdicts, and critic strength
- Tabs for every per-stage report (memory, three analysts, synthesis,
  bull/bear/arbiter, trader, three risk specialists, risk arbiter, critic)
- Decision-history browser with realised P&L from prior runs
- One-click reflection (T+5) sweep
"""
from __future__ import annotations
import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Dict, Any, List

import streamlit as st

# --- Project imports ---
from graphs.trading_graph import build_graph
from graphs import progress as progress_mod
from memory.memory_db import (
    init_db, get_recent_decisions, get_pending_evaluations,
)
from memory.reflection import run_reflection
from reporting.writer import write_reports, REPORTS_ROOT


# ============================================================================
# Curated demo universe
# ============================================================================
DEMO_UNIVERSE = [
    # ticker, company name, sector, short blurb
    ("AAPL",  "Apple Inc.",              "Consumer Tech",       "iPhone, services, Mac, wearables."),
    ("MSFT",  "Microsoft Corp.",         "Software / Cloud",    "Azure, Office, Windows, AI infra."),
    ("GOOGL", "Alphabet Inc. (Class A)", "Internet / Ads",      "Google Search, YouTube, Cloud."),
    ("META",  "Meta Platforms, Inc.",    "Internet / Social",   "Facebook, Instagram, WhatsApp, Reality Labs."),
    ("NVDA",  "NVIDIA Corp.",            "Semiconductors / AI", "GPUs, datacenter accelerators, CUDA."),
]
TICKERS = [t for t, *_ in DEMO_UNIVERSE]
TICKER_INDEX = {t: i for i, t in enumerate(TICKERS)}


# ============================================================================
# Styling
# ============================================================================
def inject_css():
    st.markdown(
        """
        <style>
            .stApp { background: #0E1117; }
            .agent-row {
                display: flex; align-items: center; justify-content: space-between;
                padding: 8px 14px; margin: 4px 0; border-radius: 8px;
                border: 1px solid #2A2F3A;
                background: #161B22;
                font-family: 'Inter', system-ui, sans-serif;
            }
            .agent-row.pending { color: #6e7681; }
            .agent-row.running { background: #1f2a44; color: #e6edf3; border-color: #58a6ff; }
            .agent-row.done    { background: #14261c; color: #a6e3a1; border-color: #2ea043; }
            .agent-status { font-size: 12px; opacity: 0.85; }
            .verdict-card {
                background: #161B22; border: 1px solid #2A2F3A; border-radius: 10px;
                padding: 14px 18px; margin-bottom: 8px;
            }
            .verdict-card .label { color: #8b949e; font-size: 12px; text-transform: uppercase; }
            .verdict-card .value { color: #e6edf3; font-size: 22px; font-weight: 700; }
            .verdict-card.good .value { color: #3fb950; }
            .verdict-card.bad  .value { color: #f85149; }
            .verdict-card.warn .value { color: #d29922; }
            .decision-banner {
                padding: 18px 22px; border-radius: 12px;
                font-family: 'Georgia', serif; font-size: 26px; font-weight: 700;
                text-align: center; margin-bottom: 12px;
            }
            .decision-banner.buy        { background: #14261c; color: #3fb950; border: 2px solid #2ea043; }
            .decision-banner.overweight { background: #14261c; color: #56d364; border: 2px solid #238636; }
            .decision-banner.hold       { background: #2c2718; color: #d29922; border: 2px solid #9e6a03; }
            .decision-banner.underweight{ background: #2d1b1b; color: #ff7b72; border: 2px solid #da3633; }
            .decision-banner.sell       { background: #2d1b1b; color: #f85149; border: 2px solid #b62324; }
            .ticker-pill {
                display: inline-block; background: #1f6feb; color: white;
                padding: 2px 10px; border-radius: 99px; font-size: 12px; margin-right: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Pipeline runner (background-threaded so UI can update live)
# ============================================================================
AGENT_ORDER = [
    "Memory layer",
    "Technical analyst",
    "Fundamentals analyst",
    "News & sentiment analyst",
    "Chain-of-thought synthesis",
    "Debate (bull/bear/arbiter)",
    "Trader",
    "Risk panel (3 specialists)",
    "Risk arbiter",
    "Devil's advocate critic",
    "Final decision & persist",
]


def run_pipeline_threaded(ticker: str, as_of: str, rounds: int,
                          events: list, result_box: dict,
                          done_flag: threading.Event):
    """Run the LangGraph pipeline in a thread, recording progress events."""
    sink = progress_mod.list_sink(events)
    progress_mod.register(sink)
    try:
        graph = build_graph()
        out = graph.invoke({
            "ticker": ticker.upper(),
            "as_of_date": as_of,
            "debate_rounds": rounds,
        })
        result_box["result"] = out
    except Exception as e:
        result_box["error"] = str(e)
    finally:
        progress_mod.unregister(sink)
        done_flag.set()


def render_agent_panel(events: List[Dict[str, Any]], placeholder):
    """Render the agent status list using the events captured so far."""
    started, completed = set(), set()
    durations: Dict[str, float] = {}
    starts: Dict[str, datetime] = {}
    debate_round_label = None

    for ev in events:
        label = ev["label"]
        # Surface debate-round substeps separately
        if label.startswith("Bull researcher") or label.startswith("Bear researcher") \
                or label == "Neutral arbiter":
            debate_round_label = label
            if ev["phase"] == "start":
                starts[label] = datetime.fromisoformat(ev["ts"])
            elif ev["phase"] == "done":
                t0 = starts.get(label)
                if t0:
                    durations[label] = (datetime.fromisoformat(ev["ts"]) - t0).total_seconds()
            continue
        if ev["phase"] == "start":
            started.add(label)
            starts[label] = datetime.fromisoformat(ev["ts"])
        elif ev["phase"] == "done":
            completed.add(label)
            t0 = starts.get(label)
            if t0:
                durations[label] = (datetime.fromisoformat(ev["ts"]) - t0).total_seconds()

    rows = []
    for agent in AGENT_ORDER:
        if agent in completed:
            took = durations.get(agent)
            tail = f"{took:.1f}s" if took is not None else "done"
            rows.append(
                f'<div class="agent-row done">'
                f'<span>✓ {agent}</span>'
                f'<span class="agent-status">{tail}</span>'
                f'</div>'
            )
        elif agent in started:
            extra = ""
            if agent == "Debate (bull/bear/arbiter)" and debate_round_label:
                extra = f" — {debate_round_label}"
            rows.append(
                f'<div class="agent-row running">'
                f'<span>▶ {agent}{extra}</span>'
                f'<span class="agent-status">running...</span>'
                f'</div>'
            )
        else:
            rows.append(
                f'<div class="agent-row pending">'
                f'<span>○ {agent}</span>'
                f'<span class="agent-status">queued</span>'
                f'</div>'
            )
    placeholder.markdown("\n".join(rows), unsafe_allow_html=True)


# ============================================================================
# Result rendering
# ============================================================================
def render_decision_banner(result: dict, company: str):
    final = (result.get("final_decision") or "HOLD").upper()
    cls_map = {
        "BUY": "buy", "OVERWEIGHT": "overweight",
        "HOLD": "hold",
        "UNDERWEIGHT": "underweight", "SELL": "sell",
    }
    cls = cls_map.get(final, "hold")
    conv = result.get("trader_conviction", 3)
    rounds = result.get("debate_rounds_completed", "?")
    crit = result.get("critic_strength", "WEAK")
    rerun = result.get("critic_triggered_rerun", False)
    rerun_tag = " · re-debate triggered" if rerun else ""
    st.markdown(
        f'<div class="decision-banner {cls}">'
        f'<span class="ticker-pill">{result["ticker"]}</span>'
        f'{company} → {final} <small style="font-weight:400;">'
        f' (conviction {conv}/5 · {rounds} round(s) · critic {crit}{rerun_tag})</small>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_risk_cards(result: dict):
    cols = st.columns(4)

    def card(col, label, value, kind):
        col.markdown(
            f'<div class="verdict-card {kind}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    tail = result.get("tail_risk_verdict", "NORMAL")
    macro = result.get("macro_verdict", "MIXED")
    liq = result.get("liquidity_verdict", "NORMAL")
    adj = result.get("risk_adjustment", "KEEP")

    card(cols[0], "Tail-Risk", tail,
         "bad" if tail == "ELEVATED" else "good" if tail == "MUTED" else "warn")
    card(cols[1], "Macro/Regime", macro,
         "bad" if macro == "HOSTILE" else "good" if macro == "SUPPORTIVE" else "warn")
    card(cols[2], "Liquidity", liq,
         "bad" if liq == "STRESSED" else "good" if liq == "DEEP" else "warn")
    card(cols[3], "Risk Arbiter", adj,
         "bad" if "OVERRIDE" in adj or "TWO" in adj
         else "warn" if "ONE" in adj else "good")


def render_market_context(result: dict):
    risk_inputs = result.get("risk_inputs") or {}
    market = risk_inputs.get("market", {}) or {}
    tk_vol = risk_inputs.get("ticker_vol_annualized")
    tk_vol_label = risk_inputs.get("ticker_vol_label", "?")

    def pct(x):
        try:
            return f"{float(x):+.2%}"
        except (TypeError, ValueError):
            return "?"

    cols = st.columns(4)
    cols[0].metric("Regime",     market.get("regime", "?"))
    cols[1].metric("SPY",        f"${market.get('spy_last','?')}",
                   delta=f"vs SMA50 ${market.get('spy_sma50','?')}")
    cols[2].metric("60d DD",     pct(market.get("drawdown_60d")))
    cols[3].metric(f"Ticker vol ({tk_vol_label})",
                   pct(tk_vol))


def render_stage_tabs(result: dict):
    stages = [
        ("📋 Memory",         result.get("memory_context", "")),
        ("📈 Technical",      result.get("technical_report", "")),
        ("📊 Fundamentals",   result.get("fundamental_report", "")),
        ("📰 News",           result.get("news_sentiment_report", "")),
        ("🧠 Synthesis",      result.get("synthesis_output", "")),
        ("🐂 Bull",           result.get("bull_argument", "")),
        ("🐻 Bear",           result.get("bear_argument", "")),
        ("⚖️  Arbiter",       result.get("arbiter_verdict", "")),
        ("🤝 Trader",         result.get("trader_output", "")),
        ("⚡ Tail-Risk",      result.get("tail_risk_output", "")),
        ("🌐 Macro",          result.get("macro_output", "")),
        ("💧 Liquidity",      result.get("liquidity_output", "")),
        ("🛡️  Risk Arbiter",  result.get("risk_output", "")),
        ("👹 Critic",         result.get("critic_output", "")),
    ]
    tabs = st.tabs([s[0] for s in stages])
    for tab, (title, content) in zip(tabs, stages):
        with tab:
            st.markdown(content if content.strip() else "_no output_")


# ============================================================================
# History panel
# ============================================================================
def render_history(ticker: str):
    rows = get_recent_decisions(ticker, limit=20)
    if not rows:
        st.info(f"No prior decisions stored for {ticker}.")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    keep = ["decision_date", "final_decision", "conviction", "bias", "confidence",
            "entry_price", "outcome_label", "outcome_return", "evaluate_after"]
    df = df[[c for c in keep if c in df.columns]]
    df["outcome_return"] = df["outcome_return"].apply(
        lambda x: f"{x:+.2%}" if isinstance(x, (int, float)) else "—"
    )
    df["entry_price"] = df["entry_price"].apply(
        lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "—"
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# Page
# ============================================================================
def main():
    st.set_page_config(
        page_title="Multi-Agent Trading Intelligence",
        page_icon="🤖",
        layout="wide",
    )
    inject_css()
    init_db()

    st.title("🤖 Multi-Agent Trading Intelligence")
    st.caption(
        "Memory → 3 parallel analysts → synthesis → bull/bear debate → trader → "
        "specialist risk panel → risk arbiter → devil's-advocate critic → final decision."
    )

    # --- Sidebar: inputs ---
    with st.sidebar:
        st.header("⚙️ Run configuration")

        labels = [f"{t} — {n}" for t, n, *_ in DEMO_UNIVERSE]
        choice = st.selectbox(
            "Company",
            options=range(len(DEMO_UNIVERSE)),
            format_func=lambda i: labels[i],
            index=4,  # default to NVDA — most fun for demo
        )
        ticker, company, sector, blurb = DEMO_UNIVERSE[choice]
        st.caption(f"**{company}** · {sector}")
        st.caption(blurb)

        as_of = st.date_input(
            "Analysis date",
            value=date.today(),
            min_value=date.today() - timedelta(days=730),
            max_value=date.today(),
            help="The pipeline treats this date as 'today'.",
        )

        rounds = st.slider(
            "Debate rounds (bull ↔ bear)", min_value=1, max_value=3, value=2,
            help="More rounds = sharper rebuttal but higher LLM cost.",
        )

        st.divider()
        st.markdown("**Past T+5 evaluations**")
        pending = get_pending_evaluations()
        st.caption(f"{len(pending)} decision(s) waiting to be graded.")
        if st.button("Run reflection now", use_container_width=True):
            with st.spinner("Grading matured decisions..."):
                processed = run_reflection()
            if processed:
                st.success(f"Graded {len(processed)} decision(s).")
                for p in processed:
                    icon = "✅" if p["outcome_label"] == "win" else \
                           "❌" if p["outcome_label"] == "loss" else "➖"
                    st.write(
                        f"{icon} **{p['ticker']}** {p['outcome_label']} "
                        f"({p['outcome_return']:+.2%})"
                    )
            else:
                st.info("Nothing to evaluate yet.")

    # --- Main area: tabs ---
    tab_run, tab_history, tab_about = st.tabs(["▶ Run analysis", "🗂️ Past decisions", "ℹ️ About"])

    # ---------- RUN TAB ----------
    with tab_run:
        st.subheader(f"{ticker} — {company}")
        st.markdown(
            f"<span class='ticker-pill'>{ticker}</span> "
            f"<span style='color:#8b949e'>as of <b>{as_of.isoformat()}</b> · "
            f"{rounds} debate round(s)</span>",
            unsafe_allow_html=True,
        )

        run_clicked = st.button(
            f"🚀 Analyze {ticker} ({company})",
            type="primary",
            use_container_width=True,
        )

        col_left, col_right = st.columns([1, 2])

        agent_panel = col_left.empty()
        with col_left:
            st.markdown("### 🏃 Agents")
            # Render initial empty panel
        result_area = col_right.container()

        if run_clicked:
            events: List[Dict[str, Any]] = []
            result_box: Dict[str, Any] = {}
            done_flag = threading.Event()

            t = threading.Thread(
                target=run_pipeline_threaded,
                args=(ticker, as_of.isoformat(), rounds, events, result_box, done_flag),
                daemon=True,
            )
            t.start()

            # Live render loop
            while not done_flag.is_set():
                render_agent_panel(events, agent_panel)
                time.sleep(0.4)
            t.join(timeout=2.0)
            render_agent_panel(events, agent_panel)

            if "error" in result_box:
                with result_area:
                    st.error(f"Pipeline failed: {result_box['error']}")
                return

            result = result_box.get("result")
            if not result:
                with result_area:
                    st.error("No result returned.")
                return

            # Persist reports to disk so the file viewer can find them later
            try:
                write_reports(result)
            except Exception as e:
                st.warning(f"Reports not written: {e}")

            with result_area:
                render_decision_banner(result, company)

                st.markdown("#### 🛡 Risk panel")
                render_risk_cards(result)

                st.markdown("#### 🌐 Market context")
                render_market_context(result)

                st.markdown("#### 📚 Stage outputs")
                render_stage_tabs(result)
        else:
            # Pre-run static panel
            render_agent_panel([], agent_panel)
            with result_area:
                st.info("Configure the run on the left, then click **Analyze** to start.")

    # ---------- HISTORY TAB ----------
    with tab_history:
        st.subheader("Past decisions")
        hist_ticker = st.selectbox(
            "Ticker", options=TICKERS, index=TICKER_INDEX.get(ticker, 0),
            format_func=lambda t: f"{t} — {dict((x[0], x[1]) for x in DEMO_UNIVERSE)[t]}",
        )
        render_history(hist_ticker)

    # ---------- ABOUT TAB ----------
    with tab_about:
        st.subheader("About this system")
        st.markdown(
            """
**Multi-Agent Trading Intelligence** orchestrates eleven specialised agents
into a directed pipeline:

1. **Memory layer** — pulls the last five decisions for this ticker plus any
   T+5 post-mortems, and feeds them into every downstream prompt.
2. **Three analysts** run in parallel — Technical (RSI, MACD, Bollinger %B,
   VWAP, volume z-score), Fundamentals (SEC EDGAR filings), and
   News & Sentiment (Alpha Vantage feed).
3. **Chain-of-thought synthesis** combines the three reports into a structured
   bias / confidence / risks / takeaway document.
4. **Debate** — bull and bear take turns over **1–3 rounds**, each rebutting the
   prior round, then a neutral arbiter scores both sides.
5. **Trader** issues a five-bucket rating (Buy / Overweight / Hold / Underweight /
   Sell) plus a 1–5 conviction.
6. **Risk panel** — three specialists (Tail-Risk, Macro/Regime, Liquidity)
   examine the same data in parallel.
7. **Risk arbiter** maps the trader's rating to KEEP / DOWNGRADE_ONE_STEP /
   DOWNGRADE_TWO_STEPS / OVERRIDE_TO_HOLD.
8. **Devil's-advocate critic** seeks the strongest counter-thesis. If rated
   STRONG, it privately reruns debate → trader → risk panel → risk arbiter
   **once** with the challenge injected. The loop is bounded.
9. **Final decision** is persisted to SQLite with an evaluate-after date set
   to five trading days ahead.
10. **Reflection** (offline) grades matured decisions, labels them
    win/flat/loss at ±3%, writes a post-mortem, and feeds it into future
    memory contexts — closing the learning loop.
            """
        )


if __name__ == "__main__":
    main()
