"""Persists per-stage markdown reports + a summary, similar to the reference.

Layout:
    reports/
      <TICKER>/
        <YYYY-MM-DD>/
          00_memory.md
          01_technical.md
          02_fundamental.md
          03_news_sentiment.md
          04_synthesis.md
          05_bull.md
          06_bear.md
          07_arbiter.md
          08_trader.md
          09_risk_tail.md
          10_risk_macro.md
          11_risk_liquidity.md
          12_risk_arbiter.md
          13_critic.md
          report.md            <- top-level summary
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict

REPORTS_ROOT = os.getenv("TRADINGAGENTS_REPORTS_DIR", "reports")


def _safe_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _section(title: str, body: str) -> str:
    return f"# {title}\n\n{body or '_no output_'}\n"


def write_reports(state: Dict) -> Dict[str, str]:
    ticker = state["ticker"].upper()
    date = state.get("as_of_date") or datetime.utcnow().date().isoformat()
    base = os.path.join(REPORTS_ROOT, ticker, date)

    files = {}

    stages = [
        ("00_memory.md",            "Memory Context",            state.get("memory_context", "")),
        ("01_technical.md",         "Technical Analyst",         state.get("technical_report", "")),
        ("02_fundamental.md",       "Fundamental Analyst",       state.get("fundamental_report", "")),
        ("03_news_sentiment.md",    "News + Sentiment Analyst",  state.get("news_sentiment_report", "")),
        ("04_synthesis.md",         "CoT Synthesis",             state.get("synthesis_output", "")),
        ("05_bull.md",              "Bull Researcher",           state.get("bull_argument", "")),
        ("06_bear.md",              "Bear Researcher",           state.get("bear_argument", "")),
        ("07_arbiter.md",           "Neutral Arbiter",           state.get("arbiter_verdict", "")),
        ("08_trader.md",            "Trader",                    state.get("trader_output", "")),
        ("09_risk_tail.md",         "Risk: Tail-Risk Analyst",   state.get("tail_risk_output", "")),
        ("10_risk_macro.md",        "Risk: Macro / Regime",      state.get("macro_output", "")),
        ("11_risk_liquidity.md",    "Risk: Liquidity",           state.get("liquidity_output", "")),
        ("12_risk_arbiter.md",      "Risk Manager (Arbiter)",    state.get("risk_output", "")),
        ("13_critic.md",            "Devil's Advocate Critic",   state.get("critic_output", "")),
    ]

    for fname, title, body in stages:
        path = os.path.join(base, fname)
        _safe_write(path, _section(title, body))
        files[fname] = path

    metrics_dump = {
        "technical_metrics": state.get("technical_metrics"),
        "news_sentiment_metrics": _strip_articles(state.get("news_sentiment_metrics")),
        "risk_inputs": _serializable_risk(state.get("risk_inputs")),
    }
    metrics_path = os.path.join(base, "metrics.json")
    _safe_write(metrics_path, json.dumps(metrics_dump, indent=2, default=str))
    files["metrics.json"] = metrics_path

    summary = _build_summary(state)
    summary_path = os.path.join(base, "report.md")
    _safe_write(summary_path, summary)
    files["report.md"] = summary_path

    return files


def _strip_articles(metrics):
    if not metrics:
        return None
    out = dict(metrics)
    out.pop("top_articles", None)
    return out


def _serializable_risk(ri):
    if not ri:
        return None
    return {k: v for k, v in ri.items() if k not in ("ticker_df", "spy_df")}


def _build_summary(state: Dict) -> str:
    ticker = state["ticker"].upper()
    date = state.get("as_of_date") or datetime.utcnow().date().isoformat()
    final = state.get("final_decision", "HOLD")
    conviction = state.get("trader_conviction", 3)
    bias = _grep_section(state.get("synthesis_output", ""), "Final Bias")
    confidence = _grep_section(state.get("synthesis_output", ""), "Confidence")
    critic_strength = state.get("critic_strength", "WEAK")
    rerun = state.get("critic_triggered_rerun", False)
    rounds = state.get("debate_rounds_completed", "?")

    risk_inputs = state.get("risk_inputs") or {}
    market = risk_inputs.get("market", {})

    lines = [
        f"# Trading Decision Report — {ticker} ({date})",
        "",
        "## Summary",
        f"- **Final Decision**: **{final}**",
        f"- Trader conviction: {conviction}/5",
        f"- Synthesis bias: {bias} | Confidence: {confidence}",
        f"- Debate rounds completed: {rounds}",
        f"- Critic strength: {critic_strength} | Triggered re-debate: {rerun}",
        "",
        "## Risk Panel Verdicts",
        f"- Tail-Risk: **{state.get('tail_risk_verdict','NORMAL')}**",
        f"- Macro / Regime: **{state.get('macro_verdict','MIXED')}**",
        f"- Liquidity: **{state.get('liquidity_verdict','NORMAL')}**",
        f"- Risk arbiter adjustment: **{state.get('risk_adjustment','KEEP')}**",
        "",
        "## Market Context",
        f"- Regime: {market.get('regime','?')}",
        f"- SPY: ${market.get('spy_last','?')} | SMA50: ${market.get('spy_sma50','?')} | SMA200: ${market.get('spy_sma200','?')}",
        f"- 60d drawdown: {_pct(market.get('drawdown_60d'))} | SPY vol: {_pct(market.get('spy_vol_annualized'))}",
        f"- Ticker realized vol: {_pct(risk_inputs.get('ticker_vol_annualized'))} ({risk_inputs.get('ticker_vol_label','?')})",
        "",
        "## Stage Index",
        "- [Memory context](00_memory.md)",
        "- [Technical analyst](01_technical.md)",
        "- [Fundamental analyst](02_fundamental.md)",
        "- [News + sentiment](03_news_sentiment.md)",
        "- [CoT synthesis](04_synthesis.md)",
        "- [Bull researcher](05_bull.md)",
        "- [Bear researcher](06_bear.md)",
        "- [Neutral arbiter](07_arbiter.md)",
        "- [Trader](08_trader.md)",
        "- [Risk: tail-risk](09_risk_tail.md)",
        "- [Risk: macro / regime](10_risk_macro.md)",
        "- [Risk: liquidity](11_risk_liquidity.md)",
        "- [Risk manager (arbiter)](12_risk_arbiter.md)",
        "- [Devil's advocate critic](13_critic.md)",
        "- [Quantitative metrics dump](metrics.json)",
        "",
        "## Final Risk Notes",
        state.get("risk_output", "_no output_"),
    ]
    return "\n".join(lines) + "\n"


def _grep_section(text: str, heading: str) -> str:
    if not text:
        return "?"
    import re
    m = re.search(rf"##\s*{re.escape(heading)}\s*\n+\s*([^\n]+)", text)
    return m.group(1).strip() if m else "?"


def _pct(x):
    try:
        return f"{float(x):+.2%}"
    except (TypeError, ValueError):
        return "?"
