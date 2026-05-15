"""Reflection agent: looks up decisions whose T+5 window has elapsed,
fetches the realized return, and writes a post-mortem back to the memory DB."""
from datetime import datetime
from llm_clients.factory import get_llm
from memory.memory_db import (
    get_pending_evaluations,
    update_postmortem,
)
from dataflows.market_data import get_price_history


REFLECTION_PROMPT = """You are a reflection agent reviewing a past trading decision.

Decision metadata:
- Ticker: {ticker}
- Decision date: {decision_date}
- Decision: {final_decision} (conviction {conviction}/5)
- Provisional bias: {bias}, confidence: {confidence}
- Entry price: {entry_price}
- Outcome (T+5): price {outcome_price}, return {outcome_return:+.2%}
- Outcome label: {outcome_label}

Synthesis at decision time:
{synthesis}

Bull argument:
{bull}

Bear argument:
{bear}

Arbiter verdict:
{arbiter}

Risk manager notes:
{risk}

Write a concise post-mortem in 4 sections:
1. What we got right (or wrong)
2. Which signal drove the outcome — technical, fundamental, news/sentiment, or market regime
3. What we should weight differently next time for this ticker
4. One actionable lesson (one sentence)

Be specific. Reference the actual numbers in the decision."""


def _label_return(r: float) -> str:
    if r > 0.03:
        return "win"
    if r < -0.03:
        return "loss"
    return "flat"


def _get_close_on_or_after(ticker: str, target_date: str) -> float | None:
    df = get_price_history(ticker, lookback_days=30)
    df = df.sort_index()
    target = datetime.fromisoformat(target_date).date()
    for idx, row in df.iterrows():
        if idx.date() >= target:
            return float(row["close"])
    if not df.empty:
        return float(df["close"].iloc[-1])
    return None


def run_reflection(config: dict = None) -> list:
    """Iterate over decisions whose evaluation window has elapsed and write post-mortems."""
    pending = get_pending_evaluations()
    if not pending:
        return []

    llm = get_llm(config or {})
    processed = []

    for d in pending:
        ticker = d["ticker"]
        try:
            outcome_price = _get_close_on_or_after(ticker, d["evaluate_after"])
        except Exception as e:
            print(f"[reflection] price lookup failed for {ticker}: {e}")
            continue

        if outcome_price is None or not d.get("entry_price"):
            continue

        ret = (outcome_price - d["entry_price"]) / d["entry_price"]
        label = _label_return(ret)

        prompt = REFLECTION_PROMPT.format(
            ticker=ticker,
            decision_date=d["decision_date"],
            final_decision=d["final_decision"],
            conviction=d.get("conviction", "?"),
            bias=d.get("bias", "?"),
            confidence=d.get("confidence", "?"),
            entry_price=d["entry_price"],
            outcome_price=outcome_price,
            outcome_return=ret,
            outcome_label=label,
            synthesis=(d.get("synthesis") or "")[:2000],
            bull=(d.get("bull_argument") or "")[:1500],
            bear=(d.get("bear_argument") or "")[:1500],
            arbiter=(d.get("arbiter_verdict") or "")[:1500],
            risk=(d.get("risk_notes") or "")[:1000],
        )

        resp = llm.invoke([
            {"role": "system", "content": "You write disciplined trading post-mortems."},
            {"role": "user", "content": prompt},
        ])
        postmortem_text = resp.content

        update_postmortem(
            decision_id=d["id"],
            outcome_price=outcome_price,
            outcome_return=ret,
            outcome_label=label,
            postmortem=postmortem_text,
        )
        processed.append({
            "id": d["id"],
            "ticker": ticker,
            "outcome_label": label,
            "outcome_return": ret,
        })

    return processed
