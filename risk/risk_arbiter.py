"""Risk Arbiter: arbitrates the three risk specialists and emits the final
risk-adjusted decision (BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL)."""
import re
from llm_clients.factory import get_llm

ARBITER_SYSTEM = """You are the Risk Manager. Three specialist analysts have
each given you a one-sided view:
- Tail-Risk Analyst (worst-case bias)
- Macro / Regime Analyst (market context bias)
- Liquidity / Microstructure Analyst (execution bias)

Each is intentionally one-sided. Your job is to synthesize their inputs into
a single risk-adjusted decision that respects the trader's call when conditions
are normal but pushes back when ANY specialist raises a credible alarm.

Decision rules:
- If 2+ specialists raise the strongest alarm (ELEVATED / HOSTILE / STRESSED),
  force HOLD or downgrade two steps (BUY -> HOLD, SELL -> HOLD).
- If exactly 1 specialist raises the strongest alarm and trader conviction
  is <= 3, downgrade one step.
- If no specialist raises an alarm, KEEP the trader's call.
- Conviction <= 2 should never sit at BUY or SELL — collapse to OVERWEIGHT/UNDERWEIGHT or HOLD.

Be explicit about which specialists drove the adjustment."""

ARBITER_TEMPLATE = """## Adjustment
<one of: KEEP / DOWNGRADE_ONE_STEP / DOWNGRADE_TWO_STEPS / OVERRIDE_TO_HOLD>

## Final Decision
<one of: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL>

## Risk Notes
4-6 sentences. Cite which specialist verdicts drove the call (or the decision
to leave the trader's call alone). Reference the strongest single number from
the inputs.
"""


def _parse_final(text: str) -> str:
    m = re.search(r"##\s*Final Decision\s*\n+\s*([A-Za-z]+)", text)
    if not m:
        return "HOLD"
    raw = m.group(1).strip().upper()
    return raw if raw in {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"} else "HOLD"


def _parse_adjustment(text: str) -> str:
    m = re.search(r"##\s*Adjustment\s*\n+\s*([A-Z_]+)", text)
    return m.group(1).strip().upper() if m else "KEEP"


def run_risk_arbiter(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    trader_rating = state.get("trader_rating", "HOLD")
    conviction = state.get("trader_conviction", 3)

    tail_v = state.get("tail_risk_verdict", "NORMAL")
    macro_v = state.get("macro_verdict", "MIXED")
    liq_v = state.get("liquidity_verdict", "NORMAL")

    tail_text = state.get("tail_risk_output", "")
    macro_text = state.get("macro_output", "")
    liq_text = state.get("liquidity_output", "")

    prompt = f"""Ticker: {ticker}
Trader's call: {trader_rating} (conviction {conviction}/5)

Specialist verdicts:
- Tail-Risk: {tail_v}
- Macro/Regime: {macro_v}
- Liquidity: {liq_v}

Tail-Risk Analyst report:
---
{tail_text}
---

Macro / Regime Analyst report:
---
{macro_text}
---

Liquidity / Microstructure Analyst report:
---
{liq_text}
---

Return your decision in this exact structure:

{ARBITER_TEMPLATE}
"""
    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": ARBITER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    text = resp.content
    return {
        "risk_output": text,
        "risk_adjustment": _parse_adjustment(text),
        "final_decision": _parse_final(text),
    }
