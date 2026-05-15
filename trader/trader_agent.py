"""Trader agent: synthesizes bull, bear, and arbiter outputs into a
provisional rating + conviction (1-5)."""
import re
from llm_clients.factory import get_llm

TRADER_SYSTEM = """You are a portfolio trader. You receive:
- The full bull and bear arguments from the debate
- The neutral arbiter's verdict and scores
- The CoT synthesis and the past decision history for this ticker

Output a disciplined trading call. Rules:
- The rating must be one of: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL.
- Conviction is an integer 1-5 (1=very low, 5=very high). Treat it as your
  probability that the directional view is right over the next ~5 trading days.
- Lean on the arbiter's verdict — but you can override it if the underlying
  evidence justifies a different sizing.
- Reflect honestly when evidence conflicts: low conviction is the correct answer
  in those cases.
- Do not invent new facts."""

TRADER_TEMPLATE = """## Rating
<one of: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL>

## Conviction
<integer 1-5>

## Rationale
3-5 sentences. Cite the strongest piece of evidence that decided the rating
and the most important caveat.

## Position Notes
1-2 sentences on entry timing, size posture, and the one signal that would
invalidate this call.
"""


def _parse_rating(text: str) -> str:
    m = re.search(r"##\s*Rating\s*\n+\s*([A-Za-z]+)", text)
    if not m:
        return "HOLD"
    raw = m.group(1).strip().upper()
    if raw not in {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}:
        return "HOLD"
    return raw


def _parse_conviction(text: str) -> int:
    m = re.search(r"##\s*Conviction\s*\n+\s*(\d)", text)
    if not m:
        return 3
    val = int(m.group(1))
    return max(1, min(5, val))


def run_trader(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    synthesis = state.get("synthesis_output", "")
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")
    arbiter = state.get("arbiter_verdict", "")
    memory_ctx = state.get("memory_context", "")

    prompt = f"""Ticker: {ticker}

Past decision history (memory context):
{memory_ctx}

CoT Synthesis:
{synthesis}

Bull argument (final):
{bull}

Bear argument (final):
{bear}

Arbiter verdict:
{arbiter}

Return your call in this exact structure:

{TRADER_TEMPLATE}
"""

    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": TRADER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    text = resp.content

    return {
        "trader_output": text,
        "trader_rating": _parse_rating(text),
        "trader_conviction": _parse_conviction(text),
    }
