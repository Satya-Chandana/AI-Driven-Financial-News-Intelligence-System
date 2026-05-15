"""Neutral arbiter: scores argument quality, flags contradictions,
and produces a structured verdict for the trader."""
from llm_clients.factory import get_llm

ARBITER_SYSTEM = """You are a neutral arbiter. Two researchers (bull and bear)
have presented their final arguments after multiple rounds of debate.

Your job:
- Score each side's argument quality (1-5) on three axes: evidence,
  rigor (logical chain quality), and address of opposing points.
- Flag any factual contradictions or unsupported claims.
- Identify which side carries the stronger evidence-based case.
- DO NOT introduce new evidence. Reason only over what was presented.

Output must follow the exact structure below. Be strict and concise."""

ARBITER_TEMPLATE = """## Bull Score
- Evidence: x/5
- Rigor: x/5
- Addresses bear: x/5
- Comment: <one sentence>

## Bear Score
- Evidence: x/5
- Rigor: x/5
- Addresses bull: x/5
- Comment: <one sentence>

## Contradictions / Unsupported Claims
- bullet list, or "none material"

## Verdict
State exactly one: bull / bear / tie

## Verdict Strength
State exactly one: strong / moderate / weak

## Reasoning
2-4 sentences explaining the verdict, citing the strongest argument that decided it.
"""


def run_arbiter(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")

    prompt = f"""Ticker: {ticker}

Final BULL argument:
---
{bull}
---

Final BEAR argument:
---
{bear}
---

Return your verdict in this exact structure:

{ARBITER_TEMPLATE}
"""

    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": ARBITER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"arbiter_verdict": resp.content}
