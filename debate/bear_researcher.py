"""Bear researcher: builds the strongest evidence-based short / avoid thesis."""
from llm_clients.factory import get_llm

BEAR_SYSTEM = """You are a bear-side equity researcher. Your job is to construct
the strongest evidence-based case to SELL, AVOID, or be SHORT this ticker.

Rules:
- Anchor every claim to a specific signal from the analyst reports or synthesis.
- Address the bull's strongest points directly. Do not strawman.
- Do not invent facts. Speculation must be labeled.
- Be sharp and concise. No filler."""


def run_bear(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    synthesis = state.get("synthesis_output", "")
    tech = state.get("technical_report", "")
    fund = state.get("fundamental_report", "")
    news = state.get("news_sentiment_report", "")
    prior_bull = state.get("bull_argument", "")
    round_num = state.get("debate_round", 1)

    critic_challenge = state.get("critic_challenge", "")
    critic_block = ""
    if critic_challenge:
        critic_block = f"""
A devil's-advocate critic flagged the following concern about the candidate decision:
---
{critic_challenge}
---

Address this concern directly. If it strengthens the bear case, sharpen accordingly.
"""

    rebuttal_block = ""
    if prior_bull:
        rebuttal_block = f"""
The bull most recently argued:
---
{prior_bull}
---

Address their strongest specific points head-on. Do not repeat prior bear
argument verbatim — sharpen it.
"""

    prompt = f"""Ticker: {ticker}
Debate round: {round_num}

CoT Synthesis:
{synthesis}

Technical report:
{tech}

Fundamental report:
{fund}

News + sentiment report:
{news}

{rebuttal_block}
{critic_block}

Write the bear case in 3 short sections:
## Thesis
One paragraph. State the bearish thesis and time horizon implied.

## Key Evidence
Bulleted, each tied to a concrete signal.

## Rebuttal of the Bull
Address their strongest claim head-on.
"""

    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": BEAR_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"bear_argument": resp.content}
