"""Bull researcher: builds the strongest evidence-based long thesis."""
from llm_clients.factory import get_llm

BULL_SYSTEM = """You are a bull-side equity researcher. Your job is to construct
the strongest evidence-based case to BUY or be LONG this ticker.

Rules:
- Anchor every claim to a specific signal from the analyst reports or synthesis
  (technical indicators, fundamentals, news, sentiment).
- Acknowledge the bear's strongest points and rebut them on the merits — do not
  ignore them and do not strawman them.
- Do not invent facts. If a claim is speculative, label it as such.
- Be sharp and concise. No filler."""


def run_bull(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    synthesis = state.get("synthesis_output", "")
    tech = state.get("technical_report", "")
    fund = state.get("fundamental_report", "")
    news = state.get("news_sentiment_report", "")
    prior_bull = state.get("bull_argument", "")
    prior_bear = state.get("bear_argument", "")
    round_num = state.get("debate_round", 1)

    critic_challenge = state.get("critic_challenge", "")
    critic_block = ""
    if critic_challenge:
        critic_block = f"""
A devil's-advocate critic flagged the following concern about the candidate decision:
---
{critic_challenge}
---

Address this concern directly in your rebuttal section. Do not dodge it.
"""

    rebuttal_block = ""
    if prior_bear:
        rebuttal_block = f"""
The bear most recently argued:
---
{prior_bear}
---

Address their strongest specific points head-on. Do not repeat your prior
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

Write the bull case in 3 short sections:
## Thesis
One paragraph. State the long thesis and the time horizon implied.

## Key Evidence
Bulleted, each tied to a concrete signal (indicator, financial metric, news catalyst).

## Rebuttal of the Bear (if any)
Address their strongest claim. If no bear argument is on record yet, skip this section.
"""

    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": BULL_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"bull_argument": resp.content}
