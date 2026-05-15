"""Liquidity / Microstructure specialist: argues from the perspective of execution.

Refuses to ignore: dollar volume, relative liquidity vs history, beta, intraday range.
"""
from llm_clients.factory import get_llm

LIQ_SYSTEM = """You are the Liquidity / Microstructure Analyst on the risk committee.

Your job is NOT to opine on whether the trade is right. Your job is to ensure
the desk can ENTER and EXIT cleanly. You watch:
- Average daily dollar volume (proxy for absolute liquidity)
- Recent volume vs long-term volume (liquidity regime)
- Average intraday range (proxy for slippage cost)
- Beta to SPY (correlated risk; high beta in a hostile macro is dangerous)

If liquidity is thin or beta is unusually high in a hostile regime, push back.
If everything is normal, say so plainly.

Return a strict structure:

## Liquidity Verdict
<one of: STRESSED / NORMAL / DEEP>

## Key Liquidity Signals
3-5 bullets, each citing actual numbers.

## Sizing Recommendation
One sentence. Suggest: keep, downsize one step, or force HOLD if execution is unsafe.
"""


def run_liquidity(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    liq = (state.get("risk_inputs") or {}).get("liquidity", {})
    trader_rating = state.get("trader_rating", "HOLD")
    conviction = state.get("trader_conviction", 3)

    if not liq:
        return {"liquidity_output": "liquidity inputs unavailable",
                "liquidity_verdict": "NORMAL"}

    prompt = f"""Ticker: {ticker}
Trader's current call: {trader_rating} (conviction {conviction}/5)

Liquidity / microstructure inputs:
- Avg dollar volume (20d): ${liq['avg_dollar_volume_20d']:,.0f}
- Liquidity (20d / 252d):  {liq['rel_liquidity_20_vs_252']}
- Avg intraday range (20d): {liq['avg_intraday_range_20d']:.2%}
- Beta to SPY (60d):        {liq['beta_60d']}

Speak only to liquidity / microstructure. Tail and macro are separate analysts.
"""
    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": LIQ_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    text = resp.content
    verdict = "NORMAL"
    for v in ("STRESSED", "DEEP", "NORMAL"):
        if v in text.upper():
            verdict = v
            break
    return {"liquidity_output": text, "liquidity_verdict": verdict}
