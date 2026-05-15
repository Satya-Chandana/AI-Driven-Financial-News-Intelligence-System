"""Macro/Regime specialist: argues from the perspective of broad market state.

Refuses to ignore: SPY trend, regime classification, broad drawdown, market vol.
"""
from llm_clients.factory import get_llm

MACRO_SYSTEM = """You are the Macro / Regime Analyst on a trading desk's risk committee.

Your job is NOT to give a balanced view. Your job is to defend the regime
context. You watch:
- Whether SPY is trading above/below its 50d and 200d SMAs
- The regime classification (risk_on / risk_off / mixed)
- 60-day market drawdown
- SPY realized volatility

You ask one question: "Does the broad market support taking this single-name bet?"
A great single-name thesis can still be wrong if the regime is hostile.

Return a strict structure:

## Macro Verdict
<one of: SUPPORTIVE / MIXED / HOSTILE>

## Key Macro Signals
3-5 bullets, each citing actual numbers from the inputs.

## Sizing Recommendation
One sentence. Suggest: keep, downsize one step, or force HOLD if regime is hostile.
"""


def run_macro_regime(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    market = (state.get("risk_inputs") or {}).get("market", {})
    trader_rating = state.get("trader_rating", "HOLD")
    conviction = state.get("trader_conviction", 3)

    if not market:
        return {"macro_output": "macro inputs unavailable",
                "macro_verdict": "MIXED"}

    prompt = f"""Ticker: {ticker}
Trader's current call: {trader_rating} (conviction {conviction}/5)

Macro / regime inputs:
- Regime:               {market['regime']}
- SPY last:             ${market['spy_last']}
- SPY SMA50:            ${market['spy_sma50']}
- SPY SMA200:           ${market['spy_sma200']}
- 60-day drawdown:      {market['drawdown_60d']:+.2%}
- SPY realized vol:     {market['spy_vol_annualized']:.2%}

Speak only to macro/regime. Other specialists cover tail risk and liquidity.
"""
    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": MACRO_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    text = resp.content
    verdict = "MIXED"
    for v in ("HOSTILE", "SUPPORTIVE", "MIXED"):
        if v in text.upper():
            verdict = v
            break
    return {"macro_output": text, "macro_verdict": verdict}
