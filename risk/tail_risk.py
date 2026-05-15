"""Tail-risk specialist: argues from the perspective of catastrophic moves.

Refuses to ignore: gap risk, downside skew, vol expansion, max-drawdown precedent.
"""
from llm_clients.factory import get_llm

TAIL_SYSTEM = """You are the Tail-Risk Analyst on a trading desk's risk committee.

Your job is NOT to give a balanced view. Your job is to defend the worst-case
scenario and force the desk to size for it. You watch:
- Worst single-day move in the lookback window
- Historical Value-at-Risk (5%) and Conditional VaR (expected loss in the tail)
- Realized-vol expansion (recent vol vs full-window vol)
- Negative skew (heavy left tail)
- Recent drawdown

Be skeptical. If the trader is BUYing into a name with vol expansion and
negative skew, say so. If tail risk is muted, acknowledge it honestly — do not
manufacture danger.

Return a strict structure:

## Tail-Risk Verdict
<one of: ELEVATED / NORMAL / MUTED>

## Key Tail Signals
3-5 bullets, each citing the actual number from the inputs.

## Sizing Recommendation
One sentence. Suggest: keep, downsize one step, or force HOLD.
"""


def run_tail_risk(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    tail = (state.get("risk_inputs") or {}).get("tail", {})
    trader_rating = state.get("trader_rating", "HOLD")
    conviction = state.get("trader_conviction", 3)

    if not tail:
        return {"tail_risk_output": "tail-risk inputs unavailable",
                "tail_risk_verdict": "NORMAL"}

    prompt = f"""Ticker: {ticker}
Trader's current call: {trader_rating} (conviction {conviction}/5)

Tail-risk inputs:
- Worst single-day return: {tail['worst_day_return']:+.2%}
- Best single-day return:  {tail['best_day_return']:+.2%}
- 5% historical VaR:        {tail['var_5']:+.2%}
- 5% historical CVaR:       {tail['cvar_5']:+.2%}
- Skew of daily returns:    {tail['skew']}
- Vol (last 20d, ann.):     {tail['vol_short_annualized']:.2%}
- Vol (full, ann.):         {tail['vol_full_annualized']:.2%}
- Vol expansion ratio:      {tail['vol_expansion_ratio']}
- 60-day drawdown:          {tail['drawdown_60d']:+.2%}

Speak only to tail risk. Other specialists will cover macro and liquidity.
"""
    llm = get_llm(config or {})
    resp = llm.invoke([
        {"role": "system", "content": TAIL_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    text = resp.content
    verdict = "NORMAL"
    for v in ("ELEVATED", "MUTED", "NORMAL"):
        if v in text.upper():
            verdict = v
            break
    return {"tail_risk_output": text, "tail_risk_verdict": verdict}
