# tradingagents_v2/agents/analysts/technical_analyst.py
from dataflows.market_data import get_price_history, compute_indicators
from llm_clients.factory import get_llm
from langchain_core.messages import HumanMessage

SYSTEM_PROMPT = """You are a technical analyst. You will be given quantitative 
indicators for a stock. Think step-by-step:
1. Interpret each indicator individually.
2. Look for confirmations or contradictions across indicators.
3. State your overall technical outlook (bullish / bearish / neutral) and explain why.
Be specific. Reference the actual numbers."""

def run_technical_analyst(state: dict, config: dict = None) -> dict:
    ticker  = state["ticker"]
    cfg     = config or {}
    llm     = get_llm(cfg)

    df      = get_price_history(ticker, cfg.get("lookback_days", 180))
    metrics = compute_indicators(df, ticker)

    prompt = f"""Ticker: {ticker}

Technical indicators (as of {metrics['ticker']}):
- Current price:     ${metrics['current_price']}
- RSI(14):           {metrics['rsi_14']}
- MACD:              {metrics['macd']}  |  Signal: {metrics['macd_signal']}  |  Histogram: {metrics['macd_histogram']}
- Bollinger %B:      {metrics['bb_pct']}  (0=lower band, 1=upper band)
  Bands: ${metrics['bb_lower']} — ${metrics['bb_upper']}
- VWAP(20d):         ${metrics['vwap_20d']}  (price vs VWAP: {metrics['price_vs_vwap']:+.2f})
- Volume z-score:    {metrics['vol_zscore']} (>2 = unusually high volume)
- 52-week range:     ${metrics['52w_low']} — ${metrics['52w_high']}

Provide your step-by-step technical analysis."""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ])

    return {"technical_report": response.content, "technical_metrics": metrics}