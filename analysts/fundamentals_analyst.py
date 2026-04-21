# tradingagents_v2/agents/analysts/fundamentals_analyst.py
from dataflows.edgar import get_company_facts
from llm_clients.factory import get_llm
from config import DEFAULT_CONFIG

SYSTEM_PROMPT = """You are a fundamental analyst. You will be given raw financial 
data from SEC EDGAR filings. Think step-by-step:
1. Assess revenue and earnings trends over the last 4 years.
2. Evaluate balance sheet health (assets vs liabilities).
3. Note any red flags or strengths.
4. State your overall fundamental outlook (bullish / bearish / neutral) with reasoning."""

def _format_series(entries: list) -> str:
    if not entries:
        return "  no data"
    lines = []
    for e in entries:
        val = e['value']
        if isinstance(val, (int, float)) and abs(val) > 1e6:
            val_str = f"${val/1e9:.2f}B" if abs(val) > 1e9 else f"${val/1e6:.1f}M"
        else:
            val_str = str(val)
        lines.append(f"  {e['period']}: {val_str}")
    return "\n".join(lines)

def run_fundamentals_analyst(state: dict, config: dict = None) -> dict:
    ticker  = state["ticker"]
    cfg     = {**DEFAULT_CONFIG, **(config or {})}
    llm     = get_llm(cfg)

    facts   = get_company_facts(ticker, cfg["edgar_facts"])

    sections = []
    label_map = {
        "NetIncomeLoss":        "Net income",
        "Revenues":             "Revenue",
        "Assets":               "Total assets",
        "Liabilities":          "Total liabilities",
        "StockholdersEquity":   "Shareholders equity",
        "EarningsPerShareBasic":"EPS (basic)",
    }
    for concept in cfg["edgar_facts"]:
        name    = concept.split(":")[1]
        label   = label_map.get(name, name)
        entries = facts.get(name, [])
        sections.append(f"{label}:\n{_format_series(entries)}")

    data_block = "\n\n".join(sections)

    prompt = f"""Ticker: {ticker}  |  CIK: {facts.get('cik', 'unknown')}

SEC EDGAR annual filing data (last 4 years):

{data_block}

Provide your step-by-step fundamental analysis."""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ])

    return {"fundamental_report": response.content, "edgar_facts": facts}