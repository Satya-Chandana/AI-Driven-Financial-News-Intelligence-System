# tradingagents_v2/dataflows/edgar.py
import requests
import time

SEC_HEADERS = {"User-Agent": "TradingAgentsV2/1.0 research@example.com"}

def _get_cik(ticker: str) -> str:
    url  = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&forms=10-K".format(ticker)
    # Use the tickers.json endpoint instead — much faster
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker: {ticker}")

def get_company_facts(ticker: str, concepts: list[str]) -> dict:
    cik  = _get_cik(ticker)
    time.sleep(0.1)   # SEC rate limit courtesy pause
    url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    facts = resp.json().get("facts", {})

    result = {"ticker": ticker, "cik": cik}
    for concept in concepts:
        namespace, name = concept.split(":")
        entries = (facts.get(namespace, {})
                        .get(name, {})
                        .get("units", {})
                        .get("USD", []))
        if not entries:
            # try shares (EPS is in USD/shares but stored differently)
            entries = (facts.get(namespace, {})
                            .get(name, {})
                            .get("units", {})
                            .get("USD/shares", []))
        # Keep only annual filings (form 10-K), most recent first
        annual = sorted(
            [e for e in entries if e.get("form") == "10-K"],
            key=lambda x: x["end"], reverse=True
        )
        result[name] = [
            {"period": e["end"], "value": e["val"], "filed": e.get("filed")}
            for e in annual[:4]     # last 4 annual readings
        ]
    return result