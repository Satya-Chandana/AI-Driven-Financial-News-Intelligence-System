# tradingagents_v2/config.py
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_CONFIG = {
    "llm_provider": os.getenv("LLM_PROVIDER", "google"),
    "llm_model": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
    "max_tokens":   2048,
    "lookback_days": 180,
    "edgar_facts": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:Revenues",
        "us-gaap:Assets",
        "us-gaap:Liabilities",
        "us-gaap:StockholdersEquity",
        "us-gaap:EarningsPerShareBasic",
    ],
}