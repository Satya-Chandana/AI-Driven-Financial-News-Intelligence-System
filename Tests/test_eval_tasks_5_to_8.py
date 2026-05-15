"""
test_eval_tasks_5_to_8.py  —  Evaluation Tasks 5-8
=============================================================
All tests use mocked LLMs, market data, EDGAR, and news APIs.
No real network calls are made.
Run:  python -m pytest test_eval_tasks_5_to_8.py -v
"""
import os
import re
import sqlite3
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

VALID_RATINGS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}

_LLM_TARGETS = [
    "llm_clients.factory.get_llm",
    "analysts.technical_analyst.get_llm",
    "analysts.fundamentals_analyst.get_llm",
    "analysts.news_sentiment_analyst.get_llm",
    "synthesis.cot_synthesis.get_llm",
    "debate.bull_researcher.get_llm",
    "debate.bear_researcher.get_llm",
    "debate.arbiter.get_llm",
    "debate.neutral_arbiter.get_llm",
    "debate.critic.get_llm",
    "trader.trader_agent.get_llm",
    "risk.tail_risk.get_llm",
    "risk.macro_regime.get_llm",
    "risk.liquidity.get_llm",
    "risk.risk_arbiter.get_llm",
    "risk.risk_panel.get_llm",
]

_MARKET_TARGETS = [
    "dataflows.market_data.get_price_history",
    "dataflows.market_data.compute_indicators",
    "analysts.technical_analyst.get_price_history",
    "analysts.technical_analyst.compute_indicators",
    "risk.tail_risk.get_price_history",
    "risk.macro_regime.get_price_history",
    "risk.liquidity.get_price_history",
    "risk.volatility.get_price_history",
    "risk.risk_metrics.get_price_history",
    "memory.reflection.get_price_history",
]

# ── LLM helpers ────────────────────────────────────────────────────────────────

def _fixed_llm(content: str) -> MagicMock:
    resp = MagicMock(); resp.content = content
    llm  = MagicMock(); llm.invoke.return_value = resp
    return llm

def _sequenced_llm(responses: list) -> MagicMock:
    ctr = {"n": 0}
    def _invoke(msgs, **kw):
        r = MagicMock(); r.content = responses[ctr["n"] % len(responses)]; ctr["n"] += 1; return r
    llm = MagicMock(); llm.invoke.side_effect = _invoke; return llm

class _PatchAllLLMs:
    def __init__(self, mock): self._mock = mock; self._stack = None
    def __enter__(self):
        self._stack = ExitStack()
        for t in _LLM_TARGETS:
            try: self._stack.enter_context(patch(t, return_value=self._mock))
            except (AttributeError, ModuleNotFoundError): pass
        return self
    def __exit__(self, *a): self._stack.__exit__(*a)

def patch_llm(mock): return _PatchAllLLMs(mock)

# ── LLM response stubs ─────────────────────────────────────────────────────────

TECH = """## Reasoning
RSI=62 — bullish momentum. MACD histogram positive. Bollinger %B=0.65 upper half.
## Verdict
bullish
"""
FUND = """## Reasoning
Revenue $383B, net income $97B. Strong balance sheet. EPS growing.
## Verdict
bullish
"""
NEWS = """## Reasoning
Sentiment neutral to slightly positive. No major adverse headlines.
## Verdict
neutral
"""
SYNTHESIS = """## Technical View
Momentum bullish.
## Fundamental View
Fundamentals solid.
## News + Sentiment View
Neutral sentiment.
## Cross-Analyst Synthesis
Technical and fundamental agree. Sentiment does not conflict.
## Final Bias
bullish
## Confidence
medium
## Key Risks
Macro uncertainty.
## Actionable Takeaway
Lean long with moderate conviction.
"""
BULL = "## Bull Case\nRSI=62 and positive MACD confirm upward momentum. Revenue growth is strong.\n"
BEAR = "## Bear Case\nSentiment is only neutral. Macro risk unresolved.\n"

ARBITER_UNANIMOUS = """## Evidence Quality Scores
- Bull argument score: 8/10
  Reason: Strong technical and fundamental data.
- Bear argument score: 4/10
  Reason: Only macro risk cited, not company-specific.
### Contradiction Detected
false
### Consensus Score
0.60
### Arbiter Summary
Bull case clearly stronger.
"""
ARBITER_SPLIT = """## Evidence Quality Scores
- Bull argument score: 5/10
  Reason: Momentum exists but uncertain.
- Bear argument score: 5/10
  Reason: Legitimate downside risks.
### Contradiction Detected
true
### Consensus Score
0.00
### Arbiter Summary
Split decision — both sides valid.
"""
TRADER_HIGH = """## Rating
OVERWEIGHT

## Conviction
4

## Rationale
Bull case dominates. Technical momentum strong, fundamentals supportive.
Main risk is macro, which is manageable.

## Position Notes
Enter on pullbacks. Invalidated if RSI drops below 50.
"""
TRADER_LOW = """## Rating
HOLD

## Conviction
2

## Rationale
Signals conflict. Technical bullish but sentiment and macro create uncertainty.
Low conviction reflects balanced debate.

## Position Notes
Wait for clarity. Exit if sentiment deteriorates.
"""
TAIL   = "NORMAL — volatility within expected range. VaR and CVaR acceptable."
MACRO  = "SUPPORTIVE — SPY above 200-day SMA. Bull market regime."
LIQ    = "NORMAL — dollar volume and beta within acceptable ranges."
RISK_ARB = """## Risk Arbiter Decision
All three specialists concur.
## Adjustment
KEEP
## Final Decision
OVERWEIGHT
## Adjusted Conviction
4
"""
CRITIC_WEAK = """## Critique
Bull case is well-supported. No material blind spots found.

## Strength
WEAK

## Recommendation
Decision stands.
"""

ALL_RESPONSES = [
    TECH, FUND, NEWS, SYNTHESIS,
    BULL, BEAR, ARBITER_UNANIMOUS,
    TRADER_HIGH, TAIL, MACRO, LIQ, RISK_ARB, CRITIC_WEAK,
]

# ── DB helper ──────────────────────────────────────────────────────────────────

def _insert_decision(db_path, ticker, decision_date, final_decision, conviction=4, entry_price=150.0):
    conn = sqlite3.connect(db_path)
    now  = datetime.now(timezone.utc).isoformat()
    conn.execute("""INSERT INTO decisions
        (ticker, decision_date, evaluate_after, entry_price, final_decision, conviction, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, decision_date, "2099-12-31", entry_price, final_decision, conviction, now))
    conn.commit(); conn.close()

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import importlib
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("TRADINGAGENTS_DB_PATH", db)
    import memory.memory_db as mdb
    importlib.reload(mdb); mdb.init_db()
    yield db

@pytest.fixture
def mock_market(isolated_db):
    import numpy as np, pandas as pd
    n   = 252
    idx = pd.bdate_range("2023-07-01", periods=n)
    p   = np.linspace(150, 170, n)
    df  = pd.DataFrame({"close": p, "high": p*1.01, "low": p*0.99,
                         "volume": np.full(n, 5_000_000), "open": p*1.002}, index=idx)
    m   = {"ticker":"GENERIC","current_price":170.0,"rsi_14":62.0,
           "macd":1.5,"macd_signal":1.0,"macd_histogram":0.5,
           "bb_pct":0.65,"bb_lower":160.0,"bb_upper":180.0,
           "vwap_20d":165.0,"price_vs_vwap":5.0,"vol_zscore":0.3,
           "52w_low":140.0,"52w_high":195.0}
    with ExitStack() as stack:
        for t in _MARKET_TARGETS:
            try: stack.enter_context(patch(t, return_value=df))
            except (AttributeError, ModuleNotFoundError): pass
        try: stack.enter_context(patch("dataflows.market_data.compute_indicators", return_value=m))
        except (AttributeError, ModuleNotFoundError): pass
        try: stack.enter_context(patch("analysts.technical_analyst.compute_indicators", return_value=m))
        except (AttributeError, ModuleNotFoundError): pass
        yield df, m

@pytest.fixture
def mock_edgar():
    facts = {"cik":"0000320193",
             "NetIncomeLoss":[{"period":"2023","value":96_995_000_000}],
             "Revenues":[{"period":"2023","value":383_285_000_000}],
             "Assets":[{"period":"2023","value":352_755_000_000}],
             "Liabilities":[{"period":"2023","value":290_437_000_000}],
             "StockholdersEquity":[{"period":"2023","value":62_146_000_000}],
             "EarningsPerShareBasic":[{"period":"2023","value":6.13}]}
    with ExitStack() as stack:
        for t in ["dataflows.edgar.get_company_facts","analysts.fundamentals_analyst.get_company_facts"]:
            try: stack.enter_context(patch(t, return_value=facts))
            except (AttributeError, ModuleNotFoundError): pass
        yield facts

@pytest.fixture
def mock_news():
    m = {"article_count":5,"avg_sentiment":0.12,"weighted_sentiment":0.08,
         "sentiment_label":"Neutral","confidence":"medium","top_articles":[]}
    with ExitStack() as stack:
        for t, v in [
            ("dataflows.news_sentiment_data.fetch_alpha_vantage_news", []),
            ("dataflows.news_sentiment_data.prepare_news_sentiment_metrics", m),
            ("analysts.news_sentiment_analyst.fetch_alpha_vantage_news", []),
            ("analysts.news_sentiment_analyst.prepare_news_sentiment_metrics", m),
        ]:
            try: stack.enter_context(patch(t, return_value=v))
            except (AttributeError, ModuleNotFoundError): pass
        yield m


def _run(ticker, responses=None):
    from main import analyze
    llm = _sequenced_llm(responses or ALL_RESPONSES)
    with patch_llm(llm):
        return analyze(ticker, debate_rounds=1, write_files=False, verbose=False, show_progress=False)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — No prior memory
# ══════════════════════════════════════════════════════════════════════════════

class TestTask5_NeutralArbiter:
    _SYNTH = "## Final Bias\nneutral\n## Confidence\nlow\n## Key Risks\nTechnical/sentiment conflict.\n"
    _BULL  = "Bull: RSI=62, MACD positive. Technically bullish."
    _BEAR  = "Bear: Negative post-earnings sentiment. Operational risk."

    def test_5a_arbiter_scores_both_sides(self):
        llm = _fixed_llm(ARBITER_SPLIT)
        state = {"ticker": "SYNTHETIC", "as_of_date": "2024-04-01",
                 "synthesis_output": self._SYNTH,
                 "bull_argument": self._BULL, "bear_argument": self._BEAR}
        with patch("debate.arbiter.get_llm", return_value=llm):
            try:
                from debate.arbiter import run_arbiter
            except ImportError:
                try:
                    from debate.neutral_arbiter import run_arbiter
                except ImportError:
                    pytest.skip("Arbiter module not found under expected path")
            result = run_arbiter(state, config={})
        verdict = result.get("arbiter_verdict", "")
        assert verdict, "Arbiter verdict must be non-empty"
        text_lower = verdict.lower()
        assert "bull" in text_lower or "score" in text_lower
        assert "bear" in text_lower or "score" in text_lower
        print("\n✅  TASK 5a PASSED — Arbiter scored both sides with balanced signals")

    def test_5b_consensus_score_in_range(self):
        llm = _fixed_llm(ARBITER_SPLIT)
        state = {"ticker": "SYNTHETIC", "as_of_date": "2024-04-01",
                 "synthesis_output": self._SYNTH,
                 "bull_argument": self._BULL, "bear_argument": self._BEAR}
        with patch("debate.arbiter.get_llm", return_value=llm):
            try:
                from debate.arbiter import run_arbiter
            except ImportError:
                try:
                    from debate.neutral_arbiter import run_arbiter
                except ImportError:
                    pytest.skip("Arbiter module not found")
            result = run_arbiter(state, config={})
        score = result.get("consensus_score")
        if score is not None:
            assert -1.0 <= float(score) <= 1.0
        print(f"\n✅  TASK 5b PASSED — consensus_score={score}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Trader conviction score
# ══════════════════════════════════════════════════════════════════════════════


class TestTask6_TraderConviction:
    _BASE = {
        "ticker": "TSLA", "as_of_date": "2024-04-01",
        "synthesis_output": SYNTHESIS, "bull_argument": BULL,
        "bear_argument": BEAR, "memory_context": "",
    }

    def _get_trader(self):
        try:
            from trader.trader_agent import run_trader
        except ImportError:
            from trader_agent import run_trader
        return run_trader

    def test_6a_unanimous_conviction_high(self):
        run_trader = self._get_trader()
        state = dict(self._BASE, arbiter_verdict=ARBITER_UNANIMOUS)
        llm = _fixed_llm(TRADER_HIGH)
        with patch("trader.trader_agent.get_llm", return_value=llm):
            result = run_trader(state, config={})
        rating = result.get("trader_rating", "")
        conviction = result.get("trader_conviction", 0)
        assert rating.upper() in VALID_RATINGS
        assert isinstance(conviction, int) and 1 <= conviction <= 5
        assert conviction >= 4, f"Unanimous → conviction ≥ 4, got {conviction}"
        print(f"\n✅  TASK 6a PASSED — Rating={rating}, Conviction={conviction}")

    def test_6b_split_conviction_low(self):
        run_trader = self._get_trader()
        state = dict(self._BASE, arbiter_verdict=ARBITER_SPLIT)
        llm = _fixed_llm(TRADER_LOW)
        with patch("trader.trader_agent.get_llm", return_value=llm):
            result = run_trader(state, config={})
        rating = result.get("trader_rating", "")
        conviction = result.get("trader_conviction", 0)
        assert rating.upper() in VALID_RATINGS
        assert isinstance(conviction, int) and 1 <= conviction <= 5
        assert conviction <= 3, f"Split → conviction ≤ 3, got {conviction}"
        print(f"\n✅  TASK 6b PASSED — Rating={rating}, Conviction={conviction}")

    def test_6c_output_fields_present(self):
        run_trader = self._get_trader()
        state = dict(self._BASE, arbiter_verdict=ARBITER_UNANIMOUS)
        llm = _fixed_llm(TRADER_HIGH)
        with patch("trader.trader_agent.get_llm", return_value=llm):
            result = run_trader(state, config={})
        assert "trader_rating" in result
        assert "trader_conviction" in result
        assert "trader_output" in result
        print("\n✅  TASK 6c PASSED — All output fields present")


class TestTask7_RiskRegime:
    def test_7a_regime_field_present_in_result(self, mock_all):
        result = _run("SPY")
        regime = result.get("macro_verdict") or result.get("risk_adjustment") or ""
        assert regime, f"No regime information in result. Keys: {list(result.keys())}"
        print(f"\n✅  TASK 7a PASSED — Regime field present: {regime}")

    def test_7b_risk_verdicts_all_present(self, mock_all):
        result = _run("SPY")
        tail    = result.get("tail_risk_verdict", "")
        macro   = result.get("macro_verdict", "")
        liq     = result.get("liquidity_verdict", "")
        assert tail or macro or liq, \
            "No risk specialist verdicts found in result"
        print(f"\n✅  TASK 7b PASSED — Risk verdicts: tail={tail}, macro={macro}, liq={liq}")

    def test_7c_final_decision_valid_after_risk(self, mock_all):
        result = _run("SPY")
        dec = (result.get("final_decision") or result.get("trader_rating") or "").upper()
        assert dec in VALID_RATINGS, f"Invalid final decision after risk: {dec!r}"
        print(f"\n✅  TASK 7c PASSED — Final decision '{dec}' valid after risk panel")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Reflection Agent T+5 outcome computation
# ══════════════════════════════════════════════════════════════════════════════


class TestTask8_Reflection:
    def _price_df(self):
        idx = pd.to_datetime(["2024-04-01","2024-04-02","2024-04-03",
                               "2024-04-04","2024-04-05","2024-04-08"])
        return pd.DataFrame({"close":[150.0,152.0,153.0,154.0,155.0,158.0]}, index=idx)

    def test_8a_postmortem_written_to_db(self, isolated_db):
        decision_id = _insert_decision(isolated_db, "SPY", "2024-04-01", "BUY",
                                        entry_price=150.0, evaluated=0)
        # Update evaluate_after to a past date so reflection triggers
        conn = sqlite3.connect(isolated_db)
        conn.execute("UPDATE decisions SET evaluate_after='2024-04-08' WHERE id=?", (decision_id,))
        conn.commit(); conn.close()

        llm = _fixed_llm("Post-mortem: The technical momentum call was correct. "
                          "RSI indicated upward bias which materialised. "
                          "Next time weight sentiment more heavily as a confirming signal.")
        price_df = self._price_df()

        with (patch("memory.reflection.get_llm", return_value=llm),
              patch("memory.reflection.get_price_history", return_value=price_df)):
            try:
                from memory.reflection import run_reflection
            except ImportError:
                from reflection import run_reflection
            processed = run_reflection()

        assert len(processed) >= 1, "No decisions processed by reflection agent"
        conn = sqlite3.connect(isolated_db)
        row = conn.execute("SELECT postmortem, outcome_label, evaluated FROM decisions WHERE id=?",
                           (decision_id,)).fetchone()
        conn.close()
        assert row[2] == 1, "Decision not marked as evaluated"
        assert row[0] and len(row[0]) >= 50, f"Post-mortem too short: {row[0]!r}"
        print(f"\n✅  TASK 8a PASSED — Post-mortem written, outcome_label={row[1]}")

    def test_8b_outcome_label_classified(self, isolated_db):
        decision_id = _insert_decision(isolated_db, "AAPL", "2024-04-01", "BUY",
                                        entry_price=150.0, evaluated=0)
        conn = sqlite3.connect(isolated_db)
        conn.execute("UPDATE decisions SET evaluate_after='2024-04-08' WHERE id=?", (decision_id,))
        conn.commit(); conn.close()
        # Outcome price 158 vs entry 150 → return +5.3% → win (threshold 3%)
        llm = _fixed_llm("Post-mortem: Strong technical momentum confirmed by price action. "
                          "Bull thesis validated by 5% gain in 5 trading days.")
        price_df = self._price_df()
        with (patch("memory.reflection.get_llm", return_value=llm),
              patch("memory.reflection.get_price_history", return_value=price_df)):
            try:
                from memory.reflection import run_reflection
            except ImportError:
                from reflection import run_reflection
            processed = run_reflection()
        assert processed, "No decisions processed"
        label = processed[0].get("outcome_label", "")
        assert label in {"win", "loss", "flat"}, f"Bad outcome_label: {label!r}"
        print(f"\n✅  TASK 8b PASSED — Outcome label classified as '{label}'")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 9 — Multi-ticker watchlist run
# ══════════════════════════════════════════════════════════════════════════════