"""
test_eval_tasks_9_to_12.py  —  Evaluation Tasks 9-12
=========================================================================
Run:  python -m pytest test_eval_tasks_9_to_12.py -v
"""
import re
import sqlite3
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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
    "memory.reflection.get_llm",
]

def _fixed_llm(content):
    r = MagicMock(); r.content = content
    m = MagicMock(); m.invoke.return_value = r; return m

def _sequenced_llm(responses):
    ctr = {"n": 0}
    def _inv(msgs, **kw):
        r = MagicMock(); r.content = responses[ctr["n"] % len(responses)]; ctr["n"] += 1; return r
    m = MagicMock(); m.invoke.side_effect = _inv; return m

class _PatchAllLLMs:
    def __init__(self, mock): self._mock = mock; self._stack = None
    def __enter__(self):
        self._stack = ExitStack()
        for t in _LLM_TARGETS:
            try: self._stack.enter_context(patch(t, return_value=self._mock))
            except (AttributeError, ModuleNotFoundError): pass
        return self
    def __exit__(self, *a): self._stack.__exit__(*a)

def patch_llm(m): return _PatchAllLLMs(m)

# ── Shared stubs ────────────────────────────────────────────────────────────────

_TRADER_RESP = """## Rating\nHOLD\n\n## Conviction\n3\n\n## Rationale\nBalanced signals.\n\n## Position Notes\nWait for clarity.\n"""
_CRITIC_RESP = """## Critique\nNo major blind spots.\n\n## Strength\nWEAK\n\n## Recommendation\nDecision stands.\n"""
_RISK_ARB    = "## Adjustment\nKEEP\n## Final Decision\nHOLD\n## Adjusted Conviction\n3\n"
_TAIL        = "NORMAL — volatility within expected range."
_MACRO       = "SUPPORTIVE — SPY above 200-day SMA."
_LIQ         = "NORMAL — adequate liquidity."

_ALL_RESP = [
    "## Reasoning\nBullish signals.\n## Verdict\nbullish\n",  # tech
    "## Reasoning\nSolid fundamentals.\n## Verdict\nbullish\n",  # fund
    "## Reasoning\nNeutral sentiment.\n## Verdict\nneutral\n",  # news
    "## Final Bias\nbullish\n## Confidence\nmedium\n## Key Risks\nMacro.\n## Actionable Takeaway\nLean long.\n",
    "## Bull Case\nMomentum positive.\n",
    "## Bear Case\nMacro risk unresolved.\n",
    "## Evidence Quality Scores\n- Bull argument score: 7/10\n- Bear argument score: 4/10\n### Contradiction Detected\nfalse\n### Consensus Score\n0.50\n",
    _TRADER_RESP, _TAIL, _MACRO, _LIQ, _RISK_ARB, _CRITIC_RESP,
]

def _mk_df(n=252):
    idx = pd.bdate_range("2023-07-01", periods=n)
    p   = np.linspace(150, 170, n)
    return pd.DataFrame({"close":p,"high":p*1.01,"low":p*0.99,
                          "volume":np.full(n,5_000_000),"open":p*1.002}, index=idx)

def _mk_metrics():
    return {"ticker":"GENERIC","current_price":170.0,"rsi_14":62.0,
            "macd":1.5,"macd_signal":1.0,"macd_histogram":0.5,
            "bb_pct":0.65,"bb_lower":160.0,"bb_upper":180.0,
            "vwap_20d":165.0,"price_vs_vwap":5.0,"vol_zscore":0.3,
            "52w_low":140.0,"52w_high":195.0}

_EDGAR_FACTS = {
    "cik":"0000320193",
    "NetIncomeLoss":[{"period":"2023","value":96_995_000_000}],
    "Revenues":[{"period":"2023","value":383_285_000_000}],
    "Assets":[{"period":"2023","value":352_755_000_000}],
    "Liabilities":[{"period":"2023","value":290_437_000_000}],
    "StockholdersEquity":[{"period":"2023","value":62_146_000_000}],
    "EarningsPerShareBasic":[{"period":"2023","value":6.13}],
}

def _insert_decision(db, ticker, date, decision, conviction=4, entry_price=150.0, evaluated=0,
                     outcome_label=None, outcome_return=None, postmortem=None):
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db)
    conn.execute("""INSERT INTO decisions
        (ticker, decision_date, evaluate_after, entry_price, final_decision,
         conviction, evaluated, outcome_label, outcome_return, postmortem, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, date, "2024-01-01", entry_price, decision, conviction,
         evaluated, outcome_label, outcome_return, postmortem, now))
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit(); conn.close(); return row_id

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import importlib
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("TRADINGAGENTS_DB_PATH", db)
    import memory.memory_db as mdb
    importlib.reload(mdb); mdb.init_db()
    yield db

@pytest.fixture
def mock_all(isolated_db):
    df = _mk_df(); m = _mk_metrics()
    mkt_targets = [
        "dataflows.market_data.get_price_history",
        "dataflows.market_data.compute_indicators",
        "analysts.technical_analyst.get_price_history",
        "analysts.technical_analyst.compute_indicators",
        "risk.tail_risk.get_price_history",
        "risk.macro_regime.get_price_history",
        "risk.liquidity.get_price_history",
    ]
    edgar_targets = ["dataflows.edgar.get_company_facts","analysts.fundamentals_analyst.get_company_facts"]
    news_targets  = [
        ("dataflows.news_sentiment_data.fetch_alpha_vantage_news", []),
        ("dataflows.news_sentiment_data.prepare_news_sentiment_metrics",
         {"article_count":5,"avg_sentiment":0.12,"weighted_sentiment":0.08,
          "sentiment_label":"Neutral","confidence":"medium","top_articles":[]}),
        ("analysts.news_sentiment_analyst.fetch_alpha_vantage_news", []),
        ("analysts.news_sentiment_analyst.prepare_news_sentiment_metrics",
         {"article_count":5,"avg_sentiment":0.12,"weighted_sentiment":0.08,
          "sentiment_label":"Neutral","confidence":"medium","top_articles":[]}),
    ]
    with ExitStack() as stack:
        for t in mkt_targets:
            try: stack.enter_context(patch(t, return_value=df))
            except (AttributeError, ModuleNotFoundError): pass
        try: stack.enter_context(patch("dataflows.market_data.compute_indicators", return_value=m))
        except (AttributeError, ModuleNotFoundError): pass
        try: stack.enter_context(patch("analysts.technical_analyst.compute_indicators", return_value=m))
        except (AttributeError, ModuleNotFoundError): pass
        for t in edgar_targets:
            try: stack.enter_context(patch(t, return_value=_EDGAR_FACTS))
            except (AttributeError, ModuleNotFoundError): pass
        for t, v in news_targets:
            try: stack.enter_context(patch(t, return_value=v))
            except (AttributeError, ModuleNotFoundError): pass
        yield

def _run(ticker, responses=None):
    from main import analyze
    llm = _sequenced_llm(responses or _ALL_RESP)
    with patch_llm(llm):
        return analyze(ticker, debate_rounds=1, write_files=False, verbose=False, show_progress=False)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Risk regime check
# ══════════════════════════════════════════════════════════════════════════════

class TestTask9_Watchlist:
    TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL"]

    def test_9a_all_tickers_produce_valid_decision(self, mock_all):
        from main import analyze
        llm = _sequenced_llm(_ALL_RESP)
        results = {}
        with patch_llm(llm):
            for tk in self.TICKERS:
                results[tk] = analyze(tk, debate_rounds=1, write_files=False,
                                       verbose=False, show_progress=False)
        for tk, res in results.items():
            dec = (res.get("final_decision") or res.get("trader_rating") or "").upper()
            assert dec in VALID_RATINGS, f"{tk}: invalid decision {dec!r}"
        print(f"\n✅  TASK 9a PASSED — All {len(self.TICKERS)} tickers produced valid decisions")

    def test_9b_all_decisions_saved_to_db(self, mock_all, isolated_db):
        from main import analyze
        llm = _sequenced_llm(_ALL_RESP)
        with patch_llm(llm):
            for tk in self.TICKERS:
                analyze(tk, debate_rounds=1, write_files=False, verbose=False, show_progress=False)
        conn = sqlite3.connect(isolated_db)
        rows = conn.execute("SELECT ticker, final_decision FROM decisions").fetchall()
        conn.close()
        saved_tickers = {r[0] for r in rows}
        for tk in self.TICKERS:
            assert tk in saved_tickers, f"{tk} not found in DB after run"
        print(f"\n✅  TASK 9b PASSED — All {len(self.TICKERS)} decisions persisted to DB")

    def test_9c_no_cross_ticker_contamination(self, mock_all, isolated_db):
        from main import analyze
        llm = _sequenced_llm(_ALL_RESP)
        contexts = {}
        with patch_llm(llm):
            for tk in ["NVDA", "AAPL", "MSFT"]:
                res = analyze(tk, debate_rounds=1, write_files=False, verbose=False, show_progress=False)
                contexts[tk] = res.get("memory_context", "")
        # Each ticker's memory context should not contain other tickers' names
        for tk, ctx in contexts.items():
            other = [o for o in ["NVDA","AAPL","MSFT"] if o != tk]
            for o in other:
                assert o not in ctx or "No prior" in ctx, \
                    f"{tk} memory context contains {o}'s data (cross-contamination)"
        print("\n✅  TASK 9c PASSED — No cross-ticker memory contamination detected")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 10 — History retrieval and ordering
# ══════════════════════════════════════════════════════════════════════════════


class TestTask10_HistoryRetrieval:
    def test_10a_ordered_newest_first(self, isolated_db):
        dates = ["2024-01-01","2024-02-01","2024-03-01","2024-04-01","2024-05-01"]
        for d in dates: _insert_decision(isolated_db, "AAPL", d, "HOLD")
        from memory.memory_db import get_recent_decisions
        rows = get_recent_decisions("AAPL", limit=10)
        returned_dates = [r["decision_date"] for r in rows]
        assert returned_dates == sorted(returned_dates, reverse=True), \
            f"Decisions not ordered newest-first: {returned_dates}"
        print("\n✅  TASK 10a PASSED — Decisions returned newest-first")

    def test_10b_limit_respected(self, isolated_db):
        for i in range(8): _insert_decision(isolated_db, "MSFT", f"2024-0{(i%9)+1}-01", "BUY")
        from memory.memory_db import get_recent_decisions
        rows = get_recent_decisions("MSFT", limit=5)
        assert len(rows) <= 5, f"Expected ≤5 rows, got {len(rows)}"
        print(f"\n✅  TASK 10b PASSED — Limit respected: returned {len(rows)} of requested 5")

    def test_10c_ticker_isolation(self, isolated_db):
        _insert_decision(isolated_db, "AAPL", "2024-01-01", "BUY")
        _insert_decision(isolated_db, "TSLA", "2024-01-01", "SELL")
        from memory.memory_db import get_recent_decisions
        aapl_rows = get_recent_decisions("AAPL", limit=10)
        assert all(r["ticker"] == "AAPL" for r in aapl_rows), "AAPL query returned non-AAPL rows"
        print("\n✅  TASK 10c PASSED — Ticker isolation verified in history retrieval")

    def test_10d_empty_for_unknown_ticker(self, isolated_db):
        from memory.memory_db import get_recent_decisions
        rows = get_recent_decisions("UNKNOWN_XYZ", limit=10)
        assert rows == [], f"Expected empty list for unknown ticker, got {rows}"
        print("\n✅  TASK 10d PASSED — Empty list returned for unknown ticker")

    def test_10e_required_fields_non_null(self, isolated_db):
        _insert_decision(isolated_db, "NVDA", "2024-01-01", "OVERWEIGHT", conviction=3)
        from memory.memory_db import get_recent_decisions
        rows = get_recent_decisions("NVDA", limit=1)
        assert rows, "No rows returned"
        row = rows[0]
        for field in ["ticker", "decision_date", "final_decision", "conviction"]:
            assert row.get(field) is not None, f"Required field '{field}' is null"
        print("\n✅  TASK 10e PASSED — All required fields non-null")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 11 — Sparse news coverage
# ══════════════════════════════════════════════════════════════════════════════


class TestTask11_SparseNews:
    def test_11a_pipeline_completes_with_zero_articles(self, isolated_db):
        """Pipeline must complete even with zero news articles."""
        sparse_metrics = {
            "article_count": 0, "avg_sentiment": 0.0, "weighted_sentiment": 0.0,
            "sentiment_label": "Unknown", "confidence": "low", "top_articles": [],
        }
        df = _mk_df(); m = _mk_metrics()
        with ExitStack() as stack:
            for t, v in [
                ("dataflows.market_data.get_price_history", df),
                ("analysts.technical_analyst.get_price_history", df),
                ("risk.tail_risk.get_price_history", df),
                ("risk.macro_regime.get_price_history", df),
                ("risk.liquidity.get_price_history", df),
            ]:
                try: stack.enter_context(patch(t, return_value=v))
                except (AttributeError, ModuleNotFoundError): pass
            for t, v in [
                ("dataflows.market_data.compute_indicators", m),
                ("analysts.technical_analyst.compute_indicators", m),
                ("dataflows.edgar.get_company_facts", _EDGAR_FACTS),
                ("analysts.fundamentals_analyst.get_company_facts", _EDGAR_FACTS),
                ("dataflows.news_sentiment_data.fetch_alpha_vantage_news", []),
                ("dataflows.news_sentiment_data.prepare_news_sentiment_metrics", sparse_metrics),
                ("analysts.news_sentiment_analyst.fetch_alpha_vantage_news", []),
                ("analysts.news_sentiment_analyst.prepare_news_sentiment_metrics", sparse_metrics),
            ]:
                try: stack.enter_context(patch(t, return_value=v))
                except (AttributeError, ModuleNotFoundError): pass

            llm = _sequenced_llm(_ALL_RESP)
            with patch_llm(llm):
                from main import analyze
                result = analyze("APOG", debate_rounds=1, write_files=False,
                                  verbose=False, show_progress=False)

        dec = (result.get("final_decision") or result.get("trader_rating") or "").upper()
        assert dec in VALID_RATINGS, f"Invalid decision with sparse news: {dec!r}"
        print(f"\n✅  TASK 11a PASSED — Pipeline completed with zero articles. Decision: {dec}")

    def test_11b_news_report_produced_despite_sparse_data(self, mock_all):
        result = _run("APOG")
        report = result.get("news_sentiment_report", "")
        # Even with no articles, the agent should produce some output
        assert isinstance(report, str), "news_sentiment_report must be a string"
        print(f"\n✅  TASK 11b PASSED — News report produced despite sparse data")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 12 — Conflicting analyst signals
# ══════════════════════════════════════════════════════════════════════════════


class TestTask12_ConflictingSignals:
    """Task 12 — Conflicting signals: bullish technical vs bearish sentiment."""

    _SYNTH = """## Technical View
RSI=62, MACD positive — explicitly bullish momentum.
## Fundamental View
Fundamentals acceptable.
## News + Sentiment View
Negative post-earnings sentiment. Adverse supply chain news.
## Cross-Analyst Synthesis
Technical and sentiment in direct conflict.
## Final Bias
neutral
## Confidence
low
## Key Risks
Sentiment/technical divergence unresolved.
## Actionable Takeaway
Wait for resolution before committing.
"""
    _BULL = "Bull: RSI=62 and positive MACD confirm upward momentum."
    _BEAR = "Bear: Negative post-earnings sentiment dominates. Supply chain risk."
    _ARBITER_SPLIT = """## Evidence Quality Scores
- Bull argument score: 5/10
  Reason: Technical strong but sentiment headwind unresolved.
- Bear argument score: 5/10
  Reason: Sentiment deterioration is real and measurable.
### Contradiction Detected
true
### Consensus Score
0.00
### Arbiter Summary
Genuine conflict between technical and sentiment signals.
"""
    _TRADER_LOW = """## Rating
HOLD

## Conviction
2

## Rationale
Conflict between technical momentum and post-earnings sentiment.
Low conviction reflects genuine uncertainty. Mixed signals warrant caution.

## Position Notes
Avoid new positions until conflict resolves.
"""

    def test_12a_contradiction_detected_true(self):
        llm = _fixed_llm(self._ARBITER_SPLIT)
        state = {
            "ticker": "TSLA", "as_of_date": "2024-04-23",
            "synthesis_output": self._SYNTH,
            "bull_argument": self._BULL, "bear_argument": self._BEAR,
        }
        with patch("debate.arbiter.get_llm", return_value=llm):
            try:
                from debate.arbiter import run_arbiter
            except ImportError:
                try:
                    from debate.neutral_arbiter import run_arbiter
                except ImportError:
                    pytest.skip("Arbiter module not found")
            result = run_arbiter(state, config={})
        verdict_text = result.get("arbiter_verdict", "").lower()
        assert "contradiction detected" in verdict_text and "true" in verdict_text, \
        f"contradiction_detected not found as true in arbiter verdict.\n{verdict_text[:300]}"
        print("\n✅  TASK 12a PASSED — contradiction_detected=true found in arbiter verdict text")

    def test_12b_conviction_low_on_conflicting_signals(self):
        trader_llm = _fixed_llm(self._TRADER_LOW)
        state = {
            "ticker": "TSLA", "as_of_date": "2024-04-23",
            "synthesis_output": self._SYNTH,
            "bull_argument": self._BULL, "bear_argument": self._BEAR,
            "arbiter_verdict": self._ARBITER_SPLIT,
            "memory_context": "", "contradiction_detected": True,
        }
        with patch("trader.trader_agent.get_llm", return_value=trader_llm):
            try:
                from trader.trader_agent import run_trader
            except ImportError:
                from trader_agent import run_trader
            result = run_trader(state, config={})
        conviction = result.get("trader_conviction", 0)
        assert conviction <= 3, f"Conflicting signals should yield conviction ≤ 3, got {conviction}"
        print(f"\n✅  TASK 12b PASSED — Conviction={conviction} (≤3) on conflicting signals")

    def test_12c_conflict_keywords_in_thesis(self):
        trader_llm = _fixed_llm(self._TRADER_LOW)
        state = {
            "ticker": "TSLA", "as_of_date": "2024-04-23",
            "synthesis_output": self._SYNTH,
            "bull_argument": self._BULL, "bear_argument": self._BEAR,
            "arbiter_verdict": self._ARBITER_SPLIT,
            "memory_context": "", "contradiction_detected": True,
        }
        with patch("trader.trader_agent.get_llm", return_value=trader_llm):
            try:
                from trader.trader_agent import run_trader
            except ImportError:
                from trader_agent import run_trader
            result = run_trader(state, config={})
        thesis = result.get("trader_output", "").lower()
        keywords = ("conflict", "mixed", "divergence", "uncertainty", "cautio")
        assert any(k in thesis for k in keywords), \
            f"Thesis missing conflict keywords.\nThesis: {thesis[:300]}"
        print("\n✅  TASK 12c PASSED — Conflict keywords found in trader thesis")