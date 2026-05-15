"""End-to-end LangGraph wiring for the multi-agent trading pipeline.

Flow:
    memory
       -> [technical | fundamentals | news_sentiment]   (parallel)
       -> synthesis
       -> debate (bull/bear/arbiter, configurable rounds)
       -> trader
       -> risk_panel (tail / macro / liquidity)
       -> risk_arbiter
       -> critic (may internally trigger one re-debate -> trader -> risk loop)
       -> final_decision
       -> END
"""
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.constants import START

from analysts.technical_analyst import run_technical_analyst
from analysts.fundamentals_analyst import run_fundamentals_analyst
from analysts.news_sentiment_analyst import run_news_sentiment_analyst
from synthesis.cot_synthesis import run_cot_synthesis

from memory.memory_agent import run_memory_agent
from debate.debate_orchestrator import run_debate
from debate.critic import run_critic
from trader.trader_agent import run_trader
from risk.risk_panel import run_risk_panel
from risk.risk_arbiter import run_risk_arbiter
from risk.final_decision import run_final_decision

from graphs.progress import emit


class TradingState(TypedDict, total=False):
    ticker: str
    as_of_date: str          # YYYY-MM-DD; if absent, "today" is used

    # config (per-run)
    debate_rounds: int

    # memory
    memory_history: List[Dict[str, Any]]
    memory_context: str

    # analysts
    technical_report: str
    technical_metrics: dict
    fundamental_report: str
    edgar_facts: dict
    news_sentiment_report: str
    news_sentiment_metrics: dict

    # synthesis
    synthesis_output: str

    # debate
    bull_argument: str
    bear_argument: str
    arbiter_verdict: str
    debate_round: int
    debate_rounds_completed: int
    critic_challenge: str

    # trader
    trader_output: str
    trader_rating: str
    trader_conviction: int

    # risk panel
    risk_inputs: dict
    tail_risk_output: str
    tail_risk_verdict: str
    macro_output: str
    macro_verdict: str
    liquidity_output: str
    liquidity_verdict: str

    # risk arbiter
    risk_output: str
    risk_adjustment: str
    final_decision: str

    # critic
    critic_output: str
    critic_strength: str
    critic_triggered_rerun: bool

    # persistence
    decision_id: int
    decision_summary: str
    evaluate_after: str


def _traced(label, fn):
    """Wrap a node function so it emits start/done progress events."""
    def wrapped(state):
        emit("start", label)
        try:
            return fn(state)
        finally:
            emit("done", label)
    return wrapped


memory_node         = _traced("Memory layer",                lambda s: run_memory_agent(s))
technical_node      = _traced("Technical analyst",           lambda s: run_technical_analyst(s))
fundamentals_node   = _traced("Fundamentals analyst",        lambda s: run_fundamentals_analyst(s))
news_sentiment_node = _traced("News & sentiment analyst",    lambda s: run_news_sentiment_analyst(s))
synthesis_node      = _traced("Chain-of-thought synthesis",  lambda s: run_cot_synthesis(s))


def _debate_inner(state):
    cfg = {"debate_rounds": state.get("debate_rounds", 2)}
    return run_debate(state, cfg)
debate_node          = _traced("Debate (bull/bear/arbiter)", _debate_inner)

trader_node          = _traced("Trader",                     lambda s: run_trader(s))
risk_panel_node      = _traced("Risk panel (3 specialists)", lambda s: run_risk_panel(s))
risk_arbiter_node    = _traced("Risk arbiter",               lambda s: run_risk_arbiter(s))
critic_node          = _traced("Devil's advocate critic",    lambda s: run_critic(s))
final_decision_node  = _traced("Final decision & persist",   lambda s: run_final_decision(s))


def build_graph():
    g = StateGraph(TradingState)

    g.add_node("memory", memory_node)
    g.add_node("technical", technical_node)
    g.add_node("fundamentals", fundamentals_node)
    g.add_node("news_sentiment", news_sentiment_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("debate", debate_node)
    g.add_node("trader", trader_node)
    g.add_node("risk_panel", risk_panel_node)
    g.add_node("risk_arbiter", risk_arbiter_node)
    g.add_node("critic", critic_node)
    g.add_node("final_decision", final_decision_node)

    g.add_edge(START, "memory")

    g.add_edge("memory", "technical")
    g.add_edge("memory", "fundamentals")
    g.add_edge("memory", "news_sentiment")

    g.add_edge("technical", "synthesis")
    g.add_edge("fundamentals", "synthesis")
    g.add_edge("news_sentiment", "synthesis")

    g.add_edge("synthesis", "debate")
    g.add_edge("debate", "trader")
    g.add_edge("trader", "risk_panel")
    g.add_edge("risk_panel", "risk_arbiter")
    g.add_edge("risk_arbiter", "critic")
    g.add_edge("critic", "final_decision")
    g.add_edge("final_decision", END)

    return g.compile()
