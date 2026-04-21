# from langgraph.graph import StateGraph, END
# from typing import TypedDict
# from agents.analysts.technical_analyst import run_technical_analyst
# from agents.analysts.fundamentals_analyst import run_fundamentals_analyst
# from agents.synthesis.cot_synthesis import run_cot_synthesis
# from langgraph.constants import START

# class TradingState(TypedDict, total=False):
#     ticker: str
#     technical_report: str
#     technical_metrics: dict
#     fundamental_report: str
#     edgar_facts: dict
#     synthesis_output: str

# def technical_node(state: TradingState) -> dict:
#     return run_technical_analyst(state)

# def fundamentals_node(state: TradingState) -> dict:
#     return run_fundamentals_analyst(state)

# def synthesis_node(state: TradingState) -> dict:
#     return run_cot_synthesis(state)

# def build_graph():
#     g = StateGraph(TradingState)

#     g.add_node("technical", technical_node)
#     g.add_node("fundamentals", fundamentals_node)
#     g.add_node("synthesis", synthesis_node)

#     g.add_edge(START, "technical")
#     g.add_edge(START, "fundamentals")
#     g.add_edge("technical", "synthesis")
#     g.add_edge("fundamentals", "synthesis")
#     g.add_edge("synthesis", END)

#     return g.compile()


from langgraph.graph import StateGraph, END
from langgraph.constants import START
from typing import TypedDict

from agents.analysts.technical_analyst import run_technical_analyst
from agents.analysts.fundamentals_analyst import run_fundamentals_analyst
from agents.analysts.news_sentiment_analyst import run_news_sentiment_analyst
from agents.synthesis.cot_synthesis import run_cot_synthesis


class TradingState(TypedDict, total=False):
    ticker: str

    technical_report: str
    technical_metrics: dict

    fundamental_report: str
    edgar_facts: dict

    news_sentiment_report: str
    news_sentiment_metrics: dict

    synthesis_output: str


def technical_node(state: TradingState) -> dict:
    return run_technical_analyst(state)


def fundamentals_node(state: TradingState) -> dict:
    return run_fundamentals_analyst(state)


def news_sentiment_node(state: TradingState) -> dict:
    return run_news_sentiment_analyst(state)


def synthesis_node(state: TradingState) -> dict:
    return run_cot_synthesis(state)


def build_graph():
    g = StateGraph(TradingState)

    g.add_node("technical", technical_node)
    g.add_node("fundamentals", fundamentals_node)
    g.add_node("news_sentiment", news_sentiment_node)
    g.add_node("synthesis", synthesis_node)

    # parallel analysts
    g.add_edge(START, "technical")
    g.add_edge(START, "fundamentals")
    g.add_edge(START, "news_sentiment")

    # all analyst outputs flow into synthesis
    g.add_edge("technical", "synthesis")
    g.add_edge("fundamentals", "synthesis")
    g.add_edge("news_sentiment", "synthesis")

    g.add_edge("synthesis", END)

    return g.compile()