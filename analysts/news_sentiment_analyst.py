from dataflows.news_sentiment_data import (
    fetch_alpha_vantage_news,
    prepare_news_sentiment_metrics,
)
from llm_clients.factory import get_llm

SYSTEM_PROMPT = """You are a news and sentiment analyst.

You are given recent Alpha Vantage news items and aggregated sentiment metrics.

Your tasks:
1. Identify the most important recent developments
2. Explain the overall sentiment trend
3. Give more weight to fresher news
4. Distinguish between weak/noisy sentiment and strong actionable sentiment
5. Provide an overall outlook: bullish / bearish / neutral

Be concise, specific, and mention the actual metrics.
"""


def run_news_sentiment_analyst(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    cfg = config or {}
    llm = get_llm(cfg)

    news_items = fetch_alpha_vantage_news(ticker, limit=50)
    metrics = prepare_news_sentiment_metrics(news_items)

    article_lines = []
    for item in metrics["top_articles"]:
        article_lines.append(
            f"- {item['title']}\n"
            f"  source={item['source']}, "
            f"sentiment={item['overall_sentiment_score']}, "
            f"hours_ago={item['hours_ago']}, "
            f"recency_weight={item['recency_weight']}, "
            f"impact_score={item['impact_score']}\n"
            f"  summary={item['summary']}"
        )

    article_block = "\n".join(article_lines) if article_lines else "No recent articles found."

    memory_block = state.get("memory_context", "")
    memory_section = f"\n\n{memory_block}\n" if memory_block else ""

    prompt = f"""Ticker: {ticker}
{memory_section}
Aggregated metrics:
- Article count: {metrics['article_count']}
- Average sentiment: {metrics['avg_sentiment']}
- Recency-weighted sentiment: {metrics['weighted_sentiment']}
- Sentiment label: {metrics['sentiment_label']}
- Confidence: {metrics['confidence']}

Top recent articles:
{article_block}

Provide a combined news + sentiment analysis report.
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    return {
        "news_sentiment_report": response.content,
        "news_sentiment_metrics": metrics,
    }