import requests
from datetime import datetime, timezone
from typing import List, Dict
import os

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def fetch_alpha_vantage_news(ticker: str, limit: int = 50) -> List[Dict]:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    feed = data.get("feed", [])
    results = []

    for item in feed:
        published = item.get("time_published")
        published_dt = None
        if published:
            try:
                published_dt = datetime.strptime(
                    published, "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                published_dt = None

        results.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", "unknown"),
            "url": item.get("url", ""),
            "published_at": published_dt,
            "overall_sentiment_score": float(item.get("overall_sentiment_score", 0.0)),
            "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
        })

    return results


def recency_weight(hours_ago: float) -> float:
    if hours_ago <= 24:
        return 1.0
    if hours_ago <= 72:
        return 0.7
    if hours_ago <= 120:
        return 0.4
    return 0.2


def label_sentiment(score: float) -> str:
    if score > 0.2:
        return "bullish"
    if score < -0.2:
        return "bearish"
    return "neutral"


def prepare_news_sentiment_metrics(news_items: List[Dict]) -> Dict:
    if not news_items:
        return {
            "article_count": 0,
            "avg_sentiment": 0.0,
            "weighted_sentiment": 0.0,
            "sentiment_label": "neutral",
            "confidence": "low",
            "top_articles": [],
        }

    now = datetime.now(timezone.utc)
    enriched = []

    raw_scores = []
    weighted_sum = 0.0
    weight_total = 0.0

    for item in news_items:
        published_at = item.get("published_at")
        if published_at is None:
            hours_ago = None
            time_weight = 0.2
        else:
            hours_ago = (now - published_at).total_seconds() / 3600
            time_weight = recency_weight(hours_ago)

        sentiment_score = float(item.get("overall_sentiment_score", 0.0))
        impact_score = time_weight * (1 + abs(sentiment_score))

        enriched_item = {
            **item,
            "hours_ago": round(hours_ago, 1) if hours_ago is not None else None,
            "recency_weight": time_weight,
            "impact_score": round(impact_score, 3),
        }
        enriched.append(enriched_item)

        raw_scores.append(sentiment_score)
        weighted_sum += sentiment_score * time_weight
        weight_total += time_weight

    avg_sentiment = sum(raw_scores) / len(raw_scores)
    weighted_sentiment = weighted_sum / weight_total if weight_total else 0.0
    spread = max(raw_scores) - min(raw_scores) if len(raw_scores) > 1 else 0.0

    if spread < 0.25:
        confidence = "high"
    elif spread < 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    enriched.sort(key=lambda x: x["impact_score"], reverse=True)

    return {
        "article_count": len(enriched),
        "avg_sentiment": round(avg_sentiment, 3),
        "weighted_sentiment": round(weighted_sentiment, 3),
        "sentiment_label": label_sentiment(weighted_sentiment),
        "confidence": confidence,
        "top_articles": enriched[:8],
    }