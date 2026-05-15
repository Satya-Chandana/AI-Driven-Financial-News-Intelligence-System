"""Helpers for the risk manager: ticker volatility and broad market regime via SPY."""
import numpy as np
import pandas as pd
from dataflows.market_data import get_price_history


def realized_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """Annualized realized vol from the last `window` daily log returns."""
    close = df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        window = max(2, len(log_ret))
    recent = log_ret.iloc[-window:]
    return float(recent.std(ddof=0) * np.sqrt(252))


def market_regime(spy_df: pd.DataFrame) -> dict:
    """Classify market regime from SPY: trend (50/200d SMA) + recent drawdown."""
    close = spy_df["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    last = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else last
    s200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else last

    if last > s50 > s200:
        regime = "risk_on"
    elif last < s50 < s200:
        regime = "risk_off"
    else:
        regime = "mixed"

    rolling_high = close.rolling(60).max().iloc[-1]
    drawdown = (last - float(rolling_high)) / float(rolling_high) if rolling_high else 0.0
    spy_vol = realized_volatility(spy_df, window=20)

    return {
        "regime": regime,
        "spy_last": round(last, 2),
        "spy_sma50": round(s50, 2),
        "spy_sma200": round(s200, 2),
        "drawdown_60d": round(drawdown, 4),
        "spy_vol_annualized": round(spy_vol, 4),
    }


def assess_volatility(ticker_vol: float) -> str:
    """Categorize annualized realized vol into low/normal/elevated/extreme."""
    if ticker_vol < 0.20:
        return "low"
    if ticker_vol < 0.35:
        return "normal"
    if ticker_vol < 0.60:
        return "elevated"
    return "extreme"


def gather_risk_context(ticker: str, ticker_df: pd.DataFrame | None = None,
                        end_date=None) -> dict:
    if ticker_df is None:
        ticker_df = get_price_history(ticker, lookback_days=300, end_date=end_date)
    spy_df = get_price_history("SPY", lookback_days=300, end_date=end_date)

    ticker_vol = realized_volatility(ticker_df, window=20)
    return {
        "ticker_vol_annualized": round(ticker_vol, 4),
        "ticker_vol_label": assess_volatility(ticker_vol),
        "market": market_regime(spy_df),
    }
