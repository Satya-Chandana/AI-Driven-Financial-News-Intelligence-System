import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_price_history(ticker: str, lookback_days: int = 180) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No price data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    return df


def compute_indicators(df: pd.DataFrame, ticker: str) -> dict:
    close = df["close"]
    volume = df["volume"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower)

    # VWAP
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()

    # Volume z-score
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_zscore = (volume - vol_mean) / vol_std

    latest = -1
    return {
        "ticker": ticker,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "current_price": round(float(close.iloc[latest]), 2),
        "rsi_14": round(float(rsi.iloc[latest]), 2),
        "macd": round(float(macd.iloc[latest]), 4),
        "macd_signal": round(float(signal.iloc[latest]), 4),
        "macd_histogram": round(float((macd - signal).iloc[latest]), 4),
        "bb_pct": round(float(bb_pct.iloc[latest]), 3),
        "bb_upper": round(float(bb_upper.iloc[latest]), 2),
        "bb_lower": round(float(bb_lower.iloc[latest]), 2),
        "vwap_20d": round(float(vwap.iloc[latest]), 2),
        "vol_zscore": round(float(vol_zscore.iloc[latest]), 2),
        "price_vs_vwap": round(float(close.iloc[latest] - vwap.iloc[latest]), 2),
        "52w_high": round(float(close.rolling(252).max().iloc[latest]), 2),
        "52w_low": round(float(close.rolling(252).min().iloc[latest]), 2),
    }