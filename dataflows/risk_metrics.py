"""Quantitative inputs for the risk specialist panel.

Tail-risk metrics:    worst-day, historical VaR/CVaR, vol expansion, max drawdown.
Liquidity metrics:    dollar volume, relative liquidity, intraday range, beta to SPY.
Macro regime is computed in dataflows/volatility.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataflows.market_data import get_price_history


def tail_risk_metrics(df: pd.DataFrame) -> dict:
    close = df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    if log_ret.empty:
        return {}

    short_window = log_ret.iloc[-20:] if len(log_ret) >= 20 else log_ret
    full = log_ret

    worst_day = float(full.min())
    best_day = float(full.max())
    var_5 = float(np.percentile(full, 5))
    cvar_5 = float(full[full <= var_5].mean()) if (full <= var_5).any() else var_5
    skew = float(((full - full.mean()) ** 3).mean() / (full.std(ddof=0) ** 3 + 1e-12))

    vol_short = float(short_window.std(ddof=0) * np.sqrt(252))
    vol_full = float(full.std(ddof=0) * np.sqrt(252))
    vol_expansion = vol_short / vol_full if vol_full > 0 else 1.0

    rolling_high_60 = close.rolling(60).max().iloc[-1]
    last = float(close.iloc[-1])
    drawdown_60 = (last - float(rolling_high_60)) / float(rolling_high_60) if rolling_high_60 else 0.0

    return {
        "worst_day_return": round(worst_day, 4),
        "best_day_return": round(best_day, 4),
        "var_5": round(var_5, 4),
        "cvar_5": round(cvar_5, 4),
        "skew": round(skew, 3),
        "vol_short_annualized": round(vol_short, 4),
        "vol_full_annualized": round(vol_full, 4),
        "vol_expansion_ratio": round(vol_expansion, 3),
        "drawdown_60d": round(drawdown_60, 4),
    }


def liquidity_metrics(df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    dollar_volume = (close * volume)
    avg_dollar_vol_20 = float(dollar_volume.rolling(20).mean().iloc[-1]) if len(dollar_volume) >= 20 else float(dollar_volume.mean())
    avg_volume_252 = float(volume.rolling(252).mean().iloc[-1]) if len(volume) >= 252 else float(volume.mean())
    avg_volume_20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
    rel_liquidity = avg_volume_20 / avg_volume_252 if avg_volume_252 > 0 else 1.0

    intraday_range = ((high - low) / close).rolling(20).mean().iloc[-1]
    intraday_range = float(intraday_range) if not pd.isna(intraday_range) else 0.0

    # Beta to SPY using the overlapping last 60 days of returns.
    ticker_ret = close.pct_change().dropna()
    spy_ret = spy_df["close"].pct_change().dropna()
    common = ticker_ret.index.intersection(spy_ret.index)
    if len(common) >= 30:
        x = spy_ret.loc[common].iloc[-60:]
        y = ticker_ret.loc[common].iloc[-60:]
        var_x = float(x.var(ddof=0))
        beta = float(((x - x.mean()) * (y - y.mean())).mean() / var_x) if var_x > 0 else 1.0
    else:
        beta = 1.0

    return {
        "avg_dollar_volume_20d": round(avg_dollar_vol_20, 0),
        "rel_liquidity_20_vs_252": round(rel_liquidity, 3),
        "avg_intraday_range_20d": round(intraday_range, 4),
        "beta_60d": round(beta, 3),
    }


def gather_full_risk_inputs(ticker: str, end_date=None) -> dict:
    """Return everything the risk specialist panel needs in one shot."""
    ticker_df = get_price_history(ticker, lookback_days=400, end_date=end_date)
    spy_df = get_price_history("SPY", lookback_days=400, end_date=end_date)
    return {
        "ticker_df": ticker_df,
        "spy_df": spy_df,
        "tail": tail_risk_metrics(ticker_df),
        "liquidity": liquidity_metrics(ticker_df, spy_df),
    }
