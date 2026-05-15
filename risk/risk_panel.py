"""Runs the three risk specialists. Pulls quant inputs once and shares them."""
from dataflows.volatility import market_regime, realized_volatility, assess_volatility
from dataflows.risk_metrics import gather_full_risk_inputs
from risk.tail_risk import run_tail_risk
from risk.macro_regime import run_macro_regime
from risk.liquidity import run_liquidity


def run_risk_panel(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    bundle = gather_full_risk_inputs(ticker, end_date=state.get("as_of_date"))

    market = market_regime(bundle["spy_df"])
    ticker_vol = realized_volatility(bundle["ticker_df"], window=20)

    risk_inputs = {
        "market": market,
        "tail": bundle["tail"],
        "liquidity": bundle["liquidity"],
        "ticker_vol_annualized": round(ticker_vol, 4),
        "ticker_vol_label": assess_volatility(ticker_vol),
    }

    panel_state = {**state, "risk_inputs": risk_inputs}

    out = {"risk_inputs": risk_inputs}
    out.update(run_tail_risk(panel_state, config))
    out.update(run_macro_regime(panel_state, config))
    out.update(run_liquidity(panel_state, config))
    return out
