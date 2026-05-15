"""Final decision node: assembles a structured record and writes it to the memory DB."""
import re
from datetime import datetime
from memory.memory_db import save_decision, add_trading_days


def _parse_bias(synthesis_text: str) -> str:
    m = re.search(r"##\s*Final Bias\s*\n+\s*([a-zA-Z]+)", synthesis_text or "")
    return m.group(1).strip().lower() if m else "unknown"


def _parse_confidence(synthesis_text: str) -> str:
    m = re.search(r"##\s*Confidence\s*\n+\s*([a-zA-Z]+)", synthesis_text or "")
    return m.group(1).strip().lower() if m else "unknown"


def run_final_decision(state: dict, config: dict = None) -> dict:
    ticker = state["ticker"]
    today = state.get("as_of_date") or datetime.utcnow().date().isoformat()
    eval_after = add_trading_days(today, n_days=5)

    metrics = state.get("technical_metrics") or {}
    entry_price = metrics.get("current_price")

    final_decision = state.get("final_decision", "HOLD")
    conviction = state.get("trader_conviction", 3)
    bias = _parse_bias(state.get("synthesis_output", ""))
    confidence = _parse_confidence(state.get("synthesis_output", ""))

    record = {
        "ticker": ticker,
        "decision_date": today,
        "evaluate_after": eval_after,
        "entry_price": entry_price,
        "final_decision": final_decision,
        "conviction": conviction,
        "bias": bias,
        "confidence": confidence,
        "synthesis": state.get("synthesis_output", ""),
        "bull_argument": state.get("bull_argument", ""),
        "bear_argument": state.get("bear_argument", ""),
        "arbiter_verdict": state.get("arbiter_verdict", ""),
        "risk_notes": state.get("risk_output", ""),
    }

    decision_id = save_decision(record)

    summary = (
        f"FINAL DECISION for {ticker}: {final_decision} | conviction {conviction}/5\n"
        f"Bias: {bias} | Confidence: {confidence}\n"
        f"Entry price: {entry_price} | Re-evaluate on/after: {eval_after}\n"
        f"Persisted as decision id={decision_id}."
    )

    return {
        "decision_id": decision_id,
        "decision_summary": summary,
        "evaluate_after": eval_after,
    }
