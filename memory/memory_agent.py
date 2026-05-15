"""Memory Agent: pulls past decisions for the ticker and formats them as
context for downstream analysts and the trader."""
from typing import Dict
from memory.memory_db import get_recent_decisions


def _format_history_block(rows: list) -> str:
    if not rows:
        return "No prior decisions on record for this ticker."

    lines = []
    for r in rows:
        outcome = (
            f"outcome={r['outcome_label']} "
            f"(return {r['outcome_return']:+.2%})"
            if r.get("evaluated") else "outcome=pending"
        )
        lines.append(
            f"- {r['decision_date']} | {r['final_decision']} "
            f"(conviction {r.get('conviction','?')}/5, "
            f"bias={r.get('bias','?')}, conf={r.get('confidence','?')}) "
            f"| {outcome}"
        )
        if r.get("postmortem"):
            pm = r["postmortem"].strip().splitlines()[0][:180]
            lines.append(f"    postmortem: {pm}")
    return "\n".join(lines)


def run_memory_agent(state: Dict, config: Dict = None) -> Dict:
    ticker = state["ticker"]
    rows = get_recent_decisions(ticker, limit=5)
    block = _format_history_block(rows)

    return {
        "memory_history": rows,
        "memory_context": (
            f"Past decision history for {ticker} (most recent first):\n"
            f"{block}\n\n"
            "Use this only as context — do not anchor to prior decisions if "
            "current evidence disagrees. Prefer the freshest signal."
        ),
    }
