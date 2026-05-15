"""Runs the bull/bear debate for N rounds, then hands off to the arbiter."""
from debate.bull_researcher import run_bull
from debate.bear_researcher import run_bear
from debate.arbiter import run_arbiter
from graphs.progress import emit


def run_debate(state: dict, config: dict = None, rounds: int = 2) -> dict:
    """Sequential debate: bull -> bear, repeated for `rounds` rounds, then arbiter."""
    cfg = config or {}
    rounds = int(cfg.get("debate_rounds", rounds))
    rounds = max(1, min(3, rounds))

    accumulated = {}
    working_state = dict(state)

    for r in range(1, rounds + 1):
        working_state["debate_round"] = r

        emit("start", f"Bull researcher (round {r}/{rounds})")
        bull_out = run_bull(working_state, cfg)
        working_state.update(bull_out)
        accumulated.update(bull_out)
        emit("done", f"Bull researcher (round {r}/{rounds})")

        emit("start", f"Bear researcher (round {r}/{rounds})")
        bear_out = run_bear(working_state, cfg)
        working_state.update(bear_out)
        accumulated.update(bear_out)
        emit("done", f"Bear researcher (round {r}/{rounds})")

    emit("start", "Neutral arbiter")
    arb_out = run_arbiter(working_state, cfg)
    accumulated.update(arb_out)
    accumulated["debate_rounds_completed"] = rounds
    emit("done", "Neutral arbiter")
    return accumulated
