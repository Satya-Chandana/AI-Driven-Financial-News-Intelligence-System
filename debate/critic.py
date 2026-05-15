"""Devil's-advocate critic.

Runs AFTER the risk arbiter has produced a candidate final decision.
Reads the entire transcript and tries to break the call. If the critique
is rated STRONG, internally fires one more round of debate -> trader -> risk
to give the system a chance to revise. Otherwise the candidate stands.
"""
import re
from llm_clients.factory import get_llm
from debate.bull_researcher import run_bull
from debate.bear_researcher import run_bear
from debate.arbiter import run_arbiter
from trader.trader_agent import run_trader
from risk.risk_panel import run_risk_panel
from risk.risk_arbiter import run_risk_arbiter

CRITIC_SYSTEM = """You are the desk's Devil's-Advocate Critic. You are the
last set of eyes before a decision ships.

You have just read:
- The bull and bear arguments
- The neutral arbiter's verdict
- The trader's rating and conviction
- All three risk specialists (tail / macro / liquidity)
- The risk arbiter's final decision

Your sole job: find the strongest single reason this decision could be wrong.
Be ruthless and specific. Cite real evidence from the transcript.

Then rate your own critique honestly:
- STRONG: a credible failure mode is being ignored or under-weighted. The
  decision should be re-debated.
- MODERATE: a worthwhile concern, but the decision can stand with a footnote.
- WEAK: nothing material — confirm the decision.

Do not manufacture critiques. WEAK is a valid honest answer.

Return strictly:

## Critique
4-6 sentences. The single strongest reason the call could be wrong, with cited evidence.

## Strength
<one of: WEAK / MODERATE / STRONG>

## Recommendation
One sentence. If STRONG, name the specific question the next debate round must answer.
"""


def _parse_strength(text: str) -> str:
    m = re.search(r"##\s*Strength\s*\n+\s*([A-Za-z]+)", text)
    if not m:
        return "WEAK"
    raw = m.group(1).strip().upper()
    return raw if raw in {"WEAK", "MODERATE", "STRONG"} else "WEAK"


def _build_critic_prompt(state: dict) -> str:
    return f"""Ticker: {state['ticker']}

Bull (final round):
{state.get('bull_argument','')}

Bear (final round):
{state.get('bear_argument','')}

Arbiter verdict:
{state.get('arbiter_verdict','')}

Trader call:
{state.get('trader_output','')}

Tail-Risk Analyst:
{state.get('tail_risk_output','')}

Macro / Regime Analyst:
{state.get('macro_output','')}

Liquidity Analyst:
{state.get('liquidity_output','')}

Risk Arbiter (candidate final decision):
{state.get('risk_output','')}

Find the strongest single reason this candidate decision could be wrong.
"""


def run_critic(state: dict, config: dict = None) -> dict:
    cfg = config or {}
    llm = get_llm(cfg)

    # First critique pass.
    resp = llm.invoke([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": _build_critic_prompt(state)},
    ])
    critique_text = resp.content
    strength = _parse_strength(critique_text)

    out = {
        "critic_output": critique_text,
        "critic_strength": strength,
        "critic_triggered_rerun": False,
    }

    if strength != "STRONG":
        return out

    # STRONG critique => one more debate round + re-trade + re-risk.
    rerun_state = dict(state)
    rerun_state["debate_round"] = (state.get("debate_rounds_completed", 1) or 1) + 1
    # Inject the critic's challenge into the bull/bear context so they engage with it.
    rerun_state["critic_challenge"] = critique_text

    bull_out = run_bull(rerun_state, cfg)
    rerun_state.update(bull_out)
    bear_out = run_bear(rerun_state, cfg)
    rerun_state.update(bear_out)
    arb_out = run_arbiter(rerun_state, cfg)
    rerun_state.update(arb_out)

    trader_out = run_trader(rerun_state, cfg)
    rerun_state.update(trader_out)

    risk_panel_out = run_risk_panel(rerun_state, cfg)
    rerun_state.update(risk_panel_out)
    risk_arb_out = run_risk_arbiter(rerun_state, cfg)
    rerun_state.update(risk_arb_out)

    out.update({
        "critic_triggered_rerun": True,
        "bull_argument": rerun_state["bull_argument"],
        "bear_argument": rerun_state["bear_argument"],
        "arbiter_verdict": rerun_state["arbiter_verdict"],
        "trader_output": rerun_state["trader_output"],
        "trader_rating": rerun_state["trader_rating"],
        "trader_conviction": rerun_state["trader_conviction"],
        "tail_risk_output": rerun_state.get("tail_risk_output", ""),
        "tail_risk_verdict": rerun_state.get("tail_risk_verdict", "NORMAL"),
        "macro_output": rerun_state.get("macro_output", ""),
        "macro_verdict": rerun_state.get("macro_verdict", "MIXED"),
        "liquidity_output": rerun_state.get("liquidity_output", ""),
        "liquidity_verdict": rerun_state.get("liquidity_verdict", "NORMAL"),
        "risk_output": rerun_state.get("risk_output", ""),
        "risk_adjustment": rerun_state.get("risk_adjustment", "KEEP"),
        "final_decision": rerun_state.get("final_decision", "HOLD"),
        "debate_rounds_completed": rerun_state["debate_round"],
    })
    return out
