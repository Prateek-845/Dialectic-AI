# agents/fallacy_checker.py
# Agent Node: The Fallacy Checker.
# Penalizes Authority Scores if Ad Hominem or Emotional Appeals are detected.
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from state import GraphState
from config import get_llm

import asyncio

async def fallacy_checker_node(state: GraphState) -> dict:
    print(f"--- [Fallacy Checker Node] Starting (iteration={state.get('iteration', 0)}) ---")
    llm = get_llm("MEDIATOR", max_tokens=150)
    
    class FallacyCheckResult(BaseModel):
        has_fallacy: bool
        reasoning: str
        penalty: float

    async def check_fallacy(text: str, current_score: float, label: str) -> tuple[float, str]:
        if not text: return current_score, ""
        print(f"--- [Fallacy Checker Node] Checking fallacies for {label} ---")
        prompt = f"Analyze this argument for severe logical fallacies. Argument: {text}\nIf it contains a severe fallacy, set 'has_fallacy' to true, explain the 'reasoning', and set 'penalty' to 0.2. Otherwise, set penalty to 0.0."
        try:
            res = await llm.with_structured_output(FallacyCheckResult).ainvoke([HumanMessage(content=prompt)])
            print(f"--- [Fallacy Checker Node] Fallacy check output for {label}: has_fallacy={res.has_fallacy}, penalty={res.penalty} ---")
            if res.has_fallacy:
                return round(max(0.0, current_score - res.penalty), 3), res.reasoning
        except Exception as e:
            print(f"--- [Fallacy Checker Node] Exception checking fallacy for {label}: {str(e)} ---")
            pass
        return current_score, ""

    print("--- [Fallacy Checker Node] Gathering fallacy checks concurrently ---")
    (new_a, crit_a), (new_b, crit_b) = await asyncio.gather(
        check_fallacy(state.get("agent_a_summary", ""), state.get("a_score", 0.0), "Challenger A"),
        check_fallacy(state.get("agent_b_summary", ""), state.get("b_score", 0.0), "Supporter B")
    )
    
    log = state.get("debate_log", []) + [{
        "iteration": state.get("iteration", 1),
        "a_score": new_a,
        "b_score": new_b,
        "highlighted_text_a": state.get("highlighted_text_a", ""),
        "highlighted_text_b": state.get("highlighted_text_b", "")
    }]
    
    print(f"--- [Fallacy Checker Node] Completed (new A={new_a}, new B={new_b}) ---")
    return {"a_score": new_a, "b_score": new_b, "debate_log": log, "critique_a": crit_a, "critique_b": crit_b}
