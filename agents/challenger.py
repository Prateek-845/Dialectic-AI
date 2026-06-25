# agents/challenger.py
# Agent A: The Challenger.
# Uses Web Search to bring in external facts and challenges the article.
import asyncio
from state import GraphState
from config import get_llm
from utils.tools import perform_web_search, generate_argument

async def challenger_node(state: GraphState) -> dict:
    print(f"--- [Challenger Node] Starting (iteration={state.get('iteration', 0)}) ---")
    article = state["original_article"]
    llm = get_llm("A", max_tokens=300)
    
    print("--- [Challenger Node] Running external context web search ---")
    try:
        search_context = await asyncio.wait_for(
            perform_web_search(article[:100] + " controversy criticism"),
            timeout=12.0
        )
    except asyncio.TimeoutError:
        print("--- [Challenger Node] Web search timed out after 12s, proceeding with fallback context ---")
        search_context = "No external context found (Search timed out)."
    print(f"--- [Challenger Node] Web search completed (context len={len(search_context)}) ---")
    
    prompt = f"You are playing the role of a {state.get('persona_a', 'Challenger')}.\n\nOriginal Article:\n{article[:2000]}\n\nExternal Context:\n{search_context}\n\n"
    
    if state.get("iteration", 0) > 0:
        prompt += f"Previous factual score was low ({state.get('a_score', 0.0)}). Use specific names/dates.\n\n"
        if state.get("critique_a"):
            prompt += f"Fix this logical fallacy:\n{state.get('critique_a')}\n\n"
        prompt += f"Opponent's argument:\n{state.get('agent_b_summary', '')}\n\n"
        
    prompt += (
        "Write a highly concise critical argument (max 150 words) challenging the main claims. "
        "Cite specific facts. Plain text. No bullets. Enclose strictly in <ARGUMENT>...</ARGUMENT> tags."
    )
    
    print("--- [Challenger Node] Generating argument from LLM ---")
    arg_summary = await generate_argument(llm, prompt, "ARGUMENT")
    print(f"--- [Challenger Node] Completed (arg len={len(arg_summary) if arg_summary else 0}) ---")
    return {"agent_a_summary": arg_summary}
