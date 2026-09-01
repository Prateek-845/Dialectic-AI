# agents/challenger.py
# Agent A: The Challenger.
# Uses Web Search to bring in external facts and challenges the article.
import asyncio
from state import GraphState
from config import get_llm
from utils.tools import perform_web_search, generate_argument

async def challenger_node(state: GraphState) -> dict:
    article = state["original_article"]
    llm = get_llm("A", max_tokens=2000)
    
    try:
        search_context = await asyncio.wait_for(
            perform_web_search(article[:100] + " controversy criticism"),
            timeout=12.0
        )
    except asyncio.TimeoutError:
        search_context = "No external context found (Search timed out)."
    
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
    
    arg_summary = await generate_argument(llm, prompt, "ARGUMENT")
    return {"agent_a_summary": arg_summary}
