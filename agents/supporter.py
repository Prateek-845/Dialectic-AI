# agents/supporter.py
# Agent B: The Supporter.
# Defends the article against the Challenger.
from state import GraphState
from config import get_llm
from utils.tools import generate_argument

async def supporter_node(state: GraphState) -> dict:
    article = state["original_article"]
    llm = get_llm("B", max_tokens=2000)
    
    prompt = f"You are playing the role of a {state.get('persona_b', 'Supporter')}.\n\nOriginal Article:\n{article[:2000]}\n\nChallenger argued:\n{state.get('agent_a_summary', '')}\n\n"
    
    if state.get("iteration", 0) > 0:
        prompt += f"Previous factual score was low ({state.get('b_score', 0.0)}). Use specific names/dates.\n\n"
        if state.get("critique_b"):
            prompt += f"Fix this logical fallacy:\n{state.get('critique_b')}\n\n"

    prompt += (
        "Write a highly concise defensive argument (max 150 words) supporting the article's core claims "
        "and countering the Challenger. Cite specific facts. Plain text. No bullets. Enclose your argument strictly in <ARGUMENT> tags, for example: <ARGUMENT> your text here </ARGUMENT>."
    )
    
    arg_summary = await generate_argument(llm, prompt, "ARGUMENT")
    return {"agent_b_summary": arg_summary}
