# agents/analyst.py
# Agent 0: The Analyst. Reads the article and dynamically decides what two perspectives should debate it.
import json
import re
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm

async def analyst_node(state: GraphState) -> dict:
    llm = get_llm("MEDIATOR", max_tokens=60)  # Only needs ~20 tokens for the JSON response
    prompt = (
        "You are a News Analyst. Read the following article snippet and identify two "
        "distinct, opposing professional personas (e.g., 'Economist', 'Privacy Advocate', 'Doctor') "
        "who would have a fierce but factual debate about this topic.\n\n"
        f"Article:\n{state['original_article'][:1500]}\n\n"
        "Return EXACTLY a JSON dictionary like this, with no other text: "
        '{"persona_a": "First Persona Name", "persona_b": "Second Persona Name"}'
    )

    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        # Strip <think>...</think> blocks first — qwen thinking model wraps its
        # reasoning in think tags which confuses the greedy {.*} regex (DOTALL).
        clean = re.sub(r'<think>.*?</think>', '', result.content, flags=re.DOTALL | re.IGNORECASE).strip()
        data = json.loads(re.search(r"\{.*\}", clean, re.DOTALL).group(0))
        return {"persona_a": data.get("persona_a", "Challenger"), "persona_b": data.get("persona_b", "Supporter")}
    except Exception:
        return {"persona_a": "Skeptic", "persona_b": "Defending Authority"}
