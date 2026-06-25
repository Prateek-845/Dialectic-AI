# agents/analyst.py
# Agent 0: The Analyst. Reads the article and dynamically decides what two perspectives should debate it.
import json
import re
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm

async def analyst_node(state: GraphState) -> dict:
    print("--- [Analyst Node] Starting ---")
    llm = get_llm("MEDIATOR")
    prompt = (
        "You are a News Analyst. Read the following article snippet and identify two "
        "distinct, opposing professional personas (e.g., 'Economist', 'Privacy Advocate', 'Doctor') "
        "who would have a fierce but factual debate about this topic.\n\n"
        f"Article:\n{state['original_article'][:1500]}\n\n"
        "Return EXACTLY a JSON dictionary like this: "
        '{"persona_a": "First Persona Name", "persona_b": "Second Persona Name"}'
    )
    
    try:
        print("--- [Analyst Node] Invoking LLM ---")
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        data = json.loads(re.search(r"\{.*\}", result.content.strip(), re.DOTALL).group(0))
        res = {"persona_a": data.get("persona_a", "Challenger"), "persona_b": data.get("persona_b", "Supporter")}
        print(f"--- [Analyst Node] Completed successfully: {res} ---")
        return res
    except Exception as e:
        print(f"--- [Analyst Node] Exception in invoking LLM: {str(e)}. Falling back to defaults. ---")
        fallback = {"persona_a": "Skeptic", "persona_b": "Defending Authority"}
        print(f"--- [Analyst Node] Completed with fallback: {fallback} ---")
        return fallback
