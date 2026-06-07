"""
agents/challenger.py
Agent A: The Challenger.
Uses Web Search to bring in external facts and challenges the article.
"""
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm
from utils.tools import perform_web_search

def challenger_node(state: GraphState) -> dict:
    article = state["original_article"]
    persona = state.get("persona_a", "Challenger")
    iteration = state.get("iteration", 0)
    llm = get_llm("A", max_tokens=300) # Heterogeneous model
    

    search_context = perform_web_search(article[:100] + " controversy criticism")
    
    prompt = f"You are playing the role of a {persona}.\n\n"
    prompt += f"Original Article:\n{article[:2000]}\n\n"
    prompt += f"External Web Search Context:\n{search_context}\n\n"
    
    if iteration > 0:
        a_score = state.get("a_score", 0.0)
        prompt += f"Your previous factual score was low ({a_score}). You must use MORE specific names and dates from the article.\n\n"
        
        critique = state.get("critique_a")
        if critique:
            prompt += f"Additionally, your previous argument was rejected by the Fallacy Checker for the following reason:\n{critique}\n"
            prompt += "Please fix this logical error and try again.\n\n"
            
        prompt += f"Opponent's argument:\n{state.get('agent_b_summary', '')}\n\n"
        
    prompt += (
        "Write a highly concise critical argument (maximum 150 words) challenging the article's main claims. "
        "You must cite specific facts from both the article and the external context. "
        "Plain text only. No bullet points.\n"
        "IMPORTANT: You must enclose your final, polished argument strictly within <ARGUMENT> and </ARGUMENT> tags. "
        "Do not include any internal thoughts or reasoning inside these tags."
    )
    
    result = llm.invoke([HumanMessage(content=prompt)])
    content = result.content.strip()
    
    import re

    match = re.search(r'<ARGUMENT>(.*?)</ARGUMENT>', content, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:

        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
    return {"agent_a_summary": content}
