"""
agents/supporter.py
Agent B: The Supporter.
Defends the article against the Challenger.
"""
from langchain_core.messages import HumanMessage
from state import GraphState
from config import get_llm

def supporter_node(state: GraphState) -> dict:
    article = state["original_article"]
    persona = state.get("persona_b", "Supporter")
    iteration = state.get("iteration", 0)
    llm = get_llm("B", max_tokens=300) # Heterogeneous model
    
    challenger_arg = state.get("agent_a_summary", "")
    
    prompt = f"You are playing the role of a {persona}.\n\n"
    prompt += f"Original Article:\n{article[:2000]}\n\n"
    prompt += f"Challenger ({state.get('persona_a', 'Opponent')}) argued:\n{challenger_arg}\n\n"
    
    if iteration > 0:
        b_score = state.get("b_score", 0.0)
        prompt += f"Your previous factual score was low ({b_score}). You must use MORE specific names and dates from the article.\n\n"
        
        critique = state.get("critique_b")
        if critique:
            prompt += f"Additionally, your previous argument was rejected by the Fallacy Checker for the following reason:\n{critique}\n"
            prompt += "Please fix this logical error and try again.\n\n"

    prompt += (
        "Write a highly concise defensive argument (maximum 150 words) supporting the article's core claims "
        "and countering the Challenger. Cite specific facts. Plain text only. No bullet points.\n"
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
        
    return {"agent_b_summary": content}
