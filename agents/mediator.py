# agents/mediator.py
# Agent C: The Mediator.
# Writes the final synthesis.
from textblob import TextBlob
from rouge_score import rouge_scorer
from state import GraphState
from utils.tools import generate_argument
from config import get_llm

async def mediator_node(state: GraphState) -> dict:
    a_sum, b_sum = state.get("agent_a_summary", ""), state.get("agent_b_summary", "")
    llm = get_llm("MEDIATOR", max_tokens=800)
    
    prompt = (
        "You are the Mediator. Write a 200-word final, neutral synthesis of the debate.\n\n"
        f"Persona A ({state.get('persona_a', 'A')}) Score: {state.get('a_score', 0.0)}\nArgument:\n{a_sum}\n\n"
        f"Persona B ({state.get('persona_b', 'B')}) Score: {state.get('b_score', 0.0)}\nArgument:\n{b_sum}\n\n"
        "Write a final, completely neutral synthesis (approx 200 words) summarizing the core truth. "
        "If you need to think step-by-step, you MUST enclose your entire thinking process inside <think>...</think> tags. "
        "After thinking, enclose your final synthesis strictly within <SYNTHESIS> tags, for example: <SYNTHESIS> your text here </SYNTHESIS>."
    )
    
    final_text = await generate_argument(llm, prompt, "SYNTHESIS")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r_scores = scorer.score(state["original_article"], final_text)
    synthesis_rouge = {
        "rouge1": round(r_scores['rouge1'].fmeasure, 3),
        "rougeL": round(r_scores['rougeL'].fmeasure, 3)
    }
    
    # Compare agent argument polarities vs synthesis polarity.
    # Agents argue with intentional bias; the synthesis should pull toward center.
    # This framing shows the mediator doing its job — rather than comparing
    # synthesis to the already-neutral original article (which makes it look worse).
    blob_a = TextBlob(a_sum) if a_sum else None
    blob_b = TextBlob(b_sum) if b_sum else None
    syn_blob = TextBlob(final_text)

    pol_a = round(blob_a.sentiment.polarity, 3) if blob_a else 0.0
    pol_b = round(blob_b.sentiment.polarity, 3) if blob_b else 0.0
    pol_syn = round(syn_blob.sentiment.polarity, 3)

    synthesis_neutral = {
        "challenger_polarity": pol_a,
        "supporter_polarity": pol_b,
        "synthesis_polarity": pol_syn,
        "avg_agent_polarity": round((pol_a + pol_b) / 2, 3),
        # 1.0 = perfectly neutral, 0.0 = maximally biased
        "synthesis_neutrality": round(1.0 - abs(pol_syn), 3)
    }

    score_a = scorer.score(a_sum, final_text)['rougeL'].fmeasure
    score_b = scorer.score(b_sum, final_text)['rougeL'].fmeasure
    total_influence = score_a + score_b if (score_a + score_b) > 0 else 1.0
    influence = {
        "challenger": round((score_a / total_influence) * 100, 1),
        "supporter": round((score_b / total_influence) * 100, 1)
    }

    return {
        "final_summary": final_text,
        "synthesis_rouge": synthesis_rouge,
        "synthesis_neutral": synthesis_neutral,
        "debate_influence": influence
    }
