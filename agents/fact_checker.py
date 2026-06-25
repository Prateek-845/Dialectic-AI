# agents/fact_checker.py
# Algorithmic Node (No LLM): The Fact Checker.
# Uses spaCy for Entity overlap, DeBERTa for Contradiction, and generates Highlighted HTML.
import asyncio
import numpy as np
from state import GraphState
from config import load_spacy_model, load_nli_model
from utils.tools import perform_web_search

async def fact_checker_node(state: GraphState) -> dict:
    article, sum_a, sum_b = state["original_article"], state.get("agent_a_summary", ""), state.get("agent_b_summary", "")
    
    nlp = load_spacy_model()
    nli_model = load_nli_model()
    iteration = state.get("iteration", 0)
    
    doc_art = nlp(article)
    article_ents = {ent.text.lower() for ent in doc_art.ents if ent.label_ not in ["CARDINAL", "ORDINAL"]}
    
    def make_span(t: str, bg: str, fg: str, title: str) -> str:
        return f'<span style="background-color: {bg}; color: {fg}; padding: 2px; border-radius: 3px;" title="{title}">{t}</span>'

    async def score_and_highlight(summary_text: str) -> tuple[float, str]:
        if not summary_text: return 0.0, ""
        doc_sum = nlp(summary_text)
        cited, total_ents, last_idx, web_search_count = 0, 0, 0, 0
        highlighted_words = []

        for ent in doc_sum.ents:
            if ent.label_ in ["CARDINAL", "ORDINAL"]: continue
            total_ents += 1
            highlighted_words.append(summary_text[last_idx:ent.start_char])

            import re
            ent_clean = ent.text.lower().replace("'s", "").replace("’s", "")
            ent_clean = re.sub(r'^[^\w]+|[^\w]+$', '', ent_clean).strip()
            
            is_in_article = ent_clean in article.lower() or any(ent_clean in a or a in ent_clean for a in article_ents)

            if is_in_article:
                cited += 1
                highlighted_words.append(make_span(ent.text, "#d4edda", "#155724", "Verified Entity"))
            elif web_search_count < 5:
                try:
                    res = await asyncio.wait_for(
                        perform_web_search(f"{ent.text} {article[:50]}"),
                        timeout=8.0
                    )
                except asyncio.TimeoutError:
                    res = "No external context found (Search timed out)"
                web_search_count += 1
                if ent_clean in res.lower() and "No external" not in res:
                    cited += 1
                    highlighted_words.append(make_span(ent.text, "#cce5ff", "#004085", "Verified via Web"))
                else:
                    highlighted_words.append(make_span(ent.text, "#f8d7da", "#721c24", "Unverified"))
            else:
                highlighted_words.append(make_span(ent.text, "#f8d7da", "#721c24", "Rate Limited"))
            last_idx = ent.end_char
            
        highlighted_words.append(summary_text[last_idx:])
        base_score = (cited / total_ents) if total_ents > 0 else 0.5
        
        penalty = 0.0
        if nli_model:
            sentences = [sent.text for sent in doc_sum.sents]
            for sent in sentences:
                logits = nli_model.predict([[article[:1000], sent]])
                probs = np.exp(logits) / np.sum(np.exp(logits))
                if probs[0][0] > 0.85: # Contradiction label
                    penalty = 0.5
                    break
                    
        final_score = max(0.0, min(1.0, base_score * (1.0 - penalty)))
        return round(float(final_score), 3), "".join(highlighted_words)

    a_score, html_a = await score_and_highlight(sum_a)
    b_score, html_b = await score_and_highlight(sum_b)
    
    return {
        "a_score": a_score, 
        "b_score": b_score, 
        "highlighted_text_a": html_a,
        "highlighted_text_b": html_b,
        "iteration": iteration + 1
    }
