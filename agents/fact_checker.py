# agents/fact_checker.py
# Algorithmic Node (No LLM): The Fact Checker.
# Uses spaCy for Entity overlap, DeBERTa for Contradiction, and generates Highlighted HTML.
import asyncio
import numpy as np
from state import GraphState
from config import load_spacy_model, load_nli_model
from utils.tools import perform_web_search

async def fact_checker_node(state: GraphState) -> dict:
    print(f"--- [Fact Checker Node] Starting (iteration={state.get('iteration', 0)}) ---")
    article, sum_a, sum_b = state["original_article"], state.get("agent_a_summary", ""), state.get("agent_b_summary", "")
    
    print("--- [Fact Checker Node] Loading spaCy and NLI models ---")
    nlp = load_spacy_model()
    nli_model = load_nli_model()
    iteration = state.get("iteration", 0)
    
    print("--- [Fact Checker Node] Parsing article with spaCy ---")
    doc_art = nlp(article)
    article_ents = {ent.text.lower() for ent in doc_art.ents if ent.label_ not in ["CARDINAL", "ORDINAL"]}
    print(f"--- [Fact Checker Node] Extracted {len(article_ents)} entities from article ---")
    
    def make_span(t: str, bg: str, fg: str, title: str) -> str:
        return f'<span style="background-color: {bg}; color: {fg}; padding: 2px; border-radius: 3px;" title="{title}">{t}</span>'

    async def score_and_highlight(summary_text: str, label: str) -> tuple[float, str]:
        if not summary_text: return 0.0, ""
        print(f"--- [Fact Checker Node] Scoring and highlighting {label} ---")
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
                print(f"--- [Fact Checker Node] entity '{ent.text}' not in article. Searching Web... ---")
                try:
                    res = await asyncio.wait_for(
                        perform_web_search(f"{ent.text} {article[:50]}"),
                        timeout=8.0
                    )
                except asyncio.TimeoutError:
                    print(f"--- [Fact Checker Node] Web search timed out for entity '{ent.text}' after 8s ---")
                    res = "No external context found (Search timed out)"
                print(f"--- [Fact Checker Node] Web search completed for '{ent.text}' (res len={len(res)}) ---")
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
        print(f"--- [Fact Checker Node] {label} score: {final_score} (ents: {cited}/{total_ents}) ---")
        return round(float(final_score), 3), "".join(highlighted_words)

    a_score, html_a = await score_and_highlight(sum_a, "Challenger A")
    b_score, html_b = await score_and_highlight(sum_b, "Supporter B")
    
    print(f"--- [Fact Checker Node] Completed (A={a_score}, B={b_score}) ---")
    return {
        "a_score": a_score, 
        "b_score": b_score, 
        "highlighted_text_a": html_a,
        "highlighted_text_b": html_b,
        "iteration": iteration + 1
    }
