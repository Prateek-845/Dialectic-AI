"""
app.py
The main Streamlit front-end entry point.
Run with: py -m streamlit run app.py
"""
import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
from utils.tools import auto_fetch_text
from graph import build_graph

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Global Font and Background */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        
        .sub-title {
            text-align: center;
            color: #8b949e;
            font-size: 1.2rem;
            margin-top: 0px;
            margin-bottom: 30px;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            border: none;
            color: white;
        }

        /* Input Area */
        .stTextArea textarea {
            background-color: #161b22;
            border: 1px solid #30363d;
            color: #c9d1d9;
            border-radius: 8px;
        }
        .stTextArea textarea:focus {
            border-color: #58a6ff;
            box-shadow: 0 0 0 1px #58a6ff;
        }

        /* Cards/Containers */
        div[data-testid="stExpander"] {
            background-color: #161b22;
            border-radius: 10px;
            border: 1px solid #30363d;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        div[data-testid="stExpander"] summary {
            background-color: #21262d;
            padding: 10px 15px;
        }
        div[data-testid="stExpander"] summary p {
            font-weight: 600;
            color: #ffffff;
            font-size: 1.1rem;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #161b22;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Highlight Legend */
        .legend-box {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 10px;
        }
        .legend-entity { color: #1e40af; background-color: #dbeafe; }
        .legend-web { color: #166534; background-color: #dcfce3; }
        .legend-unverified { color: #991b1b; background-color: #fee2e2; }

        /* Persona Cards */
        .persona-challenger {
            border-left: 4px solid #ef4444;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        .persona-supporter {
            border-left: 4px solid #3b82f6;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Dialectic AI", layout="wide")
    inject_custom_css()
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY! Please set it in your .env file (for local use) or in the Streamlit Cloud 'Secrets' settings (for deployment).")
        st.stop()
        
    st.markdown("<h1 class='main-title'>Dialectic AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>RAG based News Neutralizer via Fact Grounded Multiagent Debate</p>", unsafe_allow_html=True)
    
    # Session state initialization for UI flow
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False
        st.session_state.processed_article = ""
        st.session_state.final_end_state = None
    
    # --- Input Area (Centered if not started) ---
    if not st.session_state.debate_started:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Input Article or URL")
            article_input = st.text_area("Paste text or provide a valid URL here:", height=200, label_visibility="collapsed", placeholder="Paste article text or URL here...")
            
            if st.button("Start Debate Process", type="primary"):
                if not article_input:
                    st.warning("Please paste an article or URL.")
                    st.stop()
                    
                with st.spinner("Processing input (fetching URL if needed)..."):
                    st.session_state.processed_article = auto_fetch_text(article_input)
                    st.session_state.last_input = article_input
                    st.session_state.debate_started = True
                st.rerun()
        st.stop()
        
    actual_article = st.session_state.processed_article
        
    # --- Sidebar for Context ---
    with st.sidebar:
        st.markdown("### Original Article")
        st.markdown(f"<div style='font-size: 0.9em; color: #8b949e; max-height: 70vh; overflow-y: auto; padding-right: 5px;'>{actual_article[:3000]}...</div>", unsafe_allow_html=True)
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- Execute Graph ---
    if st.session_state.final_end_state is None:
        graph = build_graph()
        
        # Use st.status for a clean loading UI
        with st.status("Initializing Debate Agents...", expanded=True) as status:
            final_end_state = None
            for event in graph.stream({"original_article": actual_article}, stream_mode="values"):
                final_end_state = event
                if "final_summary" in event and event["final_summary"]:
                    status.update(label="Synthesis Complete!", state="complete", expanded=False)
                else:
                    iter_num = event.get('iteration', 0)
                    status.update(label=f"Generating arguments & Fact-checking (Round {iter_num+1})...")
            
            st.session_state.final_end_state = final_end_state
            st.rerun()

    final_end_state = st.session_state.final_end_state

    # --- Dashboard Layout: Tabs ---
    tab1, tab2 = st.tabs(["Live Debate & Fact Check", "Final Synthesis Report"])
    
    with tab1:
        st.markdown("### Agentic Debate Log")
        st.markdown("""
            <div style='margin-bottom: 20px;'>
                <span class='legend-box legend-entity'>Verified Entity</span>
                <span class='legend-box legend-web'>Verified Web Source</span>
                <span class='legend-box legend-unverified'>Unverified/Hallucinated</span>
            </div>
        """, unsafe_allow_html=True)

        if final_end_state and final_end_state.get("debate_log"):
            persona_a = final_end_state.get('persona_a', 'Challenger')
            persona_b = final_end_state.get('persona_b', 'Supporter')
            
            for round_data in final_end_state["debate_log"]:
                with st.expander(f"Round {round_data['iteration']}", expanded=True):
                    # Challenger
                    st.markdown(f"<div class='persona-challenger'><strong>{persona_a}</strong> (Score: <code>{round_data.get('a_score', 0)}</code>)</div>", unsafe_allow_html=True)
                    if round_data.get("highlighted_text_a"):
                        st.markdown(f"<div style='padding-left: 15px; margin-bottom: 20px;'>{round_data['highlighted_text_a']}</div>", unsafe_allow_html=True)
                    
                    # Supporter
                    st.markdown(f"<div class='persona-supporter'><strong>{persona_b}</strong> (Score: <code>{round_data.get('b_score', 0)}</code>)</div>", unsafe_allow_html=True)
                    if round_data.get("highlighted_text_b"):
                        st.markdown(f"<div style='padding-left: 15px;'>{round_data['highlighted_text_b']}</div>", unsafe_allow_html=True)

    with tab2:
        if final_end_state and final_end_state.get("final_summary"):
            col_rep, col_met = st.columns([1.5, 1])
            
            with col_rep:
                st.markdown("### Mediator's Neutral Synthesis")
                st.info(final_end_state["final_summary"])
                
            with col_met:
                st.markdown("### Evaluation Metrics")
                
                # ROUGE Metrics
                if final_end_state.get("synthesis_rouge"):
                    st.markdown("#### Information Conservation")
                    r = final_end_state["synthesis_rouge"]
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("ROUGE 1", f"{r.get('rouge1', 0):.3f}")
                    with mc2:
                        st.metric("ROUGE L", f"{r.get('rougeL', 0):.3f}")
                
                # Sentiment/Neutrality
                if final_end_state.get("synthesis_neutral"):
                    st.markdown("#### Sentiment Polarity")
                    st.caption("Values closer to 0 indicate neutrality")
                    s = final_end_state["synthesis_neutral"]
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.metric("Original Polarity", f"{s.get('original_polarity', 0):.3f}")
                    with sc2:
                        st.metric("Synthesis Polarity", f"{s.get('synthesis_polarity', 0):.3f}")
                
                # Influence
                if final_end_state.get("debate_influence"):
                    inf = final_end_state["debate_influence"]
                    st.markdown("#### Debate Influence")
                    st.caption("Percentage of influence on Mediator")
                    st.progress(inf.get('challenger', 0) / 100.0, text=f"Challenger: {inf.get('challenger', 0)}%")
                    st.progress(inf.get('supporter', 0) / 100.0, text=f"Supporter: {inf.get('supporter', 0)}%")

if __name__ == "__main__":
    main()
