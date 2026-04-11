# 🧠 Dialectic AI: The Verifiable News Synthesis Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dialectic-ai-ahdhkarxnqe9m7bddt5jhw.streamlit.app/)

Welcome to **Dialectic AI**, a research-grade NLP system designed to solve media bias and AI hallucinations through an adversarial, fact-grounded "Dialectic" framework.

## 🏛️ 1. The Core Philosophy: "The Dialectic"
Inspired by the Hegelian Dialectic (Thesis + Antithesis = Synthesis), this project moves beyond simple "AI Summarization." Modern news is often polarized; Dialectic AI neutralizes this by forcing two specialized agents into a controlled debate:
- **The Supporter (Thesis):** Defends the core claims of the input text.
- **The Challenger (Antithesis):** Scrutinizes biases, identifying logical leaps and omissions.
- **The Mediator (Synthesis):** Evaluates the debate, prioritizes factually grounded arguments, and produces a final, objective synthesis.

---

## 🏗️ 2. High-Level Architecture (LangGraph)
Built on **LangGraph**, the system operates as a state-of-the-art multi-agent state machine. Unlike a linear prompt chain, Dialectic AI features cyclic reasoning and autonomous "Reflexion" loops.

### The Graph Pipeline:
1.  **Analyst Node:** Dynamically assigns professional "Personas" to the agents (e.g., *Privacy Advocate* vs. *Silicon Valley CEO*) based on the article's topic.
2.  **Adversarial Nodes:** Challenger and Supporter agents generate contrasting summaries of the news.
3.  **Fact-Checker Node:** An algorithmic engine that performs deep linguistic cross-referencing between the AI's claims and the source.
4.  **Fallacy Detector:** A node that uses **Chain-of-Thought (CoT)** to critique the agents' logic.
5.  **Router Logic:** If an agent's "Authority Score" is too low, the graph autonomously loops back for a **Reflexion rewrite**.
6.  **Mediator Node:** Synthesizes the final report using the most verified points from both sides.

---

## 🔍 3. Advanced RAG & Multi-Source Grounding
This project utilizes a multi-layered **Retrieval-Augmented Generation (RAG)** pipeline:
- **URL Scraper RAG:** Automatically fetches and cleans HTML from live article links.
- **Just-In-Time Entity RAG:** During verification, if an agent brings in a fact not in the source text, the system invokes the **DuckDuckGo Search API** in real-time to distinguish between "helpful context" and "hallucination."

---

## 🧪 4. Technical Stack
- **Inference Cloud:** [Groq](https://groq.com/) (Llama 3.3-70b & Qwen-32b) for sub-second, high-reasoning debate rounds.
- **NER Engine:** `spaCy` (`en_core_web_md`) for localized entity tracking.
- **Semantic Verification:** `DeBERTa-v3-small` Cross-Encoder for Natural Language Inference (NLI), detecting contradictions between AI claims and source data.
- **Quantified Truth:** Calculation of **ROUGE** (Information Conservation) and **Sentiment Neutrality** metrics.

---

## 🚀 5. Deployment & Setup

### Streamlit Cloud
This app is optimized for Streamlit Cloud.
1. Fork this repo.
2. Connect to Streamlit Cloud.
3. **Important:** Add your `GROQ_API_KEY` to the **Secrets** management on the Streamlit dashboard.

### Local Installation
1.  **Clone & Install**:
    ```bash
    git clone https://github.com/Prateek-845/Dialectic-AI.git
    cd Dialectic-AI
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```
2.  **Environment Variables**: Create a `.env` file with your `GROQ_API_KEY`.
3.  **Run**:
    ```bash
    python -m streamlit run app.py
    ```

---

## 📊 6. Quantitative Authority Metrics
Dialectic AI provides a visual "Confidence Map" using color-coded HTML highlights:
- 🟢 **Verified**: Claim was found directly in the source text.
- 🔵 **Web-Verified**: Claim was confirmed via real-time DuckDuckGo RAG search.
- 🔴 **Unverified**: Potential hallucination - heavily penalized in the "Authority Score."

---

## 🛡️ License
MIT License. Built for ethical AI research.
