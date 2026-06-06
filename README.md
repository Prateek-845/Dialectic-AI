# Dialectic AI 🧠⚖️

**Dialectic AI** is an advanced, multi-agent AI debate platform that analyzes articles, links, and news, pitting LLM personas against each other in a structured, fact-checked debate before synthesizing a highly neutral summary.

By leveraging **LangGraph**, **FastAPI**, and modern **React**, Dialectic AI attempts to eradicate media bias, unverified claims, and hallucination by forcing AI agents to debate from opposing perspectives while being actively monitored by a Fact-Checker and a Fallacy Checker.

---

## 🏗️ Architecture & Tech Stack

The project relies on a highly modular microservice architecture:

1. **Frontend (React + Vite)**: A sophisticated, enterprise-grade dark-mode SPA styled with Tailwind CSS. It connects via Server-Sent Events (SSE) to stream the debate live to the user.
2. **Backend (Node.js + Express + MongoDB)**: Handles user authentication, JSON Web Tokens (JWT), session state, and saving completed debate logs/history to MongoDB.
3. **AI Engine (Python + FastAPI)**: A robust Python microservice that orchestrates the **LangGraph** multi-agent state machine.
4. **LLM Provider (Groq)**: The project uses open-source models (Llama 3.3, Mixtral) running at blistering speeds via Groq's LPU inference engine.

---

## ✨ Core Features

- **Dynamic Persona Generation**: The system reads the article and automatically creates two opposing expert personas (e.g., "Economist" vs. "Privacy Advocate").
- **Agentic Debate Log**: The Challenger and Supporter take turns arguing the topic over multiple iterations.
- **Strict Fact Checking**: A dedicated "Fact Checker" agent intercepts claims and highlights them via raw HTML injections:
  - 🟢 **Green:** Verified Entity
  - 🔵 **Blue:** Verified Web Source
  - 🔴 **Red:** Unverified/Hallucinated
- **Fallacy Detection**: A hidden "Fallacy Checker" analyzes arguments for logical fallacies and quietly penalizes the agents' scores in the background.
- **Human-In-The-Loop**: If an agent starts relying on too many unverified claims, the state machine pauses and asks the user (the "Jury") for feedback before continuing.
- **Neutral Synthesis**: The final "Mediator" reads the entire debate log and synthesizes the core truth.
- **Advanced Metrics**:
  - **ROUGE Scores**: Evaluates how much core information from the original article survived the debate.
  - **Sentiment Polarity**: Measures the emotional bias of the original article vs. the neutral synthesis using `TextBlob`.
  - **Debate Influence**: Calculates exactly what percentage of the final synthesis was influenced by the Challenger vs. the Supporter.

---

## 🧩 The LangGraph Agent Workflow

Dialectic AI relies on a state graph (`state.py` and `graph.py`) to manage the debate:

1. **Analyst (Agent 0)**: Reads the article, decides on the two opposing personas.
2. **Challenger (Agent A)**: Reads the article, drafts a highly critical argument against the premise.
3. **Supporter (Agent B)**: Reads the article and the Challenger's argument, drafts a defensive argument.
4. **Fact Checker & Evaluator**: Validates entities using `spaCy` and optionally DuckDuckGo web search. Assigns an internal score to the arguments.
5. **Fallacy Checker**: Looks for strawman arguments or ad-hominem attacks, generating a CoT (Chain of Thought) critique.
6. **Mediator (Agent C)**: Runs once the debate concludes to write the final summary and calculate ROUGE/Polarity metrics.

---

## 🚀 Running Locally

### Prerequisites
- Node.js (v18+)
- Python (3.10+)
- MongoDB (Running locally on port 27017 or a MongoDB URI)
- A Groq API Key

### 1. Setup the AI Engine (Python)
Navigate to the root directory and activate your virtual environment:
```bash
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Install requirements
pip install -r requirements.txt

# Start the FastAPI engine
python ai_engine/main.py
```
*The engine will run on `http://localhost:8000`.*

### 2. Setup the Express Backend (Node.js)
Open a new terminal tab:
```bash
cd backend
npm install

# Create a .env file and add your secrets (JWT_SECRET, MONGODB_URI)
# Start the server
node server.js
```
*The backend will run on `http://localhost:5000`.*

### 3. Setup the React Frontend (Vite)
Open a new terminal tab:
```bash
cd frontend
npm install

# Start the development server
npm run dev
```
*The frontend will run on `http://localhost:5173`.*

---

## 🌐 Deployment 
To deploy this project to the cloud for free:
1. **Database:** Deploy your database on **MongoDB Atlas** (Free M0 Cluster).
2. **AI Engine:** Deploy the `/ai_engine` folder as a Python Web Service on **Render**.
3. **Backend:** Deploy the `/backend` folder as a Node Web Service on **Render**. Ensure you set the `PYTHON_API_URL` environment variable to point to your AI Engine.
4. **Frontend:** Deploy the `/frontend` folder to **Vercel**. Ensure you set `VITE_API_URL` to point to your Node Backend.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.
