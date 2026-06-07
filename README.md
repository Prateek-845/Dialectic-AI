# Dialectic AI

Dialectic AI is a full stack,RAG based multi agent debate platform designed to critically analyze news articles and URLs. By utilizing specialized large language model personas, the system orchestrates a structured, fact-checked debate regarding the claims made in a given text. The platform ultimately generates a synthesized, neutral summary, reducing bias and highlighting verified facts.

**Live Application:** [Dialectic AI on Vercel](https://dialectic-ai-two.vercel.app/)

## Architecture

The system is separated into three distinct microservices to ensure scalability and separation of concerns:

1. **Frontend (React, Vite, Tailwind CSS)**
   The client application handles real-time Server-Sent Events (SSE) streaming, providing a live view of the AI debate. It manages user authentication workflows and features interactive metric displays for ROUGE scores and Polarity Bias analysis.

2. **Backend (Node.js, Express, MongoDB)**
   The Node.js server acts as the central middleware. It manages JWT-based user authentication, securely proxies the SSE stream from the Python AI engine to the frontend, and persists debate histories and metrics in a MongoDB Atlas database.

3. **AI Engine (Python, FastAPI, LangGraph)**
   The core intelligence engine uses LangGraph to orchestrate a cyclic state machine of AI agents. It processes the text, executes the multi-agent graph asynchronously, performs real-time fact-checking, and streams the outputs back to the Node server via FastAPI.

## Core Features

* **Multi-Agent Debate:** The system dynamically instantiates specialized agents. A Challenger agent actively attacks the claims presented in the text, while a Supporter agent defends them based on source material. A Mediator agent oversees the debate and synthesizes the final arguments.
* **Real-Time Streaming:** The user interface updates dynamically in real-time as the agents formulate their arguments and counter-arguments.
* **Automated Fact Checking:** The engine automatically performs background web searches using DuckDuckGo to incorporate external context and verify claims made during the debate.
* **Entity Verification:** Named entities extracted from the text are color-coded based on factual verification. Entities are marked as verified within the text, verified via external web search, or flagged as unverified hallucinations.
* **Fallacy Checking:** The system analyzes generated arguments for logical contradictions and severe fallacies. If scores fall below a threshold, the graph routes back to the agents for a rewrite. Critical errors pause the execution for human intervention.
* **Human-in-the-Loop:** The debate execution can be paused, allowing human operators to provide guidance or corrections when logical errors or hallucinations are detected.
* **Advanced Metrics:** The platform calculates ROUGE scores to measure information conservation between the original article and the synthesis. It also calculates TextBlob Polarity to measure bias reduction, and quantifies debate influence to determine which agent had the most impact on the final summary.

## Agent Workflow

![Agent Workflow Diagram](./diagram.png)

1. **Analyst:** Scrapes the provided URL or text and dynamically defines the professional personas best suited for the debate.
2. **Challenger (Agent A):** Reads the article, executes a web search for opposing context, and attacks the primary claims.
3. **Supporter (Agent B):** Defends the article against the specific points and citations raised by the Challenger.
4. **Fact Checker:** Utilizes spaCy to extract Named Entities, verifying them against the original source text and external web context, adjusting scores accordingly.
5. **Fallacy Checker:** Evaluates the arguments for logical consistency. If an argument is flawed, the node routes back to the Challenger or Supporter for a rewrite. If the error is critical, it halts for human intervention.
6. **Mediator:** Synthesizes the verified arguments from both sides into a neutral, balanced summary and calculates the final quantitative metrics.

## Local Installation

Running the application locally requires initializing three separate environments.

### 1. Python AI Engine

Navigate to the `ai_engine` directory and set up the Python environment:

```bash
cd ai_engine
python -m venv .venv
```

Activate the virtual environment:
* Windows: `.venv\Scripts\activate`
* Mac/Linux: `source .venv/bin/activate`

Install dependencies and start the server:

```bash
pip install -r ../requirements.txt
```

Create a `.env` file in the root directory and add your `GROQ_API_KEY`.

```bash
python main.py
```

### 2. Node Backend

Navigate to the `backend` directory, install dependencies, and start the Express server:

```bash
cd backend
npm install
node server.js
```

Ensure your MongoDB instance is running locally or provide a `MONGODB_URI` in the backend environment variables.

### 3. React Frontend

Navigate to the `frontend` directory, install dependencies, and start the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

## Cloud Deployment

The application is designed for a distributed cloud architecture, with services hosted on platforms optimized for their specific runtime requirements.

1. **Database Layer (MongoDB Atlas)**
   The primary data store is hosted on MongoDB Atlas. It is configured to accept connections from the cloud environments and stores all user credentials, debate histories, and synthesized metrics securely.

2. **AI Engine (Render)**
   The FastAPI Python engine is deployed as a Web Service on Render, operating in a Python 3.11 environment. This ensures compatibility with modern AI packages such as LangGraph and asynchronous HTTP clients. It receives requests from the Node backend, executes the multi-agent graph, and streams the output back over HTTP.

3. **Node Backend (Render)**
   The Express.js server is hosted on Render as a distinct Node.js environment. It functions as a secure middleware layer, connecting directly to the MongoDB cluster via a secure URI and proxying the Server-Sent Events (SSE) from the AI engine to the client.

4. **React Frontend (Vercel)**
   The client-side Vite React application is deployed on Vercel's Edge Network for maximum global performance and rapid content delivery. It connects to the Node backend via standard REST endpoints and maintains the live SSE connection for the real-time debate user interface.
