Healthcare Agent – Purdue GenAI Capstone
An Agentic Healthcare Assistant with Planning, Tools, RAG, Evaluation, and Streamlit UI

This project is the final capstone for the Purdue Generative AI Program, demonstrating:

Agentic LLM planning

Retrieval-Augmented Generation (RAG)

Tool execution (patient DB, PDF medical history, appointment booking, disease info)

LLMOps (evaluation, metrics, logs)

A full Streamlit application

Features
🔹 1. Agentic Workflow & Planning

The system uses an LLM to:

Parse a natural-language query

Generate a structured plan (JSON)

Choose tools

Execute them in sequence

Summarize the final answer

2. Integrated Tools

The agent uses the following tools:

Tool	Purpose
patient_lookup_tool	Search patient in Excel DB
medical_history_tool	RAG over clinical PDF reports (FAISS vector store)
appointment_booking_tool	Mock specialist appointment scheduler
disease_search_tool	Disease explanation via web search API or LLM fallback

3. Retrieval-Augmented Generation (RAG)

PDFs are parsed with PyPDFLoader

Embedded via sentence-transformers/all-MiniLM-L6-v2

Stored in a FAISS index

Retrieved chunks feed into the final summarization LLM

4. LLMOps Evaluation

Includes 5 test cases

Automated grading via QAEvalChain

Stores:

eval_runs.jsonl

eval_results.jsonl

eval_metrics.json

5. Streamlit Application

Provides three main views:

Chatbot

Evaluation Dashboard

Run Logs



📁 Repository Structure
healthcare-agent-capstone/
│
├── agent_core.py                 # Main agent: tools, planner, dispatcher, summarizer
├── app.py                        # Streamlit UI
├── healthcare_agent.ipynb        # Development notebook (Phase 1 & 2)
├── records.xlsx                  # Patient database
├── sample_patient.pdf            # PDF history files
├── sample_report_anjali.pdf
├── sample_report_david.pdf
├── sample_report_ramesh.pdf
├── eval_runs.jsonl               # Raw agent runs
├── eval_results.jsonl            # QAEvalChain graded results
├── eval_metrics.json             # Computed metrics for dashboard
├── requirements.txt              # Python dependencies
└── README.md                     # This file

🔑 API Key Setup (IMPORTANT)

For security reasons, my OpenAI API key is not included in the repo.

Before running the app:

Create a file named openai_key.txt in the project root.

Paste your OpenAI API key inside:


Save the file.

The app will automatically load it.

Installation & Running the App

1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit
streamlit run app.py

The app will open at:

http://localhost:8501

Evaluation Workflow

Open the notebook healthcare_agent.ipynb

Run:

Phase 1: Build tools, planner, RAG, agent

Phase 2: Run evaluation on test cases

Save results (JSONL)

Results will appear in:

Evaluation Dashboard tab (in Streamlit)

Run Logs tab
