#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from requests.exceptions import HTTPError
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

# Load API key
with open("openai_key.txt", "r") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

def get_llm(temp: float = 0.2):
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=temp,
    )


# In[2]:


import sys



# In[3]:


class PatientDB:
    def __init__(self, excel_path: str):
        # Load the Excel sheet into a DataFrame
        self.df = pd.read_excel(excel_path)
        # Ensure there is a patient_id column
        if "patient_id" not in self.df.columns:
            self.df["patient_id"] = range(1, len(self.df) + 1)

    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        # Case-insensitive match on the Name column
        matches = self.df[self.df["Name"].str.lower() == name.lower()]
        if matches.empty:
            return None
        # Return the first match as a dict
        return matches.iloc[0].to_dict()

# Create a global patient DB instance
patient_db = PatientDB("records.xlsx")

def patient_lookup_tool(name: str = None) -> Dict[str, Any]:
    """
    Tool: look up a patient by name in the records.xlsx file.
    """
    result = patient_db.find_by_name(name)
    if result is None:
        return {"success": False, "error": "Patient not found"}
    return {"success": True, "patient": result}

# quick sanity check (change name if needed)
patient_lookup_tool("Ramesh Kulkarni")


# In[4]:


pdf_paths = [
    "sample_patient.pdf",
    "sample_report_anjali.pdf",
    "sample_report_david.pdf",
    "sample_report_ramesh.pdf",
]

all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    for d in docs:
        d.metadata["source"] = path  # tag which PDF it came from
    all_docs.extend(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

medical_history_vs = FAISS.from_documents(all_docs, embeddings)
medical_history_vs.save_local("faiss_medical_history")

print("✅ FAISS index built and saved as 'faiss_medical_history'")


# In[5]:


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

medical_history_vs = FAISS.load_local(
    "faiss_medical_history",
    embeddings,
    allow_dangerous_deserialization=True,
)

def medical_history_tool(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Tool: search medical PDF reports for information related to the query.
    """
    docs = medical_history_vs.similarity_search(query, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    return {
        "success": True,
        "context": context,
        "docs_meta": [d.metadata for d in docs],
    }

# test
medical_history_tool("chronic kidney disease")


# In[6]:


class AppointmentScheduler:
    def __init__(self):
        # doctor_id -> list of bookings
        self.bookings = {}

    def book(self, specialty: str, patient_name: str, day: str = "2025-12-01"):
        doctor_id = f"doc_{specialty.replace(' ', '_').lower()}"
        slot = f"{day} 10:00"
        self.bookings.setdefault(doctor_id, []).append(
            {"patient": patient_name, "slot": slot}
        )
        return {
            "success": True,
            "doctor_id": doctor_id,
            "specialty": specialty,
            "slot": slot,
        }

scheduler = AppointmentScheduler()

def appointment_booking_tool(
    patient_name: str,
    specialty: str,
    day: str = "2025-12-01",
) -> Dict[str, Any]:
    """
    Tool: mock booking of an appointment for a patient.
    """
    return scheduler.book(specialty, patient_name, day)

# test
appointment_booking_tool("Ramesh Kulkarni", "nephrologist")


# In[7]:


# 7-9 define and test the Web Search disease tool

def _call_openai_web_search(prompt: str) -> str:
    """
    Low-level helper: call OpenAI Responses API with web_search_preview.
    Extract the assistant message text from the output.
    """
    url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "tools=v1",
    }

    body = {
        "model": "gpt-4.1-mini",
        "input": prompt,
        "tools": [
            {
                "type": "web_search_preview",
                "search_context_size": "medium",
            }
        ],
    }

    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()

    # ---- NEW, ROBUST PARSING ----
    try:
        output_items = data.get("output", [])

        # Find the first "message" item
        message_items = [o for o in output_items if o.get("type") == "message"]
        if not message_items:
            raise ValueError("No 'message' item found in output")

        msg = message_items[0]
        content_items = msg.get("content", [])

        # Find first content block that has text (type 'output_text')
        text_items = [c for c in content_items if c.get("type") == "output_text"]
        if not text_items:
            raise ValueError("No 'output_text' content found in message")

        text_value = text_items[0].get("text", "")
        if not isinstance(text_value, str):
            raise ValueError("Unexpected 'text' field shape")

    except Exception as e:
        logging.error(f"Unexpected Responses API format: {e}")
        logging.error(json.dumps(data, indent=2))
        raise

    return text_value


# In[8]:

def disease_search_tool(query: str) -> Dict[str, Any]:
    """
    Tool: use OpenAI web search to get up-to-date info about a disease.
    If web search hits a rate limit (429) or other HTTP error,
    fall back to a normal LLM-only summary (no web).
    """
    prompt = (
        "Use web search to gather up-to-date information from authoritative "
        "medical sources such as the World Health Organization (who.int), "
        "MedlinePlus (medlineplus.gov), and major clinical guidelines.\n\n"
        f"Condition: {query}\n\n"
        "Provide:\n"
        "- A short overview of the condition\n"
        "- Key current treatment options\n"
        "- Important monitoring or lifestyle considerations\n\n"
        "Write in clear bullet points. This is general information only and "
        "NOT medical advice; clearly advise consulting a doctor."
    )

    try:
        # 🔹 First try: external web search (Responses API)
        summary_text = _call_openai_web_search(prompt)
        source = "web_search"
    except HTTPError as e:
        # If we hit 429 or other HTTP error, fall back to LLM-only
        status = e.response.status_code if e.response is not None else "unknown"
        print(
            f"[disease_search_tool] Web search failed with HTTP {status}, "
            "falling back to LLM-only answer."
        )

        llm = get_llm(temp=0.2)
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a medical information assistant. "
                    "Provide factual, high-level information about diseases and treatments. "
                    "Do NOT give strict medical advice. Always say that a doctor should be consulted.",
                ),
                (
                    "human",
                    "Summarize the condition and common treatment options for: {disease}",
                ),
            ]
        )
        chain = fallback_prompt | llm
        response = chain.invoke({"disease": query})
        summary_text = response.content
        source = "llm_fallback"

    return {
        "success": True,
        "summary": summary_text,
        "source": source,  # helps you log whether it used web or fallback
    }


# In[9]:


res = disease_search_tool("chronic kidney disease")
print(res["summary"][:800])


# In[10]:


# Planner  model ( Subtask and Plan)

class SubTask(BaseModel):
    id: int
    description: str
    tool: str  # 'patient_lookup', 'medical_history', 'appointment_booking', 'disease_search', 'none'
    inputs: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    overall_goal: str
    subtasks: List[SubTask]



# In[18]:


# Planner Chain: Plan and Goal decomposition - step 8
def build_planner_chain():
    llm = get_llm(temp=0.1)

    system_prompt = """
You are a medical task planner for a virtual healthcare assistant.

Given a user request, you MUST:
1. Understand the overall goal.
2. Break it into an ordered list of clear subtasks.
3. For each subtask, pick exactly one tool name from this set:
   - "patient_lookup"
   - "medical_history"
   - "appointment_booking"
   - "disease_search"
   - "none"  (for reasoning-only steps)
4. For each subtask, also provide an "inputs" object (dictionary) with any needed fields,
   such as "patient_name", "specialty", "day", or "query".

You must return valid JSON with:
- a string field "overall_goal"
- a list field "subtasks"

Each element of "subtasks" must be an object with:
- integer "id"
- string "description"
- string "tool"
- object "inputs" (dictionary of arguments for that tool)

Do NOT include any extra keys. Do NOT include explanations outside the JSON.
Return ONLY the JSON object.
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User request: {user_query}"),
        ]
    )

    parser = JsonOutputParser(pydantic_object=Plan)
    chain = prompt | llm | parser
    return chain


def create_plan(user_query: str) -> Plan:
    chain = build_planner_chain()
    raw = chain.invoke({"user_query": user_query})

    logging.info(f"Raw plan from LLM/parser: {raw}")

    # If it's already a Plan instance, great
    if isinstance(raw, Plan):
        return raw

    # Else convert the raw dict into a Plan object
    plan = Plan.model_validate(raw)
    logging.info(f"Plan object: {plan}")
    return plan

# test planner again (optional)
test_plan = create_plan(
    "My 70-year-old father has chronic kidney disease. "
    "Book a nephrologist and summarize latest treatment methods."
)
test_plan





# In[20]:


# tooling/agent logic 

def dispatch_tool(subtask: SubTask, state: Dict[str, Any]) -> Dict[str, Any]:
    tool = subtask.tool
    inputs = subtask.inputs.copy()

    # Auto-fill patient_name if we already looked them up
    if "patient_name" not in inputs and "patient" in state:
        inputs["patient_name"] = state["patient"].get("Name")

    if tool == "patient_lookup":
        result = patient_lookup_tool(name=inputs.get("patient_name"))
        if result.get("success"):
            state["patient"] = result["patient"]

    elif tool == "medical_history":
        patient_name = inputs.get(
            "patient_name",
            state.get("patient", {}).get("Name", "the patient"),
        )
        q = inputs.get("query", "recent history and diagnoses")
        query = f"{patient_name}: {q}"
        result = medical_history_tool(query)

    elif tool == "appointment_booking":
        patient_name = inputs.get(
            "patient_name",
            state.get("patient", {}).get("Name", "the patient"),
        )
        specialty = inputs.get("specialty", "general physician")
        day = inputs.get("day", "2025-12-01")
        result = appointment_booking_tool(patient_name, specialty, day)

    elif tool == "disease_search":
        disease_q = inputs.get("query", "")
        result = disease_search_tool(disease_q)

    elif tool == "none":
        result = {"success": True, "info": "No tool used for this step."}

    else:
        result = {"success": False, "error": f"Unknown tool: {tool}"}

    return {"subtask": subtask.dict(), "result": result}



# In[21]:


def build_summary_chain():
    llm = get_llm(temp=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a virtual medical assistant.

You will receive:
- The original user query
- A trace of each tool call and its result

Your job:
- Explain clearly what was done (e.g., patient found, appointment booked).
- Summarize relevant medical information in simple language.
- DO NOT give strict medical advice.
- Always recommend consulting a healthcare professional.
                """.strip(),
            ),
            (
                "human",
                "User query: {user_query}\n\nTool trace:\n{trace_text}",
            ),
        ]
    )

    return prompt | llm


def run_agent(user_query: str) -> Dict[str, Any]:
    """
    Full Phase 1 pipeline:
    1. Plan (goal decomposition)
    2. Execute subtasks via tools
    3. Summarize into a final answer
    4. Return answer + plan + trace (for Phase 2 evaluation/Streamlit)
    """
    plan = create_plan(user_query)
    state: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []

    # Execute every subtask in order
    for st in plan.subtasks:
        step_output = dispatch_tool(st, state)
        trace.append(step_output)

    # Build a readable trace string
    trace_text = ""
    for step in trace:
        trace_text += f"Subtask: {step['subtask']['description']}\n"
        trace_text += f"Tool: {step['subtask']['tool']}\n"
        trace_text += f"Result: {json.dumps(step['result'], indent=2)}\n\n"

    summary_chain = build_summary_chain()
    final_answer = summary_chain.invoke(
        {"user_query": user_query, "trace_text": trace_text}
    )

    return {
        "answer": final_answer.content,
        "plan": plan.dict(),
        "trace": trace,
    }


# In[23]:


query = (
    "My 70-year-old father has chronic kidney disease. "
    "I want to book a nephrologist for him. "
    "Also, can you summarize latest treatment methods?"
)

result = run_agent(query)

print("=== FINAL ANSWER ===\n")
print(result["answer"])

print("\n=== PLAN ===\n")
print(json.dumps(result["plan"], indent=2))

print("\n=== TRACE (tool calls) ===\n")
print(json.dumps(result["trace"], indent=2))



# In[24]:


# Phase 2--- Query list

# ========= PHASE 2 – STEP 1: DEFINE EVALUATION DATASET =========

# Each item is one "test case" for my agent.
# - query: what the user asks the assistant
# - reference_answer: a rough description of what a GOOD answer should include
#  

evaluation_set = [
    {
        "id": 1,
        "query": (
            "My 70-year-old father has chronic kidney disease. "
            "Please book a nephrologist appointment for him and summarize "
            "the latest treatment options."
        ),
        "reference_answer": (
            "Should book a nephrologist appointment, mention the booked slot, "
            "and provide a clear summary of chronic kidney disease, including "
            "typical treatments (like blood pressure control, diet changes, "
            "possible dialysis or transplant in advanced stages). "
            "Should include a disclaimer to consult a doctor."
        ),
    },
    {
        "id": 2,
        "query": (
            "Find the patient record for Ramesh Kulkarni and summarize his "
            "medical history from the available PDF reports."
        ),
        "reference_answer": (
            "Should correctly identify Ramesh Kulkarni in records.xlsx, use the "
            "PDF medical history to summarize key conditions, diagnoses, and any "
            "important events. Should sound like a concise medical summary, not "
            "generic advice."
        ),
    },
    {
        "id": 3,
        "query": (
            "I think I might have type 2 diabetes. Can you explain what this "
            "condition is and what the usual treatments are?"
        ),
        "reference_answer": (
            "Should explain type 2 diabetes in simple language, describe insulin "
            "resistance, common treatments (lifestyle changes, metformin or other "
            "medications, sometimes insulin), and monitoring like HbA1c. "
            "Must clearly state it is not medical advice and to see a doctor."
        ),
    },
    {
        "id": 4,
        "query": (
            "Please book a follow-up appointment with a cardiologist for my father "
            "next week and summarize any heart-related issues from his reports."
        ),
        "reference_answer": (
            "Should schedule a cardiologist appointment (mock booking tool), show "
            "the doctor_id and slot, and summarize heart/cardiac information from "
            "available medical history if present. If not present, should say that "
            "no specific heart issues were found in the records."
        ),
    },
    {
        "id": 5,
        "query": (
            "Summarize the key health risks and lifestyle recommendations for "
            "someone with chronic kidney disease based on current medical guidance."
        ),
        "reference_answer": (
            "Should list specific CKD health risks and concrete lifestyle "
            "recommendations (diet, fluid management, blood pressure control, "
            "smoking cessation, regular monitoring), based on up-to-date web "
            "search or medical knowledge, plus a doctor-consultation disclaimer."
        ),
    },
]

print(f"Created evaluation_set with {len(evaluation_set)} test cases.")


# In[25]:


# ========= PHASE 2 – STEP 2: RUN AGENT ON EVALUATION SET =========

import time
from datetime import datetime

eval_runs = []

for case in evaluation_set:
    qid = case["id"]
    query = case["query"]
    reference = case["reference_answer"]

    print(f"\n=== Running test case {qid}: ===")
    print(f"Query: {query}\n")

    start_time = time.time()
    agent_output = run_agent(query)
    end_time = time.time()

    run_record = {
        "id": qid,
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "reference_answer": reference,
        "predicted_answer": agent_output["answer"],
        "plan": agent_output["plan"],
        "trace": agent_output["trace"],
        "latency_seconds": round(end_time - start_time, 2),
    }

    eval_runs.append(run_record)

    print(f"✅ Finished test case {qid} in {run_record['latency_seconds']} seconds.")
    print("Short preview of agent answer:\n")
    print(run_record["predicted_answer"][:400], "...\n")

# Save all runs to a JSONL file for later evaluation / Streamlit dashboard
with open("eval_runs.jsonl", "w", encoding="utf-8") as f:
    for r in eval_runs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n✅ Completed all evaluation runs.")
print(f"Saved {len(eval_runs)} runs to 'eval_runs.jsonl'.")


# In[27]:


# ========= PHASE 2 – STEP 3 (UPDATED): QAEvalChain EVALUATION =========

from langchain.evaluation.qa import QAEvalChain

# 1) Load eval_runs if not already loaded
if "eval_runs" not in globals():
    eval_runs = []
    with open("eval_runs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            eval_runs.append(json.loads(line))

print(f"Loaded {len(eval_runs)} evaluation runs for grading.")

# 2) Build examples & predictions
examples = []
predictions = []

for r in eval_runs:
    examples.append(
        {
            "query": r["query"],
            "answer": r["reference_answer"],
        }
    )
    predictions.append(
        {
            "result": r["predicted_answer"],
        }
    )

print(f"Prepared {len(examples)} examples and {len(predictions)} predictions.")

# 3) Build the evaluation chain
judge_llm = get_llm(temp=0.0)
eval_chain = QAEvalChain.from_llm(judge_llm)

# 4) Run evaluation
graded_outputs = eval_chain.evaluate(examples, predictions)

print("\nRaw graded_outputs from QAEvalChain:")
print(json.dumps(graded_outputs, indent=2))

# 5) Normalize grades into a clean list
def extract_grade_text(grade_dict):
    """
    Different LangChain versions put the explanation in different keys.
    Try several options, then fall back to the whole dict as string.
    """
    if isinstance(grade_dict, str):
        return grade_dict

    # Common shapes:
    # 1) {"text": "..."}
    if "text" in grade_dict and grade_dict["text"]:
        return grade_dict["text"]

    # 2) {"results": [{"text": "..."}]}
    if "results" in grade_dict and grade_dict["results"]:
        first = grade_dict["results"][0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]

    # 3) {"reasoning": "...", "score": ...}
    parts = []
    if "score" in grade_dict:
        parts.append(f"Score: {grade_dict['score']}")
    if "reasoning" in grade_dict:
        parts.append(f"Reasoning: {grade_dict['reasoning']}")
    if parts:
        return " | ".join(parts)

    # Fallback: just dump the dict
    return json.dumps(grade_dict)


eval_results = []
for run, grade in zip(eval_runs, graded_outputs):
    grade_text = extract_grade_text(grade)
    eval_results.append(
        {
            "id": run["id"],
            "query": run["query"],
            "reference_answer": run["reference_answer"],
            "predicted_answer": run["predicted_answer"],
            "grade": grade_text,
            "latency_seconds": run.get("latency_seconds", None),
        }
    )

# 6) Save eval_results
with open("eval_results.jsonl", "w", encoding="utf-8") as f:
    for r in eval_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n✅ QAEvalChain evaluation complete (updated).")
print(f"Saved {len(eval_results)} graded results to 'eval_results.jsonl'.")

# 7) Show preview
for r in eval_results:
    print("\n------------------------------")
    print(f"Test case ID: {r['id']}")
    print(f"Query: {r['query'][:120]}...")
    print(f"Grade / Explanation: {r['grade']}")



# In[28]:


# ========= PHASE 2 – STEP 4: COMPUTE METRICS FROM eval_results =========

import json
from collections import Counter

# Load graded evaluation results
eval_results = []
with open("eval_results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        eval_results.append(json.loads(line))

print(f"Loaded {len(eval_results)} graded results.")


# --- Helper to classify grade text into simple buckets ---
def classify_grade(grade_text: str):
    """
    Map free-form QAEvalChain evaluations into simple categories.
    This does NOT need to be perfect — the project only requires a reasonable attempt.
    """
    text = grade_text.lower()

    if "correct" in text and "partial" not in text:
        return "correct"
    if "partially" in text or "partial" in text:
        return "partially_correct"
    if "incorrect" in text or "wrong" in text:
        return "incorrect"
    if "incomplete" in text or "missing" in text:
        return "incomplete"

    # everything else
    return "other"


# --- Apply classification ---
categories = []
for r in eval_results:
    cat = classify_grade(r["grade"])
    categories.append(cat)
    r["grade_category"] = cat

# Count how many of each category
counts = Counter(categories)

# Compute simple metrics
total = len(eval_results)
metrics = {
    "total_eval_cases": total,
    "correct": counts.get("correct", 0),
    "partially_correct": counts.get("partially_correct", 0),
    "incorrect": counts.get("incorrect", 0),
    "incomplete": counts.get("incomplete", 0),
    "other": counts.get("other", 0),
    "accuracy_percent": round((counts.get("correct", 0) / total) * 100, 2) if total else 0.0,
}

print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save metrics for use in Streamlit dashboard
with open("eval_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved evaluation metrics to 'eval_metrics.json'.")


# In[ ]:




