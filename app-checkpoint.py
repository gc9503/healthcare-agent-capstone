import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Import your agent logic from the notebook-exported script
from agent_core import run_agent  # make sure agent_core.py is in the same folder


# ---------- Helper functions to load data ----------

def load_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        return []
    data = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Load evaluation artifacts (if they exist) ----------

eval_runs = load_jsonl("eval_runs.jsonl")
eval_results = load_jsonl("eval_results.jsonl")
eval_metrics = load_json("eval_metrics.json")

# Some eval files (like grade_category) may not exist if you ran only some steps
# We'll handle that gracefully.

# ---------- Streamlit layout ----------

st.set_page_config(
    page_title="Healthcare Agent – Capstone",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Chatbot", "Evaluation Dashboard", "Run Logs"],
)


# ---------- Page 1: Chatbot ----------

if page == "Chatbot":
    st.title("🩺 Healthcare Agent – Chatbot")

    st.markdown(
        "Type a healthcare-related task below. "
        "The agent will plan, call tools (patient DB, medical history, appointments, "
        "disease search), and then summarize the results."
    )

    user_query = st.text_area(
        "Your query",
        height=150,
        placeholder=(
            "Example: My 70-year-old father has chronic kidney disease. "
            "Book a nephrologist and summarize latest treatments."
        ),
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Assistant")

    if run_button and user_query.strip():
        with st.spinner("Running agent..."):
            try:
                result = run_agent(user_query)
            except Exception as e:
                st.error(f"Error while running agent: {e}")
            else:
                st.subheader("✅ Final Answer")
                st.write(result["answer"])

                with st.expander("📋 Plan (goal decomposition)", expanded=False):
                    st.json(result["plan"])

                with st.expander("🛠 Tool Trace (calls & results)", expanded=False):
                    st.json(result["trace"])

    elif run_button and not user_query.strip():
        st.warning("Please enter a query before running the assistant.")


# ---------- Page 2: Evaluation Dashboard ----------

elif page == "Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard (QAEvalChain)")

    if not eval_results:
        st.info(
            "No evaluation results found. "
            "Run the Phase 2 evaluation steps in your notebook to generate "
            "`eval_results.jsonl` and `eval_metrics.json`."
        )
    else:
        # Show metrics summary
        st.subheader("Summary Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total evaluated cases",
            eval_metrics.get("total_eval_cases", len(eval_results)),
        )
        col2.metric(
            "Accuracy (%)",
            eval_metrics.get("accuracy_percent", 0.0),
        )
        col3.metric(
            "Correct answers",
            eval_metrics.get("correct", 0),
        )

        # Build a DataFrame of grade categories for charting
        df_results = pd.DataFrame(eval_results)

        if "grade_category" in df_results.columns:
            st.subheader("Grade Category Distribution")
            cat_counts = df_results["grade_category"].value_counts()
            st.bar_chart(cat_counts)
        else:
            st.info(
                "No 'grade_category' field found in eval_results. "
                "Make sure you ran Phase 2 Step 4 in the notebook."
            )

        st.subheader("Detailed Graded Results")
        show_cols = [
            "id",
            "query",
            "grade",
        ]
        existing_cols = [c for c in show_cols if c in df_results.columns]
        st.dataframe(df_results[existing_cols], use_container_width=True)

        st.markdown("Select a test case ID to inspect full details:")
        selected_id = st.selectbox(
            "Test case ID",
            options=df_results["id"].tolist(),
        )

        row = df_results[df_results["id"] == selected_id].iloc[0]
        st.write("### Query")
        st.write(row["query"])
        st.write("### Reference Answer (your expectation)")
        st.write(row["reference_answer"])
        st.write("### Predicted Answer (agent output)")
        st.write(row["predicted_answer"])
        st.write("### Grade / Explanation (QAEvalChain)")
        st.write(row["grade"])


# ---------- Page 3: Run Logs ----------

elif page == "Run Logs":
    st.title("📜 Run Logs (Plans & Traces)")

    if not eval_runs:
        st.info(
            "No run logs found. "
            "Run Phase 2 Step 2 in your notebook to generate `eval_runs.jsonl`."
        )
    else:
        df_runs = pd.DataFrame(eval_runs)
        st.subheader("All logged runs")
        st.dataframe(
            df_runs[["id", "timestamp", "query", "latency_seconds"]],
            use_container_width=True,
        )

        st.markdown("Select a run ID to inspect its plan & tool trace:")
        selected_id = st.selectbox(
            "Run ID",
            options=df_runs["id"].tolist(),
        )

        row = df_runs[df_runs["id"] == selected_id].iloc[0]

        st.write("### Query")
        st.write(row["query"])

        st.write("### Plan")
        st.json(row["plan"])

        st.write("### Tool Trace")
        st.json(row["trace"])
