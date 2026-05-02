"""Streamlit Text-to-SQL application — run with: streamlit run app.py"""

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.benchmark import load_benchmark_dataset, run_benchmark, BenchmarkReport
from core.db import DatabaseManager
from core.llm import get_client
from core.pipeline import TextToSQLPipeline, PipelineResult, MAX_RETRIES
from core.pipeline import StreamChunk

load_dotenv()

# Streamlit Community Cloud: pull secrets into env vars so all modules
# continue to use os.getenv() without any cloud-specific code in core/.
try:
    for _key in ("NVIDIA_API_KEY", "DB_PATH"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

st.set_page_config(
    page_title="Text-to-SQL",
    page_icon="🗄️",
    layout="wide",
)


# ------------------------------------------------------------------ #
# Cached shared resources
# ------------------------------------------------------------------ #

@st.cache_resource
def get_db() -> DatabaseManager:
    db_path = os.getenv("DB_PATH", "data/sample.db")
    return DatabaseManager(db_path)


@st.cache_resource
def get_pipeline() -> TextToSQLPipeline:
    return TextToSQLPipeline(db=get_db(), client=get_client())


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Each entry: {role, content, sql, rows, columns, reasoning, attempts}

# ------------------------------------------------------------------ #
# Layout
# ------------------------------------------------------------------ #

tab_chat, tab_schema, tab_bench = st.tabs(["💬 Chat", "📋 Schema Browser", "📊 Benchmark"])


# ================================================================== #
# TAB 1 — CHAT
# ================================================================== #

with tab_chat:
    st.title("Ask your database anything")
    st.caption("Powered by GLM-4.7 via NVIDIA API · Schema-aware · Auto-correcting")

    # Render existing history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("Generated SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
                if msg.get("reasoning"):
                    with st.expander("Chain of Thought", expanded=False):
                        st.markdown(msg["reasoning"])
                if msg.get("attempts", 1) > 1:
                    st.caption(f"Required {msg['attempts']} attempt(s) to produce valid SQL")
            if msg.get("rows") is not None and msg.get("columns"):
                df = pd.DataFrame(msg["rows"], columns=msg["columns"])
                st.dataframe(df, use_container_width=True)

    # Input
    if prompt := st.chat_input("e.g. Which customers spent the most on completed orders?"):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt, "sql": None, "rows": None, "columns": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            pipeline = get_pipeline()

            reasoning_placeholder = st.empty()
            sql_placeholder = st.empty()
            status_placeholder = st.empty()
            status_placeholder.info("Generating SQL…")

            reasoning_accum = ""
            content_accum = ""
            final_result: PipelineResult | None = None

            for item in pipeline.stream_run(prompt):
                if isinstance(item, StreamChunk):
                    if item.reasoning_delta:
                        reasoning_accum += item.reasoning_delta
                        reasoning_placeholder.caption(
                            f"Thinking… {reasoning_accum[-300:]}"
                        )
                    if item.content_delta:
                        content_accum += item.content_delta
                        sql_placeholder.code(content_accum, language="sql")
                elif isinstance(item, PipelineResult):
                    final_result = item

            reasoning_placeholder.empty()
            sql_placeholder.empty()
            status_placeholder.empty()

            if final_result is None:
                st.error("No result received from pipeline.")
            elif final_result.error:
                st.error(f"Error after {final_result.attempts} attempt(s): {final_result.error}")
                with st.expander("Last SQL attempt"):
                    st.code(final_result.sql, language="sql")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I couldn't generate a valid query: {final_result.error}",
                    "sql": final_result.sql,
                    "rows": None,
                    "columns": None,
                    "reasoning": final_result.reasoning,
                    "attempts": final_result.attempts,
                })
            else:
                n = len(final_result.rows)
                response_text = f"Here are the results ({n} row{'s' if n != 1 else ''}):"
                st.markdown(response_text)

                with st.expander("Generated SQL", expanded=True):
                    st.code(final_result.sql, language="sql")
                if final_result.reasoning:
                    with st.expander("Chain of Thought", expanded=False):
                        st.markdown(final_result.reasoning)
                if final_result.attempts > 1:
                    st.caption(f"Required {final_result.attempts} attempt(s) to produce valid SQL")

                if final_result.rows:
                    df = pd.DataFrame(final_result.rows, columns=final_result.columns)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Query returned no rows.")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "sql": final_result.sql,
                    "rows": final_result.rows,
                    "columns": final_result.columns,
                    "reasoning": final_result.reasoning,
                    "attempts": final_result.attempts,
                })


# ================================================================== #
# TAB 2 — SCHEMA BROWSER
# ================================================================== #

with tab_schema:
    st.title("Database Schema")

    try:
        db = get_db()
        schema_dict = db.get_schema_dict()

        for table_name, info in schema_dict.items():
            with st.expander(f"📦 {table_name}  ({len(info['columns'])} columns)", expanded=False):
                col_data = [
                    {
                        "Column": c["name"],
                        "Type": c["type"],
                        "PK": "✓" if c["primary_key"] else "",
                        "Nullable": "✓" if c["nullable"] else "",
                    }
                    for c in info["columns"]
                ]
                st.dataframe(
                    pd.DataFrame(col_data),
                    use_container_width=True,
                    hide_index=True,
                )
                if info["foreign_keys"]:
                    st.markdown("**Foreign Keys**")
                    for fk in info["foreign_keys"]:
                        cols = ", ".join(fk["constrained_columns"])
                        ref_cols = ", ".join(fk["referred_columns"])
                        st.markdown(f"- `{cols}` → `{fk['referred_table']}({ref_cols})`")

        with st.expander("📄 Raw schema injected into prompts", expanded=False):
            st.code(db.get_schema_string(), language="sql")

    except Exception as e:
        st.error(f"Failed to load schema: {e}")
        st.info("Make sure you have run `python data/seed_db.py` to create the database.")


# ================================================================== #
# TAB 3 — BENCHMARK
# ================================================================== #

with tab_bench:
    st.title("Benchmark")
    st.markdown(
        "Upload a JSON file with `[{\"question\": ..., \"gold_sql\": ...}]` pairs "
        "or use the built-in 20-question sample dataset."
    )

    col_upload, col_builtin = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload benchmark JSON", type=["json"])
    with col_builtin:
        st.markdown("&nbsp;")
        use_sample = st.button("Use sample dataset (20 questions)")

    dataset = None

    if uploaded:
        try:
            dataset = json.load(uploaded)
            st.success(f"Loaded {len(dataset)} questions from uploaded file.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
    elif use_sample:
        try:
            dataset = load_benchmark_dataset("data/benchmark_sample.json")
            st.success(f"Loaded {len(dataset)} questions from sample dataset.")
        except Exception as e:
            st.error(f"Failed to load sample dataset: {e}")

    if dataset:
        st.divider()
        if st.button("▶ Run Benchmark", type="primary"):
            pipeline = get_pipeline()
            db = get_db()

            progress_bar = st.progress(0.0, text="Starting benchmark…")
            status_text = st.empty()

            def update_progress(current: int, total: int):
                progress_bar.progress(
                    current / total,
                    text=f"Question {current}/{total}: {dataset[current - 1]['question'][:70]}…",
                )

            with st.spinner("Running benchmark — this may take several minutes…"):
                report: BenchmarkReport = run_benchmark(
                    dataset=dataset,
                    pipeline=pipeline,
                    db=db,
                    progress_callback=update_progress,
                )

            progress_bar.empty()
            status_text.empty()

            # Summary metrics
            st.subheader("Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Exact Match", f"{report.em_score:.1%}")
            c2.metric("Execution Match", f"{report.exm_score:.1%}")
            c3.metric("Pass@1", f"{report.pass_at_1:.1%}")
            c4.metric("Pass@2", f"{report.pass_at_2:.1%}")
            c5.metric("Pass@3", f"{report.pass_at_3:.1%}")
            st.caption(
                f"Avg correction attempts per question: {report.avg_attempts:.2f} "
                f"(budget: {MAX_RETRIES})"
            )

            # Per-question table
            st.subheader("Per-question results")
            rows = [
                {
                    "#": i + 1,
                    "Question": s.question,
                    "EM": "✓" if s.exact_match else "✗",
                    "EXM": "✓" if s.execution_match else "✗",
                    "Attempts": s.attempts,
                    "Error": s.error or "",
                }
                for i, s in enumerate(report.samples)
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "EM": st.column_config.TextColumn("Exact Match", width="small"),
                    "EXM": st.column_config.TextColumn("Exec Match", width="small"),
                    "Attempts": st.column_config.NumberColumn(width="small"),
                },
            )

            # Detailed per-question breakdown
            st.subheader("Detailed breakdown")
            for i, s in enumerate(report.samples):
                icon = "✅" if s.execution_match else "❌"
                with st.expander(f"{icon} Q{i + 1}: {s.question[:70]}"):
                    col_gold, col_pred = st.columns(2)
                    with col_gold:
                        st.markdown("**Gold SQL**")
                        st.code(s.gold_sql, language="sql")
                    with col_pred:
                        st.markdown("**Predicted SQL**")
                        st.code(s.predicted_sql, language="sql")
                    if s.error:
                        st.error(s.error)
                    elif not s.execution_match:
                        st.warning("Execution results differ from gold.")
