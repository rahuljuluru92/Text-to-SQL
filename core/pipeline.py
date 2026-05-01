"""Text-to-SQL pipeline: generate → validate → execute → correct (≤3 retries)."""

import sqlite3
from collections.abc import Generator
from dataclasses import dataclass, field

from openai import OpenAI

from core.db import DatabaseManager
from core.llm import call_llm, stream_sql_generation, StreamChunk, get_client
from core.validator import validate_against_schema, clean_sql_output
from prompts.templates import (
    build_system_prompt,
    build_user_message,
    build_correction_message,
)

MAX_RETRIES = 3


@dataclass
class PipelineResult:
    sql: str = ""
    rows: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    reasoning: str = ""
    attempts: int = 1
    error: str | None = None
    validation_errors: list[str] = field(default_factory=list)


class TextToSQLPipeline:
    def __init__(self, db: DatabaseManager, client: OpenAI | None = None):
        self.db = db
        self.client = client or get_client()

    def _base_messages(self, question: str) -> list[dict]:
        schema = self.db.get_schema_string()
        return [
            {"role": "system", "content": build_system_prompt(schema)},
            {"role": "user", "content": build_user_message(question)},
        ]

    def _correction_messages(
        self, question: str, previous_sql: str, error_msg: str
    ) -> list[dict]:
        schema = self.db.get_schema_string()
        return [
            {"role": "system", "content": build_system_prompt(schema)},
            {"role": "user", "content": build_user_message(question)},
            {"role": "assistant", "content": previous_sql},
            {
                "role": "user",
                "content": build_correction_message(
                    question, previous_sql, error_msg, schema
                ),
            },
        ]

    def run(self, question: str) -> PipelineResult:
        """
        Synchronous pipeline used by the benchmark harness.
        Calls the LLM, validates, executes, and retries on failure.
        """
        schema_dict = self.db.get_schema_dict()
        result = PipelineResult()
        messages = self._base_messages(question)
        current_sql = ""

        for attempt in range(1, MAX_RETRIES + 1):
            result.attempts = attempt
            llm_resp = call_llm(messages, client=self.client)
            current_sql = clean_sql_output(llm_resp.content)
            result.reasoning = llm_resp.reasoning

            # Schema-level validation (table name check via sqlglot AST)
            val = validate_against_schema(current_sql, schema_dict)
            if not val.is_valid:
                error_msg = "; ".join(val.errors)
                result.validation_errors = val.errors
                if attempt < MAX_RETRIES:
                    messages = self._correction_messages(question, current_sql, error_msg)
                    continue
                result.sql = current_sql
                result.error = f"Schema validation failed after {attempt} attempt(s): {error_msg}"
                return result

            # Execution (catches runtime errors and PRAGMA query_only violations)
            try:
                rows, cols = self.db.execute_query(current_sql)
                result.sql = current_sql
                result.rows = rows
                result.columns = cols
                return result
            except sqlite3.OperationalError as e:
                error_msg = str(e)
                if attempt < MAX_RETRIES:
                    messages = self._correction_messages(question, current_sql, error_msg)
                    continue
                result.sql = current_sql
                result.error = f"Execution failed after {attempt} attempt(s): {error_msg}"
                return result

        result.sql = current_sql
        result.error = "Max retries exceeded"
        return result

    def stream_run(self, question: str) -> Generator:
        """
        Streaming pipeline used by the Streamlit chat tab.

        Yields StreamChunk objects during the first LLM call (for real-time
        display), then yields a final PipelineResult.

        Caller pattern:
            for item in pipeline.stream_run(question):
                if isinstance(item, StreamChunk):
                    # update live UI
                elif isinstance(item, PipelineResult):
                    # render final SQL + result table
        """
        schema_dict = self.db.get_schema_dict()
        messages = self._base_messages(question)

        # First attempt — stream tokens to the UI
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        for chunk in stream_sql_generation(messages, client=self.client):
            yield chunk
            if chunk.content_delta:
                content_parts.append(chunk.content_delta)
            if chunk.reasoning_delta:
                reasoning_parts.append(chunk.reasoning_delta)

        current_sql = clean_sql_output("".join(content_parts))
        reasoning = "".join(reasoning_parts)

        # Validate + execute (silent correction loop for retries 2 and 3)
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                llm_resp = call_llm(messages, client=self.client)
                current_sql = clean_sql_output(llm_resp.content)
                reasoning = llm_resp.reasoning

            val = validate_against_schema(current_sql, schema_dict)
            if not val.is_valid:
                error_msg = "; ".join(val.errors)
                if attempt < MAX_RETRIES:
                    messages = self._correction_messages(question, current_sql, error_msg)
                    continue
                yield PipelineResult(
                    sql=current_sql,
                    reasoning=reasoning,
                    attempts=attempt,
                    error=f"Schema validation failed: {error_msg}",
                    validation_errors=val.errors,
                )
                return

            try:
                rows, cols = self.db.execute_query(current_sql)
                yield PipelineResult(
                    sql=current_sql,
                    rows=rows,
                    columns=cols,
                    reasoning=reasoning,
                    attempts=attempt,
                )
                return
            except sqlite3.OperationalError as e:
                error_msg = str(e)
                if attempt < MAX_RETRIES:
                    messages = self._correction_messages(question, current_sql, error_msg)
                    continue
                yield PipelineResult(
                    sql=current_sql,
                    reasoning=reasoning,
                    attempts=attempt,
                    error=f"Execution error: {error_msg}",
                )
                return
