"""Prompt templates for the Text-to-SQL pipeline."""

SYSTEM_PROMPT_TEMPLATE = """You are an expert SQL assistant. Your job is to convert natural language questions into valid SQLite SQL queries.

## Database Schema
{schema}

## Rules
1. Output ONLY the raw SQL query — no markdown fences, no explanation, no commentary.
2. Use only table names and column names that exist in the schema above. Do not invent names.
3. Use standard SQLite syntax (STRFTIME for dates, no window functions unless essential).
4. When joining tables, always qualify ambiguous column names with the table name or alias.
5. If the question cannot be answered with the available schema, output exactly:
   SELECT 'Cannot answer: <reason>' AS error;

## Output Format
A single SQL statement ending with a semicolon.
"""

CORRECTION_PROMPT_TEMPLATE = """The SQL query you generated produced an error. Please fix it.

## Original Question
{question}

## Your Previous (Incorrect) SQL
{previous_sql}

## Error Message
{error_message}

## Database Schema (for reference)
{schema}

Output ONLY the corrected SQL query with no explanation or commentary.
"""


def build_system_prompt(schema_string: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(schema=schema_string)


def build_user_message(question: str) -> str:
    return f"Question: {question}"


def build_correction_message(
    question: str,
    previous_sql: str,
    error_message: str,
    schema_string: str,
) -> str:
    return CORRECTION_PROMPT_TEMPLATE.format(
        question=question,
        previous_sql=previous_sql,
        error_message=error_message,
        schema=schema_string,
    )
