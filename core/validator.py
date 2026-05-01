"""Schema-level SQL validation using sqlglot AST parsing."""

import re
from dataclasses import dataclass, field

import sqlglot
import sqlglot.errors
import sqlglot.expressions as exp


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)


def clean_sql_output(raw_output: str) -> str:
    """
    Strip markdown fences and whitespace from LLM output.
    GLM-4.7 sometimes wraps output in ```sql ... ``` despite instructions.
    """
    cleaned = re.sub(r"^```(?:sql)?\s*\n?", "", raw_output.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())
    return cleaned.strip()


def validate_against_schema(sql: str, schema_dict: dict[str, dict]) -> ValidationResult:
    """
    Parse the SQL with sqlglot (SQLite dialect) and verify every physical
    table reference exists in schema_dict.

    sqlglot's AST traversal correctly handles:
      - CTEs (WITH clauses)
      - Subqueries
      - Table aliases
      - Nested functions

    A ParseError during parsing is itself returned as a validation failure,
    which triggers the correction loop without needing a live DB round-trip.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return ValidationResult(is_valid=False, errors=[f"SQL syntax error: {e}"])

    known_tables = {t.lower() for t in schema_dict.keys()}
    errors: list[str] = []

    for table_node in tree.find_all(exp.Table):
        # Skip subquery aliases and CTE names (they have no db/catalog)
        table_name = table_node.name.lower()
        if not table_name:
            continue
        # CTE names appear as Table nodes too; skip them if they match a CTE alias
        cte_names = {
            cte.alias.lower()
            for cte in tree.find_all(exp.CTE)
            if cte.alias
        }
        if table_name in cte_names:
            continue
        if table_name not in known_tables:
            errors.append(
                f"Table '{table_name}' does not exist. "
                f"Available: {sorted(known_tables)}"
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
