"""SQLite connection, schema introspection, and query execution."""

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect


class DatabaseManager:
    def __init__(self, db_path: str = "data/sample.db"):
        self.db_path = str(Path(db_path).resolve())
        self._sqlite_conn: sqlite3.Connection | None = None
        self._sa_engine = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def get_connection(self) -> sqlite3.Connection:
        if self._sqlite_conn is None:
            self._sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._sqlite_conn.row_factory = sqlite3.Row
            # Prevent any DML/DDL — hallucinated destructive queries become
            # catchable OperationalErrors that feed the correction loop.
            self._sqlite_conn.execute("PRAGMA query_only = ON")
        return self._sqlite_conn

    def _get_sa_engine(self):
        if self._sa_engine is None:
            self._sa_engine = create_engine(f"sqlite:///{self.db_path}")
        return self._sa_engine

    # ------------------------------------------------------------------ #
    # Schema introspection (SQLAlchemy inspect)
    # ------------------------------------------------------------------ #

    def get_schema_dict(self) -> dict[str, dict]:
        """
        Returns:
          {
            table_name: {
              "columns": [{"name", "type", "nullable", "primary_key"}, ...],
              "foreign_keys": [{"constrained_columns", "referred_table", "referred_columns"}, ...]
            }
          }
        """
        engine = self._get_sa_engine()
        insp = sa_inspect(engine)
        schema: dict[str, dict] = {}
        for table_name in insp.get_table_names():
            columns = []
            for col in insp.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "primary_key": col.get("primary_key", False),
                })
            fks = []
            for fk in insp.get_foreign_keys(table_name):
                fks.append({
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                })
            schema[table_name] = {"columns": columns, "foreign_keys": fks}
        return schema

    def get_schema_string(self) -> str:
        """Compact CREATE TABLE representation injected into LLM prompts."""
        schema = self.get_schema_dict()
        parts = []
        for table_name, info in schema.items():
            fk_map: dict[str, str] = {}
            for fk in info["foreign_keys"]:
                for cc, rc in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fk_map[cc] = f"{fk['referred_table']}({rc})"

            col_defs = []
            for col in info["columns"]:
                tokens = [col["name"], col["type"]]
                if col["primary_key"]:
                    tokens.append("PRIMARY KEY")
                if not col["nullable"] and not col["primary_key"]:
                    tokens.append("NOT NULL")
                if col["name"] in fk_map:
                    tokens.append(f"REFERENCES {fk_map[col['name']]}")
                col_defs.append("  " + " ".join(tokens))

            parts.append(
                f"CREATE TABLE {table_name} (\n"
                + ",\n".join(col_defs)
                + "\n);"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Query execution
    # ------------------------------------------------------------------ #

    def execute_query(self, sql: str) -> tuple[list[dict], list[str]]:
        """
        Execute a SELECT query. Returns (rows_as_list_of_dicts, column_names).
        Raises sqlite3.OperationalError on bad SQL (caught by the pipeline).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        return rows, col_names

    def close(self):
        if self._sqlite_conn:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        if self._sa_engine:
            self._sa_engine.dispose()
            self._sa_engine = None
