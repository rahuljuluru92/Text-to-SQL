"""
Benchmark metrics for Text-to-SQL evaluation.

Metrics:
  - Exact Match (EM): normalized string equality between gold and predicted SQL.
  - Execution Match (EXM): result-set equality (order-insensitive, value-coerced).
  - Pass@k: fraction of questions answered correctly within k correction attempts.

Dataset format (JSON):
  [{"question": "...", "gold_sql": "..."}]
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from core.db import DatabaseManager
from core.pipeline import TextToSQLPipeline, PipelineResult


# ------------------------------------------------------------------ #
# Normalization
# ------------------------------------------------------------------ #

def normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql.rstrip(";").strip()


# ------------------------------------------------------------------ #
# Execution Match helpers
# ------------------------------------------------------------------ #

def _run_for_comparison(db: DatabaseManager, sql: str) -> list[tuple] | None:
    """
    Execute SQL and return sorted list of string-coerced row tuples.
    Returns None if execution fails (error treated as incorrect).
    Sorting makes comparison order-insensitive.
    """
    try:
        rows, cols = db.execute_query(sql)
        normalized = [tuple(str(v) for v in row.values()) for row in rows]
        return sorted(normalized)
    except sqlite3.OperationalError:
        return None


def execution_match(gold_sql: str, pred_sql: str, db: DatabaseManager) -> bool:
    gold = _run_for_comparison(db, gold_sql)
    pred = _run_for_comparison(db, pred_sql)
    if gold is None or pred is None:
        return False
    return gold == pred


# ------------------------------------------------------------------ #
# Pass@k
# ------------------------------------------------------------------ #

def pass_at_k(samples: list["BenchmarkSample"], k: int) -> float:
    """
    Fraction of questions answered correctly (EXM=True) within k attempts.
    A question counts as Pass@k if it has no error AND attempts <= k.
    """
    if not samples:
        return 0.0
    passed = sum(
        1 for s in samples
        if s.execution_match and s.error is None and s.attempts <= k
    )
    return passed / len(samples)


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class BenchmarkSample:
    question: str
    gold_sql: str
    predicted_sql: str = ""
    exact_match: bool = False
    execution_match: bool = False
    attempts: int = 1
    error: str | None = None


@dataclass
class BenchmarkReport:
    total: int = 0
    em_score: float = 0.0
    exm_score: float = 0.0
    pass_at_1: float = 0.0
    pass_at_2: float = 0.0
    pass_at_3: float = 0.0
    avg_attempts: float = 1.0
    samples: list[BenchmarkSample] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Dataset loading
# ------------------------------------------------------------------ #

def load_benchmark_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark dataset must be a JSON array")
    for item in data:
        if "question" not in item or "gold_sql" not in item:
            raise ValueError("Each item must have 'question' and 'gold_sql' keys")
    return data


# ------------------------------------------------------------------ #
# Main benchmark runner
# ------------------------------------------------------------------ #

def run_benchmark(
    dataset: list[dict],
    pipeline: TextToSQLPipeline,
    db: DatabaseManager,
    progress_callback=None,
) -> BenchmarkReport:
    """
    Run the pipeline over every question in the dataset.

    progress_callback(current: int, total: int) is called after each question
    if provided — used to drive a Streamlit progress bar.
    """
    report = BenchmarkReport(total=len(dataset))
    total_attempts = 0
    em_count = 0
    exm_count = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        gold_sql = item["gold_sql"]

        result: PipelineResult = pipeline.run(question)
        predicted_sql = result.sql

        em = normalize_sql(gold_sql) == normalize_sql(predicted_sql)
        exm = execution_match(gold_sql, predicted_sql, db)

        if em:
            em_count += 1
        if exm:
            exm_count += 1
        total_attempts += result.attempts

        report.samples.append(
            BenchmarkSample(
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                exact_match=em,
                execution_match=exm,
                attempts=result.attempts,
                error=result.error,
            )
        )

        if progress_callback:
            progress_callback(i + 1, len(dataset))

    n = len(dataset)
    report.em_score = em_count / n if n else 0.0
    report.exm_score = exm_count / n if n else 0.0
    report.pass_at_1 = pass_at_k(report.samples, k=1)
    report.pass_at_2 = pass_at_k(report.samples, k=2)
    report.pass_at_3 = pass_at_k(report.samples, k=3)
    report.avg_attempts = total_attempts / n if n else 1.0
    return report
