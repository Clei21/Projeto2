import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

try:
    from deepeval.metrics import BaseMetric
except Exception:
    class BaseMetric:
        pass


def extract_sql(text: str) -> str:
    if not text:
        return ""
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    sql = block.group(1) if block else text
    sql = re.sub(r"(?is)^.*?(select|with)\b", r"\1", sql.strip())
    sql = sql.strip().strip("` ")
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    return sql


def normalize_rows(rows: Iterable[tuple], ordered: bool) -> Any:
    normalized = [tuple(str(v).strip() if v is not None else "NULL" for v in row) for row in rows]
    return normalized if ordered else sorted(normalized)


class ExecutionAccuracy(BaseMetric):
    def __init__(self, database_path: str | Path | None = None, timeout: int = 10):
        self.database_path = str(database_path) if database_path else None
        self.timeout = timeout
        self.score = 0.0
        self.reason = ""
        self.name = "Execution Accuracy"

    def measure(self, test_case) -> float:
        db_path = getattr(test_case, "context", None) or self.database_path
        if isinstance(db_path, list):
            db_path = db_path[0] if db_path else self.database_path
        if not db_path:
            self.reason = "Database path not provided"
            self.score = 0.0
            return self.score

        predicted_sql = extract_sql(getattr(test_case, "actual_output", ""))
        gold_sql = extract_sql(getattr(test_case, "expected_output", ""))
        ordered = "order by" in gold_sql.lower()

        try:
            with sqlite3.connect(str(db_path), timeout=self.timeout) as conn:
                conn.execute("PRAGMA query_only = ON")
                pred_rows = conn.execute(predicted_sql).fetchall()
                gold_rows = conn.execute(gold_sql).fetchall()
            self.score = 1.0 if normalize_rows(pred_rows, ordered) == normalize_rows(gold_rows, ordered) else 0.0
            self.reason = "match" if self.score else "different result sets"
        except Exception as exc:
            self.score = 0.0
            self.reason = f"execution error: {type(exc).__name__}: {exc}"
        return self.score

    async def a_measure(self, test_case) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score == 1.0
