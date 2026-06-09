import os
import re
import sqlite3
from typing import List, Optional, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ExecutionAccuracyMetric(BaseMetric):
    def __init__(
        self, db_root_path: str, threshold: float = 1.0, timeout: float = 30.0
    ):
        self.db_root_path = db_root_path
        self.threshold = threshold
        self.timeout = timeout
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.error = None

    @property
    def __name__(self):
        return "Execution Accuracy"

    def _resolve_db_path(self, db_id: str) -> str:
        candidate = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            return candidate
        flat = os.path.join(self.db_root_path, f"{db_id}.sqlite")
        if os.path.exists(flat):
            return flat
        raise FileNotFoundError(
            f"SQLite database not found for db_id '{db_id}'"
        )

    @staticmethod
    def _extract_sql(raw_output: str) -> str:
        if raw_output is None:
            return ""
        text = raw_output.strip()

        fenced = re.search(
            r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            text = fenced.group(1).strip()

        marker = re.search(r"(?i)\bSQL\s*:\s*", text)
        if marker:
            text = text[marker.end():].strip()

        statement_match = re.search(
            r"(?is)\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*",
            text,
        )
        if statement_match:
            text = statement_match.group(0).strip()

        if ";" in text:
            text = text.split(";")[0].strip()

        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _has_order_by(sql: str) -> bool:
        return re.search(r"(?i)\border\s+by\b", sql) is not None

    def _run_query(
        self, db_path: str, sql: str
    ) -> Tuple[bool, Optional[List[tuple]]]:
        if not sql:
            return False, None
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=self.timeout)
            conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return True, rows
        except Exception:
            return False, None
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _compare(
        predicted_rows: List[tuple],
        gold_rows: List[tuple],
        order_sensitive: bool,
    ) -> bool:
        if predicted_rows is None or gold_rows is None:
            return False
        if len(predicted_rows) != len(gold_rows):
            return False

        if order_sensitive:
            return predicted_rows == gold_rows

        def normalize(rows: List[tuple]):
            return sorted(tuple(str(value) for value in row) for row in rows)

        return normalize(predicted_rows) == normalize(gold_rows)

    def measure(self, test_case: LLMTestCase) -> float:
        db_id = None
        if test_case.metadata:
            db_id = test_case.metadata.get("db_id")
        if not db_id:
            self.score = 0.0
            self.success = False
            self.reason = "Missing db_id in test case metadata"
            return self.score

        predicted_sql = self._extract_sql(test_case.actual_output)
        gold_sql = test_case.expected_output.strip()

        try:
            db_path = self._resolve_db_path(db_id)
        except FileNotFoundError as exc:
            self.score = 0.0
            self.success = False
            self.error = str(exc)
            self.reason = str(exc)
            return self.score

        pred_ok, pred_rows = self._run_query(db_path, predicted_sql)
        gold_ok, gold_rows = self._run_query(db_path, gold_sql)

        if not gold_ok:
            self.score = 0.0
            self.success = False
            self.reason = f"Gold query failed to execute for db_id '{db_id}'"
            return self.score

        if not pred_ok:
            self.score = 0.0
            self.success = False
            self.reason = "Predicted query failed to execute"
            return self.score

        order_sensitive = self._has_order_by(gold_sql)
        match = self._compare(pred_rows, gold_rows, order_sensitive)

        self.score = 1.0 if match else 0.0
        self.success = self.score >= self.threshold
        self.reason = "Result sets match" if match else "Result sets differ"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
