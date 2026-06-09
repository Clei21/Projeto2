import json
import os

import pytest

from custom_metrics.execution_accuracy import ExecutionAccuracyMetric
from scripts.config import RESULTS_DIR, SPIDER_DB_DIR


RUN_NAME = os.environ.get("EVAL_RUN_NAME", "finetuned_a")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.0"))


def load_run_records(run_name):
    path = os.path.join(RESULTS_DIR, f"text2sql_{run_name}.json")
    if not os.path.exists(path):
        pytest.skip(
            f"No results file for run '{run_name}'."
            " Run evaluate_text2sql.py first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_test_cases(run_name):
    from deepeval.test_case import LLMTestCase

    data = load_run_records(run_name)
    cases = []
    for record in data["records"]:
        cases.append(
            LLMTestCase(
                input=record["question"],
                actual_output=record["raw_output"],
                expected_output=record["gold_sql"],
                metadata={"db_id": record["db_id"]},
            )
        )
    return cases, data


def test_execution_accuracy_above_threshold():
    cases, data = build_test_cases(RUN_NAME)
    metric = ExecutionAccuracyMetric(db_root_path=SPIDER_DB_DIR)

    total = sum(metric.measure(case) for case in cases)
    accuracy = total / len(cases) if cases else 0.0

    assert accuracy >= MIN_ACCURACY, (
        f"Execution accuracy {accuracy:.4f} below threshold {MIN_ACCURACY:.4f}"
    )


def test_metric_extracts_sql_from_markdown():
    metric = ExecutionAccuracyMetric(db_root_path=SPIDER_DB_DIR)
    raw = "Here is the query:\n```sql\nSELECT name FROM students;\n```\nDone."
    assert metric._extract_sql(raw) == "SELECT name FROM students"


def test_metric_extracts_plain_sql():
    metric = ExecutionAccuracyMetric(db_root_path=SPIDER_DB_DIR)
    raw = "SQL: SELECT count(*) FROM singer"
    assert metric._extract_sql(raw) == "SELECT count(*) FROM singer"


def test_order_by_detection():
    metric = ExecutionAccuracyMetric(db_root_path=SPIDER_DB_DIR)
    assert metric._has_order_by("SELECT a FROM t ORDER BY a") is True
    assert metric._has_order_by("SELECT a FROM t") is False
