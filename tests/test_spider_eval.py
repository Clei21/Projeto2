"""Teste pytest que aplica a métrica de Execution Accuracy a um arquivo de
predições e consolida a acurácia agregada.

Uso:
    PREDICTIONS=results/predictions_lora_a.jsonl \
    SPIDER_DB_DIR=spider/database \
    pytest tests/test_spider_eval.py -s
"""

import json
import os
import sys
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_metrics.execution_accuracy import ExecutionAccuracyMetric
from scripts.spider_common import load_jsonl

PREDICTIONS = os.environ.get("PREDICTIONS", "results/predictions_baseline.jsonl")
SPIDER_DB_DIR = os.environ.get("SPIDER_DB_DIR", "spider/database")


@pytest.fixture(scope="module")
def predictions():
    if not Path(PREDICTIONS).exists():
        pytest.skip(f"arquivo de predições não encontrado: {PREDICTIONS}")
    return load_jsonl(PREDICTIONS)


def test_execution_accuracy(predictions):
    metric = ExecutionAccuracyMetric(db_root=SPIDER_DB_DIR)

    details = []
    for item in predictions:
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["raw_output"],
            expected_output=item["gold_sql"],
            additional_metadata={"db_id": item["db_id"]},
        )
        score = metric.measure(test_case)
        details.append(
            {
                "db_id": item["db_id"],
                "question": item["question"],
                "score": score,
                "reason": metric.reason,
            }
        )

    accuracy = sum(d["score"] for d in details) / len(details)

    report_path = Path("results") / f"eval_{Path(PREDICTIONS).stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"predictions_file": PREDICTIONS, "n": len(details),
             "execution_accuracy": accuracy, "details": details},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nExecution Accuracy: {accuracy:.4f} ({len(details)} exemplos)")
    print(f"Relatório detalhado: {report_path}")

    assert len(details) > 0
