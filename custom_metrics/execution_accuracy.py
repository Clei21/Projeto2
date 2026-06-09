"""Métrica customizada de Execution Accuracy para a tarefa Text-to-SQL (Spider).

A comparação segue a semântica do avaliador oficial do Spider: os conjuntos
de resultados são comparados de forma insensível à ordem das linhas, exceto
quando a consulta de referência contém ORDER BY.
"""

import re
import sqlite3
from collections import Counter
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_SELECT_STMT = re.compile(r"\b(SELECT\b.+?)(?:;|$)", re.DOTALL | re.IGNORECASE)
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def extract_sql(raw_output: str) -> str:
    """Extrai a consulta SQL da saída bruta do modelo.

    Prioriza blocos markdown ```sql ...```; na ausência deles, captura a
    primeira instrução SELECT encontrada no texto.
    """
    block = _SQL_BLOCK.search(raw_output)
    candidate = block.group(1) if block else raw_output

    stmt = _SELECT_STMT.search(candidate)
    if stmt is None:
        return ""

    sql = stmt.group(1).strip()
    return re.sub(r"\s+", " ", sql)


def _normalize_row(row) -> tuple:
    return tuple(
        round(v, 6) if isinstance(v, float) else v
        for v in row
    )


class ExecutionAccuracyMetric(BaseMetric):
    """Avalia se a consulta gerada produz o mesmo resultado da consulta gold.

    Cada LLMTestCase deve carregar o identificador do banco em
    ``additional_metadata={"db_id": ...}``. O atributo ``db_root`` aponta
    para o diretório ``database/`` do Spider, que contém um subdiretório
    por banco com o arquivo SQLite correspondente.
    """

    def __init__(self, db_root: str, threshold: float = 1.0, timeout_s: float = 30.0):
        self.db_root = Path(db_root)
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "Execution Accuracy"

    def measure(self, test_case: LLMTestCase) -> float:
        db_id = (test_case.additional_metadata or {}).get("db_id")
        if not db_id:
            return self._fail("test case sem db_id em additional_metadata")

        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return self._fail(f"banco não encontrado: {db_path}")

        predicted_sql = extract_sql(test_case.actual_output or "")
        if not predicted_sql:
            return self._fail("nenhuma consulta SELECT encontrada na saída do modelo")

        gold_sql = test_case.expected_output

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=self.timeout_s)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        try:
            try:
                predicted_rows = conn.execute(predicted_sql).fetchall()
            except sqlite3.Error as e:
                return self._fail(f"erro ao executar consulta gerada: {e}")

            try:
                gold_rows = conn.execute(gold_sql).fetchall()
            except sqlite3.Error as e:
                return self._fail(f"erro ao executar consulta gold: {e}")
        finally:
            conn.close()

        predicted = [_normalize_row(r) for r in predicted_rows]
        gold = [_normalize_row(r) for r in gold_rows]

        if _ORDER_BY.search(gold_sql):
            match = predicted == gold
        else:
            match = Counter(predicted) == Counter(gold)

        self.score = 1.0 if match else 0.0
        self.success = match
        self.reason = "resultados idênticos" if match else "resultados divergentes"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    def _fail(self, reason: str) -> float:
        self.score = 0.0
        self.success = False
        self.reason = reason
        return self.score
