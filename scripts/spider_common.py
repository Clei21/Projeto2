"""Funções compartilhadas: serialização de esquema, prompt few-shot e seeds."""

import json
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a question, "
    "write a single SQLite query that answers the question. "
    "Respond with the SQL query only, inside a ```sql code block."
)


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_schemas(tables_json: str) -> dict:
    """Carrega tables.json do Spider e retorna {db_id: esquema serializado}."""
    with open(tables_json, encoding="utf-8") as f:
        tables = json.load(f)
    return {db["db_id"]: serialize_schema(db) for db in tables}


def serialize_schema(db: dict) -> str:
    """Serializa o esquema de um banco em texto compacto.

    Formato: uma linha por tabela com colunas e tipos, seguida das chaves
    primárias e estrangeiras, suficiente para o schema linking.
    """
    table_names = db["table_names_original"]
    columns = db["column_names_original"]
    types = db["column_types"]

    cols_by_table = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx >= 0:
            cols_by_table[tbl_idx].append(f"{col_name} {types[col_idx]}")

    lines = []
    for tbl_idx, name in enumerate(table_names):
        cols = ", ".join(cols_by_table[tbl_idx])
        lines.append(f"Table {name} ({cols})")

    pks = [columns[i] for i in db.get("primary_keys", []) if isinstance(i, int)]
    for tbl_idx, col_name in pks:
        lines.append(f"Primary key: {table_names[tbl_idx]}.{col_name}")

    for src, dst in db.get("foreign_keys", []):
        s_tbl, s_col = columns[src]
        d_tbl, d_col = columns[dst]
        lines.append(
            f"Foreign key: {table_names[s_tbl]}.{s_col} -> {table_names[d_tbl]}.{d_col}"
        )
    return "\n".join(lines)


def build_user_message(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"


def build_few_shot_messages(few_shot_examples: list[dict]) -> list[dict]:
    """Converte os 3 exemplos fixos em turnos user/assistant do chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot_examples:
        messages.append(
            {"role": "user", "content": build_user_message(ex["schema"], ex["question"])}
        )
        messages.append(
            {"role": "assistant", "content": f"```sql\n{ex['query']}\n```"}
        )
    return messages


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
