import json
from typing import Dict, List, Optional

from scripts.config import SPIDER_TABLES_FILE


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_schema_lookup(
    tables_file: str = SPIDER_TABLES_FILE,
) -> Dict[str, str]:
    tables = load_json(tables_file)
    lookup = {}
    for entry in tables:
        db_id = entry["db_id"]
        table_names = entry["table_names_original"]
        columns = entry["column_names_original"]
        column_types = entry["column_types"]
        primary_keys = set(entry.get("primary_keys", []))
        foreign_keys = entry.get("foreign_keys", [])

        per_table_cols = {name: [] for name in table_names}
        for idx, (table_idx, column_name) in enumerate(columns):
            if table_idx == -1:
                continue
            col_type = column_types[idx]
            tag = " (PK)" if idx in primary_keys else ""
            per_table_cols[table_names[table_idx]].append(
                f"{column_name} {col_type}{tag}"
            )

        lines = []
        for table_name in table_names:
            cols = ", ".join(per_table_cols[table_name])
            lines.append(f"Table {table_name} ({cols})")

        for source_col, target_col in foreign_keys:
            s_table = table_names[columns[source_col][0]]
            s_col = columns[source_col][1]
            t_table = table_names[columns[target_col][0]]
            t_col = columns[target_col][1]
            lines.append(
                f"Foreign Key: {s_table}.{s_col} references {t_table}.{t_col}"
            )

        lookup[db_id] = "\n".join(lines)
    return lookup


def select_few_shot_examples(
    train_examples: List[dict],
    schema_lookup: Dict[str, str],
    num_examples: int = 3,
) -> List[dict]:
    chosen = []
    seen_dbs = set()
    for example in train_examples:
        db_id = example["db_id"]
        if db_id in seen_dbs:
            continue
        if db_id not in schema_lookup:
            continue
        seen_dbs.add(db_id)
        chosen.append(example)
        if len(chosen) == num_examples:
            break
    return chosen


def format_example_block(
    schema: str, question: str, sql: Optional[str] = None
) -> str:
    block = f"Database schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    if sql is not None:
        block += f" {sql}"
    return block


SYSTEM_INSTRUCTION = (
    "You are an expert data analyst. Given a database schema and a natural "
    "language question, write a single valid SQLite query that answers it. "
    "Return only the SQL query without any explanation."
)


def build_few_shot_prompt(
    schema_lookup: Dict[str, str],
    few_shot_examples: List[dict],
    target_db_id: str,
    target_question: str,
) -> List[dict]:
    parts = []
    for example in few_shot_examples:
        schema = schema_lookup[example["db_id"]]
        parts.append(
            format_example_block(schema, example["question"], example["query"])
        )

    target_schema = schema_lookup[target_db_id]
    parts.append(format_example_block(target_schema, target_question, None))

    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build_training_messages(
    schema_lookup: Dict[str, str],
    example: dict,
) -> List[dict]:
    schema = schema_lookup[example["db_id"]]
    user_content = format_example_block(schema, example["question"], None)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["query"]},
    ]
