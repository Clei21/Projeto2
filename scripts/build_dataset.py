import argparse
from pathlib import Path

from utils import read_json, write_jsonl


def schema_text(tables: dict, db_id: str) -> str:
    idx = tables["db_id"].index(db_id)
    table_names = tables["table_names_original"][idx]
    column_names = tables["column_names_original"][idx]
    table_cols = {name: [] for name in table_names}
    for table_idx, col_name in column_names:
        if table_idx >= 0:
            table_cols[table_names[table_idx]].append(col_name)
    return "\n".join(f"Table {table}: {', '.join(cols)}" for table, cols in table_cols.items())


def make_prompt(schema: str, question: str) -> str:
    return f"You are a Text-to-SQL assistant. Generate only the SQL query.\n\nDatabase schema:\n{schema}\n\nQuestion: {question}\nSQL:"


def convert(split_path: Path, tables: dict, out_path: Path, limit: int | None):
    rows = read_json(split_path)
    if limit:
        rows = rows[:limit]
    converted = []
    for item in rows:
        schema = schema_text(tables, item["db_id"])
        converted.append({
            "db_id": item["db_id"],
            "question": item["question"],
            "query": item["query"],
            "text": make_prompt(schema, item["question"]) + " " + item["query"],
            "prompt": make_prompt(schema, item["question"]),
        })
    write_jsonl(out_path, converted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", default="data/spider")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    spider = Path(args.spider_dir)
    tables = read_json(spider / "tables.json")
    out = Path(args.out_dir)
    convert(spider / "train_spider.json", tables, out / "train.jsonl", args.max_train_samples)
    convert(spider / "dev.json", tables, out / "dev.jsonl", None)
    print(f"Wrote files to {out}")


if __name__ == "__main__":
    main()
