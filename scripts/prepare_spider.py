"""Pré-processamento do Spider.

Gera três artefatos a partir do diretório oficial do dataset:
  - data/train.jsonl: training split no formato de chat (campo "messages")
  - data/dev.jsonl: dev split com schema, pergunta e SQL gold
  - data/few_shot.json: 3 exemplos fixos do training split para o prompt

Uso:
    python scripts/prepare_spider.py --spider_dir spider --out_dir data
"""

import argparse
import json
import random
from pathlib import Path

from spider_common import (
    SEED,
    SYSTEM_PROMPT,
    build_user_message,
    load_schemas,
    save_jsonl,
)


def to_chat_example(item: dict, schemas: dict) -> dict:
    schema = schemas[item["db_id"]]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(schema, item["question"])},
            {"role": "assistant", "content": f"```sql\n{item['query']}\n```"},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", required=True)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    spider = Path(args.spider_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemas = load_schemas(spider / "tables.json")

    with open(spider / "train_spider.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(spider / "dev.json", encoding="utf-8") as f:
        dev = json.load(f)

    rng = random.Random(SEED)
    few_shot_items = rng.sample(train, 3)
    few_shot = [
        {
            "db_id": ex["db_id"],
            "schema": schemas[ex["db_id"]],
            "question": ex["question"],
            "query": ex["query"],
        }
        for ex in few_shot_items
    ]
    with open(out / "few_shot.json", "w", encoding="utf-8") as f:
        json.dump(few_shot, f, ensure_ascii=False, indent=2)

    save_jsonl([to_chat_example(ex, schemas) for ex in train], out / "train.jsonl")

    dev_records = [
        {
            "db_id": ex["db_id"],
            "schema": schemas[ex["db_id"]],
            "question": ex["question"],
            "gold_sql": ex["query"],
        }
        for ex in dev
    ]
    save_jsonl(dev_records, out / "dev.jsonl")

    print(f"train: {len(train)} exemplos | dev: {len(dev_records)} exemplos")
    print(f"few-shot fixo: {[ex['db_id'] for ex in few_shot]}")


if __name__ == "__main__":
    main()
