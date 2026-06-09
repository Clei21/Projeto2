import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_spider_db(spider_dir: str | Path, db_id: str) -> str:
    base = Path(spider_dir)
    candidates = [
        base / "database" / db_id / f"{db_id}.sqlite",
        base / "database" / db_id / f"{db_id}.db",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"SQLite database not found for {db_id}")
