import glob
import json
from pathlib import Path

import pandas as pd

rows = []
for path in glob.glob("results/spider_*.jsonl"):
    name = Path(path).stem.replace("spider_", "")
    data = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows.append({"model": name, "metric": "spider_execution_accuracy", "value": sum(x["score"] for x in data) / len(data), "n": len(data)})

for path in glob.glob("results/mmlu_*.json"):
    name = Path(path).stem.replace("mmlu_", "")
    data = json.load(open(path, encoding="utf-8"))["summary"]
    for category, value in data.items():
        rows.append({"model": name, "metric": f"mmlu_{category}", "value": value, "n": 150 if category == "overall" else 50})

out = pd.DataFrame(rows).sort_values(["metric", "model"])
out.to_csv("results/summary.csv", index=False)
print(out)
