"""Consolida os resultados das Fases 2, 4 e 5 em uma única tabela de
trade-off, calculando a variação percentual por categoria do MMLU.

Uso:
    python scripts/analyze_tradeoff.py \
        --spider_baseline results/eval_predictions_baseline.json \
        --spider_finetuned results/eval_predictions_lora_a.json \
        --mmlu_baseline results/mmlu_baseline.json \
        --mmlu_finetuned results/mmlu_lora_a.json \
        --out results/tradeoff_lora_a.json
"""

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delta_pct(base: float, tuned: float) -> float:
    if base == 0:
        return float("inf") if tuned > 0 else 0.0
    return (tuned - base) / base * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_baseline", required=True)
    parser.add_argument("--spider_finetuned", required=True)
    parser.add_argument("--mmlu_baseline", required=True)
    parser.add_argument("--mmlu_finetuned", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spider_base = load(args.spider_baseline)["execution_accuracy"]
    spider_ft = load(args.spider_finetuned)["execution_accuracy"]
    mmlu_base = load(args.mmlu_baseline)
    mmlu_ft = load(args.mmlu_finetuned)

    summary = {
        "spider": {
            "baseline": spider_base,
            "finetuned": spider_ft,
            "delta_pct": delta_pct(spider_base, spider_ft),
        },
        "mmlu_overall": {
            "baseline": mmlu_base["overall_accuracy"],
            "finetuned": mmlu_ft["overall_accuracy"],
            "delta_pct": delta_pct(mmlu_base["overall_accuracy"], mmlu_ft["overall_accuracy"]),
        },
        "mmlu_by_category": {},
    }

    for category in mmlu_base["categories"]:
        base = mmlu_base["categories"][category]["accuracy"]
        tuned = mmlu_ft["categories"][category]["accuracy"]
        summary["mmlu_by_category"][category] = {
            "baseline": base, "finetuned": tuned, "delta_pct": delta_pct(base, tuned),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
