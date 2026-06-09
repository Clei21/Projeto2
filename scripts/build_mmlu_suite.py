"""Construção da suíte de avaliação MMLU (Fase 5).

Amostra 50 questões de cada uma das três subcategorias com seed fixa e
anexa os 5 exemplos do dev split de cada disciplina, que serão usados
como contexto 5-shot idêntico para todos os modelos.

Uso:
    python scripts/build_mmlu_suite.py --out data/mmlu_suite.json
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from spider_common import SEED

SUBJECTS = {
    "STEM": "college_computer_science",
    "Humanidades": "philosophy",
    "Ciências Sociais": "high_school_macroeconomics",
}
QUESTIONS_PER_CATEGORY = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mmlu_suite.json")
    args = parser.parse_args()

    rng = random.Random(SEED)
    suite = {"seed": SEED, "categories": {}}

    for category, subject in SUBJECTS.items():
        test = load_dataset("cais/mmlu", subject, split="test")
        dev = load_dataset("cais/mmlu", subject, split="dev")

        indices = rng.sample(range(len(test)), QUESTIONS_PER_CATEGORY)
        questions = [
            {
                "question": test[i]["question"],
                "choices": test[i]["choices"],
                "answer": int(test[i]["answer"]),
            }
            for i in sorted(indices)
        ]
        few_shot = [
            {
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": int(ex["answer"]),
            }
            for ex in dev
        ]
        suite["categories"][category] = {
            "subject": subject,
            "few_shot": few_shot,
            "questions": questions,
        }
        print(f"{category} ({subject}): {len(questions)} questões, {len(few_shot)} exemplos few-shot")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    print(f"Suíte salva em {args.out}")


if __name__ == "__main__":
    main()
