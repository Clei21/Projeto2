import argparse
import json
import os
import re

from datasets import load_dataset

from scripts.config import (
    MMLU_SUBJECTS,
    MMLU_QUESTIONS_PER_CATEGORY,
    MMLU_NUM_SHOTS,
    MAX_NEW_TOKENS_MMLU,
    RESULTS_DIR,
    SEED,
    set_global_seed,
)
from scripts.model_utils import load_model, load_tokenizer, generate_batch
from scripts.spider_utils import chunked


LETTERS = ["A", "B", "C", "D"]


def format_question(item, include_answer: bool) -> str:
    choices = item["choices"]
    block = item["question"].strip() + "\n"
    for letter, choice in zip(LETTERS, choices):
        block += f"{letter}. {choice}\n"
    block += "Answer:"
    if include_answer:
        block += f" {LETTERS[item['answer']]}"
    return block


def build_mmlu_messages(few_shot_items, question_item):
    shots = "\n\n".join(format_question(item, True) for item in few_shot_items)
    target = format_question(question_item, False)
    user_content = (
        "The following are multiple choice questions. "
        "Answer with a single letter (A, B, C or D).\n\n"
        f"{shots}\n\n{target}"
    )
    return [{"role": "user", "content": user_content}]


def parse_answer(text: str) -> str:
    text = text.strip().upper()
    if text and text[0] in "ABCD":
        return text[0]
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)

    tokenizer = load_tokenizer()
    model = load_model(
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    per_category = {}
    all_records = []

    for category, subject in MMLU_SUBJECTS.items():
        dev_split = load_dataset("cais/mmlu", subject, split="dev")
        test_split = load_dataset("cais/mmlu", subject, split="test")

        few_shot_items = [dev_split[i] for i in range(MMLU_NUM_SHOTS)]
        questions = [test_split[i] for i in range(MMLU_QUESTIONS_PER_CATEGORY)]

        prompts = [build_mmlu_messages(few_shot_items, q) for q in questions]

        predictions = []
        for batch in chunked(prompts, args.batch_size):
            predictions.extend(
                generate_batch(model, tokenizer, batch, MAX_NEW_TOKENS_MMLU)
            )

        correct = 0
        for question_item, raw_output in zip(questions, predictions):
            predicted = parse_answer(raw_output)
            gold = LETTERS[question_item["answer"]]
            is_correct = predicted == gold
            correct += int(is_correct)
            all_records.append(
                {
                    "category": category,
                    "subject": subject,
                    "predicted": predicted,
                    "gold": gold,
                    "correct": is_correct,
                }
            )

        per_category[category] = correct / MMLU_QUESTIONS_PER_CATEGORY

    overall = sum(r["correct"] for r in all_records) / len(all_records)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"mmlu_{args.run_name}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": args.run_name,
                "adapter_path": args.adapter_path,
                "overall_accuracy": overall,
                "per_category_accuracy": per_category,
                "records": all_records,
            },
            handle,
            indent=2,
        )

    print(f"Run: {args.run_name}")
    print(f"MMLU Overall Accuracy: {overall:.4f}")
    for category, acc in per_category.items():
        print(f"  {category}: {acc:.4f}")
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
