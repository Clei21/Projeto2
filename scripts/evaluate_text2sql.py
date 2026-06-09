import argparse
import json
import os

from deepeval.test_case import LLMTestCase

from custom_metrics.execution_accuracy import ExecutionAccuracyMetric
from scripts.config import (
    SPIDER_DEV_FILE,
    SPIDER_TRAIN_FILE,
    SPIDER_DB_DIR,
    RESULTS_DIR,
    MAX_NEW_TOKENS_SQL,
    SEED,
    set_global_seed,
)
from scripts.model_utils import load_model, load_tokenizer, generate_batch
from scripts.spider_utils import (
    build_schema_lookup,
    build_few_shot_prompt,
    chunked,
    select_few_shot_examples,
    load_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    set_global_seed(SEED)

    schema_lookup = build_schema_lookup()
    train_examples = load_json(SPIDER_TRAIN_FILE)
    dev_examples = load_json(SPIDER_DEV_FILE)

    if args.limit is not None:
        dev_examples = dev_examples[: args.limit]

    few_shot = select_few_shot_examples(
        train_examples, schema_lookup, num_examples=3
    )

    tokenizer = load_tokenizer()
    model = load_model(
        adapter_path=args.adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    prompts = []
    for example in dev_examples:
        messages = build_few_shot_prompt(
            schema_lookup, few_shot, example["db_id"], example["question"]
        )
        prompts.append(messages)

    predictions = []
    for batch in chunked(prompts, args.batch_size):
        predictions.extend(
            generate_batch(model, tokenizer, batch, MAX_NEW_TOKENS_SQL)
        )

    metric = ExecutionAccuracyMetric(db_root_path=SPIDER_DB_DIR)

    records = []
    total = 0.0
    for example, raw_output in zip(dev_examples, predictions):
        test_case = LLMTestCase(
            input=example["question"],
            actual_output=raw_output,
            expected_output=example["query"],
            metadata={"db_id": example["db_id"]},
        )
        score = metric.measure(test_case)
        total += score
        records.append(
            {
                "db_id": example["db_id"],
                "question": example["question"],
                "gold_sql": example["query"],
                "raw_output": raw_output,
                "predicted_sql": metric._extract_sql(raw_output),
                "score": score,
                "reason": metric.reason,
            }
        )

    accuracy = total / len(records) if records else 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"text2sql_{args.run_name}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": args.run_name,
                "adapter_path": args.adapter_path,
                "num_examples": len(records),
                "execution_accuracy": accuracy,
                "records": records,
            },
            handle,
            indent=2,
        )

    print(f"Run: {args.run_name}")
    print(f"Execution Accuracy: {accuracy:.4f} ({int(total)}/{len(records)})")
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
