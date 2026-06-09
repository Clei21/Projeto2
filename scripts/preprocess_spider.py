import argparse
import json
import os

from scripts.config import (
    SPIDER_TRAIN_FILE,
    DATA_DIR,
    SEED,
    set_global_seed,
)
from scripts.spider_utils import (
    build_schema_lookup,
    build_training_messages,
    load_json,
)


def main():
    parser = argparse.ArgumentParser()
    default_out = os.path.join(DATA_DIR, "train_chat.jsonl")
    parser.add_argument("--output", default=default_out)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    set_global_seed(SEED)

    schema_lookup = build_schema_lookup()
    train_examples = load_json(SPIDER_TRAIN_FILE)

    if args.max_examples is not None:
        train_examples = train_examples[: args.max_examples]

    written = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for example in train_examples:
            if example["db_id"] not in schema_lookup:
                continue
            messages = build_training_messages(schema_lookup, example)
            handle.write(json.dumps({"messages": messages}) + "\n")
            written += 1

    print(f"Wrote {written} training examples to {args.output}")


if __name__ == "__main__":
    main()
