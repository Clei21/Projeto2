import argparse
import json
from pathlib import Path

import torch
from deepeval.test_case import LLMTestCase
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from custom_metrics import ExecutionAccuracy, extract_sql
from utils import find_spider_db, load_config, set_seed


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(base_model, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=160):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1600).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rows = load_jsonl("data/processed/dev.jsonl")[: args.limit or cfg["max_eval_samples"]]
    model, tokenizer = load_model(cfg["base_model"], args.adapter)
    metric = ExecutionAccuracy()

    outputs = []
    correct = 0
    for i, row in enumerate(rows, 1):
        raw = generate(model, tokenizer, row["prompt"])
        db_path = find_spider_db("data/spider", row["db_id"])
        case = LLMTestCase(input=row["question"], actual_output=raw, expected_output=row["query"], context=[db_path])
        score = metric.measure(case)
        correct += int(score)
        outputs.append({**row, "raw_output": raw, "predicted_sql": extract_sql(raw), "score": score, "reason": metric.reason})
        print(f"{i}/{len(rows)} score={score} acc={correct/i:.3f}")

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"spider_{args.name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print({"name": args.name, "accuracy": correct / len(rows), "n": len(rows), "file": str(out_path)})


if __name__ == "__main__":
    main()
