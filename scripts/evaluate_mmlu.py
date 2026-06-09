import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import load_config, set_seed

LETTERS = ["A", "B", "C", "D"]


def format_question(item, include_answer=True):
    choices = item["choices"]
    text = f"Question: {item['question']}\n"
    text += "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    text += "\nAnswer:"
    if include_answer:
        text += f" {LETTERS[int(item['answer'])]}"
    return text


def build_prompt(dev_examples, item):
    shots = "\n\n".join(format_question(ex, include_answer=True) for ex in dev_examples)
    return f"Answer the multiple choice questions with only A, B, C, or D.\n\n{shots}\n\n{format_question(item, include_answer=False)}"


def parse_answer(text):
    cleaned = text.strip().upper()
    for char in cleaned:
        if char in LETTERS:
            return char
    return ""


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


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    model, tokenizer = load_model(cfg["base_model"], args.adapter)

    results = []
    summary = {}
    for group, subject in cfg["mmlu"]["categories"].items():
        dev = load_dataset("cais/mmlu", subject, split="dev")
        test = load_dataset("cais/mmlu", subject, split="test")
        shots = [dev[i] for i in range(5)]
        sample = [test[i] for i in range(cfg["mmlu"]["samples_per_category"])]
        correct = 0
        for item in sample:
            raw = generate(model, tokenizer, build_prompt(shots, item))
            pred = parse_answer(raw)
            gold = LETTERS[int(item["answer"])]
            ok = pred == gold
            correct += int(ok)
            results.append({"category": group, "subject": subject, "question": item["question"], "gold": gold, "pred": pred, "raw": raw, "correct": ok})
        summary[group] = correct / len(sample)
        print(group, summary[group])

    summary["overall"] = sum(1 for r in results if r["correct"]) / len(results)
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"mmlu_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
