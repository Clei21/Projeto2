"""Avaliação 5-shot no MMLU (Fase 5).

A resposta do modelo é determinada comparando a verossimilhança dos tokens
"A", "B", "C" e "D" na posição de resposta, procedimento determinístico e
padrão para benchmarks de múltipla escolha.

Uso:
    # baseline
    python scripts/evaluate_mmlu.py --model_name Qwen/Qwen2.5-3B-Instruct \
        --suite data/mmlu_suite.json --out results/mmlu_baseline.json

    # fine-tuned
    python scripts/evaluate_mmlu.py --model_name Qwen/Qwen2.5-3B-Instruct \
        --adapter outputs/lora_a --suite data/mmlu_suite.json \
        --out results/mmlu_lora_a.json
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from spider_common import set_global_seed

LETTERS = ["A", "B", "C", "D"]


def format_question(q: dict, with_answer: bool) -> str:
    lines = [q["question"].strip()]
    for letter, choice in zip(LETTERS, q["choices"]):
        lines.append(f"{letter}. {choice}")
    answer = f" {LETTERS[q['answer']]}" if with_answer else ""
    lines.append(f"Answer:{answer}")
    return "\n".join(lines)


def build_prompt(subject: str, few_shot: list[dict], question: dict) -> str:
    header = (
        f"The following are multiple choice questions (with answers) "
        f"about {subject.replace('_', ' ')}.\n\n"
    )
    shots = "\n\n".join(format_question(q, with_answer=True) for q in few_shot)
    target = format_question(question, with_answer=False)
    return header + shots + "\n\n" + target


def load_model(model_name: str, adapter: str | None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.inference_mode()
def predict_letter(model, tokenizer, prompt: str, letter_ids: list[int]) -> int:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    letter_logits = logits[letter_ids]
    return int(torch.argmax(letter_logits).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    set_global_seed()

    with open(args.suite, encoding="utf-8") as f:
        suite = json.load(f)

    model, tokenizer = load_model(args.model_name, args.adapter)
    letter_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in LETTERS]

    results = {"model": args.model_name, "adapter": args.adapter, "categories": {}}
    total_correct, total = 0, 0

    for category, data in suite["categories"].items():
        correct = 0
        for q in data["questions"]:
            prompt = build_prompt(data["subject"], data["few_shot"], q)
            predicted = predict_letter(model, tokenizer, prompt, letter_ids)
            correct += int(predicted == q["answer"])

        n = len(data["questions"])
        accuracy = correct / n
        results["categories"][category] = {
            "subject": data["subject"], "correct": correct, "n": n, "accuracy": accuracy,
        }
        total_correct += correct
        total += n
        print(f"{category}: {accuracy:.4f} ({correct}/{n})")

    results["overall_accuracy"] = total_correct / total
    print(f"Acurácia agregada: {results['overall_accuracy']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
