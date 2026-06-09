"""Geração de predições no Spider dev split (Fases 2 e 4).

O mesmo template de prompt few-shot é usado para o baseline e para os
modelos fine-tuned; a única diferença é o carregamento opcional do
adaptador LoRA via --adapter. A decodificação é determinística (greedy).

Uso:
    # baseline
    python scripts/generate_predictions.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --dev_file data/dev.jsonl --few_shot data/few_shot.json \
        --out results/predictions_baseline.jsonl

    # fine-tuned
    python scripts/generate_predictions.py \
        --model_name Qwen/Qwen2.5-3B-Instruct --adapter outputs/lora_a \
        --dev_file data/dev.jsonl --few_shot data/few_shot.json \
        --out results/predictions_lora_a.jsonl
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from spider_common import (
    build_few_shot_messages,
    build_user_message,
    load_jsonl,
    save_jsonl,
    set_global_seed,
)


def load_model(model_name: str, adapter: str | None, use_4bit: bool):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if quant_config is None else None,
        device_map="auto",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--few_shot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="avalia apenas as N primeiras entradas (útil para depuração)")
    args = parser.parse_args()

    set_global_seed()

    with open(args.few_shot, encoding="utf-8") as f:
        few_shot = json.load(f)
    base_messages = build_few_shot_messages(few_shot)

    dev = load_jsonl(args.dev_file)
    if args.limit:
        dev = dev[: args.limit]

    model, tokenizer = load_model(args.model_name, args.adapter, args.use_4bit)

    predictions = []
    for i, item in enumerate(dev):
        messages = base_messages + [
            {"role": "user", "content": build_user_message(item["schema"], item["question"])}
        ]
        raw_output = generate(model, tokenizer, messages, args.max_new_tokens)
        predictions.append(
            {
                "db_id": item["db_id"],
                "question": item["question"],
                "gold_sql": item["gold_sql"],
                "raw_output": raw_output,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(dev)} predições geradas")
            save_jsonl(predictions, args.out)

    save_jsonl(predictions, args.out)
    print(f"Predições salvas em {args.out}")


if __name__ == "__main__":
    main()
