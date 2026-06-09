import argparse
import json
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from scripts.config import BASE_MODEL, DATA_DIR, SEED, set_global_seed
from scripts.model_utils import load_tokenizer, make_bnb_config


def load_hyperparams(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    default_train = os.path.join(DATA_DIR, "train_chat.jsonl")
    parser.add_argument("--train_file", default=default_train)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    set_global_seed(SEED)
    params = load_hyperparams(args.config)

    tokenizer = load_tokenizer(BASE_MODEL)

    use_qlora = params.get("use_qlora", True)
    quantization_config = make_bnb_config() if use_qlora else None

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=params["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=params["num_train_epochs"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        learning_rate=params["learning_rate"],
        lr_scheduler_type=params.get("lr_scheduler_type", "cosine"),
        warmup_ratio=params.get("warmup_ratio", 0.03),
        logging_steps=params.get("logging_steps", 20),
        save_strategy="epoch",
        gradient_checkpointing=True,
        bf16=False,
        fp16=torch.cuda.is_available(),
        optim=params.get("optim", "paged_adamw_8bit"),
        max_seq_length=params.get("max_seq_length", 2048),
        packing=params.get("packing", False),
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    params_path = os.path.join(args.output_dir, "training_params.json")
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
