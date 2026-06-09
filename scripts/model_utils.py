import os
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from scripts.config import BASE_MODEL


def make_bnb_config() -> Optional[BitsAndBytesConfig]:
    if not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name: str = BASE_MODEL,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
):
    quantization_config = make_bnb_config() if load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        trust_remote_code=True,
    )

    if adapter_path is not None and os.path.isdir(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    list_of_messages: List[List[dict]],
    max_new_tokens: int,
) -> List[str]:
    prompts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [text.strip() for text in decoded]
