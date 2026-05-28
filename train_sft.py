import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def load_env(path: str = ".env") -> None:
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


PROMPT_TEMPLATE = (
    "You are evaluating commonsense morality.\n"
    "Write a short chain of thought, then return exactly one label:\n"
    "0 = morally acceptable\n"
    "1 = morally wrong\n\n"
    "Scenario: {scenario}\n"
    "Chain of thought:"
)


def format_pair(record: dict) -> tuple[str, str]:
    prompt = PROMPT_TEMPLATE.format(scenario=record["scenario"])
    completion = f" {record['chain_of_thought']}\nFinal answer: {record['final_answer']}"
    return prompt, completion


def build_tokenized_dataset(records: list[dict], tokenizer, max_length: int) -> Dataset:
    eos_id = tokenizer.eos_token_id
    input_ids_all: list[list[int]] = []
    labels_all: list[list[int]] = []
    attention_mask_all: list[list[int]] = []

    for record in records:
        prompt, completion = format_pair(record)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + completion_ids + [eos_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        input_ids_all.append(input_ids)
        labels_all.append(labels)
        attention_mask_all.append([1] * len(input_ids))

    return Dataset.from_dict(
        {
            "input_ids": input_ids_all,
            "labels": labels_all,
            "attention_mask": attention_mask_all,
        }
    )


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--data", default="training_data/synthetic_ethics_cot_training.jsonl"
    )
    parser.add_argument("--output-dir", default="checkpoints/qwen-cot-sft")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora", action="store_true", help="Train with LoRA adapters")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    load_env()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = build_tokenized_dataset(records, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )
    model.gradient_checkpointing_enable()

    if args.lora:

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if args.learning_rate == 2e-5:
            args.learning_rate = 2e-4

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="constant",
        optim="adafactor",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
