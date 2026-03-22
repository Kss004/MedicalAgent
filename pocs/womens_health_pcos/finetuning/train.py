"""Step 5: Fine-tune Gemma-2-2B-IT with QLoRA using Unsloth.

Designed to run on Google Colab free tier (T4 16GB).
Loads ShareGPT-format JSONL, applies QLoRA, trains with TRL's SFTTrainer.
"""

import json
from pathlib import Path

from config import TRAINING, TRAIN_FILE, EVAL_FILE, SYSTEM_PROMPT


def load_dataset(path: Path) -> list[dict]:
    """Load JSONL ShareGPT dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def train():
    """Run QLoRA fine-tuning with Unsloth."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    cfg = TRAINING

    # ── Load model + tokenizer ─────────────────────────────────────────────
    print(f"[Train] Loading {cfg['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )

    # ── Apply LoRA adapters ────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── Load datasets ──────────────────────────────────────────────────────
    print("[Train] Loading datasets...")
    train_data = load_dataset(TRAIN_FILE)
    eval_data = load_dataset(EVAL_FILE)

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ── Formatting function ────────────────────────────────────────────────
    def format_sharegpt(example):
        """Convert ShareGPT conversations to chat template string."""
        messages = []
        for turn in example["conversations"]:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_ds = train_ds.map(format_sharegpt)
    eval_ds = eval_ds.map(format_sharegpt)

    # ── Training arguments ─────────────────────────────────────────────────
    output_dir = str(Path(__file__).parent / "checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["save_steps"],
        save_total_limit=3,
        report_to="none",
        seed=42,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=True,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("[Train] Starting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = str(Path(__file__).parent / "model_final")
    print(f"[Train] Saving model to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("[Train] Done!")
    return model, tokenizer


if __name__ == "__main__":
    train()
