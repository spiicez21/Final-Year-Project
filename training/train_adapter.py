"""
Train a domain LoRA adapter on TinyLlama-1.1B-Chat via QLoRA (4-bit base +
LoRA adapter), using TRL's SFTTrainer. Config defaults match
DevFiles/Specs.md section 5/7 (r=8, alpha=16, q_proj/v_proj, 3 epochs).

Base model choice: TinyLlama/TinyLlama-1.1B-Chat-v1.0 — same variant used
for the Ollama baseline (Condition A), so Condition B stays comparable.

4-bit QLoRA is used because the local GPU (MX450, 2.15GB VRAM) cannot fit
fp16 full-model training — 4-bit quantization brings the base weights to
~0.6GB, leaving room for LoRA params, gradients, and activations.

Usage:
    python train_adapter.py --domain medieval --max-samples 50   # smoke test
    python train_adapter.py --domain medieval                    # full run
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATASET_PATHS = {
    "medieval": REPO_ROOT / "data" / "processed" / "medieval_npc_dataset.json",
    "healthcare": REPO_ROOT / "data" / "processed" / "healthcare_dataset.json",
    "education": REPO_ROOT / "data" / "processed" / "education_dataset.json",
}

SYSTEM_PROMPTS = {
    "medieval": "You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                "period-appropriate voice consistent with your role. Never break character.",
}


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def build_dataset(domain: str, tokenizer, max_samples: int = None) -> Dataset:
    path = DATASET_PATHS[domain]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist — {domain} dataset hasn't been built yet. "
            f"Only 'medieval' is ready as of Phase 2."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    if max_samples:
        entries = entries[:max_samples]

    system_template = SYSTEM_PROMPTS.get(domain, "You are a helpful NPC. Stay in character.")
    texts = []
    for e in entries:
        archetype = e.get("persona", {}).get("archetype", "npc")
        system = system_template.format(archetype=archetype)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": e["input"]},
            {"role": "assistant", "content": e["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def load_quantized_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--max-samples", type=int, default=None, help="cap training examples (smoke tests)")
    parser.add_argument("--epochs", type=int, default=None, help="override training_args.yaml epoch count")
    parser.add_argument("--lora-r", type=int, default=None, help="override lora_config.yaml rank (for ablation)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    lora_cfg = load_config("lora_config.yaml")
    train_cfg = load_config("training_args.yaml")
    if args.lora_r:
        lora_cfg["r"] = args.lora_r
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs

    output_dir = ADAPTERS_DIR / f"{args.domain}_r{lora_cfg['r']}"

    print(f"loading base model: {BASE_MODEL} (4-bit QLoRA)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_quantized_model()

    print(f"building dataset for domain '{args.domain}'" + (f" (max {args.max_samples} samples)" if args.max_samples else ""))
    dataset = build_dataset(args.domain, tokenizer, args.max_samples)
    print(f"dataset size: {len(dataset)}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg["logging_steps"],
        # Step-based checkpointing, not epoch-based: this hardware has crashed
        # mid-epoch under thermal load before (driver reset, no traceback,
        # lost a full hour of an r=16 run at step 56/189). Saving every 15
        # steps (~4-5min) means a crash loses minutes, not an epoch.
        save_strategy="steps",
        save_steps=15,
        save_total_limit=3,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"{args.domain}-r{lora_cfg['r']}-adapter",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_checkpoint = None
    if output_dir.exists():
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"resuming from checkpoint: {resume_checkpoint}")

    print("starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
