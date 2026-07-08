"""
Condition D (Specs.md section 8): full parameter fine-tune of TinyLlama-1.1B-
Chat on the medieval dataset — no LoRA, no quantization. This is the "cost
comparison" condition: how much does full fine-tuning cost (compute, storage)
vs. Condition B's LoRA adapter, for whatever quality difference it buys.

Cannot run on the local MX450 (2.15GB VRAM) — full fp32 Adam optimizer state
for 1.1B params alone is ~8.8GB, plus ~2.2GB bf16 weights, ~2.2GB gradients,
plus activations. Needs a real GPU (Colab T4's 16GB fits with gradient
checkpointing + 8-bit Adam; see below).

Reuses build_dataset() from train_adapter.py — same prompt/completion format,
same completion-only-loss fix, same system prompt template. Best comparability
against Condition B (medieval_r8_gutonly): run with the same --id-prefix GUT-
filter, since that's the config that actually produced archaic markers.

Usage:
    python train_full_finetune.py --domain medieval --id-prefix GUT- --max-samples 20   # smoke test
    python train_full_finetune.py --domain medieval --id-prefix GUT-                    # full run (Colab)
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_adapter import BASE_MODEL, build_dataset, load_config

ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"  # reuse the same dir for consistency


def load_full_model():
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required with gradient checkpointing
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="medieval")
    parser.add_argument("--max-samples", type=int, default=None, help="cap training examples (smoke tests)")
    parser.add_argument("--epochs", type=int, default=None, help="override training_args.yaml epoch count")
    parser.add_argument("--id-prefix", default="GUT-",
                         help="filter training entries, default GUT- to match the working Condition B config")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train_cfg = load_config("training_args.yaml")
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs

    dir_suffix = "full_finetune"
    if args.id_prefix:
        dir_suffix += f"_{args.id_prefix.rstrip('-').lower()}only"
    if args.max_samples:
        dir_suffix += f"_smoketest{args.max_samples}"
    output_dir = ADAPTERS_DIR / f"{args.domain}_{dir_suffix}"

    print(f"loading base model: {BASE_MODEL} (full fp32-optimizer fine-tune, bf16 weights)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_full_model()

    print(f"building dataset for domain '{args.domain}'" + (f" (max {args.max_samples} samples)" if args.max_samples else ""))
    dataset = build_dataset(args.domain, tokenizer, args.max_samples, args.id_prefix)
    print(f"dataset size: {len(dataset)}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        # Smaller per-device batch than the LoRA runs (2 vs 4) — full
        # fine-tune needs much more memory per sample (gradients+optimizer
        # state for every one of the 1.1B params, not just the ~1M LoRA
        # params). Same effective batch size (16) via more accumulation steps.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=train_cfg["learning_rate"],
        fp16=False,
        bf16=True,
        # 8-bit Adam (bitsandbytes) roughly halves optimizer memory vs
        # standard fp32 Adam — the difference between fitting on a T4 and not.
        optim="adamw_bnb_8bit",
        logging_steps=train_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=15,
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"{args.domain}-{dir_suffix}",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)

    print("starting full fine-tune...")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"full fine-tuned model saved -> {output_dir}")

    size_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"total size on disk: {size_bytes / 1e6:.1f} MB (vs. ~2.3MB for the LoRA adapter — "
          f"this is the storage-cost side of the Condition B vs D comparison)")


if __name__ == "__main__":
    main()
