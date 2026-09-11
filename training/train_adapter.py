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
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATASET_PATHS = {
    "medieval": REPO_ROOT / "data" / "processed" / "medieval_npc_dataset.json",
    "modern": REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json",
}

SYSTEM_PROMPTS = {
    "medieval": "You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                "period-appropriate voice consistent with your role. Never break character.",
    "modern": "You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
              "voice consistent with your role. Never break character.",
}

# Modern-domain archetypes — 1:1 role mapping onto the medieval control set
# (guard->police officer, merchant->shopkeeper, scholar->professor,
# innkeeper->bartender, clergy->social worker, herbalist->pharmacist,
# noble->executive, peasant->service worker). See Docs/DATA_PIPELINE.md.
# Supersedes the 2026-08-06 crime-city mapping (cop/dealer/boss/...).
MODERN_ARCHETYPES = ["police officer", "shopkeeper", "professor", "bartender",
                      "social worker", "pharmacist", "executive", "service worker"]


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


SPLIT_PATH = REPO_ROOT / "data" / "processed" / "modern_split.json"


def load_heldout_ids(domain: str) -> set:
    """Ids reserved for evaluation. Empty for domains with no split file.

    Before this existed every adapter trained on every entry for its
    archetype, which left training loss unfalsifiable (no way to tell learning
    from memorising) and meant evaluations that sampled from the dataset were
    scoring adapters on their own training targets. See training/make_split.py.
    """
    if domain != "modern" or not SPLIT_PATH.exists():
        return set()
    return set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["heldout_ids"])


def build_dataset(domain: str, tokenizer, max_samples: int = None, id_prefix: str = None,
                  archetype: str = None, exclude_ids: set = None, only_ids: set = None) -> Dataset:
    """Builds a prompt/completion dataset, NOT a flat 'text' field.

    `id_prefix`: keep only entries whose id starts with this (e.g. "GUT-").
    Added to test whether training on only the genuinely archaic-pronoun-dense
    subset (all 626 GUT-* entries have >=1 dialect_features marker, vs 79.5%
    across the full 1003-entry mix) produces measurable dialect markers at
    generation time — the fixed-loss-masking runs still showed zero, and the
    working theory is that chimbiwide/hand-authored entries (medieval-themed
    but often pronoun-light) dilute the "always use thee/thou" signal even
    though "sound medieval" comes through fine. See Docs/TODO.md Phase 3.

    Bug this fixes: TRL's SFTTrainer only masks the prompt out of the loss
    (completion_only_loss) when the dataset has "prompt"/"completion" columns
    (see trl.trainer.sft_trainer.SFTTrainer, line ~352: completion_only_loss
    defaults to `"prompt" in dataset_sample and "completion" in dataset_sample`).
    With a flat pre-templated "text" field, every token — including the
    templated system prompt and the player's question — contributed to the
    loss equally. Since the system prompt is long, constant, and trivially
    memorizable, and the assistant's archaic vocabulary was a small fraction
    of total tokens, six full training runs (r=8/16/32, alpha=16/32, 3/8
    epochs) all converged nicely on loss/accuracy while producing zero
    archaic dialect markers at generation time — the gradient signal for the
    thing we actually wanted to learn was being diluted by the prompt tokens.

    TinyLlama-Chat's own chat_template.jinja has no {% generation %} tags, so
    the alternative fix (assistant_masks via return_assistant_tokens_mask)
    isn't available — prompt/completion is the template-independent fix.
    """
    path = DATASET_PATHS[domain]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist — {domain} dataset hasn't been built yet. "
            f"Only 'medieval' is ready as of Phase 2."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    if id_prefix:
        entries = [e for e in entries if e["id"].startswith(id_prefix)]
    if archetype:
        entries = [e for e in entries if e.get("persona", {}).get("archetype") == archetype]
    if exclude_ids:
        entries = [e for e in entries if e["id"] not in exclude_ids]
    if only_ids is not None:
        entries = [e for e in entries if e["id"] in only_ids]
    if max_samples:
        entries = entries[:max_samples]

    system_template = SYSTEM_PROMPTS.get(domain, "You are a helpful NPC. Stay in character.")
    prompts, completions = [], []
    for e in entries:
        archetype = e.get("persona", {}).get("archetype", "npc")
        system = system_template.format(archetype=archetype)
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": e["input"]},
        ]
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        # template's add_generation_prompt block already leaves a trailing
        # newline after "<|assistant|>" (matches its own assistant-turn
        # pattern "<|assistant|>\n{content}") — don't add a second one.
        completion = e["output"] + tokenizer.eos_token
        prompts.append(prompt)
        completions.append(completion)

    return Dataset.from_dict({"prompt": prompts, "completion": completions})


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
    parser.add_argument("--lora-alpha", type=int, default=None, help="override lora_config.yaml alpha (scales adapter's effective pull)")
    parser.add_argument("--id-prefix", default=None, help="train only on entries whose id starts with this, e.g. GUT- (dialect-density experiment)")
    parser.add_argument("--archetype", default=None, help="train only on this archetype, e.g. guard (Week 2: single-archetype adapter)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-holdout", action="store_true",
                        help="train on every entry, as the original 8 adapters were. Off by "
                             "default: without a held-out set there is no eval loss, so a lower "
                             "training loss cannot be told apart from memorisation.")
    parser.add_argument("--early-stopping", type=int, default=3,
                        help="stop after this many evals with no eval_loss improvement (0 = off)")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="override training_args.yaml warmup_ratio (0 = the original no-warmup)")
    parser.add_argument("--scheduler", default=None,
                        help="override lr_scheduler_type, e.g. 'linear' to match the original runs")
    parser.add_argument("--tag", default=None,
                        help="appended to the output directory name, so ablations never collide")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="per-device batch size; gradient accumulation is adjusted so the "
                             "effective batch is unchanged (a speed knob, not a training change)")
    args = parser.parse_args()

    lora_cfg = load_config("lora_config.yaml")
    train_cfg = load_config("training_args.yaml")
    if args.lora_r:
        lora_cfg["r"] = args.lora_r
    if args.lora_alpha:
        lora_cfg["lora_alpha"] = args.lora_alpha
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    if args.warmup_ratio is not None:
        train_cfg["warmup_ratio"] = args.warmup_ratio
    if args.scheduler:
        train_cfg["lr_scheduler_type"] = args.scheduler

    dir_suffix = f"r{lora_cfg['r']}"
    if lora_cfg["lora_alpha"] != 16:
        dir_suffix += f"_a{lora_cfg['lora_alpha']}"
    if train_cfg["num_train_epochs"] != 3:
        dir_suffix += f"_e{int(train_cfg['num_train_epochs'])}"
    if args.max_samples:
        # Prevents smoke tests from colliding with (and overwriting) a real
        # full-dataset run's directory just because rank/alpha/epochs match —
        # this exact collision destroyed the original medieval_r8 baseline's
        # weights once already (harmless in that case since already
        # documented, but don't repeat it).
        dir_suffix += f"_smoketest{args.max_samples}"
    if args.id_prefix:
        dir_suffix += f"_{args.id_prefix.rstrip('-').lower()}only"
    if args.archetype:
        dir_suffix += f"_{args.archetype.replace(' ', '')}"
    heldout_ids = set() if args.no_holdout else load_heldout_ids(args.domain)
    if heldout_ids:
        # Distinct suffix, so a held-out run can never share a directory with
        # one of the original adapters. That matters more than it looks: on
        # torch >= 2.6 this script resumes from any checkpoint it finds in the
        # output dir, so a colliding name would silently *continue training a
        # production adapter* rather than start a clean run.
        dir_suffix += "_ho"
    if args.tag:
        dir_suffix += f"_{args.tag}"
    output_dir = ADAPTERS_DIR / f"{args.domain}_{dir_suffix}"

    print(f"loading base model: {BASE_MODEL} (4-bit QLoRA)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_quantized_model()

    print(f"building dataset for domain '{args.domain}'" + (f" (max {args.max_samples} samples)" if args.max_samples else ""))
    dataset = build_dataset(args.domain, tokenizer, args.max_samples, args.id_prefix,
                            args.archetype, exclude_ids=heldout_ids)
    eval_dataset = None
    if heldout_ids:
        eval_dataset = build_dataset(args.domain, tokenizer, None, args.id_prefix,
                                     args.archetype, only_ids=heldout_ids)
        if len(eval_dataset) == 0:
            eval_dataset = None
    print(f"dataset size: {len(dataset)} train"
          + (f", {len(eval_dataset)} held-out" if eval_dataset is not None
             else " (no held-out set: eval loss unavailable)"))

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    # Micro-batch is a speed knob, not a training change: accumulation is
    # derived so the effective batch (and so the optimisation) stays the same.
    # 4x4 was chosen for the MX450's 2 GB of VRAM; on a card with room, 16x1
    # does the same work in a quarter of the forward/backward calls.
    effective_batch = (train_cfg["per_device_train_batch_size"]
                       * train_cfg["gradient_accumulation_steps"])
    micro = args.micro_batch or train_cfg["per_device_train_batch_size"]
    if effective_batch % micro:
        raise SystemExit(f"--micro-batch {micro} must divide the effective batch {effective_batch}")
    accum = effective_batch // micro

    has_eval = eval_dataset is not None
    # Log and evaluate densely. The original runs are 24 optimiser steps long
    # and logged every 10, which recorded two loss values per adapter — not a
    # curve, and no way to see where training stopped helping.
    eval_every = train_cfg.get("eval_steps", 2)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=train_cfg["learning_rate"],
        # Warmup + cosine. With the default (linear decay, no warmup) the very
        # first optimiser steps run at the full 2e-4 on freshly initialised
        # adapter weights; on a 24-step run those few steps are a large share
        # of all training.
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        fp16=train_cfg["fp16"],
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg.get("logging_steps_eval", 2) if has_eval else train_cfg["logging_steps"],
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_every if has_eval else None,
        # Step-based checkpointing, not epoch-based: this hardware has crashed
        # mid-epoch under thermal load before (driver reset, no traceback,
        # lost a full hour of an r=16 run at step 56/189). With evaluation on,
        # saves must land on eval steps for load_best_model_at_end to work,
        # which also makes crashes cheaper still.
        save_strategy="steps",
        save_steps=eval_every if has_eval else 15,
        save_total_limit=2 if has_eval else 3,
        # Keep the checkpoint with the lowest *held-out* loss, not the last
        # one. Training loss keeps falling as the adapter memorises; eval loss
        # turns back up when that starts to cost generalisation.
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"{args.domain}-{dir_suffix}-adapter",
        max_length=512,
        packing=False,
    )

    callbacks = []
    if has_eval and args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # True trainer resume (optimizer/scheduler state) requires torch>=2.6 —
    # transformers refuses to torch.load optimizer.pt otherwise (CVE-2025-32434
    # restriction). On the original machine (torch 2.5.1) resume therefore
    # never ran. On the current one (torch 2.11, required for the RTX 5060's
    # sm_120) it DOES run, automatically, whenever the output directory already
    # holds a checkpoint. That is why held-out runs get their own `_ho` suffix
    # and ablations a `--tag`: a run that collided with an existing adapter's
    # directory would silently resume and keep training that adapter instead
    # of starting clean.
    resume_checkpoint = None
    torch_version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if output_dir.exists() and torch_version >= (2, 6):
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"resuming from checkpoint: {resume_checkpoint}")
    elif output_dir.exists() and any(output_dir.glob("checkpoint-*")):
        print(f"note: {output_dir} has checkpoints but torch {torch.__version__} < 2.6 "
              f"can't resume trainer state — starting fresh (old checkpoints untouched).")

    print("starting training...")
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    wall_s = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"adapter saved -> {output_dir}")

    # The original adapters kept their loss history only inside intermediate
    # checkpoints, which save_total_limit then deletes. Written here so a run's
    # numbers survive alongside its weights.
    history = trainer.state.log_history
    train_pts = [(h["step"], h["loss"]) for h in history if "loss" in h]
    eval_pts = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    summary = {
        "output_dir": output_dir.name,
        "base_model": BASE_MODEL,
        "train_examples": len(dataset),
        "heldout_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "effective_batch": effective_batch,
        "micro_batch": micro,
        "grad_accum": accum,
        "global_steps": trainer.state.global_step,
        "wall_seconds": round(wall_s, 1),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_train_loss": train_pts[-1][1] if train_pts else None,
        "train_loss_curve": train_pts,
        "eval_loss_curve": eval_pts,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"train steps: {summary['global_steps']}  wall: {summary['wall_seconds']}s")
    if eval_pts:
        print(f"eval loss: {eval_pts[0][1]:.4f} -> best {summary['best_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
