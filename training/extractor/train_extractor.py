"""Fine-tune GLiNER small on the synthetic player-fact data.

    .venv/Scripts/python.exe training/extractor/make_extraction_data.py
    .venv/Scripts/python.exe training/extractor/train_extractor.py

Starts from urchade/gliner_small-v2.1 (Apache-2.0, 166M parameters) and saves
the checkpoint with the lowest validation loss to
training/extractor/player_facts_gliner/, which backend/dialogue/extractor.py
loads. Weights are git-ignored like the adapters; this script and the data
generator are what reproduce them.
"""

import json
import shutil
import tempfile
import os
import time
from pathlib import Path

import torch
from gliner import GLiNER

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "player_facts_gliner"
BASE = "urchade/gliner_small-v2.1"


def load(name):
    rows = json.loads((DATA / ("%s.json" % name)).read_text(encoding="utf-8"))
    return [{"tokenized_text": r["tokenized_text"], "ner": r["ner"], "ner_labels": r["ner_labels"]}
            for r in rows]


def main():
    torch.manual_seed(20260915)
    train, val = load("train"), load("val")
    model = GLiNER.from_pretrained(BASE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("train %d  val %d  device %s" % (len(train), len(val), device))

    # Checkpoints go to the system temp directory, weights only: with optimizer
    # state each one was ~1.4 GB, and three of them filled the project drive
    # mid-save. Only the best model is written into the repository.
    checkpoints = Path(os.environ.get("EXTRACTOR_CHECKPOINTS",
                                      Path(tempfile.gettempdir()) / "npc_extractor_checkpoints"))
    args = model.create_training_args(
        output_dir=str(checkpoints),
        save_only_model=True,
        learning_rate=5e-6,          # encoder: small, it already knows English
        others_lr=1e-5,              # span and label heads
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        max_steps=3000,              # ~4 epochs of 6000; at 2000 eval loss was still falling
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=0,
        use_cpu=device == "cpu",
        bf16=device == "cuda",
        report_to="none",
    )
    t0 = time.time()
    trainer = model.train_model(train_dataset=train, eval_dataset=val, training_args=args)
    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    model.save_pretrained(str(OUT))
    summary = {"base": BASE, "train": len(train), "val": len(val), "minutes": round((time.time() - t0) / 60, 1),
               "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
               "eval_curve": [(h["step"], round(h["eval_loss"], 4)) for h in history]}
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    shutil.rmtree(checkpoints, ignore_errors=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
