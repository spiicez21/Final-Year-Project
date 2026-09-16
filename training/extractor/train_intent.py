"""Train the intent classifier: two small heads over a frozen sentence encoder.

    .venv/Scripts/python.exe training/extractor/make_intent_data.py
    .venv/Scripts/python.exe training/extractor/train_intent.py

The encoder is the DeBERTa-v3-small inside the fine-tuned player-fact GLiNER
(player_facts_gliner/), so no second model is downloaded or kept in memory.
Its token states are mean-pooled into one vector per line, and two linear
heads are trained on those vectors:

  intent   softmax over greeting / farewell / ack / recall / about_npc /
           question / report / statement
  slots    independent sigmoids over the remembered facts a recall question
           asks about (name, section, ..., "any")

The heads are a few thousand parameters and are saved next to the extractor as
intent_head.pt, with the label order in intent_head.json.
"""

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from dialogue.encoder import SentenceEncoder  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL_DIR = HERE / "player_facts_gliner"
INTENTS = ["greeting", "farewell", "ack", "recall", "about_npc", "question", "report", "statement"]
SLOTS = ["name", "year", "department", "section", "college", "hometown", "project", "interests",
         "feeling", "any"]


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def targets(rows):
    y = torch.tensor([INTENTS.index(r["intent"]) for r in rows])
    s = torch.zeros(len(rows), len(SLOTS))
    for i, r in enumerate(rows):
        for slot in r["slots"]:
            s[i, SLOTS.index(slot)] = 1.0
    is_recall = torch.tensor([r["intent"] == "recall" for r in rows])
    return y, s, is_recall


def main():
    torch.manual_seed(20260916)
    random.seed(20260916)
    encoder = SentenceEncoder(MODEL_DIR)
    train, val = load("intent_train.json"), load("intent_val.json")
    xt = encoder.encode([r["text"] for r in train])
    xv = encoder.encode([r["text"] for r in val])
    yt, st, rt = targets(train)
    yv, sv, rv = targets(val)

    dim = xt.shape[1]
    intent_head = torch.nn.Linear(dim, len(INTENTS))
    slot_head = torch.nn.Linear(dim, len(SLOTS))
    params = list(intent_head.parameters()) + list(slot_head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-2)
    best = (1e9, None)
    for epoch in range(400):
        intent_head.train(), slot_head.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(intent_head(xt), yt)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(slot_head(xt[rt]), st[rt])
        loss.backward()
        opt.step()
        with torch.no_grad():
            intent_head.eval(), slot_head.eval()
            vloss = (torch.nn.functional.cross_entropy(intent_head(xv), yv)
                     + torch.nn.functional.binary_cross_entropy_with_logits(slot_head(xv[rv]), sv[rv])).item()
            if vloss < best[0]:
                best = (vloss, {"intent": {k: v.clone() for k, v in intent_head.state_dict().items()},
                                "slots": {k: v.clone() for k, v in slot_head.state_dict().items()},
                                "epoch": epoch})
    intent_head.load_state_dict(best[1]["intent"])
    slot_head.load_state_dict(best[1]["slots"])
    with torch.no_grad():
        acc = (intent_head(xv).argmax(1) == yv).float().mean().item()
    torch.save({"intent": best[1]["intent"], "slots": best[1]["slots"]}, MODEL_DIR / "intent_head.pt")
    meta = {"intents": INTENTS, "slots": SLOTS, "encoder_dim": dim, "best_epoch": best[1]["epoch"],
            "val_loss": round(best[0], 4), "val_intent_accuracy": round(acc, 4),
            "train": len(train), "val": len(val)}
    (MODEL_DIR / "intent_head.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
