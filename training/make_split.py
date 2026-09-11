"""Builds a reproducible train / held-out split for the modern-city dataset.

Why this exists. Until now every adapter was trained on *every* entry for its
archetype, and two evaluations then sampled their "test" prompts from the same
file:

  * evaluation/run_bertscore.py     -- test pairs from modern_npc_dataset.json
  * evaluation/run_pdm_v2_multi.py  -- test prompts AND reference features
                                       from modern_npc_dataset.json

So both compared each adapter against text it had been trained on. Training
loss had the same blind spot: with no validation set there is no way to tell a
lower loss from better memorisation.

This writes the split as a list of ids rather than editing the dataset. The
dataset froze on 6 Sep 2026 and must not change; an id list sits alongside it,
is trivially reviewable in a diff, and every script can apply it the same way.

The split is stratified per archetype (each keeps the same fraction held out)
and seeded, so rerunning this produces an identical file.

Usage:
    .venv/Scripts/python.exe training/make_split.py
    .venv/Scripts/python.exe training/make_split.py --fraction 0.15 --seed 7
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
SPLIT = REPO_ROOT / "data" / "processed" / "modern_split.json"

# Held-out entries per archetype are floored at this, so the smallest archetypes
# (115 entries) still leave enough to compute a meaningful eval loss.
MIN_HELDOUT = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    entries = json.loads(DATASET.read_text(encoding="utf-8"))["entries"]
    by_arch = defaultdict(list)
    for e in entries:
        by_arch[e["persona"]["archetype"]].append(e["id"])

    rng = random.Random(args.seed)
    heldout, per_arch = [], {}
    for arch in sorted(by_arch):
        ids = sorted(by_arch[arch])          # sort first: order-independent of file layout
        rng.shuffle(ids)
        n = max(MIN_HELDOUT, round(len(ids) * args.fraction))
        heldout.extend(ids[:n])
        per_arch[arch] = {"total": len(ids), "heldout": n, "train": len(ids) - n}

    SPLIT.write_text(json.dumps({
        "description": "Held-out ids for the modern-city dataset. Train on "
                       "everything NOT listed; evaluate generalisation only on "
                       "these. Stratified by archetype, seeded.",
        "dataset": DATASET.name,
        "seed": args.seed,
        "fraction": args.fraction,
        "per_archetype": per_arch,
        "heldout_ids": sorted(heldout),
    }, indent=2), encoding="utf-8")

    print("wrote %s" % SPLIT.relative_to(REPO_ROOT))
    print("%-16s %6s %8s %6s" % ("archetype", "total", "heldout", "train"))
    for arch, c in per_arch.items():
        print("%-16s %6d %8d %6d" % (arch, c["total"], c["heldout"], c["train"]))
    print("%-16s %6d %8d %6d" % ("TOTAL", len(entries), len(heldout), len(entries) - len(heldout)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
