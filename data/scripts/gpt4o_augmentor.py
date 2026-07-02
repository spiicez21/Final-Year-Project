"""
Fill archetype gaps in the medieval NPC dataset with GPT-4o-generated
synthetic entries (see DevFiles/Specs.md section 6, "Archetype Distribution
Target" table).

Reads current archetype counts from data/processed/medieval_npc_dataset.json,
computes the gap against the target table, and prompts GPT-4o to generate
`quality_score`-tagged entries in schema v1.0 to close it.

Requires OPENAI_API_KEY in the environment. Costs real money per run —
defaults to a small --limit per archetype so an accidental full run doesn't
burn budget.

Usage:
    python gpt4o_augmentor.py --archetype guard --count 20
    python gpt4o_augmentor.py --all --limit-per-archetype 10   # gap-fill pass
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[1] / "processed" / "medieval_npc_dataset.json"

# Target counts from Specs.md section 6.
ARCHETYPE_TARGETS = {
    "noble": 150, "peasant": 150, "guard": 200, "merchant": 150,
    "scholar": 150, "innkeeper": 100, "herbalist": 50, "clergy": 50,
}

SYSTEM_PROMPT = """You write short medieval RPG NPC dialogue pairs for a training \
dataset. Output strict JSON matching this schema per entry:
{
  "input": "player line",
  "output": "NPC response, archaic voice (thee/thou/hath/etc where natural)",
  "persona": {"archetype": "<archetype>", "name": "", "disposition": "friendly|neutral|suspicious|hostile|reverent", "social_class": "..."},
  "context": {"location": "...", "time_of_day": "morning|midday|evening|night", "world_state": "peacetime|conflict|festival|drought"},
  "linguistic_markers": {"formality": "low|medium|high", "dialect_features": ["thee", "..."], "vocabulary_tier": "simple|mixed|elevated"},
  "metadata": {"intent": "information|quest|lore|trade|social|combat", "persona_stress_test": false, "stress_test_type": null, "source": "synthetic_gpt4o", "quality_score": 7, "tags": []}
}
Do not break persona voice. Do not reference anything anachronistic."""


def current_counts() -> Counter:
    if not DATASET_PATH.exists():
        return Counter()
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return Counter(e["persona"]["archetype"] for e in data["entries"])


def gaps() -> dict:
    counts = current_counts()
    return {
        archetype: max(0, target - counts.get(archetype, 0))
        for archetype, target in ARCHETYPE_TARGETS.items()
    }


def generate_batch(archetype: str, count: int) -> list:
    """Call GPT-4o to generate `count` dialogue entries for one archetype.

    Deferred import + API key check here so `--dry-run`/gap reporting
    works without the openai package or a configured key.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set — required for generation.")

    from openai import OpenAI

    client = OpenAI()
    user_prompt = f"Generate {count} entries for archetype '{archetype}'. Return a JSON array only."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("entries", parsed if isinstance(parsed, list) else [])


def merge_entries(new_entries: list):
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8")) if DATASET_PATH.exists() else {
        "metadata": {"description": "Medieval NPC dialogue pairs", "schema_version": "1.0", "sources": {}},
        "entries": [],
    }
    next_id = len(data["entries"]) + 1
    for entry in new_entries:
        entry["id"] = f"SYN-{next_id:04d}"
        data["entries"].append(entry)
        next_id += 1
    data["metadata"]["total_pairs"] = len(data["entries"])
    DATASET_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(new_entries)} entries -> {DATASET_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetype", help="single archetype to fill")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="fill every archetype gap")
    parser.add_argument("--limit-per-archetype", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="only print the gap report, generate nothing")
    args = parser.parse_args()

    gap_report = gaps()
    print("archetype gap report (target - current):")
    for archetype, gap in gap_report.items():
        print(f"  {archetype:12s} {gap}")

    if args.dry_run:
        return

    if args.archetype:
        entries = generate_batch(args.archetype, args.count)
        merge_entries(entries)
        return

    if args.all:
        for archetype, gap in gap_report.items():
            if gap <= 0:
                continue
            batch_size = min(gap, args.limit_per_archetype)
            entries = generate_batch(archetype, batch_size)
            merge_entries(entries)


if __name__ == "__main__":
    main()
