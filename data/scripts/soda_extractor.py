"""
Extract modern-city NPC dialogue pairs from SODA (allenai/soda) into the
project's modern schema. Weeks 3-4 (Docs/TODO.md): building the real
600-pair (75 x 8) modern-city set.

SODA's `speakers` field carries the actual character names/roles used in
each narrative-derived dialogue (e.g. "Priest", "Officer", "Teacher") — this
is a real, verified way to find archetype-matched conversations without
needing an LLM call: search for a speaker whose name matches one of the 8
target archetypes' role vocabulary, then take (preceding turn, that
speaker's turn) as the (input, output) pair.

Not a heavy rewrite like chimbiwide's register_rewrite(): SODA's raw text is
already natural, contemporary conversational English in the right register
for a modern-city NPC, so this is closer to extraction + relabeling than
persona re-voicing. Documented honestly in each entry's conversion_note
rather than overclaiming a rewrite pass that didn't happen.

MultiWOZ 2.2 turned out to have zero "police" domain dialogues on
inspection (2026-08-08) despite Docs/DATA_PIPELINE.md's original
description — SODA is the real source for police officer coverage instead.

Usage:
    python data/scripts/soda_extractor.py --per-archetype 75          # report only
    python data/scripts/soda_extractor.py --per-archetype 75 --merge  # also merge
"""

import argparse
import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "raw" / "modern" / "soda" / "train.parquet"
OUT_PATH = Path(__file__).resolve().parents[1] / "processed" / "modern_npc_dataset.json"

# archetype -> speaker-name regex. Verified against real counts in the SODA
# training split before building this (see Docs/TODO.md): executive 50252,
# professor 18425, shopkeeper 16510, service worker 11681, social worker
# 9920, police officer 8606, bartender 1758, pharmacist 99 matching rows.
ARCHETYPE_PATTERNS = {
    # "officer" alone was dropped — matched false positives like "Ski Patrol
    # Officer" and "Loan Officer" that aren't police (caught on manual
    # spot-check, 2026-08-08). "police officer"/"cop"/"policeman"/
    # "policewoman" are unambiguous.
    "police officer": r"\b(police officer|cop|policeman|policewoman)\b",
    "shopkeeper": r"\b(shopkeeper|shop owner|store owner|clerk|cashier|salesperson|vendor)\b",
    "professor": r"\b(professor|teacher|lecturer|instructor)\b",
    "bartender": r"\b(bartender|barkeep|barista)\b",
    "social worker": r"\b(social worker|counselor|therapist)\b",
    "pharmacist": r"\b(pharmacist|druggist)\b",
    "executive": r"\b(executive|ceo|manager|boss|director)\b",
    "service worker": r"\b(waiter|waitress|server|receptionist|attendant)\b",
}

MIN_WORDS, MAX_WORDS = 3, 60
STAGE_DIRECTION = re.compile(r"\([^)]*\)")


def _clean(text: str) -> str:
    # SODA's narrative-derived dialogue sometimes embeds inline stage
    # directions, e.g. "(He looks at the prescription.)" — strip these,
    # dialogue text shouldn't contain third-person narration.
    text = STAGE_DIRECTION.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _quality_ok(text: str) -> bool:
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS


def extract(per_archetype: int):
    import pandas as pd

    print(f"loading {RAW_PATH} (speakers + dialogue + original_index columns)...")
    df = pd.read_parquet(RAW_PATH, engine="pyarrow", columns=["speakers", "dialogue", "original_index"])
    print(f"loaded {len(df)} rows")

    patterns = {a: re.compile(p, re.IGNORECASE) for a, p in ARCHETYPE_PATTERNS.items()}
    kept = {a: [] for a in ARCHETYPE_PATTERNS}
    seen_outputs = {a: set() for a in ARCHETYPE_PATTERNS}

    for speakers, dialogue, orig_idx in zip(df["speakers"], df["dialogue"], df["original_index"]):
        if len(speakers) < 2 or len(dialogue) < 2:
            continue
        for archetype, pattern in patterns.items():
            if len(kept[archetype]) >= per_archetype:
                continue
            # find the first turn spoken by a name matching this archetype's
            # role vocabulary, that also has a preceding turn from someone else
            for i in range(1, len(speakers)):
                if pattern.search(str(speakers[i])) and speakers[i] != speakers[i - 1]:
                    input_text = _clean(str(dialogue[i - 1]))
                    output_text = _clean(str(dialogue[i]))
                    if not (_quality_ok(input_text) and _quality_ok(output_text)):
                        continue
                    if output_text in seen_outputs[archetype]:
                        continue
                    seen_outputs[archetype].add(output_text)
                    kept[archetype].append({
                        "input": input_text,
                        "output": output_text,
                        "speaker_name": str(speakers[i]),
                        "original_index": int(orig_idx),
                    })
                    break  # one pair per dialogue per archetype, avoid oversampling one narrative

    for archetype, rows in kept.items():
        print(f"{archetype:16s} {len(rows):4d} / {per_archetype} target")
    return kept


def build_entries(kept: dict, next_id: int) -> list:
    entries = []
    for archetype, rows in kept.items():
        for row in rows:
            entries.append({
                "id": f"SOD-{next_id:04d}",
                "input": row["input"],
                "output": row["output"],
                "persona": {
                    "archetype": archetype,
                    "name": row["speaker_name"],
                    "disposition": "neutral",
                    "social_class": archetype,
                },
                "context": {"location": "unspecified", "time_of_day": "unspecified", "world_state": "modern"},
                "linguistic_markers": {
                    "formality": "medium",
                    "dialect_features": [],
                    "vocabulary_tier": "mixed",
                },
                "metadata": {
                    "intent": "social",
                    "persona_stress_test": False,
                    "stress_test_type": None,
                    "source": "soda",
                    "quality_score": 5,
                    "tags": ["soda", archetype.replace(" ", "_")],
                    "provenance": {"source": "allenai/soda", "original_index": row["original_index"]},
                    "conversion_note": ("Extracted directly from a SODA dialogue where the speaker's own name "
                                        f"matched the '{archetype}' role vocabulary — text used as-is (already "
                                        "natural contemporary register), not lexically rewritten. See "
                                        "data/scripts/soda_extractor.py."),
                },
            })
            next_id += 1
    return entries


def merge_into_dataset(entries: list):
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    data["entries"].extend(entries)
    data["metadata"]["total_pairs"] = len(data["entries"])
    data["metadata"]["sources"]["soda"] = {
        "pairs_extracted": len(entries),
        "method": "allenai/soda (CC BY 4.0), speaker-name role matching against the 8 archetypes, no rewrite (see data/scripts/soda_extractor.py).",
    }
    OUT_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(entries)} entries -> {OUT_PATH}, total now {len(data['entries'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-archetype", type=int, default=75)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    kept = extract(args.per_archetype)
    if not args.merge:
        print("(pass --merge to write these into the dataset)")
        return

    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    next_id = len(data["entries"]) + 1
    entries = build_entries(kept, next_id)
    merge_into_dataset(entries)


if __name__ == "__main__":
    main()
