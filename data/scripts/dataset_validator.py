"""
Validate data/processed/medieval_npc_dataset.json against schema v1.0
(DevFiles/Specs.md section 6): required fields, duplicate detection,
archetype/intent balance, and a few known-artifact heuristics from the
extraction pipelines (stray dialogue tags, mojibake, empty fields).

Usage:
    python dataset_validator.py                 # full report
    python dataset_validator.py --strict         # exit 1 on any error (for CI)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[1] / "processed" / "medieval_npc_dataset.json"

REQUIRED_TOP = ["id", "input", "output", "persona", "context", "linguistic_markers", "metadata"]
REQUIRED_PERSONA = ["archetype", "disposition", "social_class"]
REQUIRED_CONTEXT = ["location", "time_of_day", "world_state"]
REQUIRED_LINGUISTIC = ["formality", "dialect_features", "vocabulary_tier"]
REQUIRED_METADATA = ["intent", "persona_stress_test", "source", "quality_score"]

VALID_ARCHETYPES = {"guard", "merchant", "scholar", "noble", "innkeeper", "herbalist", "clergy", "peasant"}
VALID_DISPOSITIONS = {"friendly", "neutral", "suspicious", "hostile", "reverent"}
VALID_INTENTS = {"information", "quest", "lore", "trade", "social", "combat"}

ARCHETYPE_TARGETS = {
    "noble": 150, "peasant": 150, "guard": 200, "merchant": 150,
    "scholar": 150, "innkeeper": 100, "herbalist": 50, "clergy": 50,
}

# Known-artifact heuristics from the extraction pipelines (Docs/DATA_PIPELINE.md).
STRAY_TAG_PATTERN = re.compile(r"\b(?:said|quoth|answered)\s+(?:the\s+)?[A-Z]?[a-z]*\b", re.IGNORECASE)


def load_entries():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data["entries"]


def check_required_fields(entries: list) -> list:
    errors = []
    for e in entries:
        eid = e.get("id", "<missing id>")
        for field in REQUIRED_TOP:
            if field not in e:
                errors.append(f"{eid}: missing top-level field '{field}'")
                continue
        if not e.get("input", "").strip():
            errors.append(f"{eid}: empty input")
        if not e.get("output", "").strip():
            errors.append(f"{eid}: empty output")

        persona = e.get("persona", {})
        for field in REQUIRED_PERSONA:
            if field not in persona:
                errors.append(f"{eid}: missing persona.{field}")
        if persona.get("archetype") not in VALID_ARCHETYPES:
            errors.append(f"{eid}: invalid archetype '{persona.get('archetype')}'")
        if persona.get("disposition") not in VALID_DISPOSITIONS:
            errors.append(f"{eid}: invalid disposition '{persona.get('disposition')}'")

        context = e.get("context", {})
        for field in REQUIRED_CONTEXT:
            if field not in context:
                errors.append(f"{eid}: missing context.{field}")

        linguistic = e.get("linguistic_markers", {})
        for field in REQUIRED_LINGUISTIC:
            if field not in linguistic:
                errors.append(f"{eid}: missing linguistic_markers.{field}")

        metadata = e.get("metadata", {})
        for field in REQUIRED_METADATA:
            if field not in metadata:
                errors.append(f"{eid}: missing metadata.{field}")
        if metadata.get("intent") not in VALID_INTENTS:
            errors.append(f"{eid}: invalid intent '{metadata.get('intent')}'")
        qs = metadata.get("quality_score")
        if not isinstance(qs, (int, float)) or not (0 <= qs <= 10):
            errors.append(f"{eid}: quality_score out of range or missing ({qs})")
    return errors


def check_duplicates(entries: list) -> list:
    errors = []
    ids = [e.get("id") for e in entries]
    id_counts = Counter(ids)
    for eid, count in id_counts.items():
        if count > 1:
            errors.append(f"duplicate id '{eid}' appears {count} times")

    seen_text = {}
    for e in entries:
        key = (e.get("input", "").strip().lower(), e.get("output", "").strip().lower())
        if key in seen_text:
            errors.append(f"{e.get('id')}: duplicate input/output pair, also seen in {seen_text[key]}")
        else:
            seen_text[key] = e.get("id")
    return errors


def check_known_artifacts(entries: list) -> list:
    """Flags (not hard errors) from known extraction quirks — see DATA_PIPELINE.md."""
    warnings = []
    for e in entries:
        eid = e.get("id", "<missing id>")
        output = e.get("output", "")
        if "�" in e.get("input", "") + output:
            warnings.append(f"{eid}: possible mojibake/encoding corruption")
        if eid.startswith("GUT-") and STRAY_TAG_PATTERN.search(output):
            warnings.append(f"{eid}: possible stray dialogue tag left in output (known Malory extraction artifact)")
        if len(output.split()) < 2:
            warnings.append(f"{eid}: suspiciously short output ('{output}')")
    return warnings


def archetype_balance_report(entries: list):
    counts = Counter(e["persona"]["archetype"] for e in entries if "persona" in e)
    print("\n--- Archetype balance vs. Specs.md target ---")
    for archetype, target in sorted(ARCHETYPE_TARGETS.items(), key=lambda kv: kv[1] - counts.get(kv[0], 0), reverse=True):
        current = counts.get(archetype, 0)
        gap = target - current
        status = f"gap {gap}" if gap > 0 else f"over by {-gap}"
        print(f"  {archetype:12s} target {target:4d}  current {current:4d}  ({status})")


def source_report(entries: list):
    counts = Counter(e.get("metadata", {}).get("source", "unknown") for e in entries)
    print("\n--- Source breakdown ---")
    for source, count in counts.most_common():
        print(f"  {source:20s} {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 if any schema errors found")
    args = parser.parse_args()

    entries = load_entries()
    print(f"loaded {len(entries)} entries from {DATASET_PATH}")

    errors = check_required_fields(entries) + check_duplicates(entries)
    warnings = check_known_artifacts(entries)

    print(f"\n--- Schema errors: {len(errors)} ---")
    for err in errors[:50]:
        print(f"  ERROR: {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more")

    print(f"\n--- Warnings (known artifacts, non-fatal): {len(warnings)} ---")
    for warn in warnings[:20]:
        print(f"  WARN: {warn}")
    if len(warnings) > 20:
        print(f"  ... and {len(warnings) - 20} more")

    archetype_balance_report(entries)
    source_report(entries)

    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
