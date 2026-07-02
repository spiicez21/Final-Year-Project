"""
Convert chimbiwide/NPC-Dialogue_v2 (HuggingFace) into the project's medieval
NPC dialogue schema (see DevFiles/Specs.md section 6).

Pipeline: download -> filter (medieval-plausible entries only) -> clean
(strip modern references) -> register rewrite (archaic voice) -> merge.

Status: skeleton. The register-rewrite step needs an LLM call (GPT-4o or
local model) and is left as a TODO — do not merge raw chimbiwide text
without it, since the source dataset is modern-register and carries an
IP risk noted in Specs.md (medium — IP scrub required before any
publication use).

Usage:
    python chimbiwide_converter.py --limit 300
"""

import argparse
import json
import re
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "processed" / "medieval_npc_dataset.json"
CACHE_DIR = Path(__file__).resolve().parents[1] / "raw" / "huggingface" / "chimbiwide"

# Background-blurb keyword -> our archetype set. Order matters (first match wins).
ARCHETYPE_REMAP = {
    "bounty hunter": "guard", "knight": "guard", "soldier": "guard", "guard": "guard",
    "assassin": "guard", "mercenary": "guard",
    "smuggler": "merchant", "shopkeeper": "merchant", "trader": "merchant", "merchant": "merchant",
    "wizard": "scholar", "sage": "scholar", "professor": "scholar", "scholar": "scholar",
    "king": "noble", "queen": "noble", "lord": "noble", "lady": "noble", "noble": "noble",
    "tavern": "innkeeper", "innkeeper": "innkeeper", "bartender": "innkeeper",
    "healer": "herbalist", "alchemist": "herbalist", "herbalist": "herbalist",
    "priest": "clergy", "monk": "clergy", "clergy": "clergy",
    "villager": "peasant", "farmer": "peasant", "thief": "peasant",
}

# Terms that mark a line as out-of-period / unusable without heavy rewrite.
MODERN_LEAKAGE = [
    "phone", "computer", "internet", "email", "wifi", "gun",
    "okay", "gonna", "wanna", "rupees", "police",
]


def load_source(limit: int):
    """Load chimbiwide/NPC-Dialogue_v2 via the `datasets` library.

    Deferred import: `datasets` is only needed here, not elsewhere in the
    data pipeline.
    """
    from datasets import load_dataset

    ds = load_dataset("chimbiwide/NPC-Dialogue_v2", "dialogue", split="train", cache_dir=str(CACHE_DIR))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def parse_row(row: dict) -> dict:
    """Each row is a `messages` list: msg[0] = user roleplay-setup prompt
    (contains "You are <Name>." + a Background: blurb), then alternating
    user/assistant turns. Extract the character name/background and the
    dialogue turns as (input, output) pairs."""
    messages = row["messages"]
    setup = messages[0]["content"]

    name_match = re.search(r"You are ([^.]+)\.", setup)
    name = name_match.group(1).strip() if name_match else ""

    bg_match = re.search(r"Background:\s*(.*?)\s*Current Location:", setup, re.DOTALL)
    background = bg_match.group(1).strip() if bg_match else setup

    # messages[0] = setup (user), messages[1] = opening greeting (assistant),
    # then real turns alternate user/assistant from messages[2] onward.
    pairs = []
    for i in range(1, len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append({"input": messages[i]["content"], "output": messages[i + 1]["content"]})

    return {"name": name, "background": background, "pairs": pairs}


def is_medieval_plausible(text: str) -> bool:
    lowered = text.lower()
    return not any(re.search(rf"\b{term}\b", lowered) for term in MODERN_LEAKAGE)


def remap_archetype(background: str) -> str:
    lowered = background.lower()
    for keyword, archetype in ARCHETYPE_REMAP.items():
        if keyword in lowered:
            return archetype
    return "peasant"


def register_rewrite(text: str) -> str:
    """Rewrite modern-register text into archaic NPC voice.

    TODO: wire up to gpt4o_augmentor's LLM client once available, or a
    local model. Passing text through unchanged is NOT acceptable for the
    final dataset — flagged here so it's impossible to miss.
    """
    raise NotImplementedError(
        "register_rewrite needs an LLM backend (see gpt4o_augmentor.py). "
        "Do not merge chimbiwide entries until this is implemented."
    )


def convert(limit: int = 300):
    rows = load_source(limit)
    kept, dropped, no_pairs = [], 0, 0
    archetype_dist = {}

    for row in rows:
        parsed = parse_row(row)
        if not parsed["pairs"]:
            no_pairs += 1
            continue

        full_text = parsed["background"] + " " + " ".join(p["input"] + " " + p["output"] for p in parsed["pairs"])
        if not is_medieval_plausible(full_text):
            dropped += 1
            continue

        archetype = remap_archetype(parsed["background"])
        archetype_dist[archetype] = archetype_dist.get(archetype, 0) + 1
        kept.append({**parsed, "archetype": archetype})

    print(f"chimbiwide: {len(kept)} medieval-plausible / {dropped} dropped (modern leakage) / {no_pairs} no usable turns")
    print(f"archetype distribution (pre-rewrite): {archetype_dist}")
    print("register_rewrite not yet implemented — stopping before merge.")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    convert(args.limit)


if __name__ == "__main__":
    main()
