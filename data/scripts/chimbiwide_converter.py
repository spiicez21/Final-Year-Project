"""
Convert chimbiwide/NPC-Dialogue_v2 (HuggingFace) into the project's medieval
NPC dialogue schema (see DevFiles/Specs.md section 6).

Pipeline: download -> filter (medieval-plausible entries only) -> register
rewrite (rule-based archaic voice) -> merge.

register_rewrite() is a deterministic lexical/grammatical rewriter, not an
LLM call — no API key, no cost, fully reproducible. It is intentionally
modest: contraction expansion, you/your -> thee/thy/thou with a handful of
irregular-verb fixups, and a small modern-vocabulary swap list. It will not
produce Shakespeare-quality archaic prose; it produces "good enough to not
read as a phone-and-wifi contemporary chatlog" prose, tagged as
auto-rewritten so it can be reviewed/upgraded later. Source content is
otherwise unmodified (character voice, plot beats), and Specs.md flags this
source as medium IP risk — do not use rewritten entries for anything beyond
local training/eval without a further scrub pass.

Usage:
    python chimbiwide_converter.py --limit 300                # filter + rewrite, print report
    python chimbiwide_converter.py --limit 300 --merge         # also merge into the dataset
    python chimbiwide_converter.py --limit 300 --merge --max-entries 150
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


# Order matters: contractions first (so "you're" doesn't get half-matched
# by the bare "you" rule), then irregular thou-verb fixups, then bare
# pronouns, then vocabulary swaps. All case-insensitive, case-preserving
# for the first letter (crude but avoids "Thee" mid-sentence looking odd
# at least at sentence starts).
CONTRACTIONS = [
    (r"\byou're\b", "thou art"), (r"\byou've\b", "thou hast"),
    (r"\byou'll\b", "thou shalt"), (r"\byou'd\b", "thou wouldst"),
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"), (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"), (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "cannot"), (r"\bcouldn't\b", "could not"),
    (r"\bwouldn't\b", "would not"), (r"\bshouldn't\b", "should not"),
    (r"\bi'm\b", "I am"), (r"\bi've\b", "I have"),
    (r"\bi'll\b", "I shall"), (r"\bi'd\b", "I would"),
    (r"\bit's\b", "'tis"), (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"), (r"\bwhat's\b", "what is"),
]

# "thou <verb>" irregular fixups — applied after the bare "you" -> "thou"
# swap, since the verb immediately follows in these common cases.
THOU_VERB_FIXUPS = [
    (r"\bthou are\b", "thou art"), (r"\bthou have\b", "thou hast"),
    (r"\bthou do\b", "thou dost"), (r"\bthou did\b", "thou didst"),
    (r"\bthou will\b", "thou wilt"), (r"\bthou can\b", "thou canst"),
    (r"\bthou were\b", "thou wast"),
    # inverted question forms: "Are you" -> "you"->"thou" gives "Are thou",
    # needs the same irregular-verb swap but with verb *before* thou.
    (r"\bare thou\b", "art thou"), (r"\bhave thou\b", "hast thou"),
    (r"\bdo thou\b", "dost thou"), (r"\bdid thou\b", "didst thou"),
    (r"\bwill thou\b", "wilt thou"), (r"\bcan thou\b", "canst thou"),
    (r"\bwere thou\b", "wast thou"),
]

VOCAB_SWAPS = [
    (r"\bokay\b", "aye, it is well"), (r"\bok\b", "aye"),
    (r"\byeah\b", "aye"), (r"\byep\b", "aye"),
    (r"\bhello\b", "well met"), (r"\bhi\b", "well met"),
    (r"\bbye\b", "farewell"), (r"\bgoodbye\b", "farewell"),
    (r"\bmoney\b", "coin"), (r"\bboss\b", "master"),
    (r"\bguys\b", "friends"), (r"\bkidding\b", "jesting"),
    (r"\bcops\b", "watchmen"), (r"\bpolice\b", "watchmen"),
]


def _apply_swaps(text: str, swaps: list) -> str:
    for pattern, replacement in swaps:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def register_rewrite(text: str) -> str:
    """Rule-based archaic-voice rewrite — see module docstring. Deterministic,
    no LLM/API call. Order: contractions -> bare you/your/yours pronouns ->
    thou-verb irregular fixups -> vocabulary swaps."""
    text = _apply_swaps(text, CONTRACTIONS)
    text = re.sub(r"\byourself\b", "thyself", text, flags=re.IGNORECASE)
    text = re.sub(r"\byours\b", "thine", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "thy", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "thou", text, flags=re.IGNORECASE)
    text = _apply_swaps(text, THOU_VERB_FIXUPS)
    text = _apply_swaps(text, VOCAB_SWAPS)
    # capitalize the first letter of the string and after sentence-ending punctuation
    text = re.sub(r"(^\s*|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


DIALECT_PATTERNS = {
    "thee": r"\bthee\b", "thou": r"\bthou\b", "thy": r"\bthy\b", "thine": r"\bthine\b",
    "dost": r"\bdost\b", "hath": r"\bhath\b", "hast": r"\bhast\b", "wast": r"\bwast\b",
    "doth": r"\bdoth\b", "wilt": r"\bwilt\b", "nay": r"\bnay\b", "art": r"\bart\b",
    "tis": r"'tis\b", "thyself": r"\bthyself\b",
}


def extract_features(text: str) -> list:
    return [feat for feat, pattern in DIALECT_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]


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
    return kept


def build_entries(kept_rows: list, next_id: int) -> list:
    """One entry per row — the first dialogue pair only, to avoid
    oversampling a single character/scene across many near-duplicate turns."""
    entries = []
    for row in kept_rows:
        first_pair = row["pairs"][0]
        input_text = register_rewrite(first_pair["input"])
        output_text = register_rewrite(first_pair["output"])
        archetype = row["archetype"]

        entries.append({
            "id": f"CHM-{next_id:04d}",
            "input": input_text,
            "output": output_text,
            "persona": {
                "archetype": archetype,
                "name": row["name"],
                "disposition": "neutral",
                "social_class": "soldier" if archetype == "guard" else archetype,
            },
            "context": {"location": "unspecified", "time_of_day": "unspecified", "world_state": "medieval"},
            "linguistic_markers": {
                "formality": "high" if archetype in ("noble", "clergy", "scholar") else "low",
                "dialect_features": extract_features(output_text),
                "vocabulary_tier": "elevated" if archetype == "noble" else "mixed",
            },
            "metadata": {
                "intent": "social",
                "persona_stress_test": False,
                "stress_test_type": None,
                "source": "chimbiwide",
                "quality_score": 5,
                "tags": ["chimbiwide", archetype, "register_rewritten", "auto_extracted"],
                "conversion_note": "Rule-based archaic rewrite (no LLM) — grammar/register imperfect, review before publication use.",
            },
        })
        next_id += 1
    return entries


def merge_into_dataset(entries: list):
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    data["entries"].extend(entries)
    data["metadata"]["total_pairs"] = len(data["entries"])
    data["metadata"]["sources"]["chimbiwide"] = {
        "pairs_extracted": len(entries),
        "method": "HF chimbiwide/NPC-Dialogue_v2, medieval-plausibility filter + rule-based register rewrite (no LLM)",
    }
    OUT_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(entries)} entries -> {OUT_PATH}, total now {len(data['entries'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--merge", action="store_true", help="rewrite + merge into medieval_npc_dataset.json")
    parser.add_argument("--max-entries", type=int, default=None, help="cap merged entries")
    args = parser.parse_args()

    kept = convert(args.limit)
    if not args.merge:
        print("(pass --merge to write these into the dataset)")
        return

    if args.max_entries and len(kept) > args.max_entries:
        kept = kept[: args.max_entries]

    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    next_id = len(data["entries"]) + 1
    entries = build_entries(kept, next_id)
    merge_into_dataset(entries)


if __name__ == "__main__":
    main()
