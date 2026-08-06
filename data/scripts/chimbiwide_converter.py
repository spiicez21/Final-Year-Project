"""
Convert chimbiwide/NPC-Dialogue_v2 (HuggingFace) into the project's NPC
dialogue schema (see DevFiles/Specs.md section 6). Supports two domains:

  --domain medieval (default): filter to medieval-plausible entries, then
  rewrite into archaic voice via register_rewrite() — a deterministic
  lexical/grammatical rewriter, not an LLM call. It is intentionally modest:
  contraction expansion, you/your -> thee/thy/thou with irregular-verb
  fixups, small vocab swaps. Won't produce Shakespeare-quality prose, just
  "good enough to not read as a phone-and-wifi contemporary chatlog," tagged
  auto-rewritten for later review. Specs.md flags this source as medium IP
  risk — do not use rewritten entries beyond local training/eval without a
  further scrub pass.

  --domain modern: chimbiwide's raw dialogue is already casual/contemporary
  in register (that's *why* it needed the archaic rewrite above) — so modern
  mode skips register_rewrite entirely and uses the text as-is, just with a
  different archetype remap (fantasy roleplay bios -> crime-city archetypes)
  and a fantasy-leakage filter (drop entries too magic/kingdom-flavored to
  read as a modern setting) instead of the medieval one.

Pipeline: download -> filter -> (rewrite if medieval) -> merge.

Usage:
    python chimbiwide_converter.py --domain medieval --limit 300 --merge
    python chimbiwide_converter.py --domain modern --limit 300 --merge
"""

import argparse
import json
import re
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "processed"
CACHE_DIR = Path(__file__).resolve().parents[1] / "raw" / "huggingface" / "chimbiwide"

OUT_PATHS = {
    "medieval": PROCESSED_DIR / "medieval_npc_dataset.json",
    "modern": PROCESSED_DIR / "modern_npc_dataset.json",
}

# Background-blurb keyword -> archetype, per domain. Order matters (first match wins).
ARCHETYPE_REMAP_MEDIEVAL = {
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

# Same fantasy-roleplay bios, remapped onto the modern crime-city archetype
# set (see Docs/TODO.md for the medieval->modern archetype mapping table).
ARCHETYPE_REMAP_MODERN = {
    "bounty hunter": "cop", "knight": "cop", "soldier": "cop", "guard": "cop", "mercenary": "cop",
    "smuggler": "dealer", "shopkeeper": "dealer", "trader": "dealer", "merchant": "dealer",
    "assassin": "boss", "king": "boss", "queen": "boss", "lord": "boss", "lady": "boss", "noble": "boss",
    "wizard": "lawyer", "sage": "lawyer", "professor": "lawyer", "scholar": "lawyer",
    "tavern": "bartender", "innkeeper": "bartender", "bartender": "bartender",
    "healer": "mechanic", "alchemist": "mechanic", "herbalist": "mechanic",
    "priest": "preacher", "monk": "preacher", "clergy": "preacher",
    "villager": "civilian", "farmer": "civilian", "thief": "civilian",
}

ARCHETYPE_REMAPS = {"medieval": ARCHETYPE_REMAP_MEDIEVAL, "modern": ARCHETYPE_REMAP_MODERN}
DEFAULT_ARCHETYPES = {"medieval": "peasant", "modern": "civilian"}

# Terms that mark a line as out-of-period for the MEDIEVAL domain — unusable
# without heavy rewrite. Not applied to the modern domain (this vocabulary
# is exactly what modern dialogue should contain).
MODERN_LEAKAGE = [
    "phone", "computer", "internet", "email", "wifi", "gun",
    "okay", "gonna", "wanna", "rupees", "police",
]

# Terms that mark a line as too fantasy-flavored for the MODERN domain.
FANTASY_LEAKAGE = [
    "dragon", "wizard", "spell", "magic", "kingdom", "castle",
    "elf", "orc", "sorcery", "enchant", "potion", "sword",
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


def is_domain_plausible(text: str, domain: str) -> bool:
    lowered = text.lower()
    leakage = MODERN_LEAKAGE if domain == "medieval" else FANTASY_LEAKAGE
    return not any(re.search(rf"\b{term}\b", lowered) for term in leakage)


def remap_archetype(background: str, domain: str) -> str:
    lowered = background.lower()
    for keyword, archetype in ARCHETYPE_REMAPS[domain].items():
        if keyword in lowered:
            return archetype
    return DEFAULT_ARCHETYPES[domain]


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


# Medieval list kept exactly as originally shipped (superset of
# evaluation/pdm_scorer.py's DIALECT_PATTERNS_MEDIEVAL — thine/wast/thyself
# included here) — not touched, so already-built medieval entries' recorded
# dialect_features stay reproducible.
DIALECT_PATTERNS_MEDIEVAL = {
    "thee": r"\bthee\b", "thou": r"\bthou\b", "thy": r"\bthy\b", "thine": r"\bthine\b",
    "dost": r"\bdost\b", "hath": r"\bhath\b", "hast": r"\bhast\b", "wast": r"\bwast\b",
    "doth": r"\bdoth\b", "wilt": r"\bwilt\b", "nay": r"\bnay\b", "art": r"\bart\b",
    "tis": r"'tis\b", "thyself": r"\bthyself\b",
}

# Mirrors evaluation/pdm_scorer.py's DIALECT_PATTERNS_MODERN — keep in sync
# if either changes, so recorded dialect_features match what eval scores.
DIALECT_PATTERNS_MODERN = {
    "gonna": r"\bgonna\b", "wanna": r"\bwanna\b", "ain't": r"\bain'?t\b",
    "gotta": r"\bgotta\b", "lemme": r"\blemme\b", "gimme": r"\bgimme\b",
    "dunno": r"\bdunno\b", "nah": r"\bnah\b", "yo": r"\byo\b",
    "bro": r"\bbro\b", "homie": r"\bhomie\b", "finna": r"\bfinna\b",
    "kinda": r"\bkinda\b", "sorta": r"\bsorta\b",
}

DIALECT_PATTERNS_BY_DOMAIN = {"medieval": DIALECT_PATTERNS_MEDIEVAL, "modern": DIALECT_PATTERNS_MODERN}

# Archetypes considered "high formality" / "elevated vocabulary", per domain
# — mirrors the medieval->modern archetype mapping (noble -> boss).
HIGH_FORMALITY = {"medieval": ("noble", "clergy", "scholar"), "modern": ("boss", "lawyer", "preacher")}
ELEVATED_VOCAB_ARCHETYPE = {"medieval": "noble", "modern": "boss"}


def extract_features(text: str, domain: str) -> list:
    patterns = DIALECT_PATTERNS_BY_DOMAIN[domain]
    return [feat for feat, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


def convert(limit: int, domain: str):
    rows = load_source(limit)
    kept, dropped, no_pairs = [], 0, 0
    archetype_dist = {}

    for row in rows:
        parsed = parse_row(row)
        if not parsed["pairs"]:
            no_pairs += 1
            continue

        full_text = parsed["background"] + " " + " ".join(p["input"] + " " + p["output"] for p in parsed["pairs"])
        if not is_domain_plausible(full_text, domain):
            dropped += 1
            continue

        archetype = remap_archetype(parsed["background"], domain)
        archetype_dist[archetype] = archetype_dist.get(archetype, 0) + 1
        kept.append({**parsed, "archetype": archetype})

    leakage_label = "modern leakage" if domain == "medieval" else "fantasy leakage"
    print(f"chimbiwide: {len(kept)} {domain}-plausible / {dropped} dropped ({leakage_label}) / {no_pairs} no usable turns")
    print(f"archetype distribution: {archetype_dist}")
    return kept


def build_entries(kept_rows: list, next_id: int, domain: str) -> list:
    """One entry per row — the first dialogue pair only, to avoid
    oversampling a single character/scene across many near-duplicate turns."""
    entries = []
    for row in kept_rows:
        first_pair = row["pairs"][0]
        if domain == "medieval":
            input_text = register_rewrite(first_pair["input"])
            output_text = register_rewrite(first_pair["output"])
        else:
            # Raw chimbiwide dialogue is already casual/contemporary — no
            # rewrite needed for the modern domain.
            input_text = first_pair["input"].strip()
            output_text = first_pair["output"].strip()
        archetype = row["archetype"]

        entries.append({
            "id": f"CHM-{next_id:04d}",
            "input": input_text,
            "output": output_text,
            "persona": {
                "archetype": archetype,
                "name": row["name"],
                "disposition": "neutral",
                "social_class": "soldier" if (domain == "medieval" and archetype == "guard") else archetype,
            },
            "context": {"location": "unspecified", "time_of_day": "unspecified", "world_state": domain},
            "linguistic_markers": {
                "formality": "high" if archetype in HIGH_FORMALITY[domain] else "low",
                "dialect_features": extract_features(output_text, domain),
                "vocabulary_tier": "elevated" if archetype == ELEVATED_VOCAB_ARCHETYPE[domain] else "mixed",
            },
            "metadata": {
                "intent": "social",
                "persona_stress_test": False,
                "stress_test_type": None,
                "source": "chimbiwide",
                "quality_score": 5,
                "tags": ["chimbiwide", archetype] + (["register_rewritten", "auto_extracted"] if domain == "medieval" else ["raw_extracted"]),
                "conversion_note": ("Rule-based archaic rewrite (no LLM) — grammar/register imperfect, review before publication use."
                                     if domain == "medieval" else
                                     "Raw chimbiwide dialogue, no rewrite — archetype remapped from fantasy bio to modern crime-city role."),
            },
        })
        next_id += 1
    return entries


def merge_into_dataset(entries: list, domain: str):
    out_path = OUT_PATHS[domain]
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["entries"].extend(entries)
    data["metadata"]["total_pairs"] = len(data["entries"])
    data["metadata"]["sources"]["chimbiwide"] = {
        "pairs_extracted": len(entries),
        "method": f"HF chimbiwide/NPC-Dialogue_v2, {domain}-plausibility filter"
                  + (" + rule-based register rewrite (no LLM)" if domain == "medieval" else " (raw text, no rewrite)"),
    }
    out_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(entries)} entries -> {out_path}, total now {len(data['entries'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["medieval", "modern"], default="medieval")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--merge", action="store_true", help="convert + merge into the domain's dataset file")
    parser.add_argument("--max-entries", type=int, default=None, help="cap merged entries")
    args = parser.parse_args()

    kept = convert(args.limit, args.domain)
    if not args.merge:
        print("(pass --merge to write these into the dataset)")
        return

    if args.max_entries and len(kept) > args.max_entries:
        kept = kept[: args.max_entries]

    out_path = OUT_PATHS[args.domain]
    data = json.loads(out_path.read_text(encoding="utf-8"))
    next_id = len(data["entries"]) + 1
    entries = build_entries(kept, next_id, args.domain)
    merge_into_dataset(entries, args.domain)


if __name__ == "__main__":
    main()
