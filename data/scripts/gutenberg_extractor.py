"""
Extract medieval NPC dialogue pairs from Project Gutenberg play texts.

Input:  data/raw/gutenberg/{play}.txt  (Shakespeare play, plain text)
Output: data/processed/medieval_npc_dataset.json (schema v1.0, GUT-#### entries)

Usage:
    python gutenberg_extractor.py --plays hamlet macbeth caesar
"""

import argparse
import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / "gutenberg"
OUT_PATH = Path(__file__).resolve().parents[1] / "processed" / "medieval_npc_dataset.json"

PLAY_META = {
    "hamlet": "Project Gutenberg — Hamlet (Shakespeare, 1599)",
    "macbeth": "Project Gutenberg — Macbeth (Shakespeare, 1606)",
    "caesar": "Project Gutenberg — Julius Caesar (Shakespeare, 1599)",
    "canterbury": "Project Gutenberg — The Canterbury Tales (Chaucer, c.1400, Purves ed.)",
}

# Sources parsed as verse/frame-narrative (quoted-speech extraction) rather than play speaker-cues.
POEM_SOURCES = {"canterbury"}

# Tale-teller -> archetype, keyed by the header text (lowercased, stripped of "the"/"'s tale").
TALE_TELLER_ARCHETYPE = {
    "knight": "noble", "miller": "peasant", "reeve": "peasant", "cook": "peasant",
    "man of law": "scholar", "wife of bath": "merchant", "friar": "clergy",
    "sompnour": "clergy", "clerk": "scholar", "merchant": "merchant",
    "squire": "noble", "franklin": "merchant", "doctor": "scholar",
    "pardoner": "clergy", "prioress": "clergy", "monk": "clergy",
    "nun's priest": "clergy", "manciple": "peasant", "parson": "clergy",
}

TALE_HEADER = re.compile(r"^[A-Z][A-Z’ .<>0-9]*TALE[A-Z’ .<>0-9]*$")
QUOTE_SPAN = re.compile(r"“([^”]+)”", re.DOTALL)

# Speaker abbreviation -> archetype. Extend as new plays are added.
ARCHETYPE_MAP = {
    "ham": "noble", "hamlet": "noble", "king": "noble", "queen": "noble",
    "claud": "noble", "laer": "noble", "oph": "noble", "pol": "noble",
    "hor": "scholar",
    "bar": "guard", "fran": "guard", "mar": "guard", "guard": "guard",
    "clown": "peasant", "grave": "peasant", "1 clo": "peasant", "2 clo": "peasant",
    "mac": "noble", "macb": "noble", "lady": "noble", "banquo": "noble",
    "duncan": "noble", "malcolm": "noble", "porter": "peasant", "murderer": "peasant",
    "caes": "noble", "brutus": "noble", "cassius": "noble", "antony": "noble",
    "cinna": "scholar", "cobler": "peasant", "carpenter": "peasant",
    "cit": "peasant", "soldier": "guard", "sentry": "guard",
    "inn": "innkeeper", "host": "innkeeper",
}

DIALECT_PATTERNS = {
    "thee": r"\bthee\b", "thou": r"\bthou\b", "thy": r"\bthy\b",
    "dost": r"\bdost\b", "hath": r"\bhath\b", "hast": r"\bhast\b",
    "doth": r"\bdoth\b", "wilt": r"\bwilt\b", "nay": r"\bnay\b",
    "art": r"\bart\b", "tis": r"'tis\b", "prithee": r"\bprithee\b",
    "wherefore": r"\bwherefore\b", "forsooth": r"\bforsooth\b",
}

SPEAKER_LINE = re.compile(r"^([A-Z][A-Za-z0-9 .']{1,20})\.\s*(.*)$")


def archetype_for(speaker: str) -> str:
    key = speaker.strip().lower().rstrip(".")
    for prefix, archetype in ARCHETYPE_MAP.items():
        if key.startswith(prefix):
            return archetype
    return "peasant"


def extract_features(text: str) -> list:
    found = []
    for feat, pattern in DIALECT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(feat)
    return found


def quality_score(input_line: str, output_line: str) -> int:
    score = 3
    if 15 <= len(output_line) <= 200:
        score += 1
    if extract_features(output_line):
        score += 1
    if len(output_line.split()) >= 4:
        score += 1
    return min(score, 10)


def parse_turns(raw_text: str):
    """Yield (speaker, line) tuples from a Gutenberg play body."""
    turns = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = SPEAKER_LINE.match(line)
        if match and len(match.group(1).split()) <= 3:
            speaker, rest = match.group(1), match.group(2)
            if rest:
                turns.append((speaker, rest))
    return turns


def tale_key_for(header: str) -> str:
    key = header.lower().replace("the ", "", 1).replace("’s tale", "").replace("'s tale", "")
    key = re.sub(r"<\d+>", "", key).strip(" .")
    return key


def parse_poem_dialogue(raw_text: str):
    """Extract quoted-speech lines from a frame-narrative poem, tagged by
    the enclosing tale's teller (used as an archetype proxy)."""
    lines = raw_text.splitlines()
    header_positions = [
        (i, tale_key_for(line.strip()))
        for i, line in enumerate(lines)
        if TALE_HEADER.match(line.strip()) and tale_key_for(line.strip()) in TALE_TELLER_ARCHETYPE
    ]

    if not header_positions:
        return []
    body_start_line = header_positions[0][0]

    quotes = []  # (archetype, text)
    for match in QUOTE_SPAN.finditer(raw_text):
        line_idx = raw_text[: match.start()].count("\n")
        if line_idx < body_start_line:
            continue  # skip preface / table of contents / front matter

        archetype = TALE_TELLER_ARCHETYPE[header_positions[0][1]]
        for pos, key in header_positions:
            if pos <= line_idx:
                archetype = TALE_TELLER_ARCHETYPE[key]
            else:
                break

        text = re.sub(r"\*[^*]*\*", " ", match.group(1))  # strip Purves glosses like *word*, keep word boundary
        text = re.sub(r"\s+", " ", text).strip().strip(",;: ")
        word_count = len(text.split())
        if word_count < 3 or word_count > 60:
            continue  # drop fragments and run-on multi-speech blocks
        if "�" in text or not re.search(r"[a-zA-Z]{2,}", text):
            continue  # drop mojibake / encoding-corrupted lines
        quotes.append((archetype, text))
    return quotes


def extract_pairs(play: str, min_quality: int = 4):
    raw_path = RAW_DIR / f"{play}.txt"
    raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")

    if play in POEM_SOURCES:
        quotes = parse_poem_dialogue(raw_text)
        pairs, skipped = [], 0
        for (arch_a, text_a), (arch_b, text_b) in zip(quotes, quotes[1:]):
            score = quality_score(text_a, text_b)
            if score < min_quality:
                skipped += 1
                continue
            pairs.append(
                {
                    "input": text_a,
                    "output": text_b,
                    "archetype": arch_b,
                    "quality_score": score,
                    "dialect_features": extract_features(text_b),
                }
            )
        return pairs, skipped, len(quotes)

    turns = parse_turns(raw_text)
    pairs, skipped = [], 0
    for (spk_a, line_a), (spk_b, line_b) in zip(turns, turns[1:]):
        if spk_a == spk_b:
            continue
        score = quality_score(line_a, line_b)
        if score < min_quality:
            skipped += 1
            continue
        pairs.append(
            {
                "input": line_a,
                "output": line_b,
                "speaker": spk_b,
                "quality_score": score,
                "dialect_features": extract_features(line_b),
            }
        )
    return pairs, skipped, len(turns)


def build_entry(play: str, idx: int, pair: dict) -> dict:
    archetype = pair["archetype"] if "archetype" in pair else archetype_for(pair["speaker"])
    return {
        "id": f"GUT-{idx:04d}",
        "input": pair["input"],
        "output": pair["output"],
        "persona": {
            "archetype": archetype,
            "name": pair.get("speaker", archetype),
            "disposition": "neutral",
            "social_class": "soldier" if archetype == "guard" else archetype,
        },
        "context": {
            "location": "unspecified",
            "time_of_day": "unspecified",
            "world_state": "medieval",
        },
        "linguistic_markers": {
            "formality": "high" if archetype in ("noble", "clergy", "scholar") else "low",
            "dialect_features": pair["dialect_features"],
            "vocabulary_tier": "elevated" if archetype == "noble" else "mixed",
        },
        "metadata": {
            "intent": "social",
            "persona_stress_test": False,
            "stress_test_type": None,
            "source": PLAY_META.get(play, f"Project Gutenberg — {play.title()}"),
            "source_play": play,
            "quality_score": pair["quality_score"],
            "tags": ["gutenberg", "public_domain", "chaucer" if play in POEM_SOURCES else "shakespeare", play, archetype, "auto_extracted"],
            "conversion_note": "Auto-extracted. Review: location, world_state. Verify archetype mapping.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plays", nargs="+", default=["hamlet", "macbeth", "caesar"])
    parser.add_argument("--min-quality", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=None, help="cap pairs kept per play, highest quality first")
    args = parser.parse_args()

    existing = {"metadata": {"sources": {}}, "entries": []}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    entries = existing["entries"]
    next_id = len(entries) + 1
    sources = existing["metadata"].get("sources", {})

    for play in args.plays:
        if play in sources:
            print(f"skip {play}: already extracted")
            continue
        pairs, skipped, raw_turns = extract_pairs(play, args.min_quality)
        if args.max_pairs and len(pairs) > args.max_pairs:
            pairs.sort(key=lambda p: p["quality_score"], reverse=True)
            skipped += len(pairs) - args.max_pairs
            pairs = pairs[: args.max_pairs]
        archetype_dist, intent_dist = {}, {}
        for pair in pairs:
            entry = build_entry(play, next_id, pair)
            entries.append(entry)
            next_id += 1
            a = entry["persona"]["archetype"]
            archetype_dist[a] = archetype_dist.get(a, 0) + 1
            i = entry["metadata"]["intent"]
            intent_dist[i] = intent_dist.get(i, 0) + 1

        avg_q = round(sum(p["quality_score"] for p in pairs) / len(pairs), 2) if pairs else 0
        sources[play] = {
            "raw_turns": raw_turns,
            "pairs_extracted": len(pairs),
            "pairs_skipped": skipped,
            "avg_quality_score": avg_q,
            "archetype_distribution": archetype_dist,
            "intent_distribution": intent_dist,
        }
        print(f"{play}: {len(pairs)} pairs extracted, {skipped} skipped")

    output = {
        "metadata": {
            "description": "Medieval NPC dialogue pairs — Project Gutenberg Shakespeare",
            "license": "Public Domain",
            "schema_version": "1.0",
            "total_pairs": len(entries),
            "min_quality_threshold": args.min_quality,
            "sources": sources,
            "todo": existing["metadata"].get(
                "todo",
                [
                    "Fill 'unspecified' time_of_day fields",
                    "Verify archetype labels for abbreviated speaker names (Ham, Mac, etc.)",
                    "Add persona_stress_test entries manually",
                    "Combine with hand-authored MED-0001..0010 entries",
                ],
            ),
        },
        "entries": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(entries)} total entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
