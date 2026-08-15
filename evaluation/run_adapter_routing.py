"""
Docs/TODO.md 'Evaluation conditions (reference)' table: adapter-routing
accuracy was listed as a required metric (README.md: "Archetype classifier
vs. player input", proposed as the >95%-figure metric for any rubric that
demands one) but never built or measured. This closes that gap.

Judge-free by design, same discipline as pdm_v2.py's lexicon family: builds
a per-archetype TF-IDF-style word lexicon from a TRAIN split of the dataset,
classifies each HELD-OUT input by which archetype's lexicon it overlaps with
most (cosine similarity over lexicon weight vectors), and reports accuracy
against the real gold archetype label already in the dataset. No LLM call,
no adapter inference — this measures whether player input alone carries
enough archetype signal to route correctly, independent of generation
quality.

Usage:
    python evaluation/run_adapter_routing.py
"""

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"

TOKEN_RE = re.compile(r"[a-z']+")
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it", "to", "and", "of", "in",
    "for", "on", "that", "this", "with", "my", "your", "me", "at", "be", "have", "has", "do",
    "did", "does", "so", "if", "just", "can", "will", "would", "there", "here", "what", "how",
    "we", "they", "he", "she", "not", "no", "yes", "really", "hey", "oh", "um", "uh", "or",
}
TRAIN_FRACTION = 0.8
SEED = 42


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]


def build_lexicon_vectors(train_entries: list) -> dict:
    """Per-archetype TF-IDF-style word weights, same method as
    pdm_v2.build_archetype_lexicons but built here directly on `input` text
    (the player-side utterance) instead of `output` (the NPC's persona
    voice) — routing has to work from what the PLAYER said, before any
    archetype has spoken."""
    archetype_docs = {}
    for e in train_entries:
        arch = e["persona"]["archetype"]
        archetype_docs.setdefault(arch, []).extend(tokenize(e["input"]))

    doc_freq = Counter()
    for arch, tokens in archetype_docs.items():
        for word in set(tokens):
            doc_freq[word] += 1
    n_archetypes = len(archetype_docs)

    vectors = {}
    for arch, tokens in archetype_docs.items():
        term_freq = Counter(tokens)
        total = sum(term_freq.values()) or 1
        vec = {}
        for word, count in term_freq.items():
            tf = count / total
            idf = math.log(n_archetypes / (1 + doc_freq[word])) + 1
            vec[word] = tf * idf
        vectors[arch] = vec
    return vectors


def classify(text: str, vectors: dict) -> str:
    tokens = Counter(tokenize(text))
    best_arch, best_score = None, -1.0
    for arch, vec in vectors.items():
        score = sum(vec.get(word, 0.0) * count for word, count in tokens.items())
        if score > best_score:
            best_score, best_arch = score, arch
    return best_arch


def main():
    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]

    by_archetype = {}
    for e in entries:
        by_archetype.setdefault(e["persona"]["archetype"], []).append(e)

    rng = random.Random(SEED)
    train_entries, test_entries = [], []
    for arch, group in by_archetype.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        split = int(len(shuffled) * TRAIN_FRACTION)
        train_entries.extend(shuffled[:split])
        test_entries.extend(shuffled[split:])

    print(f"train: {len(train_entries)}  test (held out): {len(test_entries)}")
    vectors = build_lexicon_vectors(train_entries)
    for arch, vec in vectors.items():
        print(f"  {arch:15s} lexicon size={len(vec)}")

    correct = 0
    confusion = Counter()
    per_archetype = Counter()
    per_archetype_correct = Counter()
    for e in test_entries:
        gold = e["persona"]["archetype"]
        pred = classify(e["input"], vectors)
        per_archetype[gold] += 1
        if pred == gold:
            correct += 1
            per_archetype_correct[gold] += 1
        else:
            confusion[(gold, pred)] += 1

    accuracy = correct / len(test_entries)
    print(f"\n--- per-archetype accuracy (held-out test set) ---")
    for arch in sorted(per_archetype):
        n = per_archetype[arch]
        c = per_archetype_correct[arch]
        print(f"  {arch:15s} {c}/{n} = {c/n:.1%}")

    print(f"\noverall adapter-routing accuracy: {correct}/{len(test_entries)} = {accuracy:.1%}")

    if confusion:
        print(f"\ntop confusions (gold -> predicted):")
        for (gold, pred), count in confusion.most_common(10):
            print(f"  {gold} -> {pred}: {count}")

    out_path = Path(__file__).resolve().parent / "results" / "adapter_routing_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "accuracy": accuracy, "correct": correct, "total": len(test_entries),
        "per_archetype": {a: per_archetype_correct[a] / per_archetype[a] for a in per_archetype},
        "confusion": [{"gold": g, "predicted": p, "count": c} for (g, p), c in confusion.most_common()],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
