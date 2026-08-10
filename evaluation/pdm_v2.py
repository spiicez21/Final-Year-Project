"""
PDM v2 — domain-agnostic Persona Drift Metric (Docs/TODO.md Week 5).

PDM v1 (pdm_scorer.py) is a single archaic-English word list — it produces
no signal on modern dialogue (nothing in "gonna, what's the fastest way to
report a stolen car" matches thee/thou/hath). PDM v2 replaces the single
word-list with five cheap, judge-free feature families:

  1. lexicon    — per-archetype distinctive content vocabulary, computed
                   from the real training data (TF-IDF-style), not hand-picked.
  2. register    — fixed informal/formal function-word markers (independent
                   of archetype).
  3. formality   — contraction-density bucket (high/medium/low).
  4. syntax      — sentence-length bucket + whether the response asks a question.
  5. stance      — hedging language present / assertive language present.

Two DIFFERENT comparison methods, not one Jaccard applied uniformly —
this is the real lesson from two rounds of calibration (2026-08-08):

  - lexicon/register are open-vocabulary sets (many distinct words CAN be
    "this archetype's vocabulary") — Jaccard against a unioned reference set
    is the right comparison, same as PDM v1 used.
  - formality/syntax/stance are small-cardinality CATEGORICAL attributes
    (formality has exactly 3 possible values). Union-across-training-data
    for a 3-valued attribute hits all 3 values almost immediately, so
    Jaccard(one response's bucket, {all 3 buckets}) is a CONSTANT regardless
    of what the response actually says — measures nothing. These three
    families instead compare against the archetype's MODE (most common
    value per attribute) and score match/no-match, not set overlap.

First version pooled everything into one flat Jaccard (short correct
responses scored worse than long rambling ones, since they had fewer
feature slots to hit against one big reference set). Second version kept
Jaccard but split it per family (fixed the pooling bias, but the three
categorical families still measured nothing, since a mode/no-mode question
was being asked as a set-overlap question). This is the third version.

Calibration rule (Docs/TODO.md): must be calibrated against real
generations from a trained adapter, not against the dataset text alone —
the PDM v1 baseline (mean drift 0.9833, flat, saturated) is what happens
when a metric never sees real model output before being trusted.
"""

import json
import re
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modern_npc_dataset.json"

STOPWORDS = set(
    "the a an is are was were be been being to of and or but in on at for with i you he she it we they "
    "this that these those my your his her its our their not no do does did have has had will would can "
    "could should shall may might just so if then there here what sure let like well tell about more why "
    "know get got going go want need think going gonna out up down over".split()
)

LEXICON_TOP_N = 15
LEXICON_MIN_COUNT = 3

REGISTER_MARKERS = {
    "informal": ["gonna", "wanna", "ain't", "gotta", "lemme", "gimme", "dunno", "nah", "yeah", "kinda", "sorta", "yep"],
    "formal": ["therefore", "furthermore", "hereby", "pursuant", "shall", "whom", "regarding", "nonetheless"],
}

HEDGES = [r"\bmaybe\b", r"\bperhaps\b", r"\bi think\b", r"\bmight\b", r"\bpossibly\b", r"\bi guess\b", r"\bsort of\b"]
ASSERTIVES = [r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b", r"\balways\b", r"\bnever\b", r"\bwill\b"]

CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# lexicon/register: open-vocabulary, compared via Jaccard-of-sets.
# formality/syntax/stance: small-cardinality categorical, compared via
# mode-match (does this response's attribute value match the archetype's
# most common value for that attribute?).
FAMILY_KIND = {
    "lexicon": "set",
    "register": "set",
    "formality": "attrs",
    "syntax": "attrs",
    "stance": "attrs",
}
FAMILIES = tuple(FAMILY_KIND.keys())


def _tokenize(text: str) -> list:
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOPWORDS and len(w) > 2 and "'" not in w]


def build_archetype_lexicons(entries: list = None, top_n: int = LEXICON_TOP_N) -> dict:
    """TF-IDF-style: score = raw count in this archetype's own entries,
    discounted by how many OTHER archetypes also use the word — a word
    every archetype says a lot (e.g. "help") scores low; a word only one
    archetype says a lot (e.g. "prescription" for pharmacist) scores high."""
    if entries is None:
        entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]

    by_archetype = {}
    for e in entries:
        arch = e["persona"]["archetype"]
        by_archetype.setdefault(arch, Counter()).update(_tokenize(e["output"]))

    lexicons = {}
    for arch, counts in by_archetype.items():
        scored = []
        for word, cnt in counts.items():
            if cnt < LEXICON_MIN_COUNT:
                continue
            df_other = sum(1 for a2, c2 in by_archetype.items() if a2 != arch and c2.get(word, 0) > 0)
            scored.append((cnt / (1 + df_other), word))
        scored.sort(reverse=True)
        lexicons[arch] = {word for _, word in scored[:top_n]}
    return lexicons


def _lexicon_set(text: str, archetype: str, lexicons: dict) -> set:
    lexicon = lexicons.get(archetype, set())
    return {f"lex:{token}" for token in _tokenize(text) if token in lexicon}


def _register_set(text_lower: str) -> set:
    features = set()
    for register, markers in REGISTER_MARKERS.items():
        for marker in markers:
            if re.search(rf"\b{re.escape(marker)}\b", text_lower):
                features.add(f"register:{register}:{marker}")
    return features


def _formality_attrs(text: str, words: list) -> dict:
    if not words:
        bucket = "low"
    else:
        density = len(CONTRACTION_RE.findall(text)) / len(words)
        bucket = "low" if density > 0.06 else "medium" if density > 0.02 else "high"
    return {"bucket": bucket}


def _syntax_attrs(text: str, words: list) -> dict:
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        length_bucket = "short"
    else:
        avg_len = len(words) / len(sentences)
        length_bucket = "short" if avg_len < 8 else "medium" if avg_len < 16 else "long"
    return {"length": length_bucket, "has_question": "?" in text}


def _stance_attrs(text_lower: str) -> dict:
    return {
        "hedging": any(re.search(p, text_lower) for p in HEDGES),
        "assertive": any(re.search(p, text_lower) for p in ASSERTIVES),
    }


def extract_features_v2(text: str, archetype: str, lexicons: dict) -> dict:
    """Returns {family_name: value}, where value is a set for
    lexicon/register or an attrs dict for formality/syntax/stance."""
    text_lower = text.lower()
    words = text.split()
    return {
        "lexicon": _lexicon_set(text, archetype, lexicons),
        "register": _register_set(text_lower),
        "formality": _formality_attrs(text, words),
        "syntax": _syntax_attrs(text, words),
        "stance": _stance_attrs(text_lower),
    }


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _attr_match_fraction(response_attrs: dict, mode_attrs: dict) -> float:
    matches = [response_attrs[k] == mode_attrs[k] for k in mode_attrs]
    return sum(matches) / len(matches) if matches else 1.0


def build_reference_features(archetype: str, entries: list, lexicons: dict) -> dict:
    """Reference for lexicon/register: union of that family's features
    across the archetype's own training entries (set). Reference for
    formality/syntax/stance: a per-attribute PROBABILITY DISTRIBUTION over
    values across those same entries, not a single mode. Binary mode-match
    was tried and calibrated badly (2026-08-08): police officer's formality
    is genuinely bimodal in the real data (high: 62, low: 53, medium: 19,
    out of 134) — "high" barely edges out "low" as the mode, so treating
    "low" as flatly wrong discarded 40% of the archetype's own legitimate
    variation. A distribution lets a response landing in a substantial
    secondary bucket still score reasonably, while landing in a rare bucket
    (the 19 "medium" ones) still scores worse — proportional, not binary."""
    matching = [e for e in entries if e["persona"]["archetype"] == archetype]

    ref = {}
    ref["lexicon"] = set()
    ref["register"] = set()
    attr_votes = {"formality": Counter(), "syntax": {"length": Counter(), "has_question": Counter()},
                  "stance": {"hedging": Counter(), "assertive": Counter()}}

    for e in matching:
        feats = extract_features_v2(e["output"], archetype, lexicons)
        ref["lexicon"] |= feats["lexicon"]
        ref["register"] |= feats["register"]
        attr_votes["formality"][feats["formality"]["bucket"]] += 1
        attr_votes["syntax"]["length"][feats["syntax"]["length"]] += 1
        attr_votes["syntax"]["has_question"][feats["syntax"]["has_question"]] += 1
        attr_votes["stance"]["hedging"][feats["stance"]["hedging"]] += 1
        attr_votes["stance"]["assertive"][feats["stance"]["assertive"]] += 1

    def _typicality_dist(counter: Counter) -> dict:
        # value -> (observed frequency) / (frequency of the most common
        # value) — landing exactly on the mode still scores 1.0, landing on
        # a substantial secondary value scores proportionally, landing on
        # something rare scores low. Empty (no training data) -> neutral 0.5
        # for anything, rather than crashing or silently defaulting to a
        # made-up mode.
        if not counter:
            return {}
        peak = counter.most_common(1)[0][1]
        return {value: count / peak for value, count in counter.items()}

    ref["formality"] = {"bucket": _typicality_dist(attr_votes["formality"])}
    ref["syntax"] = {
        "length": _typicality_dist(attr_votes["syntax"]["length"]),
        "has_question": _typicality_dist(attr_votes["syntax"]["has_question"]),
    }
    ref["stance"] = {
        "hedging": _typicality_dist(attr_votes["stance"]["hedging"]),
        "assertive": _typicality_dist(attr_votes["stance"]["assertive"]),
    }
    return ref


def _attr_typicality(response_attrs: dict, reference_dists: dict) -> float:
    """Average, across this family's attributes, of how typical the
    response's value is for the archetype (1.0 = matches the most common
    value seen in training, 0.0 = never seen, proportional in between)."""
    scores = []
    for attr, dist in reference_dists.items():
        scores.append(dist.get(response_attrs[attr], 0.0) if dist else 0.5)
    return sum(scores) / len(scores) if scores else 1.0


def single_turn_drift_v2(response: str, archetype: str, reference_features: dict, lexicons: dict) -> float:
    """Mean of (1 - similarity) across the five families: Jaccard for
    lexicon/register, distributional typicality for formality/syntax/stance."""
    response_features = extract_features_v2(response, archetype, lexicons)
    family_drifts = []
    for family in FAMILIES:
        if FAMILY_KIND[family] == "set":
            similarity = jaccard(response_features[family], reference_features[family])
        else:
            similarity = _attr_typicality(response_features[family], reference_features[family])
        family_drifts.append(1.0 - similarity)
    return round(sum(family_drifts) / len(family_drifts), 4)
