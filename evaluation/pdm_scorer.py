"""
Persona Drift Metric (PDM) — reference implementation from
DevFiles/Specs.md Appendix A. This is PDM v1: a single archaic-English
dialect-marker lexicon, medieval-only.

PDM(C) = 1 - (1/N) * Sum sim(dialect_features(t_i), reference_feature_set)

0.0 = no drift (perfect persona consistency), 1.0 = complete drift/collapse.

PDM v2 (Docs/TODO.md Week 5) replaces the lexicon-matching approach for the
modern domain with domain-agnostic feature families (lexicon + register
markers + formality score + syntactic profile + stance), calibrated against
real generations from a trained adapter rather than against dataset text —
not a word-list swap. Not built yet. A prior lexicon-swap attempt
(DIALECT_PATTERNS_MODERN, crime-city slang) predated that decision and has
been removed rather than left as a wrong stand-in for PDM v2.
"""

import re

DIALECT_PATTERNS_MEDIEVAL = {
    "thee": r"\bthee\b", "thou": r"\bthou\b", "thy": r"\bthy\b",
    "dost": r"\bdost\b", "hath": r"\bhath\b", "hast": r"\bhast\b",
    "doth": r"\bdoth\b", "wilt": r"\bwilt\b", "nay": r"\bnay\b",
    "art": r"\bart\b", "tis": r"\b'tis\b", "prithee": r"\bprithee\b",
    "wherefore": r"\bwherefore\b", "forsooth": r"\bforsooth\b",
}

DIALECT_PATTERNS_BY_DOMAIN = {
    "medieval": DIALECT_PATTERNS_MEDIEVAL,
}

# Default/back-compat alias — existing call sites (run_baseline.py,
# run_condition_b.py, run_stress_test.py, backend/main.py) import
# DIALECT_PATTERNS directly and call extract_features(text) with no domain
# arg; all of that keeps working unchanged, scoped to medieval.
DIALECT_PATTERNS = DIALECT_PATTERNS_MEDIEVAL


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_features(text: str, patterns: dict = DIALECT_PATTERNS) -> set:
    found = set()
    for feat, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.add(feat)
    return found


def compute_pdm(conversation_turns: list, reference_features: set, patterns: dict = DIALECT_PATTERNS) -> float:
    """PDM over a multi-turn conversation (list of NPC output strings)."""
    similarities = []
    for turn in conversation_turns:
        turn_features = extract_features(turn, patterns)
        similarities.append(jaccard(turn_features, reference_features))
    avg_similarity = sum(similarities) / len(similarities)
    return round(1.0 - avg_similarity, 4)


def single_turn_drift(response: str, reference_features: set, patterns: dict = DIALECT_PATTERNS) -> float:
    """Single-turn proxy: 1 - jaccard(response_features, reference_features).
    Used for baseline (non-conversational) evaluation where each dataset
    entry is an isolated prompt/response pair rather than a multi-turn log."""
    return round(1.0 - jaccard(extract_features(response, patterns), reference_features), 4)
