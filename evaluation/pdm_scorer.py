"""
Persona Drift Metric (PDM) — reference implementation from
DevFiles/Specs.md Appendix A.

PDM(C) = 1 - (1/N) * Sum sim(archaic_features(t_i), reference_feature_set)

0.0 = no drift (perfect persona consistency), 1.0 = complete drift/collapse.
"""

import re

DIALECT_PATTERNS = {
    "thee": r"\bthee\b", "thou": r"\bthou\b", "thy": r"\bthy\b",
    "dost": r"\bdost\b", "hath": r"\bhath\b", "hast": r"\bhast\b",
    "doth": r"\bdoth\b", "wilt": r"\bwilt\b", "nay": r"\bnay\b",
    "art": r"\bart\b", "tis": r"\b'tis\b", "prithee": r"\bprithee\b",
    "wherefore": r"\bwherefore\b", "forsooth": r"\bforsooth\b",
}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_features(text: str) -> set:
    found = set()
    for feat, pattern in DIALECT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.add(feat)
    return found


def compute_pdm(conversation_turns: list, reference_features: set) -> float:
    """PDM over a multi-turn conversation (list of NPC output strings)."""
    similarities = []
    for turn in conversation_turns:
        turn_features = extract_features(turn)
        similarities.append(jaccard(turn_features, reference_features))
    avg_similarity = sum(similarities) / len(similarities)
    return round(1.0 - avg_similarity, 4)


def single_turn_drift(response: str, reference_features: set) -> float:
    """Single-turn proxy: 1 - jaccard(response_features, reference_features).
    Used for baseline (non-conversational) evaluation where each dataset
    entry is an isolated prompt/response pair rather than a multi-turn log."""
    return round(1.0 - jaccard(extract_features(response), reference_features), 4)
