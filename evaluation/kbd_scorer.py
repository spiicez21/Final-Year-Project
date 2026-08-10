"""
Knowledge Boundary Drift (KBD) — README Contribution C1, Docs/TODO.md Week 6.

KBD = (factual references to out-of-visibility knowledge items) /
      (total factual references detected in the response)

Judge-free: detection is signature-phrase substring matching against
data/processed/knowledge_base.json, not an LLM call. Visibility resolution
is a flat set intersection (`archetype in item["visible_to"]`) — no
hierarchy, no propagation, no decay, per the explicit Docs/TODO.md
constraint ("not a tree, not Neo4j").

A response with zero detected factual references returns KBD=None, not 0 —
"said nothing checkable" is a different outcome from "said something and it
was fine," and conflating them would let a model dodge every probe with "I
don't know" and score a perfect 0.0 that means nothing (same class of
mistake as PDM v1's saturated 0.9833 baseline: a metric that can't tell
silence from correctness has no signal).
"""

import json
import re
from pathlib import Path

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "knowledge_base.json"

# Cheap, judge-free mitigation for a real precision gap: a response that
# DENIES a fact ("No, I haven't heard about that insulin shipment") still
# contains the signature phrase, and naive substring matching alone would
# count that as a violation — it isn't one, the model correctly claimed
# ignorance while echoing the question's own wording. Not a full NLI
# confirm-vs-deny classifier (that would break the judge-free/cheap design
# constraint) — just checks for a negation cue in the same sentence, before
# the matched phrase. Known limitation, not claimed to be perfect: a denial
# phrased unusually (no negation word) still counts as a match.
NEGATION_CUES = [r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bdon'?t\b", r"\bdoesn'?t\b",
                 r"\bdidn'?t\b", r"\bhaven'?t\b", r"\bhasn'?t\b", r"\bisn'?t\b", r"\bwasn'?t\b",
                 r"\bcan'?t\b", r"\bcouldn'?t\b", r"\bwouldn'?t\b", r"\bnothing\b", r"\bnobody\b"]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_knowledge_base() -> list:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))["items"]


def _is_denied(sentence: str, phrase: str) -> bool:
    idx = sentence.find(phrase)
    if idx == -1:
        return False
    before = sentence[:idx]
    return any(re.search(cue, before) for cue in NEGATION_CUES)


def detect_references(text: str, items: list) -> list:
    """Returns the knowledge items whose signature phrases appear in text
    and are NOT denied in the same sentence (see NEGATION_CUES above)."""
    text_lower = text.lower()
    sentences = SENTENCE_SPLIT.split(text_lower) or [text_lower]
    matched = []
    for item in items:
        for phrase in item["signature_phrases"]:
            if phrase not in text_lower:
                continue
            # confirmed (not denied) if ANY sentence containing the phrase
            # states it without a preceding negation cue
            confirmed = any(phrase in s and not _is_denied(s, phrase) for s in sentences)
            if confirmed:
                matched.append(item)
            break
    return matched


def compute_kbd(response: str, target_archetype: str, items: list) -> dict:
    matched = detect_references(response, items)
    if not matched:
        return {"kbd": None, "total_references": 0, "violations": [], "reason": "no_factual_reference_detected"}

    violations = [item for item in matched if target_archetype not in item["visible_to"]]
    return {
        "kbd": round(len(violations) / len(matched), 4),
        "total_references": len(matched),
        "violations": [v["id"] for v in violations],
        "matched": [m["id"] for m in matched],
    }
