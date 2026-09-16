"""What the player says about themselves, read by a model.

The first extractor was regular expressions, and it could not win: every fix
added a word list, and every word list had a hole. "i am class cse d" saved
the name "Class"; "i am tamil", "i am hosteller", "i am cr of cse d" saved
"Tamil", "Hosteller", "Cr". Patterns guess at what a name looks like; they do
not read the sentence.

So a model reads it. GLiNER is a span model: given a line and plain-English
labels ("student name", "class section"), it scores every span of the line
against every label. Nothing here says what a name, a department or a section
looks like -- the model was fine-tuned on synthetic student introductions
(training/extractor/) whose names, towns, colleges and departments are
disjoint from the evaluation lines, so what it gets right it got from
structure. Adding a fact to remember is adding a label, not a pattern.

Rules remain only as checks on the model, never as the source of a fact:

  - context    the NPC's last line goes in front of the player's, so "yuga"
               after "What's your name?" is read as a name; spans inside the
               NPC's line are ignored.
  - threshold  a span must score at least MIN_SCORE (chosen on dev lines).
  - one label per span, best-scoring slot wins (GLiNER's flat decoding).
  - confidence a saved slot keeps its score; a later, less confident reading
               cannot overwrite it (memory.merge). "my name is yugabharathi"
               then a weak misreading elsewhere cannot rename the player.

About 60 ms per line on CPU after a one-off load. Measured on held-out lines
(evaluation/run_memory_extraction.py): 35/36 fully right fine-tuned, 16/36
zero-shot; no slot learned that should not have been.
"""

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
FINE_TUNED = REPO / "training" / "extractor" / "player_facts_gliner"
BASE_MODEL = "urchade/gliner_small-v2.1"

# Slot -> label text. Must match training/extractor/make_extraction_data.py
# (test_dialogue.py checks).
LABELS = {
    "name": "student name",
    "year": "year of study",
    "department": "department or branch",
    "section": "class section",
    "college": "college",
    "hometown": "hometown",
    "project": "project topic",
    "interests": "interest",
    "feeling": "feeling",
    # A world event the player reports. Never saved as a fact about the player:
    # read() returns these separately and the game keeps them in its event log.
    "incident": "incident",
    "place": "incident location",
}
EVENT_SLOTS = {"incident", "place"}
MULTI_VALUED = {"interests"}
# Chosen on the dev lines of evaluation/memory_extraction_cases.json (0.3-0.7
# tried). With incident labels added, 0.7 was the only setting that learned no
# forbidden slot on dev ("my brother arivu is in mech" gave the player a
# department below it). See NUMBERS.md.
MIN_SCORE = 0.7
# Incident spans use the same bar. A lower one (0.4) was tried and made "the
# canteen food was bad today" an incident on the test lines -- and it had been
# picked from a report line in run_events.py, not from dev. A report the model
# cannot parse is not lost: events.event_from keeps the player's own words.
EVENT_MIN_SCORE = MIN_SCORE


class FactExtractor:
    """Lazy, thread-safe wrapper around one GLiNER model."""

    def __init__(self, path: Path | str | None = None, min_score: float = MIN_SCORE):
        self.path = path
        self.min_score = min_score
        self._model = None
        self._lock = threading.Lock()
        self._failed = False
        self.source = ""

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                from gliner import GLiNER
            except ImportError:
                log.warning("gliner is not installed: NPCs will not learn facts about the player")
                self._failed = True
                return None
            candidates = [self.path] if self.path else [FINE_TUNED, BASE_MODEL]
            for candidate in candidates:
                try:
                    if isinstance(candidate, Path) and not candidate.exists():
                        continue
                    self._model = GLiNER.from_pretrained(str(candidate), local_files_only=True)
                    self.source = str(candidate)
                    if not self.path and candidate != FINE_TUNED:
                        log.warning("player-fact extractor: fine-tuned weights not found, using %s", candidate)
                    return self._model
                except Exception as exc:  # noqa: BLE001 - try the next candidate, report at the end
                    log.info("player-fact extractor: could not load %s (%s)", candidate, exc)
            log.warning("no player-fact extractor model available: NPCs will not learn facts about the player")
            self._failed = True
            return None

    @property
    def available(self) -> bool:
        return self._load() is not None

    def read(self, message: str, context: str = "") -> tuple:
        """(facts about the player, reported event), from one model call.

        Both are {slot: (value, score)}; facts' `interests` is a list of
        (value, score). `context` is the NPC's previous line: it helps the model
        read short answers and is never a source of facts itself.
        """
        model = self._load()
        text = (message or "").strip()
        if model is None or not text:
            return {}, {}
        prefix = (context or "").strip()
        prefix = prefix + " " if prefix else ""
        spans = model.predict_entities(prefix + text, list(LABELS.values()),
                                       threshold=min(self.min_score, EVENT_MIN_SCORE), flat_ner=True)
        slot_of = {label: slot for slot, label in LABELS.items()}
        facts, event = {}, {}
        for span in spans:
            if span["start"] < len(prefix):
                continue  # inside the NPC's line
            slot = slot_of[span["label"]]
            value = span["text"].strip(" .,!?;:'\"")
            is_event = slot in EVENT_SLOTS
            if not value or span["score"] < (EVENT_MIN_SCORE if is_event else self.min_score):
                continue
            target = event if is_event else facts
            if slot in MULTI_VALUED:
                target.setdefault(slot, []).append((value, float(span["score"])))
            elif slot not in target or span["score"] > target[slot][1]:
                target[slot] = (value, float(span["score"]))
        return facts, event

    def extract(self, message: str, context: str = "") -> dict:
        """Facts the player stated about themselves (see read())."""
        return self.read(message, context)[0]


# One per process; the model is shared by every NPC.
default = FactExtractor()
