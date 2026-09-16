"""What a player message is doing.

The composer uses this to choose the ONE piece of late context a reply gets,
and the guard uses it to choose the repair.

Learned, not written. The first classifier was patterns, and it failed the
way patterns fail: 14/14 on the lines it was written against, 35/40 on unseen
ones -- "hii sir", "good evening mam", "hey there, how r u" were not
greetings; "oh nice", "thats cool" were not acknowledgements. Now two small
heads over the fact extractor's sentence encoder decide (LearnedIntent below;
trained by training/extractor/train_intent.py): which of the seven intents,
and for a recall question, which remembered facts it asks about.

rules_classify() is kept only as the fallback when no trained head is on disk,
so a fresh clone still runs. It is not consulted when the model is present.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

GREETING = "greeting"
FAREWELL = "farewell"
ACK = "ack"              # "ok", "cool", "hmm", "thanks"
RECALL = "recall"        # asks what the NPC remembers about the player
ABOUT_NPC = "about_npc"  # asks the NPC about itself
QUESTION = "question"    # any other question (world, campus, advice)
REPORT = "report"        # tells the NPC about something happening (see events.py)
STATEMENT = "statement"  # anything else, including self-introductions

# Which remembered slot a recall question is after. "any" = the whole record.
_RECALL_SLOTS = [
    ("name", r"\bmy name\b|\bwho am i\b|\bcall me\b|\bremember my name\b"),
    ("year", r"\b(what|which) year (am i|i'm)\b|\bmy year\b|\byear am i\b|\byear did i\b"),
    ("project", r"\bmy (final[- ]year )?project\b|\bproject idea\b|\bworking towards\b"
                r"|\bwanted to (build|make|do)\b"),
    ("interests", r"\bwhat am i (into|interested in)\b|\bwhat i'm (into|interested in)\b"
                  r"|\bmy interests?\b|\bi('m| am) interested in\?"),
    ("hometown", r"\bwhere am i from\b|\bwhere i'm from\b|\bmy (home ?town|native)\b"),
    ("department", r"\bwhat do i study\b|\bmy (course|major|branch|subject|department|dept)\b"),
    ("section", r"\bmy (section|class)\b|\bwhich (section|class) am i\b"),
    ("college", r"\bmy college\b|\bwhich college am i\b|\bwhere do i study\b"),
    ("any", r"\bremember me\b|\babout me\b|\bremind me\b|\bwhat did i (tell|say|mention)\b"
            r"|\bi (told|said|mentioned)\b|\bdid i (tell|say|mention)\b|\bremember\b|\brecall\b"),
]
_RECALL_SLOT_RES = [(slot, re.compile(p, re.IGNORECASE)) for slot, p in _RECALL_SLOTS]

_GREETING = re.compile(
    r"^(hi|hii+|hello|hey|heya|hiya|yo|good (morning|afternoon|evening)|namaste|vanakkam)"
    r"( there| again| everyone)?[\s!.,]*$", re.IGNORECASE)
_FAREWELL = re.compile(
    r"\b(bye|goodbye|see you|see ya|good night|take care|catch you later|gotta go|have to go)\b",
    re.IGNORECASE)
_ACK = re.compile(
    r"^(ok|okay|cool|nice|great|hmm+|hm+|oh|ah|alright|right|sure|thanks|thank you|yeah|yes|"
    r"no|nope|fine|got it|i see|wow|lol|awesome|interesting)( thanks| thank you| ok| cool)?[\s!.,]*$",
    re.IGNORECASE)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
_RECALL_REQUEST = re.compile(
    r"^(remind me|tell me (my|who|what i)|say my|recall (me|my|what)|remember (me|my|what i)"
    r"|guess (my|who))\b", re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^(what|what's|where|when|who|who's|whom|whose|why|how|which|is|are|am|do|does|did|can|could|"
    r"will|would|should|may|have|has|tell me|any|you remember|you know)\b", re.IGNORECASE)


@dataclass
class Intent:
    kind: str
    recall_slots: list = field(default_factory=list)  # for RECALL
    is_question: bool = False


def rules_classify(message: str) -> Intent:
    """The pattern classifier: fallback only. `message` should be normalised."""
    text = (message or "").strip()
    is_question = text.endswith("?") or bool(_QUESTION_START.match(text))

    slots = [slot for slot, rx in _RECALL_SLOT_RES if rx.search(text)]
    # Recall is a question or a request. "i want to do my final year project
    # on chatbots" mentions "my project" but tells, it does not ask -- treated
    # as recall, it got "You told me you wanted to do that."
    if slots and not (is_question or _RECALL_REQUEST.match(text)):
        slots = []
    if slots:
        return Intent(RECALL, recall_slots=slots, is_question=True)

    if _GREETING.match(text):
        return Intent(GREETING)
    if _FAREWELL.search(text) and len(text.split()) <= 6:
        return Intent(FAREWELL)
    if _ACK.match(text):
        return Intent(ACK)
    if is_question and _SECOND_PERSON.search(text):
        return Intent(ABOUT_NPC, is_question=True)
    if is_question:
        return Intent(QUESTION, is_question=True)
    return Intent(STATEMENT)


class LearnedIntent:
    """Intent and recall-slot heads over the extractor's encoder."""

    SLOT_THRESHOLD = 0.5

    def __init__(self, model_dir: Path | None = None):
        from . import extractor
        self.model_dir = Path(model_dir) if model_dir else extractor.FINE_TUNED
        self._extractor = extractor
        self._ready = None
        self._lock = threading.Lock()

    def _load(self):
        if self._ready is not None:
            return self._ready
        with self._lock:
            if self._ready is not None:
                return self._ready
            head, meta = self.model_dir / "intent_head.pt", self.model_dir / "intent_head.json"
            if not (head.exists() and meta.exists()):
                log.warning("no trained intent head at %s: using the pattern fallback", self.model_dir)
                self._ready = False
                return False
            model = (self._extractor.default._load()
                     if self.model_dir == self._extractor.FINE_TUNED else None)
            try:
                import torch
                from .encoder import SentenceEncoder
                self.encoder = SentenceEncoder(self.model_dir, gliner_model=model)
                info = json.loads(meta.read_text(encoding="utf-8"))
                weights = torch.load(head, map_location="cpu")
                self.intents, self.slots = info["intents"], info["slots"]
                dim = info["encoder_dim"]
                self.intent_head = torch.nn.Linear(dim, len(self.intents))
                self.slot_head = torch.nn.Linear(dim, len(self.slots))
                self.intent_head.load_state_dict(weights["intent"])
                self.slot_head.load_state_dict(weights["slots"])
                self._ready = True
            except Exception as exc:  # noqa: BLE001 - fall back rather than break the turn
                log.warning("could not load the intent head (%s): using the pattern fallback", exc)
                self._ready = False
            return self._ready

    @property
    def available(self) -> bool:
        return bool(self._load())

    def classify(self, message: str) -> Intent:
        import torch
        vector = self.encoder.encode([message or ""])
        with torch.no_grad():
            kind = self.intents[int(self.intent_head(vector).argmax(1))]
            slot_probs = torch.sigmoid(self.slot_head(vector))[0]
        if kind != RECALL:
            return Intent(kind, is_question=kind in (ABOUT_NPC, QUESTION))
        slots = [s for s, p in zip(self.slots, slot_probs.tolist()) if p >= self.SLOT_THRESHOLD]
        return Intent(RECALL, recall_slots=slots or ["any"], is_question=True)


default = LearnedIntent()


def classify(message: str) -> Intent:
    """What `message` (as the player typed it) is doing."""
    if default.available:
        return default.classify(message)
    from .text import normalize
    return rules_classify(normalize(message))
