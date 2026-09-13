"""What a player message is doing, decided by rules.

The composer uses this to choose the ONE piece of late context a reply gets,
and the guard uses it to choose the repair. Getting it wrong is cheap -- the
worst case is the context the pipeline used before intents existed -- so the
rules favour precision: a message is only "about the player" when it
unmistakably is, because the memory turn it triggers crowds out everything
else (returning players asking "what do you do" got "I don't do anything"
when memory was shown on every turn).
"""

import re
from dataclasses import dataclass, field

GREETING = "greeting"
FAREWELL = "farewell"
ACK = "ack"              # "ok", "cool", "hmm", "thanks"
RECALL = "recall"        # asks what the NPC remembers about the player
ABOUT_NPC = "about_npc"  # asks the NPC about itself
QUESTION = "question"    # any other question (world, campus, advice)
STATEMENT = "statement"  # anything else, including self-introductions

# Which remembered slot a recall question is after. "any" = the whole record.
_RECALL_SLOTS = [
    ("name", r"\bmy name\b|\bwho am i\b|\bcall me\b|\bremember my name\b"),
    ("year", r"\b(what|which) year (am i|i'm)\b|\bmy year\b|\byear am i\b|\byear did i\b"),
    ("project", r"\bmy (final[- ]year )?project\b|\bproject idea\b|\bworking towards\b"
                r"|\bwanted to (build|make|do)\b"),
    ("interests", r"\bwhat am i (into|interested in)\b|\bwhat i'm (into|interested in)\b"
                  r"|\bmy interests?\b|\bi('m| am) interested in\?"),
    ("from", r"\bwhere am i from\b|\bwhere i'm from\b"),
    ("studies", r"\bwhat do i study\b|\bmy (course|major|branch|subject)\b"),
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


def classify(message: str) -> Intent:
    """`message` should already be normalised (text.normalize)."""
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
