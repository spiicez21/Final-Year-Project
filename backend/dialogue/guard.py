"""Checks every in-character reply, and repairs what it can.

Detection is by rules over the reply and the turn's context. Repair spends at
most ONE extra generation per turn -- a second retry would push a turn past a
second of latency -- and is ordered so the most specific diagnosis wins:

  problem          seen in play                               repair
  ---------------  -----------------------------------------  -----------------------------
  list             "1. Research your topic..." (unspeakable,  flatten to one sentence
                   cut off mid-list)                           (no generation)
  asks_known       "Hi Yuga, what's your name?"                drop that question if the
                                                               reply says anything else
  recall_miss      "That's not something I can tell you", or   restart as "You told me"
                   the dodge "Yes, I remember you.", to "u     (only if the asked slot is
                   remember my name?" with the name saved      actually remembered)
  invented_name    "You told me your name was Emily."          remembered name: substitute;
                   (nothing saved for the name)                recall question, nothing
                                                               saved: restart as "I don't
                                                               think you've told me"
  fact_refusal     "I'm not sure" with the answer retrieved    restart on the fact's first
                                                               two words; if it still
                                                               refuses, say the fact itself
  self_denial      "I don't do anything." to "what do you do"  restart on the job line's
                                                               first two words
  wrong_identity   "I'm a librarian." from the police officer  same
  asks_known,      "Sure, what's your name?" straight after    restart as "Nice to meet
  nothing else     "myself Yuga"                               you, Yuga."
  repeat           the same line as an earlier reply, to a     regenerate with that exchange
                   different message                           hidden from the transcript

Prefilled restarts give the model a frame, never the answer: the words after
the prefix are still generated.
"""

import re
from dataclasses import dataclass, field

from . import memory as player_memory
from .intent import ABOUT_NPC, ACK, RECALL
from .text import similar, split_sentences

# Cut generation at the point the model starts writing the player's next line
# or drifting into document formatting.
STOP_SEQUENCES = ["\nStudent:", "\nYou:", "\nyou:", "Student:", "###", "\n\n\n"]

# Assistant-prior boilerplate that survives the prompt instructions. Matched at
# the START of a reply only, so an NPC can still use these words mid-sentence.
_PREAMBLE_RE = re.compile(r"^\s*(?:sure|certainly|of course|okay|ok|here)\b[^\n:]{0,60}:\s*",
                          re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]]*\s*$")


def clean_basic(text: str) -> str:
    """Strips assistant-prior artefacts and mid-sentence truncation.

    Unchanged from the original server, and the only cleaning the evaluation
    (no-persona) path gets, so that path stays comparable to the paper.

    Generation is capped at a low token count to keep latency inside the
    real-time budget, so replies regularly stop mid-clause. Rather than raise
    the cap, the dangling tail is dropped back to the last completed sentence.
    """
    cleaned = text.strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned, count=1).strip()
    # Models often wrap a persona line in quotes; drop them only when they
    # enclose the whole reply, so quoted speech inside a line survives.
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    if not _SENTENCE_END_RE.search(cleaned):
        cut = max(cleaned.rfind(c) for c in ".!?…")
        if cut > 0:
            cleaned = cleaned[:cut + 1]
    # Never hand back an empty string: a short odd reply is more debuggable
    # than an NPC that appears to say nothing at all.
    return cleaned.strip() or text.strip()


_LIST_ITEM = re.compile(r"(?:^|\s)(?:\d{1,2}[.)]|[-*•])\s+")
MAX_SENTENCES = 2  # the persona prompt asks for "one or two short spoken sentences"


def clean(text: str) -> str:
    """clean_basic, then shaped for speech: lists flattened, lines joined,
    at most MAX_SENTENCES sentences."""
    cleaned = clean_basic(text)
    if _LIST_ITEM.search(cleaned):
        parts = [p.strip() for p in _LIST_ITEM.split(cleaned) if p and p.strip()]
        preamble = cleaned[:_LIST_ITEM.search(cleaned).start()].strip()
        first_item = parts[1] if preamble and len(parts) > 1 else parts[0]
        keep_preamble = preamble and not preamble.endswith(":") and len(preamble.split()) >= 4
        cleaned = ((preamble + " ") if keep_preamble else "") + first_item
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned).strip()
    sentences = split_sentences(cleaned)
    if len(sentences) > MAX_SENTENCES:
        cleaned = " ".join(sentences[:MAX_SENTENCES])
    if cleaned and not _SENTENCE_END_RE.search(cleaned):
        cleaned += "."
    return cleaned or clean_basic(text)


# --- detection ---------------------------------------------------------------

# Broad: any refusal-shaped reply to a recall question.
RECALL_REFUSAL = re.compile(
    r"\b(don't|do not|can't|cannot|not sure|no idea|not something|never met|haven't|not important)\b",
    re.IGNORECASE)
# Narrow: the refusals seen on fact questions (evaluation/run_grounding.py uses
# the same wording to score refusals).
FACT_REFUSAL = re.compile(
    r"\b(don't know|do not know|not sure|no idea|not something|can't tell|cannot tell"
    r"|don't have|not allowed|i'm not able|don't keep track)\b", re.IGNORECASE)
SELF_DENIAL = re.compile(
    r"\bi (don't|do not) (do anything|have a job|work here|know anything about (that|this|it))\b"
    r"|\bi'm (just|only) a (machine|computer|program)\b|\bi'm not (really )?into anything\b"
    r"|\bi don't know anything\b", re.IGNORECASE)
STATED_NAME = re.compile(r"\b(?:your name(?: is|'s| was)|you're|you are|call you)\s+([A-Z][a-z]+)\b"
                         r"|,\s*([A-Z][a-z]+)[.!?]\s*$")
_ASKS = {
    "name": re.compile(r"\b(what's|what is|may i have|can i get) your name\b|\bwho are you\b", re.I),
    "year": re.compile(r"\b(what|which) year are you\b", re.I),
    "project": re.compile(r"\bwhat('s| is) your project\b|\bwhat (kind of )?project\b", re.I),
    "studies": re.compile(r"\bwhat do you study\b|\bwhat('s| is) your (major|course|branch)\b", re.I),
}
_NOT_PLAYER_NAMES = {"Sure", "Yes", "No", "Well", "Okay", "Thanks", "Welcome", "Right", "Great",
                     "Good", "Nice", "Hello", "Hi", "Bye", "Professor", "Officer", "Ms", "Mr"}


@dataclass
class Check:
    text: str                  # normalised player message
    intent: object
    memory: dict
    history: list              # replayed transcript turns
    facts_used: list
    npc_name: str
    job_line: str = ""
    persona_text: str = ""     # name, archetype, occupation, intro, job line: who the NPC is
    updates: dict = field(default_factory=dict)  # what this message taught


@dataclass
class Regeneration:
    """The one allowed retry: a name for the repair, and how to generate."""
    name: str
    prefill: str = ""
    hide_reply: str = ""       # drop transcript exchanges whose reply is similar to this


# "I'm a librarian" from a police officer: an identity claim with no word in
# common with who the NPC is. Stems, so "counselor" matches "counsellor".
_IDENTITY = re.compile(r"\bi'?m (?:a|an|the|just a|just the) ([a-z]+)(?: ([a-z]+))?", re.IGNORECASE)
_GENERIC_ROLES = {"student", "person", "guest", "visitor", "lot", "bit", "little", "big", "good",
                  "great", "huge", "fan", "member", "part", "new", "big", "same", "one"}


def _wrong_identity(reply: str, check: Check) -> bool:
    m = _IDENTITY.search(reply)
    if not m or not check.persona_text:
        return False
    persona_stems = {w[:5] for w in re.findall(r"[a-z]+", check.persona_text.lower()) if len(w) > 3}
    claimed = [w for w in m.groups() if w and len(w) > 3 and w.lower() not in _GENERIC_ROLES]
    return bool(claimed) and not any(w.lower()[:5] in persona_stems for w in claimed)


def _mentions_memory(reply: str, memory: dict, slots=("any",)) -> bool:
    low = reply.lower()
    keys = [k for k in memory if k != "feeling"] if "any" in slots else [s for s in slots if s in memory]
    values = []
    for k in keys:
        v = memory[k]
        values += v if isinstance(v, list) else [v]
    # "first-year" is recalled by "your first year", "robotics" by "a robot".
    stems = [str(v).lower().replace("-", " ").split()[0][:5] for v in values if str(v).strip()]
    return any(re.search(r"\b" + re.escape(s), low) for s in stems)


def _stated_player_name(reply: str, check: Check):
    npc_words = set(re.findall(r"[A-Za-z]+", check.npc_name))
    for m in STATED_NAME.finditer(reply):
        name = m.group(1) or m.group(2)
        if name and name not in npc_words and name not in _NOT_PLAYER_NAMES:
            return name
    return None


def _earlier(history: list, role: str) -> list:
    return [t["content"] for t in history if t.get("role") == role]


def detect(reply: str, check: Check) -> set:
    found = set()
    intent, memory = check.intent, check.memory
    if _LIST_ITEM.search(reply):
        found.add("list")

    name = _stated_player_name(reply, check)
    if name and name.lower() != str(memory.get("name", "")).lower():
        found.add("invented_name")

    if intent.kind == RECALL and memory:
        # A specific slot that is saved must appear in the answer: "Yes, I
        # remember you." to "u remember my name?" is a dodge, not a refusal.
        # For a general "do you remember me", only a refusal counts as a miss.
        specific = [s for s in intent.recall_slots if s != "any" and s in memory]
        if specific:
            if not _mentions_memory(reply, memory, specific):
                found.add("recall_miss")
        elif "any" in intent.recall_slots and RECALL_REFUSAL.search(reply) \
                and not _mentions_memory(reply, memory):
            found.add("recall_miss")

    if check.facts_used and FACT_REFUSAL.search(reply):
        found.add("fact_refusal")
    if intent.kind == ABOUT_NPC and SELF_DENIAL.search(reply):
        found.add("self_denial")
    if intent.kind == ABOUT_NPC and _wrong_identity(reply, check):
        found.add("wrong_identity")
    if intent.kind == ACK and FACT_REFUSAL.search(reply):
        found.add("ack_refusal")

    if any(s.endswith("?") and _is_known_question(s, check) for s in split_sentences(reply)):
        found.add("asks_known")

    users = _earlier(check.history, "user")
    replies = _earlier(check.history, "assistant")
    for u, r in zip(users, replies):
        if similar(reply, r) and not similar(check.text, u):
            found.add("repeat")
            break
    return found


# --- repair ------------------------------------------------------------------

def _is_known_question(sentence: str, check: Check) -> bool:
    """Asks for something already saved, or repeats an earlier NPC question."""
    earlier = [s for r in _earlier(check.history, "assistant")
               for s in split_sentences(r) if s.endswith("?")]
    return (any(rx.search(sentence) and check.memory.get(slot) for slot, rx in _ASKS.items())
            or any(similar(sentence, e) for e in earlier))


def _drop_questions_already_answered(reply: str, check: Check) -> str:
    return " ".join(s for s in split_sentences(reply)
                    if not (s.endswith("?") and _is_known_question(s, check)))


def _fix_name(reply: str, check: Check) -> str:
    wrong = _stated_player_name(reply, check)
    if not wrong:
        return reply
    right = check.memory.get("name")
    if right:
        return re.sub(r"\b%s\b" % re.escape(wrong), right, reply)
    # No name remembered: drop the vocative (", Emily.") or the sentence.
    fixed = re.sub(r",\s*%s(?=[.!?])" % re.escape(wrong), "", reply)
    if fixed != reply:
        return fixed
    return " ".join(s for s in split_sentences(reply) if wrong not in s)


def _first_words(sentence: str, n: int = 2) -> str:
    return " ".join((sentence or "").split()[:n])


def plan_regeneration(reply: str, problems: set, check: Check):
    """The one allowed retry, most specific diagnosis first, or None."""
    intent = check.intent
    if intent.kind == RECALL and ("recall_miss" in problems or "invented_name" in problems):
        asked_known = any(s in check.memory or (s == "any" and check.memory)
                          for s in intent.recall_slots)
        if asked_known:
            return Regeneration("recall", prefill="You told me")
        if "invented_name" in problems:
            return Regeneration("recall_unknown", prefill="I don't think you've told me")
    if "fact_refusal" in problems and check.facts_used:
        return Regeneration("fact", prefill=_first_words(check.facts_used[0]))
    if ("self_denial" in problems or "wrong_identity" in problems) and check.job_line:
        return Regeneration("self", prefill=_first_words(split_sentences(check.job_line)[0]))
    if "asks_known" in problems and not _drop_questions_already_answered(reply, check).strip():
        # Nothing but the question: "Sure, what's your name?" straight after
        # "myself Yuga". Acknowledge the introduction instead.
        if check.updates.get("name"):
            return Regeneration("introduced", prefill="Nice to meet you, %s." % check.updates["name"])
    if intent.kind == ACK and problems & {"asks_known", "repeat", "ack_refusal"}:
        # "alright" -> "what's your name?" or "I'm sorry, I don't have that
        # information." twice running. An acknowledgement needs an
        # acknowledgement back, not new content.
        return Regeneration("acknowledge", prefill="Okay.")
    if "repeat" in problems:
        # A repetition penalty does not reach it: llama.cpp penalises only the
        # last 64 tokens, and the repeated line is further back. What the model
        # cannot see it cannot copy, so the exchange it is copying is hidden
        # for the retry. (4/4 replayed repeats broke; penalty 1.3-2.0: 0/4.)
        return Regeneration("repeat", hide_reply=reply)
    return None


def after_regeneration(reply: str, regen: Regeneration, check: Check) -> tuple:
    """Last resort once the retry is spent. Returns (reply, repair or None)."""
    if regen.name == "recall":
        slots = [s for s in check.intent.recall_slots if s != "any" and s in check.memory]
        if not _mentions_memory(reply, check.memory, slots or ("any",)):
            # "You told me." -- the frame without the content. Say what is
            # saved; it states nothing the player did not say.
            return player_memory.recall_line(check.memory, slots or ["any"]), "recall_fallback"
    if regen.name == "acknowledge":
        # Keep what is left once refusals and already-answered questions go.
        kept = [s for s in split_sentences(reply)
                if not RECALL_REFUSAL.search(s) and not (s.endswith("?") and _is_known_question(s, check))]
        return (" ".join(kept) or regen.prefill), None
    if regen.name == "fact" and FACT_REFUSAL.search(reply) and check.facts_used:
        # The retry still refused ("I'm on grade 8, so I don't have that
        # information" -- the transcript pulls it back into the refusal even
        # with the start fixed). Say the retrieved fact itself: an authored
        # line, not a generated one, and reported as such.
        return check.facts_used[0], "fact_fallback"
    return reply, None


def finish(reply: str, check: Check) -> tuple:
    """Repairs that need no generation. Returns (reply, repairs applied)."""
    applied = []
    problems = detect(reply, check)
    if "asks_known" in problems:
        trimmed = _drop_questions_already_answered(reply, check)
        if trimmed.strip():
            reply = trimmed
            applied.append("asks_known")
    if "invented_name" in detect(reply, check):
        fixed = _fix_name(reply, check).strip()
        if fixed:
            reply = fixed
            applied.append("name")
    return reply, applied
