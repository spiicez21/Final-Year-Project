"""What an NPC remembers about the player, and how it learns it.

Why structured facts and not the transcript. The NPCs already replay recent
dialogue, but on a 1.1B model that is weak memory even within one visit (11/40
held-out recall probes answered) and none at all across visits, because the
transcript does not survive (0/40). Short facts, kept per NPC and persisted by
the game, survive: a returning player is recalled on 18/40 of the same probes
(evaluation/run_player_memory.py; results/player_memory_results.json). They
are also small enough to inspect, test and reset, which a transcript is not.

The honest limit: within one visit memory adds little over the transcript
(11 -> 13/40 held-out), and recall varies by adapter -- the executive, trained
hardest to refuse, recalls a returning player 0/8 times; the police officer
7/8. The adapters' "I don't remember" reflex is the ceiling here, not the
memory.

Why rules and not the model for extraction. Asking TinyLlama to summarise
"what did the player tell you" would add a second generation to every turn
and be neither reliable nor inspectable. Regular expressions over the
player's own words are cheap, deterministic and unit-testable, and they only
have to cover the handful of things a student says about themselves at an
open day: name, year, subject, interests, how they feel, where they are from.
Anything they miss is simply not remembered, which is a safer failure than
remembering something the player never said.

Memory is a plain dict of slots, not free text, so updates merge predictably:
a new name replaces the old one, interests accumulate up to a cap.

    {"name": "Priya", "year": "first-year", "studies": "computer science",
     "interests": ["compilers"], "feeling": "stressed", "from": "Chennai"}
"""

import re

MAX_INTERESTS = 4

# Capitalised words that follow "I'm" without being a name. Checked because
# the name pattern keys on capitalisation, and sentence-initial words or
# emphasis ("I'm Sorry", "I'm Here for...") would otherwise become names.
_NOT_NAMES = {
    "a", "an", "the", "here", "sorry", "fine", "good", "okay", "ok", "not",
    "just", "so", "also", "from", "in", "at", "doing", "interested", "looking",
    "thinking", "really", "very", "new", "lost", "stressed", "worried", "going",
    "trying", "studying", "curious", "wondering", "excited", "nervous", "happy",
    "tired", "confused", "back", "still", "only", "sure", "first", "second",
    "third", "final", "fourth", "hoping", "planning", "applying", "coming",
}

# Two strengths of evidence for a name. "my name is" / "call me" are explicit,
# so a lowercase name is accepted ("my name is priya" -- players rarely
# capitalise in a chat box). "I'm X" is ambiguous ("I'm lost"), so there the
# name must be capitalised and not a known non-name.
_NAME_EXPLICIT = re.compile(r"\b(?:my name is|my name's|call me|you can call me)\s+([a-z]+(?:\s[a-z]+)?)",
                            re.IGNORECASE)
_NAME = re.compile(
    r"\b(?i:i'm|im|i am|this is)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
# Either "I'm (a / in my) first year" or a self-description fragment
# ("Arjun, a first-year"). A bare "first year" is not enough: "what do
# first-year students study?" is about the NPC's world, not the player.
_ORD = r"(first|second|third|fourth|final|1st|2nd|3rd|4th)"
_YEAR = re.compile(
    r"\b(?:i'm|im|i am)(?: a| an| in)?(?: my)? " + _ORD + r"[- ]year"
    r"|(?:^|, )an? " + _ORD + r"[- ]year")
_STUDIES = re.compile(
    r"\bi(?:'m| am|m)? ?(?:studying|study|doing a degree in|major in|majoring in) "
    r"([a-z][a-z ]{1,40}?)(?=[.,!?]| and | but |$)")
_INTEREST = re.compile(
    r"\b(?:i'm interested in|im interested in|i am interested in|(?<!you )(?<!you're )interested in|"
    r"i'm into|im into|"
    r"i really like|i like|i love|i enjoy|"
    r"i want to (?:work on|learn|do|study))\s+([a-z0-9][a-z0-9 +#.-]{1,40}?)(?=[.,!?]| and | but | so |$)")
# A project is kept apart from general interests: "what was my project idea?"
# is a different question from "what am I into?", and with the project folded
# into interests the NPC missed it in all six live runs.
_PROJECT = re.compile(
    r"\b(?:project (?:on|about|in)|(?:do|doing|build|building|make|making) (?:a |my )?"
    r"(?:final[- ]year )?project (?:on|about|in))\s+([a-z0-9][a-z0-9 +#.-]{1,40}?)(?=[.,!?]| and | but | so |$)")
_FEELING = re.compile(
    r"\bi(?:'m| am| feel| feel really| feel so)? (?:really |so |very |a bit |quite )?"
    r"(stressed|worried|anxious|nervous|excited|tired|lost|confused|overwhelmed|homesick)\b")
_FROM = re.compile(r"\b(?i:i'm|i am) (?i:originally )?(?i:from) ([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)")

# Words that end a lowercase name rather than continue it: "my name is priya
# and I..." must give "Priya", not "Priya And".
_JOINERS = {"and", "but", "so", "i", "im", "from", "here", "nice", "sir", "maam",
            "ma'am", "btw", "by", "the", "a", "an", "to", "what", "how", "who"}

_YEAR_WORDS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth"}


def extract(message: str) -> dict:
    """Facts the player stated about themselves in this one message.

    Returns only the slots that were found; an empty dict means nothing was
    learned. Case matters for names and places (they key on capitals), so the
    raw message is matched for those and a lowercased copy for the rest.
    """
    text = message.strip().replace("’", "'")
    low = text.lower()
    found = {}

    m = _NAME_EXPLICIT.search(text)
    words = m.group(1).lower().split() if m else []
    if len(words) == 2 and words[1] in _JOINERS:
        words = words[:1]
    if words and words[0] not in _NOT_NAMES:
        found["name"] = " ".join(w.capitalize() for w in words)
    else:
        for m in _NAME.finditer(text):
            if m.group(1).split()[0].lower() not in _NOT_NAMES:
                found["name"] = m.group(1)
                break

    m = _YEAR.search(low)
    if m:
        ordinal = m.group(1) or m.group(2)
        found["year"] = _YEAR_WORDS.get(ordinal, ordinal) + "-year"

    # Matched on the lowercase copy, but the value is sliced from the original
    # so "AI" or "C++" keep their case (lower() preserves length for ASCII).
    def original(m):
        return text[m.start(1):m.end(1)].strip() if len(text) == len(low) else m.group(1).strip()

    m = _STUDIES.search(low)
    if m and m.group(1).strip() not in ("it", "that", "this", "well", "fine", "ok", "okay"):
        found["studies"] = original(m)

    interests = []
    for m in _INTEREST.finditer(low):
        if m.group(1).strip() not in ("", "it", "that", "this", "you", "them"):
            interests.append(original(m))
    if interests:
        found["interests"] = interests

    m = _PROJECT.search(low)
    if m and m.group(1).strip() not in ("it", "that", "this"):
        found["project"] = original(m)

    m = _FEELING.search(low)
    if m:
        found["feeling"] = m.group(1)

    m = _FROM.search(text)
    if m:
        found["from"] = m.group(1)

    return found


def merge(memory: dict, update: dict) -> dict:
    """Folds one message's facts into what is already remembered.

    Single-valued slots are replaced by the newest statement -- a player who
    corrects their name should be believed. Interests accumulate, oldest
    dropped first, capped so the prompt cannot grow without bound.
    """
    out = dict(memory or {})
    for key, value in (update or {}).items():
        if key == "interests":
            merged = list(out.get("interests", []))
            for item in value:
                if item not in merged:
                    merged.append(item)
            out["interests"] = merged[-MAX_INTERESTS:]
        else:
            out[key] = value
    return out


def render(memory: dict) -> str:
    """Memory as short third-person sentences for the system prompt.

    "They" throughout: the player's gender is never stated, so the prompt does
    not assume one.
    """
    if not memory:
        return ""
    lines = []
    if memory.get("name"):
        lines.append("Their name is %s." % memory["name"])
    if memory.get("year"):
        lines.append("They are a %s student." % memory["year"])
    if memory.get("studies"):
        lines.append("They study %s." % memory["studies"])
    if memory.get("from"):
        lines.append("They are from %s." % memory["from"])
    if memory.get("project"):
        lines.append("They want to do their final-year project on %s." % memory["project"])
    others = [i for i in memory.get("interests", []) if i != memory.get("project")]
    if others:
        lines.append("They are interested in %s." % ", ".join(others))
    if memory.get("feeling"):
        lines.append("Earlier they said they felt %s." % memory["feeling"])
    return " ".join(lines)


def _about(memory: dict) -> str:
    """Everything remembered, as one second-person sentence an NPC could say."""
    parts = []
    if memory.get("name"):
        parts.append("you're %s" % memory["name"])
    if memory.get("year"):
        parts.append("a %s" % memory["year"])
    if memory.get("studies"):
        parts.append("studying %s" % memory["studies"])
    if memory.get("from"):
        parts.append("from %s" % memory["from"])
    if not parts:
        parts.append("you're a student here")
    sentence = ", ".join(parts)
    if memory.get("project"):
        sentence += ", and you want to do your final-year project on %s" % memory["project"]
    elif memory.get("interests"):
        sentence += ", and you're into %s" % " and ".join(memory["interests"][-2:])
    return sentence[0].upper() + sentence[1:] + "."


def demonstration(memory: dict) -> list:
    """Worked examples of the NPC using what it remembers.

    Facts alone in the prompt are not enough on this model -- that was the
    lesson of the campus facts, which went from 6/11 recalled to 11/11 once
    demonstrated. Here the obstacle is specific: the adapters were trained to
    say "I don't remember" about anything outside their persona, and a single
    greeting demo ("do you remember me" -> "Of course, Priya") did not carry
    over to direct questions ("what's my name?" -> "It's not important").

    So the second demo answers a question *about the player* with every
    remembered fact at once, which shows the model that such questions are
    answerable from what it was told. Its phrasing is deliberately unlike the
    probes in evaluation/run_player_memory.py, so that eval measures transfer and not
    copying. (A third, past-tense demo -- "what did I tell you earlier" --
    was tried against the "I don't remember" reflex and lowered recall.)

    The server places these turns last, just before the player's message;
    see _build_messages for why.
    """
    if not memory:
        return []
    demo = []
    if memory.get("name"):
        demo += [
            {"role": "user", "content": "do you remember me"},
            {"role": "assistant", "content": "Of course, %s." % memory["name"]},
        ]
    demo += [
        {"role": "user", "content": "so what do you know about me"},
        {"role": "assistant", "content": _about(memory)},
    ]
    return demo
