"""What an NPC remembers about the player, and how it learns it.

Why structured facts and not the transcript. The transcript is weak memory
even within one visit (11/40 held-out recall probes) and none at all across
visits, because it does not survive (0/40). Short facts, kept per NPC and
persisted by the game, do survive, and are small enough to inspect, test and
reset. (evaluation/run_player_memory.py)

Why rules and not the model for extraction. A second generation per turn to
summarise "what did the player say" would be slow, unreliable and opaque.
Patterns over the player's own words are cheap and unit-tested, and a missed
pattern means the NPC forgets something -- a safer failure than remembering
something never said.

Written for how players actually type. The first version keyed names on
capital letters ("I'm Priya"), and a player typing "myself yuga, 3rd year cse"
had nothing saved at all -- so the NPC kept asking their name. Extraction now
runs on normalised text (text.normalize) and accepts lowercase names after a
clear cue, while refusing words that are obviously not names ("i'm fine").

Memory is a dict of slots, so updates merge predictably:
    {"name": "Yuga", "year": "third-year", "studies": "cse",
     "project": "robotics", "interests": ["ai"], "from": "Madurai"}
"""

import re

MAX_INTERESTS = 4

# Words that follow "i'm" without being a name. Checked on every "i'm X"
# candidate, because a lowercase name has no capital letter to vouch for it.
_NOT_NAMES = set("""
a an the here there sorry fine good great ok okay alright well not just so also from in at on
into with about for to of by as like doing interested looking thinking really very quite pretty
kinda new lost stressed worried going trying studying curious wondering excited nervous happy sad
tired confused back still only sure first second third final fourth fifth hoping planning applying
coming leaving waiting working learning free busy ready done late early hungry thirsty sleepy sick
ill bored scared afraid angry glad impressed aware able allowed supposed meant certain cool fresh
fresher senior junior student alumni single one your his her their my our this that it all too
more less gonna feeling homesick overwhelmed anxious stuck alive safe home outside inside around
""".split())

# Degree and branch names students give in a word. After "<ordinal> year" any
# of these is taken as the subject ("3rd year cse").
_BRANCHES = set("""
cse cs it ece eee eie ice mech mechanical civil aids aiml ai ds bca mca bsc msc btech mtech
be biotech chemical aero aeronautical automobile robotics mechatronics physics chemistry maths
mathematics commerce bcom mcom bba mba law medicine mbbs nursing pharmacy architecture
""".split())
_MULTIWORD_BRANCHES = ["computer science", "data science", "information technology",
                       "artificial intelligence", "electrical engineering",
                       "electronics and communication", "software engineering"]

_ORD = r"(first|second|third|fourth|final|1st|2nd|3rd|4th)"
_YEAR_WORDS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth"}

# Explicit name cues: a lowercase name is fine here.
_NAME_CUE = re.compile(
    r"\b(?:my name is|my name's|name is|name's|call me|you can call me|myself|this is|"
    r"i am called|i'm called|people call me)\s+([a-z][a-z]+(?:\s[a-z][a-z]+)?)", re.IGNORECASE)
# "i'm X" / "i am X": ambiguous, so X is checked against _NOT_NAMES and shape.
_NAME_IM = re.compile(r"\b(?:i'm|i am)\s+([a-z][a-z]+(?:\s[a-z][a-z]+)?)", re.IGNORECASE)
# Words that end a name rather than continue it ("arjun from madurai").
_JOINERS = {"and", "but", "so", "i", "i'm", "from", "here", "nice", "sir", "maam", "ma'am", "btw",
            "by", "the", "a", "an", "to", "what", "how", "who", "in", "at", "studying", "doing"}

_YEAR = re.compile(
    r"\b(?:i'm|i am)(?: a| an| in)?(?: my)? " + _ORD + r"[- ]?(?:year|yr)"
    r"|(?:^|,\s*|\band\s+)(?:a |an |in )?(?:my )?" + _ORD + r"[- ]?(?:year|yr)\b", re.IGNORECASE)
_STUDIES = re.compile(
    r"\bi(?:'m| am)? ?(?:studying|study|doing a degree in|doing|majoring in|major in|in) "
    r"([a-z][a-z ]{1,40}?)(?=[.,!?]| and | but |$)", re.IGNORECASE)
_AFTER_YEAR = re.compile(_ORD + r"[- ]?(?:year|yr)s?\s+([a-z&]+(?: [a-z]+)?)", re.IGNORECASE)
_PROJECT = re.compile(
    r"\b(?:project (?:on|about|in)|(?:do|doing|build|building|make|making) (?:a |my )?"
    r"(?:final[- ]year )?project (?:on|about|in))\s+([a-z0-9][a-z0-9 +#.-]{1,40}?)"
    r"(?=[.,!?]| and | but | so |$)", re.IGNORECASE)
_INTEREST = re.compile(
    r"\b(?:i'm interested in|i am interested in|(?<!you )(?<!you're )interested in|i'm into|"
    r"i really like|i like|i love|i enjoy|i want to (?:work on|learn|study))"
    r"\s+([a-z0-9][a-z0-9 +#.-]{1,40}?)(?=[.,!?]| and | but | so |$)", re.IGNORECASE)
_FEELING = re.compile(
    r"\bi(?:'m| am| feel| feel really| feel so)? (?:really |so |very |a bit |quite )?"
    r"(stressed|worried|anxious|nervous|excited|tired|lost|confused|overwhelmed|homesick)\b",
    re.IGNORECASE)
_FROM = re.compile(r"\b(?:i'm|i am|i come)\s+(?:\w+\s+)?(?:originally\s+)?from\s+"
                   r"([a-z][a-z]+(?:\s[a-z][a-z]+)?)", re.IGNORECASE)
_FROM_IN_INTRO = re.compile(r"\bfrom\s+([a-z][a-z]+(?:\s[a-z][a-z]+)?)\s*[.!]?$", re.IGNORECASE)
_PLACE_STOP = {"the", "a", "an", "here", "there", "and", "but", "so", "home", "outside", "abroad",
               "college", "school", "class", "campus", "library", "canteen", "department"}


def _looks_like_name(word: str) -> bool:
    w = word.lower()
    return (w not in _NOT_NAMES and w not in _BRANCHES and len(w) >= 2 and w.isalpha()
            and not re.search(r"(ing|ed|ly|ous|ful|ive|able)$", w))


def _clean_name(raw: str, strict: bool) -> str:
    words = raw.split()
    if len(words) == 2 and (words[1].lower() in _JOINERS or not _looks_like_name(words[1])):
        words = words[:1]
    if not words or words[0].lower() in _JOINERS:
        return ""
    if strict and not _looks_like_name(words[0]):
        return ""
    if not strict and words[0].lower() in _NOT_NAMES:
        return ""
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)


def extract(message: str) -> dict:
    """Facts the player stated about themselves in this one message.

    `message` should be normalised first (text.normalize). Returns only the
    slots found; an empty dict means nothing was learned.
    """
    text = (message or "").strip().replace("’", "'")
    found = {}
    is_question = text.endswith("?") or bool(re.match(
        r"^(what|where|when|who|why|how|which|is|are|am|do|does|did|can|could)\b", text, re.I))

    m = _NAME_CUE.search(text)
    name = _clean_name(m.group(1), strict=False) if m else ""
    if not name and not is_question:
        for m in _NAME_IM.finditer(text):
            name = _clean_name(m.group(1), strict=True)
            if name:
                break
    if name:
        found["name"] = name

    if not is_question:
        m = _YEAR.search(text)
        if m:
            ordinal = (m.group(1) or m.group(2)).lower()
            found["year"] = _YEAR_WORDS.get(ordinal, ordinal) + "-year"

    low = text.lower()
    for branch in _MULTIWORD_BRANCHES:
        if re.search(r"\b(i'm|i am|studying|study|doing|in|year)\b[^.?!]*\b" + branch + r"\b", low):
            found["studies"] = branch
            break
    if "studies" not in found:
        m = _AFTER_YEAR.search(text)
        if m and m.group(2).split()[0].lower() in _BRANCHES:
            found["studies"] = m.group(2).split()[0].lower()
    if "studies" not in found and not is_question:
        m = _STUDIES.search(text)
        if m:
            value = m.group(1).strip().lower()
            first = value.split()[0]
            if first in _BRANCHES or (first not in _NOT_NAMES and not re.match(_ORD, first)
                                      and first not in ("my", "the", "a", "an", "it", "that")):
                if not re.search(r"\b(year|project)\b", value):
                    found["studies"] = value

    m = _PROJECT.search(text)
    project_span = None
    if m and m.group(1).strip().lower() not in ("it", "that", "this"):
        found["project"] = m.group(1).strip()
        project_span = m.span()

    interests = []
    for m in _INTEREST.finditer(text):
        value = m.group(1).strip()
        overlaps_project = project_span and m.start() < project_span[1] and project_span[0] < m.end()
        if value.lower() in ("", "it", "that", "this", "you", "them") or overlaps_project \
                or re.search(r"\bproject\b", value, re.I):
            continue
        interests.append(value)
    if interests:
        found["interests"] = interests

    m = _FEELING.search(text)
    if m and not re.search(r"\bnot\s+(really\s+|so\s+|very\s+)?" + m.group(1), low):
        found["feeling"] = m.group(1).lower()

    m = _FROM.search(text)
    # In a self-introduction a bare "from X" is where the player is from:
    # "this is meena, 1st year it from coimbatore". Only there -- "i came from
    # the library" is not an introduction.
    if not m and (name or "year" in found) and not is_question:
        m = _FROM_IN_INTRO.search(text)
    if m:
        place = [w for w in m.group(1).split()]
        if place and place[0].lower() not in _PLACE_STOP:
            if len(place) == 2 and (place[1].lower() in _JOINERS or place[1].lower() in _PLACE_STOP):
                place = place[:1]
            found["from"] = " ".join(w[:1].upper() + w[1:].lower() for w in place)

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
                if item.lower() not in (m.lower() for m in merged):
                    merged.append(item)
            out["interests"] = merged[-MAX_INTERESTS:]
        else:
            out[key] = value
    return out


def render(memory: dict) -> str:
    """Memory as short third-person sentences for the system prompt ("they":
    the player's gender is never stated, so none is assumed)."""
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


def about(memory: dict) -> str:
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


def recall_line(memory: dict, slots: list) -> str:
    """The saved answer to a recall question, as the NPC would say it.

    The guard's last resort when the "You told me" restart still produced a
    non-answer ("You told me." to "what was my project about"). It states
    only what is saved, so it cannot invent anything.
    """
    say = {
        "name": lambda m: "You're %s." % m["name"],
        "year": lambda m: "You're a %s student." % m["year"],
        "studies": lambda m: "You study %s." % m["studies"],
        "from": lambda m: "You're from %s." % m["from"],
        "project": lambda m: "You want to do your final-year project on %s." % m["project"],
        "interests": lambda m: "You're into %s." % " and ".join(m["interests"][-2:]),
    }
    lines = [say[s](memory) for s in slots if s in say and memory.get(s)]
    if not lines and "project" in slots and memory.get("interests"):
        lines = [say["interests"](memory)]
    return " ".join(lines) if lines else about(memory)


def demonstration(memory: dict) -> list:
    """Worked examples of the NPC using what it remembers.

    Facts alone in the prompt were not enough on this model; a just-had
    exchange in which the NPC uses them is (run_memory_placement.py: stated
    in the system prompt 11/40, demonstrated before the question 21/40). The
    phrasings are deliberately unlike the probes in run_player_memory.py so
    that evaluation measures transfer, not copying.

    The composer shows this only when the message is about the player; shown
    on every turn, it crowded out the NPC's own job.
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
        {"role": "assistant", "content": about(memory)},
    ]
    return demo
