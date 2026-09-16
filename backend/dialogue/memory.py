"""What an NPC remembers about the player: storing, merging, and saying it.

Reading facts out of what the player types is extractor.py (a model). This
module decides what to keep and how the NPC uses it.

Why structured facts and not the transcript. The transcript is weak memory
even within one visit (11/40 held-out recall probes) and none at all across
visits, because it does not survive (0/40). Short facts, kept per NPC and
persisted by the game, do survive, and are small enough to inspect and reset.
(evaluation/run_player_memory.py)

Stored shape -- plain JSON, so the game can save it and a tester can edit it:

    {"name": "Yugabharathi", "department": "CSE", "section": "D",
     "interests": ["machine learning"],
     "_confidence": {"name": 0.93, "department": 0.88, "section": 0.81}}

Confidence decides overwrites. A slot keeps the extractor's score for the
reading that filled it, and a later reading with a different value replaces
it only if it is nearly as confident (REPLACE_MARGIN). So a clear "my name is
yugabharathi" is not renamed by a hesitant misreading of a later line, while a
clear correction ("actually my name is bharathi") still wins. Slots saved
before confidences existed count as 0, so anything the old pattern rules
misfiled ("name": "Class") is replaced by the first model reading.
"""

MAX_INTERESTS = 4
REPLACE_MARGIN = 0.1
CONFIDENCE = "_confidence"

# Slot names used before the model extractor; migrated on merge.
_RENAMED = {"studies": "department", "from": "hometown"}

SLOTS = ("name", "year", "department", "section", "college", "hometown", "project", "interests",
         "feeling")


def facts(memory: dict) -> dict:
    """The remembered slots without bookkeeping keys, old names migrated."""
    out = {}
    for key, value in (memory or {}).items():
        if key.startswith("_"):
            continue
        out[_RENAMED.get(key, key)] = value
    return out


def _display(slot: str, value: str) -> str:
    """How a value is kept and said: the player's words, tidied, never invented."""
    v = " ".join(str(value).split()).strip(" .,!?;:'\"")
    low = v.lower()
    if slot == "year":
        for suffix in (" year", " yr", " years", " yrs"):
            if low.endswith(suffix):
                v = v[: -len(suffix)]
        return v.lower()
    if slot in ("name", "hometown", "college"):
        return " ".join(w[:1].upper() + w[1:] for w in v.split())
    if slot == "department":
        # Abbreviations are spoken as letters ("CSE"); names of subjects as typed.
        return v.upper() if len(v.replace(" ", "")) <= 5 and " " not in v else v
    if slot == "section":
        return v.upper() if len(v) <= 2 else v
    return v


def merge(memory: dict, readings: dict) -> tuple:
    """Folds one message's extractor readings into what is remembered.

    `readings` is extractor output: {slot: (value, score)}, with interests as
    a list of (value, score). Returns (new memory, {slot: value} that changed).
    """
    old_conf = dict((memory or {}).get(CONFIDENCE, {}))
    out = facts(memory)
    conf = {_RENAMED.get(k, k): v for k, v in old_conf.items()}
    changed = {}
    for slot, reading in (readings or {}).items():
        if slot == "interests":
            current = list(out.get("interests", []))
            added = []
            for value, _score in reading:
                shown = _display(slot, value)
                if shown.lower() not in (c.lower() for c in current):
                    current.append(shown)
                    added.append(shown)
            if added:
                out["interests"] = current[-MAX_INTERESTS:]
                changed["interests"] = added
            continue
        value, score = reading
        shown = _display(slot, value)
        if not shown:
            continue
        saved = out.get(slot)
        saved_score = float(conf.get(slot, 0.0))
        if saved is not None and str(saved).lower() == shown.lower():
            conf[slot] = max(saved_score, score)
        elif saved is None or score >= saved_score - REPLACE_MARGIN:
            out[slot] = shown
            conf[slot] = score
            changed[slot] = shown
    if conf:
        out[CONFIDENCE] = {k: round(v, 3) for k, v in conf.items() if k in out}
    return out, changed


def _year_phrase(year: str) -> str:
    return "final year" if year == "final" else "%s year" % year


def render(memory: dict) -> str:
    """Memory as short third-person sentences for the system prompt ("they":
    the player's gender is never stated, so none is assumed)."""
    m = facts(memory)
    if not m:
        return ""
    lines = []
    if m.get("name"):
        lines.append("Their name is %s." % m["name"])
    if m.get("year"):
        lines.append("They are in %s." % _year_phrase(m["year"]))
    if m.get("department"):
        lines.append("They study %s%s." % (m["department"], (", section %s" % m["section"]) if m.get("section") else ""))
    elif m.get("section"):
        lines.append("They are in section %s." % m["section"])
    if m.get("college"):
        lines.append("They go to %s." % m["college"])
    if m.get("hometown"):
        lines.append("They are from %s." % m["hometown"])
    if m.get("project"):
        lines.append("They are doing a project on %s." % m["project"])
    if m.get("interests"):
        lines.append("They are interested in %s." % ", ".join(m["interests"]))
    if m.get("feeling"):
        lines.append("Earlier they said they felt %s." % m["feeling"])
    return " ".join(lines)


def about(memory: dict) -> str:
    """Everything remembered, as one second-person sentence an NPC could say."""
    m = facts(memory)
    parts = []
    if m.get("name"):
        parts.append("you're %s" % m["name"])
    if m.get("year"):
        parts.append("in %s" % _year_phrase(m["year"]))
    if m.get("department"):
        parts.append("studying %s%s" % (m["department"], (" %s" % m["section"]) if m.get("section") else ""))
    if m.get("hometown"):
        parts.append("from %s" % m["hometown"])
    if not parts:
        parts.append("you're a student here")
    sentence = ", ".join(parts)
    if m.get("project"):
        sentence += ", and you're doing a project on %s" % m["project"]
    elif m.get("interests"):
        sentence += ", and you're into %s" % " and ".join(m["interests"][-2:])
    return sentence[0].upper() + sentence[1:] + "."


def recall_line(memory: dict, slots: list) -> str:
    """The saved answer to a recall question, as the NPC would say it.

    The guard's last resort when the "You told me" restart still produced a
    non-answer. It states only what is saved, so it cannot invent anything.
    """
    m = facts(memory)
    say = {
        "name": lambda: "You're %s." % m["name"],
        "year": lambda: "You're in %s." % _year_phrase(m["year"]),
        "department": lambda: "You study %s." % m["department"],
        "section": lambda: "You're in section %s." % m["section"],
        "college": lambda: "You go to %s." % m["college"],
        "hometown": lambda: "You're from %s." % m["hometown"],
        "project": lambda: "You're doing a project on %s." % m["project"],
        "interests": lambda: "You're into %s." % " and ".join(m["interests"][-2:]),
    }
    lines = [say[s]() for s in slots if s in say and m.get(s)]
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
    m = facts(memory)
    if not m:
        return []
    demo = []
    if m.get("name"):
        demo += [
            {"role": "user", "content": "do you remember me"},
            {"role": "assistant", "content": "Of course, %s." % m["name"]},
        ]
    demo += [
        {"role": "user", "content": "so what do you know about me"},
        {"role": "assistant", "content": about(memory)},
    ]
    return demo
