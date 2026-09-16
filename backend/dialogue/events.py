"""News: incidents players report, and how NPCs pass them on.

When a player tells an NPC about something happening -- "i saw a man carrying
a knife near the gym" -- the extractor marks the incident and where it
happened (extractor.read). That becomes an event. The game keeps the event log
and decides who has heard what (Systems/NPC/world_event_store.gd): the NPC
that was told knows it at once, and the news then spreads NPC to NPC over
time, with no role treated specially. This module is the server's side: turning
an extraction into an event, saying an event the way an NPC would, and
checking replies for news the NPC should not have.

That last check is Knowledge Boundary Drift applied to rumours. An NPC's
visibility set here grows at runtime, as it hears things, and a reply that
mentions an event it has not heard is a leak -- the "information diffusion"
case the paper names but had not measured.

Two kinds of news, kept the same way:

  incident  something happening that the player reports ("a guy with a gun near
            the old library"). The extractor marks the incident and the place.
  note      anything else the player states that is worth passing on ("the lift
            in the physics block is stuck", "the robotics club meets on
            Friday"). Nothing is extracted: the player's own sentence is kept
            and quoted, because an invented paraphrase of a fact nobody can
            check is worse than a quote. Facts the player states *about
            themselves* are not notes -- those are player memory (memory.py).

An NPC passes an incident on when greeted, because it is urgent; a note only
when a question reaches it. Both spread NPC to NPC in the game.

Event shape, as sent by the game:
    {"id": "e3", "what": "a man carrying a knife", "where": "the gym",
     "heard_from": "player" | "<NPC name>", "fresh": true}
"fresh" means this NPC has not yet mentioned it to the player.
"""

import re

MIN_EVENT_SCORE = 0.4   # floor for an incident passed in directly; the extractor already applies its own

_CONTENT = re.compile(r"[a-z0-9]+")
_FILLER = set("""
a an the is are was were be been i you he she it they we my your his her their our me him them us
of in on at to for from with by about as and or but so near behind outside inside next way just now
someone somebody some guy man woman people person student there here this that saw see seen told
""".split())


# A note needs this many content words: "ok thanks" and "yeah" are not news.
MIN_NOTE_WORDS = 3
# ... and at most this many characters. A player once typed a department list
# that repeated "AI" forty times; it was stored whole and passed around.
MAX_NOTE_CHARS = 200


def event_from(reading: dict, message: str, is_report: bool):
    """An incident from the extractor's event reading, or None.

    The incident span is the event. If the intent classifier says this is a
    report but the extractor found no incident span, the player's own line is
    kept as the description -- their words, not an invention.
    """
    incident = reading.get("incident")
    place = reading.get("place")
    raw = False
    if incident and incident[1] >= MIN_EVENT_SCORE:
        what, score = incident
    elif is_report:
        what, score, raw = (message or "").strip(), 0.0, True
    else:
        return None
    if not what:
        return None
    return {"what": what, "where": place[0] if place else "", "score": round(float(score), 3),
            "raw": raw, "kind": "incident"}


def note_from(message: str):
    """Anything else the player tells an NPC, kept as a note, or None.

    The whole sentence is kept as typed. The caller decides *when* to ask for
    one (turn.py: a statement that taught no fact about the player), so this
    only judges whether there is enough here to be worth repeating.
    """
    what = (message or "").strip()
    words = _content_words(what)
    if len(words) < MIN_NOTE_WORDS or "?" in what:
        return None
    if len(what) > MAX_NOTE_CHARS:
        what = what[:MAX_NOTE_CHARS].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return {"what": what, "where": "", "score": 0.0, "raw": True, "kind": "note"}


def sentence(event: dict, npc_name: str = "") -> str:
    """How an NPC would state an event it has heard."""
    what = event.get("what", "").strip().rstrip(".")
    where = event.get("where", "").strip()
    source = event.get("heard_from", "player")
    if event.get("kind") == "note":
        if source in ("", "player"):
            return 'You told me: "%s".' % what
        return '%s told me a student said: "%s".' % (source, what)
    if event.get("raw"):
        # Only the player's own words were kept: quote them, don't rephrase.
        if source in ("", "player"):
            return 'You reported: "%s".' % what
        return '%s told me a student reported: "%s".' % (source, what)
    place = (" at %s" % where) if where else ""
    if source in ("", "player"):
        return "You reported %s%s." % (what, place)
    return "%s told me a student reported %s%s." % (source, what, place)


def _content_words(text: str) -> set:
    return {w for w in _CONTENT.findall((text or "").lower()) if w not in _FILLER and len(w) > 2}


def mentions(reply: str, event: dict) -> bool:
    """Does the reply talk about this event?

    At least two of the event's content words (from what happened and where),
    or all of them when the event has fewer than two -- "knife" alone could be
    about cooking; "knife" and "gym" together are the event.
    """
    words = _content_words(event.get("what", "")) | _content_words(event.get("where", ""))
    if not words:
        return False
    found = words & _content_words(reply)
    return len(found) >= min(2, len(words))


def leaks(reply: str, unheard: list) -> list:
    """Ids of events the reply mentions but this NPC has not heard."""
    return [e.get("id", "") for e in unheard or [] if mentions(reply, e)]


def lead_words(event: dict) -> str:
    """The opening of the news sentence up to what happened, for a restart:
    "Halvorsen told me a student reported" -- the model still says what."""
    full = sentence(event)
    what = event.get("what", "").strip()
    cut = full.find(what) if what else -1
    return full[:cut].rstrip(' :"') if cut > 0 else full.split(" ")[0]


def rumour_demonstration(event: dict) -> list:
    """A just-had exchange in which the NPC passes on news it heard."""
    return [
        {"role": "user", "content": "anything new around here"},
        {"role": "assistant", "content": "Have you heard? " + sentence(event)},
    ]


def report_demonstration() -> list:
    """How to take a report seriously: thank the player, say what happens next.

    Shown only when the intent classifier says the player is reporting
    something, so the reply acknowledges the report instead of treating it as
    small talk. Generic on purpose -- the NPC's persona decides what "letting
    the right people know" sounds like. Whether it helps is measured by
    evaluation/run_events.py, not assumed.
    """
    return [
        {"role": "user", "content": "someone left a bag unattended by the main gate"},
        {"role": "assistant", "content": "Thanks for telling me. I'll make sure the right people hear about it."},
    ]
