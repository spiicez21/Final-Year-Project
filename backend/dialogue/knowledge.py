"""Which of an NPC's facts does this message need?

Why this exists. With all of an NPC's campus facts in one system-prompt block,
the model used them loosely: asked "where's the canteen?", all five NPCs said
"second floor" with "The first floor has the teaching labs and the canteen"
in the prompt -- three near-identical floor sentences blur together on a 1.1B
model. So the message picks its facts, and only those are shown, as a turn
just before the question (composer.py). 31/50 -> 42/50 on held-out phrasings
(evaluation/run_grounding.py).

Word overlap, not embeddings: the fact lists are a dozen short sentences, the
scoring is inspectable, and it costs no model download or latency. A small
synonym table maps the everyday words players use onto the words the facts
use ("lunch" -> canteen). General meanings only, not tuned to test questions.

Weak words. A word that turns up in all sorts of sentences ("year", "work",
"day") is not evidence on its own: "what should I do for my final year
project" matched "about 78,000 a year" and the head of department answered a
project question with his salary. A sentence needs at least one strong word
in common with the message, except that a history question ("how long have
you...") may reach the NPC's background through a weak one.
"""

import re

from .text import split_sentences

# Everyday word -> the word(s) a fact would use.
SYNONYMS = {
    "lunch": ("canteen",), "eat": ("canteen",), "food": ("canteen",), "hungry": ("canteen",),
    "cafe": ("canteen",), "café": ("canteen",), "coffee": ("canteen",), "breakfast": ("canteen",),
    "snack": ("canteen", "shop"), "snacks": ("canteen", "shop"),
    "lecturer": ("academic",), "lecturers": ("academic",), "teachers": ("academic",),
    "academics": ("academic",), "professors": ("academic",), "faculty": ("academic",),
    "classes": ("term",), "semester": ("term",), "autumn": ("september",),
    "start": ("starts", "open"), "begin": ("starts",), "hours": ("hour", "shift", "open"),
    "finish": ("runs", "close"), "close": ("close", "runs"), "closing": ("close", "runs"),
    "ends": ("runs", "close"),
    "timing": ("open", "shift"), "timings": ("open", "shift"), "opening": ("open",),
    "office": ("offices",), "lab": ("labs",), "classroom": ("classrooms",),
    "undergrads": ("undergraduates",), "students": ("undergraduates", "postgraduates"),
    "salary": ("earn", "grade", "scale"), "paid": ("earn", "grade", "scale"),
    "pay": ("earn", "grade", "scale"), "earn": ("earn", "grade", "scale"),
    "income": ("earn", "grade", "scale"), "wage": ("earn", "grade", "scale"),
    "team": ("staff",), "report": ("staff",),
    "module": ("teach",), "modules": ("teach",), "courses": ("teach",), "course": ("teach",),
    "teaching": ("teach",),
    # "how long" asks for a duration, which the facts state in years.
    "long": ("years",),
}

# Too common across sentences to count as evidence on their own.
WEAK = {"year", "years", "day", "time", "work", "worked", "job", "thing", "way", "lot", "good",
        "new", "make", "place", "people", "department", "campus"}

# Words that ask about someone's past; they point at background sentences.
HISTORY_CUES = {"before", "became", "become", "end", "ended", "started", "got", "get", "used",
                "previously", "originally", "background", "long", "year"}  # "how many years"

STOPWORDS = set("""
a an the is are was were be been am i you your yours me my we our it its this that these those
i've i'd i'll we're they're don't doesn't can't won't isn't
of in on at to for from with by about as and or but so do does did have has had can could would
should will what where when which who whom how why there here just any some much many very really
please tell know one get got i'm what's you're it's
""".split())

_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"^(i|i'm|i've|i'd|my|we|we're|the shop is mine)\b", re.IGNORECASE)


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str, expand: bool = True) -> set:
    """Content-word stems. `expand` adds synonyms -- for the player's message
    only: synonyms map player words onto fact words, and expanding the facts
    too let the shop's "close at 6pm" also claim "runs" and outrank "The open
    day runs from 10am to 4pm" for "when does the open day finish"."""
    out = set()
    # Apostrophes stay inside words: splitting "i'm" left a stray "m" that
    # matched every fact starting "I'm ...", so "i'm in 2nd year ece" retrieved
    # the head of department's salary.
    for w in re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", (text or "").lower()):
        if w in STOPWORDS:
            continue
        out.add(_stem(w))
        if expand:
            for syn in SYNONYMS.get(w, ()):
                out.add(_stem(syn))
    return out


_WEAK_STEMS = {_stem(w) for w in WEAK}


def _query_words(question: str) -> list:
    """One set per content word the player typed: the word's stem plus its
    synonyms. Scoring counts words, not stems, so a word with two synonyms
    ("students" -> undergraduates, postgraduates) cannot score twice."""
    groups = []
    for w in re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", (question or "").lower()):
        if w in STOPWORDS:
            continue
        groups.append({_stem(w)} | {_stem(s) for s in SYNONYMS.get(w, ())})
    return groups


def select(question: str, facts: list, history: list = (), k: int = 2) -> list:
    """Up to `k` sentences the message needs, best first; [] if none qualify."""
    words = _query_words(question)
    history_question = any(g & HISTORY_CUES for g in words)
    about_npc = bool(_SECOND_PERSON.search(question))
    scored = []
    pool = [(f, False) for f in facts] + [(h, True) for h in history]
    for i, (sentence, is_history) in enumerate(pool):
        terms = _terms(sentence, expand=False)
        matched = [g for g in words if g & terms]
        strong = [g for g in matched if (g & terms) - _WEAK_STEMS]
        reaches_history = is_history and history_question
        if not strong and not (reaches_history and matched):
            continue
        # A question put to "you" prefers the NPC's own facts: "how many students
        # do you support" is the counsellor's caseload, not the department's size.
        score = (len(matched) + len(strong) + (1 if reaches_history else 0)
                 + (1.5 if about_npc and _FIRST_PERSON.match(sentence) else 0))
        scored.append((-score, i, sentence))
    scored.sort()
    return [s for _, _, s in scored[:k]]


def topic(question: str, fact: str) -> str:
    """The message's own words that matched `fact`, e.g. "the canteen"."""
    terms = _terms(fact, expand=False)
    words = [w for w in re.findall(r"[a-zA-Z0-9'-]+", question)
             if w.lower() not in STOPWORDS and (
                 _stem(w.lower()) in terms
                 or any(_stem(s) in terms for s in SYNONYMS.get(w.lower(), ())))]
    return " ".join(words) if words else "that"


def demonstration(question: str, selected: list) -> list:
    """A just-had exchange in which the NPC states the facts this message needs."""
    if not selected:
        return []
    return [
        {"role": "user", "content": "what do you know about %s" % topic(question, selected[0])},
        {"role": "assistant", "content": " ".join(selected)},
    ]


def fact_pool(facts: list, job_line: str) -> list:
    return [f.strip() for f in facts if f and f.strip()] + split_sentences(job_line or "")
