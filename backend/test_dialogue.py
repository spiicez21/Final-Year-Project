"""Unit tests for backend/dialogue. No model needed: the generator is scripted.

    .venv/Scripts/python.exe backend/test_dialogue.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialogue import Persona, TurnConfig, TurnInput, run_turn  # noqa: E402
from dialogue import composer, guard, intent, knowledge, memory  # noqa: E402
from dialogue.text import normalize, similar  # noqa: E402


def learn(message):
    return memory.extract(normalize(message))


# --- understand ----------------------------------------------------------------

def test_normalize_expands_shorthand_and_keeps_names():
    assert normalize("what do u do") == "what do you do"
    assert normalize("whats ur salary") == "what's your salary"
    assert normalize("i m Yuga") == "i'm Yuga"
    assert normalize("my student id") == "my student id"   # "id" is a word, not "i'd"
    assert normalize("I feel ill") == "I feel ill"


def test_intent():
    kinds = {
        "hi": intent.GREETING, "hi again": intent.GREETING, "ok": intent.ACK, "hmm": intent.ACK,
        "thanks bye": intent.FAREWELL, "what is my name": intent.RECALL,
        "do you remember me": intent.RECALL, "what do you do": intent.ABOUT_NPC,
        "where is the canteen": intent.QUESTION, "myself yuga, 3rd year cse": intent.STATEMENT,
        "remember to lock your bike": intent.STATEMENT,
    }
    for message, kind in kinds.items():
        assert intent.classify(normalize(message)).kind == kind, (message, intent.classify(message))
    assert intent.classify("what is my name").recall_slots == ["name"]


# --- learn -----------------------------------------------------------------------

EXTRACT = [
    ("Hi, I'm Priya.", {"name": "Priya"}),
    ("my name is Arjun Kumar", {"name": "Arjun Kumar"}),
    ("Call me Sam", {"name": "Sam"}),
    ("I'm a first-year student", {"year": "first-year"}),
    ("I am in my 2nd year", {"year": "second-year"}),
    ("I'm studying computer science", {"studies": "computer science"}),
    ("I'm interested in compilers", {"interests": ["compilers"]}),
    ("I'm thinking of doing a final-year project on machine learning.", {"project": "machine learning"}),
    ("I really like robotics and I love game design", {"interests": ["robotics", "game design"]}),
    ("I'm really stressed about exams", {"feeling": "stressed"}),
    ("I'm from Chennai", {"from": "Chennai"}),
    ("I’m Arjun", {"name": "Arjun"}),
    ("I'm Arjun, a first-year, interested in networks",
     {"name": "Arjun", "year": "first-year", "interests": ["networks"]}),
    # How players type -- none of these saved anything before the rewrite.
    ("im yuga", {"name": "Yuga"}),
    ("i am yuga", {"name": "Yuga"}),
    ("myself Yuga, 3rd year cse", {"name": "Yuga", "year": "third-year", "studies": "cse"}),
    ("this is yuga", {"name": "Yuga"}),
    ("hi im yuga bharathi", {"name": "Yuga Bharathi"}),
    ("i am arjun from madurai", {"name": "Arjun", "from": "Madurai"}),
    ("im in 2nd year ece", {"year": "second-year", "studies": "ece"}),
    ("im from chennai", {"from": "Chennai"}),
]

NOT_LEARNED = [
    ("i'm fine", "name"), ("im good thanks", "name"), ("im looking for the canteen", "name"),
    ("im new here", "name"), ("im hungry", "name"), ("I'm Sorry to bother you", "name"),
    ("what's ur name", None), ("What do you teach?", None), ("are you interested in AI?", None),
    ("what do first-year students study?", None), ("I'm not stressed at all", None),
    ("I'm doing well thanks", None), ("im in the library", None), ("im from the library", None),
    # A project is one slot, not also a junk interest.
    ("i want to do my final-year project on robotics", "interests"),
]


def test_extract():
    for message, expected in EXTRACT:
        got = learn(message)
        for k, v in expected.items():
            assert got.get(k) == v, "%r: %s expected %r got %r" % (message, k, v, got)


def test_not_learned():
    for message, slot in NOT_LEARNED:
        got = learn(message)
        assert (got == {}) if slot is None else (slot not in got), "%r learned %r" % (message, got)


def test_merge():
    m = memory.merge({}, {"name": "Priya", "interests": ["compilers"]})
    m = memory.merge(m, {"name": "Priya S", "interests": ["robotics", "Compilers"]})
    assert m["name"] == "Priya S" and m["interests"] == ["compilers", "robotics"]
    for i in range(10):
        m = memory.merge(m, {"interests": ["topic%d" % i]})
    assert len(m["interests"]) == memory.MAX_INTERESTS


def test_render_is_gender_neutral():
    text = " " + memory.render({"name": "Priya", "year": "first-year", "project": "robotics"}) + " "
    for word in (" he ", " she ", " his ", " her "):
        assert word not in text.lower()


# --- retrieve ----------------------------------------------------------------------

FACTS = [
    "The ground floor has the lecture rooms and the seminar hall.",
    "The first floor has the teaching labs and the canteen.",
    "The second floor has classrooms and the staff offices.",
    "There are 34 academic staff and 11 professional services staff.",
    "I'm on grade 10 with a head-of-department allowance, so about 78,000 a year.",
]


def test_retrieval_picks_the_right_floor():
    assert knowledge.select("where's the canteen?", FACTS)[0] == FACTS[1]
    assert knowledge.select("where are the staff offices?", FACTS)[0] == FACTS[2]
    assert FACTS[1] in knowledge.select("where can I grab lunch?", FACTS)
    assert FACTS[3] in knowledge.select("how many lecturers work here?", FACTS)


def test_retrieval_weak_word_alone_is_not_evidence():
    # "year" matched the salary sentence and a project question got a salary.
    assert FACTS[4] not in knowledge.select("what should i do for my final year project", FACTS)
    assert FACTS[4] in knowledge.select(normalize("what is ur salary"), FACTS)
    assert knowledge.select("do you like football?", FACTS) == []


# --- compose -----------------------------------------------------------------------

def _input(message, history=(), mem=None, facts=()):
    return TurnInput(archetype="executive", message=message, max_tokens=40,
                     persona=Persona(name="Halvorsen", occupation="the head of department",
                                     job_line="I run the department, so timetables and hiring.",
                                     background="I've been head for three years."),
                     facts=list(facts), history=list(history), player_memory=mem or {})


def test_memory_turn_only_when_asked_about_the_player():
    mem = {"name": "Yuga", "interests": ["ai"]}
    cfg = TurnConfig()
    job = composer.compose(_input("what do you do", mem=mem), "what do you do",
                           intent.classify("what do you do"), mem, [], cfg)
    assert not any("do you remember me" in m["content"] for m in job)
    recall = composer.compose(_input("what is my name", mem=mem), "what is my name",
                              intent.classify("what is my name"), mem, [], cfg)
    assert recall[-3]["content"] == "so what do you know about me"


def test_transcript_drops_repeated_replies():
    history = []
    for q, r in [("what do you do", "I don't know anything about that."),
                 ("ok", "I don't know anything about that."),
                 ("where is the canteen", "The first floor.")]:
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
    turns = composer.transcript(history, 8, TurnConfig())
    replies = [t["content"] for t in turns if t["role"] == "assistant"]
    assert replies == ["I don't know anything about that.", "The first floor."]


# --- check and repair ---------------------------------------------------------------

def test_clean_flattens_lists_and_caps_sentences():
    assert guard.clean("1. Research your topic thoroughly.\n\n2. Plan it.\n\n3.") == \
        "Research your topic thoroughly."
    assert guard.clean("One. Two. Three. Four.") == "One. Two."
    assert guard.clean_basic("Sure, here's a reply: Hello there") == "Hello there"


class Scripted:
    """A fake model: returns scripted replies in order, records what it was asked."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, max_tokens, prefill="", repeat_penalty=1.0):
        self.calls.append({"prefill": prefill, "repeat_penalty": repeat_penalty, "messages": messages})
        text = self.replies.pop(0)
        return prefill + text if prefill else text


def test_invented_name_with_nothing_saved_is_not_confirmed():
    gen = Scripted("You told me your name was Emily.", " your name yet.")
    out = run_turn(_input("what is my name", mem={"interests": ["ai"]}), gen)
    assert gen.calls[1]["prefill"] == "I don't think you've told me"
    assert "Emily" not in out.response and out.repairs[0] == "recall_unknown"


def test_recall_refusal_with_name_saved_is_repaired():
    gen = Scripted("That's not something I can tell you.", " your name is Yuga.")
    out = run_turn(_input("what is my name", mem={"name": "Yuga"}), gen)
    assert gen.calls[1]["prefill"] == "You told me" and "Yuga" in out.response


def test_dodged_recall_is_repaired():
    gen = Scripted("Yes, I remember you.", " your name is Arjun.")
    out = run_turn(_input("u remember my name?", mem={"name": "Arjun"}), gen)
    assert out.problems == ["recall_miss"] and "Arjun" in out.response
    fine = run_turn(_input("do you remember me", mem={"name": "Arjun"}), Scripted("Of course!"))
    assert fine.repairs == []


def test_statements_do_not_retrieve_facts():
    salary = "I'm on grade 10 with a head-of-department allowance, so about 78,000 a year."
    out = run_turn(_input("im in 2nd year ece", facts=[salary]), Scripted("Good to know."))
    assert out.facts_used == []
    assert knowledge.select(normalize("im in 2nd year ece"), [salary]) == []   # the stray "m"


def test_telling_is_not_asking():
    # "my project" in a statement is not a recall question.
    assert intent.classify("i want to do my final year project on chatbots").kind == intent.STATEMENT
    assert intent.classify(normalize("what was my project about")).kind == intent.RECALL
    assert intent.classify("remind me who i am").kind == intent.RECALL
    assert intent.classify("remember to lock your bike").kind == intent.STATEMENT


def test_from_in_an_introduction():
    assert learn("this is meena, 1st year it from coimbatore") == {
        "name": "Meena", "year": "first-year", "studies": "it", "from": "Coimbatore"}
    assert "from" not in learn("i just came from the library")


def test_recall_that_stays_empty_falls_back_to_memory():
    gen = Scripted("That's not something I can tell you.", ".")
    out = run_turn(_input("what was my project about", mem={"project": "chatbots"}), gen)
    assert out.response == "You want to do your final-year project on chatbots."
    assert out.repairs[:2] == ["recall", "recall_fallback"]


def test_acknowledgement_gets_an_acknowledgement():
    gen = Scripted("I'm sorry, I don't have that information.", " Let me know if you need anything.")
    out = run_turn(_input("alright"), gen)
    assert out.problems == ["ack_refusal"] and out.response == "Okay. Let me know if you need anything."


def test_students_question_picks_the_right_fact():
    dept = "The department has about 480 undergraduates and 95 taught postgraduates."
    caseload = "I usually have about 40 students on my caseload at any one time."
    assert knowledge.select(normalize("how many students r in the department"), [caseload, dept])[0] == dept
    assert knowledge.select("how many students do you support", [dept, caseload])[0] == caseload


def test_wrong_name_is_corrected_without_regenerating():
    gen = Scripted("Bye, Emily.")
    out = run_turn(_input("ok bye", mem={"name": "Yuga"}), gen)
    assert out.response == "Bye, Yuga." and len(gen.calls) == 1


def test_question_already_answered_is_dropped():
    gen = Scripted("Nice to meet you. What's your name?")
    out = run_turn(_input("i'm in 3rd year", mem={"name": "Yuga"}), gen)
    assert out.response == "Nice to meet you." and "asks_known" in out.repairs


def test_self_denial_restarts_on_the_job_line():
    gen = Scripted("I don't do anything.", " the department, mostly meetings.")
    out = run_turn(_input("what do u do"), gen)
    assert gen.calls[1]["prefill"] == "I run" and out.response.startswith("I run")


def test_repeat_regenerates_without_the_copied_exchange():
    history = [{"role": "user", "content": "where is the canteen"},
               {"role": "assistant", "content": "The canteen is on the first floor near the labs."}]
    gen = Scripted("The canteen is on the first floor near the labs.", "Term starts in late September.")
    out = run_turn(_input("when does term start", history=history), gen)
    retry_prompt = " ".join(m["content"] for m in gen.calls[1]["messages"])
    assert "near the labs" not in retry_prompt and out.repairs == ["repeat"]
    assert out.response == "Term starts in late September."


def test_wrong_identity_restarts_on_the_job_line():
    gen = Scripted("I'm a librarian.", " the department, mostly meetings.")
    out = run_turn(_input("what do you do"), gen)
    assert out.problems == ["wrong_identity"] and gen.calls[1]["prefill"] == "I run"
    ok = run_turn(_input("what do you do"), Scripted("I'm the head of department."))
    assert ok.repairs == []


def test_fact_refusal_that_persists_falls_back_to_the_fact():
    fact = "There are 34 academic staff and 11 professional services staff."
    gen = Scripted("I'm not sure.", " are, so I don't know.")
    out = run_turn(_input("how many lecturers work here", facts=[fact]), gen)
    assert out.response == fact and out.repairs[:2] == ["fact", "fact_fallback"]


def test_asking_the_name_just_given_becomes_an_acknowledgement():
    gen = Scripted("Sure, what's your name?", " What brings you here?")
    out = run_turn(_input("myself yuga"), gen)
    assert gen.calls[1]["prefill"] == "Nice to meet you, Yuga." and out.repairs[0] == "introduced"


def test_eval_path_is_untouched():
    gen = Scripted("Sure, here you go: I patrol the city.")
    out = run_turn(TurnInput(archetype="police officer", message="what do you do"), gen)
    assert gen.calls[0]["messages"] == [
        {"role": "system", "content": "You are a police officer NPC in a modern city. Respond in a "
         "natural, contemporary voice consistent with your role. Never break character."},
        {"role": "user", "content": "what do you do"}]
    assert out.response == "I patrol the city." and out.repairs == []


def test_similar():
    assert similar("I'm the campus liaison officer.", "I'm the campus liaison officer, here to help.")
    assert not similar("The canteen is on the first floor.", "Term starts in September.")


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("  PASS  %s" % name)
            except Exception as exc:  # noqa: BLE001 - report every failure, not just asserts
                failures += 1
                print("  FAIL  %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%s" % ("ALL PASSED" if not failures else "%d FAILED" % failures))
    raise SystemExit(1 if failures else 0)
