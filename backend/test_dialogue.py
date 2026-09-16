"""Unit tests for backend/dialogue. No model needed: the generator is scripted.

    .venv/Scripts/python.exe backend/test_dialogue.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialogue import Persona, TurnConfig, TurnInput, run_turn  # noqa: E402
from dialogue import composer, events, guard, intent, knowledge, memory  # noqa: E402
from dialogue.text import normalize, similar  # noqa: E402


def no_facts(message, context=""):
    """Fake extractor for turns that are not about learning."""
    return {}


# Turn tests must not load a 600 MB model: swap in the fake. The one test of
# the real model uses REAL_EXTRACTOR.
from dialogue import extractor as _extractor  # noqa: E402
REAL_EXTRACTOR = _extractor.default
_extractor.default = type("NoModel", (), {"extract": staticmethod(no_facts),
                                          "read": staticmethod(lambda m, c="": ({}, {}))})()
# Intent likewise runs its pattern fallback here; the learned classifier is
# measured on labelled lines by evaluation/run_intent.py.
intent.default._ready = False


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
# What a line *means* is the extractor model's job and is measured on labelled
# lines by evaluation/run_memory_extraction.py. These tests cover what is done
# with its readings.

def test_confident_name_is_not_renamed_by_a_weaker_reading():
    # The reported bug: "my name is yugabharathi", then "i am class cse d".
    m, changed = memory.merge({}, {"name": ("yugabharathi", 0.93)})
    assert m["name"] == "Yugabharathi" and changed == {"name": "Yugabharathi"}
    m, changed = memory.merge(m, {"name": ("class", 0.55), "department": ("cse", 0.9), "section": ("d", 0.8)})
    assert m["name"] == "Yugabharathi" and m["department"] == "CSE" and m["section"] == "D"
    assert "name" not in changed


def test_clear_correction_still_wins():
    m, _ = memory.merge({}, {"name": ("yuga", 0.85)})
    m, changed = memory.merge(m, {"name": ("bharathi", 0.8)})
    assert m["name"] == "Bharathi" and changed == {"name": "Bharathi"}


def test_legacy_memory_is_migrated_and_replaceable():
    # Saved by the old pattern rules: no confidences, old slot names, a wrong name.
    legacy = {"name": "Class", "studies": "cse", "from": "Chennai"}
    m, changed = memory.merge(legacy, {"name": ("yugabharathi", 0.6)})
    assert m["name"] == "Yugabharathi" and m["department"] == "cse" and m["hometown"] == "Chennai"
    assert "studies" not in m and "from" not in m


def test_interests_accumulate_capped():
    m = {}
    for i in range(10):
        m, _ = memory.merge(m, {"interests": [("topic%d" % i, 0.7)]})
    m, changed = memory.merge(m, {"interests": [("TOPIC9", 0.9)]})
    assert len(m["interests"]) == memory.MAX_INTERESTS and changed == {}


def test_values_are_tidied_not_invented():
    m, _ = memory.merge({}, {"year": ("3rd year", 0.9), "department": ("ai and ds", 0.8),
                             "section": ("d", 0.8), "hometown": ("madurai", 0.9)})
    assert m["year"] == "3rd" and m["department"] == "ai and ds" and m["section"] == "D"
    assert m["hometown"] == "Madurai"
    assert memory.render(m) == ("They are in 3rd year. They study ai and ds, section D. "
                                "They are from Madurai.")


def test_bookkeeping_never_reaches_the_npc():
    m, _ = memory.merge({}, {"name": ("meena", 0.9)})
    assert "_confidence" in m
    assert "confidence" not in memory.render(m) and "confidence" not in memory.about(m)


def test_render_is_gender_neutral():
    m, _ = memory.merge({}, {"name": ("priya", 0.9), "year": ("first", 0.9), "project": ("robotics", 0.9)})
    text = " " + memory.render(m) + " "
    for word in (" he ", " she ", " his ", " her "):
        assert word not in text.lower()


def test_extractor_labels_match_training():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "make_data", root / "training" / "extractor" / "make_extraction_data.py")
    make_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_data)
    from dialogue import extractor
    assert make_data.LABELS == extractor.LABELS
    assert set(extractor.LABELS) - extractor.EVENT_SLOTS == set(memory.SLOTS)


def test_the_reported_lines_with_the_real_model():
    """Runs only when the fine-tuned weights are on disk (they are git-ignored)."""
    if not _extractor.FINE_TUNED.exists():
        print("        (skipped: no fine-tuned extractor weights)")
        return
    m, _ = memory.merge({}, REAL_EXTRACTOR.extract("my name is yugabharathi"))
    m, _ = memory.merge(m, REAL_EXTRACTOR.extract("i am class cse d"))
    assert m.get("name") == "Yugabharathi", m


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

def test_honorifics_do_not_end_sentences():
    assert guard.clean("Hi! Ms. Okafor said a man had a knife. Prof. Adeyemi saw it too. Third.") == \
        "Hi! Ms. Okafor said a man had a knife."
    from dialogue.text import split_sentences
    assert split_sentences("Ask Dr. Rao. Or Prof. Adeyemi.") == ["Ask Dr. Rao.", "Or Prof. Adeyemi."]


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
    out = run_turn(_input("what is my name", mem={"interests": ["ai"]}), gen, extract=no_facts)
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
    out = run_turn(_input("im in 2nd year ece", facts=[salary]), Scripted("Good to know."), extract=no_facts)
    assert out.facts_used == []
    assert knowledge.select(normalize("im in 2nd year ece"), [salary]) == []   # the stray "m"


def test_telling_is_not_asking():
    # "my project" in a statement is not a recall question.
    assert intent.classify("i want to do my final year project on chatbots").kind == intent.STATEMENT
    assert intent.classify(normalize("what was my project about")).kind == intent.RECALL
    assert intent.classify("remind me who i am").kind == intent.RECALL
    assert intent.classify("remember to lock your bike").kind == intent.STATEMENT


def test_recall_that_stays_empty_falls_back_to_memory():
    gen = Scripted("That's not something I can tell you.", ".")
    out = run_turn(_input("what was my project about", mem={"project": "chatbots"}), gen)
    assert out.response == "You're doing a project on chatbots."
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


def test_short_saved_values_need_whole_words():
    # Section "D" was "recalled" by "I don't keep track of that".
    mem = {"name": "Yugabharathi", "section": "D"}
    gen = Scripted("I don't keep track of that.", " you're in section D.")
    out = run_turn(_input("which section am i in", mem=mem), gen, extract=no_facts)
    assert "recall_miss" in out.problems and out.response.endswith("section D.")
    assert guard._mentions_memory("You're in D section.", mem, ["section"])
    assert not guard._mentions_memory("I'd say ask the department.", mem, ["section"])


def test_known_question_after_new_facts_becomes_an_acknowledgement():
    mem = {"name": "Yugabharathi"}
    gen = Scripted("What's your name?", " CSE D, that's a good section.")
    out = run_turn(_input("i am class cse d", mem=mem), gen,
                   extract=lambda m, c="": {"department": ("cse", 0.9), "section": ("d", 0.8)})
    assert gen.calls[1]["prefill"] == "Got it." and "name?" not in out.response


def test_correct_recall_is_not_regenerated_as_a_repeat():
    history = [{"role": "user", "content": "i am class cse d"},
               {"role": "assistant", "content": "That's a great name, Yugabharathi."}]
    gen = Scripted("That's a great name, Yugabharathi.", "That's not something I'd know.")
    out = run_turn(_input("what is my name", history=history, mem={"name": "Yugabharathi"}), gen,
                   extract=no_facts)
    assert len(gen.calls) == 1 and "Yugabharathi" in out.response


def test_a_retry_that_is_worse_is_rejected():
    history = [{"role": "user", "content": "where is the canteen"},
               {"role": "assistant", "content": "I teach algorithms and compilers here."}]
    gen = Scripted("I teach algorithms and compilers here.", "I don't do anything.")
    out = run_turn(_input("what do you do", history=history), gen, extract=no_facts)
    assert out.response == "I teach algorithms and compilers here." and out.repairs == ["repeat_rejected"]


def test_wrong_name_is_corrected_without_regenerating():
    gen = Scripted("Bye, Emily.")
    out = run_turn(_input("ok bye", mem={"name": "Yuga"}), gen)
    assert out.response == "Bye, Yuga." and len(gen.calls) == 1


def test_question_already_answered_is_dropped():
    gen = Scripted("Nice to meet you. What's your name?")
    out = run_turn(_input("i'm in 3rd year", mem={"name": "Yuga"}), gen,
                   extract=lambda m, c="": {"year": ("3rd", 0.9)})
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
    out = run_turn(_input("myself yuga"), gen, extract=lambda m, c="": {"name": ("yuga", 0.9)})
    assert gen.calls[1]["prefill"] == "Nice to meet you, Yuga." and out.repairs[0] == "introduced"


def test_eval_path_is_untouched():
    gen = Scripted("Sure, here you go: I patrol the city.")
    out = run_turn(TurnInput(archetype="police officer", message="what do you do"), gen)
    assert gen.calls[0]["messages"] == [
        {"role": "system", "content": "You are a police officer NPC in a modern city. Respond in a "
         "natural, contemporary voice consistent with your role. Never break character."},
        {"role": "user", "content": "what do you do"}]
    assert out.response == "I patrol the city." and out.repairs == []


# --- news ----------------------------------------------------------------------

KNIFE = {"id": "e1", "what": "a man carrying a knife", "where": "the gym", "heard_from": "Ms. Okafor",
         "fresh": True}


def test_event_from_extraction_and_from_report_intent():
    reading = {"incident": ("a man carrying a knife", 0.9), "place": ("the gym", 0.8)}
    assert events.event_from(reading, "i saw a man carrying a knife near the gym", False) ==         {"what": "a man carrying a knife", "where": "the gym", "score": 0.9,
         "raw": False, "kind": "incident"}
    # A report the extractor could not parse keeps the player's own words.
    assert events.event_from({}, "something weird near block c", True)["what"] == "something weird near block c"
    assert events.event_from({}, "i like cricket", False) is None
    assert events.event_from({"incident": ("x", 0.1)}, "x", False) is None   # too unsure


def test_mentions_needs_the_event_not_one_word():
    assert events.mentions("Someone saw a man with a knife near the gym!", KNIFE)
    assert not events.mentions("I use a knife to cut fruit.", KNIFE)
    assert events.leaks("There was a knife at the gym today.", [KNIFE]) == ["e1"]
    assert events.leaks("The canteen is on the first floor.", [KNIFE]) == []


def test_reported_event_is_returned_and_the_report_demo_is_shown():
    gen = Scripted("Thank you, I'll tell security.")
    reading = ({}, {"incident": ("a man carrying a knife", 0.9), "place": ("the gym", 0.7)})
    intent.default._ready = False
    out = run_turn(_input("i saw a man carrying a knife near the gym"), gen, extract=lambda m, c="": reading)
    assert out.reported_event["what"] == "a man carrying a knife" and out.reported_event["where"] == "the gym"
    assert out.player_memory == {}            # never a fact about the player


def test_greeting_passes_on_fresh_news_and_reports_it_shared():
    inp = _input("hi")
    inp.known_events = [KNIFE]
    gen = Scripted("Hi! Ms. Okafor said a man was carrying a knife at the gym.")
    out = run_turn(inp, gen, extract=no_facts)
    assert "Have you heard?" in gen.calls[0]["messages"][-2]["content"]
    assert out.shared_event_id == "e1" and out.event_leaks == []


def test_unheard_news_in_a_reply_is_a_leak():
    inp = _input("anything happening today?")
    inp.unheard_events = [KNIFE]
    gen = Scripted("Yes, someone had a knife at the gym.")
    out = run_turn(inp, gen, extract=no_facts)
    assert out.event_leaks == ["e1"]
    # Unheard news is only for catching leaks: it must never be in the prompt.
    assert "knife" not in " ".join(m["content"] for m in gen.calls[0]["messages"])


def test_heard_news_is_retrieved_for_news_questions():
    inp = _input("is anything happening on campus?")
    inp.known_events = [dict(KNIFE, fresh=False)]
    gen = Scripted("Ms. Okafor said a man with a knife was at the gym.")
    run_turn(inp, gen, extract=no_facts)
    prompt = " ".join(m["content"] for m in gen.calls[0]["messages"])
    assert "a man carrying a knife at the gym" in prompt


def test_ignored_news_on_greeting_is_restarted_then_said():
    inp = _input("hi")
    inp.known_events = [KNIFE]
    gen = Scripted("Hi, what's going on?", " something.")
    out = run_turn(inp, gen, extract=no_facts)
    assert gen.calls[1]["prefill"] == "Have you heard? Ms. Okafor told me a student reported"
    # The restart still left out what happened: the heard line is said instead.
    assert out.response == "Have you heard? Ms. Okafor told me a student reported a man carrying a knife at the gym."
    assert out.repairs == ["news", "news_fallback"] and out.shared_event_id == "e1"


def test_retrieved_news_missing_from_the_answer_is_restarted():
    inp = _input("is anything happening on campus?")
    inp.known_events = [dict(KNIFE, fresh=False)]
    gen = Scripted("There's a lecture on history today.", " a man carrying a knife near the gym.")
    out = run_turn(inp, gen, extract=no_facts)
    assert out.repairs == ["news"] and "knife" in out.response


def test_raw_report_is_quoted_not_rephrased():
    ev = events.event_from({}, "something weird near block c", True)
    assert events.sentence(dict(ev, heard_from="Halvorsen")) == 'Halvorsen told me a student reported: "something weird near block c".'
    assert events.lead_words(dict(ev, heard_from="Halvorsen")) == "Halvorsen told me a student reported"


def test_refused_report_is_restarted_as_thanks():
    gen = Scripted("That's not allowed.", " I'll let security know.")
    reading = ({}, {"incident": ("a kid bleeding", 0.9), "place": ("the court", 0.7)})
    intent.default._ready = False
    out = run_turn(_input("there's a kid bleeding near the court"), gen, extract=lambda m, c="": reading)
    if out.intent == "report":
        assert out.response.startswith("Thanks for telling me.") and out.repairs == ["report"]


def test_other_guests_are_retrievable_by_who_they_are():
    others = ("Officer Reyes (the campus liaison officer with the city police), "
              "Ms. Okafor (a student counsellor in the college welfare office)")
    guests = knowledge.guest_facts(others)
    assert guests[0].startswith("Officer Reyes, the campus liaison officer")
    pool = guests + ["The second floor has classrooms and the staff offices."]
    assert knowledge.select("can you tell me to the officer who present here", pool) == [guests[0]]
    assert knowledge.select("is there a counsellor around?", pool) == [guests[1]]
    # A place in a guest's job title is not a reason to retrieve them.
    assert knowledge.select("where would i find a lecturer's office?", pool) == [pool[-1]]


def test_anything_else_the_player_states_is_kept_as_a_note():
    gen = Scripted("Good to know.")
    intent.default._ready = False
    out = run_turn(_input("the lift in the physics block is stuck again"), gen, extract=no_facts)
    assert out.intent == "statement"
    assert out.reported_event["kind"] == "note"
    assert out.reported_event["what"] == "the lift in the physics block is stuck again"
    assert out.player_memory == {}          # a note is never a fact about the player


def test_small_talk_and_questions_are_not_notes():
    assert events.note_from("ok thanks") is None
    assert events.note_from("where is the canteen?") is None
    intent.default._ready = False
    gen = Scripted("Nice to meet you, Yuga.")
    out = run_turn(_input("my name is yuga"), gen,
                   extract=lambda m, c="": {"name": ("Yuga", 1.0)})
    assert out.reported_event is None       # the memory took it; not news


def test_a_heard_note_answers_a_question_but_is_not_blurted_on_greeting():
    note = {"id": "e9", "what": "the lift in the physics block is stuck", "where": "",
            "heard_from": "Nadia", "fresh": True, "kind": "note", "raw": True}
    hello = _input("hi")
    hello.known_events = [note]
    gen = Scripted("Hello there.")
    assert run_turn(hello, gen, extract=no_facts).shared_event_id == ""
    asked = _input("is anything happening on campus?")
    asked.known_events = [note]
    gen = Scripted("Nadia told me a student said the lift in the physics block is stuck.")
    out = run_turn(asked, gen, extract=no_facts)
    prompt = " ".join(m["content"] for m in gen.calls[0]["messages"])
    assert 'Nadia told me a student said: "the lift in the physics block is stuck"' in prompt
    assert out.shared_event_id == "e9"


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
