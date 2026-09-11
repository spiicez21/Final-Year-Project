"""Unit tests for player_memory. Run: .venv/Scripts/python.exe -m pytest backend/test_player_memory.py -q
or directly: .venv/Scripts/python.exe backend/test_player_memory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from player_memory import demonstration, extract, merge, render

# (message, expected subset of extract() output)
EXTRACT_CASES = [
    ("Hi, I'm Priya.", {"name": "Priya"}),
    ("my name is Arjun Kumar", {"name": "Arjun Kumar"}),
    ("Call me Sam", {"name": "Sam"}),
    ("I'm a first-year student", {"year": "first-year"}),
    ("I am in my 2nd year", {"year": "second-year"}),
    ("I'm studying computer science", {"studies": "computer science"}),
    ("I'm interested in compilers", {"interests": ["compilers"]}),
    ("I'm thinking of doing a final-year project on machine learning.",
     {"project": "machine learning"}),
    ("I really like robotics and I love game design",
     {"interests": ["robotics", "game design"]}),
    ("I'm really stressed about exams", {"feeling": "stressed"}),
    ("I'm from Chennai", {"from": "Chennai"}),
    # How people actually type in a game chat box
    ("hey im Priya", {"name": "Priya"}),
    ("my name is priya", {"name": "Priya"}),
    ("my name is priya and i love compilers", {"name": "Priya", "interests": ["compilers"]}),
    ("I’m Arjun", {"name": "Arjun"}),  # curly apostrophe
    ("I'm Arjun, a first-year, interested in networks",
     {"name": "Arjun", "year": "first-year", "interests": ["networks"]}),
    ("I study AI", {"studies": "AI"}),
    ("I'm into cybersecurity", {"interests": ["cybersecurity"]}),
]

# Things that must NOT be learned. Each is a realistic line that a naive
# pattern would misread.
FALSE_POSITIVE_CASES = [
    ("I'm stressed", "name"),            # lowercase after I'm: a feeling, not a name
    ("I'm Sorry to bother you", "name"),  # capitalised, but not a name
    ("I'm Here for the open day", "name"),
    ("I'm Interested in AI", "name"),
    ("What do you teach?", None),         # a question about the NPC teaches nothing about the player
    ("Do you have some weeds?", None),
    ("I'm doing well thanks", "studies"),
    ("are you interested in AI?", "interests"),        # about the NPC, not the player
    ("what do first-year students study?", "year"),
    ("I'm not stressed at all", "feeling"),
]


def test_extract():
    for msg, expected in EXTRACT_CASES:
        got = extract(msg)
        for k, v in expected.items():
            assert got.get(k) == v, "%r: %s expected %r got %r" % (msg, k, v, got.get(k))


def test_no_false_positives():
    for msg, slot in FALSE_POSITIVE_CASES:
        got = extract(msg)
        if slot is None:
            assert got == {}, "%r should teach nothing, got %r" % (msg, got)
        else:
            assert slot not in got, "%r should not set %s, got %r" % (msg, slot, got)


def test_merge_replaces_and_accumulates():
    m = merge({}, {"name": "Priya", "interests": ["compilers"]})
    m = merge(m, {"name": "Priya S", "interests": ["robotics", "compilers"]})
    assert m["name"] == "Priya S"                      # newest statement wins
    assert m["interests"] == ["compilers", "robotics"]  # no duplicates
    for i in range(10):
        m = merge(m, {"interests": ["topic%d" % i]})
    assert len(m["interests"]) == 4                     # capped


def test_render_is_gender_neutral():
    text = render({"name": "Priya", "year": "first-year", "interests": ["compilers"]})
    assert "Priya" in text and "compilers" in text
    for word in (" he ", " she ", " his ", " her ", "He ", "She "):
        assert word not in " " + text + " ", "render assumed a gender: %r" % text


def test_project_slot():
    got = extract("I'm thinking of doing a final-year project on robotics.")
    assert got.get("project") == "robotics", got
    assert "interests" not in got, got  # a project is not also filed as an interest
    text = render({"name": "Priya", "project": "robotics", "interests": ["robotics", "AI"]})
    assert "project on robotics" in text and "interested in AI" in text
    assert text.count("robotics") == 1  # not repeated as a plain interest


def test_demonstration():
    assert demonstration({}) == []
    demo = demonstration({"name": "Priya", "year": "first-year", "project": "robotics"})
    assert demo[1]["content"] == "Of course, Priya."
    assert demo[3]["content"] == ("You're Priya, a first-year, and you want to do your "
                                  "final-year project on robotics.")
    # No name: still demonstrates, without inventing one.
    demo = demonstration({"interests": ["compilers"]})
    assert len(demo) == 2 and "compilers" in demo[1]["content"]


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("  PASS  %s" % name)
            except AssertionError as exc:
                failures += 1
                print("  FAIL  %s: %s" % (name, exc))
    print("\n%s" % ("ALL PASSED" if not failures else "%d FAILED" % failures))
    raise SystemExit(1 if failures else 0)
