"""Whole conversations, typed the way players type, over two visits.

    .venv/Scripts/python.exe evaluation/run_conversations.py [dev|test]

The recall and grounding evaluations ask one well-formed question from a fixed
state. Players do not: they type "myself Yuga, 3rd year cse", "what do u do",
"ok", and come back the next day. This drives each NPC through two visits
exactly as npc_director.gd does -- transcript capped at 16 entries, the
server's merged memory saved whenever it reports an update, transcript
cleared between visits while memory is kept -- and checks every reply for the
failures seen in play:

  repeat        reply is a near-copy of an earlier reply in the same visit,
                although the player said something different
  loop          three or more consecutive near-identical replies
  asks_known    NPC asks for something it has already been told (name, year,
                project), or asks the same question twice in a visit
  invented      NPC states a name for the player that is not the saved one
  self_denial   NPC denies having a job or knowing anything about its work
  list          reply is a numbered/bulleted list (unspeakable, and truncated)
  memory        per-slot: did the facts the player stated get saved, and did
                anything get saved that the player never said

Two scripted players, written together before any fix was attempted:
  dev   used while building the fix
  test  reported
"""
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import gguf_server as gs  # noqa: E402
import run_grounding as grounding  # noqa: E402  (game-shaped requests)

OUT_DIR = REPO_ROOT / "evaluation" / "results"
MAX_HISTORY = 16  # npc_actor.gd

PLAYERS = {
    "dev": {
        # What this player says about themselves, and the slots it should produce.
        "expected_memory": {"name": "Yuga", "year": "3", "department": "cse", "interests": ["ai"]},
        "name": "yuga",
        "visit1": ["hi", "myself Yuga, 3rd year cse", "what do u teach", "where is canteen",
                   "ok thanks", "what should i do for my final year project", "i like ai",
                   "can u help me", "what is ur salary", "ok bye"],
        "visit2": ["hi again", "do u remember me", "what is my name", "where is canteen",
                   "what do u do", "tell me about the department", "what do u do", "ok", "ok",
                   "bye"],
    },
    "test": {
        "expected_memory": {"name": "Arjun", "year": "2", "department": "ece", "hometown": "Madurai",
                            "interests": ["robotics"]},
        "name": "arjun",
        "visit1": ["hello", "i am arjun from madurai", "im in 2nd year ece", "whats ur job",
                   "wat time does the open day finish", "cool", "i really like robotics",
                   "how much do u earn", "k", "thanks bye"],
        "visit2": ["hey", "u remember my name?", "which year am i in", "where r the labs",
                   "what's ur job again", "hmm", "hmm", "anything else i should know",
                   "what do u do", "bye"],
    },
    # Written after the test player's transcripts had been read and used to
    # fix three failures (facts retrieved for statements, "open" treated as a
    # weak word, evasive recall), before any of those fixes was run.
    "test2": {
        "expected_memory": {"name": "Meena", "year": "1", "department": "it", "hometown": "Coimbatore",
                            "project": "chatbots"},
        "name": "meena",
        "visit1": ["good morning", "this is meena, 1st year it from coimbatore",
                   "what r u doing here today", "is the canteen open now", "nice",
                   "i want to do my final year project on chatbots", "who should i talk to about that",
                   "how many students r in the department", "ok ok", "see you"],
        "visit2": ["hii", "do u know who i am", "what was my project about", "which floor has the labs",
                   "tell me about ur work", "alright", "alright", "what's ur salary",
                   "what do u do", "thanks, bye"],
    },
    # The reported bug, replayed: the name was saved, then "i am class cse d"
    # renamed the player "Class". A reproduction, not a held-out test -- its
    # first lines are also in evaluation/memory_extraction_cases.json.
    "reported": {
        "expected_memory": {"name": "Yugabharathi", "department": "cse", "section": "d"},
        "name": "yugabharathi",
        "visit1": ["hi", "my name is yugabharathi", "i am class cse d", "what do u teach",
                   "which section am i in", "ok bye"],
        "visit2": ["hello again", "what is my name", "which class am i in", "ok", "bye"],
    },
}

WORDS = re.compile(r"[a-z0-9']+")
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "1st": "1", "2nd": "2",
             "3rd": "3", "4th": "4"}


def memory_facts(memory):
    """Saved slots without bookkeeping (the server's per-slot confidences)."""
    return {k: v for k, v in (memory or {}).items() if not k.startswith("_")}


def same_value(got, want):
    """Lenient: case, ordinal form ("3rd" = "third" = "3") and "year" suffixes ignored."""
    if isinstance(want, list):
        got = got if isinstance(got, list) else [got]
        return all(any(same_value(g, w) for g in got) for w in want)
    if isinstance(got, list):
        return any(same_value(g, want) for g in got)
    norm = lambda x: " ".join(_ORDINALS.get(t, t) for t in re.findall(r"[a-z0-9]+", str(x).lower())
                             if t not in ("year", "yr"))
    g, w = norm(got), norm(want)
    return bool(g) and (g == w or w in g or g in w)


def words(s):
    return WORDS.findall(s.lower())


def similar(a, b):
    A, B = set(words(a)), set(words(b))
    if len(A) < 3 or len(B) < 3:
        return a.strip().lower() == b.strip().lower()
    return len(A & B) / min(len(A), len(B)) >= 0.75


SELF_DENIAL = re.compile(
    r"\bi (don't|do not) (do anything|have a job|work here|know anything about (that|this|it))\b"
    r"|\bi'm (just|only) a (machine|computer|program)\b|\bi'm not really into anything\b",
    re.IGNORECASE)
LIST = re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s+\S")
ASK_NAME = re.compile(r"\b(what's|what is) your name\b", re.IGNORECASE)
ASK_YEAR = re.compile(r"\bwhat year are you\b|\bwhich year are you\b", re.IGNORECASE)
STATED_NAME = re.compile(r"\b(?:your name(?: is|'s| was)|you're|you are|call you)\s+([A-Z][a-z]+)")


def questions_in(reply):
    return [s.strip() for s in re.findall(r"[^.!?]*\?", reply) if s.strip()]


def run_player(npc, player, turn_fn):
    memory, report = {}, {"turns": []}
    counts = dict(repeat=0, loop=0, asks_known=0, invented=0, self_denial=0, list=0, replies=0)
    for visit in ("visit1", "visit2"):
        history, replies, asked = [], [], []
        run = 0
        for msg in player[visit]:
            out = turn_fn(npc, msg, history, memory)
            reply = out["response"]
            flags = []
            # What the NPC knows once this message is heard: asking for a name
            # the player gave in this very message counts as asking a known thing.
            if out.get("memory_updates"):
                memory = out["player_memory"]
            prev_same = [i for i, (m, r) in enumerate(replies) if similar(reply, r) and m != msg]
            if prev_same:
                flags.append("repeat")
            run = run + 1 if replies and similar(reply, replies[-1][1]) else 0
            if run >= 2:
                flags.append("loop")
            for q in questions_in(reply):
                if (ASK_NAME.search(q) and memory.get("name")) or \
                   (ASK_YEAR.search(q) and memory.get("year")) or \
                   any(similar(q, a) for a in asked):
                    flags.append("asks_known")
                    break
            asked += questions_in(reply)
            # A name the NPC gives the player that the player never gave. The
            # player's real name is not invention even if memory missed it.
            m = STATED_NAME.search(reply)
            if m and m.group(1).lower() not in (player["name"], *npc.lower().replace(".", "").split()):
                flags.append("invented")
            if SELF_DENIAL.search(reply):
                flags.append("self_denial")
            if LIST.search(reply):
                flags.append("list")
            for f in set(flags):
                counts[f] += 1
            counts["replies"] += 1
            replies.append((msg, reply))
            history = (history + [{"role": "user", "content": msg},
                                  {"role": "assistant", "content": reply}])[-MAX_HISTORY:]
            report["turns"].append({"visit": visit, "you": msg, "npc": reply, "flags": flags,
                                    "memory": memory, "repairs": out.get("repairs", [])})
    expected = player["expected_memory"]
    got = memory_facts(memory)
    slots_ok = sum(1 for k, v in expected.items() if k in got and same_value(got[k], v))
    # A slot the player never stated at all (not merely a different wording).
    spurious = [k for k in got if k not in expected]
    report.update(counts=counts, memory=got, slots_ok=slots_ok, slots_of=len(expected),
                  spurious_slots=spurious)
    return report


def server_turn(npc, msg, history, memory):
    req = grounding.build_request(npc, msg)
    req.history = [gs.ChatTurn(**t) for t in history]
    req.player_memory = memory
    out = gs.chat(req)
    return out.model_dump()


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    gs.load_scoring()
    assert gs.KNOWLEDGE_ITEMS, "KBD knowledge base did not load"
    player = PLAYERS[split]
    t0 = time.time()
    total = dict(repeat=0, loop=0, asks_known=0, invented=0, self_denial=0, list=0, replies=0)
    slots_ok = slots_of = 0
    spurious = 0
    reports = {}
    for npc in grounding.NPCS:
        rep = run_player(npc, player, server_turn)
        reports[npc] = rep
        for k in total:
            total[k] += rep["counts"][k]
        slots_ok += rep["slots_ok"]
        slots_of += rep["slots_of"]
        spurious += len(rep["spurious_slots"])
        c = rep["counts"]
        print("%-14s repeat %d loop %d asks_known %d invented %d self_denial %d list %d | memory %d/%d spurious %s"
              % (npc, c["repeat"], c["loop"], c["asks_known"], c["invented"], c["self_denial"],
                 c["list"], rep["slots_ok"], rep["slots_of"], rep["spurious_slots"]), flush=True)
        if "-v" in sys.argv:
            for t in rep["turns"]:
                print("   %s you: %-38s npc: %s %s" % (t["visit"][-1], t["you"], t["npc"][:100].replace("\n", " "),
                                                     " ".join(t["flags"])))
    print("TOTAL %s | memory slots %d/%d, spurious %d  (%.0fs)"
          % (json.dumps(total), slots_ok, slots_of, spurious, time.time() - t0))
    summary = dict(total, memory_slots_ok=slots_ok, memory_slots_of=slots_of, spurious_slots=spurious)
    (OUT_DIR / ("conversations_%s.json" % split)).write_text(
        json.dumps({"summary": summary, "by_npc": reports}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("wrote evaluation/results/conversations_%s.json" % split)


if __name__ == "__main__":
    main()
