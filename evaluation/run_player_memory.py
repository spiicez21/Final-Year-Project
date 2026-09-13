"""Does an NPC remember what the player told it?

    .venv/Scripts/python.exe evaluation/run_player_memory.py

Runs in-process through the same dialogue pipeline /chat uses
(backend/dialogue.run_turn) and the same merged Q4_K_M models, so no server is
needed and nothing about the prompt is re-implemented here.

Design
  One conversation per NPC -- an introduction and two turns of small talk --
  is generated once without memory and reused for every condition, so the
  only difference between conditions is the memory itself. Each probe is then
  asked from that same state, so probes cannot prime one another.

  Conditions   none          the rolling transcript only (what the demo had before)
               memory        + player memory slots (backend/dialogue/memory.py)
               memory+repair + the reply guard's repairs (dialogue/guard.py)
  Scenarios    session    same visit: the introduction is still in the transcript
               returning  a later visit: transcript gone, persisted memory only
  Probe sets   dev    phrasings used while choosing how memory is injected and
                      designing the retry -- do not report
               test   written before any result was seen; used to pick nothing,
                      but its results were seen before the retry was built
               test2  written after the retry was designed, before it was run
               Reported numbers pool test + test2. Neither is blind: the probes
               and the retry's intent gate share an author. A set written by
               someone who has not read the gate is the proper next check.

  Also measured: intrusion (memory keywords in answers to unrelated
  questions), false recall ("you told me ..." about things never said),
  isolation (an NPC with no memory must not know the name, retry enabled),
  and KBD leaks on every reply.

Scoring is a keyword match ("priya", "robot", "first") -- lenient, since a
hit can still embellish ("a project called 'Robotics for Everyone'"). The
outputs file keeps every reply so hits can be read, not just counted.
"""
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import gguf_server as gs  # noqa: E402
from dialogue import Persona, TurnConfig, TurnInput, run_turn  # noqa: E402
from dialogue import memory as pm  # noqa: E402
from dialogue.intent import classify  # noqa: E402
from dialogue.text import normalize  # noqa: E402


def intent_of(message):
    return classify(normalize(message)).kind


OUT_PATH = REPO_ROOT / "evaluation" / "results" / "player_memory_results.json"

EVENT = "the Computer Science department open day"
EVENT_LINE = "It's the department open day — stalls down the corridor and a lot of students asking questions."

# Same personas as SampleGame/new-game-project/Systems/NPC/npc_director.gd.
# Campus facts are left out: they are orthogonal to player memory and would
# only lengthen the prompt identically in both conditions.
NPCS = {
    "Prof. Adeyemi": ("professor", "a lecturer in the Computer Science department",
        "I came along for the open day, meeting students and talking about the modules I teach.",
        "I teach CS2011 Algorithms and CS3040 Compilers, supervise six final-year projects, and spend whatever's left on my own research.",
        "I did my doctorate in this department and came back to teach after four years in industry. I care more about students finishing than about publication counts."),
    "Ms. Okafor": ("social worker", "a student counsellor in the college welfare office",
        "I'm at the open day so students know the counselling service exists and how to reach us.",
        "I see students one to one, mostly about stress, money worries and homesickness, and I point them towards the right support.",
        "I trained as a social worker in the city before moving into student welfare nine years ago. Most of the job is listening to people who assume nobody wants to hear it."),
    "Nadia": ("shopkeeper", "the person who runs the campus store",
        "I've got a stall at the open day, so I'm chatting to students between customers.",
        "I open at seven, keep the shelves stocked, and serve a few hundred students a day between lectures.",
        "I took the campus shop over from my aunt and I've run it ever since. I know what most of the regulars want before they reach the till."),
    "Halvorsen": ("executive", "the head of the Computer Science department",
        "I'm hosting the open day, so I'm here to meet students and answer questions about the department.",
        "I run the department, so timetables, hiring, budgets and a great many meetings. I still teach one module because I'd miss it otherwise.",
        "I've been head for three years and taught here for eleven before that. I took the job thinking I could fix the timetable, and I was wrong about that."),
    "Officer Reyes": ("police officer", "the campus liaison officer with the city police",
        "I was invited along to the open day to talk to students about staying safe on campus.",
        "Mostly I'm walking the campus and talking to people. The rest is lost property, bike thefts, and making sure students get home safely at night.",
        "I've been the campus liaison for six years, after eight on regular patrol in the north of the city. I know most of the porters by name and I still get lost in the science block."),
}

# "robotics" because none of these characters would raise it unprompted, so a
# hit is recall rather than the NPC talking about its own work.
INTRO = "Hi, I'm Priya, a first-year. I'm thinking of doing a final-year project on robotics."
SMALL_TALK = ["How long have you been here?", "Is it busy today?"]

# Neither set shares a phrasing with the demonstrations in backend/dialogue/memory.py
# ("do you remember me", "so what do you know about me").
PROBES = {
    "dev": [
        ("what's my name?", ["priya"]),
        ("do you remember what I'm interested in?", ["robot"]),
        ("what was my project idea again?", ["robot"]),
        ("remind me who I am", ["priya"]),
        ("do you know my name?", ["priya"]),
        ("what year am I in?", ["first"]),
        ("what did I say I wanted to build?", ["robot"]),
        ("any advice for my project?", ["robot"]),
    ],
    "test": [
        ("sorry, what did I say my name was?", ["priya"]),
        ("who am I again?", ["priya"]),
        ("what topic did I mention for my project?", ["robot"]),
        ("remember my project?", ["robot"]),
        ("which year did I say I'm in?", ["first"]),
        ("what am I into?", ["robot"]),
        ("can you recall my name?", ["priya"]),
        ("what should I focus on, given what I told you?", ["robot"]),
    ],
    # Written after the recall retry (now guard.py recall_miss) was designed, before it was run on
    # them. The same person wrote these and the retry's intent gate, so this
    # is not a blind set -- see the docstring.
    "test2": [
        ("what do you call me?", ["priya"]),
        ("have I told you my name?", ["priya"]),
        ("what's my final-year project going to be about?", ["robot"]),
        ("what subject did I pick for my project?", ["robot"]),
        ("am I a first-year or a second-year?", ["first"]),
        ("say my name", ["priya"]),
        ("what did I tell you I was working towards?", ["robot"]),
        ("who's this talking to you?", ["priya"]),
    ],
}
REPORTED = ("test", "test2")

# Unrelated questions. The last two contain "my", so they would trip a loose
# "is this about the player?" gate; memory must stay out of all four.
NEUTRAL = ["what do you do here?", "where can I get something to eat?",
           "where should I lock my bike?", "is my student card enough for the library?"]
MEMORY_KEYS = ("priya", "robot")
# "You told me you had one" on the student-card question: a claimed memory of
# something never said. Counted separately from keyword intrusion.
FALSE_RECALL = re.compile(r"\byou (?:told me|said|mentioned)\b", re.IGNORECASE)

# "memory+repair" is the full pipeline the game gets: memory plus the reply
# guard (dialogue/guard.py), which for recall questions means the "You told
# me" restart. "memory" is the same with the guard's repairs switched off.
CONDITIONS = ("none", "memory", "memory+repair")


class TimedGenerator:
    """gs.LlamaGenerator that records how long each generation takes."""

    def __init__(self, llm):
        self.inner = gs.LlamaGenerator(llm)
        self.calls = 0

    def __call__(self, messages, max_tokens, prefill="", repeat_penalty=1.0):
        t = time.perf_counter()
        out = self.inner(messages, max_tokens, prefill=prefill, repeat_penalty=repeat_penalty)
        TIMINGS["first" if self.calls == 0 else "retry"].append((time.perf_counter() - t) * 1000)
        self.calls += 1
        return out


def ask(npc, message, history, memory, repair):
    """One turn through the real pipeline (dialogue.run_turn), as /chat runs it."""
    arch, occupation, intro, job_line, background = NPCS[npc]
    persona = Persona(name=npc, occupation=occupation, intro=intro, job_line=job_line,
                      background=background, situation=EVENT, event_line=EVENT_LINE)
    llm, _ = gs.pool.get(arch)
    turn = run_turn(TurnInput(archetype=arch, message=message, max_tokens=64, persona=persona,
                              history=list(history), player_memory=dict(memory)),
                    TimedGenerator(llm), TurnConfig(repairs=repair))
    leaked = gs.compute_kbd(turn.response, arch, gs.KNOWLEDGE_ITEMS).get("violations", [])
    return turn.response, bool(leaked), turn.generations > 1


# Generation wall-clock, ms. Warm models; a repair is a second full
# generation, so a repaired turn costs first + retry.
TIMINGS = {"first": [], "retry": []}


def _pct(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * q))], 1) if values else None


def main():
    t0 = time.time()
    gs.load_scoring()  # otherwise KNOWLEDGE_ITEMS is empty and KBD finds nothing
    assert gs.KNOWLEDGE_ITEMS, "KBD knowledge base did not load"
    memory = pm.extract(INTRO)
    print("memory learned from the introduction: %s" % json.dumps(memory))

    convos = {}
    for npc in NPCS:
        history = []
        for msg in [INTRO] + SMALL_TALK:
            reply, _, _ = ask(npc, msg, history, {}, False)
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        convos[npc] = history

    summary, rows, leaks = {}, [], 0
    for scenario in ("session", "returning"):
        for split in PROBES:
            for cond in CONDITIONS:
                n = len(PROBES[split]) * len(NPCS)
                key = "%s/%s/%s" % (scenario, split, cond)
                if scenario == "returning" and cond == "none":
                    # Nothing to recall from: no transcript and no memory.
                    summary[key] = {"recall": 0, "of": n, "note": "trivially 0"}
                    continue
                mem = memory if cond != "none" else {}
                retry = cond == "memory+repair"
                hits = retries = intrusions = false_recall = 0
                per_npc = {}
                for npc in NPCS:
                    history = convos[npc] if scenario == "session" else []
                    per_npc[npc] = 0
                    for q, keys in PROBES[split]:
                        reply, leaked, retried = ask(npc, q, history, mem, retry)
                        hit = any(k in reply.lower() for k in keys)
                        hits += hit
                        retries += retried
                        per_npc[npc] += hit
                        leaks += leaked
                        rows.append({"scenario": scenario, "split": split, "condition": cond,
                                     "npc": npc, "probe": q, "hit": hit, "retried": retried,
                                     "reply": reply})
                    if split == "test2":  # neutral questions once per scenario/condition
                        for q in NEUTRAL:
                            reply, leaked, _ = ask(npc, q, history, mem, retry)
                            intr = any(k in reply.lower() for k in MEMORY_KEYS)
                            fr = bool(FALSE_RECALL.search(reply))
                            intrusions += intr
                            false_recall += fr
                            leaks += leaked
                            rows.append({"scenario": scenario, "split": "neutral",
                                         "condition": cond, "npc": npc, "probe": q,
                                         "intrusion": intr, "false_recall": fr, "reply": reply})
                summary[key] = {"recall": hits, "of": n, "retries": retries, "per_npc": per_npc}
                extra = ""
                if split == "test2":
                    summary[key].update(intrusions=intrusions, false_recall=false_recall,
                                        neutral_of=len(NEUTRAL) * len(NPCS))
                    extra = "  intr %d  false-recall %d /%d" % (
                        intrusions, false_recall, len(NEUTRAL) * len(NPCS))
                print("%-9s %-5s %-12s recall %2d/%d  retries %2d  %s%s" % (
                    scenario, split, cond, hits, n, retries,
                    " ".join("%s=%d" % (k.split()[-1], v) for k, v in per_npc.items()), extra),
                    flush=True)

    # Pooled over the two reported sets.
    for scenario in ("session", "returning"):
        for cond in CONDITIONS:
            parts = [summary["%s/%s/%s" % (scenario, s, cond)] for s in REPORTED]
            summary["%s/reported/%s" % (scenario, cond)] = {
                "recall": sum(p["recall"] for p in parts), "of": sum(p["of"] for p in parts)}

    # Isolation: each NPC keeps its own memory, so one that was never told
    # anything is asked with an empty memory and no transcript -- with the
    # retry enabled, to show it cannot fire without memory.
    iso = iso_fr = 0
    probes = PROBES["test"] + PROBES["test2"]
    for q, _ in probes:
        reply, _, _ = ask("Halvorsen", q, [], {}, True)
        iso += any(k in reply.lower() for k in MEMORY_KEYS)
        iso_fr += bool(FALSE_RECALL.search(reply))
    summary["isolation"] = {"knew_name_or_project": iso, "claimed_you_told_me": iso_fr,
                            "of": len(probes)}
    summary["kbd_leaking_replies"] = leaks
    # The first calls load each model; drop them so the medians are warm.
    first = TIMINGS["first"][len(NPCS) * (1 + len(SMALL_TALK)):]
    summary["latency_ms"] = {
        "first_p50": _pct(first, 0.5), "first_p90": _pct(first, 0.9), "first_n": len(first),
        "retry_p50": _pct(TIMINGS["retry"], 0.5), "retry_p90": _pct(TIMINGS["retry"], 0.9),
        "retry_n": len(TIMINGS["retry"])}
    print("latency: %s" % summary["latency_ms"])
    gate = sum(intent_of(q) == "recall" for q, _ in PROBES["test2"])
    summary["test2_gate_coverage"] = {"matched": gate, "of": len(PROBES["test2"])}
    print("isolation: knew %d/%d, claimed 'you told me' %d/%d   KB-leaking replies %d   "
          "gate covers %d/%d test2 probes   (%.0fs)"
          % (iso, len(probes), iso_fr, len(probes), leaks, gate, len(PROBES["test2"]),
             time.time() - t0))
    for scenario in ("session", "returning"):
        print("REPORTED %-9s " % scenario + "  ".join(
            "%s %d/%d" % (c, summary["%s/reported/%s" % (scenario, c)]["recall"],
                          summary["%s/reported/%s" % (scenario, c)]["of"]) for c in CONDITIONS))

    OUT_PATH.write_text(json.dumps({"memory": memory, "intro": INTRO, "summary": summary,
                                    "conversations": convos, "rows": rows},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % OUT_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
