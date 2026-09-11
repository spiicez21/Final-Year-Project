"""Does an NPC remember what the player told it?

    .venv/Scripts/python.exe evaluation/run_player_memory.py

Runs in-process against the same prompt builder the game server uses
(backend/gguf_server._build_messages) and the same merged Q4_K_M models, so
no server is needed and nothing about the prompt is re-implemented here.

Design
  One conversation per NPC -- an introduction and two turns of small talk --
  is generated once without memory and reused for every condition, so the
  only difference between conditions is the memory itself. Each probe is then
  asked from that same state, so probes cannot prime one another.

  Conditions   none    the rolling transcript only (what the demo had before)
               memory  transcript + player_memory slots (backend/player_memory.py)
  Scenarios    session    same visit: the introduction is still in the transcript
               returning  a later visit: transcript gone, persisted memory only
  Probe sets   dev   the phrasings used while choosing how memory is injected
               test  written before any result on them was seen; report these

  Also measured: intrusion (memory showing up in answers to unrelated
  questions), isolation (an NPC with no memory must not know the name), and
  KBD leaks on every reply.

Scoring is a keyword match ("priya", "robot", "first") -- lenient, since a
hit can still embellish ("a project called 'Robotics for Everyone'"). The
outputs file keeps every reply so hits can be read, not just counted.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import gguf_server as gs  # noqa: E402
import player_memory as pm  # noqa: E402

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

# Neither set shares a phrasing with the demonstrations in player_memory.py
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
}
NEUTRAL = ["what do you do here?", "where can I get something to eat?"]
MEMORY_KEYS = ("priya", "robot")


def ask(npc, message, history, memory):
    arch, occupation, intro, job_line, background = NPCS[npc]
    req = gs.ChatRequest(archetype=arch, message=message, max_tokens=64, name=npc,
                         occupation=occupation, intro=intro, job_line=job_line,
                         background=background, situation=EVENT, event_line=EVENT_LINE,
                         history=[gs.ChatTurn(**t) for t in history])
    # Same order as the /chat endpoint: this message's facts are merged first.
    merged = pm.merge(memory, pm.extract(message)) if memory else {}
    llm, _ = gs.pool.get(arch)
    out = llm.create_chat_completion(messages=gs._build_messages(req, merged), max_tokens=64,
                                     temperature=0.0, stop=gs.STOP_SEQUENCES)
    reply = gs._clean_reply(out["choices"][0]["message"]["content"])
    leaked = gs.compute_kbd(reply, arch, gs.KNOWLEDGE_ITEMS).get("violations", [])
    return reply, bool(leaked)


def main():
    t0 = time.time()
    memory = pm.extract(INTRO)
    print("memory learned from the introduction: %s" % json.dumps(memory))

    convos = {}
    for npc in NPCS:
        history = []
        for msg in [INTRO] + SMALL_TALK:
            reply, _ = ask(npc, msg, history, {})
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        convos[npc] = history

    summary, rows, leaks = {}, [], 0
    for scenario in ("session", "returning"):
        for split in ("dev", "test"):
            for cond in ("none", "memory"):
                mem = memory if cond == "memory" else {}
                hits = intrusions = 0
                per_npc = {}
                for npc in NPCS:
                    history = convos[npc] if scenario == "session" else []
                    per_npc[npc] = 0
                    for q, keys in PROBES[split]:
                        reply, leaked = ask(npc, q, history, mem)
                        hit = any(k in reply.lower() for k in keys)
                        hits += hit
                        per_npc[npc] += hit
                        leaks += leaked
                        rows.append({"scenario": scenario, "split": split, "condition": cond,
                                     "npc": npc, "probe": q, "hit": hit, "reply": reply})
                    if split == "test":  # neutral questions once per scenario/condition
                        for q in NEUTRAL:
                            reply, leaked = ask(npc, q, history, mem)
                            intr = any(k in reply.lower() for k in MEMORY_KEYS)
                            intrusions += intr
                            leaks += leaked
                            rows.append({"scenario": scenario, "split": "neutral",
                                         "condition": cond, "npc": npc, "probe": q,
                                         "intrusion": intr, "reply": reply})
                n = len(PROBES[split]) * len(NPCS)
                key = "%s/%s/%s" % (scenario, split, cond)
                summary[key] = {"recall": hits, "of": n, "per_npc": per_npc}
                if split == "test":
                    summary[key]["intrusions"] = intrusions
                    summary[key]["intrusions_of"] = len(NEUTRAL) * len(NPCS)
                print("%-9s %-4s %-6s recall %2d/%d   %s" % (
                    scenario, split, cond, hits, n,
                    " ".join("%s=%d" % (k.split()[-1], v) for k, v in per_npc.items())), flush=True)

    # Isolation: each NPC keeps its own memory, so one that was never told
    # anything is asked with an empty memory and no transcript.
    iso = 0
    for q, keys in PROBES["test"]:
        reply, _ = ask("Halvorsen", q, [], {})
        iso += any(k in reply.lower() for k in MEMORY_KEYS)
    summary["isolation_leaks"] = {"count": iso, "of": len(PROBES["test"])}
    summary["kbd_leaking_replies"] = leaks
    print("isolation leaks %d/%d   KB-leaking replies %d   (%.0fs)"
          % (iso, len(PROBES["test"]), leaks, time.time() - t0))

    OUT_PATH.write_text(json.dumps({"memory": memory, "intro": INTRO, "summary": summary,
                                    "conversations": convos, "rows": rows},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % OUT_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
