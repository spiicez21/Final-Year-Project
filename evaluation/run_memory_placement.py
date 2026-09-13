"""Where in the prompt should player memory go?

    .venv/Scripts/python.exe evaluation/run_memory_placement.py

The ablation behind the design of backend/dialogue/memory.py and the late
turn in backend/dialogue/composer.py. Same NPCs, introduction, fixed conversations
and dev probes as run_player_memory.py; no repairs. Only the memory
injection varies:

  none           no memory
  clause_only    memory stated in the system prompt (MEMORY_CLAUSE), no demo
  greet_early    clause + one greeting demo ("do you remember me" ->
                 "Of course, Priya. You were telling me about robotics.")
                 before the transcript -- the first design
  demo_early     clause + the current demonstration, before the transcript
  demo_late      clause + the current demonstration, immediately before the
                 player's message -- what the server does

Dev probes only: this is the set the design was chosen on, so these numbers
explain the choice; they are not the reported recall figures.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import gguf_server as gs  # noqa: E402
from dialogue import Persona, TurnConfig, TurnInput, composer, guard  # noqa: E402
from dialogue import intent as intents, memory as pm  # noqa: E402
from dialogue.text import normalize  # noqa: E402
from run_player_memory import EVENT, EVENT_LINE, INTRO, NPCS, PROBES, SMALL_TALK  # noqa: E402

OUT_PATH = REPO_ROOT / "evaluation" / "results" / "memory_placement_results.json"
MEMORY = pm.extract(INTRO)


def greet_demo(memory):
    detail = ""
    if memory.get("interests") or memory.get("project"):
        detail = " You were telling me about %s." % (memory.get("project")
                                                     or memory["interests"][-1])
    return [{"role": "user", "content": "do you remember me"},
            {"role": "assistant", "content": "Of course, %s.%s" % (memory["name"], detail)}]


# name -> (memory in system prompt, demo builder, placement)
VARIANTS = {
    "none":        (False, None, None),
    "clause_only": (True, None, None),
    "greet_early": (True, greet_demo, "early"),
    "demo_early":  (True, pm.demonstration, "early"),
    "demo_late":   (True, pm.demonstration, "late"),
}


# The composer without its own late memory turn, so each variant can place
# the demonstration itself. Repairs are not involved: this measures placement.
BASE = TurnConfig(memory_turn="never", repairs=False)


def messages_for(npc, message, history, variant):
    use_memory, demo, placement = VARIANTS[variant]
    arch, occupation, intro, job_line, background = NPCS[npc]
    inp = TurnInput(archetype=arch, message=message, max_tokens=64, history=list(history),
                    persona=Persona(name=npc, occupation=occupation, intro=intro, job_line=job_line,
                                    background=background, situation=EVENT, event_line=EVENT_LINE))
    memory = MEMORY if use_memory else {}
    text = normalize(message)
    msgs = composer.compose(inp, text, intents.classify(text), memory, [], BASE)
    if demo:
        turns = demo(MEMORY)
        replayed = len(composer.transcript(history, BASE.max_history_turns, BASE))
        at = len(msgs) - 1 if placement == "late" else len(msgs) - 1 - replayed
        msgs[at:at] = turns
    return msgs, arch


def generate(msgs, arch):
    # clean_basic, not the speech shaping, so a keyword in a third sentence
    # still counts -- this ablation predates the sentence cap.
    return guard.clean_basic(gs.LlamaGenerator(gs.pool.get(arch)[0])(msgs, 64))


def main():
    t0 = time.time()
    convos = {}
    for npc in NPCS:
        history = []
        for msg in [INTRO] + SMALL_TALK:
            reply = generate(*messages_for(npc, msg, history, "none"))
            history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        convos[npc] = history

    summary, rows = {}, []
    for variant in VARIANTS:
        hits = 0
        for npc in NPCS:
            for q, keys in PROBES["dev"]:
                reply = generate(*messages_for(npc, q, convos[npc], variant))
                hit = any(k in reply.lower() for k in keys)
                hits += hit
                rows.append({"variant": variant, "npc": npc, "probe": q, "hit": hit,
                             "reply": reply})
        n = len(PROBES["dev"]) * len(NPCS)
        summary[variant] = {"recall": hits, "of": n}
        print("%-12s recall %2d/%d  (%.0fs)" % (variant, hits, n, time.time() - t0), flush=True)

    OUT_PATH.write_text(json.dumps({"memory": MEMORY, "summary": summary,
                                    "conversations": convos, "rows": rows},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % OUT_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
