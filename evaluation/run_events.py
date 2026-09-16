"""Does reported news stay where it has been heard, and travel when it spreads?

    .venv/Scripts/python.exe evaluation/run_events.py

In-process, through the real dialogue pipeline (dialogue.run_turn) with the
fine-tuned extractor, the learned intent classifier and the persona models,
using the same game-shaped requests as run_grounding.py. The game's event log
is simulated exactly as world_event_store.gd keeps it.

For each report and each (told NPC, other NPC) pair:

  1 report     the player tells NPC A about an incident
               -> captured: the turn returns a reported event
  2 before     NPC B, who has NOT heard it, is asked about news
               -> leak: B's reply mentions the incident (B cannot know it)
  3 greeted    B hears it from A (one gossip step), then the player says hi
               -> shared: B passes the news on unprompted
  4 asked      the player asks B about news
               -> answered: B's reply mentions the incident

Report lines were written 2026-09-15, before this script was first run, and
use places and incidents not in the extractor's training phrases. "Mentions"
is events.mentions(): two content words of the incident or place in the
reply -- lenient; transcripts are saved so replies can be read.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "evaluation"))

import gguf_server as gs  # noqa: E402
import run_grounding as grounding  # noqa: E402
from dialogue import events, run_turn  # noqa: E402

OUT = REPO / "evaluation" / "results" / "events_results.json"

REPORTS = [
    "i saw someone climbing into the principal's office through the window",
    "there's a kid bleeding badly near the basketball court",
    "a guy with a gun is walking around the old library building",
]
PAIRS = [("Prof. Adeyemi", "Nadia"), ("Ms. Okafor", "Officer Reyes"), ("Halvorsen", "Prof. Adeyemi")]
NEWS_QUESTION = "is anything happening on campus today?"


def turn(npc, message, known=(), unheard=()):
    req = grounding.build_request(npc, message)
    inp = gs.to_turn_input(req)
    inp.known_events = list(known)
    inp.unheard_events = list(unheard)
    llm, _ = gs.pool.get(req.archetype)
    return run_turn(inp, gs.LlamaGenerator(llm), gs.TURN_CONFIG)


def main():
    gs.load_scoring()
    t0 = time.time()
    rows = []
    totals = dict(trials=0, captured=0, leak_before=0, shared_on_greeting=0, answered_after=0,
                  # how many of those needed the guard: an authored fallback line is not the model
                  shared_by_fallback=0, answered_by_fallback=0, told_reply_repaired=0,
                  shared_needed_restart=0, answered_needed_restart=0)
    for report in REPORTS:
        for told, other in PAIRS:
            totals["trials"] += 1
            r1 = turn(told, report)
            ev = r1.reported_event
            row = {"report": report, "told": told, "other": other, "told_reply": r1.response,
                   "intent": r1.intent, "event": ev}
            if not ev:
                rows.append(row)
                print("  not captured: %-60s intent=%s" % (report, r1.intent), flush=True)
                continue
            totals["captured"] += 1
            event = dict(ev, id="e1")

            r2 = turn(other, NEWS_QUESTION, unheard=[event])
            leaked = bool(r2.event_leaks)
            totals["leak_before"] += leaked

            heard = dict(event, heard_from=told, fresh=True)
            r3 = turn(other, "hi", known=[heard])
            shared = bool(r3.shared_event_id)
            totals["shared_on_greeting"] += shared

            r4 = turn(other, NEWS_QUESTION, known=[dict(heard, fresh=False)])
            answered = events.mentions(r4.response, event)
            totals["answered_after"] += answered

            totals["shared_by_fallback"] += shared and "news_fallback" in r3.repairs
            totals["answered_by_fallback"] += answered and "news_fallback" in r4.repairs
            totals["told_reply_repaired"] += bool(r1.repairs)
            totals["shared_needed_restart"] += shared and "news" in r3.repairs
            totals["answered_needed_restart"] += answered and "news" in r4.repairs
            row.update(told_repairs=r1.repairs, greeted_repairs=r3.repairs, asked_repairs=r4.repairs,
                       before=r2.response, leaked=leaked, greeted=r3.response, shared=shared,
                       asked=r4.response, answered=answered)
            rows.append(row)
            print("  %-13s -> %-13s captured=%r leak=%s shared=%s answered=%s" % (
                told, other, ev.get("what"), leaked, shared, answered), flush=True)
    n = totals["trials"]
    print("\ncaptured %d/%d | leaks before hearing %d/%d | shared on greeting %d/%d | "
          "answered after hearing %d/%d   (%.0fs)" % (
              totals["captured"], n, totals["leak_before"], totals["captured"],
              totals["shared_on_greeting"], totals["captured"], totals["answered_after"],
              totals["captured"], time.time() - t0))
    print("  needed the news restart: shared %d, answered %d | by authored fallback: shared %d, answered %d"
          " | told-NPC replies repaired %d" % (
        totals["shared_needed_restart"], totals["answered_needed_restart"], totals["shared_by_fallback"],
        totals["answered_by_fallback"], totals["told_reply_repaired"]))
    OUT.write_text(json.dumps({"summary": totals, "rows": rows}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote %s" % OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
