"""Does the NPC understand what the player says about themselves?

    .venv/Scripts/python.exe evaluation/run_memory_extraction.py

Scores the player-fact extractor (backend/dialogue/extractor.py) on hand-
labelled lines in evaluation/memory_extraction_cases.json, which were written
before any extractor was run on them. For each line: every expected slot must
be learned with a matching value, and no forbidden slot may be learned ("i am
class cse d" must not produce a name).

  base        GLiNER small, zero-shot (urchade/gliner_small-v2.1)
  fine-tuned  the same, trained on training/extractor synthetic data whose
              names, towns, colleges and departments exclude every case value

The confidence threshold is chosen on dev lines only, then test lines are
scored once at that threshold.

Scenarios check the whole path through memory.merge over several lines, which
is where the reported bug lived: "my name is yugabharathi" followed by
"i am class cse d" must still leave the name as Yugabharathi. They were written
together with the cases, before any run.
"""

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from dialogue import extractor as ex  # noqa: E402
from dialogue import memory  # noqa: E402

CASES = json.loads((REPO / "evaluation" / "memory_extraction_cases.json").read_text(encoding="utf-8"))["cases"]
OUT = REPO / "evaluation" / "results" / "memory_extraction_results.json"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

# (lines as (npc_context, player_line), expected memory after all of them, slots that must be absent)
SCENARIOS = [
    ([(None, "my name is yugabharathi"), (None, "i am class cse d")],
     {"name": "yugabharathi", "department": "cse", "section": "d"}, []),
    ([(None, "i am vijay"), (None, "i am from trichy"), (None, "i am hosteller")],
     {"name": "vijay", "hometown": "trichy"}, []),
    ([("What's your name?", "meena"), ("Nice to meet you, Meena.", "i am in ece b")],
     {"name": "meena", "department": "ece", "section": "b"}, []),
    ([(None, "call me yb"), (None, "actually my name is yogesh")],
     {"name": "yogesh"}, []),
    ([(None, "i am placement coordinator"), (None, "i am tamil"), (None, "where is the canteen")],
     {}, ["name", "hometown"]),
    # Added 2026-09-15 after "i have a dog named bruno" renamed a player in play.
    ([(None, "my name is kavitha"), (None, "i have a dog named bruno"), (None, "my brother arivu is in mech")],
     {"name": "kavitha"}, ["department"]),
]

ORD = {"first": "1", "second": "2", "third": "3", "fourth": "4", "1st": "1", "2nd": "2", "3rd": "3", "4th": "4"}


def _norm(s):
    return " ".join(ORD.get(x, x) for x in re.findall(r"[a-z0-9]+", str(s).lower()) if x not in ("year", "yr"))


def matches(got, want):
    """`got` is a string or a list of strings (interests); lenient on form."""
    if isinstance(got, list):
        return any(matches(g, want) for g in got)
    g, w = _norm(got), _norm(want)
    return bool(g) and bool(w) and (w in g or g in w)


def value_of(reading):
    """Extractor reading -> plain value(s): (value, score) or [(value, score), ...]."""
    if isinstance(reading, list):
        return [v for v, _ in reading]
    return reading[0]


def score_cases(extractor, split):
    rows, full, slot_ok, slot_total, forbidden = [], 0, 0, 0, 0
    t0 = time.time()
    cases = [c for c in CASES if c["split"] == split]
    for c in cases:
        facts, event = extractor.read(c["text"], c.get("context") or "")
        got = {**facts, **event}   # incident/place cases are scored alongside player facts
        ok = True
        for slot, want in c["expect"].items():
            slot_total += 1
            hit = slot in got and matches(value_of(got[slot]), want)
            slot_ok += hit
            ok &= hit
        for slot in c.get("forbid", []):
            if slot in got:
                forbidden += 1
                ok = False
        full += ok
        rows.append({"context": c.get("context"), "text": c["text"], "ok": ok,
                     "got": {k: value_of(v) for k, v in got.items()}})
    return {"cases_ok": full, "cases": len(cases), "slots_ok": slot_ok, "slots": slot_total,
            "forbidden_learned": forbidden, "ms_per_line": round((time.time() - t0) * 1000 / len(cases), 1),
            "rows": rows}


def run_scenarios(extractor):
    out = []
    for lines, expected, absent in SCENARIOS:
        m = {}
        for context, line in lines:
            m, _ = memory.merge(m, extractor.extract(line, context or ""))
        facts = memory.facts(m)
        ok = (all(k in facts and matches(facts[k], v) for k, v in expected.items())
              and not any(k in facts for k in absent))
        out.append({"lines": [l for _, l in lines], "ok": ok, "memory": facts})
    return out


def main():
    results = {}
    for name, path in (("base", ex.BASE_MODEL), ("fine-tuned", ex.FINE_TUNED)):
        if isinstance(path, Path) and not path.exists():
            print("%-10s skipped: %s not found (run training/extractor/train_extractor.py)" % (name, path))
            continue
        model = ex.FactExtractor(path=path)
        if not model.available:
            print("%-10s skipped: could not load %s" % (name, path))
            continue
        dev = {}
        for th in THRESHOLDS:
            model.min_score = th
            dev[th] = score_cases(model, "dev")
        best = max(THRESHOLDS, key=lambda t: (dev[t]["cases_ok"] - dev[t]["forbidden_learned"],
                                               dev[t]["slots_ok"], -t))
        model.min_score = best
        test = score_cases(model, "test")
        scenarios = run_scenarios(model)
        results[name] = {"threshold_chosen_on_dev": best,
                         "dev": {str(t): {k: v for k, v in r.items() if k != "rows"} for t, r in dev.items()},
                         "test": test, "scenarios": scenarios}
        d = dev[best]
        print("%-10s threshold %.1f | dev cases %d/%d slots %d/%d forbidden %d | "
              "TEST cases %d/%d slots %d/%d forbidden %d | scenarios %d/%d | %.0f ms/line"
              % (name, best, d["cases_ok"], d["cases"], d["slots_ok"], d["slots"], d["forbidden_learned"],
                 test["cases_ok"], test["cases"], test["slots_ok"], test["slots"], test["forbidden_learned"],
                 sum(s["ok"] for s in scenarios), len(scenarios), test["ms_per_line"]), flush=True)
        if "-v" in sys.argv:
            for r in test["rows"]:
                if not r["ok"]:
                    print("     miss ctx=%-24r %-40r -> %s" % (r["context"], r["text"], r["got"]))
            for s in scenarios:
                print("     scenario %s %s -> %s" % ("ok  " if s["ok"] else "FAIL", s["lines"], s["memory"]))
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
