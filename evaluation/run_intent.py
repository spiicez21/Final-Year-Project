"""How well is the player's intent understood?

    .venv/Scripts/python.exe evaluation/run_intent.py

Scores intent classification on evaluation/intent_cases.json (labelled before
any classifier was run on them), for:

  rules    the pattern classifier (backend/dialogue/intent.py, rules_classify)
  learned  heads over the fact extractor's encoder (training/extractor/train_intent.py)

Test lines are reported; dev lines were available while building both.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from dialogue import intent  # noqa: E402
from dialogue.text import normalize  # noqa: E402

CASES = json.loads((REPO / "evaluation" / "intent_cases.json").read_text(encoding="utf-8"))["cases"]
OUT = REPO / "evaluation" / "results" / "intent_results.json"


def score(name, classify):
    result = {}
    for split in ("dev", "test"):
        cases = [c for c in CASES if c["split"] == split]
        t0 = time.time()
        rows = []
        for c in cases:
            got = classify(c["text"]).kind
            rows.append({"text": c["text"], "want": c["label"], "got": got})
        ok = sum(r["want"] == r["got"] for r in rows)
        result[split] = {"ok": ok, "of": len(cases), "ms_per_line": round((time.time() - t0) * 1000 / len(cases), 1),
                         "misses": [r for r in rows if r["want"] != r["got"]]}
    print("%-8s dev %d/%d   TEST %d/%d   %.0f ms/line" % (
        name, result["dev"]["ok"], result["dev"]["of"], result["test"]["ok"], result["test"]["of"],
        result["test"]["ms_per_line"]))
    for m in result["test"]["misses"]:
        print("     %-42r want %-10s got %s" % (m["text"], m["want"], m["got"]))
    return result


def main():
    results = {"rules": score("rules", lambda t: intent.rules_classify(normalize(t)))}
    learned = intent.default
    if learned.available:
        results["learned"] = score("learned", learned.classify)
    else:
        print("learned  skipped: no trained intent head (run training/extractor/train_intent.py)")
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
