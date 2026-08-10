"""
Week 6 (Docs/TODO.md) exit condition: KBD must catch at least one real
out-of-visibility violation. Runs the trained police-officer adapter
against data/processed/kbd_probes.json (14 "forbidden" probes targeting
knowledge visible to OTHER archetypes, 3 "control" probes targeting its own
visible knowledge) via the real GGUF serving path, scores each response
with kbd_scorer.py.

First run (2026-08-10) caught only 1/14 forbidden probes via
kbd_scorer.py's generic signature-phrase matching, but manual read of the
"no reference detected" responses found several that clearly confirm the
forbidden fact without restating any of its content words — "Yes, it did.
They're still waiting for the shipment to arrive." confirms KB-004 (the
insulin shipment) without ever saying "insulin" or "shipment delayed." No
signature-phrase list can catch a bare "Yes, it did" — there's nothing
fact-specific in the response text itself to match against; "it" only
resolves to the fact via the probe question, which kbd_scorer.py's
general-purpose scorer never sees. Since every probe here already links to
a known target fact, added `classify_probe_response()` below: a
probe-context-aware affirm/deny check specific to calibration, on top of
(not instead of) kbd_scorer.py's phrase-based scoring, which stays correct
for its documented purpose (scoring arbitrary free text with no known
probe/target).

Usage:
    python evaluation/run_kbd_calibration.py
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbd_scorer import load_knowledge_base, compute_kbd

AFFIRM_CUES = [r"^\s*yes\b", r"^\s*yeah\b", r"^\s*yep\b", r"\bthat'?s (right|correct|true)\b", r"\bconfirmed\b"]
DENY_CUES = [r"^\s*no\b", r"\bi don'?t (know|have)\b", r"\bnot sure\b", r"\bi'?m not\b",
             r"\bcan'?t confirm\b", r"\bno idea\b", r"\bdon'?t have access\b", r"\bhaven'?t heard\b"]


def classify_probe_response(response: str, probe: dict) -> str:
    """For a 'forbidden' probe: does the response affirm the linked fact,
    deny/deflect it, or is it ambiguous? Conservative — ambiguous counts as
    NOT a violation, same "don't guess a false positive" bias kbd_scorer.py
    uses for its denial check."""
    text = response.lower().strip()
    denied = any(re.search(cue, text) for cue in DENY_CUES)
    affirmed = any(re.search(cue, text) for cue in AFFIRM_CUES)
    if denied:
        return "denied"
    if affirmed:
        return "affirmed"
    return "ambiguous"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES_PATH = REPO_ROOT / "data" / "processed" / "kbd_probes.json"
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")

MODEL_PATHS = {
    "police officer": GGUF_MODELS_DIR / "modern_r16_a32_policeofficer-Q4_K_M.gguf",
}


def _add_torch_cuda_dlls_to_path():
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def main():
    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama

    items = load_knowledge_base()
    items_by_id = {i["id"]: i for i in items}
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]

    llm_cache = {}
    violations_caught = []
    results = []

    for probe in probes:
        archetype = probe["target_archetype"]
        if archetype not in llm_cache:
            model_path = MODEL_PATHS[archetype]
            print(f"loading {model_path.name}...")
            llm_cache[archetype] = Llama(model_path=str(model_path), n_ctx=2048, verbose=False,
                                          n_gpu_layers=0, n_threads=os.cpu_count(), n_batch=512,
                                          chat_format="zephyr")
        llm = llm_cache[archetype]

        system = SYSTEM_TEMPLATE.format(archetype=archetype)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": probe["player_probe"]}]
        result = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
        response = result["choices"][0]["message"]["content"].strip()

        score = compute_kbd(response, archetype, items)
        classification = classify_probe_response(response, probe) if probe["type"] == "forbidden" else None
        results.append({"probe_id": probe["id"], "type": probe["type"], "response": response,
                         "probe_classification": classification, **score})

        # A violation is either kbd_scorer.py catching a literal signature-phrase
        # confirmation, OR the probe-aware affirm/deny check catching a
        # paraphrased confirmation ("Yes, it did") that restates no content
        # words — see module docstring for why both checks are needed.
        is_violation = (score["kbd"] is not None and score["kbd"] > 0) or classification == "affirmed"
        flag = ""
        if is_violation:
            flag = "  <-- VIOLATION"
            violations_caught.append(probe["id"])
        print(f"[{probe['id']}] ({probe['type']:9s}) kbd={score['kbd']} probe_check={classification}  "
              f"{response[:70]!r}{flag}")

    print(f"\n--- KBD calibration summary ---")
    print(f"probes run: {len(results)}")
    scored = [r for r in results if r["kbd"] is not None]
    print(f"probes with a detected factual reference (phrase match): {len(scored)}/{len(results)}")
    forbidden = [r for r in results if r["type"] == "forbidden"]
    print(f"forbidden probes: {len(forbidden)}, real violations caught (phrase match OR probe-aware affirm check): "
          f"{len(violations_caught)} ({violations_caught})")
    print(f"Week 6 exit condition (KBD catches >=1 real violation): "
          f"{'PASS' if violations_caught else 'FAIL'}")

    out_path = Path(__file__).resolve().parent / "results" / "kbd_calibration_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
