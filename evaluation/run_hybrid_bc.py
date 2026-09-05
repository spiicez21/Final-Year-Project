"""
Hybrid B+C test (2026-08-15, user-directed follow-up to the KBD leak-rate
work): does combining Condition B's trained adapter with Condition C's
explicit flat-RAG system prompt ("you only know X, refuse anything else")
reduce leakage further than either alone?

Rationale: Condition B's failure mode is a weight-level bias (the model
hasn't learned precise topic discrimination); Condition C's failure mode is
that a stock model, even told exactly what it knows, still leaks under
leading questions. These are different mechanisms — worth testing whether
combining them cancels out either weakness. Zero retraining: reuses the
already-deployed adapter GGUFs and Condition C's exact fact-list template.

Runs BOTH forbidden and control probes for all 8 archetypes (the control
side is checked from the start this time — the whole point of prior KBD-fix
rounds was learning not to skip it) via `kbd_alpha_sweep_probes.json`.

Usage:
    python evaluation/run_hybrid_bc.py
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbd_scorer import load_knowledge_base, compute_kbd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES_PATH = REPO_ROOT / "data" / "processed" / "kbd_alpha_sweep_probes.json"
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"

ARCHETYPE_GGUF = {
    "police officer": "modern_r16_a32_policeofficer-Q4_K_M.gguf",
    "pharmacist": "modern_r16_a32_pharmacist-Q4_K_M.gguf",
    "professor": "modern_r16_a32_professor-Q4_K_M.gguf",
    "bartender": "modern_r16_a32_bartender-Q4_K_M.gguf",
    "social worker": "modern_r16_a32_socialworker-Q4_K_M.gguf",
    "executive": "modern_r16_a32_executive-Q4_K_M.gguf",
    "shopkeeper": "modern_r16_a32_shopkeeper-Q4_K_M.gguf",
    "service worker": "modern_r16_a32_serviceworker-Q4_K_M.gguf",
}

# Same template as run_condition_c.py, applied on top of the TRAINED adapter
# instead of the stock base model.
FLAT_RAG_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                      "voice consistent with your role. Never break character. "
                      "You personally know the following things, and nothing else about ongoing events: "
                      "{facts} If asked about anything outside this list, say you don't know or "
                      "suggest they ask someone else.")

AFFIRM_CUES = [r"^\s*yes\b", r"^\s*yeah\b", r"^\s*yep\b", r"\b(that|it)'?s (right|correct|true)\b", r"\bconfirmed\b"]
DENY_CUES = [r"^\s*no\b", r"\bi don'?t (know|have)\b", r"\bnot sure\b", r"\bi'?m not\b",
             r"\bcan'?t confirm\b", r"\bno idea\b", r"\bdon'?t have access\b", r"\bhaven'?t heard\b"]


def classify(response: str) -> str:
    text = response.lower().strip()
    if any(re.search(cue, text) for cue in DENY_CUES):
        return "denied"
    if any(re.search(cue, text) for cue in AFFIRM_CUES):
        return "affirmed"
    return "ambiguous"


def _add_torch_cuda_dlls_to_path():
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def build_fact_list(archetype: str, items: list) -> str:
    own = [i["fact"] for i in items if archetype in i["visible_to"]]
    return " ".join(f"({i + 1}) {fact}" for i, fact in enumerate(own))


def main():
    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama

    items = load_knowledge_base()
    all_probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]

    results = {}
    all_rows = []

    for archetype, gguf_name in ARCHETYPE_GGUF.items():
        my_probes = [p for p in all_probes if p["target_archetype"] == archetype]
        forbidden = [p for p in my_probes if p["type"] == "forbidden"]
        control = [p for p in my_probes if p["type"] == "control"]
        if not forbidden and not control:
            continue

        print(f"\n=== {archetype} ({len(forbidden)} forbidden, {len(control)} control) ===")
        llm = Llama(model_path=str(GGUF_MODELS_DIR / gguf_name), n_ctx=2048, verbose=False,
                    n_gpu_layers=0, n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")
        facts = build_fact_list(archetype, items)
        system = FLAT_RAG_TEMPLATE.format(archetype=archetype, facts=facts)

        forbidden_violations = 0
        control_correct = 0
        for probe in forbidden + control:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": probe["player_probe"]}]
            r = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
            response = r["choices"][0]["message"]["content"].strip()
            score = compute_kbd(response, archetype, items)
            c = classify(response)
            is_violation = (score["kbd"] is not None and score["kbd"] > 0) or c == "affirmed"

            if probe["type"] == "forbidden":
                if is_violation:
                    forbidden_violations += 1
                flag = "VIOLATION" if is_violation else "ok"
            else:
                if is_violation:
                    control_correct += 1
                flag = "ok (confirmed)" if is_violation else "WRONG (refused own fact)"

            print(f"  [{probe['id']}] ({probe['type']:9s}) {flag:24s} {response[:60]!r}")
            all_rows.append({"archetype": archetype, "probe_id": probe["id"], "type": probe["type"],
                              "response": response, "violation_or_confirmed": is_violation})

        del llm
        f_rate = forbidden_violations / len(forbidden) if forbidden else None
        c_rate = control_correct / len(control) if control else None
        results[archetype] = {"forbidden_violation_rate": f_rate, "forbidden_n": len(forbidden),
                               "control_correct_rate": c_rate, "control_n": len(control)}
        print(f"  forbidden violated: {forbidden_violations}/{len(forbidden)}"
              + (f" ({f_rate:.0%})" if f_rate is not None else "")
              + f"  |  control correctly confirmed: {control_correct}/{len(control)}"
              + (f" ({c_rate:.0%})" if c_rate is not None else ""))

    print("\n--- hybrid B+C summary ---")
    print(f"{'archetype':15s} {'forbidden viol':>15s} {'control correct':>16s}")
    for arch, r in results.items():
        fr = f"{r['forbidden_violation_rate']:.0%}" if r["forbidden_violation_rate"] is not None else "n/a"
        cr = f"{r['control_correct_rate']:.0%}" if r["control_correct_rate"] is not None else "n/a"
        print(f"{arch:15s} {fr:>15s} {cr:>16s}")

    out_path = Path(__file__).resolve().parent / "results" / "hybrid_bc_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"summary": results, "rows": all_rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
