"""
Condition C (Week 11, Docs/TODO.md): flat-RAG baseline. Stock base model
(no LoRA adapter at all), system prompt lists the archetype's OWN visible
facts explicitly and instructs refusal for anything else. Without this
condition, a reviewer can attribute Condition B's KBD improvements to "the
adapter just has more relevant context" rather than to visibility-set
training specifically — Condition C tests whether prompt-only fact
injection gets you the same result for free.

Reuses the exact same probes already scored for Condition B's pure-adapter
endpoints in the alpha-sweep (alpha=1.0 = pure archetype A, alpha=0.0 =
pure archetype B) so the comparison is direct, not a new probe set that
could confound the read. Runs all 3 alpha-sweep pairs.

"Matched token budget": the injected fact list is short (2-3 facts per
archetype, matching what's actually in knowledge_base.json) — not padded to
exaggerate Condition C's overhead, but also not pretending token cost is
identical to Condition B (which has zero extra prompt tokens, since its
knowledge lives in LoRA weights, not the prompt). Reports the real token
overhead directly rather than asserting a false equivalence.

Usage:
    python evaluation/run_condition_c.py
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

# pair_key -> (archetype_a violation rate at Condition B alpha=1.0, archetype_b at alpha=0.0)
# from evaluation/results/alpha_sweep_results.json / Docs/TODO.md Weeks 9-10 table.
CONDITION_B_ENDPOINTS = {
    "police officer+pharmacist": ("police officer", 0.00, "pharmacist", 1.00),
    "social worker+executive": ("social worker", 1.00, "executive", 0.75),
    "bartender+professor": ("bartender", 0.50, "professor", 1.00),
}

FLAT_RAG_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                      "voice consistent with your role. Never break character. "
                      "You personally know the following things, and nothing else about ongoing events: "
                      "{facts} If asked about anything outside this list, say you don't know or "
                      "suggest they ask someone else.")

AFFIRM_CUES = [r"^\s*yes\b", r"^\s*yeah\b", r"^\s*yep\b", r"\b(that|it)'?s (right|correct|true)\b", r"\bconfirmed\b"]
DENY_CUES = [r"^\s*no\b", r"\bi don'?t (know|have)\b", r"\bnot sure\b", r"\bi'?m not\b",
             r"\bcan'?t confirm\b", r"\bno idea\b", r"\bdon'?t have access\b", r"\bhaven'?t heard\b"]


def classify_probe_response(response: str) -> str:
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

    model_path = GGUF_MODELS_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    print(f"loading stock baseline: {model_path.name} (no adapter)")
    llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")
    tokenizer_encode = llm.tokenize

    fact_list_cache = {}
    all_results = []

    for pair_key, (arch_a, b_endpoint_a, arch_b, b_endpoint_b) in CONDITION_B_ENDPOINTS.items():
        print(f"\n{'='*70}\nPAIR: {pair_key}\n{'='*70}")
        for archetype in (arch_a, arch_b):
            if archetype not in fact_list_cache:
                fact_list_cache[archetype] = build_fact_list(archetype, items)
                n_tokens = len(tokenizer_encode(fact_list_cache[archetype].encode("utf-8")))
                print(f"{archetype}: injected fact-list token overhead = {n_tokens} tokens")

        pair_probes = [p for p in all_probes if p["pair"] == pair_key and p["type"] == "forbidden"]
        violations, total = 0, 0
        for probe in pair_probes:
            archetype = probe["target_archetype"]
            system = FLAT_RAG_TEMPLATE.format(archetype=archetype, facts=fact_list_cache[archetype])
            messages = [{"role": "system", "content": system}, {"role": "user", "content": probe["player_probe"]}]
            result = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
            response = result["choices"][0]["message"]["content"].strip()

            score = compute_kbd(response, archetype, items)
            classification = classify_probe_response(response)
            is_violation = (score["kbd"] is not None and score["kbd"] > 0) or classification == "affirmed"
            total += 1
            if is_violation:
                violations += 1
            flag = "  <-- VIOLATION" if is_violation else ""
            print(f"[{probe['id']}] ({archetype}) {response[:70]!r}{flag}")
            all_results.append({"pair": pair_key, "probe_id": probe["id"], "archetype": archetype,
                                 "response": response, "classification": classification,
                                 "kbd": score["kbd"], "violation": is_violation})

        rate = violations / total if total else None
        print(f"Condition C violation rate: {violations}/{total} ({rate:.0%})")
        print(f"Compare to Condition B pure-adapter endpoints (same probes, alpha-sweep): "
              f"{arch_a}={b_endpoint_a:.0%}, {arch_b}={b_endpoint_b:.0%}")

    out_path = Path(__file__).resolve().parent / "results" / "condition_c_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
