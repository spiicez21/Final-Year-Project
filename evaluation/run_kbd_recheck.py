"""
Quick recheck: after retraining pharmacist and professor with refusal data
(Docs/TODO.md Week 11 follow-up), re-run their exact held-out probes from
kbd_alpha_sweep_probes.json (AS-05/06 for pharmacist, AS-21/22 for
professor) through the real GGUF serving path, same scoring as everywhere
else. Direct single-adapter eval, no blending.

Usage:
    python evaluation/run_kbd_recheck.py
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

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")

CHECKS = [
    ("pharmacist", "modern_r16_a32_pharmacist-Q4_K_M.gguf", ["AS-05", "AS-06"]),
    ("professor", "modern_r16_a32_professor-Q4_K_M.gguf", ["AS-21", "AS-22"]),
]

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


def main():
    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama

    items = load_knowledge_base()
    probes_by_id = {p["id"]: p for p in json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]}

    for archetype, gguf_name, probe_ids in CHECKS:
        model_path = GGUF_MODELS_DIR / gguf_name
        print(f"\n=== {archetype} ({gguf_name}) ===")
        llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                    n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")

        violations, total = 0, 0
        for pid in probe_ids:
            probe = probes_by_id[pid]
            system = SYSTEM_TEMPLATE.format(archetype=archetype)
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
            print(f"[{pid}] {response[:80]!r}{flag}")

        print(f"{archetype}: {violations}/{total} ({violations/total:.0%}) violation rate")


if __name__ == "__main__":
    main()
