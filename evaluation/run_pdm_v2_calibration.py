"""
Week 5 (Docs/TODO.md) acceptance check: PDM v2 must show variance across
turns when scored against real generations — not a flat near-1.0 score,
which is what happened when PDM v1's baseline was calibrated without ever
looking at real model output (the 0.9833 mistake, see Docs/TODO.md).

Generates real responses from two conditions via the GGUF path
(run_gguf_latency.py's model registry) against real police-officer prompts
pulled from the actual dataset, scores each with PDM v2, and reports the
distribution — mean/stdev, not just a single number.

Usage:
    python evaluation/run_pdm_v2_calibration.py
"""

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdm_v2 import build_archetype_lexicons, build_reference_features, single_turn_drift_v2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"

ARCHETYPE = "police officer"
NUM_TEST_PROMPTS = 15
SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")


def _add_torch_cuda_dlls_to_path():
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def load_test_prompts(archetype: str, n: int) -> list:
    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    matching = [e["input"] for e in entries if e["persona"]["archetype"] == archetype]
    # Even stride across the pool instead of the first N, so the sample
    # isn't just whichever source happened to be merged first (all the
    # POL-* hand-authored entries are first in the file, before the SOD-*
    # SODA ones) — want prompts from both sources represented.
    step = max(1, len(matching) // n)
    return matching[::step][:n]


def generate_all(model_key: str, prompts: list) -> list:
    from llama_cpp import Llama

    if model_key == "tinyllama":
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                                      filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                                      local_dir=str(GGUF_MODELS_DIR))
    else:
        model_path = GGUF_MODELS_DIR / "modern_r16_a32_policeofficer-Q4_K_M.gguf"

    llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")

    system = SYSTEM_TEMPLATE.format(archetype=ARCHETYPE)
    responses = []
    for prompt in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        result = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
        responses.append(result["choices"][0]["message"]["content"].strip())
    return responses


def main():
    _add_torch_cuda_dlls_to_path()

    print("building PDM v2 lexicons + reference features from the dataset...")
    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    lexicons = build_archetype_lexicons(entries)
    reference = build_reference_features(ARCHETYPE, entries, lexicons)
    print(f"reference ({ARCHETYPE}): lexicon={len(reference['lexicon'])} words, "
          f"register={len(reference['register'])} markers, "
          f"formality_mode={reference['formality']}, syntax_mode={reference['syntax']}, "
          f"stance_mode={reference['stance']}")

    prompts = load_test_prompts(ARCHETYPE, NUM_TEST_PROMPTS)
    print(f"test prompts: {len(prompts)}")

    for label, model_key in [("baseline (no adapter)", "tinyllama"), ("modern_police adapter", "modern_police")]:
        print(f"\n=== {label} ===")
        responses = generate_all(model_key, prompts)
        drifts = []
        for prompt, response in zip(prompts, responses):
            drift = single_turn_drift_v2(response, ARCHETYPE, reference, lexicons)
            drifts.append(drift)
            print(f"  drift={drift:.4f}  {response[:70]!r}")
        print(f"  mean={statistics.mean(drifts):.4f}  stdev={statistics.stdev(drifts):.4f}  "
              f"min={min(drifts):.4f}  max={max(drifts):.4f}")
        variance_ok = statistics.stdev(drifts) > 0.01
        print(f"  Week 5 acceptance (variance across turns, not flat): {'PASS' if variance_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
