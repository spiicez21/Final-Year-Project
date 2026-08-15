"""
Docs/TODO.md Week 5 follow-up (L122): PDM v2 was only calibrated against the
police officer archetype/adapter pair. Runs the same baseline-vs-adapter
drift comparison across the other 7 modern-city archetypes to check whether
the "trained adapter drifts less than stock TinyLlama" ordering found for
police officer holds generally, or was specific to that archetype's data.

Reuses run_pdm_v2_calibration.py's exact method (15 real dataset prompts per
archetype, GGUF path, PDM v2 per-family averaged drift) rather than a new
one, so results are comparable across archetypes.

Usage:
    python evaluation/run_pdm_v2_multi.py
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

NUM_TEST_PROMPTS = 15
SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")

ARCHETYPE_GGUF = {
    "pharmacist": "modern_r16_a32_pharmacist-Q4_K_M.gguf",
    "professor": "modern_r16_a32_professor-Q4_K_M.gguf",
    "bartender": "modern_r16_a32_bartender-Q4_K_M.gguf",
    "social worker": "modern_r16_a32_socialworker-Q4_K_M.gguf",
    "shopkeeper": "modern_r16_a32_shopkeeper-Q4_K_M.gguf",
    "executive": "modern_r16_a32_executive-Q4_K_M.gguf",
    "service worker": "modern_r16_a32_serviceworker-Q4_K_M.gguf",
}


def _add_torch_cuda_dlls_to_path():
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def load_test_prompts(archetype: str, n: int, entries: list) -> list:
    matching = [e["input"] for e in entries if e["persona"]["archetype"] == archetype]
    step = max(1, len(matching) // n)
    return matching[::step][:n]


def generate_all(llm, archetype: str, prompts: list) -> list:
    system = SYSTEM_TEMPLATE.format(archetype=archetype)
    responses = []
    for prompt in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        result = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
        responses.append(result["choices"][0]["message"]["content"].strip())
    return responses


def main():
    _add_torch_cuda_dlls_to_path()
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    lexicons = build_archetype_lexicons(entries)

    baseline_path = hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                                     filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                                     local_dir=str(GGUF_MODELS_DIR))

    summary = {}
    for archetype, gguf_name in ARCHETYPE_GGUF.items():
        print(f"\n=== {archetype} ===")
        reference = build_reference_features(archetype, entries, lexicons)
        prompts = load_test_prompts(archetype, NUM_TEST_PROMPTS, entries)
        print(f"reference: lexicon={len(reference['lexicon'])} words, prompts={len(prompts)}")

        results = {}
        for label, model_path in [("baseline", baseline_path), ("adapter", GGUF_MODELS_DIR / gguf_name)]:
            llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                        n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")
            responses = generate_all(llm, archetype, prompts)
            drifts = [single_turn_drift_v2(r, archetype, reference, lexicons) for r in responses]
            mean = statistics.mean(drifts)
            stdev = statistics.stdev(drifts) if len(drifts) > 1 else 0.0
            results[label] = {"mean": mean, "stdev": stdev, "drifts": drifts}
            print(f"  {label:9s} mean={mean:.4f} stdev={stdev:.4f} min={min(drifts):.4f} max={max(drifts):.4f}")
            del llm

        gap = results["baseline"]["mean"] - results["adapter"]["mean"]
        order_ok = gap > 0
        summary[archetype] = {"baseline_mean": results["baseline"]["mean"],
                               "adapter_mean": results["adapter"]["mean"], "gap": gap, "order_ok": order_ok}
        print(f"  gap (baseline - adapter): {gap:+.4f}  {'OK (adapter drifts less)' if order_ok else 'REVERSED'}")

    print("\n--- summary across all 7 archetypes ---")
    for a, s in summary.items():
        print(f"{a:15s} baseline={s['baseline_mean']:.4f}  adapter={s['adapter_mean']:.4f}  "
              f"gap={s['gap']:+.4f}  {'OK' if s['order_ok'] else 'REVERSED'}")
    n_ok = sum(1 for s in summary.values() if s["order_ok"])
    print(f"\ncorrect ordering (adapter drifts less than baseline): {n_ok}/{len(summary)}")

    out_path = Path(__file__).resolve().parent / "results" / "pdm_v2_multi_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
