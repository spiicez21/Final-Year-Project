"""
Docs/TODO.md 'Evaluation conditions (reference)' table: BERTScore F1 was
listed as a required metric ("Semantic similarity to reference output",
Specs.md) but never measured. Closes that gap.

For each of the 8 modern-city archetypes, samples real dataset prompts
(with their real gold `output` as reference), generates a response via the
GGUF serving path for both stock baseline (Condition A) and the trained
adapter (Condition B), and scores each against the gold reference with
BERTScore F1 (`bert-score` library, roberta-large default).

Usage:
    python evaluation/run_bertscore.py
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"

NUM_TEST_PROMPTS = 10
SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")

ARCHETYPE_GGUF = {
    "police officer": "modern_r16_a32_policeofficer-Q4_K_M.gguf",
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


def load_test_pairs(archetype: str, n: int, entries: list) -> list:
    matching = [(e["input"], e["output"]) for e in entries if e["persona"]["archetype"] == archetype]
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
    from bert_score import score as bertscore

    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]

    baseline_path = hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                                     filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                                     local_dir=str(GGUF_MODELS_DIR))

    all_cands = {"baseline": [], "adapter": []}
    all_refs = {"baseline": [], "adapter": []}
    per_archetype_pairs = {"baseline": {}, "adapter": {}}

    for archetype, gguf_name in ARCHETYPE_GGUF.items():
        print(f"\n=== {archetype} ===")
        pairs = load_test_pairs(archetype, NUM_TEST_PROMPTS, entries)
        prompts = [p for p, _ in pairs]
        gold = [g for _, g in pairs]
        print(f"test prompts: {len(prompts)}")

        for label, model_path in [("baseline", baseline_path), ("adapter", GGUF_MODELS_DIR / gguf_name)]:
            llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=False, n_gpu_layers=0,
                        n_threads=os.cpu_count(), n_batch=512, chat_format="zephyr")
            responses = generate_all(llm, archetype, prompts)
            all_cands[label].extend(responses)
            all_refs[label].extend(gold)
            per_archetype_pairs[label][archetype] = (responses, gold)
            del llm

    print("\ncomputing BERTScore F1 (this downloads/loads the scoring model on first run)...")
    results = {}
    for label in ("baseline", "adapter"):
        P, R, F1 = bertscore(all_cands[label], all_refs[label], lang="en", verbose=False)
        results[label] = {"mean_f1": F1.mean().item(), "per_archetype": {}}
        idx = 0
        for archetype in ARCHETYPE_GGUF:
            n = len(per_archetype_pairs[label][archetype][0])
            arch_f1 = F1[idx:idx + n]
            results[label]["per_archetype"][archetype] = arch_f1.mean().item()
            idx += n

    print(f"\n--- BERTScore F1 summary ---")
    print(f"{'archetype':15s} {'baseline':>10s} {'adapter':>10s} {'gap':>8s}")
    for archetype in ARCHETYPE_GGUF:
        b = results["baseline"]["per_archetype"][archetype]
        a = results["adapter"]["per_archetype"][archetype]
        print(f"{archetype:15s} {b:10.4f} {a:10.4f} {a - b:+8.4f}")
    print(f"{'OVERALL':15s} {results['baseline']['mean_f1']:10.4f} {results['adapter']['mean_f1']:10.4f} "
          f"{results['adapter']['mean_f1'] - results['baseline']['mean_f1']:+8.4f}")

    out_path = Path(__file__).resolve().parent / "results" / "bertscore_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
