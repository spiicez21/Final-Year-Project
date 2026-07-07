"""
Phase 4 evaluation — Condition B (Specs.md section 8): TinyLlama + medieval
LoRA adapter, run against the exact same 326 prompts used for the Condition A
baseline (evaluation/results/baseline_outputs.json), so the comparison is
apples-to-apples.

Runs locally via transformers+peft (not Ollama, which can't load a raw PEFT
LoRA adapter directly — that needs a merge+GGUF-export step not yet built).
Latency here is therefore NOT directly comparable to Condition A's
Ollama-served latency; the PDM/drift comparison is the primary thing this
script is for.

Usage:
    python run_condition_b.py --adapter training/adapters/medieval_r8_gutonly
    python run_condition_b.py --adapter training/adapters/medieval_r8_gutonly --limit 20   # smoke test
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdm_scorer import extract_features, single_turn_drift

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "evaluation" / "results" / "baseline_outputs.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUTS_PATH = RESULTS_DIR / "condition_b_outputs.json"
METRICS_CSV = RESULTS_DIR / "condition_b_metrics.csv"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                   "period-appropriate voice consistent with your role. Never break character.")


def load_model(adapter_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, archetype: str, prompt: str) -> tuple:
    system = SYSTEM_TEMPLATE.format(archetype=archetype)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip(), latency_ms


def run(adapter_path: str, limit: int = None):
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if limit:
        baseline = baseline[:limit]

    print(f"loading base model + adapter: {adapter_path}")
    model, tokenizer = load_model(adapter_path)

    results = []
    for i, entry in enumerate(baseline, 1):
        reference_features = set(entry["reference_features"])
        try:
            response, latency_ms = generate(model, tokenizer, entry["archetype"], entry["prompt"])
        except Exception as exc:
            print(f"[{i}/{len(baseline)}] {entry['id']} FAILED: {exc}")
            continue

        drift = single_turn_drift(response, reference_features)
        result = {
            "id": entry["id"],
            "archetype": entry["archetype"],
            "prompt": entry["prompt"],
            "reference_output": entry["reference_output"],
            "condition_b_response": response,
            "latency_ms": latency_ms,
            "reference_features": sorted(reference_features),
            "response_features": sorted(extract_features(response)),
            "drift_score": drift,
        }
        results.append(result)
        print(f"[{i}/{len(baseline)}] {entry['id']} ({entry['archetype']}) {latency_ms:.0f}ms drift={drift}")

        if i % 20 == 0:
            _save(results)

    _save(results)
    _compare(results, baseline[: len(results)])


def _save(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(METRICS_CSV, "w", encoding="utf-8") as f:
        f.write("id,archetype,latency_ms,drift_score,response_len\n")
        for r in results:
            f.write(f"{r['id']},{r['archetype']},{r['latency_ms']},{r['drift_score']},{len(r['condition_b_response'])}\n")


def _compare(results_b, baseline_a):
    if not results_b:
        print("no results")
        return
    drift_b = [r["drift_score"] for r in results_b]
    drift_a = [r["drift_score"] for r in baseline_a]
    lat_b = [r["latency_ms"] for r in results_b]

    print("\n--- Condition A (no adapter) vs Condition B (medieval LoRA, gutonly) ---")
    print(f"entries compared: {len(results_b)}")
    print(f"mean drift — A: {sum(drift_a)/len(drift_a):.4f}   B: {sum(drift_b)/len(drift_b):.4f}")
    print(f"drift < 1.0 (any archaic marker present) — A: {sum(1 for d in drift_a if d < 1.0)}/{len(drift_a)}"
          f"   B: {sum(1 for d in drift_b if d < 1.0)}/{len(drift_b)}")
    print(f"mean B latency: {sum(lat_b)/len(lat_b):.0f}ms (transformers+peft local, NOT comparable to A's Ollama latency)")
    print(f"wrote {OUTPUTS_PATH} and {METRICS_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.adapter, args.limit)


if __name__ == "__main__":
    main()
