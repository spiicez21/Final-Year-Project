"""
Phase 1 baseline evaluation — Condition A (Specs.md section 8): TinyLlama,
zero fine-tuning, run against every prompt in the processed dataset. This is
the comparison floor every later condition (B: +LoRA, C: GPT-4o, D: full
fine-tune) gets measured against.

Requires `ollama serve` running locally with the `tinyllama` model pulled.

Usage:
    python run_baseline.py                 # full dataset run
    python run_baseline.py --limit 20       # smoke test
    python run_baseline.py --resume         # skip ids already in the output file
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdm_scorer import extract_features, single_turn_drift

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "medieval_npc_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUTS_PATH = RESULTS_DIR / "baseline_outputs.json"
METRICS_CSV = RESULTS_DIR / "baseline_metrics.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"


def load_entries(limit: int = None):
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    entries = data["entries"]
    return entries[:limit] if limit else entries


def load_existing():
    if OUTPUTS_PATH.exists():
        return {r["id"]: r for r in json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))}
    return {}


SESSION = requests.Session()


def call_ollama(prompt: str, timeout: int = 120, retries: int = 4):
    last_exc = None
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = SESSION.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            body = resp.json()
            return body.get("response", ""), latency_ms
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s backoff
    raise last_exc


def run(limit: int = None, resume: bool = False):
    entries = load_entries(limit)
    existing = load_existing() if resume else {}
    results = list(existing.values())

    for i, entry in enumerate(entries, 1):
        if entry["id"] in existing:
            continue

        reference_features = set(entry.get("linguistic_markers", {}).get("dialect_features", []))
        try:
            response_text, latency_ms = call_ollama(entry["input"])
        except Exception as exc:
            print(f"[{i}/{len(entries)}] {entry['id']} FAILED after retries: {exc}", flush=True)
            continue

        drift = single_turn_drift(response_text, reference_features)
        result = {
            "id": entry["id"],
            "archetype": entry["persona"]["archetype"],
            "prompt": entry["input"],
            "reference_output": entry["output"],
            "baseline_response": response_text,
            "latency_ms": latency_ms,
            "reference_features": sorted(reference_features),
            "response_features": sorted(extract_features(response_text)),
            "drift_score": drift,
        }
        results.append(result)
        print(f"[{i}/{len(entries)}] {entry['id']} ({entry['persona']['archetype']}) "
              f"{latency_ms:.0f}ms drift={drift}", flush=True)

        _save(results)  # flush every entry — a crash mid-run shouldn't lose more than one call
        time.sleep(0.5)  # be gentle on the 2GB-VRAM server between calls

    _save(results)
    _summarize(results)


def _save(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(METRICS_CSV, "w", encoding="utf-8") as f:
        f.write("id,archetype,latency_ms,drift_score,response_len\n")
        for r in results:
            f.write(f"{r['id']},{r['archetype']},{r['latency_ms']},{r['drift_score']},{len(r['baseline_response'])}\n")


def _summarize(results):
    if not results:
        print("no results")
        return
    latencies = [r["latency_ms"] for r in results]
    drifts = [r["drift_score"] for r in results]
    print("\n--- Condition A (TinyLlama, no adapter) baseline summary ---")
    print(f"entries: {len(results)}")
    print(f"latency ms  — mean: {sum(latencies)/len(latencies):.0f}  min: {min(latencies):.0f}  max: {max(latencies):.0f}")
    print(f"drift score — mean: {sum(drifts)/len(drifts):.4f}  min: {min(drifts):.4f}  max: {max(drifts):.4f}")
    print(f"wrote {OUTPUTS_PATH} and {METRICS_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
