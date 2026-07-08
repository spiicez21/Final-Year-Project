"""
Phase 4 persona stress test protocol (Specs.md section 8): run all 50
held-out stress test conversations (data/processed/stress_test_corpus.json)
through a given condition, turn by turn with real conversation history, and
record the turn at which persona breaks.

Persona break (per spec, literal definition): the turn's single-turn drift
score is > 0.7 AND no archaic dialect features are present in that turn's
response. Reference feature set is the archetype's aggregate dialect
vocabulary from the training data (data/processed/medieval_npc_dataset.json),
filtered to only words pdm_scorer.py's regex can actually detect.

Two backends:
  --condition baseline          Ollama /api/chat, tinyllama, no adapter (Condition A)
  --condition adapter --model-path ...   transformers+peft LoRA adapter (Condition B)
  --condition full --model-path ...      transformers full fine-tune checkpoint (Condition D)

Usage:
    python run_stress_test.py --condition baseline
    python run_stress_test.py --condition adapter --model-path training/adapters/medieval_r8_gutonly
    python run_stress_test.py --condition full --model-path training/adapters/medieval_full_finetune_gutonly
"""

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdm_scorer import extract_features, single_turn_drift, DIALECT_PATTERNS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "medieval_npc_dataset.json"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "stress_test_corpus.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                   "period-appropriate voice consistent with your role. Never break character.")


def build_archetype_references() -> dict:
    """Per-archetype reference dialect-feature set, aggregated from the
    training data, filtered to pdm_scorer's actual detectable vocabulary."""
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    valid = set(DIALECT_PATTERNS.keys())
    refs = {}
    for e in data["entries"]:
        arch = e["persona"]["archetype"]
        refs.setdefault(arch, set()).update(e["linguistic_markers"]["dialect_features"])
    return {arch: feats & valid for arch, feats in refs.items()}


def load_local_model(model_path: str, is_full: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if is_full:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    else:
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer


def generate_local(model, tokenizer, messages: list) -> str:
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


OLLAMA_SESSION = requests.Session()


def generate_ollama(messages: list, retries: int = 4) -> str:
    # Same resilience pattern as run_baseline.py — persistent session
    # (reduces socket churn) + longer backoff. Without these, this
    # crashed on WinError 10013 (socket exhaustion under sustained
    # Ollama load) partway through a 50-conversation run with zero
    # partial results saved. See Docs/TODO.md.
    last_exc = None
    for attempt in range(retries):
        try:
            resp = OLLAMA_SESSION.post(OLLAMA_CHAT_URL, json={"model": "tinyllama", "messages": messages, "stream": False},
                                        timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                import time
                time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s
    raise last_exc


def run_conversation(entry: dict, references: dict, backend: str, model=None, tokenizer=None) -> dict:
    archetype = entry["archetype"]
    reference_features = references.get(archetype, set())
    system = SYSTEM_TEMPLATE.format(archetype=archetype)
    messages = [{"role": "system", "content": system}]

    turn_results = []
    break_turn = None
    for i, player_turn in enumerate(entry["turns"], 1):
        messages.append({"role": "user", "content": player_turn})
        response = generate_ollama(messages) if backend == "ollama" else generate_local(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        features = extract_features(response)
        drift = single_turn_drift(response, reference_features)
        broke = drift > 0.7 and not features
        if broke and break_turn is None:
            break_turn = i

        turn_results.append({"turn": i, "player": player_turn, "response": response,
                              "features": sorted(features), "drift": drift, "broke": broke})

    return {
        "id": entry["id"], "archetype": archetype, "stress_test_type": entry["stress_test_type"],
        "total_turns": len(entry["turns"]), "break_turn": break_turn, "turns": turn_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["baseline", "adapter", "full"])
    parser.add_argument("--model-path", default=None, help="required for --condition adapter/full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    if args.condition != "baseline" and not args.model_path:
        parser.error("--model-path required for --condition adapter/full")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["entries"]
    if args.limit:
        corpus = corpus[: args.limit]
    references = build_archetype_references()

    backend = "ollama" if args.condition == "baseline" else "local"
    model = tokenizer = None
    if backend == "local":
        print(f"loading model: {args.model_path} ({'full fine-tune' if args.condition == 'full' else 'LoRA adapter'})", flush=True)
        model, tokenizer = load_local_model(args.model_path, is_full=(args.condition == "full"))

    output_name = args.output_name or f"stress_test_{args.condition}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs_path = RESULTS_DIR / f"{output_name}_outputs.json"

    existing = {}
    if outputs_path.exists():
        existing = {r["id"]: r for r in json.loads(outputs_path.read_text(encoding="utf-8"))}
    results = list(existing.values())

    for i, entry in enumerate(corpus, 1):
        if entry["id"] in existing:
            continue
        try:
            result = run_conversation(entry, references, backend, model, tokenizer)
        except Exception as exc:
            print(f"[{i}/{len(corpus)}] {entry['id']} FAILED after retries: {exc}", flush=True)
            continue
        results.append(result)
        bt = result["break_turn"]
        print(f"[{i}/{len(corpus)}] {entry['id']} ({entry['archetype']}, {entry['stress_test_type']}) "
              f"break_turn={bt if bt else 'never'}/{result['total_turns']}", flush=True)

        outputs_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    _summarize(results, outputs_path)


def _summarize(results, outputs_path):
    if not results:
        print("no results")
        return
    broke = [r for r in results if r["break_turn"] is not None]
    never_broke = [r for r in results if r["break_turn"] is None]
    print(f"\n--- Persona stress test summary ({len(results)} conversations) ---")
    print(f"broke at some turn: {len(broke)}/{len(results)} ({100*len(broke)/len(results):.1f}%)")
    print(f"never broke: {len(never_broke)}/{len(results)} ({100*len(never_broke)/len(results):.1f}%)")
    if broke:
        mean_break_turn = sum(r["break_turn"] for r in broke) / len(broke)
        print(f"mean turn of break (among those that broke): {mean_break_turn:.2f}")
    # censored mean: non-breaking conversations count as breaking at total_turns+1 (never within the window)
    censored = [r["break_turn"] if r["break_turn"] else r["total_turns"] + 1 for r in results]
    print(f"mean turns-before-break (censored, non-breaks counted at total_turns+1): {sum(censored)/len(censored):.2f}")

    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r["stress_test_type"]].append(r)
    print("\nby stress_test_type:")
    for stype, rs in sorted(by_type.items()):
        b = sum(1 for r in rs if r["break_turn"] is not None)
        print(f"  {stype:24s} broke {b}/{len(rs)}")
    print(f"\nwrote {outputs_path}")


if __name__ == "__main__":
    main()
