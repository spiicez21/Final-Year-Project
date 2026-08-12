"""
Weeks 9-10 (Docs/TODO.md) — the alpha-sweep, RQ2's headline result: when two
persona adapters are interpolated at ratio alpha, does the epistemic
boundary INTERPOLATE (each parent's facts known with reduced confidence) or
COLLAPSE TO THE UNION (both parents' facts fully known — a leak)?

For each of 3 archetype pairs, blends the two trained LoRA adapters via
PEFT's add_weighted_adapter (same mechanism as training/blend_adapters.py,
but kept in-memory across all 5 alpha values instead of saving+reloading
from disk each time — this is an evaluation sweep, not a deployment
artifact, so skips the GGUF export step Week 1/7-8 used for serving).

At each alpha, generates real responses to data/processed/kbd_alpha_sweep_probes.json
(bidirectional forbidden/control probes covering both parents' exclusive
knowledge) and scores with:
  - KBD (violation rate on forbidden probes, using the same probe-aware
    affirm/deny classifier from run_kbd_calibration.py)
  - PDM v2, scored against BOTH parents' reference feature sets separately
    (pdm_vs_a, pdm_vs_b) — this is what "interpolate vs collapse" looks like
    for persona voice specifically, alongside KBD's epistemic-boundary read.

Usage:
    python evaluation/run_alpha_sweep.py
"""

import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbd_scorer import load_knowledge_base, compute_kbd
from pdm_v2 import build_archetype_lexicons, build_reference_features, single_turn_drift_v2

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_DIR = REPO_ROOT / "training" / "adapters"
PROBES_PATH = REPO_ROOT / "data" / "processed" / "kbd_alpha_sweep_probes.json"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                    "voice consistent with your role. Never break character.")

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

PAIRS = [
    ("police officer", "modern_r16_a32_policeofficer", "pharmacist", "modern_r16_a32_pharmacist"),
    ("social worker", "modern_r16_a32_socialworker", "executive", "modern_r16_a32_executive"),
    ("bartender", "modern_r16_a32_bartender", "professor", "modern_r16_a32_professor"),
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


def load_base():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config,
                                                        device_map="auto", dtype=torch.bfloat16)
    return base_model, tokenizer


def generate(model, tokenizer, archetype: str, prompt: str) -> str:
    system = SYSTEM_TEMPLATE.format(archetype=archetype)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    items = load_knowledge_base()
    all_probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]
    dataset_entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    lexicons = build_archetype_lexicons(dataset_entries)

    all_results = []

    for archetype_a, adapter_a_name, archetype_b, adapter_b_name in PAIRS:
        pair_key = f"{archetype_a}+{archetype_b}"
        pair_probes = [p for p in all_probes if p["pair"] == pair_key]
        print(f"\n{'='*70}\nPAIR: {pair_key}  ({len(pair_probes)} probes)\n{'='*70}")

        ref_a = build_reference_features(archetype_a, dataset_entries, lexicons)
        ref_b = build_reference_features(archetype_b, dataset_entries, lexicons)

        print("loading base model + both adapters...")
        base_model, tokenizer = load_base()
        model = PeftModel.from_pretrained(base_model, str(ADAPTERS_DIR / adapter_a_name), adapter_name="a")
        model.load_adapter(str(ADAPTERS_DIR / adapter_b_name), adapter_name="b")

        for alpha in ALPHAS:
            # PyTorch module names can't contain "." (nn.ModuleDict key
            # rule) — a literal blend_0.25 crashes inside add_weighted_adapter
            # with KeyError. Use an integer-percent name instead.
            blend_name = f"blend_{int(round(alpha * 100))}"
            model.base_model.add_weighted_adapter(
                adapters=["a", "b"], weights=[alpha, 1 - alpha],
                adapter_name=blend_name, combination_type="linear",
            )
            model.set_adapter(blend_name)
            model.eval()

            forbidden_violations, forbidden_total = 0, 0
            pdm_a_scores, pdm_b_scores = [], []

            for probe in pair_probes:
                archetype = probe["target_archetype"]
                response = generate(model, tokenizer, archetype, probe["player_probe"])

                if probe["type"] == "forbidden":
                    forbidden_total += 1
                    score = compute_kbd(response, archetype, items)
                    classification = classify_probe_response(response)
                    is_violation = (score["kbd"] is not None and score["kbd"] > 0) or classification == "affirmed"
                    if is_violation:
                        forbidden_violations += 1

                pdm_a_scores.append(single_turn_drift_v2(response, archetype_a, ref_a, lexicons))
                pdm_b_scores.append(single_turn_drift_v2(response, archetype_b, ref_b, lexicons))

                all_results.append({
                    "pair": pair_key, "alpha": alpha, "probe_id": probe["id"], "type": probe["type"],
                    "target_archetype": archetype, "response": response,
                })

            violation_rate = forbidden_violations / forbidden_total if forbidden_total else None
            mean_pdm_a = sum(pdm_a_scores) / len(pdm_a_scores)
            mean_pdm_b = sum(pdm_b_scores) / len(pdm_b_scores)
            print(f"  alpha={alpha:.2f}  KBD violation rate={violation_rate:.2f} "
                  f"({forbidden_violations}/{forbidden_total})  "
                  f"PDM-vs-{archetype_a}={mean_pdm_a:.3f}  PDM-vs-{archetype_b}={mean_pdm_b:.3f}")

            model.delete_adapter(blend_name)

        del model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_path = Path(__file__).resolve().parent / "results" / "alpha_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
