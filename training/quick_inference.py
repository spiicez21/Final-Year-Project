"""
Quick sanity check — load the base model + a trained LoRA adapter, run a
handful of prompts, print outputs. Not a full evaluation (see evaluation/),
just a fast "did training actually do anything" check.

Usage:
    python quick_inference.py --adapter training/adapters/medieval_r8
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from pdm_scorer import extract_features

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TEST_PROMPTS = [
    ("guard", "Where can I find the blacksmith?"),
    ("merchant", "What's the price on that sword?"),
    ("scholar", "What do the old texts say of the dragon beneath the mountain?"),
    ("innkeeper", "Have you a room for the night?"),
    ("noble", "What news from the capital?"),
]


DIRECTIVE_SUFFIX = (
    " Use archaic words such as thee, thou, thy, hath, dost, doth, wilt, nay, art, "
    "'tis, prithee, and forsooth wherever they fit naturally."
)


def generate(model, tokenizer, archetype: str, prompt: str, directive: bool = False) -> str:
    system = (f"You are a {archetype} NPC in a medieval RPG world. "
              f"Respond in an archaic, period-appropriate voice consistent with your role. "
              f"Never break character.")
    if directive:
        system += DIRECTIVE_SUFFIX
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--directive", action="store_true",
                         help="add explicit archaic-vocabulary instruction to the system prompt")
    args = parser.parse_args()

    print(f"loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16)

    print(f"loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    label = "directive prompt" if args.directive else "standard prompt"
    print(f"\n--- Condition B (TinyLlama + medieval LoRA, {label}) sample outputs ---\n")
    for archetype, prompt in TEST_PROMPTS:
        response = generate(model, tokenizer, archetype, prompt, directive=args.directive)
        features = extract_features(response)
        print(f"[{archetype}] {prompt}")
        print(f"  -> {response}")
        print(f"  dialect features: {features}\n")


if __name__ == "__main__":
    main()
